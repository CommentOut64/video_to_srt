"""
统一进度发射器 - V3.1.0

封装进度更新和 SSE 推送，解决双流流水线进度不同步的问题。

核心职责:
1. 计算综合进度 (根据模式和权重)
2. 同步更新 job.progress 和 job.phase
3. 推送 SSE 事件 (overall + 阶段级别)
4. 生成 checkpoint 兼容数据

使用方式:
    emitter = ProgressEventEmitter(job, sse_manager, mode)
    emitter.update_preprocess(100, "completed")
    emitter.update_fast(5, 10)  # 5/10 chunks
    emitter.update_slow(3, 10)  # 3/10 chunks
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, TYPE_CHECKING
from enum import Enum
import logging
import time

if TYPE_CHECKING:
    from app.models.job_models import JobState
    from app.services.sse_service import SSEManager

logger = logging.getLogger(__name__)


class ProgressMode(Enum):
    """进度模式 - 对应不同的转录流水线"""
    SENSEVOICE_ONLY = "sensevoice_only"    # 极速模式: 仅 SenseVoice
    WHISPER_PATCH = "whisper_patch"        # 补刀模式: SenseVoice + Whisper 局部
    DUAL_STREAM = "dual_stream"            # 双流模式: SenseVoice + Whisper 全量


@dataclass
class ProgressDetail:
    """阶段化进度详情"""
    preprocess: float = 0.0      # 预处理进度 (0-100)
    fast: float = 0.0            # 快流/SenseVoice 进度 (0-100)
    slow: float = 0.0            # 慢流/Whisper 进度 (0-100)
    align: float = 0.0           # 对齐进度 (0-100)
    total: float = 0.0           # 综合总进度 (0-100)
    mode: ProgressMode = ProgressMode.DUAL_STREAM

    # 阶段消息
    preprocess_message: str = ""
    fast_message: str = ""
    slow_message: str = ""
    align_message: str = ""

    # 计数器
    fast_processed: int = 0
    fast_total: int = 0
    slow_processed: int = 0
    slow_total: int = 0
    align_processed: int = 0
    align_total: int = 0


class ProgressEventEmitter:
    """
    统一进度发射器

    解决的问题:
    - 双流流水线只操作 progress_tracker 但不推送 SSE
    - job.progress 始终为 0
    - 前端收不到实时进度更新
    """

    # 阶段权重配置 (不同模式)
    # V3.1.0 调整: 提高 fast (SenseVoice) 权重，让用户能看到明显的进度变化
    WEIGHTS = {
        ProgressMode.SENSEVOICE_ONLY: {
            "preprocess": 0.15,
            "fast": 0.85,
            "slow": 0.0,
            "align": 0.0
        },
        ProgressMode.WHISPER_PATCH: {
            "preprocess": 0.10,
            "fast": 0.50,  # V3.1.0: 45% → 50%，让 SenseVoice 完成时进度更明显
            "slow": 0.30,  # V3.1.0: 35% → 30%
            "align": 0.10
        },
        ProgressMode.DUAL_STREAM: {
            "preprocess": 0.10,
            "fast": 0.50,  # V3.1.0: 35% → 50%，让 SenseVoice 完成时进度达到 60%
            "slow": 0.30,  # V3.1.0: 40% → 30%
            "align": 0.10  # V3.1.0: 15% → 10%
        }
    }

    def __init__(
        self,
        job: 'JobState',
        sse_manager: 'SSEManager',
        mode: ProgressMode = ProgressMode.DUAL_STREAM,
        throttle_interval: float = 0.5  # SSE 推送节流间隔 (秒)
    ):
        """
        初始化进度发射器

        Args:
            job: 任务状态对象
            sse_manager: SSE 管理器
            mode: 进度模式
            throttle_interval: SSE 推送节流间隔，避免过于频繁
        """
        self.job = job
        self.sse_manager = sse_manager
        self.mode = mode
        self.detail = ProgressDetail(mode=mode)
        self.throttle_interval = throttle_interval
        self._last_push_time = 0.0
        self._weights = self.WEIGHTS.get(mode, self.WEIGHTS[ProgressMode.DUAL_STREAM])

        logger.debug(f"[ProgressEmitter] 初始化: job={job.job_id}, mode={mode.value}")

    def set_mode(self, mode: ProgressMode):
        """动态设置模式 (用于运行时切换)"""
        self.mode = mode
        self.detail.mode = mode
        self._weights = self.WEIGHTS.get(mode, self.WEIGHTS[ProgressMode.DUAL_STREAM])
        logger.debug(f"[ProgressEmitter] 模式切换: {mode.value}")

    def update_preprocess(self, percent: float, stage: str = "", message: str = ""):
        """
        更新预处理进度

        Args:
            percent: 进度百分比 (0-100)
            stage: 当前阶段名称 (extract/vad/spectral/separation)
            message: 进度消息
        """
        self.detail.preprocess = min(100.0, max(0.0, percent))
        self.detail.preprocess_message = message or f"预处理: {stage}"

        # 更新 job.phase
        if percent < 100:
            self.job.phase = stage or "extract"

        self._recalculate_and_push("preprocess", {
            "percent": self.detail.preprocess,
            "stage": stage,
            "message": message
        })

    def update_fast(
        self,
        processed: int,
        total: int,
        message: str = "",
        force_push: bool = False
    ):
        """
        更新快流 (SenseVoice) 进度

        Args:
            processed: 已处理 Chunk 数
            total: 总 Chunk 数
            message: 进度消息
            force_push: 强制推送 (忽略节流)
        """
        self.detail.fast_processed = processed
        self.detail.fast_total = total
        self.detail.fast = (processed / total * 100) if total > 0 else 0
        self.detail.fast_message = message or f"SenseVoice: {processed}/{total}"

        # 更新 job
        self.job.phase = "sensevoice"
        self.job.processed = processed
        self.job.total = total

        self._recalculate_and_push("fast", {
            "percent": self.detail.fast,
            "processed": processed,
            "total": total,
            "message": message
        }, force_push=force_push)

    def update_slow(
        self,
        processed: int,
        total: int,
        message: str = "",
        force_push: bool = False
    ):
        """
        更新慢流 (Whisper) 进度

        Args:
            processed: 已处理 Chunk 数
            total: 总 Chunk 数
            message: 进度消息
            force_push: 强制推送 (忽略节流)
        """
        self.detail.slow_processed = processed
        self.detail.slow_total = total
        self.detail.slow = (processed / total * 100) if total > 0 else 0
        self.detail.slow_message = message or f"Whisper: {processed}/{total}"

        # 更新 job.phase (仅在非极速模式)
        if self.mode != ProgressMode.SENSEVOICE_ONLY:
            self.job.phase = "whisper"

        self._recalculate_and_push("slow", {
            "percent": self.detail.slow,
            "processed": processed,
            "total": total,
            "message": message
        }, force_push=force_push)

    def update_align(
        self,
        processed: int,
        total: int,
        message: str = "",
        force_push: bool = False
    ):
        """
        更新对齐进度

        Args:
            processed: 已处理 Chunk 数
            total: 总 Chunk 数
            message: 进度消息
            force_push: 强制推送 (忽略节流)
        """
        self.detail.align_processed = processed
        self.detail.align_total = total
        self.detail.align = (processed / total * 100) if total > 0 else 0
        self.detail.align_message = message or f"对齐: {processed}/{total}"

        # 更新 job.phase
        self.job.phase = "align"

        self._recalculate_and_push("align", {
            "percent": self.detail.align,
            "processed": processed,
            "total": total,
            "message": message
        }, force_push=force_push)

    def complete(self, message: str = "转录完成"):
        """
        标记任务完成

        Args:
            message: 完成消息
        """
        self.detail.preprocess = 100
        self.detail.fast = 100
        self.detail.slow = 100
        self.detail.align = 100
        self.detail.total = 100

        self.job.phase = "complete"
        self.job.progress = 100

        self._push_overall(force=True)

        # 推送完成信号
        self._push_signal("job_complete", message)

        logger.info(f"[ProgressEmitter] 任务完成: {self.job.job_id}")

    def fail(self, error_message: str):
        """
        标记任务失败

        Args:
            error_message: 错误消息
        """
        self.job.error = error_message

        # 推送失败信号
        self._push_signal("job_failed", error_message)

        logger.error(f"[ProgressEmitter] 任务失败: {self.job.job_id} - {error_message}")

    def _recalculate_and_push(
        self,
        phase: str,
        phase_data: Dict[str, Any],
        force_push: bool = False
    ):
        """重新计算总进度并推送 SSE"""
        # 计算加权总进度
        w = self._weights
        new_total = (
            self.detail.preprocess * w["preprocess"] +
            self.detail.fast * w["fast"] +
            self.detail.slow * w["slow"] +
            self.detail.align * w["align"]
        )

        # V3.1.0: 防止进度倒退（只允许进度增加或保持不变）
        # 这解决了节流导致的阶段不同步问题
        if new_total < self.detail.total:
            # 进度倒退，不更新 total，只更新阶段进度
            logger.debug(
                f"[ProgressEmitter] 检测到进度倒退: {self.detail.total:.1f}% -> {new_total:.1f}%, 保持原值"
            )
        else:
            self.detail.total = new_total

        # 同步到 job.progress
        self.job.progress = round(self.detail.total, 1)

        # V3.1.0: 关键节点强制推送（阶段完成时）
        is_milestone = False
        if phase == "fast" and self.detail.fast >= 100:
            is_milestone = True
        elif phase == "slow" and self.detail.slow >= 100:
            is_milestone = True
        elif phase == "align" and self.detail.align >= 100:
            is_milestone = True
        elif phase == "preprocess" and self.detail.preprocess >= 100:
            is_milestone = True

        # 节流检查
        now = time.time()
        should_push = force_push or is_milestone or (now - self._last_push_time) >= self.throttle_interval
        if not should_push:
            return

        self._last_push_time = now

        # 推送阶段进度事件
        self._push_phase(phase, phase_data)

        # 推送总体进度事件
        self._push_overall()

    def _push_phase(self, phase: str, data: Dict[str, Any]):
        """推送阶段级别进度事件"""
        if self.sse_manager is None:
            logger.warning(f"[ProgressEmitter] SSE管理器为None，无法推送进度: {phase}")
            return

        channel_id = f"job:{self.job.job_id}"
        event_type = f"progress.{phase}"

        logger.debug(f"[ProgressEmitter] 推送进度: {phase}={data.get('percent', 0):.1f}%")
        self.sse_manager.broadcast_sync(channel_id, event_type, data)

    def _push_overall(self, force: bool = False):
        """推送总体进度事件"""
        if self.sse_manager is None:
            logger.warning(f"[ProgressEmitter] SSE管理器为None，无法推送总体进度")
            return

        channel_id = f"job:{self.job.job_id}"
        job_id = self.job.job_id

        # 构建总体进度数据
        overall_data = {
            "job_id": job_id,
            "phase": self.job.phase,
            "percent": self.detail.total,
            "phase_percent": self._get_current_phase_percent(),
            "message": self._get_current_message(),
            "status": self.job.status,
            "processed": self.job.processed,
            "total": self.job.total,
            # V3.1.0 新增: 阶段细节
            "mode": self.mode.value,
            "detail": {
                "preprocess": self.detail.preprocess,
                "fast": self.detail.fast,
                "slow": self.detail.slow,
                "align": self.detail.align
            }
        }

        logger.debug(f"[ProgressEmitter] 推送总体进度: {self.detail.total:.1f}% (fast={self.detail.fast:.1f}%, slow={self.detail.slow:.1f}%)")
        # 推送到任务频道
        self.sse_manager.broadcast_sync(channel_id, "progress.overall", overall_data)

        # 同时推送到全局频道 (用于任务列表)
        global_data = {
            "id": job_id,
            "percent": self.detail.total,
            "message": self._get_current_message(),
            "status": self.job.status
        }
        self.sse_manager.broadcast_sync("global", "job_progress", global_data)

    def _push_signal(self, signal_type: str, message: str):
        """推送信号事件"""
        if self.sse_manager is None:
            return

        channel_id = f"job:{self.job.job_id}"
        signal_data = {
            "job_id": self.job.job_id,
            "signal": signal_type,
            "message": message
        }

        self.sse_manager.broadcast_sync(channel_id, f"signal.{signal_type}", signal_data)

        # 全局频道也推送
        self.sse_manager.broadcast_sync("global", f"signal.{signal_type}", signal_data)

    def _get_current_phase_percent(self) -> float:
        """获取当前阶段内进度"""
        phase = self.job.phase
        if phase in ("extract", "vad", "spectral", "separation"):
            return self.detail.preprocess
        elif phase == "sensevoice":
            return self.detail.fast
        elif phase == "whisper":
            return self.detail.slow
        elif phase == "align":
            return self.detail.align
        else:
            return self.detail.total

    def _get_current_message(self) -> str:
        """获取当前阶段消息"""
        phase = self.job.phase
        if phase in ("extract", "vad", "spectral", "separation"):
            return self.detail.preprocess_message
        elif phase == "sensevoice":
            return self.detail.fast_message
        elif phase == "whisper":
            return self.detail.slow_message
        elif phase == "align":
            return self.detail.align_message
        else:
            return self.job.message or "处理中..."

    # ========== Checkpoint 集成 ==========

    def to_checkpoint_data(self) -> Dict[str, Any]:
        """生成 checkpoint 兼容的进度数据"""
        return {
            "progress": {
                "preprocess": self.detail.preprocess,
                "fast": self.detail.fast,
                "slow": self.detail.slow,
                "align": self.detail.align,
                "total": self.detail.total,
                "mode": self.mode.value,
                "fast_processed": self.detail.fast_processed,
                "fast_total": self.detail.fast_total,
                "slow_processed": self.detail.slow_processed,
                "slow_total": self.detail.slow_total,
                "align_processed": self.detail.align_processed,
                "align_total": self.detail.align_total
            }
        }

    def restore_from_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """
        从 checkpoint 恢复进度

        支持两种格式：
        1. ProgressEmitter 自身的格式（包含 "progress" 字段）
        2. CheckpointV37 的格式（包含 "preprocessing" 和 "transcription" 字段）

        恢复后会立即推送一次 SSE，让前端显示正确的进度

        Args:
            checkpoint_data: checkpoint 数据字典
        """
        # 尝试从 ProgressEmitter 格式恢复
        progress = checkpoint_data.get("progress", {})
        if progress:
            # 恢复进度值
            self.detail.preprocess = progress.get("preprocess", 0)
            self.detail.fast = progress.get("fast", 0)
            self.detail.slow = progress.get("slow", 0)
            self.detail.align = progress.get("align", 0)
            self.detail.total = progress.get("total", 0)

            # 恢复计数器
            self.detail.fast_processed = progress.get("fast_processed", 0)
            self.detail.fast_total = progress.get("fast_total", 0)
            self.detail.slow_processed = progress.get("slow_processed", 0)
            self.detail.slow_total = progress.get("slow_total", 0)
            self.detail.align_processed = progress.get("align_processed", 0)
            self.detail.align_total = progress.get("align_total", 0)

            # 恢复模式
            mode_str = progress.get("mode", "dual_stream")
            try:
                self.mode = ProgressMode(mode_str)
                self.detail.mode = self.mode
                self._weights = self.WEIGHTS.get(self.mode, self.WEIGHTS[ProgressMode.DUAL_STREAM])
            except ValueError:
                logger.warning(f"[ProgressEmitter] 未知模式: {mode_str}, 使用默认")
        else:
            # 从 CheckpointV37 格式恢复
            preprocessing = checkpoint_data.get("preprocessing", {})
            transcription = checkpoint_data.get("transcription", {})

            # 预处理进度
            total_chunks = preprocessing.get("total_chunks", 0)
            if preprocessing.get("separation", {}).get("completed", False):
                self.detail.preprocess = 100.0
            elif preprocessing.get("spectral_triage", {}).get("completed", False):
                self.detail.preprocess = 80.0
            elif preprocessing.get("vad_completed", False):
                self.detail.preprocess = 60.0
            elif preprocessing.get("audio_extracted", False):
                self.detail.preprocess = 30.0

            # FastWorker 进度
            fast_worker = transcription.get("fast_worker", {})
            fast_count = fast_worker.get("completed_count", 0) or len(fast_worker.get("processed_indices", []))
            self.detail.fast_processed = fast_count
            self.detail.fast_total = total_chunks
            self.detail.fast = (fast_count / total_chunks * 100) if total_chunks > 0 else 0

            # SlowWorker 进度
            slow_worker = transcription.get("slow_worker", {})
            slow_count = slow_worker.get("completed_count", 0) or len(slow_worker.get("processed_indices", []))
            self.detail.slow_processed = slow_count
            self.detail.slow_total = total_chunks
            self.detail.slow = (slow_count / total_chunks * 100) if total_chunks > 0 else 0

            # AlignmentWorker 进度
            alignment = transcription.get("alignment", {})
            align_count = alignment.get("completed_count", 0) or len(alignment.get("finalized_indices", []))
            self.detail.align_processed = align_count
            self.detail.align_total = total_chunks
            self.detail.align = (align_count / total_chunks * 100) if total_chunks > 0 else 0

            # 重新计算总进度
            w = self._weights
            self.detail.total = (
                self.detail.preprocess * w["preprocess"] +
                self.detail.fast * w["fast"] +
                self.detail.slow * w["slow"] +
                self.detail.align * w["align"]
            )

        # 同步到 job
        self.job.progress = round(self.detail.total, 1)
        self.job.processed = self.detail.fast_processed
        self.job.total = self.detail.fast_total

        logger.info(
            f"[ProgressEmitter] 从 checkpoint 恢复: "
            f"total={self.detail.total:.1f}%, "
            f"fast={self.detail.fast_processed}/{self.detail.fast_total}, "
            f"slow={self.detail.slow_processed}/{self.detail.slow_total}"
        )

        # 立即推送一次，让前端恢复显示
        self._push_overall(force=True)


# ========== 工厂函数 ==========

_emitter_instances: Dict[str, ProgressEventEmitter] = {}


def get_progress_emitter(
    job: 'JobState',
    sse_manager: 'SSEManager',
    mode: ProgressMode = None,
    transcription_profile: str = None
) -> ProgressEventEmitter:
    """
    获取或创建进度发射器

    Args:
        job: 任务状态对象
        sse_manager: SSE 管理器
        mode: 进度模式 (可选，优先使用)
        transcription_profile: 转录配置名称 (用于推断模式)

    Returns:
        ProgressEventEmitter 实例
    """
    job_id = job.job_id

    if job_id not in _emitter_instances:
        # 推断模式
        if mode is None:
            if transcription_profile == "sensevoice_only":
                mode = ProgressMode.SENSEVOICE_ONLY
            elif transcription_profile == "sv_whisper_patch":
                mode = ProgressMode.WHISPER_PATCH
            else:
                mode = ProgressMode.DUAL_STREAM

        _emitter_instances[job_id] = ProgressEventEmitter(job, sse_manager, mode)

    return _emitter_instances[job_id]


def remove_progress_emitter(job_id: str):
    """移除进度发射器"""
    if job_id in _emitter_instances:
        del _emitter_instances[job_id]
        logger.debug(f"[ProgressEmitter] 移除: {job_id}")
