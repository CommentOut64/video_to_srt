"""
转录处理服务
整合了processor.py和原transcription_service.py的所有功能
"""
import os, threading, json, math, gc, logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import OrderedDict  # 新增导入
from pydub import AudioSegment, silence
from app.services.whisper_service import get_whisper_service, load_audio as whisper_load_audio
# 从新架构导入 VAD 配置（2025-12-17 统一配置定义）
from app.services.audio.vad_service import VADConfig, VADMethod, get_vad_service
from app.services.audio.chunk_engine import ChunkEngine, AudioChunk
import torch
import psutil
import numpy as np


class ProcessingMode(Enum):
    """
    处理模式枚举
    用于智能决策使用内存模式还是硬盘模式进行音频处理

    注意：DISK 模式（硬盘分段）已废弃，仅保留 MEMORY 模式
    """
    MEMORY = "memory"  # 内存模式（默认，高性能）
    DISK = "disk"      # 硬盘模式（已废弃，仅保留用于向后兼容）


class BreakToGlobalSeparation(Exception):
    """熔断异常：触发时需要升级为全局人声分离模式"""
    pass


@dataclass
class CircuitBreakerState:
    """
    熔断器状态（支持模型升级）

    用于监控转录质量，当大量段落需要重试时：
    1. 优先尝试升级模型（如果允许且未达上限）
    2. 无法升级时才触发熔断
    """
    consecutive_retries: int = 0        # 连续重试计数
    total_retries: int = 0              # 总重试次数
    total_segments: int = 0             # 总段落数
    processed_segments: int = 0         # 已处理段落数

    # === Phase 3: 升级跟踪 ===
    escalation_count: int = 0                           # 已升级次数
    current_model: Optional[str] = None                 # 当前使用的模型
    escalation_history: List[str] = field(default_factory=list)  # 升级历史

    def record_retry(self):
        """记录一次重试"""
        self.consecutive_retries += 1
        self.total_retries += 1

    def record_success(self):
        """记录一次成功（重置连续计数）"""
        self.consecutive_retries = 0
        self.processed_segments += 1

    def record_escalation(self, new_model: str):
        """
        记录一次模型升级

        Args:
            new_model: 升级后的模型名称
        """
        if self.current_model:
            self.escalation_history.append(f"{self.current_model} -> {new_model}")
        self.current_model = new_model
        self.escalation_count += 1
        # 升级后重置连续重试计数，给新模型机会
        self.consecutive_retries = 0

    def should_escalate(self, demucs_settings) -> bool:
        """
        判断是否应该升级模型（优先于熔断）

        升级条件：
        1. 允许自动升级 (auto_escalation=True)
        2. 未达到最大升级次数
        3. 满足熔断条件（连续重试或比例过高）

        Args:
            demucs_settings: Demucs配置对象

        Returns:
            bool: True表示应该升级模型
        """
        if not demucs_settings.auto_escalation:
            return False

        if self.escalation_count >= demucs_settings.max_escalations:
            return False

        # 满足熔断条件时，优先升级
        return self._check_break_condition(demucs_settings)

    def should_break(self, demucs_settings) -> bool:
        """
        判断是否应该触发熔断

        注意：只有在无法升级时才触发熔断

        Args:
            demucs_settings: Demucs配置对象

        Returns:
            bool: True表示应该触发熔断
        """
        if not demucs_settings.circuit_breaker_enabled:
            return False

        # 如果还能升级，不触发熔断
        if self.should_escalate(demucs_settings):
            return False

        return self._check_break_condition(demucs_settings)

    def _check_break_condition(self, demucs_settings) -> bool:
        """
        检查是否满足熔断/升级条件

        熔断条件（满足任一即触发）：
        1. 连续 N 个 segment 都触发重试（默认N=3）
        2. 总重试比例超过阈值（默认20%）

        Args:
            demucs_settings: Demucs配置对象

        Returns:
            bool: True表示满足熔断/升级条件
        """
        # 条件1：连续重试次数
        if self.consecutive_retries >= demucs_settings.consecutive_threshold:
            return True

        # 条件2：总重试比例（至少处理5个segment后才检查）
        if self.processed_segments >= 5:
            retry_ratio = self.total_retries / self.processed_segments
            if retry_ratio >= demucs_settings.ratio_threshold:
                return True

        return False

    def get_stats(self) -> Dict:
        """获取统计信息（扩展）"""
        return {
            "consecutive_retries": self.consecutive_retries,
            "total_retries": self.total_retries,
            "total_segments": self.total_segments,
            "processed_segments": self.processed_segments,
            "retry_ratio": self.total_retries / max(1, self.processed_segments),
            # Phase 3 新增
            "escalation_count": self.escalation_count,
            "current_model": self.current_model,
            "escalation_history": self.escalation_history,
        }


class CircuitBreakAction(Enum):
    """熔断后的处理动作"""
    CONTINUE = "continue"           # 继续处理，标记问题段落
    FALLBACK_ORIGINAL = "fallback"  # 降级使用原始音频
    FAIL = "fail"                   # 任务失败
    PAUSE = "pause"                 # 暂停等待人工介入


class CircuitBreakHandler:
    """
    熔断异常处理器

    负责在熔断触发时执行用户配置的处理策略
    """

    def __init__(self, job: "JobState", settings):
        """
        初始化熔断处理器

        Args:
            job: 任务状态对象
            settings: 预处理配置对象
        """
        self.job = job
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.problem_segments: List[int] = []  # 记录问题段落索引

    def handle(
        self,
        breaker_state: CircuitBreakerState,
        current_segment_idx: int,
        sse_manager = None
    ) -> CircuitBreakAction:
        """
        处理熔断异常

        Args:
            breaker_state: 熔断器状态
            current_segment_idx: 当前段落索引
            sse_manager: SSE管理器（用于推送事件）

        Returns:
            CircuitBreakAction: 处理动作
        """
        action_str = self.settings.on_break

        # 解析处理动作
        try:
            action = CircuitBreakAction(action_str)
        except ValueError:
            # 如果配置值无效，默认使用 CONTINUE
            self.logger.warning(f"无效的熔断处理策略: {action_str}，使用默认值 continue")
            action = CircuitBreakAction.CONTINUE

        # 记录问题段落
        self.problem_segments.append(current_segment_idx)

        # 推送 SSE 事件
        if sse_manager:
            self._push_circuit_break_event(breaker_state, action, sse_manager)

        # 根据策略执行操作
        if action == CircuitBreakAction.FAIL:
            self.logger.error(
                f"熔断触发，任务终止。问题段落: {self.problem_segments}"
            )
            raise BreakToGlobalSeparation(
                f"熔断触发，任务终止。问题段落: {self.problem_segments}"
            )

        elif action == CircuitBreakAction.PAUSE:
            self.logger.warning(
                f"熔断触发，等待人工介入。问题段落: {self.problem_segments}"
            )
            self.job.paused = True
            self.job.status = "paused"
            self.job.message = f"熔断触发，等待人工介入。问题段落: {self.problem_segments}"
            raise BreakToGlobalSeparation(self.job.message)

        else:  # CONTINUE 或 FALLBACK_ORIGINAL
            self.logger.warning(
                f"熔断触发，采用 {action.value} 策略继续处理。"
                f"问题段落: {self.problem_segments}"
            )

        return action

    def get_problem_report(self) -> Dict:
        """
        获取问题报告

        Returns:
            包含问题统计和建议的字典
        """
        return {
            "total_problem_segments": len(self.problem_segments),
            "problem_indices": self.problem_segments,
            "suggestion": self._get_suggestion()
        }

    def _get_suggestion(self) -> str:
        """
        根据问题段落数量给出建议

        Returns:
            建议文本
        """
        count = len(self.problem_segments)
        if count == 0:
            return "所有段落处理正常"
        elif count <= 3:
            return "少量段落可能需要手动调整时间轴"
        elif count <= 10:
            return "建议检查这些段落的字幕准确性"
        else:
            return "大量段落有问题，建议使用更高质量的模型重新处理"

    def _push_circuit_break_event(
        self,
        state: CircuitBreakerState,
        action: CircuitBreakAction,
        sse_manager
    ):
        """
        推送熔断处理事件

        Args:
            state: 熔断器状态
            action: 处理动作
            sse_manager: SSE管理器
        """
        try:
            sse_manager.push_event(
                self.job.job_id,
                "circuit_breaker_handled",
                {
                    "action": action.value,
                    "problem_segments": self.problem_segments,
                    "stats": state.get_stats(),
                    "suggestion": self._get_suggestion()
                }
            )
        except Exception as e:
            self.logger.debug(f"SSE推送失败（非致命）: {e}")


from app.models.job_models import JobSettings, JobState
from app.models.hardware_models import HardwareInfo, OptimizationConfig
from app.services.hardware_profile_service import get_hardware_profile_provider
from app.services.job_lifecycle_service import get_job_lifecycle_service
from app.core.config import config  # 导入统一配置

class TranscriptionService:
    """
    转录处理服务
    整合了所有转录相关功能
    """

    def __init__(self, jobs_root: str):
        """
        初始化转录服务

        Args:
            jobs_root: 任务工作目录根路径
        """
        self.jobs_root = Path(jobs_root)
        self.jobs_root.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # 集成硬件能力提供者
        self.hardware_profile_provider = get_hardware_profile_provider()
        self._hardware_info: Optional[HardwareInfo] = None
        self._optimization_config: Optional[OptimizationConfig] = None

        # 集成任务生命周期服务（创建/恢复/持久化）
        self.job_lifecycle = get_job_lifecycle_service(self.jobs_root, logger=self.logger)

        # 集成SSE管理器（用于实时进度推送）
        from app.services.sse_service import get_sse_manager
        self.sse_manager = get_sse_manager()
        self.logger.info("SSE管理器已集成")

        # 记录CPU信息
        sys_info = self.hardware_profile_provider.get_cpu_system_info()
        if sys_info.get('supported', False):
            self.logger.info(
                f" CPU信息: {sys_info['logical_cores']}个逻辑核心, "
                f"{sys_info.get('physical_cores', '?')}个物理核心, "
                f"平台: {sys_info.get('platform', '?')}"
            )
        else:
            self.logger.warning("CPU亲和性功能不可用")

        # 执行硬件检测
        self._detect_hardware()

        # 任务加载已由 JobLifecycleService 负责

    def _detect_hardware(self):
        """执行硬件检测并生成优化配置"""
        try:
            self.logger.info("开始硬件检测...")
            self._hardware_info = self.hardware_profile_provider.get_hardware_info(is_force_refresh=True)
            self._optimization_config = self.hardware_profile_provider.get_optimization_config(self._hardware_info)

            # 记录检测结果
            hw = self._hardware_info
            opt = self._optimization_config
            self.logger.info(f"硬件检测完成GPU: {'' if hw.cuda_available else ''}, "
                             f"CPU: {hw.cpu_cores}核/{hw.cpu_threads}线程, "
                             f"内存: {hw.memory_total_mb}MB, "
                             f"优化配置: batch={opt.batch_size}, device={opt.recommended_device}")

        except Exception as e:
            self.logger.error(f"硬件检测失败: {e}")

    def _build_asr_engines(
        self,
        job: "JobState",
        transcription_profile: str
    ) -> Tuple[Optional["ASREngine"], Optional["ASREngine"]]:
        """
        根据任务配置构建 ASR 引擎实例（用于流水线注入）。

        Args:
            job: 任务状态对象
            transcription_profile: 转录模式

        Returns:
            Tuple[Optional[ASREngine], Optional[ASREngine]]:
                (draft_engine, patch_engine)
        """
        from app.core.asr.engine import ASREngine
        from app.engines.factory import ASREngineFactory

        draft_engine: Optional[ASREngine] = ASREngineFactory.create("sensevoice")

        if transcription_profile == "sensevoice_only":
            return draft_engine, None

        transcription = getattr(job.settings, "transcription", None)
        model_name = getattr(transcription, "whisper_model", "medium")
        device = (
            self._optimization_config.recommended_device
            if self._optimization_config
            else "cuda"
        )
        patch_engine: Optional[ASREngine] = ASREngineFactory.create(
            "whisper",
            model_name=model_name,
            device=device,
            compute_type=None,
        )

        return draft_engine, patch_engine

    async def _run_pipeline_v2(self, job: JobState):
        """
        新架构 Pipeline 入口方法（2025-12-17 架构改造）

        职责：
        1. 触发：接收任务，传给 Pipeline
        2. 收尾：拿到结果，生成 SRT 文件

        注意：此方法是简化版，暂不支持断点续传、任务状态管理等复杂功能
        这些功能将在后续阶段逐步集成

        使用新架构 PreprocessingPipeline + AsyncDualPipeline

        Args:
            job: 任务状态对象
        """
        try:
            from app.pipelines.async_dual_pipeline import AsyncDualPipeline

            job_dir = Path(job.dir)
            input_path = job_dir / job.filename

            # ==========================================
            # 阶段 1: 音频前处理
            # ==========================================
            self._update_progress(job, 'audio_processing', 0, '音频处理中...')

            # 使用新架构：PreprocessingPipeline（Stage模式）
            self.logger.info("使用新架构预处理流水线（Stage模式）")
            chunks = await self._run_new_preprocessing(job, input_path)

            job.total = len(chunks)
            self.logger.info(f"音频处理完成: {len(chunks)} 个 Chunk")

            # ==========================================
            # 阶段 2: 异步双流转录（AsyncDualPipeline）
            # V3.5: 根据 transcription_profile 动态创建流水线
            # ==========================================
            self._update_progress(job, 'transcription', 0, '转录中...')

            # 获取转录模式配置
            transcription = getattr(job.settings, "transcription", None)
            transcription_profile = (
                transcription.transcription_profile
                if transcription else "sensevoice_only"
            )
            self.logger.info(f"转录模式: {transcription_profile}")

            draft_engine, patch_engine = self._build_asr_engines(job, transcription_profile)
            draft_name = draft_engine.get_engine_name() if draft_engine else "none"
            patch_name = patch_engine.get_engine_name() if patch_engine else "none"
            self.logger.info(
                "新 ASR 引擎已启用: draft=%s, patch=%s, profile=%s",
                draft_name,
                patch_name,
                transcription_profile,
            )

            from app.core.thresholds import ThresholdConfig
            patching_threshold_value = getattr(
                transcription,
                "patching_threshold",
                0.60,
            )
            patching_threshold = ThresholdConfig(
                whisper_patch_trigger_confidence=patching_threshold_value
            )

            # 动态创建转录流水线
            transcription_pipeline = AsyncDualPipeline(
                job_id=job.job_id,
                transcription_profile=transcription_profile,
                draft_engine=draft_engine,
                patch_engine=patch_engine,
                patching_threshold=patching_threshold,
                logger=self.logger
            )

            # 调用转录流水线
            results = await transcription_pipeline.run(
                audio_chunks=chunks
            )

            # 从 ProcessingContext 中提取句子
            final_sentences = []
            for ctx in results:
                if hasattr(ctx, 'sv_result') and ctx.sv_result:
                    # 从 streaming_subtitle 获取句子
                    from app.services.streaming_subtitle import get_streaming_subtitle_manager
                    subtitle_manager = get_streaming_subtitle_manager(job.job_id)
                    chunk_indices = subtitle_manager.chunk_sentences.get(ctx.chunk_index, [])
                    for idx in chunk_indices:
                        if idx in subtitle_manager.sentences:
                            final_sentences.append(subtitle_manager.sentences[idx])

            self.logger.info(f"转录完成: {len(final_sentences)} 个句子")

            # ==========================================
            # 阶段 3: 生成 SRT 文件（收尾）
            # ==========================================
            self._update_progress(job, 'finalize', 0, '生成字幕文件...')

            srt_path = job_dir / f"{Path(job.filename).stem}.srt"
            self._generate_srt_from_sentences(final_sentences, srt_path)

            job.srt_path = str(srt_path)
            job.status = 'completed'
            job.message = '转录完成'
            job.progress = 100

            self.logger.info(f"任务完成: {job.job_id}")

        except Exception as e:
            self.logger.error(f"Pipeline 执行失败: {e}", exc_info=True)
            job.status = 'failed'
            job.error = str(e)
            job.message = f'失败: {str(e)}'
            raise

    def _generate_srt_from_sentences(self, sentences: List, output_path: Path):
        """
        从句子列表生成 SRT 文件
        V3.1.1+dev.20260106.03: 生成前自动修复时间戳重叠

        Args:
            sentences: 句子列表
            output_path: 输出路径
        """
        # V3.1.1+dev.20260106.03: 转换为字典列表以便修复重叠
        from app.utils.text_utils import repair_timestamp_overlaps, detect_timestamp_overlaps

        segments = []
        for sentence in sentences:
            segments.append({
                'start': sentence.start,
                'end': sentence.end,
                'text': sentence.text
            })

        # 检测并修复重叠
        overlaps = detect_timestamp_overlaps(segments)
        if overlaps:
            self.logger.warning(f"检测到 {len(overlaps)} 处时间戳重叠，自动修复中...")
            segments = repair_timestamp_overlaps(segments, gap_ms=1.0)
            self.logger.info(f"已修复 {len(overlaps)} 处时间戳重叠")

        # 生成 SRT 内容
        srt_content = []
        for i, seg in enumerate(segments, 1):
            start = self._format_srt_timestamp(seg['start'])
            end = self._format_srt_timestamp(seg['end'])
            text = seg['text']

            srt_content.append(f"{i}\n{start} --> {end}\n{text}\n")

        output_path.write_text('\n'.join(srt_content), encoding='utf-8')
        self.logger.info(f"SRT 文件已生成: {output_path}")

    def _format_srt_timestamp(self, seconds: float) -> str:
        """
        格式化 SRT 时间戳

        Args:
            seconds: 秒数

        Returns:
            SRT 格式时间戳 (HH:MM:SS,mmm)
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    async def _run_new_preprocessing(self, job: JobState, input_path: Path) -> List:
        """
        使用新架构预处理流水线处理音频

        Args:
            job: 任务状态对象
            input_path: 输入文件路径

        Returns:
            List[AudioChunk]: 预处理完成的 Chunk 列表
        """
        from app.pipelines.preprocessing_pipeline import PreprocessingPipeline
        from app.services.audio.chunk_engine import ChunkEngine
        from app.services.runtime_param_resolver import build_vad_config_for_profile

        # V3.9: 根据引擎类型选择 VAD 配置
        # 英语使用 Whisper，需要合并 VAD（避免幻觉）
        # 其他语言使用 SenseVoice，需要保留停顿信息
        language = getattr(job.settings, 'language', 'auto')
        is_english = language in {'en', 'english'}

        profile = "whisper" if is_english else "sensevoice"
        vad_config = build_vad_config_for_profile(profile)
        self.logger.info("使用 %s VAD 配置", "Whisper" if is_english else "SenseVoice")

        # 创建自定义 ChunkEngine
        chunk_engine = ChunkEngine(logger=self.logger)

        # 创建 PreprocessingPipeline 实例
        preprocessing_pipeline = PreprocessingPipeline(
            config=job.settings.preprocessing,
            chunk_engine=chunk_engine,
            vad_config=vad_config,
            logger=self.logger
        )

        # 执行预处理
        chunks = await preprocessing_pipeline.process(
            video_path=str(input_path),
            job_state=job
        )

        # 记录统计信息
        stats = preprocessing_pipeline.get_statistics(chunks)
        self.logger.info(
            f"预处理统计: "
            f"总chunk数={stats['total_chunks']}, "
            f"需要分离={stats['need_separation']}, "
            f"已分离={stats['separated']}, "
            f"分离比例={stats['separation_ratio']:.2%}, "
            f"熔断重试总次数={stats['fuse_retry_total']}, "
            f"最大重试次数={stats['fuse_retry_max']}"
        )

        return chunks

    def _load_all_jobs_from_disk(self):
        """
        启动时扫描并加载所有任务到内存（修复重启后无法打开旧任务的问题）

        这个方法会扫描 jobs 目录中的所有任务，并加载到内存中，
        避免重启后因为内存为空导致无法访问旧任务
        """
        self.job_lifecycle.load_all_jobs_from_disk()
    
    def get_hardware_info(self) -> Optional[HardwareInfo]:
        """获取硬件信息"""
        return self._hardware_info
    
    def get_optimization_config(self) -> Optional[OptimizationConfig]:
        """获取优化配置"""  
        return self._optimization_config
    
    def get_optimized_job_settings(self, base_settings: Optional[JobSettings] = None) -> JobSettings:
        """获取基于硬件优化的任务设置"""
        settings = base_settings or JobSettings()

        # 基于硬件推荐优化 SenseVoice 设备选择
        if (
            self._optimization_config
            and settings.transcription.sensevoice_device == "auto"
        ):
            settings.transcription.sensevoice_device = self._optimization_config.recommended_device

        return settings

    def create_job(
        self,
        filename: str,
        src_path: str,
        settings: JobSettings,
        job_id: Optional[str] = None
    ) -> JobState:
        """
        创建转录任务

        Args:
            filename: 文件名
            src_path: 源文件路径
            settings: 任务设置
            job_id: 任务ID（可选，不提供则自动生成）

        Returns:
            JobState: 创建的任务状态对象
        """
        return self.job_lifecycle.create_job(
            filename=filename,
            src_path=src_path,
            settings=settings,
            job_id=job_id
        )

    def save_job_meta(self, job: JobState) -> bool:
        """
        保存任务元信息到状态仓库（用于重启后恢复）

        使用原子写入确保断电安全：先写临时文件，再rename替换

        Args:
            job: 任务状态对象

        Returns:
            bool: 是否成功保存
        """
        return self.job_lifecycle.save_job_meta(job)

    def load_job_meta(self, job_id: str) -> Optional[JobState]:
        """
        从状态仓库加载任务元信息

        Args:
            job_id: 任务ID

        Returns:
            Optional[JobState]: 恢复的任务状态对象
        """
        return self.job_lifecycle.load_job_meta(job_id)

    def get_job(self, job_id: str) -> Optional[JobState]:
        """
        获取任务状态

        Args:
            job_id: 任务ID

        Returns:
            Optional[JobState]: 任务状态对象，不存在则返回None
        """
        return self.job_lifecycle.get_job(job_id)

    def scan_incomplete_jobs(self) -> List[Dict]:
        """
        扫描所有未完成的任务（有checkpoint.json的任务）

        Returns:
            List[Dict]: 未完成任务列表
        """
        return self.job_lifecycle.scan_incomplete_jobs()

    def restore_job_from_checkpoint(self, job_id: str) -> Optional[JobState]:
        """
        从检查点恢复任务状态（无 checkpoint 时从头开始）

        Args:
            job_id: 任务ID

        Returns:
            Optional[JobState]: 恢复的任务状态对象
        """
        return self.job_lifecycle.restore_job_from_checkpoint(job_id)

    def check_file_checkpoint(self, file_path: str) -> Optional[Dict]:
        """
        检查文件是否有可用的断点

        Args:
            file_path: 文件路径

        Returns:
            Optional[Dict]: 断点信息，无断点则返回None
        """
        return self.job_lifecycle.check_file_checkpoint(file_path)

    def start_job(self, job_id: str):
        """
        启动转录任务（V2.2: 废弃，由队列服务调用_run_pipeline_v2）

        注意: 此方法保留是为了向后兼容，不再自动创建线程

        Args:
            job_id: 任务ID
        """
        self.job_lifecycle.start_job(job_id)

    def pause_job(self, job_id: str) -> bool:
        """
        暂停转录任务（保存断点）

        Args:
            job_id: 任务ID

        Returns:
            bool: 是否成功设置暂停标志
        """
        return self.job_lifecycle.pause_job(job_id)

    def cancel_job(self, job_id: str, delete_data: bool = False):
        """
        取消转录任务

        Args:
            job_id: 任务ID
            delete_data: 是否删除任务数据

        Returns:
            Tuple[bool, Optional[str]]: (是否成功, 失败原因)
        """
        return self.job_lifecycle.cancel_job(job_id, delete_data=delete_data)

    def _update_progress(
        self,
        job: JobState,
        phase: str,
        phase_ratio: float,
        message: str = ""
    ):
        """
        更新任务进度

        Args:
            job: 任务状态对象
            phase: 当前阶段 (extract/split/transcribe/srt)
            phase_ratio: 当前阶段完成比例 (0.0-1.0)
            message: 进度消息
        """
        job.phase = phase

        # 计算阶段内进度（0-100，保留1位小数）
        job.phase_percent = round(max(0.0, min(1.0, phase_ratio)) * 100, 1)

        # 使用配置中的进度权重
        phase_weights = config.PHASE_WEIGHTS
        total_weight = config.TOTAL_WEIGHT

        # 计算累计进度
        done_weight = 0
        for p, w in phase_weights.items():
            if p == phase:
                break
            done_weight += w

        current_weight = phase_weights.get(phase, 0) * max(0.0, min(1.0, phase_ratio))
        # 改为1位小数
        job.progress = round((done_weight + current_weight) / total_weight * 100, 1)

        if message:
            job.message = message

        # 推送SSE进度更新（线程安全）
        self._push_sse_progress(job)

    def _push_sse_progress(self, job: JobState):
        """
        推送SSE进度更新（线程安全）
        同时推送到单任务频道和全局频道，确保 TaskMonitor 实时更新

        Args:
            job: 任务状态对象
        """
        try:
            # 动态获取SSE管理器（确保获取到已设置loop的实例）
            from app.services.sse_service import get_sse_manager
            sse_manager = get_sse_manager()

            # 1. 推送到单任务频道（EditorView 使用）
            channel_id = f"job:{job.job_id}"
            progress_data = {
                "job_id": job.job_id,
                "phase": job.phase,
                "percent": job.progress,
                "phase_percent": job.phase_percent,  # 新增：阶段内进度
                "message": job.message,
                "status": job.status,
                "processed": job.processed,
                "total": job.total,
                "language": job.language or ""
            }
            sse_manager.broadcast_sync(channel_id, "progress.overall", progress_data)

            # 2. 推送到全局频道（TaskMonitor 使用）
            global_progress_data = {
                "id": job.job_id,  # 全局频道使用 "id"
                "percent": job.progress,
                "phase_percent": job.phase_percent,  # 新增：阶段内进度
                "message": job.message,
                "status": job.status,
                "phase": job.phase,
                "processed": job.processed,
                "total": job.total
            }
            sse_manager.broadcast_sync("global", "job_progress", global_progress_data)

        except Exception as e:
            # SSE推送失败不应影响转录流程
            self.logger.debug(f"SSE推送失败: {e}")

    def _push_sse_signal(self, job: JobState, signal_code: str, message: str = ""):
        """
        推送SSE信号事件（用于关键节点通知）

        Args:
            job: 任务状态对象
            signal_code: 信号代码（如 "job_complete", "job_failed", "job_canceled"）
            message: 附加消息
        """
        try:
            # 动态获取SSE管理器（确保获取到已设置loop的实例）
            from app.services.sse_service import get_sse_manager
            sse_manager = get_sse_manager()

            channel_id = f"job:{job.job_id}"
            sse_manager.broadcast_sync(
                channel_id,
                f"signal.{signal_code}",
                {
                    "job_id": job.job_id,
                    "signal": signal_code,
                    "message": message or job.message,
                    "status": job.status,
                    "percent": job.progress
                }
            )
        except Exception as e:
            self.logger.debug(f"SSE信号推送失败（非致命）: {e}")

    def _trigger_media_post_process(self, job_id: str):
        """
        异步触发媒体预处理（转录完成后调用）
        生成波形峰值、视频缩略图、Proxy视频等，为编辑器做准备

        Args:
            job_id: 任务ID
        """
        try:
            import asyncio
            import aiohttp

            async def do_post_process():
                """异步执行预处理请求"""
                try:
                    # 调用媒体预处理接口
                    async with aiohttp.ClientSession() as session:
                        url = f"http://127.0.0.1:8000/api/media/{job_id}/post-process"
                        async with session.post(url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                            if resp.status == 200:
                                result = await resp.json()
                                self.logger.info(f"媒体预处理完成: peaks={result.get('peaks')}, thumbnails={result.get('thumbnails')}, proxy={result.get('proxy')}")
                            else:
                                self.logger.warning(f"媒体预处理请求失败: {resp.status}")
                except asyncio.TimeoutError:
                    self.logger.warning(f"媒体预处理超时: {job_id}")
                except Exception as e:
                    self.logger.warning(f"媒体预处理异常: {e}")

            # 尝试在现有事件循环中执行
            try:
                loop = asyncio.get_running_loop()
                asyncio.ensure_future(do_post_process(), loop=loop)
            except RuntimeError:
                # 没有运行中的事件循环，创建新线程执行
                import threading
                def run_in_thread():
                    asyncio.run(do_post_process())
                thread = threading.Thread(target=run_in_thread, daemon=True)
                thread.start()

            self.logger.info(f"已触发媒体预处理任务: {job_id}")

            # 触发 720p 高清转码（队列空闲时执行）
            self._trigger_720p_transcode_if_idle(job_id)

        except Exception as e:
            # 预处理失败不影响转录结果
            self.logger.warning(f"触发媒体预处理失败（非致命）: {e}")

    def _trigger_720p_transcode_if_idle(self, job_id: str):
        """
        在转录完成后触发 720p 高清转码（仅在队列空闲时）

        策略:
        1. 检查视频是否需要转码（H.265 等）
        2. 检查转录队列是否空闲
        3. 如果空闲，立即启动 720p 转码
        4. 如果队列有任务，延迟触发（使用低优先级）
        """
        try:
            from pathlib import Path
            from app.core.config import config
            from app.services.media_prep_service import get_media_prep_service

            self.logger.info(f"[720p] 开始检查是否需要触发720p转码: {job_id}")

            job_dir = config.JOBS_DIR / job_id
            if not job_dir.exists():
                self.logger.info(f"[720p] 任务目录不存在，跳过: {job_id}")
                return

            # 查找视频文件
            video_file = None
            video_exts = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.webm', '.flv', '.m4v']
            for file in job_dir.iterdir():
                if file.is_file() and file.suffix.lower() in video_exts:
                    video_file = file
                    break

            if not video_file:
                self.logger.info(f"[720p] 未找到视频文件，跳过: {job_id}")
                return

            self.logger.info(f"[720p] 找到视频文件: {video_file.name}")

            # 使用正确的文件名：proxy_720p.mp4
            proxy_720p = job_dir / "proxy_720p.mp4"

            # 720p 已存在则跳过
            if proxy_720p.exists():
                self.logger.info(f"[720p] 已存在，跳过: {job_id}")
                return

            media_prep = get_media_prep_service()

            # 检查 720p 是否已在队列中
            proxy_status = media_prep.get_proxy_status(job_id)
            if proxy_status and proxy_status.get("status") in ["queued", "processing", "completed"]:
                self.logger.info(f"[720p] 任务已存在或完成（状态={proxy_status.get('status')}），跳过: {job_id}")
                return

            # 检查 360p 预览是否已完成（必须先有360p才能启动720p）
            # 修复：优先检查文件是否存在，因为内存状态可能在服务重启后丢失
            preview_360p = job_dir / "preview_360p.mp4"

            # 文件存在即认为360p已完成（内存状态可能丢失）
            if preview_360p.exists():
                preview_completed = True
                self.logger.debug(f"[720p] 360p文件存在，认为已完成: {job_id}")
            else:
                # 文件不存在，检查内存状态（可能正在转码中）
                preview_status = media_prep.get_preview_status(job_id)
                preview_completed = preview_status and preview_status.get("status") == "completed"

            if not preview_completed:
                self.logger.info(f"[720p] 360p预览未完成，跳过720p转码: {job_id}")
                return

            # V3.1.2+dev.20260114.05: 交给 720p 调度器统一排队，避免重复入队
            from app.services.proxy_720_scheduler import get_proxy_scheduler

            scheduler = get_proxy_scheduler()
            scheduler.request(
                job_id,
                video_file,
                trigger_type="transcription_done",
                auto_enabled=config.PROXY_CONFIG.get('auto_trigger_720p', False),
                force=False,
                priority=100
            )

        except Exception as e:
            self.logger.warning(f"[720p] 触发转码失败（非致命）: {e}")

    def _is_transcription_queue_idle(self) -> bool:
        """检查转录任务队列是否空闲"""
        try:
            from app.services.job_queue_service import get_job_queue

            job_queue = get_job_queue()
            if not job_queue:
                return True

            # 检查是否有正在处理或等待的任务
            active_jobs = [
                j for j in job_queue.jobs.values()
                if j.status in ['pending', 'processing', 'queued']
            ]

            return len(active_jobs) == 0

        except Exception as e:
            self.logger.debug(f"检查队列状态失败: {e}")
            return True  # 默认认为空闲

    def _push_sse_segment(self, job: JobState, segment_result: dict, processed: int, total: int):
        """
        推送单个segment的转录结果（流式输出）

        Args:
            job: 任务状态对象
            segment_result: 单个segment的转录结果（未对齐）
            processed: 已处理的segment数量
            total: 总segment数量
        """
        try:
            # 动态获取SSE管理器
            from app.services.sse_service import get_sse_manager
            sse_manager = get_sse_manager()

            channel_id = f"job:{job.job_id}"
            sse_manager.broadcast_sync(
                channel_id,
                "subtitle.segment",
                {
                    "segment_index": segment_result.get('segment_index', 0),
                    "segments": segment_result.get('segments', []),
                    "language": segment_result.get('language', job.language),
                    "progress": {
                        "processed": processed,
                        "total": total,
                        "percentage": round(processed / max(1, total) * 100, 2)
                    }
                }
            )
            self.logger.debug(f"推送segment #{segment_result.get('segment_index', 0)} 转录结果")
        except Exception as e:
            # SSE推送失败不应影响转录流程
            self.logger.debug(f"SSE segment推送失败（非致命）: {e}")

    def _push_sse_aligned(self, job: JobState, aligned_results: List[Dict]):
        """
        推送对齐完成事件（流式输出）

        Args:
            job: 任务状态对象
            aligned_results: 对齐后的结果列表
        """
        try:
            # 动态获取SSE管理器
            from app.services.sse_service import get_sse_manager
            sse_manager = get_sse_manager()

            channel_id = f"job:{job.job_id}"

            # 提取对齐后的segments
            segments = []
            word_segments = []
            if aligned_results and len(aligned_results) > 0:
                segments = aligned_results[0].get('segments', [])
                word_segments = aligned_results[0].get('word_segments', [])

            sse_manager.broadcast_sync(
                channel_id,
                "subtitle.aligned",
                {
                    "segments": segments,
                    "word_segments": word_segments,
                    "message": "对齐完成"
                }
            )
            self.logger.info(f"推送对齐完成事件，共 {len(segments)} 条字幕")
        except Exception as e:
            # SSE推送失败不应影响转录流程
            self.logger.debug(f"SSE aligned推送失败（非致命）: {e}")

    def _save_checkpoint(self, job_dir: Path, data: dict, job: JobState):
        """
        原子性保存检查点
        使用"写临时文件 -> 重命名"策略，确保文件要么完整写入，要么保持原样

        Args:
            job_dir: 任务目录
            data: 检查点数据
            job: 任务状态对象（用于获取settings）
        """
        self.job_lifecycle._save_checkpoint(job_dir, data, job)

    def _load_checkpoint(self, job_dir: Path) -> Optional[dict]:
        """
        加载检查点，如果文件损坏则返回 None

        Args:
            job_dir: 任务目录

        Returns:
            Optional[dict]: 检查点数据，不存在或损坏则返回 None
        """
        return self.job_lifecycle._load_checkpoint(job_dir)

    def _flush_checkpoint_after_split(
        self,
        job_dir: Path,
        job: JobState,
        segments: List[Dict],
        processing_mode: ProcessingMode,
        demucs_state: Dict = None
    ):
        """
        分段完成后强制刷新checkpoint（确保断点续传一致性）

        这是断点续传的关键节点！
        只有分段元数据被持久化后，后续的转录索引才有意义。

        Args:
            job_dir: 任务目录
            job: 任务状态对象
            segments: 分段元数据列表
            processing_mode: 当前处理模式
            demucs_state: Demucs状态数据（可选）
        """
        self.job_lifecycle._flush_checkpoint_after_split(
            job=job,
            job_dir=job_dir,
            processing_mode=processing_mode,
            segments=segments,
            demucs_state=demucs_state,
        )


    # ========== 核心处理方法 ==========

    def _get_audio_duration(self, audio_path: str) -> float:
        """
        获取音频时长（秒）

        Args:
            audio_path: 音频文件路径

        Returns:
            float: 音频时长（秒）
        """
        try:
            # 方法1: 使用pydub（精确但较慢）
            audio = AudioSegment.from_wav(audio_path)
            duration = len(audio) / 1000.0
            self.logger.debug(f"音频时长（pydub）: {duration:.1f}秒")
            return duration
        except Exception as e:
            self.logger.warning(f"pydub获取时长失败，使用文件大小估算: {e}")
            # 方法2: 根据文件大小估算（16kHz, 16bit, mono ≈ 32KB/秒）
            try:
                file_size = os.path.getsize(audio_path)
                duration = file_size / 32000
                self.logger.debug(f"音频时长（估算）: {duration:.1f}秒")
                return duration
            except Exception as e2:
                self.logger.error(f"获取音频时长失败: {e2}")
                return 0.0

    def _decide_processing_mode(self, audio_path: str, job: JobState) -> ProcessingMode:
        """
        智能决策处理模式（内存模式 vs 硬盘模式）

        决策逻辑：
        1. 估算音频内存需求
        2. 检测系统可用内存
        3. 预留安全余量（模型、转录中间变量等）
        4. 决定使用哪种模式

        Args:
            audio_path: 音频文件路径
            job: 任务状态对象

        Returns:
            ProcessingMode: 处理模式
        """
        # 获取音频时长（秒）
        audio_duration_sec = self._get_audio_duration(audio_path)

        # 估算音频内存需求 (16kHz, float32)
        # 公式: duration * 16000 * 4 bytes
        estimated_audio_mb = (audio_duration_sec * 16000 * 4) / (1024 * 1024)

        # 预留额外内存（模型加载、VAD处理、转录中间变量等）
        # 保守估计：音频内存的2倍 + 500MB基础开销
        total_estimated_mb = estimated_audio_mb * 2 + 500

        # 获取系统可用内存
        mem_info = psutil.virtual_memory()
        available_mb = mem_info.available / (1024 * 1024)
        total_mb = mem_info.total / (1024 * 1024)

        # 安全阈值：动态计算，综合考虑多种因素
        # 1. 基础保留：2GB（保证系统基本运行）
        # 2. 动态保留：可用内存的10%（而非总内存的20%，避免过度保守）
        # 3. 最大保留上限：4GB（避免在大内存系统上过度保留）
        base_reserve_mb = 2048
        dynamic_reserve_mb = available_mb * 0.1
        safety_reserve_mb = min(base_reserve_mb + dynamic_reserve_mb, 4096)
        usable_mb = available_mb - safety_reserve_mb

        self.logger.info(f"内存评估:")
        self.logger.info(f"音频时长: {audio_duration_sec/60:.1f}分钟")
        self.logger.info(f"预估需求: {total_estimated_mb:.0f}MB")
        self.logger.info(f"可用内存: {available_mb:.0f}MB")
        self.logger.info(f"安全余量: {safety_reserve_mb:.0f}MB")
        self.logger.info(f"可用于处理: {usable_mb:.0f}MB")

        # 决策
        if usable_mb >= total_estimated_mb:
            self.logger.info("选择【内存模式】- 内存充足，使用高性能模式")
            job.message = "内存充足，使用高性能模式"
            return ProcessingMode.MEMORY
        else:
            self.logger.warning(f"选择【硬盘模式】- 内存不足（需要{total_estimated_mb:.0f}MB，可用{usable_mb:.0f}MB）")
            job.message = "内存受限，使用稳定模式"
            return ProcessingMode.DISK

    def _safe_load_audio(self, audio_path: str, job: JobState) -> np.ndarray:
        """
        安全加载音频到内存（带异常处理）

        用于内存模式下将完整音频一次性加载到内存中。
        包含加载验证和详细的异常处理，加载失败时抛出RuntimeError触发降级。

        Args:
            audio_path: 音频文件路径
            job: 任务状态对象（用于更新状态消息）

        Returns:
            np.ndarray: 音频数组（float32, 16kHz采样率）

        Raises:
            RuntimeError: 音频加载失败时抛出，调用方可据此触发硬盘模式降级
        """
        try:
            self.logger.info(f"加载音频到内存: {audio_path}")
            # 使用 whisper_service 提供的 load_audio 函数加载音频
            audio_array = whisper_load_audio(audio_path)

            # 验证加载结果
            if audio_array is None or len(audio_array) == 0:
                raise ValueError("音频数组为空")

            # 记录加载信息
            duration_sec = len(audio_array) / 16000
            memory_mb = audio_array.nbytes / (1024 * 1024)
            # Audio loaded: {duration_sec/60:.1f}分钟")
            self.logger.info(f"内存占用: {memory_mb:.1f}MB")
            self.logger.info(f"采样点数: {len(audio_array):,}")

            return audio_array

        except MemoryError as e:
            self.logger.error(f"内存不足，无法加载音频: {e}")
            job.message = "内存不足，自动切换到硬盘模式"
            raise RuntimeError(f"内存不足: {e}")

        except Exception as e:
            self.logger.error(f"音频加载失败: {e}")
            job.message = f"音频加载失败: {e}"
            raise RuntimeError(f"音频加载失败（可能文件损坏）: {e}")

    # ==========================================
    # Demucs 人声分离相关方法
    # ==========================================

    def _detect_bgm(self, audio_path: str, job: JobState):
        """
        执行BGM检测，更新进度
        
        【v2.1 重构】使用频谱分诊替代旧的分位数采样 + Demucs 检测
        新方法：纯频谱特征分析，无需运行 Demucs，速度更快且更准确

        Args:
            audio_path: 音频文件路径
            job: 任务状态对象

        Returns:
            Tuple[BGMLevel, List[float]]: (BGM强度级别, 各采样点的音乐得分列表)
        """
        from app.services.demucs_service import BGMLevel
        from app.services.audio_spectrum_classifier import get_spectrum_classifier
        import librosa

        self._update_progress(job, 'bgm_detect', 0, 'BGM检测中（频谱分诊）...')

        try:
            # 加载音频
            audio_array, sr = librosa.load(audio_path, sr=16000)
            
            # 使用频谱分诊器进行快速全局预判
            from app.services.runtime_param_resolver import get_demucs_runtime_params

            spectrum_classifier = get_spectrum_classifier()
            runtime_demucs = get_demucs_runtime_params()
            sample_duration = runtime_demucs.get("bgm_sample_duration", 10.0)
            level_str, avg_score = spectrum_classifier.quick_global_diagnosis(
                audio_array, sr=sr, sample_duration=sample_duration
            )
            
            # 转换为 BGMLevel 枚举
            level_map = {
                "none": BGMLevel.NONE,
                "light": BGMLevel.LIGHT,
                "heavy": BGMLevel.HEAVY,
                "unknown": BGMLevel.LIGHT  # unknown 保守处理为 light
            }
            level = level_map.get(level_str, BGMLevel.LIGHT)
            
            # 构造兼容的 ratios 列表（用于日志和 SSE）
            # 频谱分诊返回的是音乐得分，这里用 avg_score 填充
            ratios = [avg_score, avg_score, avg_score]

            self._update_progress(job, 'bgm_detect', 1, f'BGM检测完成: {level.value}')

            # 推送SSE事件
            self._push_sse_bgm_detected(job, level, ratios)

            self.logger.info(
                f"BGM检测结果（频谱分诊）: {level.value}, "
                f"音乐得分={avg_score:.2f}"
            )

            return level, ratios

        except Exception as e:
            self.logger.warning(f"BGM检测失败，将跳过Demucs: {e}")
            # 失败时返回 NONE 级别，不影响主流程
            from app.services.demucs_service import BGMLevel
            return BGMLevel.NONE, []
    
    def _separate_vocals_global(self, audio_path: str, job: JobState) -> str:
        """
        执行全局人声分离，更新进度

        Args:
            audio_path: 原始音频路径
            job: 任务状态对象

        Returns:
            str: 分离后的人声文件路径
        """
        from app.services.demucs_service import get_demucs_service

        self._update_progress(job, 'demucs_global', 0, '人声分离中...')

        try:
            demucs = get_demucs_service()

            def progress_callback(progress: float, message: str):
                """进度回调"""
                self._update_progress(job, 'demucs_global', progress, message)

            # 执行人声分离
            vocals_path = demucs.separate_vocals(
                audio_path,
                progress_callback=progress_callback
            )

            self._update_progress(job, 'demucs_global', 1, '人声分离完成')
            self.logger.info(f"全局人声分离完成: {vocals_path}")

            return vocals_path

        except Exception as e:
            self.logger.error(f"人声分离失败: {e}")
            # 分离失败时返回原始音频路径（降级处理）
            self.logger.warning("人声分离失败，将使用原始音频继续处理")
            return audio_path

    def _push_sse_bgm_detected(self, job: JobState, level, ratios):
        """
        推送BGM检测结果事件

        Args:
            job: 任务状态对象
            level: BGM强度级别
            ratios: 各采样点的BGM比例列表
        """
        try:
            from app.services.sse_service import get_sse_manager

            sse_manager = get_sse_manager()
            channel_id = f"job:{job.job_id}"

            # 将numpy类型转换为Python原生类型，避免JSON序列化错误
            native_ratios = [float(r) for r in ratios] if ratios else []
            max_ratio = float(max(ratios)) if ratios else 0.0

            # 构造事件数据
            event_data = {
                "level": level.value,
                "ratios": native_ratios,
                "max_ratio": max_ratio,
                "recommendation": self._get_demucs_recommendation(level)
            }

            # 广播事件
            sse_manager.broadcast_sync(channel_id, "signal.bgm_detected", event_data)

        except Exception as e:
            # SSE推送失败不应影响主流程
            self.logger.debug(f"SSE推送失败（非致命）: {e}")

    def _push_sse_separation_strategy(self, job: JobState, strategy):
        """
        推送分离策略决策事件

        Args:
            job: 任务状态对象
            strategy: SeparationStrategy 对象
        """
        try:
            from app.services.sse_service import get_sse_manager

            sse_manager = get_sse_manager()
            channel_id = f"job:{job.job_id}"

            # 使用 strategy.to_dict() 获取事件数据
            event_data = strategy.to_dict()

            # 广播事件
            sse_manager.broadcast_sync(channel_id, "signal.separation_strategy", event_data)

            self.logger.debug(f"分离策略事件已推送: {strategy.reason}")

        except Exception as e:
            # SSE推送失败不应影响主流程
            self.logger.debug(f"SSE推送失败（非致命）: {e}")

    def _push_sse_model_escalated(
        self,
        job: JobState,
        from_model: str,
        to_model: str,
        reason: str,
        breaker_state: CircuitBreakerState
    ):
        """
        推送模型升级事件

        Args:
            job: 任务状态对象
            from_model: 原模型名称
            to_model: 新模型名称
            reason: 升级原因
            breaker_state: 熔断器状态
        """
        try:
            from app.services.sse_service import get_sse_manager

            sse_manager = get_sse_manager()
            channel_id = f"job:{job.job_id}"

            preprocessing = getattr(job.settings, "preprocessing", None)
            max_escalations = max(1, getattr(preprocessing, "demucs_shifts", 1))

            # 构造事件数据
            event_data = {
                "from_model": from_model,
                "to_model": to_model,
                "reason": reason,
                "escalation_count": breaker_state.escalation_count,
                "max_escalations": max_escalations,
                "stats": breaker_state.get_stats()
            }

            # 广播事件
            sse_manager.broadcast_sync(channel_id, "signal.model_upgrade", event_data)

            self.logger.info(f"模型升级事件已推送: {from_model} -> {to_model}")

        except Exception as e:
            # SSE推送失败不应影响主流程
            self.logger.debug(f"SSE推送失败（非致命）: {e}")

    def _push_sse_circuit_breaker_triggered(
        self,
        job: JobState,
        circuit_breaker: Optional[CircuitBreakerState]
    ):
        """
        推送熔断触发事件（Phase 4: 扩展包含升级信息）

        Args:
            job: 任务状态对象
            circuit_breaker: 熔断器状态对象
        """
        try:
            from app.services.sse_service import get_sse_manager

            sse_manager = get_sse_manager()
            channel_id = f"job:{job.job_id}"

            # 构造事件数据（扩展包含升级历史）
            stats = circuit_breaker.get_stats() if circuit_breaker else {}
            event_data = {
                "triggered": True,
                "reason": self._get_circuit_break_reason(circuit_breaker),
                "stats": stats,
                "action": "升级为全局人声分离模式"
            }

            # 广播事件
            sse_manager.broadcast_sync(channel_id, "signal.circuit_breaker", event_data)

            self.logger.info("熔断事件已推送到前端")

        except Exception as e:
            # SSE推送失败不应影响主流程
            self.logger.debug(f"SSE推送失败（非致命）: {e}")

    def _get_circuit_break_reason(self, circuit_breaker: Optional[CircuitBreakerState]) -> str:
        """
        生成熔断原因描述

        Args:
            circuit_breaker: 熔断器状态

        Returns:
            熔断原因描述字符串
        """
        if not circuit_breaker:
            return "转录质量低，触发熔断升级"

        stats = circuit_breaker.get_stats()

        # 如果有升级历史，说明已经尝试过升级
        if circuit_breaker.escalation_count > 0:
            return (
                f"已升级 {circuit_breaker.escalation_count} 次模型仍未改善，触发熔断。"
                f"升级历史: {', '.join(circuit_breaker.escalation_history)}"
            )
        else:
            return (
                f"连续 {stats['consecutive_retries']} 个段落重试失败，"
                f"总重试率 {stats['retry_ratio']:.1%}，触发熔断"
            )

    def _get_demucs_recommendation(self, level) -> str:
        """
        根据BGM级别返回建议的处理模式

        Args:
            level: BGM强度级别

        Returns:
            str: 建议的处理模式描述
        """
        from app.services.demucs_service import BGMLevel

        if level == BGMLevel.HEAVY:
            return "全局分离"
        elif level == BGMLevel.LIGHT:
            return "按需分离"
        else:
            return "无需分离"

    # ==========================================
    # 按需分离与熔断机制相关方法
    # ==========================================

    def _check_transcription_confidence(
        self,
        result: Dict,
        logprob_threshold: float,
        no_speech_threshold: float
    ) -> bool:
        """
        检查转录结果的置信度

        Args:
            result: 转录结果字典
            logprob_threshold: logprob阈值（低于此值需要重试）
            no_speech_threshold: no_speech_prob阈值（高于此值需要重试）

        Returns:
            bool: True表示置信度低，需要重试
        """
        segments = result.get('segments', [])

        if not segments:
            return True  # 没有识别出内容，需要重试

        # 计算平均置信度
        total_logprob = 0
        total_no_speech = 0
        count = 0

        for seg in segments:
            if 'avg_logprob' in seg:
                total_logprob += seg['avg_logprob']
                count += 1
            if 'no_speech_prob' in seg:
                total_no_speech += seg['no_speech_prob']

        if count == 0:
            return False  # 没有置信度信息，不重试

        avg_logprob = total_logprob / count
        avg_no_speech = total_no_speech / count if count > 0 else 0

        # 判断是否需要重试
        if avg_logprob < logprob_threshold:
            self.logger.debug(f"avg_logprob={avg_logprob:.2f} < {logprob_threshold}, 需要重试")
            return True

        if avg_no_speech > no_speech_threshold:
            self.logger.debug(f"no_speech_prob={avg_no_speech:.2f} > {no_speech_threshold}, 需要重试")
            return True

        return False

    def _is_better_result(self, new_result: Dict, old_result: Dict) -> bool:
        """
        比较两个转录结果，判断新结果是否更好

        Args:
            new_result: 新的转录结果
            old_result: 旧的转录结果

        Returns:
            bool: True表示新结果更好
        """
        new_segments = new_result.get('segments', [])
        old_segments = old_result.get('segments', [])

        # 如果新结果没有内容，旧的更好
        if not new_segments:
            return False

        # 如果旧结果没有内容，新的更好
        if not old_segments:
            return True

        # 比较平均logprob
        def get_avg_logprob(segments):
            logprobs = [s.get('avg_logprob', -1) for s in segments if 'avg_logprob' in s]
            return np.mean(logprobs) if logprobs else -1

        new_logprob = get_avg_logprob(new_segments)
        old_logprob = get_avg_logprob(old_segments)

        # 新结果的logprob更高（更接近0）则更好
        return new_logprob > old_logprob

    def _build_whisper_transcribe_params(
        self,
        job: JobState,
        overrides: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """构建 Whisper 推理参数（运行参数 + 局部覆盖）。"""
        from app.config.model_config import get_whisper_suppress_tokens
        from app.services.model_manager_v2 import get_model_manager_v2
        from app.services.model_runtime_config_service import get_model_runtime_config_service
        from app.services.runtime_param_resolver import get_runtime_group
        from app.services.whisper_service import get_whisper_service

        transcription = getattr(job.settings, "transcription", None)
        model_name = getattr(transcription, "whisper_model", "medium")
        whisper_service = get_whisper_service()
        override_keys = {key for key, value in (overrides or {}).items() if value is not None}
        sources: Dict[str, str] = {}

        try:
            model_id = whisper_service.resolve_model_id(model_name)
            manager = get_model_manager_v2()
            spec = manager.registry.get(model_id)
            runtime_data = get_model_runtime_config_service().get_effective_runtime_for_model(spec)
            runtime = runtime_data.get("effective", {})
            sources = runtime_data.get("sources", {})
        except Exception as exc:
            self.logger.debug("Whisper 运行参数回退分组: %s", exc)
            runtime = get_runtime_group("whisper")

        params: Dict[str, Any] = {
            "language": runtime.get("language"),
            "initial_prompt": runtime.get("initial_prompt"),
            "word_timestamps": runtime.get("word_timestamps"),
            "beam_size": runtime.get("beam_size"),
            "vad_filter": runtime.get("vad_filter"),
            "vad_parameters": runtime.get("vad_parameters"),
            "temperature": runtime.get("temperature"),
            "condition_on_previous_text": runtime.get("condition_on_previous_text"),
            "suppress_tokens": runtime.get("suppress_tokens"),
            "repetition_penalty": runtime.get("repetition_penalty"),
            "no_repeat_ngram_size": runtime.get("no_repeat_ngram_size"),
        }

        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    params[key] = value

        language = params.get("language")
        if language is None or language == "auto" or language == "":
            params["language"] = None

        if params.get("word_timestamps") is None:
            params["word_timestamps"] = False
        if params.get("beam_size") is None:
            params["beam_size"] = 5
        if params.get("vad_filter") is None:
            params["vad_filter"] = True
        if params.get("temperature") is None:
            params["temperature"] = 0.0
        if params.get("condition_on_previous_text") is None:
            params["condition_on_previous_text"] = True
        if params.get("repetition_penalty") is None:
            params["repetition_penalty"] = 1.0
        if params.get("no_repeat_ngram_size") is None:
            params["no_repeat_ngram_size"] = 0

        if context == "patch":
            if not sources:
                if "word_timestamps" not in override_keys:
                    params["word_timestamps"] = False
                if "condition_on_previous_text" not in override_keys:
                    params["condition_on_previous_text"] = False
            else:
                if (
                    "word_timestamps" not in override_keys
                    and sources.get("word_timestamps") == "default"
                ):
                    params["word_timestamps"] = False
                if (
                    "condition_on_previous_text" not in override_keys
                    and sources.get("condition_on_previous_text") == "default"
                ):
                    params["condition_on_previous_text"] = False

        if params.get("suppress_tokens") is None:
            suppress_tokens = get_whisper_suppress_tokens(model_name)
            params["suppress_tokens"] = suppress_tokens if suppress_tokens else None

        return params

    def _transcribe_segment_unaligned(
        self,
        seg: Dict,
        model,
        job: JobState
    ) -> Optional[Dict]:
        """
        转录单个音频段（仅转录，不对齐）

        Args:
            seg: 段信息 {file, start_ms, duration_ms, index}
            model: Faster-Whisper 模型
            job: 任务状态

        Returns:
            Dict: 未对齐的转录结果
            {
                "segment_index": 0,
                "language": "zh",
                "segments": [{"id": 0, "start": 10.5, "end": 15.2, "text": "..."}]
            }
        """
        # 使用 whisper_service 提供的 load_audio 函数
        audio = whisper_load_audio(seg['file'])

        try:
            # 使用 Faster-Whisper 转录
            params = self._build_whisper_transcribe_params(
                job,
                overrides={"language": job.language, "vad_filter": True},
            )
            segments_gen, info = model.transcribe(audio, **params)

            # 转换生成器为列表
            segments_list = list(segments_gen)

            if not segments_list:
                return None

            # 检测语言（首次）
            if not job.language and info.language:
                job.language = info.language
                self.logger.info(f"检测到语言: {job.language}")

            # 时间偏移校正（针对粗略时间戳）
            start_offset = seg['start_ms'] / 1000.0
            adjusted_segments = []

            for idx, s in enumerate(segments_list):
                adjusted_segments.append({
                    'id': idx,
                    'start': s.start + start_offset,
                    'end': s.end + start_offset,
                    'text': s.text.strip()
                })

            return {
                'segment_index': seg.get('index', 0),
                'language': info.language or job.language,
                'segments': adjusted_segments
            }

        finally:
            del audio
            gc.collect()

    def _transcribe_segment_in_memory(
        self,
        audio_array: np.ndarray,
        seg_meta: Dict,
        model,
        job: JobState,
        is_vocals: bool = False
    ) -> Optional[Dict]:
        """
        从内存切片转录（Zero-copy，高性能）

        内存模式下使用，直接从完整音频数组中切片，无需磁盘IO。

        Args:
            audio_array: 完整音频数组
            seg_meta: 分段元数据 {"index": 0, "start": 0.0, "end": 30.5, "mode": "memory"}
            model: Faster-Whisper 模型
            job: 任务状态
            is_vocals: 是否是Demucs分离后的人声（用于日志）

        Returns:
            Dict: 未对齐的转录结果
        """
        sr = 16000
        start_sample = int(seg_meta['start'] * sr)
        end_sample = int(seg_meta['end'] * sr)

        # Zero-copy切片（numpy view，不复制数据）
        audio_slice = audio_array[start_sample:end_sample]

        try:
            # 使用 Faster-Whisper 转录
            params = self._build_whisper_transcribe_params(
                job,
                overrides={"language": job.language, "vad_filter": False},
            )
            segments_gen, info = model.transcribe(audio_slice, **params)

            # 转换生成器为列表
            segments_list = list(segments_gen)

            if not segments_list:
                return None

            # 检测语言（首次）
            if not job.language and info.language:
                job.language = info.language
                self.logger.info(f"detected language: {job.language}")

            # 时间偏移校正
            start_offset = seg_meta['start']
            adjusted_segments = []

            for idx, s in enumerate(segments_list):
                adjusted_segments.append({
                    'id': idx,
                    'start': s.start + start_offset,
                    'end': s.end + start_offset,
                    'text': s.text.strip()
                })

            return {
                'segment_index': seg_meta['index'],
                'language': info.language or job.language,
                'segments': adjusted_segments
            }

        finally:
            # 注意：audio_slice是view，不需要单独释放
            gc.collect()

    def _transcribe_segment_from_disk(
        self,
        seg: Dict,
        model,
        job: JobState
    ) -> Optional[Dict]:
        """
        从文件加载转录（硬盘模式）

        硬盘模式下使用，从segment文件加载音频进行转录。

        Args:
            seg: 分段信息 {"index": 0, "file": "segment_0.wav", "start": 0.0, "end": 30.0, "mode": "disk"}
            model: Faster-Whisper 模型
            job: 任务状态

        Returns:
            Dict: 未对齐的转录结果
        """
        # 使用 whisper_service 提供的 load_audio 函数
        audio = whisper_load_audio(seg['file'])

        try:
            # 使用 Faster-Whisper 转录
            params = self._build_whisper_transcribe_params(
                job,
                overrides={"language": job.language, "vad_filter": True},
            )
            segments_gen, info = model.transcribe(audio, **params)

            # 转换生成器为列表
            segments_list = list(segments_gen)

            if not segments_list:
                return None

            # 检测语言（首次）
            if not job.language and info.language:
                job.language = info.language
                self.logger.info(f"detected language: {job.language}")

            # 时间偏移校正（使用start字段，秒为单位）
            start_offset = seg.get('start', seg.get('start_ms', 0) / 1000.0)
            adjusted_segments = []

            for idx, s in enumerate(segments_list):
                adjusted_segments.append({
                    'id': idx,
                    'start': s.start + start_offset,
                    'end': s.end + start_offset,
                    'text': s.text.strip()
                })

            return {
                'segment_index': seg['index'],
                'language': info.language or job.language,
                'segments': adjusted_segments
            }

        finally:
            del audio
            gc.collect()

    def _transcribe_segment(
        self,
        seg_meta: Dict,
        model,
        job: JobState,
        audio_array: Optional[np.ndarray] = None
    ) -> Optional[Dict]:
        """
        统一转录入口（根据模式自动选择）

        Args:
            seg_meta: 分段元数据
            model: Whisper模型
            job: 任务状态
            audio_array: 音频数组（内存模式时必须提供）

        Returns:
            Dict: 未对齐的转录结果
        """
        mode = seg_meta.get('mode', 'disk')

        if mode == 'memory':
            if audio_array is None:
                raise ValueError("memory mode requires audio_array parameter")
            return self._transcribe_segment_in_memory(audio_array, seg_meta, model, job)
        else:
            return self._transcribe_segment_from_disk(seg_meta, model, job)

    def _check_memory_during_transcription(self, job: JobState) -> bool:
        """
        转录过程中检查内存状态

        如果内存严重不足，暂停任务并警告用户。

        Args:
            job: 任务状态对象

        Returns:
            bool: True=继续处理，False=需要暂停
        """
        mem_info = psutil.virtual_memory()
        available_mb = mem_info.available / (1024 * 1024)
        percent_used = mem_info.percent

        # 危险阈值：可用内存<500MB 或 使用率>95%
        if available_mb < 500 or percent_used > 95:
            self.logger.error(f"memory critically low! available: {available_mb:.0f}MB, usage: {percent_used}%")
            job.status = 'paused'
            job.message = f"memory insufficient (available {available_mb:.0f}MB), please close other programs"
            job.paused = True

            # 推送警告SSE
            self._push_sse_signal(job, "memory_warning",
                f"memory critically low (available {available_mb:.0f}MB), task paused")

            return False

        # 警告阈值：可用内存<1GB 或 使用率>90%
        if available_mb < 1024 or percent_used > 90:
            self.logger.warning(f"memory tight: available {available_mb:.0f}MB, usage {percent_used}%")
            # 不暂停，但记录警告

        return True

    def _align_all_results(
        self,
        unaligned_results: List[Dict],
        job: JobState,
        audio_path: str
    ) -> List[Dict]:
        """
        合并转录结果的分段

        此方法合并所有 segments 并返回，不执行对齐操作。
        """
        self.logger.info(f"合并 {len(unaligned_results)} 个分段的转录结果（跳过强制对齐）")

        # 合并所有segments
        all_segments = []
        for result in unaligned_results:
            all_segments.extend(result['segments'])

        if not all_segments:
            self.logger.warning("没有可处理的内容")
            return []

        # 直接返回合并后的结果（Faster-Whisper 已提供时间戳）
        return [{
            'segments': all_segments,
            'word_segments': []  # 新架构使用伪对齐生成字级时间戳
        }]

    def _push_sse_align_progress(
        self,
        job: JobState,
        current_batch: int,
        total_batches: int,
        aligned_count: int,
        total_count: int
    ):
        """
        推送对齐进度SSE事件（前端进度条实时更新）

        事件类型: "align_progress"

        Args:
            job: 任务状态对象
            current_batch: 当前批次号（1-based）
            total_batches: 总批次数
            aligned_count: 已对齐的segment数量
            total_count: 总segment数量
        """
        try:
            from app.services.sse_service import get_sse_manager
            sse_manager = get_sse_manager()

            channel_id = f"job:{job.job_id}"

            # 计算百分比
            batch_progress = (current_batch / total_batches) * 100 if total_batches > 0 else 0
            segment_progress = (aligned_count / total_count) * 100 if total_count > 0 else 0

            sse_manager.broadcast_sync(
                channel_id,
                "progress.align",
                {
                    "job_id": job.job_id,
                    "phase": "align",
                    "batch": {
                        "current": current_batch,
                        "total": total_batches,
                        "progress": round(batch_progress, 2)
                    },
                    "segments": {
                        "aligned": aligned_count,
                        "total": total_count,
                        "progress": round(segment_progress, 2)
                    },
                    "message": f"aligning batch {current_batch}/{total_batches} ({aligned_count}/{total_count} segments)"
                }
            )

        except Exception as e:
            self.logger.debug(f"SSE align progress push failed (non-fatal): {e}")

    def _align_all_results_batched(
        self,
        unaligned_results: List[Dict],
        job: JobState,
        audio_source,  # Union[np.ndarray, str]
        processing_mode: ProcessingMode
    ) -> List[Dict]:
        """
        批量处理转录结果的分段

        此方法合并所有 segments，应用时间校验和微调，然后返回。
        """
        self.logger.info(f"处理 {len(unaligned_results)} 个分段的转录结果（跳过强制对齐）")

        # 1. 合并所有segments
        all_segments = []
        for result in unaligned_results:
            all_segments.extend(result['segments'])

        if not all_segments:
            self.logger.warning("没有可处理的内容")
            return []

        # 2. 对结果进行边界校验，过滤异常结果
        valid_segments = []
        for seg in all_segments:
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            text = seg.get('text', '').strip()

            # 校验1：时间戳必须有效
            if start is None or end is None or start < 0 or end <= start:
                self.logger.warning(f"过滤无效时间戳: start={start}, end={end}, text={text[:20] if text else ''}...")
                continue

            # 校验2：字幕时长不能过长（超过30秒可能是异常）
            duration = end - start
            if duration > 30:
                self.logger.warning(f"过滤过长字幕({duration:.1f}s): {text[:30] if text else ''}...")
                continue

            # 校验3：字幕时长不能过短（小于0.1秒可能是噪音）
            if duration < 0.1 and len(text) > 0:
                self.logger.warning(f"过滤过短字幕({duration:.2f}s): {text}")
                continue

            valid_segments.append(seg)

        # 3. 字幕时间微调 - 修正"抢先出现"问题
        valid_segments = self._adjust_subtitle_timing(valid_segments)

        self.logger.info(f"处理完成: {len(valid_segments)}/{len(all_segments)} 个有效字幕段")

        return [{
            'segments': valid_segments,
            'word_segments': []  # 新架构使用伪对齐生成字级时间戳
        }]

    def _adjust_subtitle_timing(
        self,
        segments: List[Dict],
        start_delay_ms: int = 25,
        end_padding_ms: int = 25
    ) -> List[Dict]:
        """
        字幕时间微调 - 修正对齐偏差

        问题背景：
        转录后的字幕可能出现时间偏差，需要适当调整。

        解决方案：
        1. 将字幕开始时间延后 start_delay_ms（默认25ms）
        2. 将字幕结束时间延后 end_padding_ms（默认25ms，给一点余量）
        3. 确保相邻字幕不重叠

        Args:
            segments: 对齐后的字幕列表
            start_delay_ms: 开始时间延迟（毫秒），推荐20-50ms
            end_padding_ms: 结束时间延长（毫秒），推荐20-50ms

        Returns:
            调整后的字幕列表
        """
        if not segments:
            return segments

        start_delay = start_delay_ms / 1000.0
        end_padding = end_padding_ms / 1000.0

        adjusted = []
        for i, seg in enumerate(segments):
            new_seg = seg.copy()
            old_start = seg.get('start', 0)
            old_end = seg.get('end', 0)

            # 延迟开始时间
            new_start = old_start + start_delay

            # 延长结束时间
            new_end = old_end + end_padding

            # 确保开始时间不超过结束时间
            if new_start >= new_end:
                new_start = old_start  # 回退到原始开始时间

            # 确保不与下一条字幕重叠
            if i < len(segments) - 1:
                next_start = segments[i + 1].get('start', float('inf'))
                # 下一条也会被延迟，所以比较时要考虑
                next_adjusted_start = next_start + start_delay
                if new_end > next_adjusted_start:
                    new_end = next_adjusted_start - 0.05  # 留50ms间隔

            # 确保结束时间仍然有效
            if new_end <= new_start:
                new_end = old_end  # 回退到原始结束时间

            new_seg['start'] = round(new_start, 3)
            new_seg['end'] = round(new_end, 3)
            adjusted.append(new_seg)

        self.logger.info(f"字幕时间微调: 延迟开始25ms, 延长结束25ms")
        return adjusted

    def _format_ts(self, sec: float) -> str:
        """
        格式化时间戳为SRT格式

        Args:
            sec: 秒数

        Returns:
            str: SRT时间戳 (HH:MM:SS,mmm)
        """
        if sec < 0:
            sec = 0

        ms = int(round(sec * 1000))
        h = ms // 3600000
        ms %= 3600000
        m = ms // 60000
        ms %= 60000
        s = ms // 1000
        ms %= 1000

        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    def _generate_srt(self, results: List[Dict], path: str, word_level: bool):
        """
        生成SRT字幕文件
        V3.1.1+dev.20260106.03: 生成前自动修复时间戳重叠

        Args:
            results: 转录结果列表
            path: 输出文件路径
            word_level: 是否使用词级时间戳
        """
        # V3.1.1+dev.20260106.03: 导入重叠修复函数
        from app.utils.text_utils import repair_timestamp_overlaps, detect_timestamp_overlaps

        all_entries = []

        for r in results:
            if not r:
                continue

            # 词级时间戳模式
            if word_level and r.get('word_segments'):
                for w in r['word_segments']:
                    if w.get('start') is not None and w.get('end') is not None:
                        txt = (w.get('word') or '').strip()
                        if txt:
                            all_entries.append({
                                'start': w['start'],
                                'end': w['end'],
                                'text': txt
                            })

            # 句子级时间戳模式（默认）
            elif r.get('segments'):
                for s in r['segments']:
                    if s.get('start') is not None and s.get('end') is not None:
                        txt = (s.get('text') or '').strip()
                        if txt:
                            all_entries.append({
                                'start': s['start'],
                                'end': s['end'],
                                'text': txt
                            })

        # 过滤无效时间戳
        all_entries = [e for e in all_entries if e['end'] > e['start']]

        # V3.1.1+dev.20260106.03: 检测并修复重叠
        if all_entries:
            overlaps = detect_timestamp_overlaps(all_entries)
            if overlaps:
                self.logger.warning(f"检测到 {len(overlaps)} 处时间戳重叠，自动修复中...")
                all_entries = repair_timestamp_overlaps(all_entries, gap_ms=1.0)
                self.logger.info(f"已修复 {len(overlaps)} 处时间戳重叠")

        # 写入SRT格式
        lines = []
        for n, e in enumerate(all_entries, 1):
            lines.append(str(n))  # 序号
            lines.append(
                f"{self._format_ts(e['start'])} --> {self._format_ts(e['end'])}"
            )  # 时间戳
            lines.append(e['text'])  # 字幕文本
            lines.append("")  # 空行

        # 写入文件
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        self.logger.info(f"SRT文件已生成: {path}, 共{len(all_entries)}条字幕")

    def clear_model_cache(self):
        """
        清空模型缓存（供队列服务调用）

        注意: 新架构已移除对齐模型，仅清理 Whisper 模型
        """
        from app.services.model_manager_v2 import get_model_manager_v2

        manager = get_model_manager_v2()
        manager.unload_all()
        self.logger.info("ModelManagerV2 缓存已清空")

    # ==========================================
    # SenseVoice 集成方法（Phase 3）
    # ==========================================

    def _sensevoice_transcribe(
        self,
        audio_array: np.ndarray,
        job: 'JobState',
        sample_rate: int = 16000
    ) -> 'SenseVoiceResult':
        """
        调用 SenseVoice 服务进行转录（返回带真实字级时间戳的结果）

        Args:
            audio_array: 音频数组
            job: 任务状态对象
            sample_rate: 采样率

        Returns:
            SenseVoiceResult: 转录结果，包含:
                - text: 原始文本（带标签）
                - text_clean: 清洗后文本
                - words: 字级时间戳列表（真实时间戳，非伪对齐）
                - confidence: 平均置信度
                - language: 检测到的语言
                - emotion: 情感标签
                - event: 事件标签
        """
        from app.services.sensevoice_onnx_service import get_sensevoice_service
        from app.models.sensevoice_models import SenseVoiceResult, WordTimestamp

        self.logger.debug("调用 SenseVoice 转录服务")

        try:
            service = get_sensevoice_service()

            # 确保模型已加载
            if not service.is_loaded:
                self.logger.info("加载 SenseVoice 模型...")
                service.load_model()

            # 调用转录（返回字典）
            result_dict = service.transcribe_audio_array(
                audio_array=audio_array,
                sample_rate=sample_rate
            )

            # 转换为 SenseVoiceResult 对象
            words = [
                WordTimestamp(**w) if isinstance(w, dict) else w
                for w in result_dict.get('words', [])
            ]

            result = SenseVoiceResult(
                text=result_dict.get('text', ''),
                text_clean=result_dict.get('text_clean', ''),
                confidence=result_dict.get('confidence', 1.0),
                words=words,
                start=0.0,  # Chunk 级别的起始时间，由调用者设置
                end=len(audio_array) / sample_rate,
                language=result_dict.get('language'),
                emotion=result_dict.get('emotion'),
                event=result_dict.get('event'),
                raw_result=result_dict
            )

            self.logger.debug(f"SenseVoice 转录完成: {len(result.text_clean)} 字符, {len(words)} 个字")
            return result

        except Exception as e:
            self.logger.error(f"SenseVoice 转录失败: {e}")
            raise

    def _split_sentences(
        self,
        sv_result: 'SenseVoiceResult',
        chunk_start_time: float = 0.0,
        split_config: Optional['SplitConfig'] = None,  # Layer 1: 允许传入配置
        enable_grouping: bool = True                    # Layer 2: 是否启用语义分组 (预留)
    ) -> List['SentenceSegment']:
        """
        将 SenseVoice 结果切分为句子（基于真实字级时间戳）

        Args:
            sv_result: SenseVoice 转录结果（包含真实字级时间戳）
            chunk_start_time: Chunk 在完整音频中的起始时间（用于时间偏移）
            split_config: 分句配置（Layer 1 新增，可选）
            enable_grouping: 是否启用语义分组（Layer 2 预留，可选）

        Returns:
            List[SentenceSegment]: 句子列表
        """
        from app.models.sensevoice_models import SentenceSegment, TextSource
        from app.services.sentence_splitter import SentenceSplitter, SplitConfig

        self.logger.debug(f"开始句子切分: {len(sv_result.words)} 个字")

        if not sv_result.words:
            return []

        # Layer 1: 使用传入的配置或默认配置
        config = split_config or SplitConfig()
        splitter = SentenceSplitter(config)
        sentences = splitter.split(sv_result.words, sv_result.text_clean)

        # Layer 2: 语义分组 (可选，已实现)
        if enable_grouping:
            from app.services.semantic_grouper import SemanticGrouper, GroupConfig
            grouper = SemanticGrouper(GroupConfig(language=config.language))
            sentences = grouper.group(sentences)

        # 【阶段三】VAD 边缘吸附 (Head Snap)
        # 逻辑：如果是 Chunk 的第一句话，且 CTC 延迟在合理范围内（<0.6s），
        # 强制将其 start 对齐到 Chunk 的物理起始点 (0.0 相对时间)
        HEAD_SNAP_THRESHOLD = 0.6  # 最大允许吸附的延迟（秒）

        if sentences:
            first_sent = sentences[0]
            # 检查第一个词的相对开始时间（相对于 chunk）
            # 如果 > 0 且 < 阈值，说明 CTC 有延迟，需要吸附到 VAD 边缘
            if 0 < first_sent.start < HEAD_SNAP_THRESHOLD:
                self.logger.debug(
                    f"Head Snap: '{first_sent.text[:15]}...' 延迟修正 "
                    f"{first_sent.start:.3f}s -> 0.0s (相对VAD)"
                )
                # 修正句子开始时间
                first_sent.start = 0.0
                # 同时修正第一个单词的开始时间（保持一致性）
                if first_sent.words:
                    first_sent.words[0].start = 0.0

        # 调整时间偏移（将 Chunk 内的相对时间转换为绝对时间）
        for sentence in sentences:
            sentence.start += chunk_start_time
            sentence.end += chunk_start_time
            sentence.source = TextSource.SENSEVOICE
            # V3.1.2+dev.20260111.01: 使用 update_confidence 确保 display_confidence 同步更新
            sentence.update_confidence(sv_result.confidence, source="sensevoice")

            # 调整字级时间戳的偏移
            for word in sentence.words:
                word.start += chunk_start_time
                word.end += chunk_start_time

        self.logger.debug(f"句子切分完成: {len(sentences)} 句")
        return sentences

    def _split_text_by_punctuation(self, text: str) -> List[str]:
        """
        基于标点符号切分文本

        Args:
            text: 原始文本

        Returns:
            List[str]: 句子列表
        """
        import re

        # 句末标点
        sentence_end_pattern = r'([。？！.?!])'

        # 使用正则切分，保留标点
        parts = re.split(sentence_end_pattern, text)

        # 合并标点到前一个句子
        sentences = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and re.match(sentence_end_pattern, parts[i + 1]):
                sentences.append(parts[i] + parts[i + 1])
                i += 2
            else:
                if parts[i].strip():
                    sentences.append(parts[i])
                i += 1

        return [s.strip() for s in sentences if s.strip()]

    def _generate_pseudo_word_timestamps(
        self,
        text: str,
        start: float,
        end: float,
        confidence: float = 1.0
    ) -> List['WordTimestamp']:
        """
        生成伪字级时间戳（均匀分布）

        Args:
            text: 句子文本
            start: 句子开始时间
            end: 句子结束时间
            confidence: 置信度

        Returns:
            List[WordTimestamp]: 字级时间戳列表
        """
        from app.models.sensevoice_models import WordTimestamp

        if not text:
            return []

        duration = end - start
        char_duration = duration / len(text)

        words = []
        for i, char in enumerate(text):
            word_start = start + i * char_duration
            word_end = start + (i + 1) * char_duration

            words.append(WordTimestamp(
                word=char,
                start=word_start,
                end=word_end,
                confidence=confidence,
                is_pseudo=True  # 标记为伪对齐
            ))

        return words

    def _generate_subtitle_from_sentences(
        self,
        sentences: List['SentenceSegment'],
        output_path: str,
        include_translation: bool = False
    ) -> str:
        """
        从句子列表生成 SRT 字幕文件

        Args:
            sentences: 句子列表
            output_path: 输出文件路径
            include_translation: 是否包含翻译（双语字幕）

        Returns:
            str: 生成的SRT文件路径
        """
        self.logger.info(f"生成SRT字幕: {len(sentences)} 句 -> {output_path}")

        lines = []
        for idx, sentence in enumerate(sentences, 1):
            # 序号
            lines.append(str(idx))

            # 时间戳
            start_ts = self._format_ts(sentence.start)
            end_ts = self._format_ts(sentence.end)
            lines.append(f"{start_ts} --> {end_ts}")

            # 字幕文本（使用清洗后的文本）
            if include_translation and sentence.translation:
                # 双语字幕：原文 + 翻译
                lines.append(sentence.text_clean or sentence.text)
                lines.append(sentence.translation)
            else:
                lines.append(sentence.text_clean or sentence.text)

            # 空行分隔
            lines.append("")

        # 写入文件
        # 确保父目录存在
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        self.logger.info(f"SRT字幕生成完成: {output_path}")
        return output_path

    def _save_raw_transcription(
        self,
        job: 'JobState',
        sentences: List['SentenceSegment']
    ):
        """
        保存原始转录数据（未经 LLM 处理的 SenseVoice 原始输出）

        Args:
            job: 任务状态
            sentences: 句子列表
        """
        import json
        from pathlib import Path

        try:
            # 准备保存数据
            raw_data = {
                "job_id": job.job_id,
                "filename": job.filename,
                "timestamp": self._get_current_timestamp(),
                "total_sentences": len(sentences),
                "sentences": []
            }

            # 收集所有句子的原始数据
            for idx, sentence in enumerate(sentences):
                sentence_data = {
                    "index": idx,
                    "start": sentence.start,
                    "end": sentence.end,
                    "text": sentence.text,  # 包含原始标签的文本
                    "confidence": sentence.confidence,
                    "source": sentence.source.value if hasattr(sentence.source, 'value') else str(sentence.source),
                    "words": []
                }

                # 添加字级时间戳
                if hasattr(sentence, 'words') and sentence.words:
                    for word in sentence.words:
                        word_data = {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "confidence": word.confidence,
                            "is_pseudo": word.is_pseudo
                        }
                        sentence_data["words"].append(word_data)

                raw_data["sentences"].append(sentence_data)

            # 保存到文件
            output_path = Path(job.dir) / f"{job.job_id}_raw_transcription.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"原始转录数据已保存: {output_path}")

        except Exception as e:
            self.logger.error(f"保存原始转录数据失败: {e}", exc_info=True)

    def _get_current_timestamp(self) -> str:
        """获取当前时间戳字符串"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _init_chunk_states(
        self,
        vad_segments: List[dict],
        audio_array: np.ndarray,
        sr: int = 16000
    ) -> List['ChunkProcessState']:
        """
        初始化所有 Chunk 的处理状态

        关键：保存原始音频引用，用于熔断回溯

        Args:
            vad_segments: VAD 切分结果
            audio_array: 完整音频数组
            sr: 采样率

        Returns:
            List[ChunkProcessState]: Chunk 状态列表
        """
        from app.models.circuit_breaker_models import ChunkProcessState, SeparationLevel

        self.logger.info(f"初始化 {len(vad_segments)} 个 Chunk 状态...")

        states = []
        for seg in vad_segments:
            start_sample = int(seg['start'] * sr)
            end_sample = int(seg['end'] * sr)
            chunk_audio = audio_array[start_sample:end_sample]

            state = ChunkProcessState(
                chunk_index=seg['index'],
                start_time=seg['start'],
                end_time=seg['end'],
                original_audio=chunk_audio.copy(),  # 关键：保存原始音频副本
                current_audio=chunk_audio,           # 当前使用的音频（可能被分离后替换）
                sample_rate=sr,
                separation_level=SeparationLevel.NONE
            )
            states.append(state)

        self.logger.info(f"Chunk 状态初始化完成")
        return states

    def _convert_audio_chunks_to_states(
        self,
        audio_chunks: List['AudioChunk']
    ) -> List['ChunkProcessState']:
        """
        将 AudioChunk 列表转换为 ChunkProcessState 列表

        用于将新架构 PreprocessingPipeline 的输出转换为旧架构的数据结构，
        以便复用现有的 _transcribe_chunk_with_fusing 方法。

        Args:
            audio_chunks: PreprocessingPipeline 输出的 AudioChunk 列表

        Returns:
            List[ChunkProcessState]: 转换后的 Chunk 状态列表
        """
        from app.models.circuit_breaker_models import ChunkProcessState
        from app.services.audio.chunk_engine import AudioChunk

        self.logger.info(f"转换 {len(audio_chunks)} 个 AudioChunk 到 ChunkProcessState...")

        states = []
        for chunk in audio_chunks:
            state = ChunkProcessState(
                chunk_index=chunk.index,
                start_time=chunk.start,
                end_time=chunk.end,
                # 音频引用：original_audio 用于熔断回溯
                original_audio=chunk.original_audio if chunk.original_audio is not None else chunk.audio.copy(),
                current_audio=chunk.audio,
                sample_rate=chunk.sample_rate,
                # 分离状态
                separation_level=chunk.separation_level,
                separation_model_used=chunk.separation_model,
                # 熔断状态
                fuse_retry_count=chunk.fuse_retry_count,
            )
            states.append(state)

        self.logger.info(f"AudioChunk 转换完成: {len(states)} 个 ChunkProcessState")
        return states

    async def _transcribe_chunk_with_fusing(
        self,
        chunk_state: 'ChunkProcessState',
        job: 'JobState',
        subtitle_manager: 'StreamingSubtitleManager',
        demucs_service
    ) -> List['SentenceSegment']:
        """
        单个 Chunk 的转录流程（含熔断回溯）

        流程：
        1. 使用 current_audio 进行 SenseVoice 转录
        2. 评估置信度和事件标签
        3. 熔断决策
        4. 如需熔断：回溯到 original_audio，升级分离，重新转录
        5. 止损点：max_retry=1

        Args:
            chunk_state: Chunk 处理状态
            job: 任务状态对象
            subtitle_manager: 流式字幕管理器
            demucs_service: Demucs 服务实例

        Returns:
            List[SentenceSegment]: 句子列表
        """
        from app.services.fuse_breaker import get_fuse_breaker, execute_fuse_upgrade, FuseAction

        fuse_breaker = get_fuse_breaker()

        while True:
            # 1. SenseVoice 转录
            sv_result = self._sensevoice_transcribe(chunk_state.current_audio, job, chunk_state.sample_rate)

            # 2. 分句
            sentences = self._split_sentences(sv_result, chunk_state.start_time)

            # 3. 计算置信度和事件标签
            avg_confidence = sum(s.confidence for s in sentences) / len(sentences) if sentences else 0.0
            event_tag = sv_result.event  # SenseVoice 检测到的事件（BGM/Noise等）

            # 4. 熔断决策
            decision = fuse_breaker.should_fuse(
                chunk_state=chunk_state,
                confidence=avg_confidence,
                event_tag=event_tag
            )

            self.logger.debug(
                f"Chunk {chunk_state.chunk_index} 熔断决策: {decision.action.value}, "
                f"置信度={avg_confidence:.2f}, 事件={event_tag}"
            )

            # 5. 处理决策
            if decision.action == FuseAction.ACCEPT:
                # 接受结果，推送草稿事件（Phase 5: 双模态架构）
                # 使用 add_draft_sentences 批量推送，触发 subtitle.draft 事件
                subtitle_manager.add_draft_sentences(chunk_state.chunk_index, sentences)
                return sentences

            elif decision.action == FuseAction.UPGRADE_SEPARATION:
                # 熔断回溯：使用原始音频重新分离
                self.logger.info(
                    f"Chunk {chunk_state.chunk_index} 触发熔断，"
                    f"升级分离: {chunk_state.separation_level.value} → {decision.next_separation_level.value}"
                )

                chunk_state = execute_fuse_upgrade(
                    chunk_state=chunk_state,
                    next_level=decision.next_separation_level,
                    demucs_service=demucs_service
                )

                # 继续循环，使用升级后的音频重新转录
                continue

            else:
                # 未知动作，接受当前结果
                subtitle_manager.add_draft_sentences(chunk_state.chunk_index, sentences)
                return sentences

    async def _whisper_text_patch_with_arbitration(
        self,
        sentence: 'SentenceSegment',
        sentence_index: int,
        audio_array: np.ndarray,
        job: 'JobState',
        subtitle_manager: 'StreamingSubtitleManager',
        is_trash_suspect: bool = False
    ) -> 'SentenceSegment':
        """
        Whisper 复核 + 仲裁判决（二次安检机制）

        核心逻辑：
        - 如果是垃圾嫌疑样本，根据 Whisper 反馈判决去留
        - 如果是常规复核，直接采纳 Whisper 结果

        Args:
            sentence: 原始句子
            sentence_index: 句子索引
            audio_array: 完整音频数组
            job: 任务状态
            subtitle_manager: 流式字幕管理器
            is_trash_suspect: 是否是垃圾嫌疑样本

        Returns:
            SentenceSegment: 更新后的句子（或标记删除）
        """
        from app.services.whisper_service import get_whisper_service
        from app.models.sensevoice_models import TextSource

        whisper_service = get_whisper_service()

        # 确保 Whisper 模型已加载（首次运行会自动下载）
        if not whisper_service.is_loaded:
            self.logger.info("加载 Whisper 模型...")
            whisper_service.load_model()  # 自动下载 CTranslate2 格式模型

        # 提取对应时间段的音频（增加前向重叠，为 Whisper 提供上下文预热）
        sr = 16000
        WHISPER_OVERLAP_SEC = 0.5  # Whisper 上下文重叠时长（秒）

        # 计算重叠后的起始位置（不小于0）
        overlap_start = max(0.0, sentence.start - WHISPER_OVERLAP_SEC)
        start_sample = int(overlap_start * sr)
        end_sample = int(sentence.end * sr)
        audio_segment = audio_array[start_sample:end_sample]

        # 记录日志（调试用）
        if overlap_start < sentence.start:
            self.logger.debug(
                f"Whisper 复核添加 {sentence.start - overlap_start:.2f}s 前向重叠: "
                f"[{overlap_start:.2f}s, {sentence.end:.2f}s]"
            )

        # 获取上下文提示（现在使用清洗后的文本，避免下划线等原始 token）
        context = subtitle_manager.get_context_window(sentence_index)

        # Whisper 转录
        params = self._build_whisper_transcribe_params(
            job,
            overrides={
                "language": getattr(job.settings, 'language', 'auto'),
                "initial_prompt": context,
            },
            context="patch",
        )
        result = whisper_service.transcribe(audio=audio_segment, **params)

        whisper_text = result.get('text', '').strip()
        whisper_conf = self._estimate_whisper_confidence(result)
        segments = result.get('segments', [])

        # ========== 第三道防线: 双流校验与置信度门控 ==========
        from app.services.text_normalizer import TextNormalizer

        # A. avg_logprob 检查（模型在瞎猜）
        avg_logprob = -0.5  # 默认值
        avg_no_speech = 0.0  # 默认值
        if segments:
            avg_logprob = sum(s.get('avg_logprob', -0.5) for s in segments) / len(segments)
            avg_no_speech = sum(s.get('no_speech_prob', 0.0) for s in segments) / len(segments)

            # 熔断条件1: avg_logprob 过低（模型不确信）
            if avg_logprob < -1.0:
                self.logger.warning(
                    f"Whisper 熔断(avg_logprob={avg_logprob:.2f} < -1.0) {sentence_index}: "
                    f"'{whisper_text[:50]}...', 回退到 SenseVoice"
                )
                return sentence

            # 熔断条件2: no_speech_prob 高但仍输出文本（静音段被强行翻译）
            if avg_no_speech > 0.6 and whisper_text:
                self.logger.warning(
                    f"Whisper 熔断(no_speech={avg_no_speech:.2f} > 0.6) {sentence_index}: "
                    f"'{whisper_text[:50]}...', 回退到 SenseVoice"
                )
                return sentence

        # B. 幻觉正则检测（使用升级后的 TextNormalizer）
        if TextNormalizer.is_whisper_hallucination(whisper_text):
            self.logger.warning(
                f"Whisper 熔断(幻觉检测) {sentence_index}: '{whisper_text[:50]}...', 回退到 SenseVoice"
            )
            return sentence

        # C. 双流长度/内容对比（关键创新）
        sensevoice_text = sentence.text_clean or sentence.text or ""

        # 特殊情况: SenseVoice 为空，但 Whisper 有输出
        if not sensevoice_text and whisper_text:
            # 如果 Whisper 置信度极高，给予特权放行
            if avg_logprob > -0.5:
                self.logger.info(
                    f"SenseVoice 为空，Whisper 高置信度(logprob={avg_logprob:.2f})特权放行 {sentence_index}: "
                    f"'{whisper_text[:50]}...'"
                )
                # 继续正常流程
            else:
                # Whisper 置信度不够高，保守起见不采纳
                self.logger.warning(
                    f"SenseVoice 为空，Whisper 置信度不足(logprob={avg_logprob:.2f}) {sentence_index}: "
                    f"'{whisper_text[:50]}...', 保持 SenseVoice 空结果"
                )
                return sentence

        # 长度暴涨检测: Whisper 输出远超 SenseVoice
        if sensevoice_text:
            len_sv = len(sensevoice_text)
            len_w = len(whisper_text)

            # 公式: len(whisper) > 3 * len(sensevoice) + 10
            if len_w > 3 * len_sv + 10:
                # 特权放行: 如果 Whisper 置信度极高，可能是 SenseVoice 漏识别
                if avg_logprob > -0.5:
                    self.logger.info(
                        f"Whisper 长度暴涨但置信度高(logprob={avg_logprob:.2f}) {sentence_index}, 特权放行"
                    )
                else:
                    self.logger.warning(
                        f"Whisper 熔断(长度暴涨 {len_w} > 3*{len_sv}+10) {sentence_index}: "
                        f"'{whisper_text[:50]}...', 回退到 SenseVoice"
                    )
                    return sentence

        # D. 原有检测逻辑（保留兼容）
        if context and whisper_text:
            # 检测是否包含大量下划线（幻觉标志）
            underscore_ratio = whisper_text.count('_') / max(len(whisper_text), 1)
            if underscore_ratio > 0.3:  # 超过 30% 是下划线
                self.logger.warning(
                    f"Whisper 复核检测到下划线幻觉 {sentence_index}: "
                    f"'{whisper_text[:50]}...' (下划线占比 {underscore_ratio:.1%}), 回退到 SenseVoice"
                )
                return sentence

            # 检测是否重复了提示词内容（相似度过高）
            context_words = set(context.split())
            whisper_words = set(whisper_text.split())
            if len(context_words) > 0:
                overlap_ratio = len(context_words & whisper_words) / len(context_words)
                # 如果 Whisper 输出与 context 重叠度超过 80%，且长度相近，可能是照抄
                if overlap_ratio > 0.8 and abs(len(whisper_text) - len(context)) < len(context) * 0.3:
                    self.logger.warning(
                        f"Whisper 复核检测到提示词重复 {sentence_index}: "
                        f"与 context 重叠度 {overlap_ratio:.1%}, 回退到 SenseVoice"
                    )
                    return sentence

        # === 判决时刻 ===
        if is_trash_suspect:
            # 仲裁阈值
            WHISPER_ARBITRATION_CONF = 0.5  # Whisper 也必须有一定确信度
            MIN_TEXT_LENGTH = 1  # 最少文本长度

            # V3.8 修复: Whisper 空输出时，保留 SenseVoice 而非删除
            # 有音频却没有 Whisper 输出，说明 Whisper 幻觉，应回退到 SenseVoice
            if len(whisper_text) < MIN_TEXT_LENGTH:
                self.logger.warning(
                    f"仲裁结果: Whisper 空输出，保留 SenseVoice 结果 {sentence_index} - "
                    f"SenseVoice({sentence.confidence:.2f}): '{sensevoice_text[:50]}...'"
                )
                # 不删除，直接返回原 SenseVoice 句子
                return sentence

            # 判死刑条件：Whisper 有输出但置信度极低（两者都不可靠）
            if whisper_conf < WHISPER_ARBITRATION_CONF:
                # V3.8: 如果 SenseVoice 有实质内容，优先保留而非删除
                if sensevoice_text and len(sensevoice_text) >= 2:
                    self.logger.warning(
                        f"仲裁结果: Whisper 低置信度({whisper_conf:.2f})，保留 SenseVoice {sentence_index}: "
                        f"'{sensevoice_text[:50]}...'"
                    )
                    return sentence
                else:
                    # SenseVoice 也为空或单字符，且 Whisper 低置信度，才删除
                    self.logger.info(
                        f"仲裁结果: 删除垃圾片段 {sentence_index} - "
                        f"SenseVoice({sentence.confidence:.2f}, '{sensevoice_text}') + Whisper({whisper_conf:.2f}, '{whisper_text}')"
                    )
                    subtitle_manager.mark_for_deletion(
                        index=sentence_index,
                        reason=f"whisper_arbitration_failed_conf_{whisper_conf:.2f}"
                    )
                    return sentence
            else:
                # 挽救成功！
                self.logger.info(
                    f"仲裁结果: 挽救片段 {sentence_index} - "
                    f"'{whisper_text}' (Whisper Conf {whisper_conf:.2f})"
                )

        # 常规复核 OR 垃圾样本通过仲裁 => 采纳 Whisper 结果
        if not whisper_text:
            self.logger.warning(f"Whisper 复核返回空文本，保留原结果")
            return sentence

        # 保存 Whisper 备选文本
        sentence.whisper_alternative = whisper_text

        # 使用伪对齐更新句子
        subtitle_manager.update_sentence(
            index=sentence_index,
            new_text=whisper_text,
            source=TextSource.WHISPER_PATCH,
            confidence=whisper_conf
        )

        return subtitle_manager.sentences[sentence_index]

    async def _whisper_text_patch(
        self,
        sentence: 'SentenceSegment',
        sentence_index: int,
        audio_array: np.ndarray,
        job: 'JobState',
        subtitle_manager: 'StreamingSubtitleManager'
    ) -> 'SentenceSegment':
        """
        Whisper 复核（时空解耦版：仅取文本）

        核心原则：
        - SenseVoice 确定的时间轴（start/end）不可变
        - 仅使用 Whisper 的文本结果
        - 新文本使用伪对齐生成字级时间戳

        Args:
            sentence: 原始句子（由 SenseVoice 生成）
            sentence_index: 句子索引
            audio_array: 完整音频数组
            job: 任务状态
            subtitle_manager: 流式字幕管理器

        Returns:
            SentenceSegment: 更新后的句子
        """
        from app.services.whisper_service import get_whisper_service
        from app.models.sensevoice_models import TextSource

        whisper_service = get_whisper_service()

        # 确保 Whisper 模型已加载（首次运行会自动下载）
        if not whisper_service.is_loaded:
            self.logger.info("加载 Whisper 模型...")
            whisper_service.load_model()  # 自动下载 CTranslate2 格式模型

        # 提取对应时间段的音频（增加前向重叠，为 Whisper 提供上下文预热）
        sr = 16000
        WHISPER_OVERLAP_SEC = 0.5  # Whisper 上下文重叠时长（秒）

        # 计算重叠后的起始位置（不小于0）
        overlap_start = max(0.0, sentence.start - WHISPER_OVERLAP_SEC)
        start_sample = int(overlap_start * sr)
        end_sample = int(sentence.end * sr)
        audio_segment = audio_array[start_sample:end_sample]

        # 记录日志（调试用）
        if overlap_start < sentence.start:
            self.logger.debug(
                f"Whisper 复核添加 {sentence.start - overlap_start:.2f}s 前向重叠: "
                f"[{overlap_start:.2f}s, {sentence.end:.2f}s]"
            )

        # 获取上下文提示（现在使用清洗后的文本，避免下划线等原始 token）
        context = subtitle_manager.get_context_window(sentence_index)

        # Whisper 转录（仅取文本，弃用时间戳）
        params = self._build_whisper_transcribe_params(
            job,
            overrides={
                "language": getattr(job.settings, 'language', 'auto'),
                "initial_prompt": context,
            },
            context="patch",
        )
        result = whisper_service.transcribe(audio=audio_segment, **params)

        whisper_text = result.get('text', '').strip()
        segments = result.get('segments', [])

        # ========== 第三道防线: 双流校验与置信度门控 ==========
        from app.services.text_normalizer import TextNormalizer

        # A. avg_logprob 检查（模型在瞎猜）
        avg_logprob = -0.5  # 默认值
        avg_no_speech = 0.0  # 默认值
        if segments:
            avg_logprob = sum(s.get('avg_logprob', -0.5) for s in segments) / len(segments)
            avg_no_speech = sum(s.get('no_speech_prob', 0.0) for s in segments) / len(segments)

            # 熔断条件1: avg_logprob 过低（模型不确信）
            if avg_logprob < -1.0:
                self.logger.warning(
                    f"Whisper 熔断(avg_logprob={avg_logprob:.2f} < -1.0) {sentence_index}: "
                    f"'{whisper_text[:50]}...', 回退到 SenseVoice"
                )
                return sentence

            # 熔断条件2: no_speech_prob 高但仍输出文本（静音段被强行翻译）
            if avg_no_speech > 0.6 and whisper_text:
                self.logger.warning(
                    f"Whisper 熔断(no_speech={avg_no_speech:.2f} > 0.6) {sentence_index}: "
                    f"'{whisper_text[:50]}...', 回退到 SenseVoice"
                )
                return sentence

        # B. 幻觉正则检测（使用升级后的 TextNormalizer）
        if TextNormalizer.is_whisper_hallucination(whisper_text):
            self.logger.warning(
                f"Whisper 熔断(幻觉检测) {sentence_index}: '{whisper_text[:50]}...', 回退到 SenseVoice"
            )
            return sentence

        # C. 双流长度对比
        sensevoice_text = sentence.text_clean or sentence.text or ""
        if sensevoice_text:
            len_sv = len(sensevoice_text)
            len_w = len(whisper_text)

            # 长度暴涨检测
            if len_w > 3 * len_sv + 10:
                if avg_logprob > -0.5:
                    self.logger.info(
                        f"Whisper 长度暴涨但置信度高(logprob={avg_logprob:.2f}) {sentence_index}, 特权放行"
                    )
                else:
                    self.logger.warning(
                        f"Whisper 熔断(长度暴涨 {len_w} > 3*{len_sv}+10) {sentence_index}: "
                        f"'{whisper_text[:50]}...', 回退到 SenseVoice"
                    )
                    return sentence

        # D. 原有检测逻辑（保留兼容）
        if context and whisper_text:
            # 检测是否包含大量下划线（幻觉标志）
            underscore_ratio = whisper_text.count('_') / max(len(whisper_text), 1)
            if underscore_ratio > 0.3:  # 超过 30% 是下划线
                self.logger.warning(
                    f"Whisper 复核检测到下划线幻觉 {sentence_index}: "
                    f"'{whisper_text[:50]}...' (下划线占比 {underscore_ratio:.1%}), 回退到 SenseVoice"
                )
                return sentence

            # 检测是否重复了提示词内容（相似度过高）
            context_words = set(context.split())
            whisper_words = set(whisper_text.split())
            if len(context_words) > 0:
                overlap_ratio = len(context_words & whisper_words) / len(context_words)
                # 如果 Whisper 输出与 context 重叠度超过 80%，且长度相近，可能是照抄
                if overlap_ratio > 0.8 and abs(len(whisper_text) - len(context)) < len(context) * 0.3:
                    self.logger.warning(
                        f"Whisper 复核检测到提示词重复 {sentence_index}: "
                        f"与 context 重叠度 {overlap_ratio:.1%}, 回退到 SenseVoice"
                    )
                    return sentence

        if not whisper_text:
            self.logger.warning(f"Whisper 复核返回空文本，保留原结果")
            return sentence

        # 保存 Whisper 备选文本
        sentence.whisper_alternative = whisper_text

        # 使用伪对齐更新句子
        subtitle_manager.update_sentence(
            index=sentence_index,
            new_text=whisper_text,
            source=TextSource.WHISPER_PATCH,
            confidence=self._estimate_whisper_confidence(result)
        )

        return subtitle_manager.sentences[sentence_index]

    def _estimate_whisper_confidence(self, result: dict) -> float:
        """估算 Whisper 结果置信度"""
        segments = result.get('segments', [])
        if not segments:
            # 无可靠片段时不返回置信度，让前端隐藏徽章
            return None

        # 基于 avg_logprob 和 no_speech_prob 计算
        total_logprob = sum(s.get('avg_logprob', -0.5) for s in segments)
        avg_logprob = total_logprob / len(segments)

        avg_no_speech = sum(s.get('no_speech_prob', 0.1) for s in segments) / len(segments)

        # 转换为 0-1 置信度
        confidence = min(1.0, max(0.0, 1.0 + avg_logprob))  # logprob 越接近 0 越好
        confidence *= (1.0 - avg_no_speech)  # no_speech 越低越好

        return round(confidence, 3)

    async def _whisper_buffer_pool_process(
        self,
        patch_queue: List[Dict],
        audio_array: np.ndarray,
        job: 'JobState',
        subtitle_manager: 'StreamingSubtitleManager',
        progress_tracker
    ):
        """
        使用 Whisper 缓冲池批量处理句子（DEEP_LISTEN 模式）

        优势:
        - 累积多个短 Chunk，拼接后一次性推理
        - 利用 Whisper 长上下文能力，从根源消除短音频幻觉
        - 长文本回填对齐到原始时间戳

        Args:
            patch_queue: 需要复核的句子队列
            audio_array: 完整音频数组
            job: 任务状态
            subtitle_manager: 流式字幕管理器
            progress_tracker: 进度追踪器
        """
        from app.services.whisper_service import get_whisper_service
        from app.services.whisper_buffer_pool import WhisperBufferService, WhisperBufferConfig
        from app.services.text_normalizer import TextNormalizer
        from app.models.sensevoice_models import TextSource
        from app.services.progress_tracker import ProcessPhase

        whisper_service = get_whisper_service()

        # 确保 Whisper 模型已加载
        if not whisper_service.is_loaded:
            self.logger.info("加载 Whisper 模型...")
            whisper_service.load_model()

        # 初始化缓冲池服务
        buffer_config = WhisperBufferConfig(
            min_duration_sec=5.0,       # 累积 5 秒后触发
            max_chunk_count=3,          # 或累积 3 个 Chunk 后触发
            silence_trigger_sec=1.0,    # 长静音也触发
            max_duration_sec=30.0,      # 最大 30 秒
        )
        buffer_service = WhisperBufferService(buffer_config)

        sr = 16000
        processed_count = 0
        total_count = len(patch_queue)

        self.logger.info(f"Whisper 缓冲池模式: 处理 {total_count} 个句子")

        # 按顺序处理句子，累积到缓冲池
        for idx, item in enumerate(patch_queue):
            sent_idx = item["index"]
            sentence = item["sentence"]

            # 提取音频片段
            start_sample = int(sentence.start * sr)
            end_sample = int(sentence.end * sr)
            audio_segment = audio_array[start_sample:end_sample]

            # 获取 SenseVoice 文本
            sv_text = getattr(sentence, 'text_clean', None) or sentence.text or ""
            sv_conf = sentence.confidence

            # 添加到缓冲池
            buffer_service.add_chunk(
                index=sent_idx,
                start=sentence.start,
                end=sentence.end,
                audio=audio_segment,
                sensevoice_text=sv_text,
                sensevoice_confidence=sv_conf
            )

            # 检测是否与下一个句子有长静音间隔
            has_long_silence = False
            if idx < len(patch_queue) - 1:
                next_sentence = patch_queue[idx + 1]["sentence"]
                gap = next_sentence.start - sentence.end
                if gap > buffer_config.silence_trigger_sec:
                    has_long_silence = True

            is_last = (idx == len(patch_queue) - 1)

            # 检查是否触发缓冲池处理
            if buffer_service.should_trigger(has_long_silence=has_long_silence, is_eof=is_last):
                # 获取上下文提示
                context = subtitle_manager.get_context_window(sent_idx)

                # 处理缓冲池
                aligned_results = buffer_service.process_buffer(
                    whisper_service=whisper_service,
                    language=getattr(job.settings, 'language', 'auto'),
                    initial_prompt=context
                )

                # 应用对齐结果
                for result in aligned_results:
                    chunk_idx = result["chunk_index"]
                    whisper_text = result.get("whisper_text", "").strip()
                    whisper_conf = result.get("whisper_confidence", 0.5)
                    sv_text_orig = result.get("sensevoice_text", "")

                    # 跳过空结果
                    if not whisper_text:
                        self.logger.debug(f"缓冲池对齐: Chunk {chunk_idx} Whisper 输出为空，保留 SenseVoice")
                        processed_count += 1
                        continue

                    # 幻觉检测
                    if TextNormalizer.is_whisper_hallucination(whisper_text):
                        self.logger.warning(
                            f"缓冲池对齐: Chunk {chunk_idx} Whisper 幻觉 '{whisper_text[:30]}...', 保留 SenseVoice"
                        )
                        processed_count += 1
                        continue

                    # 双流长度对比
                    if sv_text_orig:
                        len_sv = len(sv_text_orig)
                        len_w = len(whisper_text)
                        if len_w > 3 * len_sv + 10 and whisper_conf < 0.7:
                            self.logger.warning(
                                f"缓冲池对齐: Chunk {chunk_idx} 长度暴涨 ({len_w} > 3*{len_sv}+10), 保留 SenseVoice"
                            )
                            processed_count += 1
                            continue

                    # 更新句子
                    if chunk_idx in subtitle_manager.sentences:
                        subtitle_manager.update_sentence(
                            index=chunk_idx,
                            new_text=whisper_text,
                            source=TextSource.WHISPER_PATCH,
                            confidence=whisper_conf
                        )
                        self.logger.debug(
                            f"缓冲池对齐: Chunk {chunk_idx} 更新为 '{whisper_text[:50]}...'"
                        )

                    processed_count += 1

                # 更新进度
                progress_tracker.update_phase(ProcessPhase.WHISPER_PATCH, increment=len(aligned_results))

        # 处理缓冲池中的剩余内容
        if not buffer_service.is_empty:
            context = subtitle_manager.get_context_window(patch_queue[-1]["index"]) if patch_queue else ""
            remaining_results = buffer_service.flush_remaining(
                whisper_service=whisper_service,
                language=getattr(job.settings, 'language', 'auto'),
                initial_prompt=context
            )

            for result in remaining_results:
                chunk_idx = result["chunk_index"]
                whisper_text = result.get("whisper_text", "").strip()
                whisper_conf = result.get("whisper_confidence", 0.5)

                if whisper_text and not TextNormalizer.is_whisper_hallucination(whisper_text):
                    if chunk_idx in subtitle_manager.sentences:
                        subtitle_manager.update_sentence(
                            index=chunk_idx,
                            new_text=whisper_text,
                            source=TextSource.WHISPER_PATCH,
                            confidence=whisper_conf
                        )
                processed_count += 1

            progress_tracker.update_phase(ProcessPhase.WHISPER_PATCH, increment=len(remaining_results))

        self.logger.info(f"Whisper 缓冲池处理完成: {processed_count}/{total_count} 个句子")

    async def _post_process_enhancement(
        self,
        sentences: List['SentenceSegment'],
        audio_array: np.ndarray,
        job: 'JobState',
        subtitle_manager: 'StreamingSubtitleManager',
        solution_config: 'SolutionConfig'
    ) -> List['SentenceSegment']:
        """
        后处理增强层（所有 Chunk 转录完成后执行）

        根据用户配置执行：
        1. 低置信度句子 → Whisper 复核（仅文本 + 伪对齐）
        2. [可选] LLM 校对
        3. [可选] LLM 翻译

        注意：这不是熔断，熔断在转录阶段已经处理完成

        Args:
            sentences: 所有句子列表
            audio_array: 完整音频数组
            job: 任务状态
            subtitle_manager: 流式字幕管理器
            solution_config: 方案配置

        Returns:
            List[SentenceSegment]: 增强后的句子列表
        """
        from app.services.progress_tracker import get_progress_tracker, ProcessPhase
        from app.services.solution_matrix import EnhancementMode, ProofreadMode, TranslateMode
        from app.core.thresholds import needs_whisper_patch, is_critical_patch_needed

        progress_tracker = get_progress_tracker(job.job_id, solution_config.preset_id)

        # 检查任务是否已取消
        if job.canceled:
            self.logger.info(f"任务已取消，跳过后处理增强: {job.job_id}")
            return sentences

        # 调试日志：确认方法被调用
        self.logger.debug(f"开始后处理增强: {len(sentences)} 句, enhancement={solution_config.enhancement.value}")

        # V3.1.0: 极速模式（sensevoice_only）完全跳过 Whisper 复核
        # 极速模式的设计目标是纯 SenseVoice 输出，不加载 Whisper 模型
        if solution_config.enhancement == EnhancementMode.OFF:
            self.logger.info("极速模式: 跳过所有 Whisper 复核和仲裁")
            # 仍然执行 LLM 校对/翻译（如果配置了）
            if solution_config.proofread != ProofreadMode.OFF:
                self.logger.info("LLM 校对功能待实现")
            if solution_config.translate != TranslateMode.OFF:
                self.logger.info("LLM 翻译功能待实现")
            return sentences

        # 1. 收集需要 Whisper 复核的句子（含强制复核、常规复核、垃圾核查）
        patch_queue = []
        # 阈值配置
        GARBAGE_CONFIDENCE_THRESHOLD = 0.4  # 低于此值触发 Whisper 仲裁

        for i, sentence in enumerate(sentences):
            should_patch = False
            is_critical = False
            is_trash_suspect = False  # 是否是"垃圾嫌疑"需要 Whisper 仲裁

            # === DEEP_LISTEN 快速路径 ===
            # DEEP_LISTEN 模式下所有句子都要复核，直接入队，跳过后续冗余判断
            if solution_config.enhancement == EnhancementMode.DEEP_LISTEN:
                patch_queue.append({
                    "index": i,
                    "sentence": sentence,
                    "is_critical": False,
                    "is_trash_suspect": False
                })
                self.logger.debug(f"[DEEP_LISTEN] 句子{i} 直接加入双流队列")
                continue

            # 计算片段时长和清洗后文本长度
            duration = sentence.end - sentence.start
            text_clean = getattr(sentence, 'text_clean', None) or sentence.text
            clean_text = text_clean.strip() if text_clean else ""
            text_length = len(clean_text)

            # 【阶段四】强制关键复核条件（无论用户设置如何，必须修）
            if is_critical_patch_needed(clean_text, duration, sentence.confidence):
                should_patch = True
                is_critical = True
                self.logger.warning(
                    f"触发强制复核: '{clean_text}' "
                    f"(conf={sentence.confidence:.2f}, dur={duration:.2f}s)"
                )

            # 【新增】低置信度"二次安检"逻辑
            # 如果整句置信度极低（可能是幻觉/噪音），用 Whisper 仲裁
            elif sentence.confidence < GARBAGE_CONFIDENCE_THRESHOLD:
                should_patch = True
                is_trash_suspect = True
                self.logger.info(
                    f"触发低置信度核查: '{clean_text[:30]}...' "
                    f"(conf={sentence.confidence:.2f})"
                )

            # 【新增】字级强制复核（独立检查，不受 enhancement 配置影响）
            # 条件1: 单字符实词且置信度 < 0.9
            # 条件2: 任意实词置信度极低 (< 0.35)，几乎肯定是识别错误
            if not should_patch and sentence.words:
                SINGLE_CHAR_CONF_THRESHOLD = 0.9  # 单字符词的高置信度要求
                CRITICAL_WORD_CONF_THRESHOLD = 0.35  # 极低置信度阈值（低于此值几乎肯定是错误）
                LOW_CONF_WORD_COUNT_THRESHOLD = 3  # 多个低置信度词的数量阈值
                LOW_WORD_CONF_THRESHOLD = 0.5  # 低置信度词的阈值

                low_conf_word_count = 0  # 统计低置信度词数量

                for w in sentence.words:
                    # 去除空白和 SentencePiece 边界标记 (U+2581)
                    word_text = w.word.strip().lstrip('\u2581').lower()
                    word_conf = w.confidence

                    # 跳过空字符、标点
                    if not word_text or (len(word_text) == 1 and not word_text.isalnum()):
                        continue

                    # 条件1: 单字符实词 + 置信度 < 0.9
                    if len(word_text) == 1 and word_text.isalnum() and word_conf < SINGLE_CHAR_CONF_THRESHOLD:
                        should_patch = True
                        is_critical = True
                        self.logger.warning(
                            f"触发字级单字符强制复核: Sentence {i} 含单字符词 '{word_text}' "
                            f"(conf={word_conf:.2f})"
                        )
                        break

                    # 条件2: 任意实词置信度极低 (< 0.35)，几乎肯定是识别错误
                    if len(word_text) >= 2 and word_conf < CRITICAL_WORD_CONF_THRESHOLD:
                        should_patch = True
                        is_critical = True
                        self.logger.warning(
                            f"触发字级极低置信度强制复核: Sentence {i} 含极低置信度词 '{word_text}' "
                            f"(conf={word_conf:.2f})"
                        )
                        break

                    # 统计低置信度词数量
                    if len(word_text) >= 2 and word_conf < LOW_WORD_CONF_THRESHOLD:
                        low_conf_word_count += 1

                # 条件3: 多个低置信度词（>= 3个），整句可能有问题
                if not should_patch and low_conf_word_count >= LOW_CONF_WORD_COUNT_THRESHOLD:
                    should_patch = True
                    is_critical = True
                    self.logger.warning(
                        f"触发字级多低置信度词强制复核: Sentence {i} 含 {low_conf_word_count} 个低置信度词"
                    )

            # 常规复核条件（遵循用户设置，仅 SMART_PATCH 模式）
            # 注: DEEP_LISTEN 模式已在循环开头通过快速路径处理
            if not should_patch and solution_config.enhancement == EnhancementMode.SMART_PATCH:
                # SMART_PATCH 模式: 仅低置信度句子触发复核
                # 【阶段五】构建字级时间戳列表
                words_data = [{"word": w.word, "confidence": w.confidence} for w in sentence.words]

                # 增强版复核判断：置信度、短片段、单字符、字级触发
                if needs_whisper_patch(
                    sentence.confidence,
                    duration=duration,
                    text_length=text_length,
                    words=words_data  # 【阶段五】传入字级数据
                ):
                    should_patch = True

            if should_patch:
                patch_queue.append({
                    "index": i,
                    "sentence": sentence,
                    "is_critical": is_critical,
                    "is_trash_suspect": is_trash_suspect
                })

        # 调试日志：输出复核队列统计
        self.logger.debug(f"复核队列构建完成: {len(patch_queue)} 个句子需要复核")

        # 2. Whisper 复核阶段（含仲裁判决）
        if patch_queue:
            progress_tracker.start_phase(ProcessPhase.WHISPER_PATCH, len(patch_queue), "Whisper 复核中...")

            # === DEEP_LISTEN 模式: 使用 Whisper 缓冲池批量处理 ===
            if solution_config.enhancement == EnhancementMode.DEEP_LISTEN:
                await self._whisper_buffer_pool_process(
                    patch_queue=patch_queue,
                    audio_array=audio_array,
                    job=job,
                    subtitle_manager=subtitle_manager,
                    progress_tracker=progress_tracker
                )
            else:
                # === SMART_PATCH 模式: 逐句处理 ===
                for idx, item in enumerate(patch_queue):
                    # 检查任务是否已取消
                    if job.canceled:
                        self.logger.info(f"任务已取消，停止 Whisper 复核: {job.job_id}")
                        break

                    sent_idx = item["index"]
                    sentence = item["sentence"]
                    is_trash_suspect = item["is_trash_suspect"]

                    # 执行 Whisper 复核并获取仲裁结果
                    await self._whisper_text_patch_with_arbitration(
                        sentence=sentence,
                        sentence_index=sent_idx,
                        audio_array=audio_array,
                        job=job,
                        subtitle_manager=subtitle_manager,
                        is_trash_suspect=is_trash_suspect
                    )
                    progress_tracker.update_phase(ProcessPhase.WHISPER_PATCH, increment=1)

            progress_tracker.complete_phase(ProcessPhase.WHISPER_PATCH)

        # 3. 清理被标记为垃圾的句子
        deleted_count = subtitle_manager.remove_marked_sentences()
        if deleted_count > 0:
            self.logger.info(f"Whisper 仲裁：删除了 {deleted_count} 个确认为垃圾的片段")

        # 4. [可选] LLM 校对
        if solution_config.proofread != ProofreadMode.OFF:
            # TODO: 实现 LLM 校对
            self.logger.info("LLM 校对功能待实现")

        # 4. [可选] LLM 翻译
        if solution_config.translate != TranslateMode.OFF:
            # TODO: 实现 LLM 翻译
            self.logger.info("LLM 翻译功能待实现")

        return subtitle_manager.get_all_sentences()

# 单例处理器
_service_instance: Optional[TranscriptionService] = None


def get_transcription_service(root: str) -> TranscriptionService:
    """获取转录服务实例（单例模式）"""
    global _service_instance
    if _service_instance is None:
        _service_instance = TranscriptionService(root)
    return _service_instance
