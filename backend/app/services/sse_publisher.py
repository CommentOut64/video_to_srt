"""
SSE 发布服务 - V3.2.0+dev.20260125.11

统一管理 SSE 事件推送，包装 StreamingSubtitleManager 和 ProgressEventEmitter。

核心职责:
1. 进度事件推送（phase/overall）
2. 字幕事件推送（委托 StreamingSubtitleManager）
3. 信号事件推送（bgm_detected/separation_strategy 等）
4. Checkpoint 快照支持

设计原则:
- 包装而非替换：保留 StreamingSubtitleManager 和 ProgressEventEmitter 的完整能力
- 延迟初始化：按需创建内部管理器实例
- 统一入口：所有 SSE 推送通过此服务

使用方式:
    publisher = SSEPublisher(job_id, sse_manager)
    publisher.publish_progress("sensevoice", 50.0, "处理中...")
    publisher.add_sentence(sentence)
    publisher.publish_signal("job_complete", "转录完成")
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from app.services.streaming_subtitle import (
    StreamingSubtitleManager,
    get_streaming_subtitle_manager,
    push_subtitle_event,
)

if TYPE_CHECKING:
    from app.models.sensevoice_models import SentenceSegment, TextSource
    from app.services.sse_service import SSEManager
    from app.services.progress_emitter import ProgressEventEmitter, ProgressMode
    from app.models.job_models import JobState

logger = logging.getLogger(__name__)


class SSEPublisher:
    """
    SSE 发布服务 - 统一管理事件推送

    设计模式: Facade + Wrapper
    - Facade: 提供统一的 SSE 推送接口
    - Wrapper: 包装 StreamingSubtitleManager 和 ProgressEventEmitter
    """

    def __init__(
        self,
        job_id: str,
        sse_manager: "SSEManager",
        job: Optional["JobState"] = None,
        progress_mode: Optional["ProgressMode"] = None,
    ) -> None:
        """
        初始化 SSE 发布器

        Args:
            job_id: 任务 ID
            sse_manager: SSE 管理器
            job: 任务状态对象（用于 ProgressEventEmitter）
            progress_mode: 进度模式（用于 ProgressEventEmitter）
        """
        self.job_id = job_id
        self.sse_manager = sse_manager
        self._job = job
        self._progress_mode = progress_mode

        # 延迟初始化
        self._subtitle_manager: Optional[StreamingSubtitleManager] = None
        self._progress_emitter: Optional["ProgressEventEmitter"] = None

        logger.debug(f"[SSEPublisher] 初始化: job_id={job_id}")

    def _resolve_channel_identifier(self, job: Optional["JobState"] = None) -> str:
        """统一解析 project 频道标识。"""
        target_job = job or self._job
        if target_job is not None:
            project_id = str(getattr(target_job, "project_id", "") or "").strip()
            if project_id:
                return project_id
            runtime_job_id = str(getattr(target_job, "job_id", "") or "").strip()
            if runtime_job_id:
                return runtime_job_id
        return str(self.job_id)

    # ========== 进度事件 ==========

    def publish_progress(
        self,
        phase: str,
        percent: float,
        message: str = ""
    ) -> None:
        """
        推送阶段进度事件

        Args:
            phase: 阶段名称（sensevoice/whisper/align 等）
            percent: 进度百分比（0-100）
            message: 进度消息
        """
        data = {
            "percent": percent,
            "message": message,
            "phase": phase,
            "updated_at": int(time.time() * 1000),
        }

        channel_id = f"project:{self._resolve_channel_identifier()}"
        event_type = f"progress.{phase}"

        self.sse_manager.broadcast_sync(channel_id, event_type, data)
        logger.debug(f"[SSEPublisher] 推送进度: {phase}={percent:.1f}%")

    def publish_overall_progress(self, data: Dict[str, Any]) -> None:
        """
        推送总体进度事件

        Args:
            data: 进度数据，应包含 percent, phase, status 等字段
        """
        payload = dict(data)
        payload.setdefault("job_id", self.job_id)

        updated_at_ms = payload.get("updated_at") or int(time.time() * 1000)
        payload["updated_at"] = updated_at_ms
        payload.setdefault("timestamp", time.time())
        payload.setdefault("project_id", self._resolve_channel_identifier())

        channel_id = f"project:{self._resolve_channel_identifier()}"

        # 推送到任务频道
        self.sse_manager.broadcast_sync(channel_id, "progress.overall", payload)

        # 同时推送到全局频道（用于任务列表）
        global_payload = {
            "id": payload.get("project_id") or self._resolve_channel_identifier(),
            "percent": payload.get("percent", 0),
            "message": payload.get("message", ""),
            "status": payload.get("status", ""),
            "updated_at": updated_at_ms,
            "timestamp": payload.get("timestamp"),
        }
        self.sse_manager.broadcast_sync("global", "job_progress", global_payload)

        logger.debug(f"[SSEPublisher] 推送总体进度: {payload.get('percent', 0):.1f}%")

    def publish_job_progress(self, job: "JobState") -> None:
        """
        推送与旧实现兼容的总体进度事件（包含全量字段）

        Args:
            job: 任务状态对象
        """
        updated_at_ms = int(time.time() * 1000)
        channel_identifier = self._resolve_channel_identifier(job)
        channel_id = f"project:{channel_identifier}"

        progress_data = {
            "job_id": job.job_id,
            "project_id": channel_identifier,
            "phase": job.phase,
            "percent": job.progress,
            "phase_percent": job.phase_percent,
            "message": job.message,
            "status": job.status,
            "processed": job.processed,
            "total": job.total,
            "language": job.language or "",
            "updated_at": updated_at_ms,
            "timestamp": time.time(),
        }
        self.sse_manager.broadcast_sync(channel_id, "progress.overall", progress_data)

        global_progress_data = {
            "id": channel_identifier,
            "percent": job.progress,
            "phase_percent": job.phase_percent,
            "message": job.message,
            "status": job.status,
            "phase": job.phase,
            "processed": job.processed,
            "total": job.total,
            "updated_at": updated_at_ms,
            "timestamp": time.time(),
        }
        self.sse_manager.broadcast_sync("global", "job_progress", global_progress_data)

        logger.debug(f"[SSEPublisher] 推送总体进度(兼容): {job.progress:.1f}%")

    def publish_align_progress(
        self,
        job: "JobState",
        current_batch: int,
        total_batches: int,
        aligned_count: int,
        total_count: int,
    ) -> None:
        """
        推送对齐进度事件（旧实现兼容）

        Args:
            job: 任务状态对象
            current_batch: 当前批次号
            total_batches: 总批次数
            aligned_count: 已对齐段落数
            total_count: 总段落数
        """
        channel_identifier = self._resolve_channel_identifier(job)
        channel_id = f"project:{channel_identifier}"
        batch_progress = (current_batch / total_batches) * 100 if total_batches > 0 else 0
        segment_progress = (aligned_count / total_count) * 100 if total_count > 0 else 0

        self.sse_manager.broadcast_sync(
            channel_id,
            "progress.align",
            {
                "job_id": job.job_id,
                "project_id": channel_identifier,
                "phase": "align",
                "batch": {
                    "current": current_batch,
                    "total": total_batches,
                    "progress": round(batch_progress, 2),
                },
                "segments": {
                    "aligned": aligned_count,
                    "total": total_count,
                    "progress": round(segment_progress, 2),
                },
                "message": (
                    f"aligning batch {current_batch}/{total_batches} "
                    f"({aligned_count}/{total_count} segments)"
                ),
            },
        )

        logger.debug(
            "[SSEPublisher] 推送对齐进度: batch=%s/%s, aligned=%s/%s",
            current_batch,
            total_batches,
            aligned_count,
            total_count,
        )

    # ========== 字幕事件（委托 StreamingSubtitleManager）==========

    def publish_subtitle_event(
        self,
        event_type: str,
        data: Dict[str, Any]
    ) -> None:
        """
        推送字幕事件

        Args:
            event_type: 事件类型（segment/draft/replace_chunk 等）
            data: 事件数据
        """
        push_subtitle_event(self.sse_manager, self.job_id, event_type, data)

    def publish_segment(
        self,
        job: "JobState",
        segment_result: Dict[str, Any],
        processed: int,
        total: int,
    ) -> None:
        """
        推送单个 segment 转录结果（旧实现兼容）

        Args:
            job: 任务状态对象
            segment_result: 单个 segment 转录结果
            processed: 已处理 segment 数量
            total: 总 segment 数量
        """
        payload = {
            "segment_index": segment_result.get("segment_index", 0),
            "segments": segment_result.get("segments", []),
            "language": segment_result.get("language", job.language),
            "progress": {
                "processed": processed,
                "total": total,
                "percentage": round(processed / max(1, total) * 100, 2),
            },
        }
        self.publish_subtitle_event("segment", payload)
        logger.debug("[SSEPublisher] 推送 segment #%s", payload.get("segment_index", 0))

    def publish_aligned(self, job: "JobState", aligned_results: List[Dict[str, Any]]) -> None:
        """
        推送对齐完成事件（旧实现兼容）

        Args:
            job: 任务状态对象
            aligned_results: 对齐结果列表
        """
        segments = []
        word_segments = []
        if aligned_results:
            first = aligned_results[0] or {}
            segments = first.get("segments", [])
            word_segments = first.get("word_segments", [])

        self.publish_subtitle_event(
            "aligned",
            {
                "segments": segments,
                "word_segments": word_segments,
                "message": "对齐完成",
            },
        )
        logger.info("[SSEPublisher] 推送对齐完成事件，共 %s 条字幕", len(segments))

    def add_sentence(self, sentence: "SentenceSegment") -> int:
        """
        添加新句子（委托 StreamingSubtitleManager）

        Args:
            sentence: 句子段落

        Returns:
            int: 句子索引
        """
        return self._get_subtitle_manager().add_sentence(sentence)

    def update_sentence(
        self,
        index: int,
        new_text: str,
        source: "TextSource",
        confidence: float = None,
        perplexity: float = None,
        confidence_source: str = None,
    ) -> None:
        """
        更新已有句子（委托 StreamingSubtitleManager）

        Args:
            index: 句子索引
            new_text: 新文本
            source: 文本来源
            confidence: 新置信度
            perplexity: LLM 困惑度
            confidence_source: 置信度来源
        """
        self._get_subtitle_manager().update_sentence(
            index=index,
            new_text=new_text,
            source=source,
            confidence=confidence,
            perplexity=perplexity,
            confidence_source=confidence_source,
        )

    def add_draft_sentences(
        self,
        chunk_index: int,
        sentences: List["SentenceSegment"]
    ) -> List[int]:
        """
        添加草稿句子（快流推送）

        Args:
            chunk_index: Chunk 索引
            sentences: 句子列表

        Returns:
            List[int]: 句子索引列表
        """
        return self._get_subtitle_manager().add_draft_sentences(chunk_index, sentences)

    def replace_chunk(
        self,
        chunk_index: int,
        sentences: List["SentenceSegment"]
    ) -> List[int]:
        """
        替换 Chunk 的所有句子（慢流定稿）

        Args:
            chunk_index: Chunk 索引
            sentences: 定稿句子列表

        Returns:
            List[int]: 新的句子索引列表
        """
        return self._get_subtitle_manager().replace_chunk(chunk_index, sentences)

    def add_finalized_sentences(
        self,
        chunk_index: int,
        sentences: List["SentenceSegment"]
    ) -> List[int]:
        """
        添加定稿句子（极速模式专用）

        Args:
            chunk_index: Chunk 索引
            sentences: 定稿句子列表

        Returns:
            List[int]: 句子索引列表
        """
        return self._get_subtitle_manager().add_finalized_sentences(chunk_index, sentences)

    def get_all_sentences(self) -> List["SentenceSegment"]:
        """获取所有句子（按时间排序）"""
        return self._get_subtitle_manager().get_all_sentences()

    # ========== 信号事件 ==========

    def publish_job_signal(
        self,
        job: "JobState",
        signal_type: str,
        message: str = "",
    ) -> None:
        """
        推送任务信号事件（旧实现兼容，仅任务频道）

        Args:
            job: 任务状态对象
            signal_type: 信号类型
            message: 信号消息
        """
        signal_data = {
            "job_id": job.job_id,
            "project_id": self._resolve_channel_identifier(job),
            "signal": signal_type,
            "message": message or job.message,
            "status": job.status,
            "percent": job.progress,
            "updated_at": int(time.time() * 1000),
        }
        channel_id = f"project:{self._resolve_channel_identifier(job)}"
        self.sse_manager.broadcast_sync(channel_id, f"signal.{signal_type}", signal_data)
        logger.debug(f"[SSEPublisher] 推送任务信号: {signal_type}")

    def publish_signal(
        self,
        signal_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        推送信号事件

        Args:
            signal_type: 信号类型（job_complete/job_failed 等）
            message: 信号消息
            data: 附加数据
        """
        signal_data = {
            "job_id": self.job_id,
            "project_id": self._resolve_channel_identifier(),
            "signal": signal_type,
            "message": message,
            "updated_at": int(time.time() * 1000),
        }
        if data:
            signal_data.update(data)

        channel_id = f"project:{self._resolve_channel_identifier()}"

        # 推送到任务频道
        self.sse_manager.broadcast_sync(channel_id, f"signal.{signal_type}", signal_data)

        # 同时推送到全局频道
        self.sse_manager.broadcast_sync("global", f"signal.{signal_type}", signal_data)

        logger.debug(f"[SSEPublisher] 推送信号: {signal_type}")

    def publish_bgm_detected(
        self,
        level: str,
        ratios: List[float],
        max_ratio: float,
        recommendation: str,
        include_timestamp: bool = False,
    ) -> None:
        """
        推送 BGM 检测结果

        Args:
            level: BGM 强度级别
            ratios: 各频段比例
            max_ratio: 最大比例
            recommendation: 处理建议
        """
        level_value = getattr(level, "value", level)
        payload = {
            "level": level_value,
            "ratios": ratios,
            "max_ratio": max_ratio,
            "recommendation": recommendation,
        }
        if include_timestamp:
            payload["updated_at"] = int(time.time() * 1000)

        channel_id = f"project:{self._resolve_channel_identifier()}"
        self.sse_manager.broadcast_sync(channel_id, "signal.bgm_detected", payload)

        logger.debug(f"[SSEPublisher] 推送 BGM 检测: level={level}")

    def publish_separation_strategy(
        self,
        strategy: Any,
        reason: str = "",
        include_timestamp: bool = False,
    ) -> None:
        """
        推送分离策略

        Args:
            strategy: 分离策略（可以是字典或有 to_dict 方法的对象）
            reason: 策略原因
        """
        if hasattr(strategy, "to_dict"):
            payload = strategy.to_dict()
        elif isinstance(strategy, dict):
            payload = dict(strategy)
        else:
            payload = {"strategy": str(strategy)}

        if reason:
            payload.setdefault("reason", reason)

        if include_timestamp:
            payload["updated_at"] = int(time.time() * 1000)

        channel_id = f"project:{self._resolve_channel_identifier()}"
        self.sse_manager.broadcast_sync(channel_id, "signal.separation_strategy", payload)

        logger.debug(f"[SSEPublisher] 推送分离策略: {payload.get('strategy', 'unknown')}")

    # ========== Checkpoint 支持 ==========

    def to_checkpoint_data(self) -> Dict[str, Any]:
        """
        生成 checkpoint 兼容的数据

        Returns:
            Dict: 包含字幕快照的 checkpoint 数据
        """
        subtitle_data = {}
        if self._subtitle_manager is not None:
            subtitle_data = self._subtitle_manager.to_checkpoint_data()

        progress_data = {}
        if self._progress_emitter is not None:
            progress_data = self._progress_emitter.to_checkpoint_data()

        return {
            "subtitle": subtitle_data,
            "progress": progress_data.get("progress", {}),
        }

    def restore_from_checkpoint(self, data: Dict[str, Any]) -> bool:
        """
        从 checkpoint 恢复状态

        Args:
            data: checkpoint 数据

        Returns:
            bool: 恢复是否成功
        """
        try:
            # 恢复字幕状态
            subtitle_data = data.get("subtitle", {})
            if subtitle_data:
                self._get_subtitle_manager().restore_from_checkpoint(subtitle_data)

            # 恢复进度状态
            progress_data = data.get("progress", {})
            if progress_data and self._progress_emitter is not None:
                self._progress_emitter.restore_from_checkpoint({"progress": progress_data})

            logger.info(f"[SSEPublisher] 从 checkpoint 恢复成功: job_id={self.job_id}")
            return True

        except Exception as e:
            logger.error(f"[SSEPublisher] 从 checkpoint 恢复失败: {e}", exc_info=True)
            return False

    # ========== 属性代理 ==========

    @property
    def subtitle_manager(self) -> StreamingSubtitleManager:
        """获取内部的 StreamingSubtitleManager"""
        return self._get_subtitle_manager()

    @property
    def progress_emitter(self) -> Optional["ProgressEventEmitter"]:
        """获取内部的 ProgressEventEmitter（可能为 None）"""
        return self._progress_emitter

    def set_progress_emitter(self, emitter: "ProgressEventEmitter") -> None:
        """
        设置外部创建的 ProgressEventEmitter

        Args:
            emitter: ProgressEventEmitter 实例
        """
        self._progress_emitter = emitter

    def set_subtitle_manager(self, manager: StreamingSubtitleManager) -> None:
        """
        设置外部创建的 StreamingSubtitleManager

        Args:
            manager: StreamingSubtitleManager 实例
        """
        self._subtitle_manager = manager

    # ========== 内部方法 ==========

    def _get_subtitle_manager(self) -> StreamingSubtitleManager:
        """延迟获取 StreamingSubtitleManager"""
        if self._subtitle_manager is None:
            self._subtitle_manager = get_streaming_subtitle_manager(
                self.job_id,
                project_id=self._resolve_channel_identifier(),
            )
            # 确保使用相同的 SSE 管理器
            self._subtitle_manager.sse_manager = self.sse_manager
        return self._subtitle_manager

    def _get_progress_emitter(self) -> "ProgressEventEmitter":
        """延迟获取 ProgressEventEmitter"""
        if self._progress_emitter is None:
            from app.services.progress_emitter import ProgressEventEmitter, ProgressMode
            from app.models.job_models import JobState

            # 如果没有提供 job，创建一个临时的
            job = self._job
            if job is None:
                logger.warning("[SSEPublisher] 未提供 job，创建占位 JobState")
                job = JobState(job_id=self.job_id, filename="", dir="")

            mode = self._progress_mode or ProgressMode.DUAL_STREAM
            self._progress_emitter = ProgressEventEmitter(job, self.sse_manager, mode)

        return self._progress_emitter


# ========== 工厂函数 ==========

_sse_publishers: Dict[str, SSEPublisher] = {}


def get_sse_publisher(
    job_id: str,
    sse_manager: "SSEManager",
    job: Optional["JobState"] = None,
    progress_mode: Optional["ProgressMode"] = None,
) -> SSEPublisher:
    """
    获取或创建 SSE 发布器

    Args:
        job_id: 任务 ID
        sse_manager: SSE 管理器
        job: 任务状态对象
        progress_mode: 进度模式

    Returns:
        SSEPublisher 实例
    """
    publisher = _sse_publishers.get(job_id)
    if publisher is None:
        publisher = SSEPublisher(
            job_id=job_id,
            sse_manager=sse_manager,
            job=job,
            progress_mode=progress_mode,
        )
        _sse_publishers[job_id] = publisher
        return publisher

    if publisher.sse_manager is not sse_manager:
        publisher.sse_manager = sse_manager
        if publisher._subtitle_manager is not None:
            publisher._subtitle_manager.sse_manager = sse_manager

    if job is not None:
        publisher._job = job
        if publisher._progress_emitter is not None:
            publisher._progress_emitter.job = job

    if progress_mode is not None:
        publisher._progress_mode = progress_mode
        if publisher._progress_emitter is not None:
            publisher._progress_emitter.set_mode(progress_mode)

    return publisher


def remove_sse_publisher(job_id: str) -> None:
    """移除 SSE 发布器"""
    if job_id in _sse_publishers:
        del _sse_publishers[job_id]
        logger.debug(f"[SSEPublisher] 移除: {job_id}")


def get_sse_publisher_cache_snapshot() -> Dict[str, Any]:
    """获取 SSE 发布器缓存快照（观测用途，不参与业务逻辑）。"""
    job_ids = list(_sse_publishers.keys())
    return {
        "count": len(job_ids),
        "job_ids": job_ids,
    }
