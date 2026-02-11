"""
SpeakerTimelineService（Phase 2）。

设计模式：Facade Pattern。
原因：统一编排“声纹聚类 + segmentation 边界 + turn 构建 + 断点提交”，
为上层 S-Epoch 暴露单一入口并隔离底层实现细节。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np

from app.models.speaker_timeline_models import SpeakerTimeline
from app.services.timeline.cluster_manager import (
    ClusterManager,
    ClusterManagerConfig,
)
from app.services.timeline.segmentation_service import (
    PyannoteSegmentationConfig,
    PyannoteSegmentationService,
    SegmentationResult,
)
from app.services.timeline.turn_builder import (
    TimelineChunkAssignment,
    TurnBuilder,
    TurnBuilderConfig,
)
from app.utils.cancellation_token import CancellationToken, CancelledException, PausedException


class RuntimeCheckpointSnapshotLike(Protocol):
    """运行时快照协议。"""

    last_unit_commits: dict[str, str]


class RuntimeCheckpointServiceLike(Protocol):
    """运行时断点服务最小协议。"""

    def record_unit_started(
        self,
        *,
        stage: str,
        unit_id: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        ...

    def record_unit_committed(
        self,
        *,
        stage: str,
        unit_id: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        ...

    def is_pause_requested(self) -> bool:
        ...

    def is_cancel_requested(self) -> bool:
        ...

    def load_snapshot(self) -> RuntimeCheckpointSnapshotLike | dict[str, Any]:
        ...


@dataclass(frozen=True)
class SpeakerChunkInput:
    """Timeline 构建输入片段。"""

    chunk_id: str
    start: float
    end: float
    embedding: list[float] | None
    quality_score: float = 1.0
    is_overlap: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class SpeakerTimelineServiceConfig:
    """Timeline 服务配置。"""

    cluster: ClusterManagerConfig = field(default_factory=ClusterManagerConfig)
    turn_builder: TurnBuilderConfig = field(default_factory=TurnBuilderConfig)
    segmentation: PyannoteSegmentationConfig = field(default_factory=PyannoteSegmentationConfig)
    segmentation_unit_stage: str = "timeline"
    segmentation_unit_id: str = "segmentation_epoch"


class SpeakerTimelineService:
    """块级说话人时间线构建服务。"""

    def __init__(
        self,
        config: Optional[SpeakerTimelineServiceConfig] = None,
        *,
        cluster_manager: Optional[ClusterManager] = None,
        turn_builder: Optional[TurnBuilder] = None,
        segmentation_service: Optional[PyannoteSegmentationService] = None,
        runtime_checkpoint_service: Optional[RuntimeCheckpointServiceLike] = None,
        cancellation_token: Optional[CancellationToken] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or SpeakerTimelineServiceConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.cluster_manager = cluster_manager or ClusterManager(
            config=self.config.cluster,
            logger=self.logger,
        )
        self.turn_builder = turn_builder or TurnBuilder(config=self.config.turn_builder)
        self.segmentation_service = segmentation_service or PyannoteSegmentationService(
            config=self.config.segmentation,
            logger=self.logger,
        )
        self.runtime_checkpoint_service = runtime_checkpoint_service
        self.cancellation_token = cancellation_token

    def build_timeline(
        self,
        *,
        block_id: str,
        chunks: list[SpeakerChunkInput],
        audio: np.ndarray,
        sample_rate: int,
    ) -> SpeakerTimeline:
        """生成块级 `SpeakerTimeline`。"""
        if not chunks:
            return SpeakerTimeline(block_id=block_id, turns=[], speakers=[])

        self.cluster_manager.reset()
        assignments: list[TimelineChunkAssignment] = []

        for chunk in sorted(chunks, key=lambda item: (item.start, item.end)):
            if self.cancellation_token:
                self.cancellation_token.raise_if_canceled()
                self.cancellation_token.raise_if_paused()

            embedding = chunk.embedding or []
            cluster_result = self.cluster_manager.assign_chunk(
                embedding=embedding,
                duration=chunk.duration,
                quality_score=chunk.quality_score,
                is_overlap=chunk.is_overlap,
            )
            speaker_id = self.cluster_manager.resolve_speaker_id(cluster_result.speaker_id)
            status = self.cluster_manager.get_status(speaker_id)

            assignments.append(
                TimelineChunkAssignment(
                    chunk_id=chunk.chunk_id,
                    start=chunk.start,
                    end=chunk.end,
                    speaker_id=speaker_id,
                    speaker_status=status,
                    is_overlap=chunk.is_overlap,
                )
            )

        segmentation_result = self._run_segmentation_with_checkpoint(
            block_id=block_id,
            audio=audio,
            sample_rate=sample_rate,
            chunk_count=len(chunks),
        )

        turns = self.turn_builder.build_turns(
            block_id=block_id,
            assignments=assignments,
            segmentation_boundaries=segmentation_result.boundaries,
        )
        speakers = self.cluster_manager.build_profiles()

        return SpeakerTimeline(
            block_id=block_id,
            turns=turns,
            speakers=speakers,
        )

    def _run_segmentation_with_checkpoint(
        self,
        *,
        block_id: str,
        audio: np.ndarray,
        sample_rate: int,
        chunk_count: int,
    ):
        stage = self.config.segmentation_unit_stage
        unit_id = f"{self.config.segmentation_unit_id}:{block_id}"

        if self.runtime_checkpoint_service and hasattr(self.runtime_checkpoint_service, "load_snapshot"):
            snapshot = self.runtime_checkpoint_service.load_snapshot()
            last_unit_commits = self._extract_last_unit_commits(snapshot)
            last_unit_id = last_unit_commits.get(stage)
            if last_unit_id == unit_id:
                # Why: 崩溃恢复后如果该块 S-Epoch 已提交，则不重复执行，
                # 将重算范围限制在最后未提交单元。
                self.logger.info(
                    "segmentation_epoch 命中已提交单元，跳过重算: stage=%s unit=%s",
                    stage,
                    unit_id,
                )
                return SegmentationResult(boundaries=[], frames=[])

        payload = {
            "block_id": block_id,
            "chunk_count": chunk_count,
            "sample_rate": sample_rate,
        }
        if self.runtime_checkpoint_service and hasattr(self.runtime_checkpoint_service, "record_unit_started"):
            self.runtime_checkpoint_service.record_unit_started(
                stage=stage,
                unit_id=unit_id,
                payload=payload,
            )

        self._raise_if_stopped()
        result = self.segmentation_service.run(audio=audio, sample_rate=sample_rate)

        if self.runtime_checkpoint_service and hasattr(self.runtime_checkpoint_service, "record_unit_committed"):
            self.runtime_checkpoint_service.record_unit_committed(
                stage=stage,
                unit_id=unit_id,
                payload={
                    "block_id": block_id,
                    "boundary_count": len(result.boundaries),
                },
            )

        self._raise_if_stopped()

        return result

    def _raise_if_stopped(self) -> None:
        if self.cancellation_token:
            if self.cancellation_token.is_canceled:
                raise CancelledException(job_id="timeline", message="segmentation_epoch 已取消")
            if self.cancellation_token.is_paused:
                raise PausedException(job_id="timeline", message="segmentation_epoch 已暂停")

        service = self.runtime_checkpoint_service
        if service is None:
            return

        if hasattr(service, "is_cancel_requested") and service.is_cancel_requested():
            raise CancelledException(job_id="timeline", message="segmentation_epoch 已取消")
        if hasattr(service, "is_pause_requested") and service.is_pause_requested():
            raise PausedException(job_id="timeline", message="segmentation_epoch 已暂停")

    @staticmethod
    def _extract_last_unit_commits(snapshot: RuntimeCheckpointSnapshotLike | dict[str, Any]) -> dict[str, str]:
        if isinstance(snapshot, dict):
            return dict(snapshot.get("last_unit_commits", {}))
        commits = getattr(snapshot, "last_unit_commits", {})
        return dict(commits or {})
