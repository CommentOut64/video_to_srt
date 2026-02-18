"""
SpeakerTimelineService（Phase 2）。

设计模式：Facade Pattern。
原因：统一编排“声纹聚类 + segmentation 边界 + turn 构建 + 断点提交”，
为上层 S-Epoch 暴露单一入口并隔离底层实现细节。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

from app.models.speaker_timeline_models import SpeakerTimeline
from app.models.speaker_timeline_models import SpeakerProfile, SpeakerTurn
from app.services.timeline.cluster_manager import (
    ClusterManager,
    ClusterManagerConfig,
)
from app.services.timeline.diarization_service import (
    DiarizationSegment,
    DiarizationResult,
    PyannoteDiarizationConfig,
    PyannoteDiarizationService,
)
from app.services.timeline.segmentation_service import (
    PyannoteSegmentationConfig,
    PyannoteSegmentationService,
    SegmentationFrame,
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
    diarization: PyannoteDiarizationConfig = field(default_factory=PyannoteDiarizationConfig)
    segmentation_unit_stage: str = "timeline"
    segmentation_unit_id: str = "segmentation_epoch"
    diarization_unit_stage: str = "timeline"
    diarization_unit_id: str = "diarization_epoch"


class SpeakerTimelineService:
    """块级说话人时间线构建服务。"""

    def __init__(
        self,
        config: Optional[SpeakerTimelineServiceConfig] = None,
        *,
        cluster_manager: Optional[ClusterManager] = None,
        turn_builder: Optional[TurnBuilder] = None,
        diarization_service: Optional[PyannoteDiarizationService] = None,
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
        self.diarization_service = diarization_service or PyannoteDiarizationService(
            config=self.config.diarization,
            logger=self.logger,
        )
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

        if self.diarization_service.is_enabled:
            diarization_result = self._run_diarization_with_checkpoint(
                block_id=block_id,
                audio=audio,
                sample_rate=sample_rate,
                chunk_count=len(chunks),
            )
            if diarization_result.segments:
                return self._build_timeline_from_diarization(
                    block_id=block_id,
                    diarization_result=diarization_result,
                )
            self.logger.warning("diarization 启用但未产出说话人分段，回退聚类+segmentation")

        self.cluster_manager.reset()
        ordered_chunks = sorted(chunks, key=lambda item: (item.start, item.end))
        assignments, start_index = self._load_cluster_progress(
            block_id=block_id,
            chunks=ordered_chunks,
        )

        for index in range(start_index, len(ordered_chunks)):
            chunk = ordered_chunks[index]
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
            self._save_cluster_progress(
                block_id=block_id,
                chunks=ordered_chunks,
                assignments=assignments,
                processed_count=index + 1,
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

    def _run_diarization_with_checkpoint(
        self,
        *,
        block_id: str,
        audio: np.ndarray,
        sample_rate: int,
        chunk_count: int,
    ) -> DiarizationResult:
        stage = self.config.diarization_unit_stage
        unit_id = f"{self.config.diarization_unit_id}:{block_id}"

        if self.runtime_checkpoint_service and hasattr(self.runtime_checkpoint_service, "load_snapshot"):
            snapshot = self.runtime_checkpoint_service.load_snapshot()
            last_unit_commits = self._extract_last_unit_commits(snapshot)
            last_unit_id = last_unit_commits.get(stage)
            if last_unit_id == unit_id:
                cached_result = self._load_cached_diarization(block_id=block_id)
                if cached_result is not None:
                    self.logger.info(
                        "diarization_epoch 命中已提交单元，命中缓存并跳过重算: stage=%s unit=%s",
                        stage,
                        unit_id,
                    )
                    return cached_result
                self.logger.info(
                    "diarization_epoch 命中已提交单元但缓存缺失，执行重算: stage=%s unit=%s",
                    stage,
                    unit_id,
                )

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
        result = self.diarization_service.run(audio=audio, sample_rate=sample_rate)
        is_cache_saved = self._save_cached_diarization(block_id=block_id, result=result)

        if self.runtime_checkpoint_service and hasattr(self.runtime_checkpoint_service, "record_unit_committed"):
            self.runtime_checkpoint_service.record_unit_committed(
                stage=stage,
                unit_id=unit_id,
                payload={
                    "block_id": block_id,
                    "segment_count": len(result.segments),
                    "speaker_count": len(result.speaker_ids),
                    "cache_saved": is_cache_saved,
                },
            )

        self._raise_if_stopped()
        return result

    def _build_timeline_from_diarization(
        self,
        *,
        block_id: str,
        diarization_result: DiarizationResult,
    ) -> SpeakerTimeline:
        turns: list[SpeakerTurn] = []
        speaker_stats: dict[str, dict[str, float]] = {}

        for idx, item in enumerate(diarization_result.segments):
            turn_id = f"{block_id}-turn-{idx:04d}"
            turns.append(
                SpeakerTurn(
                    turn_id=turn_id,
                    block_id=block_id,
                    speaker_id=item.speaker_id,
                    start=float(item.start),
                    end=float(item.end),
                    boundary_confidence=float(item.confidence),
                    is_overlap=bool(item.is_overlap),
                    source="merged",
                )
            )

            stats = speaker_stats.setdefault(
                item.speaker_id,
                {"count": 0.0, "duration": 0.0},
            )
            stats["count"] += 1.0
            stats["duration"] += float(item.duration)

        speakers = [
            SpeakerProfile(
                speaker_id=speaker_id,
                embedding_centroid=[],
                sample_count=int(values["count"]),
                quality_score=max(0.0, min(1.0, values["duration"] / max(values["count"], 1.0))),
            )
            for speaker_id, values in sorted(speaker_stats.items(), key=lambda pair: pair[0])
        ]

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
                cached_result = self._load_cached_segmentation(block_id=block_id)
                if cached_result is not None:
                    self.logger.info(
                        "segmentation_epoch 命中已提交单元，命中缓存并跳过重算: stage=%s unit=%s",
                        stage,
                        unit_id,
                    )
                    return cached_result
                # Why: 已提交但缓存缺失时若直接返回空边界，会破坏后续 turn 构建与切分质量；
                # 这里回退为重算并回填缓存，保证恢复行为与首次执行一致。
                self.logger.info(
                    "segmentation_epoch 命中已提交单元但缓存缺失，执行重算: stage=%s unit=%s",
                    stage,
                    unit_id,
                )

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
        is_cache_saved = self._save_cached_segmentation(block_id=block_id, result=result)

        if self.runtime_checkpoint_service and hasattr(self.runtime_checkpoint_service, "record_unit_committed"):
            self.runtime_checkpoint_service.record_unit_committed(
                stage=stage,
                unit_id=unit_id,
                payload={
                    "block_id": block_id,
                    "boundary_count": len(result.boundaries),
                    "cache_saved": is_cache_saved,
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

    def _load_cached_diarization(self, *, block_id: str) -> DiarizationResult | None:
        cache_path = self._build_cache_file_path(cache_prefix="diarization", block_id=block_id)
        if cache_path is None:
            return None

        payload = self._read_json_file(cache_path)
        if not isinstance(payload, dict):
            return None
        return self._deserialize_diarization_result(payload)

    def _save_cached_diarization(self, *, block_id: str, result: DiarizationResult) -> bool:
        cache_path = self._build_cache_file_path(cache_prefix="diarization", block_id=block_id)
        if cache_path is None:
            return False
        payload = self._serialize_diarization_result(result=result)
        return self._write_json_file(path=cache_path, payload=payload)

    def _load_cached_segmentation(self, *, block_id: str) -> SegmentationResult | None:
        cache_path = self._build_cache_file_path(cache_prefix="segmentation", block_id=block_id)
        if cache_path is None:
            return None
        payload = self._read_json_file(cache_path)
        if not isinstance(payload, dict):
            return None
        return self._deserialize_segmentation_result(payload)

    def _save_cached_segmentation(self, *, block_id: str, result: SegmentationResult) -> bool:
        cache_path = self._build_cache_file_path(cache_prefix="segmentation", block_id=block_id)
        if cache_path is None:
            return False
        payload = self._serialize_segmentation_result(result=result)
        return self._write_json_file(path=cache_path, payload=payload)

    def _load_cluster_progress(
        self,
        *,
        block_id: str,
        chunks: list[SpeakerChunkInput],
    ) -> tuple[list[TimelineChunkAssignment], int]:
        cache_path = self._build_cache_file_path(cache_prefix="cluster_progress", block_id=block_id)
        if cache_path is None:
            return [], 0

        payload = self._read_json_file(cache_path)
        if not isinstance(payload, dict):
            return [], 0

        expected_signature = self._build_chunk_signature(chunks=chunks)
        cached_signature = payload.get("chunk_signature", [])
        if cached_signature != expected_signature:
            return [], 0

        cluster_state_payload = payload.get("cluster_state", {})
        if not isinstance(cluster_state_payload, dict):
            return [], 0
        is_cluster_restored = self.cluster_manager.load_from_dict(cluster_state_payload)
        if not is_cluster_restored:
            self.cluster_manager.reset()
            return [], 0

        raw_assignments = payload.get("assignments", [])
        if not isinstance(raw_assignments, list):
            self.cluster_manager.reset()
            return [], 0

        processed_count = max(0, int(payload.get("processed_count", 0)))
        processed_count = min(processed_count, len(raw_assignments))
        assignments: list[TimelineChunkAssignment] = []
        for raw_item in raw_assignments[:processed_count]:
            parsed = self._deserialize_timeline_assignment(raw_item)
            if parsed is None:
                self.cluster_manager.reset()
                return [], 0
            assignments.append(parsed)

        if processed_count > 0:
            self.logger.info(
                "cluster_progress 命中缓存并恢复: block=%s processed=%s/%s",
                block_id,
                processed_count,
                len(chunks),
            )

        return assignments, processed_count

    def _save_cluster_progress(
        self,
        *,
        block_id: str,
        chunks: list[SpeakerChunkInput],
        assignments: list[TimelineChunkAssignment],
        processed_count: int,
    ) -> bool:
        cache_path = self._build_cache_file_path(cache_prefix="cluster_progress", block_id=block_id)
        if cache_path is None:
            return False

        payload = {
            "version": "3.2.0+dev.20260216.03",
            "block_id": block_id,
            "chunk_signature": self._build_chunk_signature(chunks=chunks),
            "processed_count": int(processed_count),
            "cluster_state": self.cluster_manager.to_dict(),
            "assignments": [
                self._serialize_timeline_assignment(item=item)
                for item in assignments[:processed_count]
            ],
        }
        return self._write_json_file(path=cache_path, payload=payload)

    @staticmethod
    def _serialize_diarization_result(*, result: DiarizationResult) -> dict[str, Any]:
        return {
            "version": "3.2.0+dev.20260216.03",
            "segments": [
                {
                    "speaker_id": str(item.speaker_id),
                    "start": float(item.start),
                    "end": float(item.end),
                    "confidence": float(item.confidence),
                    "is_overlap": bool(item.is_overlap),
                }
                for item in result.segments
            ],
            "speaker_ids": [str(item) for item in result.speaker_ids],
        }

    @staticmethod
    def _deserialize_diarization_result(payload: dict[str, Any]) -> DiarizationResult | None:
        try:
            raw_segments = payload.get("segments", [])
            raw_speaker_ids = payload.get("speaker_ids", [])
            if not isinstance(raw_segments, list) or not isinstance(raw_speaker_ids, list):
                return None

            segments: list[DiarizationSegment] = []
            for item in raw_segments:
                if not isinstance(item, dict):
                    return None
                segments.append(
                    DiarizationSegment(
                        speaker_id=str(item.get("speaker_id") or "unknown"),
                        start=float(item.get("start", 0.0)),
                        end=float(item.get("end", 0.0)),
                        confidence=float(item.get("confidence", 0.85)),
                        is_overlap=bool(item.get("is_overlap", False)),
                    )
                )

            return DiarizationResult(
                segments=segments,
                speaker_ids=[str(item) for item in raw_speaker_ids],
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _serialize_segmentation_result(*, result: SegmentationResult) -> dict[str, Any]:
        return {
            "version": "3.2.0+dev.20260216.03",
            "boundaries": [float(item) for item in result.boundaries],
            "frames": [
                {"time": float(item.time), "score": float(item.score)}
                for item in result.frames
            ],
        }

    @staticmethod
    def _deserialize_segmentation_result(payload: dict[str, Any]) -> SegmentationResult | None:
        try:
            raw_boundaries = payload.get("boundaries", [])
            raw_frames = payload.get("frames", [])
            if not isinstance(raw_boundaries, list) or not isinstance(raw_frames, list):
                return None

            boundaries = [float(item) for item in raw_boundaries]
            frames: list[SegmentationFrame] = []
            for item in raw_frames:
                if not isinstance(item, dict):
                    return None
                frames.append(
                    SegmentationFrame(
                        time=float(item.get("time", 0.0)),
                        score=float(item.get("score", 0.0)),
                    )
                )
            return SegmentationResult(boundaries=boundaries, frames=frames)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _serialize_timeline_assignment(item: TimelineChunkAssignment) -> dict[str, Any]:
        return {
            "chunk_id": str(item.chunk_id),
            "start": float(item.start),
            "end": float(item.end),
            "speaker_id": str(item.speaker_id),
            "speaker_status": str(item.speaker_status),
            "is_overlap": bool(item.is_overlap),
        }

    @staticmethod
    def _deserialize_timeline_assignment(payload: object) -> TimelineChunkAssignment | None:
        if not isinstance(payload, dict):
            return None
        try:
            return TimelineChunkAssignment(
                chunk_id=str(payload.get("chunk_id") or ""),
                start=float(payload.get("start", 0.0)),
                end=float(payload.get("end", 0.0)),
                speaker_id=str(payload.get("speaker_id") or "unknown"),
                speaker_status=str(payload.get("speaker_status") or "unknown"),
                is_overlap=bool(payload.get("is_overlap", False)),
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _build_chunk_signature(*, chunks: list[SpeakerChunkInput]) -> list[dict[str, Any]]:
        signature: list[dict[str, Any]] = []
        for chunk in chunks:
            embedding = chunk.embedding or []
            embedding_array = np.asarray(embedding, dtype=np.float32) if embedding else np.array([], dtype=np.float32)
            signature.append(
                {
                    "chunk_id": str(chunk.chunk_id),
                    "start": round(float(chunk.start), 6),
                    "end": round(float(chunk.end), 6),
                    "quality_score": round(float(chunk.quality_score), 6),
                    "is_overlap": bool(chunk.is_overlap),
                    "embedding_dim": int(len(embedding)),
                    "embedding_checksum": round(float(np.sum(embedding_array)), 6),
                }
            )
        return signature

    def _build_cache_file_path(self, *, cache_prefix: str, block_id: str) -> Path | None:
        cache_dir = self._resolve_timeline_cache_dir()
        if cache_dir is None:
            return None
        safe_block_id = "".join(
            item if item.isalnum() or item in {"-", "_"} else "_"
            for item in str(block_id)
        )
        if not safe_block_id:
            safe_block_id = "unknown"
        return cache_dir / f"{cache_prefix}_{safe_block_id}.json"

    def _resolve_timeline_cache_dir(self) -> Path | None:
        service = self.runtime_checkpoint_service
        if service is None:
            return None

        job_dir = getattr(service, "job_dir", None)
        if job_dir is None:
            return None

        cache_dir = Path(job_dir) / "timeline_cache"
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.logger.warning("创建 timeline_cache 目录失败: %s", exc)
            return None
        return cache_dir

    def _write_json_file(self, *, path: Path, payload: dict[str, Any]) -> bool:
        try:
            temp_path = path.with_suffix(path.suffix + ".tmp")
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_path.replace(path)
            return True
        except OSError as exc:
            self.logger.warning("写入 timeline 缓存失败: path=%s error=%s", path, exc)
            return False

    def _read_json_file(self, path: Path) -> dict[str, Any] | None:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            self.logger.warning("读取 timeline 缓存失败: path=%s error=%s", path, exc)
            return None

        try:
            payload = json.loads(text)
            if not isinstance(payload, dict):
                return None
            return payload
        except json.JSONDecodeError as exc:
            self.logger.warning("解析 timeline 缓存失败: path=%s error=%s", path, exc)
            return None

    @staticmethod
    def _extract_last_unit_commits(snapshot: RuntimeCheckpointSnapshotLike | dict[str, Any]) -> dict[str, str]:
        if isinstance(snapshot, dict):
            return dict(snapshot.get("last_unit_commits", {}))
        commits = getattr(snapshot, "last_unit_commits", {})
        return dict(commits or {})
