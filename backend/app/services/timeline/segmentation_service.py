"""
Pyannote segmentation 服务（Phase 2）。

设计模式：Adapter Pattern。
原因：对业务层隐藏 pyannote 推理细节，统一输出 `SegmentationResult`，
即使底层边界证据改由 community-1 diarization 投影得到，也不影响上层断点与缓存契约。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

from app.services.timeline.diarization_service import (
    DiarizationResult,
    DiarizationSegment,
    PyannoteDiarizationConfig,
    PyannoteDiarizationService,
)


@dataclass(frozen=True)
class SegmentationFrame:
    """单帧 segmentation 输出。"""

    time: float
    score: float


@dataclass(frozen=True)
class SegmentationResult:
    """segmentation 推理结果。"""

    boundaries: list[float]
    frames: list[SegmentationFrame]


@dataclass(frozen=True)
class PyannoteSegmentationConfig:
    """基于 community-1 边界投影的 segmentation 配置。"""

    model_id: str = "pyannote-speaker-diarization-community-1"
    local_path: str = ""
    hf_token: Optional[str] = None
    boundary_threshold: float = 0.55
    min_boundary_interval_sec: float = 0.20
    prefer_device: str = "auto"
    min_segment_duration_sec: float = 0.15
    max_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    num_speakers: Optional[int] = None


class DiarizationRunnerLike(Protocol):
    """供 segmentation 适配器复用的 diarization 最小协议。"""

    def run(
        self,
        *,
        audio: np.ndarray,
        sample_rate: int,
    ) -> DiarizationResult:
        ...


def map_frame_times_to_word_boundaries(
    *,
    frame_times: Sequence[float],
    word_boundaries: Sequence[float],
    tolerance_sec: float = 0.22,
    is_enable_mapping: bool = False,
) -> List[Dict[str, Any]]:
    """
    将 pyannote 帧时间映射到词边界（阶段3契约）。

    Why:
    - 统一 `raw_time/snapped_time/delta_ms/mapping_reason/mapping_quality` 字段口径，
      供 FactBuilder 在不改主链裁决的前提下产出可观测事实。
    """
    normalized_frames = sorted({float(item) for item in frame_times or []})
    normalized_boundaries = sorted({float(item) for item in word_boundaries or []})
    mappings: List[Dict[str, Any]] = []
    if not normalized_frames:
        return mappings

    if not is_enable_mapping:
        for raw_time in normalized_frames:
            mappings.append(
                {
                    "raw_time": raw_time,
                    "snapped_time": raw_time,
                    "delta_ms": 0.0,
                    "mapping_reason": "time_mapping_disabled",
                    "mapping_quality": "disabled",
                }
            )
        return mappings

    if not normalized_boundaries:
        for raw_time in normalized_frames:
            mappings.append(
                {
                    "raw_time": raw_time,
                    "snapped_time": raw_time,
                    "delta_ms": 0.0,
                    "mapping_reason": "no_word_boundaries",
                    "mapping_quality": "deferred",
                }
            )
        return mappings

    tolerance = max(0.0, float(tolerance_sec))
    for raw_time in normalized_frames:
        snapped_time = min(normalized_boundaries, key=lambda item: abs(item - raw_time))
        delta_ms = (float(snapped_time) - float(raw_time)) * 1000.0
        if abs(float(snapped_time) - float(raw_time)) <= tolerance:
            mappings.append(
                {
                    "raw_time": float(raw_time),
                    "snapped_time": float(snapped_time),
                    "delta_ms": float(delta_ms),
                    "mapping_reason": "word_boundary_mapper",
                    "mapping_quality": "snapped",
                }
            )
        else:
            mappings.append(
                {
                    "raw_time": float(raw_time),
                    "snapped_time": float(raw_time),
                    "delta_ms": 0.0,
                    "mapping_reason": "no_anchor_within_tolerance",
                    "mapping_quality": "deferred",
                }
            )
    return mappings


class PyannoteSegmentationService:
    """使用 community-1 diarization 结果投影说话人变化边界。"""

    def __init__(
        self,
        config: Optional[PyannoteSegmentationConfig] = None,
        *,
        diarization_service: Optional[DiarizationRunnerLike] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or PyannoteSegmentationConfig()
        self.logger = logger or logging.getLogger(__name__)
        self._diarization_service = diarization_service or PyannoteDiarizationService(
            config=self._build_diarization_config(),
            logger=self.logger,
        )

    def run(
        self,
        *,
        audio: np.ndarray,
        sample_rate: int,
    ) -> SegmentationResult:
        """执行 community-1 推理并返回边界点。"""
        if audio.size == 0:
            return SegmentationResult(boundaries=[], frames=[])

        diarization_result = self._diarization_service.run(
            audio=audio,
            sample_rate=sample_rate,
        )
        audio_duration_sec = float(audio.shape[0]) / float(sample_rate)
        return self._build_result_from_diarization(
            diarization_result=diarization_result,
            audio_duration_sec=audio_duration_sec,
            boundary_threshold=self.config.boundary_threshold,
            min_boundary_interval_sec=self.config.min_boundary_interval_sec,
        )

    def _build_diarization_config(self) -> PyannoteDiarizationConfig:
        return PyannoteDiarizationConfig(
            enabled=True,
            model_id=str(self.config.model_id or "").strip(),
            local_path=str(self.config.local_path or "").strip(),
            hf_token=self.config.hf_token,
            prefer_device=str(self.config.prefer_device or "auto"),
            min_segment_duration_sec=max(0.0, float(self.config.min_segment_duration_sec)),
            max_speakers=self.config.max_speakers,
            min_speakers=self.config.min_speakers,
            num_speakers=self.config.num_speakers,
        )

    @staticmethod
    def _build_result_from_diarization(
        *,
        diarization_result: DiarizationResult,
        audio_duration_sec: float,
        boundary_threshold: float,
        min_boundary_interval_sec: float,
    ) -> SegmentationResult:
        segments = sorted(
            diarization_result.segments,
            key=lambda item: (float(item.start), float(item.end), str(item.speaker_id)),
        )
        if not segments:
            return SegmentationResult(boundaries=[], frames=[])

        clamped_duration = max(0.0, float(audio_duration_sec))
        frames = PyannoteSegmentationService._build_frames_from_segments(
            segments=segments,
            audio_duration_sec=clamped_duration,
        )
        boundaries = PyannoteSegmentationService._build_boundaries_from_segments(
            segments=segments,
            audio_duration_sec=clamped_duration,
            boundary_threshold=boundary_threshold,
            min_boundary_interval_sec=min_boundary_interval_sec,
        )
        return SegmentationResult(boundaries=boundaries, frames=frames)

    @staticmethod
    def _build_frames_from_segments(
        *,
        segments: Sequence[DiarizationSegment],
        audio_duration_sec: float,
    ) -> list[SegmentationFrame]:
        frames: list[SegmentationFrame] = []
        seen_times: set[float] = set()
        for segment in segments:
            confidence = max(0.0, min(1.0, float(segment.confidence)))
            for raw_time in (segment.start, segment.end):
                time_sec = max(0.0, min(float(raw_time), audio_duration_sec))
                rounded_time = round(time_sec, 6)
                if rounded_time in seen_times:
                    continue
                seen_times.add(rounded_time)
                frames.append(SegmentationFrame(time=time_sec, score=confidence))
        frames.sort(key=lambda item: item.time)
        return frames

    @staticmethod
    def _build_boundaries_from_segments(
        *,
        segments: Sequence[DiarizationSegment],
        audio_duration_sec: float,
        boundary_threshold: float,
        min_boundary_interval_sec: float,
    ) -> list[float]:
        threshold = max(0.0, min(1.0, float(boundary_threshold)))
        min_interval = max(0.0, float(min_boundary_interval_sec))
        boundaries: list[float] = []
        last_boundary = -1e9

        for prev_item, next_item in zip(segments, segments[1:]):
            is_speaker_changed = str(prev_item.speaker_id) != str(next_item.speaker_id)
            if not is_speaker_changed:
                continue

            confidence = (
                max(0.0, min(1.0, float(prev_item.confidence)))
                + max(0.0, min(1.0, float(next_item.confidence)))
            ) / 2.0
            if confidence < threshold:
                continue

            boundary_time = (float(prev_item.end) + float(next_item.start)) / 2.0
            boundary_time = max(0.0, min(boundary_time, audio_duration_sec))
            if boundary_time - last_boundary < min_interval:
                continue

            boundaries.append(boundary_time)
            last_boundary = boundary_time

        return boundaries
