"""
集合层 FactBuilder。
V3.2.0+dev.20260215.13
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from app.services.alignment.types import AlignedFacts, AlignmentResult, AnnotatedWord
from app.services.timeline.segmentation_service import map_frame_times_to_word_boundaries


@dataclass
class FactBuilderConfig:
    """集合层配置。"""

    anchor_snap_tolerance_sec: float = 0.22
    is_enable_time_mapping: bool = False
    time_axis_version: str = "m2_nw_v2"


class FactBuilder:
    """
    集合层事实构建器（Builder Pattern）。

    Why:
    - 将快慢词、turn、时间映射拼装收敛到单入口，避免编排层散落组装。
    - 只生产事实对象，不参与切分与裁决，便于后续阶段复用。
    """

    def __init__(self, config: Optional[FactBuilderConfig] = None) -> None:
        self.config = config or FactBuilderConfig()

    def build(
        self,
        *,
        annotated_words: Sequence[AnnotatedWord],
        alignment_result: Optional[AlignmentResult],
        speaker_turns: Sequence[Dict[str, Any]],
        fast_draft_cuts: Sequence[float],
        pyannote_frame_times: Sequence[float],
    ) -> AlignedFacts:
        normalized_words = list(annotated_words or [])
        normalized_turns = self._normalize_speaker_turns(speaker_turns)
        normalized_cuts = self._normalize_cut_times(fast_draft_cuts)
        normalized_frames = self._normalize_cut_times(pyannote_frame_times)
        if not normalized_frames:
            # Why: 阶段3仅要求契约可观测；无 pyannote 帧时使用快流边界作为最小候选。
            normalized_frames = list(normalized_cuts)
        boundaries = self._collect_word_boundaries(normalized_words)
        time_mappings = map_frame_times_to_word_boundaries(
            frame_times=normalized_frames,
            word_boundaries=boundaries,
            tolerance_sec=float(self.config.anchor_snap_tolerance_sec),
            is_enable_mapping=bool(self.config.is_enable_time_mapping),
        )

        alignment_score = (
            float(alignment_result.alignment_score)
            if alignment_result is not None
            else 0.0
        )
        gap_ratio = (
            float(alignment_result.gap_ratio)
            if alignment_result is not None
            else 0.0
        )
        gap_positions = (
            list(alignment_result.gap_positions)
            if alignment_result is not None
            else []
        )

        return AlignedFacts(
            annotated_words=normalized_words,
            alignment_score=alignment_score,
            gap_ratio=gap_ratio,
            gap_positions=gap_positions,
            speaker_turns=normalized_turns,
            fast_draft_cuts=normalized_cuts,
            time_axis_version=str(self.config.time_axis_version),
            time_mappings=time_mappings,
        )

    @staticmethod
    def _normalize_speaker_turns(
        turns: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in turns or []:
            if not isinstance(item, dict):
                continue
            turn_id = str(item.get("turn_id", "") or "").strip()
            speaker_id = str(item.get("speaker_id", "") or "unknown").strip() or "unknown"
            start_raw = item.get("start")
            end_raw = item.get("end")
            if start_raw is None or end_raw is None:
                continue
            start = float(start_raw)
            end = float(end_raw)
            if end <= start:
                continue
            normalized.append(
                {
                    "turn_id": turn_id,
                    "speaker_id": speaker_id,
                    "start": start,
                    "end": end,
                    "source": str(item.get("source", "") or ""),
                    "boundary_confidence": (
                        float(item.get("boundary_confidence"))
                        if item.get("boundary_confidence") is not None
                        else 0.0
                    ),
                }
            )
        return normalized

    @staticmethod
    def _collect_word_boundaries(
        words: Sequence[AnnotatedWord],
    ) -> List[float]:
        boundaries: List[float] = []
        for word in words:
            if word.start is not None:
                boundaries.append(float(word.start))
            if word.end is not None:
                boundaries.append(float(word.end))
        return sorted(set(boundaries))

    @staticmethod
    def _normalize_cut_times(values: Sequence[float]) -> List[float]:
        normalized: List[float] = []
        for value in values or []:
            try:
                normalized.append(float(value))
            except (TypeError, ValueError):
                continue
        return sorted(set(normalized))


__all__ = [
    "FactBuilder",
    "FactBuilderConfig",
]
