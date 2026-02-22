"""
对齐 Gap 处理器（插值/合并/降级）。
V3.2.0+dev.20260202.07
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from app.models.confidence_models import AlignedWord, AlignmentStatus


class GapResolution(Enum):
    """Gap 处理策略。"""

    NONE = "none"
    INTERPOLATED = "interpolated"
    MERGED = "merged"
    DEGRADED = "degraded"


@dataclass
class GapResolutionResult:
    """Gap 处理输出。"""

    words: List[AlignedWord]
    resolution: GapResolution
    gap_ratio: float
    gap_positions: List[int]


class GapResolver:
    """GapResolver：根据 Gap 比例选择策略处理。"""
    _MIN_TIMELINE_EPSILON = 1e-3

    def __init__(
        self,
        gap_ratio_low: float = 0.1,
        gap_ratio_mid: float = 0.3,
        min_valid_neighbors: int = 1,
        min_word_duration: float = 0.1,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._gap_ratio_low = gap_ratio_low
        self._gap_ratio_mid = gap_ratio_mid
        self._min_valid_neighbors = min_valid_neighbors
        self._min_word_duration = min_word_duration
        self._logger = logger or logging.getLogger(__name__)

    def resolve_gaps(
        self,
        words: List[AlignedWord],
        vad_intervals: Optional[List[Tuple[float, float]]] = None,
        vad_range: Optional[Tuple[float, float]] = None,
    ) -> GapResolutionResult:
        if not words:
            return GapResolutionResult(words=words, resolution=GapResolution.NONE, gap_ratio=0.0, gap_positions=[])

        gap_indices = self._find_gap_indices(words)
        gap_ratio = len(gap_indices) / max(len(words), 1)
        gap_positions = self._find_gap_clusters(gap_indices)
        valid_count = len(words) - len(gap_indices)

        if gap_ratio == 0.0:
            return GapResolutionResult(words=words, resolution=GapResolution.NONE, gap_ratio=0.0, gap_positions=[])

        if valid_count < self._min_valid_neighbors:
            resolution = GapResolution.DEGRADED
            resolved = self._degrade_gaps(words, gap_indices, vad_intervals, vad_range)
        elif gap_ratio < self._gap_ratio_low:
            resolution = GapResolution.INTERPOLATED
            resolved = self._interpolate_gaps(words, gap_indices, vad_intervals, vad_range)
        elif gap_ratio < self._gap_ratio_mid:
            resolution = GapResolution.MERGED
            resolved = self._merge_gaps(words, gap_indices, vad_intervals, vad_range)
        else:
            resolution = GapResolution.DEGRADED
            resolved = self._degrade_gaps(words, gap_indices, vad_intervals, vad_range)

        return GapResolutionResult(
            words=resolved,
            resolution=resolution,
            gap_ratio=gap_ratio,
            gap_positions=gap_positions,
        )

    @staticmethod
    def _find_gap_indices(words: List[AlignedWord]) -> List[int]:
        indices: List[int] = []
        for idx, word in enumerate(words):
            if word.is_pseudo:
                indices.append(idx)
                continue
            if word.alignment_status in {AlignmentStatus.INSERTED, AlignmentStatus.PSEUDO}:
                indices.append(idx)
        return indices

    @staticmethod
    def _find_gap_clusters(gap_indices: List[int]) -> List[int]:
        if not gap_indices:
            return []
        groups = GapResolver._group_consecutive(gap_indices)
        return [group[0] for group in groups]

    def _interpolate_gaps(
        self,
        words: List[AlignedWord],
        gap_indices: List[int],
        vad_intervals: Optional[List[Tuple[float, float]]],
        vad_range: Optional[Tuple[float, float]],
    ) -> List[AlignedWord]:
        resolved = list(words)
        segments = self._group_consecutive(gap_indices)
        for segment in segments:
            prev_idx = self._find_prev_valid_index(resolved, segment[0])
            next_idx = self._find_next_valid_index(resolved, segment[-1])

            prev_end = resolved[prev_idx].end if prev_idx is not None else self._fallback_start(vad_range)
            next_start = resolved[next_idx].start if next_idx is not None else self._fallback_end(
                vad_range, prev_end, len(segment)
            )

            if next_start < prev_end:
                next_start = prev_end + self._min_word_duration * (len(segment) + 1)

            gap_step = (next_start - prev_end) / (len(segment) + 1)
            segment_prev_end = prev_end

            for i, idx in enumerate(segment, start=1):
                start = prev_end + gap_step * i
                end = min(start + max(self._min_word_duration, gap_step * 0.8), next_start)
                if end <= start:
                    end = start + self._min_word_duration
                start, end = self._apply_vad_anchor(start, end, vad_intervals)
                start, end = self._enforce_monotonic_bounds(
                    start=start,
                    end=end,
                    min_start=segment_prev_end,
                    max_end=next_start if next_idx is not None else None,
                )
                resolved[idx].start = start
                resolved[idx].end = end
                resolved[idx].is_pseudo = True
                segment_prev_end = end
        return resolved

    def _merge_gaps(
        self,
        words: List[AlignedWord],
        gap_indices: List[int],
        vad_intervals: Optional[List[Tuple[float, float]]],
        vad_range: Optional[Tuple[float, float]],
    ) -> List[AlignedWord]:
        resolved = list(words)
        for idx in gap_indices:
            prev_idx = self._find_prev_valid_index(resolved, idx)
            next_idx = self._find_next_valid_index(resolved, idx)
            prev_end = resolved[prev_idx].end if prev_idx is not None else None
            next_start = resolved[next_idx].start if next_idx is not None else None

            if prev_idx is not None:
                start = resolved[prev_idx].end
                end = start + self._min_word_duration
            elif next_idx is not None:
                end = resolved[next_idx].start
                start = end - self._min_word_duration
            else:
                start = self._fallback_start(vad_range)
                end = start + self._min_word_duration

            start, end = self._apply_vad_anchor(start, end, vad_intervals)
            start, end = self._enforce_monotonic_bounds(
                start=start,
                end=end,
                min_start=prev_end if prev_end is not None else start,
                max_end=next_start,
            )
            resolved[idx].start = start
            resolved[idx].end = end
            resolved[idx].is_pseudo = True
        return resolved

    def _degrade_gaps(
        self,
        words: List[AlignedWord],
        gap_indices: List[int],
        vad_intervals: Optional[List[Tuple[float, float]]],
        vad_range: Optional[Tuple[float, float]],
    ) -> List[AlignedWord]:
        resolved = self._merge_gaps(words, gap_indices, vad_intervals, vad_range)
        for idx in gap_indices:
            resolved[idx].alignment_status = AlignmentStatus.PSEUDO
            resolved[idx].is_pseudo = True
        return resolved

    @staticmethod
    def _group_consecutive(indices: List[int]) -> List[List[int]]:
        if not indices:
            return []
        groups: List[List[int]] = []
        current = [indices[0]]
        for idx in indices[1:]:
            if idx == current[-1] + 1:
                current.append(idx)
            else:
                groups.append(current)
                current = [idx]
        groups.append(current)
        return groups

    @staticmethod
    def _find_prev_valid_index(words: List[AlignedWord], start_index: int) -> Optional[int]:
        for idx in range(start_index - 1, -1, -1):
            if not words[idx].is_pseudo:
                return idx
        return None

    @staticmethod
    def _find_next_valid_index(words: List[AlignedWord], start_index: int) -> Optional[int]:
        for idx in range(start_index + 1, len(words)):
            if not words[idx].is_pseudo:
                return idx
        return None

    def _apply_vad_anchor(
        self,
        start: float,
        end: float,
        vad_intervals: Optional[List[Tuple[float, float]]],
    ) -> Tuple[float, float]:
        if not vad_intervals:
            return start, end

        mid = (start + end) / 2
        if self._is_in_speech(mid, vad_intervals):
            return start, end

        target_interval = self._find_nearest_interval(mid, vad_intervals)
        if not target_interval:
            return start, end

        interval_start, interval_end = target_interval
        interval_duration = max(interval_end - interval_start, 0.0)
        if interval_duration <= 0:
            return start, end

        duration = min(self._min_word_duration, interval_duration)
        anchored_start = max(interval_start, (interval_start + interval_end - duration) / 2)
        anchored_end = min(interval_end, anchored_start + duration)
        if anchored_end <= anchored_start:
            anchored_start = interval_start
            anchored_end = min(interval_end, interval_start + duration)
        return anchored_start, anchored_end

    def _enforce_monotonic_bounds(
        self,
        *,
        start: float,
        end: float,
        min_start: float,
        max_end: Optional[float],
    ) -> Tuple[float, float]:
        """
        兜底时间单调修正：避免伪词被 VAD 回锚后回退到前词之前。
        """
        safe_start = max(float(start), float(min_start))
        min_duration = max(self._MIN_TIMELINE_EPSILON, min(self._min_word_duration, 0.05))
        safe_end = max(float(end), safe_start + min_duration)

        if max_end is None:
            return safe_start, safe_end

        max_allowed_end = float(max_end) - self._MIN_TIMELINE_EPSILON
        if max_allowed_end <= safe_start:
            # Why: 邻接词时间窗过窄时，优先保证不回退，允许极短伪词占位。
            return safe_start, safe_start + self._MIN_TIMELINE_EPSILON

        safe_end = min(safe_end, max_allowed_end)
        if safe_end <= safe_start:
            safe_end = min(max_allowed_end, safe_start + self._MIN_TIMELINE_EPSILON)
        return safe_start, safe_end

    @staticmethod
    def _is_in_speech(time_point: float, intervals: List[Tuple[float, float]]) -> bool:
        for start, end in intervals:
            if start <= time_point <= end:
                return True
        return False

    @staticmethod
    def _nearest_speech_boundary(time_point: float, intervals: List[Tuple[float, float]]) -> float:
        best = intervals[0][0]
        best_dist = abs(time_point - best)
        for start, end in intervals:
            for boundary in (start, end):
                dist = abs(time_point - boundary)
                if dist < best_dist:
                    best = boundary
                    best_dist = dist
        return best

    @staticmethod
    def _find_nearest_interval(
        time_point: float,
        intervals: List[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        best_interval: Optional[Tuple[float, float]] = None
        best_dist = float("inf")
        for start, end in intervals:
            if time_point < start:
                dist = start - time_point
            elif time_point > end:
                dist = time_point - end
            else:
                dist = 0.0
            if dist < best_dist:
                best_dist = dist
                best_interval = (start, end)
        return best_interval

    @staticmethod
    def _fallback_start(vad_range: Optional[Tuple[float, float]]) -> float:
        if vad_range:
            return vad_range[0]
        return 0.0

    def _fallback_end(self, vad_range: Optional[Tuple[float, float]], base: float, gap_count: int) -> float:
        if vad_range:
            return vad_range[1]
        return base + self._min_word_duration * (gap_count + 1)
