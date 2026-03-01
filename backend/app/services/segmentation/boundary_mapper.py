"""
词边界映射器。
V3.2.0+dev.20260215.01
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from app.models.sensevoice_models import WordTimestamp


@dataclass(frozen=True)
class BoundarySelection:
    """一次边界选择结果。"""

    split_idx: int
    score: float
    is_in_gap: bool


class WordBoundaryMapper:
    """
    词边界映射器（Strategy Pattern）。

    Why:
    - 统一裁决层 cut plan 与句级修复两条链路的“时间点 -> 词边界”逻辑，避免策略漂移。
    - 事件落在词内时按“前后半词”定向，避免默认偏向词尾导致“延后一词切”。
    """

    _EPS = 1e-6
    _DISTANCE_SCALE_SEC = 0.8
    _GAP_HIT_BONUS = 1.2
    _PAUSE_BONUS_SCALE_SEC = 0.35
    _PAUSE_BONUS_WEIGHT = 0.25
    _IN_WORD_DIRECTION_WEIGHT = 0.9
    _IN_WORD_OPPOSITE_WEIGHT = 0.45
    _IN_WORD_OTHER_PENALTY = 0.8

    def select_best_boundary(
        self,
        *,
        words: Sequence[WordTimestamp],
        event_time: float,
    ) -> Optional[BoundarySelection]:
        """返回分数最高的词边界。"""
        candidates = self.rank_boundaries(words=words, event_time=event_time)
        if not candidates:
            return None
        return candidates[0]

    def rank_boundaries(
        self,
        *,
        words: Sequence[WordTimestamp],
        event_time: float,
    ) -> List[BoundarySelection]:
        """按分数由高到低返回所有可用词边界。"""
        if len(words) <= 1:
            return []

        in_word_index, in_word_ratio = self._locate_event_word(
            words=words,
            event_time=event_time,
        )
        candidates: List[BoundarySelection] = []

        for split_idx in range(len(words) - 1):
            left_end = self._to_float(getattr(words[split_idx], "end", None), 0.0)
            right_start = self._to_float(getattr(words[split_idx + 1], "start", None), left_end)

            gap_left = min(left_end, right_start)
            gap_right = max(left_end, right_start)
            is_in_gap = (gap_left - self._EPS) <= event_time <= (gap_right + self._EPS)

            distance_sec = 0.0
            if not is_in_gap:
                distance_sec = min(
                    abs(event_time - left_end),
                    abs(event_time - right_start),
                )
            distance_score = max(0.0, 1.0 - (distance_sec / self._DISTANCE_SCALE_SEC))

            pause_duration = max(0.0, right_start - left_end)
            pause_bonus = (
                min(1.0, pause_duration / self._PAUSE_BONUS_SCALE_SEC) * self._PAUSE_BONUS_WEIGHT
            )

            score = distance_score + pause_bonus
            if is_in_gap:
                score += self._GAP_HIT_BONUS

            score += self._score_in_word_direction(
                split_idx=split_idx,
                in_word_index=in_word_index,
                in_word_ratio=in_word_ratio,
            )
            candidates.append(
                BoundarySelection(
                    split_idx=split_idx,
                    score=score,
                    is_in_gap=is_in_gap,
                )
            )

        candidates.sort(
            key=lambda item: (
                -item.score,
                abs(self._resolve_boundary_center(words=words, split_idx=item.split_idx) - event_time),
                item.split_idx,
            )
        )
        return candidates

    def _score_in_word_direction(
        self,
        *,
        split_idx: int,
        in_word_index: Optional[int],
        in_word_ratio: float,
    ) -> float:
        if in_word_index is None:
            return 0.0

        before_idx = in_word_index - 1
        after_idx = in_word_index
        if split_idx == before_idx:
            if in_word_ratio <= 0.5:
                return (0.5 - in_word_ratio) * self._IN_WORD_DIRECTION_WEIGHT
            return -((in_word_ratio - 0.5) * self._IN_WORD_OPPOSITE_WEIGHT)
        if split_idx == after_idx:
            if in_word_ratio >= 0.5:
                return (in_word_ratio - 0.5) * self._IN_WORD_DIRECTION_WEIGHT
            return -((0.5 - in_word_ratio) * self._IN_WORD_OPPOSITE_WEIGHT)
        return -self._IN_WORD_OTHER_PENALTY

    def _locate_event_word(
        self,
        *,
        words: Sequence[WordTimestamp],
        event_time: float,
    ) -> Tuple[Optional[int], float]:
        for idx, word in enumerate(words):
            word_start, word_end = self._resolve_word_span(word)
            if word_end < word_start:
                continue
            if (word_start - self._EPS) <= event_time <= (word_end + self._EPS):
                duration = max(word_end - word_start, self._EPS)
                ratio = (event_time - word_start) / duration
                ratio = max(0.0, min(1.0, ratio))
                return idx, ratio
        return None, 0.0

    def _resolve_boundary_center(
        self,
        *,
        words: Sequence[WordTimestamp],
        split_idx: int,
    ) -> float:
        left_end = self._to_float(getattr(words[split_idx], "end", None), 0.0)
        right_start = self._to_float(getattr(words[split_idx + 1], "start", None), left_end)
        return (left_end + right_start) / 2.0

    def _resolve_word_span(self, word: WordTimestamp) -> Tuple[float, float]:
        start = self._to_float(getattr(word, "start", None), 0.0)
        end = self._to_float(getattr(word, "end", None), start)
        if end < start:
            end = start
        return start, end

    @staticmethod
    def _to_float(value: Optional[float], default: float) -> float:
        if value is None:
            return default
        return float(value)
