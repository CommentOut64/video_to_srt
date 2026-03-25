from __future__ import annotations

from app.models.sensevoice_models import SentenceSegment, WordTimestamp
from app.services.textflow.segmentation_rules import (
    compute_pause_strength,
    is_temporal_backtrack_boundary,
    select_best_long_split_boundary,
    should_block_pause_split,
    should_merge_continuation_pair,
    should_merge_short_with_next,
    should_merge_short_with_previous,
)


class _Strategy:
    def __init__(self, *, continuations: tuple[str, ...] = (), incomplete_endings: tuple[str, ...] = ()) -> None:
        self._continuations = tuple(item.lower() for item in continuations)
        self._incomplete_endings = tuple(item.lower() for item in incomplete_endings)

    def is_continuation(self, text: str) -> bool:
        first = str(text or "").strip().split(" ")[0].lower() if str(text or "").strip() else ""
        return first in self._continuations

    def is_incomplete_ending(self, word: str) -> bool:
        return str(word or "").strip().lower() in self._incomplete_endings


def _word(word: str, start: float, end: float, *, is_pseudo: bool = False) -> WordTimestamp:
    return WordTimestamp(
        word=word,
        start=start,
        end=end,
        confidence=1.0,
        is_pseudo=is_pseudo,
    )


def _segment(*words: WordTimestamp) -> SentenceSegment:
    return SentenceSegment(
        text=" ".join(item.word for item in words),
        text_clean=" ".join(item.word for item in words),
        start=words[0].start,
        end=words[-1].end,
        words=list(words),
    )


def test_compute_pause_strength_skips_pseudo_gap_fillers() -> None:
    words = [
        _word("hello", 0.0, 0.2),
        _word("um", 0.2, 0.25, is_pseudo=True),
        _word("world", 1.0, 1.2),
    ]

    assert compute_pause_strength(words, 0, soft_pause=0.35, long_pause=0.8) == 0
    assert compute_pause_strength(words, 1, soft_pause=0.35, long_pause=0.8) == 2


def test_should_block_pause_split_on_lowercase_continuation() -> None:
    words = [
        _word("we", 0.0, 0.2),
        _word("paused", 0.2, 0.4),
        _word("and", 0.8, 1.0),
        _word("continued", 1.0, 1.3),
    ]

    assert should_block_pause_split(
        words,
        start_idx=0,
        boundary_idx=1,
        strategy=_Strategy(continuations=("and",)),
    ) is True


def test_is_temporal_backtrack_boundary_detects_rightward_time_rewind() -> None:
    words = [
        _word("first", 1.0, 1.5),
        _word("second", 1.2, 1.3),
    ]

    assert is_temporal_backtrack_boundary(words, 0, tolerance_sec=0.02) is True


def test_select_best_long_split_boundary_prefers_strong_boundary_over_weak() -> None:
    strengths = {0: 0, 1: 1, 2: 3, 3: 1}

    boundary = select_best_long_split_boundary(
        start_idx=0,
        limit_idx=3,
        get_boundary_strength=lambda idx: strengths[idx],
    )

    assert boundary == 2


def test_short_merge_decision_helpers_follow_limits() -> None:
    assert should_merge_short_with_previous(
        merged_exists=True,
        candidate_token_count=6,
        candidate_duration=4.0,
        max_tokens=8,
        max_duration=4.0,
    ) is True
    assert should_merge_short_with_previous(
        merged_exists=True,
        candidate_token_count=9,
        candidate_duration=4.0,
        max_tokens=8,
        max_duration=4.0,
    ) is False
    assert should_merge_short_with_next(
        next_exists=True,
        candidate_token_count=7,
        max_tokens=8,
    ) is True
    assert should_merge_short_with_next(
        next_exists=True,
        candidate_token_count=9,
        max_tokens=8,
    ) is False


def test_should_merge_continuation_pair_handles_cjk_and_discourse_break() -> None:
    strategy = _Strategy(continuations=("因为",), incomplete_endings=("因为",))
    left = _segment(_word("这是", 0.0, 0.4), _word("因为", 0.4, 0.6))
    right = _segment(_word("证据", 0.7, 1.0))
    blocked = _segment(_word("然而", 0.7, 0.9), _word("后来", 0.9, 1.2))

    assert should_merge_continuation_pair(
        left,
        right,
        strategy=strategy,
        continuation_merge_max_gap_sec=0.55,
        cjk_continuation_merge_max_words=4,
        cjk_continuation_merge_max_duration_sec=2.2,
        cjk_discourse_break_markers=("然而", "但是"),
    ) is True
    assert should_merge_continuation_pair(
        left,
        blocked,
        strategy=strategy,
        continuation_merge_max_gap_sec=0.55,
        cjk_continuation_merge_max_words=4,
        cjk_continuation_merge_max_duration_sec=2.2,
        cjk_discourse_break_markers=("然而", "但是"),
    ) is False
