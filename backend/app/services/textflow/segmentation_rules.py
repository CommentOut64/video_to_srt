"""统一切分规则函数。"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, Sequence

from app.models.sensevoice_models import SentenceSegment, WordTimestamp
from app.services.text_protection import is_sentence_end_punct


class BoundaryLanguageStrategy(Protocol):
    def is_continuation(self, text: str) -> bool: ...

    def is_incomplete_ending(self, word: str) -> bool: ...


def normalize_boundary_token(token: str) -> str:
    trim_chars = "。？！.!?，、,;；:：\"'”’）)]}】」』"
    return str(token or "").replace("▁", " ").strip().strip(trim_chars)


def is_ascii_word(token: str) -> bool:
    return bool(token) and all(ch.isascii() for ch in token)


def is_cjk_text(text: str) -> bool:
    for char in str(text or ""):
        if "\u4e00" <= char <= "\u9fff":
            return True
        if "\u3040" <= char <= "\u30ff":
            return True
        if "\uac00" <= char <= "\ud7af":
            return True
    return False


def next_real_word_index(words: Sequence[WordTimestamp], boundary_idx: int) -> Optional[int]:
    next_idx = int(boundary_idx) + 1
    while next_idx < len(words) and getattr(words[next_idx], "is_pseudo", False):
        next_idx += 1
    if next_idx >= len(words):
        return None
    return next_idx


def compute_pause_strength(
    words: Sequence[WordTimestamp],
    idx: int,
    *,
    soft_pause: float,
    long_pause: float,
) -> int:
    if idx >= len(words) - 1:
        return 0
    next_idx = idx + 1
    while next_idx < len(words) and getattr(words[next_idx], "is_pseudo", False):
        next_idx += 1
    if next_idx >= len(words):
        return 0
    boundary_idx = next_idx - 1
    if idx != boundary_idx:
        return 0

    current_end = float(getattr(words[idx], "end", 0.0) or 0.0)
    if getattr(words[idx], "is_pseudo", False):
        for prev in range(idx - 1, -1, -1):
            if not getattr(words[prev], "is_pseudo", False):
                current_end = float(getattr(words[prev], "end", current_end) or current_end)
                break

    gap = float(getattr(words[next_idx], "start", 0.0) or 0.0) - current_end
    if gap >= float(long_pause):
        return 2
    if gap >= float(soft_pause):
        return 1
    return 0


def is_temporal_backtrack_boundary(
    words: Sequence[WordTimestamp],
    idx: int,
    *,
    tolerance_sec: float,
) -> bool:
    if idx < 0 or idx >= len(words) - 1:
        return False

    left_end = float(getattr(words[idx], "end", 0.0) or 0.0)
    if getattr(words[idx], "is_pseudo", False):
        for prev in range(idx - 1, -1, -1):
            if not getattr(words[prev], "is_pseudo", False):
                left_end = float(getattr(words[prev], "end", left_end) or left_end)
                break

    immediate_right_start = float(getattr(words[idx + 1], "start", left_end) or left_end)
    if immediate_right_start < (left_end - tolerance_sec):
        return True

    next_idx = next_real_word_index(words, idx)
    if next_idx is None:
        return False
    right_start = float(getattr(words[next_idx], "start", left_end) or left_end)
    return right_start < (left_end - tolerance_sec)


def should_block_pause_split(
    words: Sequence[WordTimestamp],
    *,
    start_idx: int,
    boundary_idx: int,
    strategy: BoundaryLanguageStrategy,
) -> bool:
    next_idx = next_real_word_index(words, boundary_idx)
    if next_idx is None:
        return False

    next_probe = build_probe_text(words, next_idx)
    if next_probe and strategy.is_continuation(next_probe):
        return True

    current_token = normalize_boundary_token(getattr(words[boundary_idx], "word", "") or "")
    if current_token and strategy.is_incomplete_ending(current_token):
        return True

    prev_raw = str(getattr(words[boundary_idx], "word", "") or "").strip()
    next_first = normalize_boundary_token(getattr(words[next_idx], "word", "") or "")
    if (
        next_first
        and is_ascii_word(next_first)
        and next_first[:1].islower()
        and not is_sentence_end_punct(prev_raw, getattr(words[next_idx], "word", "") or "")
        and boundary_idx >= start_idx
    ):
        return True

    return False


def select_best_long_split_boundary(
    *,
    start_idx: int,
    limit_idx: int,
    get_boundary_strength: Callable[[int], int],
) -> Optional[int]:
    best_weak: Optional[int] = None
    for idx in range(limit_idx, start_idx - 1, -1):
        strength = int(get_boundary_strength(idx))
        if strength >= 3:
            return idx
        if strength >= 1 and best_weak is None:
            best_weak = idx
    return best_weak


def should_merge_short_with_previous(
    *,
    merged_exists: bool,
    candidate_token_count: int,
    candidate_duration: float,
    max_tokens: int,
    max_duration: float,
) -> bool:
    return (
        bool(merged_exists)
        and int(candidate_token_count) <= int(max_tokens)
        and float(candidate_duration) <= float(max_duration) * 1.2
    )


def should_merge_short_with_next(
    *,
    next_exists: bool,
    candidate_token_count: int,
    max_tokens: int,
) -> bool:
    return bool(next_exists) and int(candidate_token_count) <= int(max_tokens)


def should_merge_continuation_pair(
    left: SentenceSegment,
    right: SentenceSegment,
    *,
    strategy: BoundaryLanguageStrategy,
    continuation_merge_max_gap_sec: float,
    cjk_continuation_merge_max_words: int,
    cjk_continuation_merge_max_duration_sec: float,
    cjk_discourse_break_markers: Sequence[str],
) -> bool:
    if not left.words or not right.words:
        return False

    left_last = normalize_boundary_token(getattr(left.words[-1], "word", "") or "")
    right_probe = build_probe_text(right.words, 0)
    left_end = float(getattr(left.words[-1], "end", 0.0) or 0.0)
    right_start = float(getattr(right.words[0], "start", left_end) or left_end)
    gap_sec = max(0.0, right_start - left_end)
    if gap_sec > float(continuation_merge_max_gap_sec):
        return False

    right_real_word_count = sum(1 for item in right.words if not getattr(item, "is_pseudo", False))
    if right_real_word_count <= 0:
        return False
    right_duration = max(
        0.0,
        float(getattr(right, "end", right_start) or right_start)
        - float(getattr(right, "start", right_start) or right_start),
    )
    right_probe_compact = str(right_probe or "").replace(" ", "")
    is_cjk_context = is_cjk_text(left_last) or is_cjk_text(right_probe_compact)
    is_cjk_discourse_break = bool(
        is_cjk_context
        and any(right_probe_compact.startswith(marker) for marker in cjk_discourse_break_markers)
    )

    if right_probe and strategy.is_continuation(right_probe):
        if is_cjk_discourse_break:
            return False
        if is_cjk_context:
            return (
                right_real_word_count <= int(cjk_continuation_merge_max_words)
                and right_duration <= float(cjk_continuation_merge_max_duration_sec)
            )
        return True
    if left_last and strategy.is_incomplete_ending(left_last):
        if is_cjk_context:
            if is_cjk_discourse_break:
                return False
            return (
                right_real_word_count <= int(cjk_continuation_merge_max_words) + 1
                and right_duration <= float(cjk_continuation_merge_max_duration_sec) + 0.8
            )
        return True
    return False


def build_probe_text(
    words: Sequence[WordTimestamp],
    start_idx: int,
    window_size: int = 3,
) -> str:
    tokens: list[str] = []
    idx = int(start_idx)
    while idx < len(words) and len(tokens) < int(window_size):
        if getattr(words[idx], "is_pseudo", False):
            idx += 1
            continue
        token = normalize_boundary_token(getattr(words[idx], "word", "") or "")
        if token:
            tokens.append(token)
        idx += 1
    return " ".join(tokens)


__all__ = [
    "build_probe_text",
    "compute_pause_strength",
    "is_ascii_word",
    "is_cjk_text",
    "is_temporal_backtrack_boundary",
    "next_real_word_index",
    "normalize_boundary_token",
    "select_best_long_split_boundary",
    "should_block_pause_split",
    "should_merge_continuation_pair",
    "should_merge_short_with_next",
    "should_merge_short_with_previous",
]
