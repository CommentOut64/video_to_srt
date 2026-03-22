"""TextAligner 多证据评分器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from app.services.timeanchored_alignment.contracts import AcousticCandidate


@dataclass(frozen=True)
class TextAlignmentScoreConfig:
    """多证据评分参数。"""

    exact_match_score: float = 1.0
    normalized_match_score: float = 0.78
    protected_span_bonus: float = 0.15
    pronunciation_tie_break_bonus: float = 0.08
    top_candidate_bonus: float = 0.06
    ambiguous_top_delta_max: float = 0.08
    no_text_match_pronunciation_cap: float = 0.02
    no_text_match_top_candidate_cap: float = 0.03


@dataclass(frozen=True)
class TextAlignmentScoreBreakdown:
    """评分分项（便于调试和测试验证）。"""

    text_exact: float
    text_normalized: float
    protected_bonus: float
    pronunciation_prior: float
    top_candidate_bonus: float
    total_score: float
    is_ambiguous_position: bool


class TextAlignmentScorer:
    """组合文本与弱先验证据。"""

    def __init__(
        self,
        *,
        config: TextAlignmentScoreConfig | None = None,
        normalizer: Callable[[str], str] | None = None,
    ) -> None:
        self._config = config or TextAlignmentScoreConfig()
        self._normalizer = normalizer or default_token_normalizer

    def score(
        self,
        *,
        time_token: str,
        text_token: str,
        is_protected_span: bool = False,
        pronunciation_match: bool = False,
        top_candidates: Sequence[AcousticCandidate] = (),
    ) -> TextAlignmentScoreBreakdown:
        left = str(time_token or "")
        right = str(text_token or "")
        left_norm = self._normalizer(left)
        right_norm = self._normalizer(right)

        exact = left == right
        normalized = bool(left_norm and right_norm and left_norm == right_norm and not exact)

        exact_score = self._config.exact_match_score if exact else 0.0
        normalized_score = self._config.normalized_match_score if normalized else 0.0

        has_text_evidence = exact or normalized
        protected_bonus = (
            self._config.protected_span_bonus
            if has_text_evidence and bool(is_protected_span)
            else 0.0
        )
        top_candidate_bonus, is_ambiguous = self._score_top_candidates(
            text_token=right,
            top_candidates=top_candidates,
        )
        pronunciation_prior = (
            self._config.pronunciation_tie_break_bonus
            if bool(pronunciation_match) and (not exact) and (is_ambiguous or normalized)
            else 0.0
        )

        # pronunciation 与 top-candidates 只做 tie-break，不允许在纯文本失配时越权。
        if not has_text_evidence:
            pronunciation_prior = min(pronunciation_prior, self._config.no_text_match_pronunciation_cap)
            top_candidate_bonus = min(top_candidate_bonus, self._config.no_text_match_top_candidate_cap)

        total = exact_score + normalized_score + protected_bonus + pronunciation_prior + top_candidate_bonus
        return TextAlignmentScoreBreakdown(
            text_exact=exact_score,
            text_normalized=normalized_score,
            protected_bonus=protected_bonus,
            pronunciation_prior=pronunciation_prior,
            top_candidate_bonus=top_candidate_bonus,
            total_score=total,
            is_ambiguous_position=is_ambiguous,
        )

    def _score_top_candidates(
        self,
        *,
        text_token: str,
        top_candidates: Sequence[AcousticCandidate],
    ) -> tuple[float, bool]:
        if not top_candidates or len(top_candidates) < 2:
            return 0.0, False

        sorted_candidates = sorted(top_candidates, key=lambda item: float(item.score), reverse=True)
        first = sorted_candidates[0]
        second = sorted_candidates[1]
        delta = abs(float(first.score) - float(second.score))
        is_ambiguous = delta <= self._config.ambiguous_top_delta_max
        if not is_ambiguous:
            return 0.0, False

        target_norm = self._normalizer(text_token)
        for item in sorted_candidates:
            if self._normalizer(item.text) == target_norm and target_norm:
                return self._config.top_candidate_bonus, True
        return 0.0, True


def default_token_normalizer(token: str) -> str:
    """统一轻量归一化，兼容中英日基础比较。"""
    lowered = _katakana_to_hiragana(str(token or "").strip().lower())
    if not lowered:
        return ""
    kept: list[str] = []
    for ch in lowered:
        if ch.isalnum():
            kept.append(ch)
            continue
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x309F:
            kept.append(ch)
    return "".join(kept)


def _katakana_to_hiragana(text: str) -> str:
    chars: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            chars.append(chr(code - 0x60))
        else:
            chars.append(ch)
    return "".join(chars)


__all__ = [
    "TextAlignmentScoreBreakdown",
    "TextAlignmentScoreConfig",
    "TextAlignmentScorer",
    "default_token_normalizer",
]
