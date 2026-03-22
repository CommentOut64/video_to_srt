"""窗口级语言分类器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from app.services.punctuation.semantic_buffer import SemanticChunk


SUPPORTED_LANGUAGES = ("zh", "ja", "en")
MIXED_DECISION_DOMAINS = ("flush", "split", "fallback")
MAIN_CHAIN_DECISION_DOMAINS = ("main_chain", "flush", "split", "fallback")


@dataclass(frozen=True)
class WindowLanguageDecision:
    """窗口语言分类结果。"""

    primary_language: str
    is_mixed_window: bool
    decision_domains: Tuple[str, ...]
    scores: Mapping[str, float]

    @property
    def can_enter_main_chain(self) -> bool:
        return not self.is_mixed_window and "main_chain" in self.decision_domains


class WindowLanguageClassifier:
    """基于 script run + 快流时长占比 + 慢流 hint 的窗口分类器。"""

    _FAST_RATIO_WEIGHT = 0.60
    _SCRIPT_RATIO_WEIGHT = 0.35
    _HINT_BONUS = 0.15
    _MIXED_MIN_PRIMARY_SCORE = 0.55
    _MIXED_MIN_MARGIN = 0.12
    _MIXED_SECONDARY_SCORE = 0.28

    def classify(
        self,
        chunks: Sequence[SemanticChunk],
        *,
        slow_language_hint: Optional[str] = None,
    ) -> WindowLanguageDecision:
        fast_ratios = self._collect_fast_duration_ratios(chunks)
        script_ratios = self._collect_script_ratios(chunks)
        hint = self._normalize_language(slow_language_hint)

        scores: Dict[str, float] = {}
        for language in SUPPORTED_LANGUAGES:
            fast_score = fast_ratios.get(language, 0.0)
            script_score = script_ratios.get(language, 0.0)
            score = fast_score * self._FAST_RATIO_WEIGHT + script_score * self._SCRIPT_RATIO_WEIGHT
            if hint == language:
                score += self._HINT_BONUS
            scores[language] = score

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_lang, top_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0

        if hint in SUPPORTED_LANGUAGES and top_score <= 0.0:
            return WindowLanguageDecision(
                primary_language=hint,
                is_mixed_window=False,
                decision_domains=MAIN_CHAIN_DECISION_DOMAINS,
                scores=scores,
            )

        if self._is_mixed(top_score=top_score, second_score=second_score):
            return WindowLanguageDecision(
                primary_language="mixed",
                is_mixed_window=True,
                decision_domains=MIXED_DECISION_DOMAINS,
                scores=scores,
            )

        return WindowLanguageDecision(
            primary_language=top_lang,
            is_mixed_window=False,
            decision_domains=MAIN_CHAIN_DECISION_DOMAINS,
            scores=scores,
        )

    def _is_mixed(self, *, top_score: float, second_score: float) -> bool:
        if top_score < self._MIXED_MIN_PRIMARY_SCORE:
            return True
        if (top_score - second_score) < self._MIXED_MIN_MARGIN and second_score >= self._MIXED_SECONDARY_SCORE:
            return True
        return False

    def _collect_fast_duration_ratios(self, chunks: Sequence[SemanticChunk]) -> Dict[str, float]:
        durations = {language: 0.0 for language in SUPPORTED_LANGUAGES}
        total = 0.0
        for chunk in chunks:
            start, end = chunk.audio_range
            duration = max(float(end) - float(start), 0.0)
            language = self._normalize_language(getattr(chunk, "language", ""))
            if language not in SUPPORTED_LANGUAGES:
                continue
            durations[language] += duration
            total += duration
        if total <= 0.0:
            return {language: 0.0 for language in SUPPORTED_LANGUAGES}
        return {language: value / total for language, value in durations.items()}

    def _collect_script_ratios(self, chunks: Sequence[SemanticChunk]) -> Dict[str, float]:
        script_weights = {language: 0.0 for language in SUPPORTED_LANGUAGES}
        total = 0.0
        for chunk in chunks:
            start, end = chunk.audio_range
            duration = max(float(end) - float(start), 0.0) or 1.0
            chunk_weights = self._detect_script_weights(getattr(chunk, "text", ""))
            for language, weight in chunk_weights.items():
                script_weights[language] += weight * duration
                total += weight * duration
        if total <= 0.0:
            return {language: 0.0 for language in SUPPORTED_LANGUAGES}
        return {language: value / total for language, value in script_weights.items()}

    def _detect_script_weights(self, text: str) -> Dict[str, float]:
        counts = {language: 0.0 for language in SUPPORTED_LANGUAGES}
        if not text:
            return counts

        for char in text:
            code = ord(char)
            if 0x3040 <= code <= 0x30FF:
                counts["ja"] += 1.0
                continue
            if 0x4E00 <= code <= 0x9FFF:
                counts["zh"] += 1.0
                continue
            if char.isascii() and (char.isalpha() or char.isdigit()):
                counts["en"] += 1.0

        total = sum(counts.values())
        if total <= 0.0:
            return counts
        return {language: value / total for language, value in counts.items()}

    @staticmethod
    def _normalize_language(language: Optional[str]) -> str:
        normalized = str(language or "").strip().lower()
        alias_map = {
            "cmn": "zh",
            "zh-cn": "zh",
            "zh-hans": "zh",
            "zh-hant": "zh",
            "jp": "ja",
            "eng": "en",
        }
        normalized = alias_map.get(normalized, normalized)
        if normalized in SUPPORTED_LANGUAGES:
            return normalized
        return ""
