"""LanguageRun 前端：run 级切分与窗口语言判定。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from app.services.language_policy.compiler import build_language_policy_snapshot
from app.services.language_policy.resolver import classify_window_language_distribution
from app.services.language_policy.types import WINDOW_KIND_TRUE_MIXED
from app.services.timeanchored_alignment.contracts import LanguageRun, LanguageRunPackage


_SUPPORTED_LANGUAGES = {"zh", "ja", "en"}
_GENERIC_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u3040-\u30ff]+|[\u4e00-\u9fff]+|[^\s]")
_ALNUM_OR_CJK = re.compile(r"[0-9A-Za-z\u4e00-\u9fffぁ-んァ-ン]")
_HIRA_KATA = re.compile(r"[\u3040-\u30ff]")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_ASCII_WORD = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class _ProtectedSpan:
    start: int
    end: int


class LanguageRunFrontend:
    """构建 LanguageRun + 窗口级语言分类。"""

    _PROTECTED_PATTERNS = (
        re.compile(r"\bv\d+(?:\.\d+)+\b", re.IGNORECASE),
        re.compile(r"\b(?:[A-Za-z]\.){2,}[A-Za-z]?\.?"),
        re.compile(r"\b[A-Za-z]+(?:-[A-Za-z]+)+\b"),
        re.compile(r"\b[A-Za-z]+(?:'[A-Za-z]+)+\b"),
        re.compile(r"・"),
    )

    def build_runs(
        self,
        *,
        text: str,
        language_hint: Optional[str] = None,
    ) -> LanguageRunPackage:
        source_text = str(text or "")
        if not source_text.strip():
            return LanguageRunPackage(
                runs=tuple(),
                dominant_language="mixed",
                window_kind=WINDOW_KIND_TRUE_MIXED,
                foreign_run_ratio=1.0,
                source_text=source_text,
            )

        snapshot = build_language_policy_snapshot(
            language_hint=language_hint,
            feature_scope="timeanchored_alignment",
        )
        explicit_hint = self._normalize_input_hint(language_hint)
        hint_language = explicit_hint or self._normalize_language(snapshot.language_tag)
        protected_spans = self._collect_protected_spans(source_text)
        runs = self._build_run_items(
            source_text,
            protected_spans=protected_spans,
            hint_language=hint_language,
        )

        thresholds = dict(snapshot.thresholds or {})
        classification = classify_window_language_distribution(
            language_char_counts=self._build_language_counts(
                runs,
                boost_language=explicit_hint,
            ),
            dominant_language_min_ratio=self._safe_float(thresholds.get("dominant_language_min_ratio"), 0.55),
            foreign_run_max_ratio=self._safe_float(thresholds.get("foreign_run_max_ratio"), 0.50),
            foreign_run_char_max=self._safe_int(thresholds.get("foreign_run_char_max"), 32),
        )
        if snapshot.language_tag == "mixed":
            dominant_language = "mixed"
            window_kind = WINDOW_KIND_TRUE_MIXED
            foreign_run_ratio = 1.0
            can_enter_main_chain = False
        else:
            dominant_language = classification.dominant_language
            window_kind = classification.window_kind
            foreign_run_ratio = classification.foreign_run_ratio
            can_enter_main_chain = classification.can_enter_main_chain

        final_runs: List[LanguageRun] = []
        for item in runs:
            is_foreign_island = (
                window_kind == "dominant_with_islands"
                and can_enter_main_chain
                and item.run_language in _SUPPORTED_LANGUAGES
                and item.run_language != dominant_language
            )
            final_runs.append(
                LanguageRun(
                    run_text=item.run_text,
                    run_language=item.run_language,
                    char_start=item.char_start,
                    char_end=item.char_end,
                    is_protected=item.is_protected,
                    is_foreign_island=is_foreign_island,
                )
            )

        return LanguageRunPackage(
            runs=tuple(final_runs),
            dominant_language=dominant_language,
            window_kind=window_kind,
            foreign_run_ratio=max(0.0, min(1.0, float(foreign_run_ratio))),
            source_text=source_text,
        )

    def _build_run_items(
        self,
        text: str,
        *,
        protected_spans: Tuple[_ProtectedSpan, ...],
        hint_language: str,
    ) -> List[LanguageRun]:
        runs: List[LanguageRun] = []
        cursor = 0
        for span in protected_spans:
            if cursor < span.start:
                runs.extend(
                    self._tokenize_free_text(
                        text=text[cursor:span.start],
                        offset=cursor,
                        hint_language=hint_language,
                    )
                )
            protected_text = text[span.start:span.end]
            language = self._detect_token_language(protected_text, hint_language=hint_language)
            runs.append(
                LanguageRun(
                    run_text=protected_text,
                    run_language=language,
                    char_start=span.start,
                    char_end=span.end,
                    is_protected=True,
                )
            )
            cursor = span.end

        if cursor < len(text):
            runs.extend(
                self._tokenize_free_text(
                    text=text[cursor:],
                    offset=cursor,
                    hint_language=hint_language,
                )
            )
        return runs

    def _tokenize_free_text(
        self,
        *,
        text: str,
        offset: int,
        hint_language: str,
    ) -> Iterable[LanguageRun]:
        rows: List[LanguageRun] = []
        for match in _GENERIC_TOKEN_PATTERN.finditer(text):
            token_text = match.group(0)
            if not token_text or token_text.isspace():
                continue
            start = offset + match.start()
            end = offset + match.end()
            language = self._detect_token_language(token_text, hint_language=hint_language)
            rows.append(
                LanguageRun(
                    run_text=token_text,
                    run_language=language,
                    char_start=start,
                    char_end=end,
                    is_protected=False,
                )
            )
        return rows

    def _collect_protected_spans(self, text: str) -> Tuple[_ProtectedSpan, ...]:
        matched: List[_ProtectedSpan] = []
        occupied = [False] * len(text)

        raw_spans: List[_ProtectedSpan] = []
        for pattern in self._PROTECTED_PATTERNS:
            for item in pattern.finditer(text):
                raw_spans.append(_ProtectedSpan(start=item.start(), end=item.end()))
        raw_spans.sort(key=lambda item: (item.start, -(item.end - item.start)))

        for span in raw_spans:
            if span.start >= span.end:
                continue
            if any(occupied[idx] for idx in range(span.start, span.end)):
                continue
            for idx in range(span.start, span.end):
                occupied[idx] = True
            matched.append(span)
        matched.sort(key=lambda item: item.start)
        return tuple(matched)

    def _build_language_counts(
        self,
        runs: Iterable[LanguageRun],
        *,
        boost_language: str,
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in runs:
            if item.run_language not in _SUPPORTED_LANGUAGES:
                continue
            if not _ALNUM_OR_CJK.search(item.run_text):
                continue
            counts[item.run_language] = counts.get(item.run_language, 0) + 1
        if boost_language in _SUPPORTED_LANGUAGES and boost_language in counts:
            counts[boost_language] += 2
        return counts

    @staticmethod
    def _detect_token_language(token_text: str, *, hint_language: str) -> str:
        token = str(token_text or "")
        if not token:
            return hint_language or "en"
        if token == "・" or _HIRA_KATA.search(token):
            return "ja"
        if _ASCII_WORD.search(token):
            return "en"
        if _CJK.search(token):
            if hint_language in {"zh", "ja"}:
                return hint_language
            return "zh"
        if any(char.isdigit() for char in token):
            if hint_language in _SUPPORTED_LANGUAGES:
                return hint_language
            return "en"
        if hint_language in _SUPPORTED_LANGUAGES:
            return hint_language
        return "en"

    @staticmethod
    def _normalize_language(language_tag: str) -> str:
        normalized = str(language_tag or "").strip().lower()
        if normalized in _SUPPORTED_LANGUAGES:
            return normalized
        return ""

    @staticmethod
    def _normalize_input_hint(language_hint: Optional[str]) -> str:
        raw = str(language_hint or "").strip().lower()
        alias_map = {
            "zh-cn": "zh",
            "zh-hans": "zh",
            "zh-hant": "zh",
            "en-us": "en",
            "en-gb": "en",
            "ja-jp": "ja",
        }
        normalized = alias_map.get(raw, raw)
        if normalized in _SUPPORTED_LANGUAGES:
            return normalized
        return ""

    @staticmethod
    def _safe_float(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _safe_int(value: object, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)
