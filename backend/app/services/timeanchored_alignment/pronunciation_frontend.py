"""轻量发音前端：token -> phone span 统一输出。"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import app.services.homophone.tokenizers as homophone_tokenizers
from app.services.homophone.tokenizers import HomophoneTokenizer, TokenReading
from app.services.timeanchored_alignment.contracts import (
    LanguageRun,
    PhoneUnit,
    PronunciationPackage,
    TokenToPhoneSpan,
    TokenUnit,
)
from app.services.timeanchored_alignment.language_run_frontend import LanguageRunFrontend


_SUPPORTED_LANGUAGES = {"zh", "ja", "en"}


class PronunciationFrontend:
    """基于 HomophoneTokenizer 的轻量发音前端。"""

    def __init__(
        self,
        *,
        tokenizer: Optional[HomophoneTokenizer] = None,
        language_run_frontend: Optional[LanguageRunFrontend] = None,
        dependency_mode_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        self._tokenizer = tokenizer or HomophoneTokenizer()
        self._language_run_frontend = language_run_frontend or LanguageRunFrontend()
        self._dependency_mode_overrides = dict(dependency_mode_overrides or {})

    def build_package(
        self,
        *,
        text: str,
        language_hint: Optional[str] = None,
        language_runs: Optional[Sequence[LanguageRun]] = None,
        dominant_language: Optional[str] = None,
    ) -> PronunciationPackage:
        runs: Sequence[LanguageRun]
        resolved_language = str(dominant_language or "").strip().lower()
        if language_runs is None:
            run_package = self._language_run_frontend.build_runs(
                text=text,
                language_hint=language_hint,
            )
            runs = run_package.runs
            if not resolved_language:
                resolved_language = run_package.dominant_language
        else:
            runs = list(language_runs)
            if not resolved_language:
                resolved_language = self._infer_dominant_from_runs(runs)

        token_units: list[TokenUnit] = []
        phone_units: list[PhoneUnit] = []
        spans: list[TokenToPhoneSpan] = []

        for run in runs:
            run_language = str(run.run_language or "").strip().lower()
            if run_language not in _SUPPORTED_LANGUAGES:
                continue
            token_rows = self._tokenize_run(run)
            for reading in token_rows:
                token_units.append(
                    TokenUnit(
                        token_text=reading.token_text,
                        language=run_language,
                        char_start=run.char_start + int(reading.char_start),
                        char_end=run.char_start + int(reading.char_end),
                        is_protected=bool(run.is_protected),
                        is_foreign_island=bool(run.is_foreign_island),
                    )
                )
                token_index = len(token_units) - 1
                phones = self._split_phone_texts(reading.reading_key)
                phone_start = len(phone_units)
                for phone in phones:
                    phone_units.append(
                        PhoneUnit(
                            phone_text=phone,
                            language=run_language,
                            source="homophone_tokenizer",
                        )
                    )
                phone_end = len(phone_units) - 1
                spans.append(
                    TokenToPhoneSpan(
                        token_index=token_index,
                        phone_start=phone_start,
                        phone_end=phone_end,
                    )
                )

        dependency_mode = self._resolve_dependency_mode()
        return PronunciationPackage(
            token_units=tuple(token_units),
            phone_units=tuple(phone_units),
            token_to_phone_spans=tuple(spans),
            frontend_source="homophone_tokenizer",
            dependency_mode=dependency_mode,
            language=resolved_language or "auto",
            metadata={
                "token_count": len(token_units),
                "phone_count": len(phone_units),
            },
        )

    def _tokenize_run(self, run: LanguageRun) -> list[TokenReading]:
        rows = self._tokenizer.tokenize(run.run_text, run.run_language) or []
        if rows:
            return list(rows)
        fallback_key = str(run.run_text or "").strip().lower()
        if not fallback_key:
            return []
        return [
            TokenReading(
                token_text=run.run_text,
                reading_key=fallback_key,
                reading_key_fuzzy=fallback_key,
                reading_key_no_punct=fallback_key,
                reading_key_fuzzy_no_punct=fallback_key,
                char_start=0,
                char_end=len(run.run_text),
            )
        ]

    @staticmethod
    def _split_phone_texts(reading_key: str) -> list[str]:
        key = str(reading_key or "").strip()
        if not key:
            return [""]
        if "|" in key:
            phones = [item for item in key.split("|") if item]
            if phones:
                return phones
        if " " in key:
            phones = [item for item in key.split(" ") if item]
            if phones:
                return phones
        return [key]

    def _resolve_dependency_mode(self) -> Dict[str, str]:
        modes = {
            "zh": "pypinyin" if getattr(homophone_tokenizers, "pinyin", None) is not None else "lexicon_fallback",
            "en": "cmudict"
            if getattr(homophone_tokenizers, "cmudict", None) is not None
            else (
                "phonemizer"
                if getattr(homophone_tokenizers, "_phonemize_text", None) is not None
                else "rule_fallback"
            ),
            "ja": "sudachi"
            if getattr(homophone_tokenizers, "sudachi_dictionary", None) is not None
            else "lexicon_fallback",
        }
        for language, mode in self._dependency_mode_overrides.items():
            normalized = str(language or "").strip().lower()
            if normalized in _SUPPORTED_LANGUAGES:
                modes[normalized] = str(mode or "").strip().lower() or modes[normalized]
        return modes

    @staticmethod
    def _infer_dominant_from_runs(runs: Iterable[LanguageRun]) -> str:
        counter: dict[str, int] = {}
        for item in runs:
            language = str(item.run_language or "").strip().lower()
            if language not in _SUPPORTED_LANGUAGES:
                continue
            counter[language] = counter.get(language, 0) + 1
        if not counter:
            return "auto"
        return max(counter.items(), key=lambda row: row[1])[0]
