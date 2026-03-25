"""Preparation 发音提示构建器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.services.timeanchored_alignment.contracts import (
    LanguageRun,
    PronunciationPackage,
)
from app.services.timeanchored_alignment.preparation.contracts import PronunciationHint
from app.services.timeanchored_alignment.pronunciation_frontend import PronunciationFrontend


@dataclass(frozen=True)
class PronunciationHintBuildResult:
    package: PronunciationPackage
    hints: tuple[PronunciationHint, ...]
    report: dict[str, float | str]


class PronunciationHintBuilder:
    """构建发音提示，同时保留 compat pronunciation package。"""

    def __init__(self, *, frontend: PronunciationFrontend | None = None) -> None:
        self._frontend = frontend or PronunciationFrontend()

    def build(
        self,
        *,
        text: str,
        language_hint: str,
        language_runs: Sequence[LanguageRun],
        dominant_language: str,
    ) -> PronunciationHintBuildResult:
        package = self._frontend.build_package(
            text=str(text or ""),
            language_hint=language_hint,
            language_runs=tuple(language_runs),
            dominant_language=dominant_language,
        )
        hints: list[PronunciationHint] = []
        for span in package.token_to_phone_spans:
            if span.token_index >= len(package.token_units):
                continue
            token = package.token_units[span.token_index]
            phones = package.phone_units[span.phone_start : span.phone_end + 1]
            reading_key = " ".join(phone.phone_text for phone in phones).strip()
            hints.append(
                PronunciationHint(
                    token_text=token.token_text,
                    reading_key=reading_key or token.token_text,
                    language=token.language,
                    char_start=int(token.char_start),
                    char_end=int(token.char_end),
                    source=str(package.frontend_source),
                )
            )
        report = {
            "token_count": float(len(package.token_units)),
            "phone_count": float(len(package.phone_units)),
            "language": str(package.language),
        }
        return PronunciationHintBuildResult(
            package=package,
            hints=tuple(hints),
            report=report,
        )
