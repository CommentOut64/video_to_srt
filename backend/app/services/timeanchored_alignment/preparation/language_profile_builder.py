"""Preparation 语言画像构建器。"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.timeanchored_alignment.contracts import LanguageRunPackage
from app.services.timeanchored_alignment.language_run_frontend import LanguageRunFrontend


@dataclass(frozen=True)
class LanguageProfileBuildResult:
    package: LanguageRunPackage


class LanguageProfileBuilder:
    """复用现有 language run 前端，但把调用边界收口到 Preparation 层。"""

    def __init__(self, *, frontend: LanguageRunFrontend | None = None) -> None:
        self._frontend = frontend or LanguageRunFrontend()

    def build(self, *, text: str, language_hint: str) -> LanguageProfileBuildResult:
        package = self._frontend.build_runs(
            text=str(text or ""),
            language_hint=language_hint,
        )
        return LanguageProfileBuildResult(package=package)
