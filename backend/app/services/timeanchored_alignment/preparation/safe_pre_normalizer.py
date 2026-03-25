"""Preparation 安全预规范化。"""

from __future__ import annotations

from dataclasses import dataclass
import re


_MULTI_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SafePreNormalizedText:
    source_text: str
    normalized_text: str
    language: str


class SafePreNormalizer:
    """只做轻量空白规范化，不在此阶段写入最终标点。"""

    def normalize(self, *, text: str, language: str) -> SafePreNormalizedText:
        source_text = str(text or "").strip()
        normalized = _MULTI_SPACE_RE.sub(" ", source_text).strip()
        return SafePreNormalizedText(
            source_text=source_text,
            normalized_text=normalized,
            language=str(language or "auto").strip().lower() or "auto",
        )
