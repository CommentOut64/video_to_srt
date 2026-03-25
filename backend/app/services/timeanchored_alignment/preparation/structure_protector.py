"""Preparation 结构保护器。"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.text_protection import extract_protected_spans
from app.services.timeanchored_alignment.contracts import ProtectedSpan
from app.services.timeanchored_alignment.preparation.contracts import ProtectedUnit


@dataclass(frozen=True)
class StructureProtectionResult:
    source_text: str
    protected_units: tuple[ProtectedUnit, ...]
    protected_spans: tuple[ProtectedSpan, ...]


class StructureProtector:
    """冻结需要保护的结构 span。"""

    def protect(self, *, text: str) -> StructureProtectionResult:
        spans = tuple(extract_protected_spans(str(text or "")))
        protected_units = tuple(
            ProtectedUnit(
                start=int(span.start),
                end=int(span.end),
                kind=str(span.kind),
                text=str(span.text),
            )
            for span in spans
        )
        return StructureProtectionResult(
            source_text=str(text or ""),
            protected_units=protected_units,
            protected_spans=spans,
        )
