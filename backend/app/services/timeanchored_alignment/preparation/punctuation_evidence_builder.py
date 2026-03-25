"""Preparation 标点事实构建器。"""

from __future__ import annotations

import unicodedata

from app.services.timeanchored_alignment.preparation.contracts import (
    ProtectedUnit,
    PunctuationEvidence,
)


def _is_punctuation(char: str) -> bool:
    if not char:
        return False
    return unicodedata.category(char).startswith("P")


def _is_protected_position(index: int, protected_units: tuple[ProtectedUnit, ...]) -> bool:
    for unit in protected_units:
        if unit.start <= index < unit.end:
            return True
    return False


class PunctuationEvidenceBuilder:
    """标点只冻结为 evidence，不直接写回 lexical text。"""

    def build(
        self,
        *,
        source_text: str,
        source_char_to_text_index: tuple[int | None, ...],
        protected_units: tuple[ProtectedUnit, ...],
    ) -> tuple[PunctuationEvidence, ...]:
        evidences: list[PunctuationEvidence] = []
        for index, char in enumerate(str(source_text or "")):
            if not _is_punctuation(char) or _is_protected_position(index, protected_units):
                continue
            mapped = source_char_to_text_index[index] if index < len(source_char_to_text_index) else None
            attach_side = "after"
            if mapped is None:
                mapped = 0
                attach_side = "before"
            evidences.append(
                PunctuationEvidence(
                    mark=char,
                    source_char_index=int(mapped),
                    attach_side=attach_side,
                )
            )
        return tuple(evidences)
