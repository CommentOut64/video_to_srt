"""Preparation 标点事实构建器。"""

from __future__ import annotations

import unicodedata

from app.services.punctuation.base import PuncPosition
from app.services.punctuation.punct_position_mapper import PunctPositionMapper
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

    def __init__(
        self,
        *,
        position_mapper: PunctPositionMapper | None = None,
    ) -> None:
        self._position_mapper = position_mapper or PunctPositionMapper()

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

    def build_from_punct_track(
        self,
        *,
        track_clean_text_ref: str,
        window_text: str,
        positions: tuple[PuncPosition, ...],
        evidence_source: str = "punct_track",
        remap_mode: str = "tolerant",
    ) -> tuple[PunctuationEvidence, ...]:
        if not window_text or not positions:
            return tuple()
        mapped_positions = self._position_mapper.remap(
            source_ref=str(track_clean_text_ref or ""),
            target_ref=str(window_text or ""),
            positions=positions,
            mode=remap_mode,
        )
        evidences: list[PunctuationEvidence] = []
        max_char_index = len(str(window_text or ""))
        for position in mapped_positions:
            try:
                char_index = int(position.char_index)
            except (TypeError, ValueError):
                continue
            if char_index < 0 or char_index > max_char_index:
                continue
            mark = str(position.punctuation or "")
            if not mark:
                continue
            evidences.append(
                PunctuationEvidence(
                    mark=mark,
                    source_char_index=char_index,
                    attach_side="after",
                    evidence_source=str(evidence_source or "punct_track"),
                )
            )
        return tuple(evidences)
