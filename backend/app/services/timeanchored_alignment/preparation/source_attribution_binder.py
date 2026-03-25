"""Preparation 来源归属绑定器。"""

from __future__ import annotations

import unicodedata

from app.services.timeanchored_alignment.preparation.contracts import SlowSlot
from app.services.timeanchored_alignment.slow_window.contracts import WindowSourceUnit


def _normalize_slot_text(text: str) -> str:
    chars: list[str] = []
    for char in str(text or ""):
        if unicodedata.category(char).startswith("P"):
            continue
        chars.append(char)
    return "".join(chars)


class SourceAttributionBinder:
    """把 lexical text 绑定回 source unit provenance。"""

    def bind(
        self,
        *,
        text: str,
        source_units: tuple[WindowSourceUnit, ...],
    ) -> tuple[SlowSlot, ...]:
        lexical_text = str(text or "")
        if not lexical_text:
            return tuple()

        slots: list[SlowSlot] = []
        cursor = 0
        text_length = len(lexical_text)
        normalized_units = [
            (unit, _normalize_slot_text(unit.text))
            for unit in source_units
        ]

        for index, (unit, normalized_text) in enumerate(normalized_units):
            if cursor >= text_length:
                break
            span_length = len(normalized_text)
            if span_length <= 0:
                continue
            if index == len(normalized_units) - 1:
                end = text_length
            else:
                end = min(text_length, cursor + span_length)
            if end <= cursor:
                continue
            slots.append(
                SlowSlot(
                    slot_id=f"{unit.unit_id}:{cursor}-{end}",
                    text=lexical_text[cursor:end],
                    char_start=cursor,
                    char_end=end,
                    speaker_id=unit.speaker_id,
                    turn_id=unit.turn_id,
                    source_chunk_ids=tuple(unit.source_chunk_ids),
                    source_chunk_indices=tuple(unit.source_chunk_indices),
                    source_unit_ids=(unit.unit_id,),
                )
            )
            cursor = end

        if not slots and source_units:
            unit = source_units[0]
            return (
                SlowSlot(
                    slot_id=f"{unit.unit_id}:0-{text_length}",
                    text=lexical_text,
                    char_start=0,
                    char_end=text_length,
                    speaker_id=unit.speaker_id,
                    turn_id=unit.turn_id,
                    source_chunk_ids=tuple(unit.source_chunk_ids),
                    source_chunk_indices=tuple(unit.source_chunk_indices),
                    source_unit_ids=(unit.unit_id,),
                ),
            )

        if cursor < text_length and slots:
            last = slots[-1]
            slots[-1] = SlowSlot(
                slot_id=last.slot_id,
                text=lexical_text[last.char_start:text_length],
                char_start=last.char_start,
                char_end=text_length,
                speaker_id=last.speaker_id,
                turn_id=last.turn_id,
                source_chunk_ids=last.source_chunk_ids,
                source_chunk_indices=last.source_chunk_indices,
                source_unit_ids=last.source_unit_ids,
            )

        return tuple(slots)
