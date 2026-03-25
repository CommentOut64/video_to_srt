"""Preparation 来源归属绑定器。"""

from __future__ import annotations

from app.services.timeanchored_alignment.preparation.contracts import SlowSlot
from app.services.timeanchored_alignment.slow_window.contracts import WindowSourceUnit


def _is_key_char(char: str) -> bool:
    return char.isalnum() or ("\u4e00" <= char <= "\u9fff")


def _normalize_slot_text(text: str) -> str:
    return "".join(char.lower() for char in str(text or "") if _is_key_char(char))


def _build_lexical_key_positions(text: str) -> tuple[str, tuple[int, ...]]:
    key_chars: list[str] = []
    char_positions: list[int] = []
    for index, char in enumerate(str(text or "")):
        if not _is_key_char(char):
            continue
        key_chars.append(char.lower())
        char_positions.append(index)
    return "".join(key_chars), tuple(char_positions)


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
        text_length = len(lexical_text)
        lexical_key, lexical_key_positions = _build_lexical_key_positions(lexical_text)
        key_length = len(lexical_key)
        key_cursor = 0
        normalized_units = [
            (unit, _normalize_slot_text(unit.text))
            for unit in source_units
            if _normalize_slot_text(unit.text)
        ]

        for index, (unit, normalized_text) in enumerate(normalized_units):
            if key_cursor >= key_length:
                break
            start_key = key_cursor
            if index == len(normalized_units) - 1:
                end_key = key_length
            else:
                end_key = min(key_length, key_cursor + len(normalized_text))
            if end_key <= start_key:
                continue
            start = lexical_key_positions[start_key]
            end = lexical_key_positions[end_key - 1] + 1
            if end <= start:
                continue
            slots.append(
                SlowSlot(
                    slot_id=f"{unit.unit_id}:{start}-{end}",
                    text=lexical_text[start:end],
                    char_start=start,
                    char_end=end,
                    speaker_id=unit.speaker_id,
                    turn_id=unit.turn_id,
                    source_chunk_ids=tuple(unit.source_chunk_ids),
                    source_chunk_indices=tuple(unit.source_chunk_indices),
                    source_unit_ids=(unit.unit_id,),
                )
            )
            key_cursor = end_key

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

        if slots and slots[-1].char_end < text_length:
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
