"""Preparation 来源归属绑定器。"""

from __future__ import annotations

from app.services.timeanchored_alignment.preparation.contracts import (
    PunctuationEvidence,
    PronunciationHint,
    SlowSlot,
)
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
        punctuation_evidences: tuple[PunctuationEvidence, ...] = tuple(),
        pronunciation_hints: tuple[PronunciationHint, ...] = tuple(),
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

        slots = self._split_slots_by_punctuation(
            slots=slots,
            lexical_text=lexical_text,
            punctuation_evidences=punctuation_evidences,
        )
        slots = self._split_slots_by_pronunciation_hints(
            slots=slots,
            lexical_text=lexical_text,
            pronunciation_hints=pronunciation_hints,
        )
        return tuple(slots)

    @staticmethod
    def _split_slots_by_punctuation(
        *,
        slots: list[SlowSlot],
        lexical_text: str,
        punctuation_evidences: tuple[PunctuationEvidence, ...],
    ) -> list[SlowSlot]:
        split_marks = frozenset({"。", "！", "？", ".", "!", "?", "，", ",", "；", ";", "：", ":"})
        split_points = sorted(
            {
                int(item.source_char_index) + (1 if str(item.attach_side) == "after" else 0)
                for item in punctuation_evidences
                if str(item.mark or "") in split_marks
            }
        )
        if not split_points:
            return slots
        return SourceAttributionBinder._split_slots_by_boundaries(
            slots=slots,
            lexical_text=lexical_text,
            split_points=tuple(split_points),
            tag="punct",
        )

    @staticmethod
    def _split_slots_by_pronunciation_hints(
        *,
        slots: list[SlowSlot],
        lexical_text: str,
        pronunciation_hints: tuple[PronunciationHint, ...],
    ) -> list[SlowSlot]:
        # under-segmentation 防线：保持完整 window 直传，只在 slot 粒度明显过粗时按 token 边界细分。
        # 触发条件使用“hints/slots 密度”而非 long/short window 分类，避免引入窗口分型语义。
        hint_count = len(pronunciation_hints)
        slot_count = len(slots)
        if slot_count <= 0 or hint_count < 16:
            return slots
        avg_hints_per_slot = hint_count / float(slot_count)
        if avg_hints_per_slot < 10.0:
            return slots
        split_points = sorted(
            {
                int(item.char_end)
                for item in pronunciation_hints
                if int(item.char_end) > 0
            }
        )
        if not split_points:
            return slots
        return SourceAttributionBinder._split_slots_by_boundaries(
            slots=slots,
            lexical_text=lexical_text,
            split_points=tuple(split_points),
            tag="hint",
        )

    @staticmethod
    def _split_slots_by_boundaries(
        *,
        slots: list[SlowSlot],
        lexical_text: str,
        split_points: tuple[int, ...],
        tag: str,
    ) -> list[SlowSlot]:
        refined: list[SlowSlot] = []
        for slot in slots:
            boundaries = [slot.char_start]
            boundaries.extend(
                point
                for point in split_points
                if slot.char_start < point < slot.char_end
            )
            boundaries.append(slot.char_end)

            if len(boundaries) <= 2:
                refined.append(slot)
                continue

            created = 0
            for index in range(len(boundaries) - 1):
                raw_start = int(boundaries[index])
                raw_end = int(boundaries[index + 1])
                start = SourceAttributionBinder._trim_left_whitespace(
                    text=lexical_text,
                    start=raw_start,
                    end=raw_end,
                )
                end = SourceAttributionBinder._trim_right_whitespace(
                    text=lexical_text,
                    start=start,
                    end=raw_end,
                )
                if end <= start:
                    continue
                created += 1
                refined.append(
                    SlowSlot(
                        slot_id=f"{slot.slot_id}:{tag}-{index}:{start}-{end}",
                        text=lexical_text[start:end],
                        char_start=start,
                        char_end=end,
                        speaker_id=slot.speaker_id,
                        turn_id=slot.turn_id,
                        source_chunk_ids=slot.source_chunk_ids,
                        source_chunk_indices=slot.source_chunk_indices,
                        source_unit_ids=slot.source_unit_ids,
                    )
                )

            if created <= 0:
                refined.append(slot)

        return refined

    @staticmethod
    def _trim_left_whitespace(*, text: str, start: int, end: int) -> int:
        cursor = int(start)
        while cursor < int(end) and str(text[cursor]).isspace():
            cursor += 1
        return cursor

    @staticmethod
    def _trim_right_whitespace(*, text: str, start: int, end: int) -> int:
        cursor = int(end)
        while cursor > int(start) and str(text[cursor - 1]).isspace():
            cursor -= 1
        return cursor
