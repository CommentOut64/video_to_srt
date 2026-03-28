"""Preparation token provenance 绑定器。"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.timeanchored_alignment.contracts import TokenUnit
from app.services.timeanchored_alignment.preparation.contracts import PreparedTokenUnit
from app.services.timeanchored_alignment.slow_window.contracts import WindowSourceUnit


def _is_key_char(char: str) -> bool:
    return char.isalnum() or ("\u4e00" <= char <= "\u9fff")


def _normalize_key(text: str) -> str:
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


def _dedup_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = str(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def _dedup_ints(values: tuple[int, ...]) -> tuple[int, ...]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        normalized = int(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


@dataclass(frozen=True)
class _SourceUnitSpan:
    unit: WindowSourceUnit
    char_start: int
    char_end: int


class TokenProvenanceBinder:
    """把 token-unit 绑定回 source unit provenance。"""

    def bind(
        self,
        *,
        text: str,
        token_units: tuple[TokenUnit, ...],
        source_units: tuple[WindowSourceUnit, ...],
    ) -> tuple[PreparedTokenUnit, ...]:
        lexical_text = str(text or "")
        if not lexical_text or not token_units:
            return tuple()

        source_spans = self._build_source_unit_spans(
            text=lexical_text,
            source_units=source_units,
        )
        if not source_spans:
            return tuple()

        prepared: list[PreparedTokenUnit] = []
        for index, token_unit in enumerate(token_units):
            char_start = int(token_unit.char_start)
            char_end = int(token_unit.char_end)
            if char_end <= char_start:
                continue
            matched_spans = self._match_source_spans(
                char_start=char_start,
                char_end=char_end,
                source_spans=source_spans,
            )
            if not matched_spans:
                continue
            source_chunk_ids = _dedup_strings(
                tuple(
                    chunk_id
                    for span in matched_spans
                    for chunk_id in tuple(span.unit.source_chunk_ids)
                )
            )
            source_chunk_indices = _dedup_ints(
                tuple(
                    chunk_index
                    for span in matched_spans
                    for chunk_index in tuple(span.unit.source_chunk_indices)
                )
            )
            source_unit_ids = tuple(str(span.unit.unit_id) for span in matched_spans)
            lead_span = matched_spans[0]
            prepared.append(
                PreparedTokenUnit(
                    unit_id=f"token-{index}:{char_start}-{char_end}",
                    token_text=lexical_text[char_start:char_end],
                    normalized_text=str(token_unit.token_text or lexical_text[char_start:char_end]),
                    char_start=char_start,
                    char_end=char_end,
                    speaker_id=lead_span.unit.speaker_id,
                    turn_id=lead_span.unit.turn_id,
                    source_chunk_ids=source_chunk_ids,
                    source_chunk_indices=source_chunk_indices,
                    source_unit_ids=source_unit_ids,
                )
            )
        return tuple(prepared)

    @staticmethod
    def _build_source_unit_spans(
        *,
        text: str,
        source_units: tuple[WindowSourceUnit, ...],
    ) -> tuple[_SourceUnitSpan, ...]:
        if not text or not source_units:
            return tuple()

        lexical_key, lexical_key_positions = _build_lexical_key_positions(text)
        if not lexical_key:
            return tuple()

        spans: list[_SourceUnitSpan] = []
        key_length = len(lexical_key)
        key_cursor = 0
        normalized_units = [
            (unit, _normalize_key(unit.text))
            for unit in source_units
            if _normalize_key(unit.text)
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
            char_start = int(lexical_key_positions[start_key])
            char_end = int(lexical_key_positions[end_key - 1]) + 1
            if char_end <= char_start:
                continue
            spans.append(
                _SourceUnitSpan(
                    unit=unit,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            key_cursor = end_key

        if not spans and source_units:
            lead_unit = source_units[0]
            return (
                _SourceUnitSpan(
                    unit=lead_unit,
                    char_start=0,
                    char_end=len(text),
                ),
            )

        if spans and spans[-1].char_end < len(text):
            last = spans[-1]
            spans[-1] = _SourceUnitSpan(
                unit=last.unit,
                char_start=last.char_start,
                char_end=len(text),
            )
        return tuple(spans)

    @staticmethod
    def _match_source_spans(
        *,
        char_start: int,
        char_end: int,
        source_spans: tuple[_SourceUnitSpan, ...],
    ) -> tuple[_SourceUnitSpan, ...]:
        overlaps = tuple(
            span
            for span in source_spans
            if int(span.char_start) < int(char_end) and int(span.char_end) > int(char_start)
        )
        if overlaps:
            return overlaps
        midpoint = (int(char_start) + int(char_end)) / 2.0
        for span in source_spans:
            if float(span.char_start) <= midpoint <= float(span.char_end):
                return (span,)
        if midpoint < float(source_spans[0].char_start):
            return (source_spans[0],)
        return (source_spans[-1],)
