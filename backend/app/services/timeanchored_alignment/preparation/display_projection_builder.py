"""Preparation 展示投影构建器。"""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata

from app.services.timeanchored_alignment.preparation.contracts import (
    ProtectedUnit,
    SlowWindowTextPackage,
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


@dataclass(frozen=True)
class DisplayProjectionResult:
    window_text: SlowWindowTextPackage
    protected_units: tuple[ProtectedUnit, ...]
    source_char_to_text_index: tuple[int | None, ...]


class DisplayProjectionBuilder:
    """把原始慢流文本投影为 lexical surface。"""

    def build(
        self,
        *,
        source_text: str,
        protected_units: tuple[ProtectedUnit, ...],
        source_language: str,
    ) -> DisplayProjectionResult:
        text_chars: list[str] = []
        source_char_to_text_index: list[int | None] = []
        last_kept_index: int | None = None

        for index, char in enumerate(str(source_text or "")):
            protected = _is_protected_position(index, protected_units)
            keep_char = protected or not _is_punctuation(char)
            if keep_char:
                text_chars.append(char)
                last_kept_index = len(text_chars) - 1
                source_char_to_text_index.append(last_kept_index)
            else:
                source_char_to_text_index.append(last_kept_index)

        remapped_units: list[ProtectedUnit] = []
        for unit in protected_units:
            mapped_indices = [
                source_char_to_text_index[position]
                for position in range(unit.start, min(unit.end, len(source_char_to_text_index)))
                if source_char_to_text_index[position] is not None
            ]
            if not mapped_indices:
                continue
            remapped_units.append(
                ProtectedUnit(
                    start=min(mapped_indices),
                    end=max(mapped_indices) + 1,
                    kind=unit.kind,
                    text=unit.text,
                )
            )

        lexical_text = "".join(text_chars).strip()
        window_text = SlowWindowTextPackage(
            text=lexical_text,
            display_text=lexical_text,
            source_language=str(source_language or "auto"),
        )
        return DisplayProjectionResult(
            window_text=window_text,
            protected_units=tuple(remapped_units),
            source_char_to_text_index=tuple(source_char_to_text_index),
        )
