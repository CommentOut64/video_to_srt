"""标点位置映射工具（strict / tolerant）。"""

from __future__ import annotations

from bisect import bisect_left
import difflib
from typing import Dict, Sequence

from app.services.punctuation.base import PuncPosition


class PunctPositionMapper:
    """在不同 clean_text_ref 之间映射标点位置。"""

    def remap(
        self,
        *,
        source_ref: str,
        target_ref: str,
        positions: Sequence[PuncPosition],
        mode: str = "strict",
    ) -> list[PuncPosition]:
        if not positions:
            return []
        normalized_mode = str(mode or "strict").strip().lower()
        if normalized_mode not in {"strict", "tolerant"}:
            normalized_mode = "strict"
        if normalized_mode == "strict":
            if not self.is_compatible(source_ref=source_ref, target_ref=target_ref):
                return []
            return list(positions)
        if self.is_compatible(source_ref=source_ref, target_ref=target_ref):
            return list(positions)
        return self._remap_tolerant(
            source_ref=str(source_ref or ""),
            target_ref=str(target_ref or ""),
            positions=positions,
        )

    def is_compatible(self, *, source_ref: str, target_ref: str) -> bool:
        if source_ref == target_ref:
            return True
        if not source_ref or not target_ref:
            return False
        if len(source_ref) != len(target_ref):
            return False
        for source_char, target_char in zip(source_ref, target_ref):
            if source_char == target_char:
                continue
            if source_char.isspace() and target_char.isspace():
                continue
            if self._normalize_ref_char(source_char) == self._normalize_ref_char(target_char):
                continue
            return False
        return True

    def _remap_tolerant(
        self,
        *,
        source_ref: str,
        target_ref: str,
        positions: Sequence[PuncPosition],
    ) -> list[PuncPosition]:
        if not source_ref or not target_ref or not positions:
            return []
        source_chars = [self._normalize_ref_char(ch) for ch in source_ref]
        target_chars = [self._normalize_ref_char(ch) for ch in target_ref]
        matcher = difflib.SequenceMatcher(a=source_chars, b=target_chars, autojunk=False)
        if matcher.ratio() < 0.55:
            return []

        direct_map: Dict[int, int] = {}
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                continue
            span = min(i2 - i1, j2 - j1)
            for offset in range(span):
                direct_map[i1 + offset] = j1 + offset
        if not direct_map:
            return []

        mapped_indexes = sorted(direct_map.keys())
        remapped: list[PuncPosition] = []
        for position in positions:
            target_index = self._project_char_index(
                source_index=int(position.char_index),
                direct_map=direct_map,
                mapped_indexes=mapped_indexes,
                target_len=len(target_ref),
            )
            if target_index is None:
                continue
            remapped.append(
                PuncPosition(
                    char_index=target_index,
                    punctuation=position.punctuation,
                    confidence=position.confidence,
                )
            )
        if not remapped:
            return []

        dedup: Dict[tuple[int, str], PuncPosition] = {}
        for position in remapped:
            key = (int(position.char_index), str(position.punctuation))
            exists = dedup.get(key)
            if exists is None or float(position.confidence) > float(exists.confidence):
                dedup[key] = position
        return sorted(dedup.values(), key=lambda item: int(item.char_index))

    @staticmethod
    def _project_char_index(
        *,
        source_index: int,
        direct_map: Dict[int, int],
        mapped_indexes: Sequence[int],
        target_len: int,
    ) -> int | None:
        if target_len <= 0:
            return None
        if source_index in direct_map:
            return max(0, min(target_len - 1, int(direct_map[source_index])))
        if not mapped_indexes:
            return None
        pos = bisect_left(mapped_indexes, source_index)
        left = mapped_indexes[pos - 1] if pos > 0 else None
        right = mapped_indexes[pos] if pos < len(mapped_indexes) else None
        candidate: int | None
        if left is not None and right is not None and right != left:
            left_target = direct_map[left]
            right_target = direct_map[right]
            ratio = float(source_index - left) / float(right - left)
            candidate = int(round(left_target + ratio * (right_target - left_target)))
        elif left is not None:
            candidate = int(direct_map[left]) + (source_index - left)
        elif right is not None:
            candidate = int(direct_map[right]) - (right - source_index)
        else:
            candidate = None
        if candidate is None:
            return None
        return max(0, min(target_len - 1, int(candidate)))

    @staticmethod
    def _normalize_ref_char(value: str) -> str:
        if not value:
            return ""
        quote_alias = {
            "’": "'",
            "‘": "'",
            "`": "'",
            "“": '"',
            "”": '"',
        }
        return quote_alias.get(value, value).lower()
