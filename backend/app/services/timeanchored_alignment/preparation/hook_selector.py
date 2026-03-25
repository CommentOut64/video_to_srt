"""Preparation fast hook 选择器。"""

from __future__ import annotations

from app.services.timeanchored_alignment.preparation.contracts import FastHook
from app.services.timeanchored_alignment.window_time_base_assembler import WindowTimeBasePackage


class HookSelector:
    """只允许从 WindowTimeBasePackage 选择 fast hooks。"""

    def select(self, *, window_time_base: WindowTimeBasePackage) -> tuple[FastHook, ...]:
        units = tuple(window_time_base.word_units or window_time_base.raw_units)
        hooks: list[FastHook] = []
        for unit in units:
            binding = self._resolve_binding(
                start=float(unit.start),
                end=float(unit.end),
                window_time_base=window_time_base,
            )
            hooks.append(
                FastHook(
                    hook_text=str(unit.text),
                    start=float(unit.start),
                    end=float(unit.end),
                    confidence=float(unit.confidence),
                    source_chunk_id=str(binding.chunk_id),
                    source_chunk_index=int(binding.chunk_index),
                    token_type=str(unit.token_type),
                )
            )
        return tuple(hooks)

    @staticmethod
    def _resolve_binding(*, start: float, end: float, window_time_base: WindowTimeBasePackage):
        best_binding = window_time_base.chunk_bindings[0]
        best_overlap = -1.0
        for binding in window_time_base.chunk_bindings:
            overlap = max(
                0.0,
                min(float(end), float(binding.chunk_end)) - max(float(start), float(binding.chunk_start)),
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_binding = binding
        return best_binding
