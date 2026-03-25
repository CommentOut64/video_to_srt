"""Window 级 time base 聚合器。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from app.schemas.pipeline_context import ProcessingContext
from app.services.timeanchored_alignment.contracts import (
    CONTRACT_VERSION,
    TimeBasePackage,
    TimeBaseQuality,
    TimeBaseUnit,
)
from app.services.timeanchored_alignment.slow_window.contracts import ReadySlowWindow, WindowChunkBinding


@dataclass(frozen=True)
class WindowTimeBasePackage:
    window_id: str
    language: str
    raw_units: tuple[TimeBaseUnit, ...]
    word_units: tuple[TimeBaseUnit, ...]
    quality: TimeBaseQuality
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]
    chunk_bindings: tuple[WindowChunkBinding, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    frame_stride: float = 0.06
    source: str = "sensevoice_window"
    contract_version: str = CONTRACT_VERSION


class WindowTimeBaseAssembler:
    """把多个 chunk time base 聚成 window 级稳定包。"""

    def assemble(
        self,
        *,
        ready_window: ReadySlowWindow,
        source_contexts: Iterable[ProcessingContext],
    ) -> WindowTimeBasePackage:
        packages_by_index: dict[int, TimeBasePackage] = {}
        for ctx in source_contexts:
            if ctx.time_base_chunk is not None:
                packages_by_index[int(ctx.chunk_index)] = ctx.time_base_chunk

        raw_units: list[TimeBaseUnit] = []
        word_units: list[TimeBaseUnit] = []
        blank_ratio = 0.0
        avg_max_prob = 0.0
        low_prob_ratio = 0.0
        package_count = 0
        frame_stride = 0.06
        language = ready_window.language_profile.primary_language or "auto"

        for chunk_index in ready_window.source_chunk_indices:
            package = packages_by_index.get(int(chunk_index))
            if package is None:
                continue
            raw_units.extend(package.raw_units)
            word_units.extend(package.word_units)
            blank_ratio += float(package.quality.blank_ratio)
            avg_max_prob += float(package.quality.avg_max_prob)
            low_prob_ratio += float(package.quality.low_prob_ratio)
            frame_stride = float(getattr(package, "frame_stride", frame_stride) or frame_stride)
            language = str(getattr(package, "language", language) or language)
            package_count += 1

        if package_count == 0:
            raise ValueError("WindowTimeBaseAssembler 缺少可聚合的 chunk time base")

        quality = TimeBaseQuality(
            blank_ratio=blank_ratio / package_count,
            avg_max_prob=avg_max_prob / package_count,
            low_prob_ratio=low_prob_ratio / package_count,
            unit_count=len(raw_units),
            word_count=len(word_units),
        )
        metadata = {
            "candidate_window_ids": [ready_window.window_id],
            "coverage_roles": {
                str(binding.chunk_index): binding.role for binding in ready_window.coverage.chunk_bindings
            },
        }
        return WindowTimeBasePackage(
            window_id=ready_window.window_id,
            language=language,
            raw_units=tuple(raw_units),
            word_units=tuple(word_units),
            quality=quality,
            source_chunk_ids=ready_window.source_chunk_ids,
            source_chunk_indices=ready_window.source_chunk_indices,
            chunk_bindings=ready_window.coverage.chunk_bindings,
            metadata=metadata,
            frame_stride=frame_stride,
        )
