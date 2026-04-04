"""Window 级 time base 聚合器。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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

        chunk_bindings_by_index = {
            int(binding.chunk_index): binding
            for binding in ready_window.coverage.chunk_bindings
        }

        for chunk_index in ready_window.source_chunk_indices:
            package = packages_by_index.get(int(chunk_index))
            if package is None:
                continue
            binding = chunk_bindings_by_index.get(int(chunk_index))
            raw_units.extend(self._rebase_units(package.raw_units, binding=binding))
            word_units.extend(self._rebase_units(package.word_units, binding=binding))
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
        blank_track: list[float] = []
        sparse_logits: list[dict[str, Any]] = []
        encoder_out_lens_total = 0
        for chunk_index in ready_window.source_chunk_indices:
            package = packages_by_index.get(int(chunk_index))
            if package is None:
                continue
            package_metadata = dict(getattr(package, "metadata", {}) or {})
            blank_track.extend(float(item) for item in (package_metadata.get("blank_track") or ()))
            encoder_out_lens_value = package_metadata.get("encoder_out_lens")
            if encoder_out_lens_value is not None:
                try:
                    encoder_out_lens_total += int(encoder_out_lens_value)
                except (TypeError, ValueError):
                    pass
            for item in (package_metadata.get("sparse_logits") or ()):
                if not isinstance(item, dict):
                    continue
                sparse_logits.append({"chunk_index": int(chunk_index), **item})
        if blank_track:
            metadata["blank_track"] = blank_track
        if encoder_out_lens_total > 0:
            metadata["encoder_out_lens"] = encoder_out_lens_total
        if sparse_logits:
            metadata["sparse_logits"] = sparse_logits
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

    @staticmethod
    def _rebase_units(
        units: Iterable[TimeBaseUnit],
        *,
        binding: WindowChunkBinding | None,
    ) -> tuple[TimeBaseUnit, ...]:
        resolved_units = tuple(units or ())
        if not resolved_units or binding is None:
            return resolved_units

        chunk_start = float(getattr(binding, "chunk_start", 0.0) or 0.0)
        chunk_end = float(getattr(binding, "chunk_end", chunk_start) or chunk_start)
        chunk_duration = max(chunk_end - chunk_start, 0.0)
        min_start = min(float(unit.start) for unit in resolved_units)
        max_end = max(float(unit.end) for unit in resolved_units)
        tolerance = 1e-4

        is_already_absolute = (
            min_start >= (chunk_start - tolerance)
            and max_end <= (chunk_end + tolerance)
        )
        if is_already_absolute:
            return resolved_units

        is_chunk_relative = (
            min_start >= -tolerance
            and max_end <= (chunk_duration + tolerance)
        )
        if not is_chunk_relative:
            return resolved_units

        return tuple(
            replace(
                unit,
                start=float(unit.start) + chunk_start,
                end=float(unit.end) + chunk_start,
            )
            for unit in resolved_units
        )
