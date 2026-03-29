from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

from app.services.textflow.contracts import SubtitleBatch
from app.services.timeanchored_alignment.slow_window.contracts import WindowCoverage


@dataclass(frozen=True)
class CommitScope:
    window_id: str
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]
    coverage_segments: tuple[tuple[float, float], ...]
    affected_segment_ids: tuple[str, ...]
    timeline_validity: str
    generation_id: str


@dataclass(frozen=True)
class CommitScopeResolver:
    """按 window coverage 解析提交覆盖域。"""

    def build(
        self,
        *,
        window_id: str,
        source_chunk_ids: tuple[str, ...],
        source_chunk_indices: tuple[int, ...],
        coverage: WindowCoverage,
        owner_batch: SubtitleBatch,
        timeline_validity: str,
    ) -> CommitScope:
        normalized_chunk_ids = tuple(str(item) for item in source_chunk_ids if str(item))
        normalized_chunk_indices = tuple(int(item) for item in source_chunk_indices)
        coverage_segments = self._normalize_coverage_segments(coverage)
        affected_segment_ids = self._collect_affected_segment_ids(owner_batch=owner_batch)
        generation_id = self._build_generation_id(
            window_id=window_id,
            source_chunk_ids=normalized_chunk_ids,
            source_chunk_indices=normalized_chunk_indices,
            coverage_segments=coverage_segments,
            affected_segment_ids=affected_segment_ids,
            timeline_validity=timeline_validity,
        )
        return CommitScope(
            window_id=str(window_id),
            source_chunk_ids=normalized_chunk_ids,
            source_chunk_indices=normalized_chunk_indices,
            coverage_segments=coverage_segments,
            affected_segment_ids=affected_segment_ids,
            timeline_validity=str(timeline_validity or "valid"),
            generation_id=generation_id,
        )

    @staticmethod
    def _normalize_coverage_segments(coverage: WindowCoverage) -> tuple[tuple[float, float], ...]:
        segments = tuple(getattr(coverage, "core_segments", ()) or ())
        normalized = tuple(
            (float(start), float(end))
            for start, end in segments
            if float(end) > float(start)
        )
        if normalized:
            return normalized
        bindings = tuple(getattr(coverage, "chunk_bindings", ()) or ())
        return tuple(
            (float(binding.chunk_start), float(binding.chunk_end))
            for binding in bindings
            if float(binding.chunk_end) > float(binding.chunk_start)
        )

    @staticmethod
    def _collect_affected_segment_ids(*, owner_batch: SubtitleBatch) -> tuple[str, ...]:
        seen_segment_ids: set[str] = set()
        segment_ids: list[str] = []
        for item in tuple(owner_batch.items or ()):
            segment_id = str(item.segment_id or "").strip()
            if not segment_id or segment_id in seen_segment_ids:
                continue
            seen_segment_ids.add(segment_id)
            segment_ids.append(segment_id)
        return tuple(segment_ids)

    @staticmethod
    def _build_generation_id(
        *,
        window_id: str,
        source_chunk_ids: tuple[str, ...],
        source_chunk_indices: tuple[int, ...],
        coverage_segments: tuple[tuple[float, float], ...],
        affected_segment_ids: tuple[str, ...],
        timeline_validity: str,
    ) -> str:
        payload = "|".join(
            (
                str(window_id),
                ",".join(source_chunk_ids),
                ",".join(str(item) for item in source_chunk_indices),
                ",".join(f"{start:.6f}-{end:.6f}" for start, end in coverage_segments),
                ",".join(affected_segment_ids),
                str(timeline_validity or "valid"),
            )
        )
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
        generation_rank = f"{time.time_ns():020d}"
        return f"{window_id}:{generation_rank}:{digest}"
