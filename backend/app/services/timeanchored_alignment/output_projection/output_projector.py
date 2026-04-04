"""OutputProjection minimal contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.services.timeanchored_alignment.slow_window.contracts import WindowCoverage

if TYPE_CHECKING:
    from app.services.textflow.contracts import (
        ChunkSentenceIndex,
        SentenceRecord,
        SubtitleBatch,
    )


@dataclass(frozen=True)
class OutputProjectionResult:
    """输出层内部 sentence-first 投影结果。"""

    chunk_id: str
    sentence_records: tuple["SentenceRecord", ...]
    chunk_sentence_indices: tuple["ChunkSentenceIndex", ...]
    subtitle_batch_compat: "SubtitleBatch"
    chunk_index: int | None = None


@dataclass(frozen=True)
class OutputProjectionInput:
    """OutputProjection minimal input boundary."""

    window_id: str
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]
    coverage: WindowCoverage
    carrier_chunk_id: str
    carrier_sentence_records: tuple["SentenceRecord", ...]
    carrier_chunk_sentence_indices: tuple["ChunkSentenceIndex", ...]
    decision_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.window_id:
            raise ValueError("OutputProjectionInput.window_id 不能为空")
        if not self.source_chunk_ids:
            raise ValueError("OutputProjectionInput.source_chunk_ids 不能为空")
        if not self.source_chunk_indices:
            raise ValueError("OutputProjectionInput.source_chunk_indices 不能为空")
        if len(self.source_chunk_ids) != len(self.source_chunk_indices):
            raise ValueError(
                "OutputProjectionInput.source_chunk_ids/source_chunk_indices 长度必须一致"
            )
        if not tuple(getattr(self.coverage, "chunk_bindings", ()) or ()):
            raise ValueError("OutputProjectionInput.coverage.chunk_bindings 不能为空")
        if not str(self.carrier_chunk_id or "").strip():
            raise ValueError("OutputProjectionInput.carrier_chunk_id 不能为空")
        if not self.carrier_chunk_sentence_indices:
            raise ValueError("OutputProjectionInput.carrier_chunk_sentence_indices 不能为空")

    @property
    def carrier_role(self) -> str:
        return "window_text_carrier"


@dataclass(frozen=True)
class OutputProjector:
    """Thin projector shell for window->output-group dispatch."""

    def validate_input(self, data: OutputProjectionInput) -> OutputProjectionInput:
        return data

    def project(self, data: OutputProjectionInput) -> tuple[OutputProjectionResult, ...]:
        from app.services.textflow.subtitle_delivery import SubtitleDelivery

        validated = self.validate_input(data)
        bindings = tuple(validated.coverage.chunk_bindings or ())
        if not bindings:
            return tuple()

        internal_chunk_id = str(validated.carrier_chunk_id)
        output_group_chunk_id = self._build_output_group_chunk_id(validated.window_id)
        source_chunk_ids = [str(item) for item in validated.source_chunk_ids]
        source_chunk_indices = [int(item) for item in validated.source_chunk_indices]
        binding_chunk_ids = [str(binding.chunk_id) for binding in bindings]
        binding_chunk_indices = [int(binding.chunk_index) for binding in bindings]
        projection_meta = {
            "window_id": str(validated.window_id),
            "carrier_chunk_id": str(validated.carrier_chunk_id),
            "source_chunk_ids": source_chunk_ids,
            "source_chunk_indices": source_chunk_indices,
            "overlap_chunk_ids": binding_chunk_ids,
            "overlap_chunk_indices": binding_chunk_indices,
            "replace_scope_chunk_ids": source_chunk_ids,
            "replace_scope_chunk_indices": source_chunk_indices,
            "binding_chunk_ids": binding_chunk_ids,
            "binding_chunk_indices": binding_chunk_indices,
            "projection_mode": "window_group",
            "carrier_role": validated.carrier_role,
            "decision_metadata": dict(validated.decision_metadata or {}),
        }
        projected_records = self._project_sentence_records(
            carrier_sentence_records=validated.carrier_sentence_records,
            source_chunk_ids=tuple(source_chunk_ids),
            overlap_chunk_ids=tuple(binding_chunk_ids),
            replace_scope_chunk_ids=tuple(source_chunk_ids),
        )
        projected_indices = self._project_chunk_sentence_indices(
            chunk_id=internal_chunk_id,
            carrier_chunk_sentence_indices=validated.carrier_chunk_sentence_indices,
            sentence_records=projected_records,
            projection_meta=projection_meta,
        )
        subtitle_delivery = SubtitleDelivery()
        subtitle_batch_compat = subtitle_delivery.build_batch_from_sentence_records(
            chunk_id=output_group_chunk_id,
            chunk_index=None,
            sentence_records=projected_records,
            diagnostics={
                "source": "output_projector",
                "projection": dict(projection_meta),
            },
        )
        return (
            OutputProjectionResult(
                chunk_id=internal_chunk_id,
                chunk_index=None,
                sentence_records=projected_records,
                chunk_sentence_indices=projected_indices,
                subtitle_batch_compat=subtitle_batch_compat,
            ),
        )

    @staticmethod
    def _build_output_group_chunk_id(window_id: str) -> str:
        normalized_window_id = str(window_id or "").strip() or "window-unknown"
        return f"ow-{normalized_window_id}"

    @staticmethod
    def _project_sentence_records(
        *,
        carrier_sentence_records: tuple["SentenceRecord", ...],
        source_chunk_ids: tuple[str, ...],
        overlap_chunk_ids: tuple[str, ...],
        replace_scope_chunk_ids: tuple[str, ...],
    ) -> tuple["SentenceRecord", ...]:
        from app.services.textflow.contracts import SentenceRecord

        projected: list[SentenceRecord] = []
        seen_sentence_ids: set[str] = set()
        for record in carrier_sentence_records:
            sentence_id = str(record.sentence_id)
            if sentence_id in seen_sentence_ids:
                continue
            seen_sentence_ids.add(sentence_id)
            projected.append(
                SentenceRecord(
                    sentence_id=sentence_id,
                    text=str(record.text),
                    start=float(record.start),
                    end=float(record.end),
                    source_chunk_ids=tuple(record.source_chunk_ids or source_chunk_ids),
                    overlap_chunk_ids=tuple(record.overlap_chunk_ids or overlap_chunk_ids),
                    replace_scope_chunk_ids=tuple(
                        record.replace_scope_chunk_ids or replace_scope_chunk_ids
                    ),
                    route=str(record.route or ""),
                    trace=dict(record.trace or {}),
                    metadata=dict(record.metadata or {}),
                )
            )
        return tuple(projected)

    @staticmethod
    def _project_chunk_sentence_indices(
        *,
        chunk_id: str,
        carrier_chunk_sentence_indices: tuple["ChunkSentenceIndex", ...],
        sentence_records: tuple["SentenceRecord", ...],
        projection_meta: dict[str, Any],
    ) -> tuple["ChunkSentenceIndex", ...]:
        from app.services.textflow.contracts import ChunkSentenceIndex

        carrier_metadata = (
            dict(carrier_chunk_sentence_indices[0].metadata or {})
            if carrier_chunk_sentence_indices
            else {}
        )
        merged_metadata = dict(carrier_metadata)
        merged_metadata.update({"projection": dict(projection_meta)})
        return (
            ChunkSentenceIndex(
                chunk_id=str(chunk_id),
                sentence_ids=tuple(str(record.sentence_id) for record in sentence_records),
                metadata=merged_metadata,
            ),
        )
