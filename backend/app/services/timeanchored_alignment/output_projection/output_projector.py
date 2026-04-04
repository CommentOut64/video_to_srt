"""OutputProjection minimal contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.services.timeanchored_alignment.slow_window.contracts import WindowCoverage

if TYPE_CHECKING:
    from app.services.textflow.contracts import SubtitleBatch, SubtitleItem


@dataclass(frozen=True)
class OutputProjectionInput:
    """OutputProjection minimal input boundary."""

    window_id: str
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]
    coverage: WindowCoverage
    carrier_batch: "SubtitleBatch"
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
        if not str(getattr(self.carrier_batch, "chunk_id", "") or "").strip():
            raise ValueError("OutputProjectionInput.carrier_batch.chunk_id 不能为空")

    @property
    def carrier_role(self) -> str:
        return "window_text_carrier"


@dataclass(frozen=True)
class OutputProjector:
    """Thin projector shell for window->output-group dispatch."""

    def validate_input(self, data: OutputProjectionInput) -> OutputProjectionInput:
        return data

    def project(self, data: OutputProjectionInput) -> tuple["SubtitleBatch", ...]:
        from app.services.textflow.contracts import SubtitleBatch

        validated = self.validate_input(data)
        bindings = tuple(validated.coverage.chunk_bindings or ())
        if not bindings:
            return tuple()

        output_group_chunk_id = self._build_output_group_chunk_id(validated.window_id)
        grouped_items: list["SubtitleItem"] = []
        seen_segment_ids: set[str] = set()
        for item in tuple(validated.carrier_batch.items or ()):
            segment_id = str(item.segment_id)
            if segment_id in seen_segment_ids:
                continue
            seen_segment_ids.add(segment_id)
            grouped_items.append(
                self._clone_item_for_chunk(item=item, chunk_id=output_group_chunk_id)
            )

        source_chunk_ids = [str(item) for item in validated.source_chunk_ids]
        source_chunk_indices = [int(item) for item in validated.source_chunk_indices]
        binding_chunk_ids = [str(binding.chunk_id) for binding in bindings]
        binding_chunk_indices = [int(binding.chunk_index) for binding in bindings]
        projection_meta = {
            "window_id": str(validated.window_id),
            "carrier_chunk_id": str(validated.carrier_batch.chunk_id),
            "source_chunk_ids": source_chunk_ids,
            "source_chunk_indices": source_chunk_indices,
            "replace_scope_chunk_ids": source_chunk_ids,
            "replace_scope_chunk_indices": source_chunk_indices,
            "binding_chunk_ids": binding_chunk_ids,
            "binding_chunk_indices": binding_chunk_indices,
            "projection_mode": "window_group",
            "carrier_role": validated.carrier_role,
            "decision_metadata": dict(validated.decision_metadata or {}),
        }
        render_report = dict(validated.carrier_batch.render_report or {})
        base_diagnostics = dict(validated.carrier_batch.diagnostics or {})
        grouped_tuple = tuple(grouped_items)
        diagnostics = dict(base_diagnostics)
        diagnostics["projection"] = dict(projection_meta)
        diagnostics["output_trace"] = self._build_output_trace(grouped_tuple)
        return (
            SubtitleBatch(
                chunk_id=output_group_chunk_id,
                chunk_index=None,
                items=grouped_tuple,
                render_report=dict(render_report),
                diagnostics=diagnostics,
            ),
        )

    @staticmethod
    def _build_output_group_chunk_id(window_id: str) -> str:
        normalized_window_id = str(window_id or "").strip() or "window-unknown"
        return f"ow-{normalized_window_id}"

    @staticmethod
    def _clone_item_for_chunk(*, item: "SubtitleItem", chunk_id: str) -> "SubtitleItem":
        from app.services.textflow.contracts import SubtitleItem

        return SubtitleItem(
            segment_id=str(item.segment_id),
            chunk_id=str(chunk_id),
            start=float(item.start),
            end=float(item.end),
            text=str(item.text),
            status=str(item.status or "final"),
            source=str(item.source or "unknown"),
            speaker_id=item.speaker_id,
            turn_id=item.turn_id,
            trace=dict(item.trace or {}),
        )

    @staticmethod
    def _build_output_trace(items: tuple["SubtitleItem", ...]) -> list[dict[str, Any]]:
        traces: list[dict[str, Any]] = []
        for sentence_index, item in enumerate(items):
            trace = dict(item.trace or {})
            traces.append(
                {
                    "sentence_index": int(sentence_index),
                    "split_reason": str(trace.get("split_reason", "") or ""),
                    "split_risk": str(trace.get("split_risk", "") or ""),
                    "window_id": str(trace.get("window_id", "") or ""),
                    "pyannote_frame_time": trace.get("pyannote_frame_time"),
                    "mapped_cut_time": trace.get("mapped_cut_time", float(item.end)),
                    "mapping_quality": str(trace.get("mapping_quality", "") or ""),
                    "mapping_reason": str(trace.get("mapping_reason", "") or ""),
                    "sentence_start": float(item.start),
                    "sentence_end": float(item.end),
                }
            )
        return traces
