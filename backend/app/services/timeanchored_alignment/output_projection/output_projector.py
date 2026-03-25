"""OutputProjection minimal contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.services.timeanchored_alignment.slow_window.contracts import WindowChunkBinding, WindowCoverage

if TYPE_CHECKING:
    from app.services.textflow.contracts import SubtitleBatch, SubtitleItem


@dataclass(frozen=True)
class OutputProjectionInput:
    """OutputProjection minimal input boundary."""

    window_id: str
    owner_chunk_id: str
    owner_chunk_index: int
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]
    coverage: WindowCoverage
    owner_carrier_batch: "SubtitleBatch"
    decision_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.window_id:
            raise ValueError("OutputProjectionInput.window_id 不能为空")
        if not self.owner_chunk_id:
            raise ValueError("OutputProjectionInput.owner_chunk_id 不能为空")
        if int(self.owner_chunk_index) < 0:
            raise ValueError("OutputProjectionInput.owner_chunk_index 必须 >= 0")
        if not self.source_chunk_ids:
            raise ValueError("OutputProjectionInput.source_chunk_ids 不能为空")
        if not self.source_chunk_indices:
            raise ValueError("OutputProjectionInput.source_chunk_indices 不能为空")
        if len(self.source_chunk_ids) != len(self.source_chunk_indices):
            raise ValueError(
                "OutputProjectionInput.source_chunk_ids/source_chunk_indices 长度必须一致"
            )
        if self.owner_chunk_id not in self.source_chunk_ids:
            raise ValueError("OutputProjectionInput.owner_chunk_id 必须属于 source_chunk_ids")
        if int(self.owner_chunk_index) not in self.source_chunk_indices:
            raise ValueError(
                "OutputProjectionInput.owner_chunk_index 必须属于 source_chunk_indices"
            )
        if not tuple(getattr(self.coverage, "chunk_bindings", ()) or ()):
            raise ValueError("OutputProjectionInput.coverage.chunk_bindings 不能为空")
        if str(self.owner_carrier_batch.chunk_id) != str(self.owner_chunk_id):
            raise ValueError(
                "OutputProjectionInput.owner_carrier_batch.chunk_id 必须等于 owner_chunk_id"
            )

    @property
    def owner_carrier_role(self) -> str:
        return "text_carrier_only"


@dataclass(frozen=True)
class OutputProjector:
    """Thin projector shell for window->chunk fan-out."""

    def validate_input(self, data: OutputProjectionInput) -> OutputProjectionInput:
        return data

    def project(self, data: OutputProjectionInput) -> tuple["SubtitleBatch", ...]:
        from app.services.textflow.contracts import SubtitleBatch

        validated = self.validate_input(data)
        bindings = tuple(validated.coverage.chunk_bindings or ())
        if not bindings:
            return tuple()

        buckets: list[list["SubtitleItem"]] = [[] for _ in bindings]
        bucket_segment_ids: list[set[str]] = [set() for _ in bindings]
        for item in tuple(validated.owner_carrier_batch.items or ()):
            target_index = self._resolve_target_binding(item=item, bindings=bindings)
            segment_id = str(item.segment_id)
            if segment_id in bucket_segment_ids[target_index]:
                continue
            bucket_segment_ids[target_index].add(segment_id)
            buckets[target_index].append(
                self._clone_item_for_chunk(item=item, chunk_id=str(bindings[target_index].chunk_id))
            )

        projection_meta = {
            "window_id": str(validated.window_id),
            "owner_chunk_id": str(validated.owner_chunk_id),
            "source_chunk_ids": [str(item) for item in validated.source_chunk_ids],
            "source_chunk_indices": [int(item) for item in validated.source_chunk_indices],
            "projection_mode": "coverage_chunk_bindings",
            "owner_carrier_role": validated.owner_carrier_role,
            "decision_metadata": dict(validated.decision_metadata or {}),
        }
        render_report = dict(validated.owner_carrier_batch.render_report or {})
        base_diagnostics = dict(validated.owner_carrier_batch.diagnostics or {})

        projected_batches: list["SubtitleBatch"] = []
        for index, binding in enumerate(bindings):
            items = tuple(buckets[index])
            diagnostics = dict(base_diagnostics)
            diagnostics["projection"] = dict(projection_meta)
            diagnostics["output_trace"] = self._build_output_trace(items)
            projected_batches.append(
                SubtitleBatch(
                    chunk_id=str(binding.chunk_id),
                    chunk_index=int(binding.chunk_index),
                    items=items,
                    render_report=dict(render_report),
                    diagnostics=diagnostics,
                )
            )
        return tuple(projected_batches)

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

    @classmethod
    def _resolve_target_binding(
        cls,
        *,
        item: "SubtitleItem",
        bindings: tuple[WindowChunkBinding, ...],
    ) -> int:
        best_index = 0
        best_score = -1.0
        best_owner_bonus = -1
        for index, binding in enumerate(bindings):
            overlap = cls._compute_overlap(
                start=float(item.start),
                end=float(item.end),
                binding=binding,
            )
            owner_bonus = 1 if bool(binding.is_owner) else 0
            if overlap > best_score or (overlap == best_score and owner_bonus > best_owner_bonus):
                best_index = index
                best_score = overlap
                best_owner_bonus = owner_bonus
        return best_index

    @staticmethod
    def _compute_overlap(*, start: float, end: float, binding: WindowChunkBinding) -> float:
        left = max(float(start), float(binding.chunk_start))
        right = min(float(end), float(binding.chunk_end))
        return max(0.0, right - left)

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
