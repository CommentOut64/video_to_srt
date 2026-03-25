"""时间区间构建。"""

from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.chain_solver import ChainSolveResult
from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorMountInputView,
    AnchorMountItem,
    TemporalEnvelope,
)


class TemporalEnvelopeBuilder:
    """把链路结果收口成 envelope 与 token 真相。"""

    def build(
        self,
        *,
        input_view: AnchorMountInputView,
        solve_result: ChainSolveResult,
    ) -> tuple[tuple[AnchorMountItem, ...], tuple[TemporalEnvelope, ...]]:
        total_slots = max(len(input_view.slots), 1)
        coverage_start = min(binding.chunk_start for binding in input_view.coverage.chunk_bindings)
        coverage_end = max(binding.chunk_end for binding in input_view.coverage.chunk_bindings)
        coverage_span = max(coverage_end - coverage_start, 0.01)
        items: list[AnchorMountItem] = []
        envelopes: list[TemporalEnvelope] = []

        resolved_bounds: dict[int, tuple[float, float]] = {}
        for slot_index, hook_indices in enumerate(solve_result.slot_to_hook_indices):
            if not hook_indices:
                continue
            starts = [input_view.fast_hooks[index].start for index in hook_indices]
            ends = [input_view.fast_hooks[index].end for index in hook_indices]
            resolved_bounds[slot_index] = (min(starts), max(ends))

        for slot_index, slot in enumerate(input_view.slots):
            hook_indices = solve_result.slot_to_hook_indices[slot_index]
            anchor_kind = solve_result.slot_anchor_kinds[slot_index]
            hook_ids = tuple(f"hook-{index}" for index in hook_indices)
            if hook_indices:
                left_bound, right_bound = resolved_bounds[slot_index]
                envelope_kind = "merged" if len(hook_indices) > 1 else "anchored"
                confidence = 1.0 if envelope_kind == "anchored" else 0.85
            else:
                previous_resolved = max(
                    (index for index in resolved_bounds if index < slot_index),
                    default=None,
                )
                next_resolved = min(
                    (index for index in resolved_bounds if index > slot_index),
                    default=None,
                )
                if previous_resolved is not None and next_resolved is not None:
                    previous_end = resolved_bounds[previous_resolved][1]
                    next_start = resolved_bounds[next_resolved][0]
                    local_start = previous_end
                    local_end = max(next_start, previous_end + 0.01)
                    span = max(local_end - local_start, 0.03)
                    ratio = (slot_index - previous_resolved) / max(next_resolved - previous_resolved, 1)
                    left_bound = local_start + span * max(0.0, ratio - 0.15)
                    right_bound = local_start + span * min(1.0, ratio + 0.15)
                    envelope_kind = "inferred"
                    confidence = 0.45
                else:
                    slice_width = coverage_span / float(total_slots)
                    left_bound = coverage_start + slice_width * slot_index
                    right_bound = left_bound + slice_width
                    envelope_kind = "unresolved"
                    confidence = 0.2
            provisional_start = float(left_bound)
            provisional_end = max(float(right_bound), provisional_start + 0.01)
            envelopes.append(
                TemporalEnvelope(
                    slot_id=slot.slot_id,
                    envelope_kind=envelope_kind,
                    left_bound=float(left_bound),
                    right_bound=float(right_bound),
                    preferred_start=None if envelope_kind in {"inferred", "unresolved"} else provisional_start,
                    preferred_end=None if envelope_kind in {"inferred", "unresolved"} else provisional_end,
                    provisional_start=provisional_start,
                    provisional_end=provisional_end,
                    confidence=confidence,
                    source_hook_ids=hook_ids,
                    source_chunk_ids=slot.source_chunk_ids,
                    source_chunk_indices=slot.source_chunk_indices,
                    cross_chunk_lock=False,
                )
            )
            items.append(
                AnchorMountItem(
                    slot_id=slot.slot_id,
                    slot_index=slot_index,
                    text_core=slot.text,
                    display_text=slot.text,
                    normalized_text=slot.text.strip().lower() or slot.text,
                    speaker_id=slot.speaker_id,
                    turn_id=slot.turn_id,
                    source_chunk_ids=slot.source_chunk_ids,
                    source_chunk_indices=slot.source_chunk_indices,
                    mount_status=envelope_kind,
                    anchor_kind=anchor_kind,
                    envelope_index=slot_index,
                    source_hook_ids=hook_ids,
                    match_confidence=confidence,
                    cross_chunk_lock_ids=tuple(),
                )
            )
        return tuple(items), tuple(envelopes)
