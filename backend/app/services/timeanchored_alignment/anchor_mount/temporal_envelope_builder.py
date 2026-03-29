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

    _MAX_INTERPOLATION_RESIDUAL = 1

    @staticmethod
    def _build_unit_block_map(
        *,
        solve_result: ChainSolveResult,
    ) -> dict[int, str]:
        unit_to_block: dict[int, str] = {}
        for block in solve_result.committed_blocks:
            for unit_index in block.unit_indices:
                unit_to_block[int(unit_index)] = str(block.block_id)
        return unit_to_block

    @staticmethod
    def _build_block_lookup(
        *,
        solve_result: ChainSolveResult,
    ) -> dict[str, object]:
        return {
            str(block.block_id): block
            for block in solve_result.committed_blocks
        }

    @staticmethod
    def _unit_weight(token_unit) -> float:
        char_span = int(token_unit.char_end) - int(token_unit.char_start)
        return float(max(char_span, 1))

    def _build_resolved_bounds(
        self,
        *,
        input_view: AnchorMountInputView,
        solve_result: ChainSolveResult,
    ) -> dict[int, tuple[float, float]]:
        resolved_bounds: dict[int, tuple[float, float]] = {}
        for block in solve_result.committed_blocks:
            if not block.hook_indices:
                continue
            block_start = min(float(input_view.fast_hooks[index].start) for index in block.hook_indices)
            block_end = max(float(input_view.fast_hooks[index].end) for index in block.hook_indices)
            if len(block.unit_indices) <= 1:
                resolved_bounds[int(block.unit_indices[0])] = (block_start, block_end)
                continue
            total_weight = sum(
                self._unit_weight(input_view.token_units[unit_index])
                for unit_index in block.unit_indices
            )
            block_span = max(block_end - block_start, 0.01)
            cursor = block_start
            cumulative_weight = 0.0
            for offset, unit_index in enumerate(block.unit_indices):
                weight = self._unit_weight(input_view.token_units[unit_index])
                cumulative_weight += weight
                if offset == len(block.unit_indices) - 1 or total_weight <= 0.0:
                    right_bound = block_end
                else:
                    ratio = cumulative_weight / total_weight
                    right_bound = block_start + block_span * ratio
                right_bound = max(right_bound, cursor + 0.01)
                resolved_bounds[int(unit_index)] = (cursor, right_bound)
                cursor = right_bound
        return resolved_bounds

    def build(
        self,
        *,
        input_view: AnchorMountInputView,
        solve_result: ChainSolveResult,
    ) -> tuple[tuple[AnchorMountItem, ...], tuple[TemporalEnvelope, ...]]:
        total_units = max(len(input_view.token_units), 1)
        coverage_start = min(binding.chunk_start for binding in input_view.coverage.chunk_bindings)
        coverage_end = max(binding.chunk_end for binding in input_view.coverage.chunk_bindings)
        coverage_span = max(coverage_end - coverage_start, 0.01)
        items: list[AnchorMountItem] = []
        envelopes: list[TemporalEnvelope] = []
        unit_to_block_id = self._build_unit_block_map(solve_result=solve_result)
        block_lookup = self._build_block_lookup(solve_result=solve_result)
        resolved_bounds = self._build_resolved_bounds(
            input_view=input_view,
            solve_result=solve_result,
        )
        unresolved_span_sizes = self._build_unresolved_span_sizes(solve_result=solve_result)

        for unit_index, token_unit in enumerate(input_view.token_units):
            hook_indices = solve_result.unit_to_hook_indices[unit_index]
            anchor_kind = solve_result.unit_anchor_kinds[unit_index]
            hook_ids = tuple(f"hook-{index}" for index in hook_indices)
            block_id = unit_to_block_id.get(unit_index)
            block = block_lookup.get(str(block_id)) if block_id is not None else None
            trust_tier = getattr(block, "trust_tier", "unreviewed")
            ambiguity_cluster_ids = tuple(getattr(block, "ambiguity_cluster_ids", ()) or ())
            if hook_indices:
                left_bound, right_bound = resolved_bounds[unit_index]
                envelope_kind = "merged" if block is not None and len(block.unit_indices) > 1 else "anchored"
                confidence = 1.0 if envelope_kind == "anchored" else 0.85
                gap_state = "resolved"
            else:
                unresolved_span_size = unresolved_span_sizes.get(unit_index, 0)
                previous_resolved = max(
                    (index for index in resolved_bounds if index < unit_index),
                    default=None,
                )
                next_resolved = min(
                    (index for index in resolved_bounds if index > unit_index),
                    default=None,
                )
                if previous_resolved is not None and next_resolved is not None:
                    previous_end = resolved_bounds[previous_resolved][1]
                    next_start = resolved_bounds[next_resolved][0]
                    local_start = previous_end
                    local_end = max(next_start, previous_end + 0.01)
                    span = max(local_end - local_start, 0.03)
                    ratio = (unit_index - previous_resolved) / max(next_resolved - previous_resolved, 1)
                    left_bound = local_start + span * max(0.0, ratio - 0.15)
                    right_bound = local_start + span * min(1.0, ratio + 0.15)
                    if unresolved_span_size <= self._MAX_INTERPOLATION_RESIDUAL:
                        envelope_kind = "inferred"
                        confidence = 0.45
                        gap_state = "residual"
                    else:
                        envelope_kind = "unresolved"
                        confidence = 0.2
                        gap_state = "large_residual"
                else:
                    slice_width = coverage_span / float(total_units)
                    left_bound = coverage_start + slice_width * unit_index
                    right_bound = left_bound + slice_width
                    envelope_kind = "unresolved"
                    confidence = 0.2
                    gap_state = (
                        "large_residual"
                        if unresolved_span_size > self._MAX_INTERPOLATION_RESIDUAL
                        else "unexamined"
                    )
            provisional_start = float(left_bound)
            provisional_end = max(float(right_bound), provisional_start + 0.01)
            envelopes.append(
                TemporalEnvelope(
                    unit_id=token_unit.unit_id,
                    envelope_kind=envelope_kind,
                    left_bound=float(left_bound),
                    right_bound=float(right_bound),
                    preferred_start=None if envelope_kind in {"inferred", "unresolved"} else provisional_start,
                    preferred_end=None if envelope_kind in {"inferred", "unresolved"} else provisional_end,
                    provisional_start=provisional_start,
                    provisional_end=provisional_end,
                    confidence=confidence,
                    source_hook_ids=hook_ids,
                    source_chunk_ids=token_unit.source_chunk_ids,
                    source_chunk_indices=token_unit.source_chunk_indices,
                    cross_chunk_lock=False,
                    gap_state=gap_state,
                    diagnostics={
                        "alignment_block_id": block_id,
                        "ambiguity_cluster_ids": list(ambiguity_cluster_ids),
                    },
                )
            )
            items.append(
                AnchorMountItem(
                    unit_id=token_unit.unit_id,
                    unit_index=unit_index,
                    token_text=token_unit.token_text,
                    display_text=token_unit.token_text,
                    normalized_text=token_unit.normalized_text,
                    speaker_id=token_unit.speaker_id,
                    turn_id=token_unit.turn_id,
                    source_chunk_ids=token_unit.source_chunk_ids,
                    source_chunk_indices=token_unit.source_chunk_indices,
                    mount_status=envelope_kind,
                    anchor_kind=anchor_kind,
                    envelope_index=unit_index,
                    source_hook_ids=hook_ids,
                    match_confidence=confidence,
                    cross_chunk_lock_ids=tuple(),
                    alignment_block_id=unit_to_block_id.get(unit_index),
                    trust_tier=trust_tier,
                    ambiguity_cluster_id=ambiguity_cluster_ids[0] if ambiguity_cluster_ids else None,
                )
            )
        return tuple(items), tuple(envelopes)

    @staticmethod
    def _build_unresolved_span_sizes(
        *,
        solve_result: ChainSolveResult,
    ) -> dict[int, int]:
        span_sizes: dict[int, int] = {}
        current_run: list[int] = []
        for unit_index, hook_indices in enumerate(solve_result.unit_to_hook_indices):
            if hook_indices:
                if current_run:
                    for unresolved_index in current_run:
                        span_sizes[unresolved_index] = len(current_run)
                    current_run = []
                continue
            current_run.append(unit_index)
        if current_run:
            for unresolved_index in current_run:
                span_sizes[unresolved_index] = len(current_run)
        return span_sizes
