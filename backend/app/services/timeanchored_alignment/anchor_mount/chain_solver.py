"""局部块串联为单调主链。"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorMountInputView,
    LocalAlignmentBlock,
)


@dataclass(frozen=True)
class ChainSolveResult:
    committed_blocks: tuple[LocalAlignmentBlock, ...]
    unit_to_hook_indices: tuple[tuple[int, ...], ...]
    unit_anchor_kinds: tuple[str, ...]
    unresolved_unit_indices: tuple[int, ...]
    reseed_count: int
    compatibility_edge_count: int = 0
    avg_in_degree: float = 0.0
    max_in_degree: int = 0
    solver_elapsed_ms: float = 0.0


class ChainSolver:
    """在稀疏兼容图上做全局最优单调链 DP。"""

    _MONOTONIC_CORRIDOR = 3
    _MAX_HOOK_JUMP = 12
    _MAX_UNIT_JUMP = 12
    _DISTANCE_RATIO_LIMIT = 3.0
    _HARD_BOUNDARY_MARKS = {"。", "！", "？", ".", "!", "?", ";", "；"}

    def solve(
        self,
        *,
        input_view: AnchorMountInputView,
        blocks: tuple[LocalAlignmentBlock, ...],
    ) -> ChainSolveResult:
        start_ts = perf_counter()
        ordered_blocks = sorted(
            blocks,
            key=lambda item: (
                min(item.unit_indices),
                min(item.hook_indices),
                max(item.unit_indices),
                max(item.hook_indices),
            ),
        )
        incoming_edges = self._build_sparse_incoming_edges(
            input_view=input_view,
            ordered_blocks=ordered_blocks,
        )
        best_scores = [float("-inf")] * len(ordered_blocks)
        parents = [-1] * len(ordered_blocks)

        for index, block in enumerate(ordered_blocks):
            best_scores[index] = float(block.score)
            for previous_index in incoming_edges[index]:
                candidate_score = (
                    best_scores[previous_index]
                    + self._transition_bonus(
                        previous=ordered_blocks[previous_index],
                        current=block,
                    )
                    + float(block.score)
                )
                if candidate_score > best_scores[index]:
                    best_scores[index] = candidate_score
                    parents[index] = previous_index

        committed = self._reconstruct_best_chain(
            ordered_blocks=ordered_blocks,
            best_scores=best_scores,
            parents=parents,
        )

        unit_to_hook_indices: list[tuple[int, ...]] = [tuple() for _ in input_view.token_units]
        unit_anchor_kinds: list[str] = ["none" for _ in input_view.token_units]
        for block in committed:
            if len(block.unit_indices) == len(block.hook_indices):
                for unit_index, hook_index in zip(block.unit_indices, block.hook_indices, strict=False):
                    unit_to_hook_indices[unit_index] = (int(hook_index),)
                    unit_anchor_kinds[unit_index] = block.anchor_kind
                continue
            for unit_index in block.unit_indices:
                unit_to_hook_indices[unit_index] = tuple(block.hook_indices)
                unit_anchor_kinds[unit_index] = block.anchor_kind

        unresolved = tuple(
            index for index, hook_indices in enumerate(unit_to_hook_indices) if not hook_indices
        )
        edge_count = sum(len(edges) for edges in incoming_edges.values())
        in_degrees = [len(incoming_edges[index]) for index in range(len(ordered_blocks))]
        elapsed_ms = (perf_counter() - start_ts) * 1000.0
        return ChainSolveResult(
            committed_blocks=tuple(committed),
            unit_to_hook_indices=tuple(unit_to_hook_indices),
            unit_anchor_kinds=tuple(unit_anchor_kinds),
            unresolved_unit_indices=unresolved,
            reseed_count=0,
            compatibility_edge_count=edge_count,
            avg_in_degree=(sum(in_degrees) / len(in_degrees)) if in_degrees else 0.0,
            max_in_degree=max(in_degrees) if in_degrees else 0,
            solver_elapsed_ms=elapsed_ms,
        )

    def _build_sparse_incoming_edges(
        self,
        *,
        input_view: AnchorMountInputView,
        ordered_blocks: list[LocalAlignmentBlock],
    ) -> dict[int, list[int]]:
        incoming_edges: dict[int, list[int]] = {index: [] for index in range(len(ordered_blocks))}
        hard_boundaries_after_unit = self._collect_hard_boundaries_after_unit(input_view=input_view)
        for current_index, current in enumerate(ordered_blocks):
            for previous_index in range(current_index):
                previous = ordered_blocks[previous_index]
                if not self._is_compatible(
                    input_view=input_view,
                    previous=previous,
                    current=current,
                    hard_boundaries_after_unit=hard_boundaries_after_unit,
                ):
                    continue
                incoming_edges[current_index].append(previous_index)
        return incoming_edges

    def _is_compatible(
        self,
        *,
        input_view: AnchorMountInputView,
        previous: LocalAlignmentBlock,
        current: LocalAlignmentBlock,
        hard_boundaries_after_unit: set[int],
    ) -> bool:
        prev_unit_end = max(previous.unit_indices)
        curr_unit_start = min(current.unit_indices)
        prev_hook_end = max(previous.hook_indices)
        curr_hook_start = min(current.hook_indices)
        if prev_unit_end >= curr_unit_start:
            return False
        if prev_hook_end >= curr_hook_start:
            return False

        unit_jump = curr_unit_start - prev_unit_end
        hook_jump = curr_hook_start - prev_hook_end
        if unit_jump > self._MAX_UNIT_JUMP or hook_jump > self._MAX_HOOK_JUMP:
            return False
        if abs(unit_jump - hook_jump) > self._MONOTONIC_CORRIDOR:
            return False

        min_gap = min(max(unit_jump, 1), max(hook_jump, 1))
        max_gap = max(max(unit_jump, 1), max(hook_jump, 1))
        if (max_gap / float(min_gap)) > self._DISTANCE_RATIO_LIMIT:
            return False

        for boundary_after in hard_boundaries_after_unit:
            if prev_unit_end <= boundary_after < curr_unit_start:
                return False

        prev_unit = input_view.token_units[prev_unit_end]
        curr_unit = input_view.token_units[curr_unit_start]
        prev_speaker_id = getattr(prev_unit, "speaker_id", None)
        curr_speaker_id = getattr(curr_unit, "speaker_id", None)
        if prev_speaker_id is not None and curr_speaker_id is not None and prev_speaker_id != curr_speaker_id:
            return False
        prev_turn_id = getattr(prev_unit, "turn_id", None)
        curr_turn_id = getattr(curr_unit, "turn_id", None)
        if prev_turn_id is not None and curr_turn_id is not None and prev_turn_id != curr_turn_id:
            return False
        return True

    def _collect_hard_boundaries_after_unit(
        self,
        *,
        input_view: AnchorMountInputView,
    ) -> set[int]:
        boundaries: set[int] = set()
        token_units = tuple(input_view.token_units or ())
        for evidence in (getattr(input_view, "punctuation_evidences", ()) or ()):
            mark = str(getattr(evidence, "mark", "") or "")
            if mark not in self._HARD_BOUNDARY_MARKS:
                continue
            char_index = int(getattr(evidence, "source_char_index", -1) or -1)
            for unit_index, token_unit in enumerate(token_units):
                if token_unit.char_start <= char_index < token_unit.char_end:
                    if unit_index < len(token_units) - 1:
                        boundaries.add(unit_index)
                    break
        return boundaries

    @staticmethod
    def _transition_bonus(
        *,
        previous: LocalAlignmentBlock,
        current: LocalAlignmentBlock,
    ) -> float:
        unit_gap = min(current.unit_indices) - max(previous.unit_indices) - 1
        hook_gap = min(current.hook_indices) - max(previous.hook_indices) - 1
        diagonal_penalty = abs(unit_gap - hook_gap)
        distance_penalty = 0.03 * float(max(unit_gap, 0) + max(hook_gap, 0))
        return max(0.0, 0.12 - 0.02 * float(diagonal_penalty) - distance_penalty)

    @staticmethod
    def _reconstruct_best_chain(
        *,
        ordered_blocks: list[LocalAlignmentBlock],
        best_scores: list[float],
        parents: list[int],
    ) -> list[LocalAlignmentBlock]:
        if not ordered_blocks:
            return []
        best_index = max(range(len(ordered_blocks)), key=lambda index: best_scores[index])
        chain: list[LocalAlignmentBlock] = []
        cursor = best_index
        while cursor >= 0:
            chain.append(ordered_blocks[cursor])
            cursor = parents[cursor]
        chain.reverse()
        return chain
