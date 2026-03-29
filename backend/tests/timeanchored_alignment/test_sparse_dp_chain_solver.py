from __future__ import annotations

from types import SimpleNamespace

from app.services.timeanchored_alignment.anchor_mount.chain_solver import ChainSolver
from app.services.timeanchored_alignment.anchor_mount.contracts import LocalAlignmentBlock


def _build_input_view(unit_count: int):
    token_units = tuple(
        SimpleNamespace(
            speaker_id="speaker-a",
            turn_id="turn-a",
            char_start=index * 2,
            char_end=index * 2 + 1,
        )
        for index in range(unit_count)
    )
    return SimpleNamespace(token_units=token_units, punctuation_evidences=tuple())


def test_sparse_dp_chain_solver_selects_globally_best_chain_instead_of_local_greedy_pick() -> None:
    solver = ChainSolver()

    result = solver.solve(
        input_view=_build_input_view(3),
        blocks=(
            LocalAlignmentBlock(
                block_id="block-a",
                unit_indices=(0,),
                hook_indices=(0,),
                score=0.70,
                block_kind="anchored",
                anchor_kind="exact",
            ),
            LocalAlignmentBlock(
                block_id="block-b",
                unit_indices=(1,),
                hook_indices=(1,),
                score=0.05,
                block_kind="anchored",
                anchor_kind="exact",
            ),
            LocalAlignmentBlock(
                block_id="block-c",
                unit_indices=(0, 1),
                hook_indices=(1, 2),
                score=1.0,
                block_kind="anchored",
                anchor_kind="merge",
            ),
            LocalAlignmentBlock(
                block_id="block-d",
                unit_indices=(2,),
                hook_indices=(3,),
                score=0.75,
                block_kind="anchored",
                anchor_kind="exact",
            ),
        ),
    )

    assert [block.block_id for block in result.committed_blocks] == ["block-c", "block-d"]
    assert result.compatibility_edge_count > 0
    assert result.solver_elapsed_ms >= 0.0


def test_sparse_dp_chain_solver_chooses_higher_value_conflicting_block() -> None:
    solver = ChainSolver()

    result = solver.solve(
        input_view=_build_input_view(2),
        blocks=(
            LocalAlignmentBlock(
                block_id="block-0",
                unit_indices=(0,),
                hook_indices=(1,),
                score=0.30,
                block_kind="anchored",
                anchor_kind="exact",
            ),
            LocalAlignmentBlock(
                block_id="block-1",
                unit_indices=(1,),
                hook_indices=(0,),
                score=0.95,
                block_kind="anchored",
                anchor_kind="exact",
            ),
        ),
    )

    assert result.reseed_count == 0
    assert [block.block_id for block in result.committed_blocks] == ["block-1"]


def test_sparse_dp_chain_solver_records_unit_to_hook_mapping_for_selected_chain() -> None:
    solver = ChainSolver()

    result = solver.solve(
        input_view=_build_input_view(3),
        blocks=(
            LocalAlignmentBlock(
                block_id="block-0",
                unit_indices=(0,),
                hook_indices=(0,),
                score=0.8,
                block_kind="anchored",
                anchor_kind="exact",
            ),
            LocalAlignmentBlock(
                block_id="block-1",
                unit_indices=(1, 2),
                hook_indices=(1, 2),
                score=0.9,
                block_kind="anchored",
                anchor_kind="merge",
            ),
        ),
    )

    assert result.unit_to_hook_indices == ((0,), (1,), (2,))
    assert result.unresolved_unit_indices == tuple()
