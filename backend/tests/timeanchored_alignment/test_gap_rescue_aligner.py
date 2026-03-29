from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.chain_solver import ChainSolveResult
from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorMountInputView,
    LocalAlignmentBlock,
)
from app.services.timeanchored_alignment.anchor_mount.gap_rescue_aligner import (
    GapRescueAligner,
)
from app.services.timeanchored_alignment.anchor_mount.window_alignment_state import (
    WindowAlignmentState,
)
from app.services.timeanchored_alignment.preparation.contracts import (
    FastHook,
    PreparedTokenUnit,
    SlowWindowTextPackage,
)
from app.services.timeanchored_alignment.slow_window.contracts import (
    WindowChunkBinding,
    WindowCoverage,
)


def _build_input_view() -> AnchorMountInputView:
    return AnchorMountInputView(
        window_id="window-gap-rescue-001",
        owner_chunk_id="chunk-1",
        owner_chunk_index=1,
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        language="en",
        token_units=(
            PreparedTokenUnit(
                unit_id="unit-0",
                token_text="hello",
                normalized_text="hello",
                char_start=0,
                char_end=5,
                speaker_id="speaker-a",
                turn_id="turn-a",
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
            ),
            PreparedTokenUnit(
                unit_id="unit-1",
                token_text="brave",
                normalized_text="brave",
                char_start=6,
                char_end=11,
                speaker_id="speaker-a",
                turn_id="turn-a",
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
            ),
            PreparedTokenUnit(
                unit_id="unit-2",
                token_text="world",
                normalized_text="world",
                char_start=12,
                char_end=17,
                speaker_id="speaker-a",
                turn_id="turn-a",
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
            ),
        ),
        window_text=SlowWindowTextPackage(
            text="hello brave world",
            display_text="hello brave world",
            source_language="en",
        ),
        punctuation_evidences=tuple(),
        fast_hooks=(
            FastHook(
                hook_text="hello",
                start=0.0,
                end=0.3,
                confidence=0.95,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
            ),
            FastHook(
                hook_text="brave",
                start=0.45,
                end=0.72,
                confidence=0.95,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
            ),
            FastHook(
                hook_text="world",
                start=0.9,
                end=1.2,
                confidence=0.95,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
            ),
        ),
        pronunciation_hints=tuple(),
        policy_snapshot=None,
        coverage=WindowCoverage(
            core_segments=((0.0, 1.2),),
            left_guard_sec=0.0,
            right_guard_sec=0.0,
            chunk_bindings=(
                WindowChunkBinding(
                    chunk_id="chunk-1",
                    chunk_index=1,
                    chunk_start=0.0,
                    chunk_end=1.2,
                    overlap_ratio=1.0,
                    role="owner",
                    is_owner=True,
                ),
            ),
        ),
    )


def _build_state() -> WindowAlignmentState:
    input_view = _build_input_view()
    solve_result = ChainSolveResult(
        committed_blocks=(
            LocalAlignmentBlock(
                block_id="block-left",
                unit_indices=(0,),
                hook_indices=(0,),
                score=1.0,
                block_kind="anchored",
                anchor_kind="exact",
                trust_tier="primary",
            ),
            LocalAlignmentBlock(
                block_id="block-right",
                unit_indices=(2,),
                hook_indices=(2,),
                score=1.0,
                block_kind="anchored",
                anchor_kind="exact",
                trust_tier="primary",
            ),
        ),
        unit_to_hook_indices=((0,), tuple(), (2,)),
        unit_anchor_kinds=("exact", "none", "exact"),
        unresolved_unit_indices=(1,),
        reseed_count=0,
    )
    return WindowAlignmentState.bootstrap(input_view).with_updates(
        local_blocks=solve_result.committed_blocks,
        main_chain=solve_result,
    )


def test_gap_rescue_aligner_builds_open_gap_and_collects_rescue_matches() -> None:
    aligner = GapRescueAligner()

    state = aligner.run(state=_build_state())

    assert len(state.anchor_islands) == 2
    assert len(state.open_gaps) == 1
    gap = state.open_gaps[0]
    assert gap.unit_indices == (1,)
    assert gap.hook_indices == (1,)
    assert len(gap.rescue_matches) == 1
    assert gap.rescue_matches[0].unit_indices == (1,)
    assert gap.rescue_matches[0].hook_indices == (1,)
    assert gap.rescue_matches[0].anchor_kind == "exact"
    assert state.rescue_round_index == 1
    assert state.diagnostics["open_gap_count"] == 1
    assert state.diagnostics["gap_rescue_match_count"] == 1
