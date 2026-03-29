from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.contracts import AnchorMountInputView
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
        window_id="window-state-001",
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
                speaker_id=None,
                turn_id=None,
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
            ),
        ),
        window_text=SlowWindowTextPackage(
            text="hello",
            display_text="hello",
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
        ),
        pronunciation_hints=tuple(),
        policy_snapshot=None,
        coverage=WindowCoverage(
            core_segments=((0.0, 0.3),),
            left_guard_sec=0.0,
            right_guard_sec=0.0,
            chunk_bindings=(
                WindowChunkBinding(
                    chunk_id="chunk-1",
                    chunk_index=1,
                    chunk_start=0.0,
                    chunk_end=0.3,
                    overlap_ratio=1.0,
                    role="owner",
                    is_owner=True,
                ),
            ),
        ),
    )


def test_window_alignment_state_bootstrap_builds_minimal_immutable_state() -> None:
    state = WindowAlignmentState.bootstrap(_build_input_view())

    assert state.timeline_validity == "repairable"
    assert state.raw_seed_candidates == tuple()
    assert state.promoted_secondary_anchors == tuple()
    assert state.diagnostics["window_id"] == "window-state-001"


def test_window_alignment_state_with_updates_returns_new_state_without_mutating_original() -> None:
    state = WindowAlignmentState.bootstrap(_build_input_view())

    updated = state.with_updates(
        rescue_round_index=1,
        timeline_validity="valid",
        diagnostics={**state.diagnostics, "seed_candidate_count": 2},
    )

    assert state.rescue_round_index == 0
    assert updated.rescue_round_index == 1
    assert updated.timeline_validity == "valid"
    assert updated.diagnostics["seed_candidate_count"] == 2
