from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorCandidate,
    AnchorMountInputView,
)
from app.services.timeanchored_alignment.anchor_mount.densification_promoter import (
    DensificationPromoter,
)
from app.services.timeanchored_alignment.anchor_mount.gap_rescue_aligner import (
    AnchorGap,
    AnchorIsland,
    GapRescueMatch,
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
        window_id="window-promoter-001",
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


def _build_state(*, primary_candidates: tuple[AnchorCandidate, ...]) -> WindowAlignmentState:
    input_view = _build_input_view()
    return WindowAlignmentState.bootstrap(input_view).with_updates(
        primary_candidates=primary_candidates,
        anchor_islands=(
            AnchorIsland(
                island_id="island-left",
                block_ids=("block-left",),
                unit_indices=(0,),
                hook_indices=(0,),
            ),
            AnchorIsland(
                island_id="island-right",
                block_ids=("block-right",),
                unit_indices=(2,),
                hook_indices=(2,),
            ),
        ),
        open_gaps=(
            AnchorGap(
                gap_id="gap-0",
                left_island_id="island-left",
                right_island_id="island-right",
                unit_indices=(1,),
                hook_indices=(1,),
                rescue_matches=(
                    GapRescueMatch(
                        unit_indices=(1,),
                        hook_indices=(1,),
                        anchor_kind="exact",
                        score=0.88,
                    ),
                ),
            ),
        ),
    )


def test_densification_promoter_promotes_unique_rescue_match_into_secondary_anchor() -> None:
    promoter = DensificationPromoter()
    state = _build_state(
        primary_candidates=(
            AnchorCandidate(
                candidate_id="exact:0:0",
                unit_indices=(0,),
                hook_indices=(0,),
                anchor_kind="exact",
                score=1.0,
                is_hard=True,
                trust_tier="primary",
            ),
            AnchorCandidate(
                candidate_id="exact:2:2",
                unit_indices=(2,),
                hook_indices=(2,),
                anchor_kind="exact",
                score=1.0,
                is_hard=True,
                trust_tier="primary",
            ),
        ),
    )

    updated = promoter.run(state=state)

    assert len(updated.promoted_secondary_anchors) == 1
    promoted = updated.promoted_secondary_anchors[0]
    assert promoted.unit_indices == (1,)
    assert promoted.hook_indices == (1,)
    assert promoted.trust_tier == "secondary"
    assert promoted.source_gap_id == "gap-0"
    assert updated.diagnostics["promoted_anchor_count"] == 1


def test_densification_promoter_does_not_override_existing_primary_anchor_span() -> None:
    promoter = DensificationPromoter()
    state = _build_state(
        primary_candidates=(
            AnchorCandidate(
                candidate_id="exact:1:1",
                unit_indices=(1,),
                hook_indices=(1,),
                anchor_kind="exact",
                score=1.0,
                is_hard=True,
                trust_tier="primary",
            ),
        ),
    )

    updated = promoter.run(state=state)

    assert updated.promoted_secondary_anchors == tuple()
    assert updated.diagnostics["promoted_anchor_count"] == 0
