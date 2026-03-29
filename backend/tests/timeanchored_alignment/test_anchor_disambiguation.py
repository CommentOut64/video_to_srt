from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.ambiguity_cluster_builder import (
    AmbiguityClusterBuilder,
)
from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorCandidate,
    AnchorMountInputView,
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
        window_id="window-cluster-001",
        owner_chunk_id="chunk-1",
        owner_chunk_index=1,
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        language="en",
        token_units=(
            PreparedTokenUnit(
                unit_id="unit-0",
                token_text="alpha",
                normalized_text="alpha",
                char_start=0,
                char_end=5,
                speaker_id="speaker-a",
                turn_id="turn-a",
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
            ),
            PreparedTokenUnit(
                unit_id="unit-1",
                token_text="beta",
                normalized_text="beta",
                char_start=6,
                char_end=10,
                speaker_id="speaker-a",
                turn_id="turn-a",
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
            ),
        ),
        window_text=SlowWindowTextPackage(
            text="alpha beta",
            display_text="alpha beta",
            source_language="en",
        ),
        punctuation_evidences=tuple(),
        fast_hooks=(
            FastHook(
                hook_text="alpha",
                start=0.0,
                end=0.2,
                confidence=0.95,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
            ),
            FastHook(
                hook_text="alphabeta",
                start=0.0,
                end=0.4,
                confidence=0.95,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
            ),
        ),
        pronunciation_hints=tuple(),
        policy_snapshot=None,
        coverage=WindowCoverage(
            core_segments=((0.0, 0.4),),
            left_guard_sec=0.0,
            right_guard_sec=0.0,
            chunk_bindings=(
                WindowChunkBinding(
                    chunk_id="chunk-1",
                    chunk_index=1,
                    chunk_start=0.0,
                    chunk_end=0.4,
                    overlap_ratio=1.0,
                    role="owner",
                    is_owner=True,
                ),
            ),
        ),
    )


def test_ambiguity_cluster_builder_groups_candidates_that_compete_for_same_units() -> None:
    input_view = _build_input_view()
    candidates = (
        AnchorCandidate(
            candidate_id="exact:0:0",
            unit_indices=(0,),
            hook_indices=(0,),
            anchor_kind="exact",
            score=1.0,
            is_hard=True,
        ),
        AnchorCandidate(
            candidate_id="merge:0-1:1",
            unit_indices=(0, 1),
            hook_indices=(1,),
            anchor_kind="merge",
            score=0.82,
            is_hard=True,
        ),
    )

    clusters = AmbiguityClusterBuilder().build(input_view=input_view, candidates=candidates)

    assert len(clusters) == 1
    assert set(clusters[0].candidate_ids) == {"exact:0:0", "merge:0-1:1"}
    assert "same_unit_span" in clusters[0].conflict_basis
    assert "merge_width_competition" in clusters[0].conflict_basis


def test_ambiguity_cluster_builder_ignores_isolated_candidates() -> None:
    input_view = _build_input_view()
    candidates = (
        AnchorCandidate(
            candidate_id="exact:0:0",
            unit_indices=(0,),
            hook_indices=(0,),
            anchor_kind="exact",
            score=1.0,
            is_hard=True,
        ),
    )

    clusters = AmbiguityClusterBuilder().build(input_view=input_view, candidates=candidates)

    assert clusters == tuple()
