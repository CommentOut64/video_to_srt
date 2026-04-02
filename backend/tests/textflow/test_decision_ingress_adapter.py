from __future__ import annotations

from dataclasses import replace

from app.services.textflow.decision_ingress_adapter import DecisionIngressAdapter
from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchoredTokenUnit,
    CrossChunkLock,
    DecisionIngressPackage,
    PunctuationFact,
    PunctuationPairState,
)
from app.services.timeanchored_alignment.contracts import BoundaryEvidence
from app.services.timeanchored_alignment.slow_window.contracts import (
    WindowChunkBinding,
    WindowCoverage,
)


def _build_package() -> DecisionIngressPackage:
    return DecisionIngressPackage(
        window_id="window-11",
        owner_chunk_id="chunk-11",
        owner_chunk_index=11,
        source_chunk_ids=("chunk-11", "chunk-12"),
        source_chunk_indices=(11, 12),
        language="zh",
        policy_snapshot=None,
        anchored_token_units=(
            AnchoredTokenUnit(
                unit_id="token-1",
                token_text="hello",
                normalized_text="hello",
                start=0.0,
                end=0.5,
                left_bound=0.0,
                right_bound=0.55,
                speaker_id="spk-1",
                turn_id="turn-1",
                mount_status="anchored",
                anchor_kind="lexical",
                source_chunk_ids=("chunk-11",),
                source_chunk_indices=(11,),
                source_hook_ids=("hook-1",),
                match_confidence=0.91,
                cross_chunk_lock_ids=("lock-1",),
            ),
            AnchoredTokenUnit(
                unit_id="token-2",
                token_text="world",
                normalized_text="world",
                start=0.55,
                end=1.0,
                left_bound=0.5,
                right_bound=1.0,
                speaker_id="spk-1",
                turn_id="turn-1",
                mount_status="merged",
                anchor_kind="soft_pronunciation",
                source_chunk_ids=("chunk-12",),
                source_chunk_indices=(12,),
                source_hook_ids=("hook-2",),
                match_confidence=0.93,
                cross_chunk_lock_ids=("lock-1",),
            ),
        ),
        punctuation_facts=(
            PunctuationFact(
                fact_id="fact-1",
                left_token_index=1,
                right_token_index=None,
                attach_mode="trailing",
                normalized_text=".",
                punct_class="sentence_end",
                confidence=0.87,
                source="aligned",
                group_id="grp-1",
                boundary_weight=0.8,
                render_default=True,
                metadata={},
            ),
        ),
        punctuation_pair_states=(
            PunctuationPairState(
                group_id="grp-quote",
                pair_kind="quote",
                open_fact_id="open-1",
                close_fact_id=None,
                state="open",
                metadata={},
            ),
        ),
        boundary_evidences=(
            BoundaryEvidence(
                split_idx=0,
                event_time=0.52,
                left_end=0.5,
                right_start=0.55,
                reason="lexical_boundary",
                score=0.9,
                hard_flag=False,
                metadata={
                    "blocked_by_lock": False,
                    "left_mount_status": "anchored",
                    "right_mount_status": "merged",
                },
            ),
        ),
        cross_chunk_locks=(
            CrossChunkLock(
                lock_id="lock-1",
                unit_ids=("token-1", "token-2"),
                hook_ids=("hook-1",),
                reason="quote_scope",
                source_chunk_ids=("chunk-11", "chunk-12"),
                source_chunk_indices=(11, 12),
            ),
        ),
        coverage=WindowCoverage(
            core_segments=((0.0, 1.0),),
            left_guard_sec=0.1,
            right_guard_sec=0.1,
            chunk_bindings=(
                WindowChunkBinding(
                    chunk_id="chunk-11",
                    chunk_index=11,
                    chunk_start=0.0,
                    chunk_end=0.7,
                    overlap_ratio=0.7,
                    role="owner",
                    is_owner=True,
                ),
                WindowChunkBinding(
                    chunk_id="chunk-12",
                    chunk_index=12,
                    chunk_start=0.55,
                    chunk_end=1.2,
                    overlap_ratio=0.45,
                    role="core",
                    is_owner=False,
                ),
            ),
        ),
        quality_metrics={"alignment_score": 0.92, "soft_anchor_ratio": 0.88},
        timeline_validity="repairable",
        should_fallback=True,
    )


def test_decision_ingress_adapter_keeps_compat_fields_but_cleans_projection_boundary() -> None:
    result = DecisionIngressAdapter().build(package=_build_package())

    ingress_context = result.ingress_context
    assert isinstance(ingress_context.source_chunk_ids, tuple)
    assert ingress_context.source_chunk_ids == ("chunk-11", "chunk-12")
    assert ingress_context.projection_chunk_ids == ()

    metadata = ingress_context.metadata
    assert metadata["source_chunk_indices"] == [11, 12]
    assert metadata["quality_metrics"] == {"alignment_score": 0.92, "soft_anchor_ratio": 0.88}
    assert metadata["timeline_validity"] == "repairable"
    assert metadata["should_fallback"] is True
    assert metadata["cross_chunk_locks"][0]["lock_id"] == "lock-1"
    assert metadata["decision_ingress_version"] == "window_first_compat_v1"
    assert metadata["fallback_projection_mode"] == "layer_internal_compat"
    assert metadata["candidate_boundary_count"] == 1

    decision_input = result.decision_input
    assert decision_input.cut_plan is None
    assert decision_input.allow_fast_draft_fallback is False
    assert decision_input.fallback_clean_text_ref
    assert len(decision_input.fallback_punctuation_positions) == 1
    assert decision_input.aligned_facts is not None
    assert decision_input.aligned_facts.fast_draft_cuts == []

    assert result.compat_report["fallback_projection_mode"] == "layer_internal_compat"
    assert result.compat_report["timeline_validity"] == "repairable"
    assert result.compat_report["canonical_boundary_count"] == 1


def test_decision_ingress_adapter_forwards_alignment_boundary_evidences_into_canonical_input() -> None:
    result = DecisionIngressAdapter().build(package=_build_package())

    assert result.decision_input.cut_plan is None
    assert len(result.decision_input.canonical_candidate_boundaries) == 1
    assert result.decision_input.canonical_candidate_boundaries[0].reason == "lexical_boundary"
    assert result.compat_report["canonical_boundary_count"] == 1
    assert result.ingress_context.metadata["candidate_boundary_count"] == 1


def test_decision_ingress_adapter_drops_blocked_boundary_evidence_from_projection_metrics() -> None:
    package = _build_package()
    blocked = replace(
        package.boundary_evidences[0],
        metadata={
            **dict(package.boundary_evidences[0].metadata),
            "blocked_by_lock": True,
        },
    )
    package = replace(package, boundary_evidences=(blocked,))

    result = DecisionIngressAdapter().build(package=package)
    assert result.decision_input.cut_plan is None
    assert result.decision_input.canonical_candidate_boundaries == ()
    assert result.compat_report["canonical_boundary_count"] == 0


def test_decision_ingress_adapter_keeps_alignment_boundaries_single_sourced() -> None:
    result = DecisionIngressAdapter().build(package=_build_package())

    reasons = {
        item.reason for item in result.decision_input.canonical_candidate_boundaries
    }
    assert reasons <= {"lexical_boundary", "anchor_block_close"}
    assert result.decision_input.fused_evidence is not None
    assert result.decision_input.fused_evidence.speaker_changes == []
    assert result.decision_input.fused_evidence.pause_anchors == []


def test_decision_ingress_adapter_keeps_split_token_parts_free_of_legacy_position_metadata() -> None:
    package = _build_package()
    first = replace(
        package.anchored_token_units[0],
        token_text="what do",
        normalized_text="what do",
    )
    package = replace(package, anchored_token_units=(first, package.anchored_token_units[1]))

    result = DecisionIngressAdapter().build(package=package)
    annotated_words = list(result.decision_input.annotated_words)
    assert [item.word for item in annotated_words] == ["what", "do", "world"]
    for item in annotated_words:
        assert set(vars(item).keys()) == {
            "word",
            "start",
            "end",
            "trailing_punct",
            "confidence",
            "confidence_source",
            "is_pseudo",
            "speaker_id",
            "turn_id",
            "track_id",
        }


def test_decision_ingress_adapter_can_build_span_scoped_input() -> None:
    package = DecisionIngressPackage(
        window_id="window-12",
        owner_chunk_id="chunk-12",
        owner_chunk_index=12,
        source_chunk_ids=("chunk-12",),
        source_chunk_indices=(12,),
        language="en",
        policy_snapshot=None,
        anchored_token_units=(
            AnchoredTokenUnit(
                unit_id="token-0",
                token_text="hello",
                normalized_text="hello",
                start=0.0,
                end=0.2,
                left_bound=0.0,
                right_bound=0.2,
                speaker_id="spk-1",
                turn_id="turn-1",
                mount_status="anchored",
                anchor_kind="lexical",
                source_chunk_ids=("chunk-12",),
                source_chunk_indices=(12,),
                source_hook_ids=("hook-0",),
                match_confidence=0.95,
                cross_chunk_lock_ids=tuple(),
                token_index=0,
                char_start=0,
                char_end=5,
            ),
            AnchoredTokenUnit(
                unit_id="token-1",
                token_text="brave",
                normalized_text="brave",
                start=0.2,
                end=0.4,
                left_bound=0.2,
                right_bound=0.4,
                speaker_id="spk-1",
                turn_id="turn-1",
                mount_status="anchored",
                anchor_kind="lexical",
                source_chunk_ids=("chunk-12",),
                source_chunk_indices=(12,),
                source_hook_ids=("hook-1",),
                match_confidence=0.95,
                cross_chunk_lock_ids=tuple(),
                token_index=1,
                char_start=6,
                char_end=11,
            ),
            AnchoredTokenUnit(
                unit_id="token-2",
                token_text="world",
                normalized_text="world",
                start=0.4,
                end=0.6,
                left_bound=0.4,
                right_bound=0.6,
                speaker_id="spk-1",
                turn_id="turn-1",
                mount_status="anchored",
                anchor_kind="lexical",
                source_chunk_ids=("chunk-12",),
                source_chunk_indices=(12,),
                source_hook_ids=("hook-2",),
                match_confidence=0.95,
                cross_chunk_lock_ids=tuple(),
                token_index=2,
                char_start=12,
                char_end=17,
            ),
        ),
        punctuation_facts=tuple(),
        punctuation_pair_states=tuple(),
        boundary_evidences=(
            BoundaryEvidence(
                split_idx=0,
                event_time=0.19,
                left_end=0.2,
                right_start=0.2,
                reason="lexical_boundary",
                score=0.9,
                hard_flag=False,
                metadata={},
            ),
            BoundaryEvidence(
                split_idx=1,
                event_time=0.39,
                left_end=0.4,
                right_start=0.4,
                reason="lexical_boundary",
                score=0.9,
                hard_flag=False,
                metadata={},
            ),
        ),
        cross_chunk_locks=tuple(),
        coverage=WindowCoverage(
            core_segments=((0.0, 0.6),),
            left_guard_sec=0.0,
            right_guard_sec=0.0,
            chunk_bindings=(
                WindowChunkBinding(
                    chunk_id="chunk-12",
                    chunk_index=12,
                    chunk_start=0.0,
                    chunk_end=0.6,
                    overlap_ratio=1.0,
                    role="owner",
                    is_owner=True,
                ),
            ),
        ),
        quality_metrics={"alignment_score": 0.95},
    )

    result = DecisionIngressAdapter().build(package=package, span_selector=(1, 3))

    assert [item.word for item in result.decision_input.annotated_words] == ["brave", "world"]
    assert len(result.decision_input.canonical_candidate_boundaries) == 1
    assert result.decision_input.canonical_candidate_boundaries[0].split_idx == 0


def test_decision_ingress_adapter_reports_span_selector_in_compat_report() -> None:
    result = DecisionIngressAdapter().build(package=_build_package(), span_selector=(0, 1))

    assert result.compat_report["span_selector"] == [0, 1]
