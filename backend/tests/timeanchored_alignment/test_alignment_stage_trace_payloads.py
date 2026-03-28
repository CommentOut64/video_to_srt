from __future__ import annotations

from app.services.alignment.types import DecisionLayerInput
from app.services.textflow.decision_ingress_adapter import DecisionIngressAdapterResult
from app.services.textflow.contracts import SegmentationIngressContext
from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchoredTokenUnit,
    CrossChunkLock,
    DecisionIngressPackage,
)
from app.services.timeanchored_alignment.contracts import BoundaryEvidence
from app.services.timeanchored_alignment.slow_window.contracts import (
    WindowChunkBinding,
    WindowCoverage,
)
from app.pipelines.dual_pipeline.services.alignment_stage_service import AlignmentStageService


def test_alignment_stage_service_builds_detailed_decision_ingress_trace_summary() -> None:
    coverage = WindowCoverage(
        core_segments=((0.0, 1.0),),
        left_guard_sec=0.0,
        right_guard_sec=0.0,
        chunk_bindings=(
            WindowChunkBinding(
                chunk_id="chunk-1",
                chunk_index=1,
                chunk_start=0.0,
                chunk_end=1.0,
                overlap_ratio=1.0,
                role="owner",
                is_owner=True,
            ),
        ),
    )
    package = DecisionIngressPackage(
        window_id="sw-000001",
        owner_chunk_id="chunk-1",
        owner_chunk_index=1,
        source_chunk_ids=("chunk-1", "chunk-2"),
        source_chunk_indices=(1, 2),
        language="en",
        policy_snapshot=None,
        anchored_token_units=(
            AnchoredTokenUnit(
                unit_id="unit-0",
                token_text="beneficial",
                normalized_text="beneficial",
                start=21.08,
                end=21.62,
                left_bound=21.08,
                right_bound=21.62,
                speaker_id="speaker-a",
                turn_id="turn-a",
                mount_status="anchored",
                anchor_kind="lexical",
                source_chunk_ids=("chunk-1", "chunk-2"),
                source_chunk_indices=(1, 2),
                source_hook_ids=("hook-36",),
                match_confidence=1.0,
                cross_chunk_lock_ids=(),
            ),
            AnchoredTokenUnit(
                unit_id="unit-1",
                token_text="for",
                normalized_text="for",
                start=21.70,
                end=21.82,
                left_bound=21.70,
                right_bound=21.82,
                speaker_id="speaker-a",
                turn_id="turn-a",
                mount_status="anchored",
                anchor_kind="lexical",
                source_chunk_ids=("chunk-2",),
                source_chunk_indices=(2,),
                source_hook_ids=("hook-37",),
                match_confidence=1.0,
                cross_chunk_lock_ids=("lock-1",),
            ),
        ),
        punctuation_facts=(),
        punctuation_pair_states=(),
        boundary_evidences=(
            BoundaryEvidence(
                split_idx=0,
                event_time=21.66,
                left_end=21.62,
                right_start=21.70,
                reason="gap_pause",
                score=0.68,
                hard_flag=False,
                metadata={"blocked_by_lock": False},
            ),
        ),
        cross_chunk_locks=(
            CrossChunkLock(
                lock_id="lock-1",
                unit_ids=("unit-0", "unit-1"),
                hook_ids=("hook-36", "hook-37"),
                reason="cross_chunk_phrase",
                source_chunk_ids=("chunk-1", "chunk-2"),
                source_chunk_indices=(1, 2),
            ),
        ),
        coverage=coverage,
        quality_metrics={"alignment_score": 1.0},
        should_fallback=False,
    )
    adapter_result = DecisionIngressAdapterResult(
        decision_input=DecisionLayerInput(
            annotated_words=[],
            vad_intervals=[],
        ),
        stream_id="timeanchored:sw-000001",
        chunk_index=1,
        ingress_context=SegmentationIngressContext(
            unit_kind="slow_window",
            unit_id="sw-000001",
            chunk_id="chunk-1",
            chunk_index=1,
        ),
        compat_report={"canonical_boundary_count": 1},
    )

    summary = AlignmentStageService._build_decision_ingress_trace_summary(
        package=package,
        adapter_result=adapter_result,
    )

    assert summary["token_units"][0]["unit_id"] == "unit-0"
    assert summary["token_units"][0]["start"] == 21.08
    assert summary["token_units"][0]["source_chunk_ids"] == ["chunk-1", "chunk-2"]
    assert summary["boundary_evidences"][0]["reason"] == "gap_pause"
    assert summary["boundary_by_split"][0]["split_idx"] == 0
    assert summary["boundary_by_split"][0]["left_text"] == "beneficial"
    assert summary["boundary_by_split"][0]["right_text"] == "for"
    assert summary["cross_chunk_locks"][0]["lock_id"] == "lock-1"
