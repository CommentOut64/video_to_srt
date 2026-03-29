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
from unittest.mock import Mock


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
        timeline_validity="valid",
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


def test_alignment_stage_service_injects_timeline_turns_from_anchored_token_window() -> None:
    coverage = WindowCoverage(
        core_segments=((10.0, 12.0),),
        left_guard_sec=0.0,
        right_guard_sec=0.0,
        chunk_bindings=(
            WindowChunkBinding(
                chunk_id="chunk-10",
                chunk_index=10,
                chunk_start=10.0,
                chunk_end=11.0,
                overlap_ratio=1.0,
                role="owner",
                is_owner=True,
            ),
            WindowChunkBinding(
                chunk_id="chunk-11",
                chunk_index=11,
                chunk_start=11.0,
                chunk_end=12.0,
                overlap_ratio=1.0,
                role="core",
                is_owner=False,
            ),
        ),
    )
    service = AlignmentStageService(
        host=type(
            "Host",
            (),
            {
                "logger": Mock(),
                "_postprocess_trace_enabled": False,
                "_postprocess_trace_level": "summary",
                "_anchor_mount_graph": "off",
                "_timeline_turns": [
                    type("Turn", (), {"turn_id": "turn-a", "speaker_id": "speaker-a", "start": 10.2, "end": 10.9, "boundary_confidence": 0.9})(),
                    type("Turn", (), {"turn_id": "turn-b", "speaker_id": "speaker-b", "start": 11.0, "end": 11.8, "boundary_confidence": 0.9})(),
                ],
            },
        )()
    )
    decision_input = DecisionLayerInput(annotated_words=[], vad_intervals=[])
    decision_ingress = DecisionIngressPackage(
        window_id="window-speaker-001",
        owner_chunk_id="chunk-10",
        owner_chunk_index=10,
        source_chunk_ids=("chunk-10", "chunk-11"),
        source_chunk_indices=(10, 11),
        language="en",
        policy_snapshot=None,
        anchored_token_units=(
            AnchoredTokenUnit(
                unit_id="unit-0",
                token_text="hello",
                normalized_text="hello",
                start=10.25,
                end=10.55,
                left_bound=10.2,
                right_bound=10.55,
                speaker_id=None,
                turn_id=None,
                mount_status="anchored",
                anchor_kind="exact",
                source_chunk_ids=("chunk-10",),
                source_chunk_indices=(10,),
                source_hook_ids=("hook-0",),
                match_confidence=1.0,
                cross_chunk_lock_ids=(),
            ),
            AnchoredTokenUnit(
                unit_id="unit-1",
                token_text="world",
                normalized_text="world",
                start=11.1,
                end=11.45,
                left_bound=11.0,
                right_bound=11.5,
                speaker_id=None,
                turn_id=None,
                mount_status="anchored",
                anchor_kind="exact",
                source_chunk_ids=("chunk-11",),
                source_chunk_indices=(11,),
                source_hook_ids=("hook-1",),
                match_confidence=1.0,
                cross_chunk_lock_ids=(),
            ),
        ),
        punctuation_facts=(),
        punctuation_pair_states=(),
        boundary_evidences=(),
        cross_chunk_locks=(),
        coverage=coverage,
        quality_metrics={"alignment_score": 1.0},
        timeline_validity="valid",
        should_fallback=False,
    )

    diagnostics = service._inject_timeline_turns_into_decision_input(
        decision_input=decision_input,
        decision_ingress=decision_ingress,
    )

    assert [item["turn_id"] for item in decision_input.aligned_facts.speaker_turns] == ["turn-a", "turn-b"]
    assert diagnostics["window_span_source"] == "anchored_token_units"
    assert diagnostics["selected_turn_count"] == 2
    assert diagnostics["window_spans"] == [[10.25, 11.45]]
