from __future__ import annotations

from app.services.alignment.types import DecisionLayerInput
from app.services.textflow.alignment_path_adapter import AlignmentPathAdapterResult
from app.services.textflow.contracts import SegmentationIngressContext
from app.services.timeanchored_alignment.contracts import (
    AlignedToken,
    AlignmentPath,
    BoundaryCandidate,
    LayerSummary,
    LowConfidenceSpan,
)
from app.pipelines.dual_pipeline.services.alignment_stage_service import AlignmentStageService


def test_alignment_stage_service_builds_detailed_segmentation_ingress_trace_summary() -> None:
    alignment_path = AlignmentPath(
        path_id="path-1",
        aligned_tokens=(
            AlignedToken(
                token_id="unit-0",
                text="beneficial",
                start=21.08,
                end=21.62,
                source_chunk_ids=("chunk-1", "chunk-2"),
                confidence=1.0,
                trace={"token_index": 0},
            ),
            AlignedToken(
                token_id="unit-1",
                text="for",
                start=21.70,
                end=21.82,
                source_chunk_ids=("chunk-2",),
                confidence=1.0,
                trace={"token_index": 1},
            ),
        ),
        route_confidence=0.93,
        low_confidence_spans=(
            LowConfidenceSpan(
                start_token_index=1,
                end_token_index=1,
                reason="low_confidence",
                score=0.4,
            ),
        ),
        summary=LayerSummary(layer="alignment", status="warning"),
        source_chunk_ids=("chunk-1", "chunk-2"),
        boundary_candidates=(
            BoundaryCandidate(
                split_token_index=0,
                event_time=21.66,
                reason="gap_pause",
                score=0.68,
                hard_boundary=False,
                source_chunk_ids=("chunk-1", "chunk-2"),
                metadata={"gap": 0.08},
            ),
        ),
    )

    adapter_result = AlignmentPathAdapterResult(
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
        compat_report={"alignment_boundary_count": 1},
    )

    summary = AlignmentStageService._build_segmentation_ingress_trace_summary(
        alignment_path=alignment_path,
        adapter_result=adapter_result,
    )

    assert summary["aligned_tokens"][0]["token_id"] == "unit-0"
    assert summary["aligned_tokens"][0]["start"] == 21.08
    assert summary["aligned_tokens"][0]["source_chunk_ids"] == ["chunk-1", "chunk-2"]
    assert summary["boundary_candidates"][0]["reason"] == "gap_pause"
    assert summary["boundary_by_split"][0]["split_idx"] == 0
    assert summary["boundary_by_split"][0]["left_text"] == "beneficial"
    assert summary["boundary_by_split"][0]["right_text"] == "for"
    assert summary["compat_report"]["alignment_boundary_count"] == 1
