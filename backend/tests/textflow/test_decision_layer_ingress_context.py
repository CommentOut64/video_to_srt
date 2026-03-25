from __future__ import annotations

from app.models.sensevoice_models import SentenceSegment, WordTimestamp
from app.services.alignment.types import DecisionLayerInput, DecisionLayerOutput
from app.services.punctuation.final_splitter import FinalSplitter
from app.services.textflow.contracts import (
    PunctuationFact as CanonicalPunctuationFact,
    SegmentPlan,
    SegmentationIngressContext,
    SegmentationResult,
)
from app.services.textflow.decision_layer import SegmentationProcessor
from app.services.timeanchored_alignment.contracts import BoundaryEvidence


def test_decision_layer_preserves_future_ready_ingress_context_in_canonical_stream() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    ingress_context = SegmentationIngressContext(
        unit_kind="slow_window",
        unit_id="window-7",
        slow_window_id="window-7",
        turn_group_id="turn-group-4",
        window_coverage=0.8,
        source_chunk_ids=("chunk-1", "chunk-2"),
        projection_chunk_ids=(),
        metadata={"trace_scope": "chunk6"},
    )
    decision_input = DecisionLayerInput(
        annotated_words=[],
        vad_intervals=[],
        ingress_context=ingress_context,
    )

    stream = processor._build_canonical_stream_from_words(
        words=[
            WordTimestamp(word="Hello", start=0.0, end=0.4),
            WordTimestamp(word="world", start=0.5, end=0.9),
        ],
        data=decision_input,
        stream_id="main",
        chunk_index=3,
    )

    assert stream.metadata["ingress_context"]["unit_kind"] == "slow_window"
    assert stream.metadata["ingress_context"]["slow_window_id"] == "window-7"
    assert stream.metadata["ingress_context"]["turn_group_id"] == "turn-group-4"
    assert stream.metadata["ingress_context"]["window_coverage"] == 0.8
    assert stream.metadata["ingress_context"]["source_chunk_ids"] == ["chunk-1", "chunk-2"]
    assert stream.metadata["ingress_context"]["projection_chunk_ids"] == []
    assert stream.diagnostics.ingress_context["metadata"]["trace_scope"] == "chunk6"


def test_decision_layer_applies_canonical_facts_and_boundaries_without_projection_alias() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    ingress_context = SegmentationIngressContext(
        unit_kind="slow_window",
        unit_id="window-9",
        slow_window_id="window-9",
        turn_group_id="turn-group-9",
        window_coverage=1.0,
        source_chunk_ids=("chunk-9", "chunk-10"),
        projection_chunk_ids=(),
        metadata={"trace_scope": "window9"},
    )
    canonical_fact = CanonicalPunctuationFact(
        fact_id="fact-end-1",
        left_token_index=1,
        right_token_index=None,
        attach_mode="trailing",
        raw_text="。",
        normalized_text="。",
        punct_class="sentence_end",
        source="aligned",
        priority=100,
        is_sentence_end=True,
        metadata={},
    )
    canonical_boundary = BoundaryEvidence(
        split_idx=0,
        event_time=0.4,
        left_end=0.38,
        right_start=0.41,
        reason="pause_long",
        score=0.92,
        hard_flag=False,
        metadata={},
    )
    decision_input = DecisionLayerInput(
        annotated_words=[],
        vad_intervals=[],
        ingress_context=ingress_context,
        canonical_punctuation_facts=(canonical_fact,),
        canonical_candidate_boundaries=(canonical_boundary,),
    )

    stream = processor._build_canonical_stream_from_words(
        words=[
            WordTimestamp(word="你好", start=0.0, end=0.38),
            WordTimestamp(word="世界", start=0.41, end=0.9),
        ],
        data=decision_input,
        stream_id="timeanchored:window-9",
        chunk_index=9,
    )

    assert stream.metadata["ingress_context"]["projection_chunk_ids"] == []
    assert [item.fact_id for item in stream.punctuation_facts] == ["fact-end-1"]
    assert stream.candidate_boundaries == (canonical_boundary,)
    assert stream.diagnostics.ingress_context["metadata"]["trace_scope"] == "window9"


def test_decision_layer_render_keeps_owner_carrier_subtitle_batch_for_slow_window_ingress() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    ingress_context = SegmentationIngressContext(
        unit_kind="slow_window",
        unit_id="window-12",
        chunk_id="chunk-owner-12",
        chunk_index=12,
        slow_window_id="window-12",
        turn_group_id="turn-group-12",
        window_coverage=1.0,
        source_chunk_ids=("chunk-owner-12", "chunk-13"),
        projection_chunk_ids=(),
        metadata={"trace_scope": "window12"},
    )
    decision_input = DecisionLayerInput(
        annotated_words=[],
        vad_intervals=[],
        ingress_context=ingress_context,
    )
    segmentation_output = DecisionLayerOutput(
        sentence_segments=[
            SentenceSegment(
                text="你好世界",
                text_clean="你好世界",
                start=0.0,
                end=1.0,
            )
        ],
        words_for_split=[
            WordTimestamp(word="你好", start=0.0, end=0.5, confidence=0.9, confidence_source="slow"),
            WordTimestamp(word="世界", start=0.5, end=1.0, confidence=0.8, confidence_source="slow"),
        ],
        segmentation_result=SegmentationResult(
            segments=(
                SegmentPlan(
                    segment_id="timeanchored:window-12:chunk-owner-12:seg:0",
                    token_start=0,
                    token_end=1,
                    start=0.0,
                    end=1.0,
                    boundary_reason="segmentation_core",
                    boundary_score=1.0,
                ),
            )
        ),
    )

    output = processor._render_with_unified_pipeline(
        segmentation_output=segmentation_output,
        data=decision_input,
        stream_id="timeanchored:window-12",
        chunk_index=12,
    )

    assert output.subtitle_batch is not None
    assert output.subtitle_batch.chunk_id == "chunk-owner-12"
    assert output.subtitle_batch.diagnostics["ingress_context"]["projection_chunk_ids"] == []
