from __future__ import annotations

from app.models.sensevoice_models import SentenceSegment, WordTimestamp
from app.services.alignment.types import AnnotatedWord, DecisionLayerInput, DecisionLayerOutput
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


def test_decision_layer_render_emits_window_carrier_batch_and_aligned_sentence_contract() -> None:
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
    assert output.subtitle_batch.chunk_id == "window-12"
    assert output.subtitle_batch.diagnostics["ingress_context"]["projection_chunk_ids"] == []
    assert output.aligned_sentences[0].source_chunk_ids == ("chunk-owner-12", "chunk-13")
    assert output.segmentation_report_contract.summary.layer == "segmentation"
    assert output.segmentation_report_contract.sentence_count == 1


def test_decision_layer_slow_window_canonical_stream_keeps_protected_decimal_span() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    ingress_context = SegmentationIngressContext(
        unit_kind="slow_window",
        unit_id="window-protected-1",
        chunk_id="window-protected-1",
        slow_window_id="window-protected-1",
        source_chunk_ids=("chunk-21", "chunk-22"),
        projection_chunk_ids=(),
    )
    decision_input = DecisionLayerInput(
        annotated_words=[],
        vad_intervals=[],
        ingress_context=ingress_context,
    )

    stream = processor._build_canonical_stream_from_words(
        words=[
            WordTimestamp(word="版本", start=0.0, end=0.4, confidence=0.9, confidence_source="slow"),
            WordTimestamp(word="3.14", start=0.4, end=0.8, confidence=0.9, confidence_source="slow"),
            WordTimestamp(word="发布", start=0.8, end=1.2, confidence=0.9, confidence_source="slow"),
        ],
        data=decision_input,
        stream_id="timeanchored:window-protected-1",
        chunk_index=21,
    )

    assert stream.chunk_ref == "window-protected-1"
    assert [span.text for span in stream.protected_spans] == ["3.14"]


def test_decision_layer_builds_scored_cut_plan_from_timeanchored_candidate_boundaries() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    ingress_context = SegmentationIngressContext(
        unit_kind="slow_window",
        unit_id="window-42",
        chunk_id="chunk-42",
        chunk_index=42,
        slow_window_id="window-42",
        source_chunk_ids=("chunk-42",),
        projection_chunk_ids=(),
    )
    decision_input = DecisionLayerInput(
        annotated_words=[
            AnnotatedWord(word="alpha", start=0.0, end=0.5, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="beta", start=0.5, end=1.0, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="gamma", start=1.0, end=1.5, confidence=0.9, confidence_source="aligned"),
        ],
        vad_intervals=[],
        ingress_context=ingress_context,
        canonical_candidate_boundaries=(
            BoundaryEvidence(
                split_idx=0,
                event_time=0.5,
                left_end=0.5,
                right_start=0.5,
                reason="anchor_block_close",
                score=0.52,
                hard_flag=False,
                metadata={},
            ),
            BoundaryEvidence(
                split_idx=0,
                event_time=0.5,
                left_end=0.5,
                right_start=0.5,
                reason="punctuation_sentence_end",
                score=1.0,
                hard_flag=True,
                metadata={},
            ),
            BoundaryEvidence(
                split_idx=1,
                event_time=1.0,
                left_end=1.0,
                right_start=1.0,
                reason="punctuation_soft",
                score=0.6,
                hard_flag=False,
                metadata={},
            ),
        ),
    )

    output = processor._run_segmentation_core(
        decision_input,
        stream_id="timeanchored:window-42",
        chunk_index=42,
        is_last_chunk=True,
    )

    assert output.applied_cut_plan is not None
    assert output.applied_cut_plan.generation_report["generated_by"] == "timeanchored_segment_planner"
    assert output.applied_cut_plan.generation_report["input_boundary_count"] == 3
    assert output.applied_cut_plan.generation_report["decision_count"] == 1
    assert output.segmentation_report["soft_cut_stats"]["decision_count"] == 1
    assert output.segmentation_report["soft_cut_stats"]["fallback_reason"] == ""


def test_decision_layer_rejects_low_quality_mount_boundary_in_timeanchored_ingress() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    ingress_context = SegmentationIngressContext(
        unit_kind="slow_window",
        unit_id="window-43",
        chunk_id="chunk-43",
        chunk_index=43,
        slow_window_id="window-43",
        source_chunk_ids=("chunk-43",),
        projection_chunk_ids=(),
    )
    decision_input = DecisionLayerInput(
        annotated_words=[
            AnnotatedWord(word="What do yo", start=0.0, end=0.8, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="u do", start=0.8, end=1.6, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="next", start=1.6, end=2.2, confidence=0.9, confidence_source="aligned"),
        ],
        vad_intervals=[],
        ingress_context=ingress_context,
        canonical_candidate_boundaries=(
            BoundaryEvidence(
                split_idx=0,
                event_time=0.8,
                left_end=0.8,
                right_start=0.8,
                reason="speaker_change",
                score=0.95,
                hard_flag=True,
                metadata={
                    "left_mount_status": "inferred",
                    "right_mount_status": "inferred",
                },
            ),
        ),
    )

    output = processor._run_segmentation_core(
        decision_input,
        stream_id="timeanchored:window-43",
        chunk_index=43,
        is_last_chunk=True,
    )

    assert output.applied_cut_plan is not None
    assert output.applied_cut_plan.generation_report["decision_count"] == 0
    assert output.applied_cut_plan.generation_report["fallback_reason"] == "no_boundary_passed_scoring"


def test_decision_layer_rejects_alpha_fragment_boundary_in_timeanchored_ingress() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    ingress_context = SegmentationIngressContext(
        unit_kind="slow_window",
        unit_id="window-44",
        chunk_id="chunk-44",
        chunk_index=44,
        slow_window_id="window-44",
        source_chunk_ids=("chunk-44",),
        projection_chunk_ids=(),
    )
    decision_input = DecisionLayerInput(
        annotated_words=[
            AnnotatedWord(word="bo", start=0.0, end=0.5, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="x with", start=0.5, end=1.1, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="context", start=1.1, end=1.8, confidence=0.9, confidence_source="aligned"),
        ],
        vad_intervals=[],
        ingress_context=ingress_context,
        canonical_candidate_boundaries=(
            BoundaryEvidence(
                split_idx=0,
                event_time=0.5,
                left_end=0.5,
                right_start=0.5,
                reason="gap_pause",
                score=0.92,
                hard_flag=False,
                metadata={
                    "left_mount_status": "merged",
                    "right_mount_status": "merged",
                },
            ),
        ),
    )

    output = processor._run_segmentation_core(
        decision_input,
        stream_id="timeanchored:window-44",
        chunk_index=44,
        is_last_chunk=True,
    )

    assert output.applied_cut_plan is not None
    assert output.applied_cut_plan.generation_report["decision_count"] == 0
    assert output.applied_cut_plan.generation_report["fallback_reason"] == "no_boundary_passed_scoring"


def test_decision_layer_builds_cut_plan_without_ingress_candidate_boundaries() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    ingress_context = SegmentationIngressContext(
        unit_kind="slow_window",
        unit_id="window-45",
        chunk_id="chunk-45",
        chunk_index=45,
        slow_window_id="window-45",
        source_chunk_ids=("chunk-45",),
        projection_chunk_ids=(),
    )
    canonical_fact = CanonicalPunctuationFact(
        fact_id="fact-end-45",
        left_token_index=1,
        right_token_index=None,
        attach_mode="trailing",
        raw_text=".",
        normalized_text=".",
        punct_class="sentence_end",
        source="aligned",
        priority=100,
        is_sentence_end=True,
        metadata={},
    )
    decision_input = DecisionLayerInput(
        annotated_words=[
            AnnotatedWord(word="alpha", start=0.0, end=0.6, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="beta", start=0.6, end=1.2, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="gamma", start=1.2, end=1.8, confidence=0.9, confidence_source="aligned"),
        ],
        vad_intervals=[],
        ingress_context=ingress_context,
        canonical_candidate_boundaries=tuple(),
        canonical_punctuation_facts=(canonical_fact,),
    )

    output = processor._run_segmentation_core(
        decision_input,
        stream_id="timeanchored:window-45",
        chunk_index=45,
        is_last_chunk=True,
    )

    assert output.applied_cut_plan is not None
    assert output.applied_cut_plan.generation_report["generated_by"] == "timeanchored_segment_planner"
    assert output.applied_cut_plan.generation_report["decision_count"] >= 1
    assert output.segmentation_report["soft_cut_stats"]["source_stats"]["timeanchored_segment_planner"] >= 1


def test_decision_layer_treats_hard_limit_as_rolling_constraint_not_static_candidate() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    ingress_context = SegmentationIngressContext(
        unit_kind="slow_window",
        unit_id="window-qvvs-hard-limit",
        chunk_id="chunk-qvvs-hard-limit",
        chunk_index=46,
        slow_window_id="window-qvvs-hard-limit",
        source_chunk_ids=("chunk-qvvs-hard-limit",),
        projection_chunk_ids=(),
    )
    decision_input = DecisionLayerInput(
        annotated_words=[
            AnnotatedWord(word="Oh", start=16.04, end=16.10, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="God,", start=16.52, end=16.94, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="killinging", start=18.20, end=18.68, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="five", start=18.74, end=19.52, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="strangers", start=19.94, end=20.24, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="is", start=20.30, end=20.44, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="probably", start=20.46, end=20.78, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="more", start=20.82, end=21.04, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="beneficial", start=21.08, end=21.62, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="for", start=21.70, end=21.82, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="society,", start=21.84, end=22.30, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="so", start=22.36, end=22.50, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="I'll", start=22.52, end=22.72, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="do", start=22.74, end=22.88, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="that", start=22.90, end=23.30, confidence=0.9, confidence_source="aligned"),
        ],
        vad_intervals=[],
        ingress_context=ingress_context,
    )

    output = processor.process(
        decision_input,
        stream_id="timeanchored:window-qvvs-hard-limit",
        chunk_index=46,
        is_last_chunk=True,
    )

    assert output.applied_cut_plan is not None
    assert output.applied_cut_plan.generation_report["generated_by"] == "timeanchored_segment_planner"
    assert output.applied_cut_plan.generation_report["reason_stats"].get("hard_limit_forced", 0) == 0
    assert "killinging five" not in [segment.text for segment in output.sentence_segments]


def test_decision_layer_exposes_planner_candidate_diagnostics_in_soft_cut_stats() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    ingress_context = SegmentationIngressContext(
        unit_kind="slow_window",
        unit_id="window-trace-47",
        chunk_id="chunk-trace-47",
        chunk_index=47,
        slow_window_id="window-trace-47",
        source_chunk_ids=("chunk-trace-47",),
        projection_chunk_ids=(),
    )
    decision_input = DecisionLayerInput(
        annotated_words=[
            AnnotatedWord(word="Bennet,", start=30.84, end=31.20, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="okay,", start=31.22, end=31.60, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="sure", start=31.64, end=31.98, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="towards", start=33.40, end=33.96, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="five", start=34.26, end=34.60, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="people,", start=34.62, end=35.10, confidence=0.9, confidence_source="aligned"),
        ],
        vad_intervals=[],
        ingress_context=ingress_context,
        canonical_candidate_boundaries=(
            BoundaryEvidence(
                split_idx=3,
                event_time=34.11,
                left_end=33.96,
                right_start=34.26,
                reason="gap_pause",
                score=0.70,
                hard_flag=False,
                metadata={},
            ),
        ),
    )

    output = processor.process(
        decision_input,
        stream_id="timeanchored:window-trace-47",
        chunk_index=47,
        is_last_chunk=True,
    )

    planner_diagnostics = output.segmentation_report["soft_cut_stats"]["planner_diagnostics"]
    assert planner_diagnostics["feature_stats"]["alignment_feature_count"] >= 1
    assert planner_diagnostics["candidate_diagnostics"]
    assert planner_diagnostics["candidate_diagnostics"][0]["left_text"] == "Bennet,"
    assert planner_diagnostics["candidate_diagnostics"][-1]["right_text"] == "people,"
    assert any(
        "gap_pause" in item["reasons"]
        for item in planner_diagnostics["candidate_diagnostics"]
    )
    assert any(
        item["split_idx"] == 3 and item["selection_status"] in {"selected", "best_rejected", "candidate"}
        for item in planner_diagnostics["candidate_diagnostics"]
    )


def test_timeanchored_planner_reports_pause_cut_suppressed_by_tail_hold_on_near_tie() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    ingress_context = SegmentationIngressContext(
        unit_kind="slow_window",
        unit_id="window-tail-hold-48",
        chunk_id="chunk-tail-hold-48",
        chunk_index=48,
        slow_window_id="window-tail-hold-48",
        source_chunk_ids=("chunk-tail-hold-48",),
        projection_chunk_ids=(),
    )
    decision_input = DecisionLayerInput(
        annotated_words=[
            AnnotatedWord(word="Probably", start=21.348, end=21.708, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="Monica", start=21.888, end=22.428, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="she", start=22.548, end=22.608, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="has", start=22.848, end=22.908, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="the", start=23.088, end=23.148, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="right", start=23.268, end=23.328, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="idea", start=23.448, end=23.508, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="in", start=23.928, end=23.988, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="a", start=24.108, end=24.168, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="lot", start=24.228, end=24.288, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="of", start=24.408, end=24.468, confidence=0.9, confidence_source="aligned"),
            AnnotatedWord(word="ways", start=24.588, end=25.428, confidence=0.9, confidence_source="aligned"),
        ],
        vad_intervals=[],
        ingress_context=ingress_context,
        canonical_candidate_boundaries=(
            BoundaryEvidence(
                split_idx=4,
                event_time=23.208,
                left_end=23.148,
                right_start=23.268,
                reason="gap_pause",
                score=0.24,
                hard_flag=False,
                metadata={"gap_sec": 0.12},
            ),
            BoundaryEvidence(
                split_idx=5,
                event_time=23.388,
                left_end=23.328,
                right_start=23.448,
                reason="gap_pause",
                score=0.24,
                hard_flag=False,
                metadata={"gap_sec": 0.12},
            ),
        ),
    )

    output = processor.process(
        decision_input,
        stream_id="timeanchored:window-tail-hold-48",
        chunk_index=48,
        is_last_chunk=True,
    )

    diagnostics = output.segmentation_report["soft_cut_stats"]["planner_diagnostics"]
    assert diagnostics["rejection_stats"].get("tail_hold_preferred", 0) >= 1


def test_decision_layer_consumes_token_indexed_punctuation_facts_without_slot_mapping() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    words_for_split = [
        WordTimestamp(word="what", start=0.0, end=0.2),
        WordTimestamp(word="do", start=0.2, end=0.4),
        WordTimestamp(word="you", start=0.4, end=0.6),
        WordTimestamp(word="think", start=0.6, end=0.8),
        WordTimestamp(word="about", start=0.8, end=1.0),
        WordTimestamp(word="it", start=1.0, end=1.2),
    ]

    facts = (
        CanonicalPunctuationFact(
            fact_id="fact-token-1",
            left_token_index=1,
            right_token_index=None,
            attach_mode="trailing",
            raw_text=".",
            normalized_text=".",
            punct_class="sentence_end",
            source="aligned",
            priority=100,
            is_sentence_end=True,
            metadata={},
        ),
        CanonicalPunctuationFact(
            fact_id="fact-token-3",
            left_token_index=3,
            right_token_index=None,
            attach_mode="trailing",
            raw_text=".",
            normalized_text=".",
            punct_class="sentence_end",
            source="aligned",
            priority=100,
            is_sentence_end=True,
            metadata={},
        ),
        CanonicalPunctuationFact(
            fact_id="fact-right-token-4",
            left_token_index=None,
            right_token_index=4,
            attach_mode="leading",
            raw_text=".",
            normalized_text=".",
            punct_class="sentence_end",
            source="aligned",
            priority=100,
            is_sentence_end=True,
            metadata={},
        ),
    )

    boundaries = processor._build_punctuation_fact_boundaries(
        punctuation_facts=facts,
        words_for_split=words_for_split,
    )
    split_idx_by_fact = {
        str(item.metadata.get("fact_id", "")): int(item.split_idx)
        for item in boundaries
    }

    assert split_idx_by_fact["fact-token-1"] == 1
    assert split_idx_by_fact["fact-token-3"] == 3
    assert split_idx_by_fact["fact-right-token-4"] == 3
