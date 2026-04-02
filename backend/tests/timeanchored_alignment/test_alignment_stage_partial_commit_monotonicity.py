from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.models.sensevoice_models import SentenceSegment
from app.pipelines.dual_pipeline.services.alignment_stage_service import AlignmentStageService
from app.services.textflow.contracts import SubtitleBatch, SubtitleItem
from app.services.timeanchored_alignment.window_recovery_contracts import (
    RecoveredWindowPlan,
    WindowSpanDecision,
)


def _build_service() -> AlignmentStageService:
    return AlignmentStageService(
        host=SimpleNamespace(
            logger=Mock(),
            _edge_selection_mode="force_slow",
            _postprocess_trace_enabled=False,
            _postprocess_trace_level="summary",
            _anchor_mount_graph="off",
        )
    )


def _item(*, segment_id: str, start: float, end: float, text: str) -> SubtitleItem:
    return SubtitleItem(
        segment_id=segment_id,
        chunk_id="chunk-owner",
        start=start,
        end=end,
        text=text,
        source="aligned",
    )


def _batch(*items: SubtitleItem) -> SubtitleBatch:
    return SubtitleBatch(
        chunk_id="chunk-owner",
        chunk_index=19,
        items=tuple(items),
    )


def test_merge_partial_commit_batches_combines_monotonic_items_into_owner_batch() -> None:
    service = _build_service()

    merged = service._merge_partial_commit_batches(
        owner_chunk_id="chunk-owner",
        owner_chunk_index=19,
        partial_batches=(
            _batch(_item(segment_id="seg-1", start=1.0, end=1.5, text="alpha")),
            _batch(_item(segment_id="seg-2", start=1.5, end=2.0, text="beta")),
        ),
    )

    assert merged.chunk_id == "chunk-owner"
    assert merged.chunk_index == 19
    assert [(item.segment_id, item.start, item.end) for item in merged.items] == [
        ("seg-1", 1.0, 1.5),
        ("seg-2", 1.5, 2.0),
    ]


def test_merge_partial_commit_batches_rejects_overlapping_sentence_ranges() -> None:
    service = _build_service()

    with pytest.raises(ValueError, match="partial_commit_non_monotonic"):
        service._merge_partial_commit_batches(
            owner_chunk_id="chunk-owner",
            owner_chunk_index=19,
            partial_batches=(
                _batch(_item(segment_id="seg-1", start=1.0, end=1.6, text="alpha")),
                _batch(_item(segment_id="seg-2", start=1.5, end=2.0, text="beta")),
            ),
        )


def test_partial_commit_merges_multiple_decision_spans_before_output_dispatch(
    monkeypatch,
) -> None:
    host = SimpleNamespace(
        logger=Mock(),
        _edge_selection_mode="force_slow",
        _postprocess_trace_enabled=False,
        _postprocess_trace_level="summary",
        _anchor_mount_graph="off",
        _is_last_chunk_index=lambda _index: False,
        _final_grouper=None,
        _decision_processor=SimpleNamespace(),
        _final_splitter=SimpleNamespace(last_split_stats={}),
        _assign_sentence_identity_by_timeline_overlap=lambda *_args, **_kwargs: None,
        _normalize_output_traces_for_sentences=lambda **_kwargs: [],
        _resolve_sentence_confidence_source=lambda _words: "aligned",
        _update_soft_cut_observability=lambda **_kwargs: {},
        _emit_output_layer=lambda **_kwargs: SimpleNamespace(output_payload={"errors": []}),
    )
    service = AlignmentStageService(host=host)
    stage_result = SimpleNamespace(
        decision_ingress=SimpleNamespace(
            window_id="window-partial",
            owner_chunk_id="chunk-owner",
            owner_chunk_index=19,
            source_chunk_ids=("chunk-owner",),
            source_chunk_indices=(19,),
            anchored_token_units=(object(),) * 8,
            punctuation_facts=tuple(),
            boundary_evidences=tuple(),
            cross_chunk_locks=tuple(),
        ),
        anchor_mount_result=SimpleNamespace(metrics={}),
        window_recovery_plan=RecoveredWindowPlan(
            spans=(
                WindowSpanDecision(
                    span_id="span-1",
                    span_kind="trusted",
                    route="decision",
                    token_start=1,
                    token_end=3,
                ),
                WindowSpanDecision(
                    span_id="span-2",
                    span_kind="trusted",
                    route="decision",
                    token_start=5,
                    token_end=7,
                ),
            )
        ),
    )
    ctx = SimpleNamespace(
        chunk_index=19,
        audio_chunk=SimpleNamespace(start=1.0, end=3.0),
        arbitration_result=SimpleNamespace(chosen_source="slow"),
        alignment_preparation=SimpleNamespace(
            compat=SimpleNamespace(text_truth=SimpleNamespace(language="en")),
            slow_text=SimpleNamespace(window_text=SimpleNamespace(text="alpha beta")),
        ),
        text_tracks=None,
        punct_track=None,
    )
    span_selector_calls: list[tuple[int, int] | None] = []
    dispatched_batches: list[SubtitleBatch] = []
    process_count = {"value": 0}

    monkeypatch.setattr(
        service,
        "_resolve_anchor_mount_routes",
        lambda **_kwargs: ("slow", "partial_commit", "partial_commit", None),
    )
    monkeypatch.setattr(service, "_inject_timeline_turns_into_decision_input", lambda **_kwargs: {})
    monkeypatch.setattr(service, "_trace_write", lambda **_kwargs: None)
    monkeypatch.setattr(
        service,
        "_build_punctuation_chain_health_metrics",
        lambda **_kwargs: {
            "chosen_source": "slow",
            "punct_track_positions_total": 0,
            "preparation_punctuation_count": 0,
            "punctuation_fact_count": 0,
            "punct_track_sentence_end_count": 0,
            "punctuation_chain_broken_flag": 0,
            "punctuation_chain_broken_reason": "",
        },
    )

    def _mock_build(*, package, span_selector=None, speaker_id=None, turn_id=None):
        span_selector_calls.append(span_selector)
        return SimpleNamespace(
            decision_input=SimpleNamespace(
                aligned_facts=SimpleNamespace(
                    annotated_words=[],
                    speaker_turns=[],
                    time_mappings=[],
                ),
                fused_evidence=SimpleNamespace(
                    speaker_changes=[],
                    pause_anchors=[],
                    semantic_anchors=[],
                    punctuation_anchors=[],
                ),
                fallback_clean_text_ref="",
            ),
            stream_id=f"timeanchored:{package.window_id}",
            chunk_index=19,
            compat_report={"fallback_punctuation_position_count": 0},
        )

    def _mock_process(*_args, **_kwargs):
        process_count["value"] += 1
        index = process_count["value"]
        return SimpleNamespace(
            sentence_segments=[
                SentenceSegment(
                    text=f"seg{index}",
                    text_clean=f"seg{index}",
                    start=float(index),
                    end=float(index) + 0.4,
                    words=[],
                )
            ],
            output_traces=[],
            segmentation_report={},
            subtitle_batch=SubtitleBatch(
                chunk_id="chunk-owner",
                chunk_index=19,
                items=(
                    SubtitleItem(
                        segment_id=f"seg-{index}",
                        chunk_id="chunk-owner",
                        start=float(index),
                        end=float(index) + 0.4,
                        text=f"seg{index}",
                        source="aligned",
                    ),
                ),
            ),
        )

    def _mock_build_projected_output_batches(*, stage_result, decision_output, split_stats):
        dispatched_batches.append(decision_output.subtitle_batch)
        return (decision_output.subtitle_batch,)

    monkeypatch.setattr(service._decision_ingress_adapter, "build", _mock_build)
    host._decision_processor.process = _mock_process
    monkeypatch.setattr(
        service,
        "_build_projected_output_batches",
        _mock_build_projected_output_batches,
    )

    service._commit_timeanchored_main_chain_result(
        ctx=ctx,
        stage_result=stage_result,
        whisper_result={"language": "en"},
        sv_result={},
        speaker_id=None,
        turn_id=None,
    )

    assert span_selector_calls == [(1, 3), (5, 7)]
    assert len(dispatched_batches) == 1
    assert [item.segment_id for item in dispatched_batches[0].items] == ["seg-1", "seg-2"]
    assert [sentence.text for sentence in ctx.final_sentences] == ["seg1", "seg2"]


def test_partial_commit_preserves_span_fallback_content_in_merged_owner_batch(
    monkeypatch,
) -> None:
    host = SimpleNamespace(
        logger=Mock(),
        _edge_selection_mode="force_slow",
        _postprocess_trace_enabled=False,
        _postprocess_trace_level="summary",
        _anchor_mount_graph="off",
        _is_last_chunk_index=lambda _index: False,
        _final_grouper=None,
        _decision_processor=SimpleNamespace(),
        _final_splitter=SimpleNamespace(last_split_stats={}),
        _assign_sentence_identity_by_timeline_overlap=lambda *_args, **_kwargs: None,
        _normalize_output_traces_for_sentences=lambda **_kwargs: [],
        _resolve_sentence_confidence_source=lambda _words: "aligned",
        _update_soft_cut_observability=lambda **_kwargs: {},
        _emit_output_layer=lambda **_kwargs: SimpleNamespace(output_payload={"errors": []}),
    )
    service = AlignmentStageService(host=host)
    stage_result = SimpleNamespace(
        decision_ingress=SimpleNamespace(
            window_id="window-partial-fallback",
            owner_chunk_id="chunk-owner",
            owner_chunk_index=19,
            source_chunk_ids=("chunk-owner",),
            source_chunk_indices=(19,),
            anchored_token_units=(object(),) * 3,
            punctuation_facts=tuple(),
            boundary_evidences=tuple(),
            cross_chunk_locks=tuple(),
        ),
        anchor_mount_result=SimpleNamespace(metrics={}),
        window_recovery_plan=RecoveredWindowPlan(
            spans=(
                WindowSpanDecision(
                    span_id="span-fallback",
                    span_kind="fallback",
                    route="span_fallback",
                    token_start=0,
                    token_end=1,
                    char_start=0,
                    char_end=5,
                    time_start=1.0,
                    time_end=1.4,
                ),
                WindowSpanDecision(
                    span_id="span-decision",
                    span_kind="trusted",
                    route="decision",
                    token_start=1,
                    token_end=3,
                    char_start=6,
                    char_end=16,
                    time_start=1.4,
                    time_end=2.1,
                ),
            )
        ),
    )
    ctx = SimpleNamespace(
        chunk_index=19,
        audio_chunk=SimpleNamespace(start=1.0, end=2.1),
        arbitration_result=SimpleNamespace(chosen_source="slow"),
        alignment_preparation=SimpleNamespace(
            compat=SimpleNamespace(text_truth=SimpleNamespace(language="en")),
            slow_text=SimpleNamespace(window_text=SimpleNamespace(text="alpha beta gamma")),
        ),
        text_tracks=None,
        punct_track=None,
    )
    span_selector_calls: list[tuple[int, int] | None] = []
    dispatched_batches: list[SubtitleBatch] = []

    monkeypatch.setattr(
        service,
        "_resolve_anchor_mount_routes",
        lambda **_kwargs: ("slow", "partial_commit", "partial_commit", None),
    )
    monkeypatch.setattr(service, "_inject_timeline_turns_into_decision_input", lambda **_kwargs: {})
    monkeypatch.setattr(service, "_trace_write", lambda **_kwargs: None)
    monkeypatch.setattr(
        service,
        "_build_punctuation_chain_health_metrics",
        lambda **_kwargs: {
            "chosen_source": "slow",
            "punct_track_positions_total": 0,
            "preparation_punctuation_count": 0,
            "punctuation_fact_count": 0,
            "punct_track_sentence_end_count": 0,
            "punctuation_chain_broken_flag": 0,
            "punctuation_chain_broken_reason": "",
        },
    )

    def _mock_build(*, package, span_selector=None, speaker_id=None, turn_id=None):
        span_selector_calls.append(span_selector)
        return SimpleNamespace(
            decision_input=SimpleNamespace(
                aligned_facts=SimpleNamespace(
                    annotated_words=[],
                    speaker_turns=[],
                    time_mappings=[],
                ),
                fused_evidence=SimpleNamespace(
                    speaker_changes=[],
                    pause_anchors=[],
                    semantic_anchors=[],
                    punctuation_anchors=[],
                ),
                fallback_clean_text_ref="",
            ),
            stream_id=f"timeanchored:{package.window_id}",
            chunk_index=19,
            compat_report={"fallback_punctuation_position_count": 0},
        )

    def _mock_process(*_args, **_kwargs):
        return SimpleNamespace(
            sentence_segments=[
                SentenceSegment(
                    text="beta gamma",
                    text_clean="beta gamma",
                    start=1.4,
                    end=2.1,
                    words=[],
                )
            ],
            output_traces=[],
            segmentation_report={},
            subtitle_batch=SubtitleBatch(
                chunk_id="chunk-owner",
                chunk_index=19,
                items=(
                    SubtitleItem(
                        segment_id="seg-decision",
                        chunk_id="chunk-owner",
                        start=1.4,
                        end=2.1,
                        text="beta gamma",
                        source="aligned",
                    ),
                ),
            ),
        )

    def _mock_build_projected_output_batches(*, stage_result, decision_output, split_stats):
        dispatched_batches.append(decision_output.subtitle_batch)
        return (decision_output.subtitle_batch,)

    monkeypatch.setattr(service._decision_ingress_adapter, "build", _mock_build)
    host._decision_processor.process = _mock_process
    monkeypatch.setattr(
        service,
        "_build_projected_output_batches",
        _mock_build_projected_output_batches,
    )

    service._commit_timeanchored_main_chain_result(
        ctx=ctx,
        stage_result=stage_result,
        whisper_result={"language": "en"},
        sv_result={},
        speaker_id=None,
        turn_id=None,
    )

    assert span_selector_calls == [(1, 3)]
    assert len(dispatched_batches) == 1
    assert [item.text for item in dispatched_batches[0].items] == ["alpha", "beta gamma"]
    assert [sentence.text for sentence in ctx.final_sentences] == ["alpha", "beta gamma"]


def test_partial_commit_non_monotonic_upgrades_to_safe_window_fallback(
    monkeypatch,
) -> None:
    service = _build_service()
    stage_result = SimpleNamespace(
        decision_ingress=SimpleNamespace(
            window_id="window-partial",
            owner_chunk_id="chunk-owner",
            owner_chunk_index=19,
        ),
        anchor_mount_result=SimpleNamespace(metrics={}),
        window_recovery_plan=RecoveredWindowPlan(
            spans=(
                WindowSpanDecision(
                    span_id="span-1",
                    span_kind="trusted",
                    route="decision",
                    token_start=1,
                    token_end=3,
                ),
                WindowSpanDecision(
                    span_id="span-2",
                    span_kind="trusted",
                    route="decision",
                    token_start=5,
                    token_end=7,
                ),
            )
        ),
    )
    ctx = SimpleNamespace(
        chunk_index=19,
        alignment_preparation=SimpleNamespace(
            compat=SimpleNamespace(text_truth=SimpleNamespace(language="en")),
            slow_text=SimpleNamespace(window_text=SimpleNamespace(text="alpha beta")),
        ),
        text_tracks=None,
    )
    safe_fallback_calls: list[dict] = []
    decision_runs = iter(
        (
            SimpleNamespace(
                adapter_result=SimpleNamespace(
                    decision_input=SimpleNamespace(
                        aligned_facts=SimpleNamespace(
                            annotated_words=[],
                            speaker_turns=[],
                            time_mappings=[],
                        ),
                        fused_evidence=SimpleNamespace(
                            speaker_changes=[],
                            pause_anchors=[],
                            semantic_anchors=[],
                            punctuation_anchors=[],
                        ),
                    ),
                    compat_report={"fallback_punctuation_position_count": 0},
                ),
                decision_output=SimpleNamespace(
                    subtitle_batch=_batch(
                        _item(segment_id="seg-1", start=1.0, end=1.6, text="alpha")
                    ),
                    output_traces=[],
                ),
                final_sentences=tuple(),
                fallback_error_code="",
            ),
            SimpleNamespace(
                adapter_result=SimpleNamespace(
                    decision_input=SimpleNamespace(
                        aligned_facts=SimpleNamespace(
                            annotated_words=[],
                            speaker_turns=[],
                            time_mappings=[],
                        ),
                        fused_evidence=SimpleNamespace(
                            speaker_changes=[],
                            pause_anchors=[],
                            semantic_anchors=[],
                            punctuation_anchors=[],
                        ),
                    ),
                    compat_report={"fallback_punctuation_position_count": 0},
                ),
                decision_output=SimpleNamespace(
                    subtitle_batch=_batch(
                        _item(segment_id="seg-2", start=1.5, end=2.0, text="beta")
                    ),
                    output_traces=[],
                ),
                final_sentences=tuple(),
                fallback_error_code="",
            ),
        )
    )

    monkeypatch.setattr(
        service,
        "_resolve_anchor_mount_routes",
        lambda **_kwargs: ("slow", "partial_commit", "partial_commit", None),
    )
    monkeypatch.setattr(
        service,
        "_execute_timeanchored_decision_pass",
        lambda **_kwargs: next(decision_runs),
    )
    monkeypatch.setattr(
        service,
        "_commit_safe_window_fallback_result",
        lambda **kwargs: safe_fallback_calls.append(kwargs),
    )

    service._commit_timeanchored_main_chain_result(
        ctx=ctx,
        stage_result=stage_result,
        whisper_result={"language": "en"},
        sv_result={},
        speaker_id=None,
        turn_id=None,
    )

    assert len(safe_fallback_calls) == 1
    assert safe_fallback_calls[0]["reason"] == "partial_commit_non_monotonic"
