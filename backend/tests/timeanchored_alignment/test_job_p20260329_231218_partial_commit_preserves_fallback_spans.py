from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.models.sensevoice_models import SentenceSegment
from app.pipelines.dual_pipeline.services.alignment_stage_service import AlignmentStageService
from app.services.textflow.contracts import SubtitleBatch, SubtitleItem
from app.services.timeanchored_alignment.window_recovery_planner import WindowRecoveryPlanner


def _job_root() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "jobs" / "p-20260329-231218-tr-test-en-1-dpgm"


def _load_stage_result(chunk_index: int) -> SimpleNamespace:
    graph_path = (
        _job_root()
        / "debug"
        / "postprocess"
        / f"chunk_{chunk_index:04d}"
        / "21_anchor_mount.graph.json"
    )
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    token_units = []
    for item in graph["mounts"]:
        span_text = str(item["unit_id"]).split(":", 1)[1]
        char_start_text, char_end_text = span_text.split("-", 1)
        envelope = item["envelope"]
        token_units.append(
            SimpleNamespace(
                unit_id=str(item["unit_id"]),
                token_index=int(item["unit_index"]),
                token_text=str(item["text"]),
                normalized_text=str(item["text"]),
                char_start=int(char_start_text),
                char_end=int(char_end_text),
                start=float(envelope["provisional_start"]),
                end=float(envelope["provisional_end"]),
                mount_status=str(item["mount_status"]),
            )
        )
    stage_result = SimpleNamespace(
        decision_ingress=SimpleNamespace(
            window_id=str(graph["window_id"]),
            owner_chunk_id=f"chunk-{chunk_index}",
            owner_chunk_index=chunk_index,
            source_chunk_ids=(f"chunk-{chunk_index}",),
            source_chunk_indices=(chunk_index,),
            anchored_token_units=tuple(token_units),
            punctuation_facts=tuple(),
            boundary_evidences=tuple(),
            cross_chunk_locks=tuple(),
            timeline_validity=str(graph["metrics"]["timeline_validity"]),
        ),
        anchor_mount_result=SimpleNamespace(
            should_fallback=True,
            timeline_validity=str(graph["metrics"]["timeline_validity"]),
            metrics=dict(graph["metrics"]),
        ),
    )
    stage_result.window_recovery_plan = WindowRecoveryPlanner().build(stage_result=stage_result)
    return stage_result


def _decision_run(
    *,
    owner_chunk_id: str,
    owner_chunk_index: int,
    span_selector: tuple[int, int],
    start: float,
    end: float,
) -> SimpleNamespace:
    token_start, token_end = span_selector
    text = f"decision-{token_start}-{token_end}"
    sentence = SentenceSegment(
        text=text,
        text_clean=text,
        start=start,
        end=end,
        words=[],
    )
    return SimpleNamespace(
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
            sentence_segments=(sentence,),
            output_traces=[],
            segmentation_report={},
            subtitle_batch=SubtitleBatch(
                chunk_id=owner_chunk_id,
                chunk_index=owner_chunk_index,
                items=(
                    SubtitleItem(
                        segment_id=f"decision:{token_start}:{token_end}",
                        chunk_id=owner_chunk_id,
                        start=start,
                        end=end,
                        text=text,
                        source="aligned",
                    ),
                ),
            ),
        ),
        final_sentences=(sentence,),
        fallback_error_code="",
    )


@pytest.mark.parametrize(
    ("chunk_index", "expected_fallback_selector"),
    (
        (5, (0, 28)),
        (32, (17, 35)),
    ),
)
def test_job_p20260329_231218_partial_commit_keeps_span_fallback_in_owner_batch(
    monkeypatch,
    chunk_index: int,
    expected_fallback_selector: tuple[int, int],
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
    stage_result = _load_stage_result(chunk_index)
    window_spans = service._build_partial_window_spans(stage_result=stage_result)
    decision_spans = service._build_partial_commit_spans(stage_result=stage_result)
    decision_span_time_map = {
        (int(span.token_start), int(span.token_end)): (
            float(span.time_start),
            float(span.time_end),
        )
        for span in decision_spans
    }
    dispatched_batches: list[SubtitleBatch] = []

    ctx = SimpleNamespace(
        chunk_index=chunk_index,
        audio_chunk=SimpleNamespace(start=0.0, end=float(len(stage_result.decision_ingress.anchored_token_units))),
        arbitration_result=SimpleNamespace(chosen_source="slow"),
        alignment_preparation=SimpleNamespace(
            compat=SimpleNamespace(text_truth=SimpleNamespace(language="en")),
            slow_text=SimpleNamespace(window_text=SimpleNamespace(text="")),
        ),
        text_tracks=None,
        punct_track=None,
    )

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
    monkeypatch.setattr(
        service,
        "_execute_timeanchored_decision_pass",
        lambda **kwargs: _decision_run(
            owner_chunk_id=str(stage_result.decision_ingress.owner_chunk_id),
            owner_chunk_index=int(stage_result.decision_ingress.owner_chunk_index),
            span_selector=kwargs["span_selector"],
            start=decision_span_time_map[kwargs["span_selector"]][0],
            end=decision_span_time_map[kwargs["span_selector"]][1],
        ),
    )
    monkeypatch.setattr(
        service,
        "_build_projected_output_batches",
        lambda *, stage_result, decision_output, split_stats: dispatched_batches.append(
            decision_output.subtitle_batch
        ) or (decision_output.subtitle_batch,),
    )

    service._commit_timeanchored_main_chain_result(
        ctx=ctx,
        stage_result=stage_result,
        whisper_result={"language": "en"},
        sv_result={},
        speaker_id=None,
        turn_id=None,
    )

    assert [span.route for span in window_spans].count("span_fallback") == 1
    assert [span.route for span in decision_spans].count("span_fallback") == 0
    fallback_span = next(span for span in window_spans if span.route == "span_fallback")
    assert (fallback_span.token_start, fallback_span.token_end) == expected_fallback_selector

    assert len(dispatched_batches) == 1
    batch = dispatched_batches[0]
    assert len(batch.items) == len(window_spans)
    assert sum(1 for item in batch.items if item.source == "span_fallback") == 1
    assert any(item.text.strip() for item in batch.items if item.source == "span_fallback")
