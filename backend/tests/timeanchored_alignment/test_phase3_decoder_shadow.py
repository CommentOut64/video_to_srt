from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from app.pipelines.dual_pipeline.services.alignment_stage_service import AlignmentStageService
from app.services.timeanchored_alignment.contracts import (
    AlignmentReport,
    SelectedTextTruth,
    TimeBaseQuality,
    TimeBaseUnit,
)
from app.services.timeanchored_alignment.decoder.contracts import DecoderShadowResult
from app.services.timeanchored_alignment.decoder.service import AlignmentDecoderService
from app.services.timeanchored_alignment.preparation.assembler import AlignmentPreparationAssembler
from app.services.timeanchored_alignment.slow_window.contracts import (
    DialogueShapeSnapshot,
    PromptSeed,
    ReadySlowWindow,
    WindowBatchHint,
    WindowChunkBinding,
    WindowCoverage,
    WindowLanguageProfile,
    WindowSourceUnit,
)
from app.services.timeanchored_alignment.window_time_base_assembler import WindowTimeBasePackage


def _build_ready_window() -> ReadySlowWindow:
    return ReadySlowWindow(
        window_id="phase3-window-001",
        owner_chunk_id="chunk-1",
        owner_chunk_index=1,
        window_mode="steady",
        flush_reason="test",
        audio_segments=((0.0, 1.0), (1.0, 2.0)),
        coverage=WindowCoverage(
            core_segments=((0.0, 2.0),),
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
                WindowChunkBinding(
                    chunk_id="chunk-2",
                    chunk_index=2,
                    chunk_start=1.0,
                    chunk_end=2.0,
                    overlap_ratio=1.0,
                    role="core",
                    is_owner=False,
                ),
            ),
        ),
        source_semantic_chunk_ids=("sem-1", "sem-2"),
        source_chunk_ids=("chunk-1", "chunk-2"),
        source_chunk_indices=(1, 2),
        source_units=(
            WindowSourceUnit(
                unit_id="unit-1",
                semantic_chunk_id="sem-1",
                text="你好",
                audio_start=0.0,
                audio_end=1.0,
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
                speaker_id="speaker-a",
                turn_id="turn-a",
                language="zh",
                arrived_at=1.0,
            ),
            WindowSourceUnit(
                unit_id="unit-2",
                semantic_chunk_id="sem-2",
                text="世界",
                audio_start=1.0,
                audio_end=2.0,
                source_chunk_ids=("chunk-2",),
                source_chunk_indices=(2,),
                speaker_id="speaker-b",
                turn_id="turn-b",
                language="zh",
                arrived_at=2.0,
            ),
        ),
        dialogue_shape=DialogueShapeSnapshot(
            shape="two_party_stable",
            speaker_count=2,
            dominant_speaker_id="speaker-a",
            dominant_speaker_ratio=0.5,
            speaker_switch_count=1,
            speaker_switch_density=0.5,
            turn_count=2,
            avg_turn_duration_sec=1.0,
        ),
        language_profile=WindowLanguageProfile(
            primary_language="zh",
            language_mix_state="single_language",
            decision_domains=("timeanchored_alignment",),
            should_bypass_whisper=False,
        ),
        prompt_seed=PromptSeed(text="你好世界"),
        batch_hint=WindowBatchHint(
            duration_bucket="short",
            token_estimate=4,
            acoustic_density_hint="medium",
            queue_priority=1,
        ),
        created_at=2.0,
    )


def _build_window_time_base() -> WindowTimeBasePackage:
    units = (
        TimeBaseUnit(text="你", start=0.0, end=0.1, confidence=0.95, token_type="word"),
        TimeBaseUnit(text="好", start=0.1, end=0.2, confidence=0.95, token_type="word"),
        TimeBaseUnit(text="世", start=1.0, end=1.1, confidence=0.95, token_type="word"),
        TimeBaseUnit(text="界", start=1.1, end=1.2, confidence=0.95, token_type="word"),
    )
    ready_window = _build_ready_window()
    return WindowTimeBasePackage(
        window_id=ready_window.window_id,
        language="zh",
        raw_units=units,
        word_units=units,
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.9, low_prob_ratio=0.05),
        source_chunk_ids=ready_window.source_chunk_ids,
        source_chunk_indices=ready_window.source_chunk_indices,
        chunk_bindings=ready_window.coverage.chunk_bindings,
    )


def _build_preparation_bundle(*, selected_source: str = "slow"):
    ready_window = _build_ready_window()
    selected_text_truth = SelectedTextTruth(
        text="你好世界",
        text_source=selected_source,
        language_hint="zh",
        source_chunk_ids=ready_window.source_chunk_ids,
        quality={"confidence": 0.9},
        metadata={"raw_text": "你好，世界！"},
    )
    return AlignmentPreparationAssembler().prepare(
        ready_window=ready_window,
        window_time_base=_build_window_time_base(),
        selected_text_truth=selected_text_truth,
        whisper_result={
            "text": "你好，世界！",
            "text_clean": "你好世界",
            "text_itn_raw": "你好，世界！",
            "confidence": 0.9,
            "language": "zh",
            "raw_result": {"segments": [{"avg_logprob": -0.1}]},
        },
        default_language="zh",
    )


def test_phase3_decoder_builds_alignment_path_from_preparation_bundle() -> None:
    preparation = _build_preparation_bundle()

    result = AlignmentDecoderService().execute(preparation=preparation)

    assert result.alignment_path is not None
    assert result.alignment_report.route == "alignment_path"
    assert len(result.alignment_path.aligned_tokens) == len(preparation.canonical_sequence.tokens)
    assert result.alignment_path.source_chunk_ids == preparation.source_chunk_ids
    assert result.boundary_candidates == result.alignment_path.boundary_candidates


def test_phase3_decoder_reports_selection_reject_when_slow_text_not_selected() -> None:
    preparation = _build_preparation_bundle(selected_source="fast")

    result = AlignmentDecoderService().execute(preparation=preparation)

    assert result.alignment_path is None
    assert result.alignment_report.route == "selection_reject_slow"
    assert result.alignment_report.failure_semantic == "selection_reject_slow"


def test_phase3_decoder_reports_true_mixed_unresolved_for_mixed_window() -> None:
    preparation = _build_preparation_bundle()
    mixed_sequence = replace(
        preparation.canonical_sequence,
        language_hint="mixed",
    )
    mixed_provenance = replace(
        preparation.provenance,
        text_source="slow",
    )
    mixed_bundle = replace(
        preparation,
        canonical_sequence=mixed_sequence,
        provenance=mixed_provenance,
    )

    result = AlignmentDecoderService().execute(preparation=mixed_bundle)

    assert result.alignment_path is None
    assert result.alignment_report.route == "true_mixed_unresolved"
    assert result.alignment_report.failure_semantic == "true_mixed_unresolved"


def test_phase3_decoder_keeps_alignment_path_route_when_only_failure_semantic_is_low_confidence() -> None:
    preparation = _build_preparation_bundle()
    ambiguous_node = replace(
        preparation.pronunciation_graph.token_nodes[1],
        metadata={
            **dict(preparation.pronunciation_graph.token_nodes[1].metadata or {}),
            "pronunciation_ambiguous": True,
        },
    )
    ambiguous_graph = replace(
        preparation.pronunciation_graph,
        token_nodes=(
            preparation.pronunciation_graph.token_nodes[0],
            ambiguous_node,
            *preparation.pronunciation_graph.token_nodes[2:],
        ),
    )
    ambiguous_bundle = replace(
        preparation,
        pronunciation_graph=ambiguous_graph,
    )

    result = AlignmentDecoderService().execute(preparation=ambiguous_bundle)

    assert result.alignment_path is not None
    assert result.alignment_report.route == "alignment_path"
    assert result.alignment_report.failure_semantic == "alignment_low_confidence"


def test_phase3_alignment_stage_writes_decoder_shadow_trace_and_diff(tmp_path: Path) -> None:
    host = SimpleNamespace(
        logger=SimpleNamespace(
            debug=lambda *args, **kwargs: None,
            exception=lambda *args, **kwargs: None,
        ),
        _postprocess_trace_enabled=True,
        _postprocess_trace_level="summary",
        _alignment_pipeline_shadow_sample_rate=1.0,
        _alignment_pipeline_write_debug_artifacts=False,
    )
    service = AlignmentStageService(host=host)
    ctx = SimpleNamespace(
        chunk_index=0,
        job_id="phase3-shadow-job",
        job_dir=tmp_path,
    )
    preparation = _build_preparation_bundle()

    stage_result = service._execute_prepared_timeanchored_stage(
        ctx=ctx,
        preparation=preparation,
        language="zh",
    )

    assert isinstance(stage_result, DecoderShadowResult)
    shadow_output = tmp_path / "debug" / "postprocess" / "chunk_0000" / "21_alignment_decoder.output.json"
    shadow_summary = tmp_path / "debug" / "postprocess" / "summaries" / "alignment.summary.json"

    assert shadow_output.exists()
    assert shadow_summary.exists()

    output_payload = json.loads(shadow_output.read_text(encoding="utf-8"))
    summary_payload = json.loads(shadow_summary.read_text(encoding="utf-8"))

    assert output_payload["route"] in {
        "alignment_path",
        "alignment_low_confidence",
    }
    assert summary_payload["layer"] == "alignment"
    assert summary_payload["status"] in {"ok", "warning", "error"}
