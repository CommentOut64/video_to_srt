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
from app.services.timeanchored_alignment.decoder.contracts import DecoderPath, DecoderShadowResult, DecoderStep
from app.services.timeanchored_alignment.decoder.local_repair import LocalRepair
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


def _build_future_hijack_ready_window() -> ReadySlowWindow:
    observation_tokens = ("alpha", "beta", "gamma", "box")
    bindings = []
    source_units = []
    cursor = 0.0
    for index, token_text in enumerate(observation_tokens):
        start = cursor
        end = start + 0.2
        bindings.append(
            WindowChunkBinding(
                chunk_id=f"chunk-future-{index}",
                chunk_index=index,
                chunk_start=start,
                chunk_end=end,
                overlap_ratio=1.0,
                role="owner" if index == 0 else "core",
                is_owner=index == 0,
            )
        )
        source_units.append(
            WindowSourceUnit(
                unit_id=f"future-unit-{index}",
                semantic_chunk_id=f"future-sem-{index}",
                text=token_text,
                audio_start=start,
                audio_end=end,
                source_chunk_ids=(f"chunk-future-{index}",),
                source_chunk_indices=(index,),
                speaker_id="speaker-a",
                turn_id="turn-a",
                language="en",
                arrived_at=end,
            )
        )
        cursor = end
    return ReadySlowWindow(
        window_id="phase3-window-future-hijack",
        owner_chunk_id="chunk-future-0",
        owner_chunk_index=0,
        window_mode="steady",
        flush_reason="test",
        audio_segments=tuple((unit.audio_start, unit.audio_end) for unit in source_units),
        coverage=WindowCoverage(
            core_segments=((0.0, cursor),),
            left_guard_sec=0.0,
            right_guard_sec=0.0,
            chunk_bindings=tuple(bindings),
        ),
        source_semantic_chunk_ids=tuple(f"future-sem-{index}" for index in range(len(observation_tokens))),
        source_chunk_ids=tuple(f"chunk-future-{index}" for index in range(len(observation_tokens))),
        source_chunk_indices=tuple(range(len(observation_tokens))),
        source_units=tuple(source_units),
        dialogue_shape=DialogueShapeSnapshot(
            shape="single_party",
            speaker_count=1,
            dominant_speaker_id="speaker-a",
            dominant_speaker_ratio=1.0,
            speaker_switch_count=0,
            speaker_switch_density=0.0,
            turn_count=1,
            avg_turn_duration_sec=cursor,
        ),
        language_profile=WindowLanguageProfile(
            primary_language="en",
            language_mix_state="single_language",
            decision_domains=("timeanchored_alignment",),
            should_bypass_whisper=False,
        ),
        prompt_seed=PromptSeed(text="alpha box beta gamma box"),
        batch_hint=WindowBatchHint(
            duration_bucket="short",
            token_estimate=5,
            acoustic_density_hint="medium",
            queue_priority=1,
        ),
        created_at=cursor,
    )


def _build_future_hijack_window_time_base() -> WindowTimeBasePackage:
    ready_window = _build_future_hijack_ready_window()
    units = (
        TimeBaseUnit(text="alpha", start=0.0, end=0.2, confidence=0.95, token_type="word"),
        TimeBaseUnit(text="beta", start=0.2, end=0.4, confidence=0.95, token_type="word"),
        TimeBaseUnit(text="gamma", start=0.4, end=0.6, confidence=0.95, token_type="word"),
        TimeBaseUnit(text="box", start=0.6, end=0.8, confidence=0.95, token_type="word"),
    )
    return WindowTimeBasePackage(
        window_id=ready_window.window_id,
        language="en",
        raw_units=units,
        word_units=units,
        quality=TimeBaseQuality(blank_ratio=0.0, avg_max_prob=0.9, low_prob_ratio=0.0),
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


def _build_future_hijack_preparation_bundle():
    ready_window = _build_future_hijack_ready_window()
    selected_text = "alpha box beta gamma box"
    return AlignmentPreparationAssembler().prepare(
        ready_window=ready_window,
        window_time_base=_build_future_hijack_window_time_base(),
        selected_text_truth=SelectedTextTruth(
            text=selected_text,
            text_source="slow",
            language_hint="en",
            source_chunk_ids=ready_window.source_chunk_ids,
            quality={"confidence": 0.9},
            metadata={"raw_text": selected_text},
        ),
        whisper_result={
            "text": selected_text,
            "text_clean": selected_text,
            "text_itn_raw": selected_text,
            "confidence": 0.9,
            "language": "en",
            "raw_result": {"segments": [{"avg_logprob": -0.1}]},
        },
        default_language="en",
    )


def test_phase3_decoder_builds_alignment_path_from_preparation_bundle() -> None:
    preparation = _build_preparation_bundle()

    result = AlignmentDecoderService().execute(preparation=preparation)

    assert result.alignment_path is not None
    assert result.alignment_report.route == "alignment_path"
    assert len(result.alignment_path.aligned_tokens) == len(preparation.canonical_sequence.tokens)
    assert result.alignment_path.source_chunk_ids == preparation.source_chunk_ids
    assert result.boundary_candidates == result.alignment_path.boundary_candidates


def test_phase3_decoder_uses_null_align_instead_of_future_hijack() -> None:
    preparation = _build_future_hijack_preparation_bundle()

    assert [
        item.primary_token for item in preparation.acoustic_observation_pack.slices
    ] == ["alpha", "beta", "gamma", "box"]

    result = AlignmentDecoderService().execute(preparation=preparation)

    assert result.alignment_path is not None
    first_box = result.alignment_path.aligned_tokens[1]
    beta = result.alignment_path.aligned_tokens[2]
    gamma = result.alignment_path.aligned_tokens[3]
    final_box = result.alignment_path.aligned_tokens[4]

    assert first_box.trace["synthetic"] is True
    assert first_box.trace["synthetic_reason"] == "null_align"
    assert first_box.trace["slice_primary_token"] == ""
    assert beta.trace["slice_primary_token"] == "beta"
    assert gamma.trace["slice_primary_token"] == "gamma"
    assert final_box.trace["slice_primary_token"] == "box"


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


def test_phase3_local_repair_interpolates_null_span_between_real_anchors() -> None:
    preparation = _build_preparation_bundle()
    decode_path = DecoderPath(
        steps=(
            DecoderStep(
                token_index=0,
                slice_index=0,
                score=0.9,
                confidence=0.9,
                lexical_exact=True,
                synthetic=False,
                metadata={"slice_start": 0.0, "slice_end": 0.1, "match_kind": "direct"},
            ),
            DecoderStep(
                token_index=1,
                slice_index=1,
                score=0.16,
                confidence=0.16,
                lexical_exact=False,
                synthetic=True,
                metadata={"slice_start": None, "slice_end": None, "match_kind": "null", "synthetic_reason": "null_align"},
            ),
            DecoderStep(
                token_index=2,
                slice_index=2,
                score=0.16,
                confidence=0.16,
                lexical_exact=False,
                synthetic=True,
                metadata={"slice_start": None, "slice_end": None, "match_kind": "null", "synthetic_reason": "null_align"},
            ),
            DecoderStep(
                token_index=3,
                slice_index=3,
                score=0.9,
                confidence=0.9,
                lexical_exact=True,
                synthetic=False,
                metadata={"slice_start": 1.1, "slice_end": 1.2, "match_kind": "direct"},
            ),
        ),
        total_score=2.12,
        matched_token_count=2,
        direct_token_count=2,
        coverage_ratio=0.5,
        direct_ratio=0.5,
    )

    repaired = LocalRepair().repair(preparation=preparation, decode_path=decode_path)

    assert repaired.steps[1].metadata["resolved_start"] == 0.1
    assert repaired.steps[1].metadata["resolved_end"] == 0.6
    assert repaired.steps[2].metadata["resolved_start"] == 0.6
    assert repaired.steps[2].metadata["resolved_end"] == 1.1
    assert repaired.steps[2].metadata["repair_reason"] == "bounded_span_interpolation"


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
