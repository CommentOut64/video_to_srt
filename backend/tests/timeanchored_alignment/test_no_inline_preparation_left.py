from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import Mock

from app.schemas.pipeline_context import ProcessingContext
from app.services.timeanchored_alignment.preparation import AlignmentPreparationAssembler
from app.pipelines.dual_pipeline.services.alignment_stage_service import AlignmentStageService
from app.services.timeanchored_alignment.contracts import (
    TimeBaseQuality,
    TimeBaseUnit,
)
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
from app.services.timeanchored_alignment.window_time_base_assembler import (
    WindowTimeBasePackage,
)


def _build_ready_window() -> ReadySlowWindow:
    return ReadySlowWindow(
        window_id="window-inline-test",
        owner_chunk_id="chunk-1",
        owner_chunk_index=1,
        window_mode="steady",
        flush_reason="test",
        audio_segments=((0.0, 1.0),),
        coverage=WindowCoverage(
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
        ),
        source_semantic_chunk_ids=("sem-1",),
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        source_units=(
            WindowSourceUnit(
                unit_id="unit-1",
                semantic_chunk_id="sem-1",
                text="你好世界",
                audio_start=0.0,
                audio_end=1.0,
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
                speaker_id="speaker-a",
                turn_id="turn-a",
                language="zh",
                arrived_at=1.0,
            ),
        ),
        dialogue_shape=DialogueShapeSnapshot(
            shape="single_speaker",
            speaker_count=1,
            dominant_speaker_id="speaker-a",
            dominant_speaker_ratio=1.0,
            speaker_switch_count=0,
            speaker_switch_density=0.0,
            turn_count=1,
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
        created_at=1.0,
    )


def _build_window_time_base() -> WindowTimeBasePackage:
    units = (
        TimeBaseUnit(text="你", start=0.0, end=0.1, confidence=0.9, token_type="word"),
        TimeBaseUnit(text="好", start=0.1, end=0.2, confidence=0.9, token_type="word"),
        TimeBaseUnit(text="世", start=0.2, end=0.3, confidence=0.9, token_type="word"),
        TimeBaseUnit(text="界", start=0.3, end=0.4, confidence=0.9, token_type="word"),
    )
    return WindowTimeBasePackage(
        window_id="window-inline-test",
        language="zh",
        raw_units=units,
        word_units=units,
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.9, low_prob_ratio=0.05),
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
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


def _build_preparation_package():
    return AlignmentPreparationAssembler().prepare(
        ready_window=_build_ready_window(),
        window_time_base=_build_window_time_base(),
        whisper_result={
            "text": "你好，世界！",
            "text_clean": "你好，世界！",
            "text_itn_raw": "你好，世界！",
            "confidence": 0.9,
            "language": "zh",
            "raw_result": {"segments": [{"avg_logprob": -0.1}]},
        },
        default_language="zh",
    )


def test_alignment_stage_service_no_longer_contains_inline_preparation_calls() -> None:
    source = inspect.getsource(AlignmentStageService._run_timeanchored_main_chain)

    assert "build_text_truth_package" not in source
    assert "build_runs(" not in source
    assert "build_package(" not in source
    assert "extract_protected_spans" not in source


def test_stage_prepared_entry_only_accepts_alignment_preparation_package() -> None:
    signature = inspect.signature(AlignmentStageService._execute_prepared_timeanchored_stage)
    parameter_names = tuple(signature.parameters.keys())

    assert "preparation" in parameter_names
    assert "text_truth" not in parameter_names
    assert "language_runs" not in parameter_names
    assert "pronunciation" not in parameter_names


def test_run_timeanchored_main_chain_delegates_to_preparation_service(monkeypatch) -> None:
    host = SimpleNamespace(
        _whisper_sanitizer=None,
        _hallucination_detector=None,
        _edge_selection_mode="auto",
        _select_text_for_alignment=lambda track: str(getattr(track, "text_clean", "") or ""),
        logger=Mock(),
    )
    service = AlignmentStageService(host=host)
    preparation = _build_preparation_package()
    captured: dict[str, object] = {}
    expected_result = object()

    monkeypatch.setattr(
        service._timeanchored_preparation_assembler,
        "prepare",
        lambda **_kwargs: preparation,
    )

    def _capture_execute(**kwargs):
        captured.update(kwargs)
        return expected_result

    monkeypatch.setattr(service._timeanchored_stage_service, "execute", _capture_execute)

    ctx = ProcessingContext(
        job_id="job-inline-test",
        chunk_index=1,
        audio_chunk=SimpleNamespace(start=0.0, end=1.0),
        whisper_result={"text": "你好，世界！", "language": "zh"},
        sv_result={"text": "你好世界"},
        time_base_chunk=preparation.compat.time_base,
    )
    ctx.ready_slow_window = _build_ready_window()
    ctx.window_time_base = _build_window_time_base()

    result = service._run_timeanchored_main_chain(
        ctx=ctx,
        tracks=SimpleNamespace(chosen_track=SimpleNamespace(text_clean="你好世界")),
        sv_result={"text": "你好世界"},
        whisper_result={"text": "你好，世界！", "language": "zh"},
        language_hint="zh",
        speaker_id="speaker-a",
        turn_id="turn-a",
    )

    assert result is expected_result
    assert captured["preparation"] is preparation
    assert captured["language"] == "zh"
    assert "time_base" not in captured
    assert "text_truth" not in captured
    assert "language_runs" not in captured
    assert "pronunciation" not in captured
    assert "chunk_window" not in captured
