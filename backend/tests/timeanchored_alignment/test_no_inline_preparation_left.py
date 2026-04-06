from __future__ import annotations

import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

from app.schemas.pipeline_context import ProcessingContext
from app.services.timeanchored_alignment.preparation import AlignmentPreparationAssembler
from app.pipelines.dual_pipeline.services.alignment_stage_service import AlignmentStageService
from app.services.timeanchored_alignment.contracts import (
    SelectedTextTruth,
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


def _build_multi_source_ready_window() -> ReadySlowWindow:
    return ReadySlowWindow(
        window_id="sw-000017",
        owner_chunk_id="chunk-33",
        owner_chunk_index=33,
        window_mode="steady",
        flush_reason="target_duration",
        audio_segments=((240.18, 247.4), (247.46, 259.21)),
        coverage=WindowCoverage(
            core_segments=((240.18, 259.21),),
            left_guard_sec=0.0,
            right_guard_sec=0.0,
            chunk_bindings=(
                WindowChunkBinding(
                    chunk_id="chunk-32",
                    chunk_index=32,
                    chunk_start=240.18,
                    chunk_end=247.4,
                    overlap_ratio=1.0,
                    role="core",
                    is_owner=False,
                ),
                WindowChunkBinding(
                    chunk_id="chunk-33",
                    chunk_index=33,
                    chunk_start=247.46,
                    chunk_end=259.21,
                    overlap_ratio=1.0,
                    role="owner",
                    is_owner=True,
                ),
            ),
        ),
        source_semantic_chunk_ids=("sem-32", "sem-33"),
        source_chunk_ids=("chunk-32", "chunk-33"),
        source_chunk_indices=(32, 33),
        source_units=(
            WindowSourceUnit(
                unit_id="unit-32",
                semantic_chunk_id="sem-32",
                text="在中午12点50分，果然在距离会元名他们捡到可乐的地方600米处。",
                audio_start=240.18,
                audio_end=247.4,
                source_chunk_ids=("chunk-32",),
                source_chunk_indices=(32,),
                speaker_id="speaker-a",
                turn_id="turn-a",
                language="zh",
                arrived_at=247.4,
            ),
            WindowSourceUnit(
                unit_id="unit-33",
                semantic_chunk_id="sem-33",
                text="这家商店的老板有个15岁的儿子，他说自己之前就看到这瓶可乐。",
                audio_start=247.46,
                audio_end=259.21,
                source_chunk_ids=("chunk-33",),
                source_chunk_indices=(33,),
                speaker_id="speaker-a",
                turn_id="turn-a",
                language="zh",
                arrived_at=259.21,
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
            avg_turn_duration_sec=19.03,
        ),
        language_profile=WindowLanguageProfile(
            primary_language="zh",
            language_mix_state="single_language",
            decision_domains=("timeanchored_alignment",),
            should_bypass_whisper=False,
        ),
        prompt_seed=PromptSeed(
            text="在中午12点50分，果然在距离会元名他们捡到可乐的地方600米处。 这家商店的老板有个15岁的儿子，他说自己之前就看到这瓶可乐。"
        ),
        batch_hint=WindowBatchHint(
            duration_bucket="medium",
            token_estimate=32,
            acoustic_density_hint="medium",
            queue_priority=1,
        ),
        created_at=259.21,
    )


def _build_multi_source_window_time_base() -> WindowTimeBasePackage:
    units = (
        TimeBaseUnit(text="在", start=240.2, end=240.5, confidence=0.9, token_type="word"),
        TimeBaseUnit(text="中午", start=240.5, end=241.1, confidence=0.9, token_type="word"),
        TimeBaseUnit(text="商店", start=249.2, end=249.8, confidence=0.9, token_type="word"),
        TimeBaseUnit(text="可乐", start=250.6, end=251.2, confidence=0.9, token_type="word"),
    )
    return WindowTimeBasePackage(
        window_id="sw-000017",
        language="zh",
        raw_units=units,
        word_units=units,
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.9, low_prob_ratio=0.05),
        source_chunk_ids=("chunk-32", "chunk-33"),
        source_chunk_indices=(32, 33),
        chunk_bindings=(
            WindowChunkBinding(
                chunk_id="chunk-32",
                chunk_index=32,
                chunk_start=240.18,
                chunk_end=247.4,
                overlap_ratio=1.0,
                role="core",
                is_owner=False,
            ),
            WindowChunkBinding(
                chunk_id="chunk-33",
                chunk_index=33,
                chunk_start=247.46,
                chunk_end=259.21,
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
        selected_text_truth=SelectedTextTruth(
            text="你好，世界！",
            text_source="slow",
            language_hint="zh",
            source_chunk_ids=("chunk-1",),
            quality={"confidence": 0.9},
            metadata={"raw_text": "你好，世界！"},
        ),
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
        logger=Mock(),
    )
    service = AlignmentStageService(host=host)
    preparation = _build_preparation_package()
    captured: dict[str, object] = {}
    expected_result = object()

    def _capture_prepare(**kwargs):
        captured.update({f"prepare_{key}": value for key, value in kwargs.items()})
        return preparation

    monkeypatch.setattr(
        service._timeanchored_preparation_assembler,
        "prepare",
        _capture_prepare,
    )

    def _capture_execute(**kwargs):
        captured.update(kwargs)
        return expected_result

    monkeypatch.setattr(service, "_execute_prepared_timeanchored_stage", _capture_execute)

    ctx = ProcessingContext(
        job_id="job-inline-test",
        chunk_index=1,
        audio_chunk=SimpleNamespace(start=0.0, end=1.0),
        whisper_result={"text": "你好，世界！", "language": "zh"},
        sv_result={"text": "你好世界"},
        time_base_chunk=preparation.compat.time_base,
        selected_text_truth=SelectedTextTruth(
            text="你好世界",
            text_source="slow",
            language_hint="zh",
            source_chunk_ids=("chunk-1",),
            quality={"confidence": 0.9},
        ),
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
    assert captured["prepare_selected_text_truth"] is ctx.selected_text_truth
    assert captured["preparation"] is preparation
    assert captured["language"] == "zh"
    assert "time_base" not in captured
    assert "text_truth" not in captured
    assert "language_runs" not in captured
    assert "pronunciation" not in captured
    assert "chunk_window" not in captured


def test_run_timeanchored_main_chain_promotes_window_source_text_when_owner_text_is_shorter(
    monkeypatch,
) -> None:
    host = SimpleNamespace(
        _whisper_sanitizer=None,
        _hallucination_detector=None,
        _edge_selection_mode="auto",
        logger=Mock(),
    )
    service = AlignmentStageService(host=host)
    preparation = _build_preparation_package()
    captured_prepare: dict[str, object] = {}

    def _capture_prepare(**kwargs):
        captured_prepare.update(kwargs)
        return preparation

    monkeypatch.setattr(
        service._timeanchored_preparation_assembler,
        "prepare",
        _capture_prepare,
    )
    monkeypatch.setattr(service, "_execute_prepared_timeanchored_stage", lambda **_: object())

    ctx = ProcessingContext(
        job_id="job-window-promote",
        chunk_index=33,
        audio_chunk=SimpleNamespace(start=247.46, end=259.21),
        whisper_result={
            "text": "这家商店的老板有个15岁的儿子，他说自己之前就看到这瓶可乐。",
            "language": "zh",
        },
        sv_result={"text": "这家商店的老板有个15岁的儿子"},
        time_base_chunk=_build_multi_source_window_time_base(),
        selected_text_truth=SelectedTextTruth(
            text="这家商店的老板有个15岁的儿子，他说自己之前就看到这瓶可乐。",
            text_source="slow",
            language_hint="zh",
            source_chunk_ids=("chunk-33",),
            quality={"confidence": 0.9},
        ),
    )
    ctx.ready_slow_window = _build_multi_source_ready_window()
    ctx.window_time_base = _build_multi_source_window_time_base()

    service._run_timeanchored_main_chain(
        ctx=ctx,
        tracks=SimpleNamespace(
            chosen_track=SimpleNamespace(
                text_clean="这家商店的老板有个15岁的儿子，他说自己之前就看到这瓶可乐。"
            )
        ),
        sv_result={"text": "这家商店的老板有个15岁的儿子"},
        whisper_result={
            "text": "这家商店的老板有个15岁的儿子，他说自己之前就看到这瓶可乐。",
            "language": "zh",
        },
        language_hint="zh",
        speaker_id="speaker-a",
        turn_id="turn-a",
    )

    promoted_selected_text_truth = captured_prepare["selected_text_truth"]
    promoted_whisper = captured_prepare["whisper_result"]

    assert isinstance(promoted_selected_text_truth, SelectedTextTruth)
    assert "在中午12点50分" in promoted_selected_text_truth.text
    assert "这家商店的老板" in promoted_selected_text_truth.text
    assert promoted_selected_text_truth.source_chunk_ids == ("chunk-32", "chunk-33")
    assert promoted_selected_text_truth.metadata.get("promoted_from_window_source_units") is True
    assert isinstance(promoted_whisper, dict)
    assert "在中午12点50分" in str(promoted_whisper.get("text", ""))


def test_preparation_context_meta_uses_formal_preparation_metrics_only() -> None:
    host = SimpleNamespace(
        _whisper_sanitizer=None,
        _hallucination_detector=None,
        _edge_selection_mode="auto",
        logger=Mock(),
    )
    service = AlignmentStageService(host=host)
    preparation = _build_preparation_package()
    ctx = ProcessingContext(job_id="job-meta-test", chunk_index=1, audio_chunk=None)

    service._commit_preparation_context(ctx=ctx, preparation=preparation)

    assert ctx.slow_window_meta["window_id"] == preparation.window_id
    assert ctx.slow_window_meta["source_chunk_ids"] == list(preparation.source_chunk_ids)
    assert ctx.slow_window_meta["canonical_token_count"] == len(preparation.canonical_sequence.tokens)
    assert "owner_chunk_id" not in ctx.slow_window_meta
    assert "compat_hook_count" not in ctx.slow_window_meta


def test_run_timeanchored_main_chain_writes_preparation_layer_summary(monkeypatch) -> None:
    host = SimpleNamespace(
        _whisper_sanitizer=None,
        _hallucination_detector=None,
        _edge_selection_mode="auto",
        logger=Mock(),
    )
    service = AlignmentStageService(host=host)
    preparation = _build_preparation_package()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        service._timeanchored_preparation_assembler,
        "prepare",
        lambda **_kwargs: preparation,
    )
    monkeypatch.setattr(service, "_execute_prepared_timeanchored_stage", lambda **_kwargs: object())
    monkeypatch.setattr(service, "_trace_write", lambda **_kwargs: None)

    with TemporaryDirectory() as tmpdir:
        ctx = ProcessingContext(
            job_id="job-summary-test",
            chunk_index=1,
            audio_chunk=SimpleNamespace(start=0.0, end=1.0),
            job_dir=Path(tmpdir),
            whisper_result={"text": "你好，世界！", "language": "zh"},
            sv_result={"text": "你好世界"},
            time_base_chunk=preparation.compat.time_base,
            selected_text_truth=SelectedTextTruth(
                text="你好世界",
                text_source="slow",
                language_hint="zh",
                source_chunk_ids=("chunk-1",),
                quality={"confidence": 0.9},
            ),
        )
        ctx.ready_slow_window = _build_ready_window()
        ctx.window_time_base = _build_window_time_base()

        service._postprocess_trace_writer._enabled = True
        monkeypatch.setattr(
            service._postprocess_trace_writer,
            "write_layer_summary",
            lambda **kwargs: captured.update(kwargs),
        )

        service._run_timeanchored_main_chain(
            ctx=ctx,
            tracks=SimpleNamespace(chosen_track=SimpleNamespace(text_clean="你好世界")),
            sv_result={"text": "你好世界"},
            whisper_result={"text": "你好，世界！", "language": "zh"},
            language_hint="zh",
            speaker_id="speaker-a",
            turn_id="turn-a",
        )

    assert captured["job_dir"] == ctx.job_dir
    assert captured["layer_summary"] == preparation.report.summary


def test_public_exports_do_not_expose_alignment_preparation_package() -> None:
    preparation_module = __import__(
        "app.services.timeanchored_alignment.preparation",
        fromlist=["__all__"],
    )
    timeanchored_module = __import__(
        "app.services.timeanchored_alignment",
        fromlist=["__all__"],
    )

    preparation_exports = tuple(getattr(preparation_module, "__all__", ()) or ())
    timeanchored_exports = tuple(getattr(timeanchored_module, "__all__", ()) or ())

    assert "PreparationBundle" in preparation_exports
    assert "PreparationBundle" in timeanchored_exports
    assert "AlignmentPreparationPackage" not in preparation_exports
    assert "AlignmentPreparationPackage" not in timeanchored_exports
