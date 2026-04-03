from __future__ import annotations

from dataclasses import replace

from app.services.language_policy.types import (
    WINDOW_KIND_SINGLE_LANGUAGE,
    WINDOW_KIND_TRUE_MIXED,
)
from app.services.timeanchored_alignment.chunk_projector import ChunkWindow
from app.services.timeanchored_alignment.contracts import (
    LanguageRun,
    LanguageRunPackage,
    PhoneUnit,
    PronunciationPackage,
    SelectedTextTruth,
    TextTruthPackage,
    TextTruthQuality,
    TextTruthUnit,
    TimeBasePackage,
    TimeBaseQuality,
    TimeBaseUnit,
    TokenToPhoneSpan,
    TokenUnit,
)
from app.services.timeanchored_alignment.preparation.assembler import (
    AlignmentPreparationAssembler,
)
from app.services.timeanchored_alignment.preparation.contracts import PreparationBundle
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
from app.services.timeanchored_alignment.stage_service import (
    TimeanchoredAlignmentStageService,
)
from app.services.timeanchored_alignment.window_time_base_assembler import (
    WindowTimeBasePackage,
)


def _build_time_base(tokens: list[str], *, language: str = "zh") -> TimeBasePackage:
    raw_units = tuple(
        TimeBaseUnit(
            text=token,
            start=idx * 0.1,
            end=idx * 0.1 + 0.08,
            confidence=0.92,
            token_type="raw",
        )
        for idx, token in enumerate(tokens)
    )
    return TimeBasePackage(
        raw_units=raw_units,
        word_units=raw_units,
        quality=TimeBaseQuality(blank_ratio=0.2, avg_max_prob=0.8, low_prob_ratio=0.1),
        language=language,
    )


def _build_text_truth(tokens: list[str], *, language: str = "zh") -> TextTruthPackage:
    units = tuple(
        TextTruthUnit(
            text=token,
            normalized_text=token,
            confidence=0.95,
            language=language,
        )
        for token in tokens
    )
    text = "".join(tokens)
    return TextTruthPackage(
        units=units,
        quality=TextTruthQuality(hallucination_risk=0.0, repetition_ratio=0.0, length_ratio=1.0),
        language=language,
        raw_text=text,
        normalized_text=text,
    )


def _build_language_runs(text: str, *, kind: str, language: str = "zh") -> LanguageRunPackage:
    if kind == WINDOW_KIND_TRUE_MIXED:
        runs = (
            LanguageRun(run_text="Hello", run_language="en", char_start=0, char_end=5),
            LanguageRun(run_text="你", run_language="zh", char_start=6, char_end=7),
        )
        return LanguageRunPackage(
            runs=runs,
            dominant_language="mixed",
            window_kind=kind,
            foreign_run_ratio=0.5,
            source_text=text,
        )
    return LanguageRunPackage(
        runs=(LanguageRun(run_text=text, run_language=language, char_start=0, char_end=len(text)),),
        dominant_language=language,
        window_kind=kind,
        foreign_run_ratio=0.0,
        source_text=text,
    )


def _build_pronunciation(tokens: list[str], *, language: str = "zh") -> PronunciationPackage:
    token_units = []
    phone_units = []
    spans = []
    cursor = 0
    for token in tokens:
        token_units.append(
            TokenUnit(
                token_text=token,
                language=language,
                char_start=cursor,
                char_end=cursor + len(token),
            )
        )
        phone_start = len(phone_units)
        phone_units.append(PhoneUnit(phone_text=f"p-{token}", language=language))
        spans.append(
            TokenToPhoneSpan(
                token_index=len(token_units) - 1,
                phone_start=phone_start,
                phone_end=phone_start,
            )
        )
        cursor += len(token)
    return PronunciationPackage(
        token_units=tuple(token_units),
        phone_units=tuple(phone_units),
        token_to_phone_spans=tuple(spans),
        frontend_source="test",
        dependency_mode={language: "rule_fallback"},
        language=language,
    )


def _build_ready_window(tokens: list[str], *, language: str = "zh") -> ReadySlowWindow:
    text = "".join(tokens)
    return ReadySlowWindow(
        window_id="window-stage-001",
        owner_chunk_id="chunk-1",
        owner_chunk_index=1,
        window_mode="steady",
        flush_reason="test",
        audio_segments=((0.0, max(len(tokens) * 0.1, 0.1)),),
        coverage=WindowCoverage(
            core_segments=((0.0, max(len(tokens) * 0.1, 0.1)),),
            left_guard_sec=0.0,
            right_guard_sec=0.0,
            chunk_bindings=(
                WindowChunkBinding(
                    chunk_id="chunk-1",
                    chunk_index=1,
                    chunk_start=0.0,
                    chunk_end=max(len(tokens) * 0.1, 0.1),
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
                text=text,
                audio_start=0.0,
                audio_end=max(len(tokens) * 0.1, 0.1),
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
                speaker_id="spk-1",
                turn_id="turn-1",
                language=language,
                arrived_at=max(len(tokens) * 0.1, 0.1),
            ),
        ),
        dialogue_shape=DialogueShapeSnapshot(
            shape="single_speaker",
            speaker_count=1,
            dominant_speaker_id="spk-1",
            dominant_speaker_ratio=1.0,
            speaker_switch_count=0,
            speaker_switch_density=0.0,
            turn_count=1,
            avg_turn_duration_sec=max(len(tokens) * 0.1, 0.1),
        ),
        language_profile=WindowLanguageProfile(
            primary_language=language,
            language_mix_state="single_language",
            decision_domains=("timeanchored_alignment",),
            should_bypass_whisper=False,
        ),
        prompt_seed=PromptSeed(text=text),
        batch_hint=WindowBatchHint(
            duration_bucket="short",
            token_estimate=len(tokens),
            acoustic_density_hint="medium",
            queue_priority=1,
        ),
        created_at=max(len(tokens) * 0.1, 0.1),
    )


def _build_window_time_base_package(tokens: list[str], *, language: str = "zh") -> WindowTimeBasePackage:
    time_base = _build_time_base(tokens, language=language)
    chunk_end = max(len(tokens) * 0.1, 0.1)
    return WindowTimeBasePackage(
        window_id="window-stage-001",
        language=language,
        raw_units=time_base.raw_units,
        word_units=time_base.word_units,
        quality=time_base.quality,
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        chunk_bindings=(
            WindowChunkBinding(
                chunk_id="chunk-1",
                chunk_index=1,
                chunk_start=0.0,
                chunk_end=chunk_end,
                overlap_ratio=1.0,
                role="owner",
                is_owner=True,
            ),
        ),
    )


def _build_preparation(
    tokens: list[str],
    *,
    language: str = "zh",
    window_kind: str = WINDOW_KIND_SINGLE_LANGUAGE,
    chunk_window: ChunkWindow | None = None,
) -> PreparationBundle:
    text = "".join(tokens)
    preparation = AlignmentPreparationAssembler().prepare(
        ready_window=_build_ready_window(tokens, language=language),
        window_time_base=_build_window_time_base_package(tokens, language=language),
        selected_text_truth=SelectedTextTruth(
            text=text,
            text_source="slow",
            language_hint=language,
            source_chunk_ids=("chunk-1",),
            quality={"confidence": 0.95},
            metadata={"raw_text": text},
        ),
        default_language=language,
    )
    if window_kind != WINDOW_KIND_SINGLE_LANGUAGE:
        preparation = replace(
            preparation,
            compat=replace(
                preparation.compat,
                language_runs=_build_language_runs("Hello 你", kind=window_kind, language=language),
            ),
        )
    if chunk_window is not None:
        preparation = replace(
            preparation,
            compat=replace(preparation.compat, chunk_window=chunk_window),
        )
    return preparation


def test_stage_service_execute_builds_pipeline_report() -> None:
    service = TimeanchoredAlignmentStageService()
    tokens = ["你", "好", "世", "界"]
    result = service.execute(
        preparation=_build_preparation(tokens),
        language="zh",
        edge_selection_mode="auto",
        speaker_id="spk-1",
        turn_id="turn-1",
    )

    assert result.base_result.route in {"text", "phonetic", "fast", "slow", "mixed"}
    assert result.final_stream
    assert result.sentence_segments == ()
    assert result.output_inputs == ()
    assert result.projections == ()
    assert result.pipeline_report.alignment_report.name == "alignment"
    assert result.pipeline_report.segmentation_report.metrics["boundary_candidate_count"] >= 0
    assert result.pipeline_report.output_report.metrics["output_chunk_count"] == 0
    raw_mount_trace = result.pipeline_report.alignment_report.metrics["raw_mount_trace"]
    assert raw_mount_trace["mapping_pairs"][0]["text_index"] == 0
    assert raw_mount_trace["mapping_pairs"][0]["time_index"] == 0
    assert raw_mount_trace["time_units"][0]["text"] == "你"
    assert raw_mount_trace["text_units"][0]["text"] == "你"


def test_stage_service_mixed_window_falls_back_to_edge_selector() -> None:
    service = TimeanchoredAlignmentStageService()
    tokens = ["你", "好", "世", "界"]
    result = service.execute(
        preparation=_build_preparation(
            tokens,
            window_kind=WINDOW_KIND_TRUE_MIXED,
            chunk_window=ChunkWindow(chunk_ref=3, start=3.0, end=4.0),
        ),
        language="zh",
        edge_selection_mode="force_fast",
    )

    assert result.text_result.route == "mixed"
    assert result.edge_result.route == "fast"
    assert result.base_result.route == "fast"
    assert len(result.output_inputs) == 0
    assert result.final_stream[0].start >= 3.0
    assert result.pipeline_report.segmentation_report.metrics["boundary_candidate_count"] >= 0


def test_stage_service_applies_chunk_global_offset_for_local_stream() -> None:
    service = TimeanchoredAlignmentStageService()
    tokens = ["你", "好", "世", "界"]
    result = service.execute(
        preparation=_build_preparation(
            tokens,
            chunk_window=ChunkWindow(chunk_ref=9, start=12.0, end=13.0),
        ),
        language="zh",
        edge_selection_mode="force_fast",
    )

    assert result.final_stream
    assert result.final_stream[0].start >= 12.0
    assert result.final_stream[-1].end <= 13.0 + 1.0
    assert result.pipeline_report.segmentation_report.metrics["boundary_candidate_count"] >= 0
