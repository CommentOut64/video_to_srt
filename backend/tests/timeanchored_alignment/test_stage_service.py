from __future__ import annotations

from app.services.language_policy.types import (
    WINDOW_KIND_SINGLE_LANGUAGE,
    WINDOW_KIND_TRUE_MIXED,
)
from app.services.timeanchored_alignment import ChunkWindow, TimeanchoredAlignmentStageService
from app.services.timeanchored_alignment.contracts import (
    LanguageRun,
    LanguageRunPackage,
    PhoneUnit,
    PronunciationPackage,
    TextTruthPackage,
    TextTruthQuality,
    TextTruthUnit,
    TimeBasePackage,
    TimeBaseQuality,
    TimeBaseUnit,
    TokenToPhoneSpan,
    TokenUnit,
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


def test_stage_service_execute_builds_pipeline_report() -> None:
    service = TimeanchoredAlignmentStageService()
    tokens = ["你", "好", "世", "界"]
    result = service.execute(
        time_base=_build_time_base(tokens),
        text_truth=_build_text_truth(tokens),
        language_runs=_build_language_runs("你好世界", kind=WINDOW_KIND_SINGLE_LANGUAGE),
        pronunciation=_build_pronunciation(tokens),
        chunk_window=ChunkWindow(chunk_ref=0, start=0.0, end=1.0),
        language="zh",
        edge_selection_mode="auto",
        speaker_id="spk-1",
        turn_id="turn-1",
    )

    assert result.base_result.route in {"text", "phonetic", "fast", "slow", "mixed"}
    assert len(result.sentence_segments) > 0
    assert result.pipeline_report.alignment_report.name == "alignment"
    assert result.pipeline_report.output_report.metrics["output_chunk_count"] >= 1


def test_stage_service_mixed_window_falls_back_to_edge_selector() -> None:
    service = TimeanchoredAlignmentStageService()
    tokens = ["你", "好", "世", "界"]
    result = service.execute(
        time_base=_build_time_base(tokens),
        text_truth=_build_text_truth(tokens),
        language_runs=_build_language_runs("Hello 你", kind=WINDOW_KIND_TRUE_MIXED),
        pronunciation=_build_pronunciation(tokens),
        chunk_window=ChunkWindow(chunk_ref=3, start=3.0, end=4.0),
        language="zh",
        edge_selection_mode="force_fast",
    )

    assert result.text_result.route == "mixed"
    assert result.edge_result.route == "fast"
    assert result.base_result.route == "fast"
    assert len(result.output_inputs) == 1
    assert result.final_stream[0].start >= 3.0
    assert result.sentence_segments[0].start >= 3.0


def test_stage_service_applies_chunk_global_offset_for_local_stream() -> None:
    service = TimeanchoredAlignmentStageService()
    tokens = ["你", "好", "世", "界"]
    result = service.execute(
        time_base=_build_time_base(tokens),
        text_truth=_build_text_truth(tokens),
        language_runs=_build_language_runs("你好世界", kind=WINDOW_KIND_SINGLE_LANGUAGE),
        pronunciation=_build_pronunciation(tokens),
        chunk_window=ChunkWindow(chunk_ref=9, start=12.0, end=13.0),
        language="zh",
        edge_selection_mode="force_fast",
    )

    assert result.final_stream
    assert result.final_stream[0].start >= 12.0
    assert result.final_stream[-1].end <= 13.0 + 1.0
    assert result.sentence_segments[0].start >= 12.0
