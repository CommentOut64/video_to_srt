from __future__ import annotations

from app.schemas.pipeline_context import ProcessingContext
from app.services.timeanchored_alignment.contracts import (
    LanguageRun,
    PhoneUnit,
    PronunciationPackage,
    ProtectedSpan,
    TokenToPhoneSpan,
    TokenUnit,
    TextTruthPackage,
    TextTruthQuality,
    TextTruthUnit,
)


def _build_dummy_text_truth() -> TextTruthPackage:
    unit = TextTruthUnit(
        text="原文",
        normalized_text="原文",
        confidence=0.9,
        language="zh",
    )
    return TextTruthPackage(
        units=(unit,),
        quality=TextTruthQuality(
            hallucination_risk=0.0,
            repetition_ratio=0.0,
            length_ratio=1.0,
        ),
        language="zh",
        raw_text="原文",
        normalized_text="原文",
    )


def _build_dummy_pronunciation() -> PronunciationPackage:
    token = TokenUnit(token_text="你", language="zh", char_start=0, char_end=1)
    phone = PhoneUnit(phone_text="ni3", language="zh")
    span = TokenToPhoneSpan(token_index=0, phone_start=0, phone_end=0)
    return PronunciationPackage(
        token_units=(token,),
        phone_units=(phone,),
        token_to_phone_spans=(span,),
        frontend_source="homophone_tokenizer",
        dependency_mode={"zh": "pypinyin"},
        language="zh",
    )


def test_preparation_context_release_artifacts_cleans_large_intermediate_data() -> None:
    context = ProcessingContext(
        job_id="job-1",
        chunk_index=0,
        audio_chunk=None,
        sv_result={
            "text": "你好",
            "ctc_logits": [[0.1, 0.9]],
            "top_candidates": [{"token": "你"}],
            "compact_trace": {"frames": [1, 2, 3]},
        },
        alignment_preparation=object(),
        text_truth=_build_dummy_text_truth(),
        protected_spans=[ProtectedSpan(start=0, end=2, kind="decimal", text="3.14")],
        language_runs=[LanguageRun(run_text="你好", run_language="zh", char_start=0, char_end=2)],
        pronunciation_package=_build_dummy_pronunciation(),
        slow_window_meta={"window_id": "sw-001"},
        pronunciation_report={"cache_size": 1024},
    )

    context.release_preparation_artifacts()

    assert context.text_truth is None
    assert context.alignment_preparation is None
    assert context.protected_spans == []
    assert context.language_runs == []
    assert context.pronunciation_package is None
    assert context.slow_window_meta == {}
    assert context.pronunciation_report == {}
    assert context.sv_result is not None
    assert "ctc_logits" not in context.sv_result
    assert "top_candidates" not in context.sv_result
    assert "compact_trace" not in context.sv_result
    assert context.sv_result.get("text") == "你好"


def test_preparation_context_release_artifacts_is_idempotent() -> None:
    context = ProcessingContext(
        job_id="job-2",
        chunk_index=1,
        audio_chunk=None,
        sv_result={"text": "hello"},
    )
    context.release_preparation_artifacts()
    context.release_preparation_artifacts()

    assert context.text_truth is None
    assert context.alignment_preparation is None
    assert context.protected_spans == []
    assert context.language_runs == []
