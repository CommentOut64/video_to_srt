from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from app.services.timeanchored_alignment.contracts import (
    AcousticCandidate,
    AlignmentItem,
    AlignmentMetrics,
    FinalAlignmentResult,
    PhoneticPackage,
    PhoneticUnit,
    PipelineReport,
    ProtectedSpan,
    SlowInferenceWindow,
    SlowInferenceWindowEnvelope,
    TextTruthPackage,
    TextTruthQuality,
    TextTruthUnit,
    TimeBasePackage,
    TimeBaseQuality,
    TimeBaseUnit,
)


def _make_time_base_package() -> TimeBasePackage:
    unit = TimeBaseUnit(text="你", start=0.0, end=0.06, confidence=0.95)
    quality = TimeBaseQuality(blank_ratio=0.2, avg_max_prob=0.8, low_prob_ratio=0.1)
    return TimeBasePackage(
        raw_units=(unit,),
        word_units=(unit,),
        quality=quality,
        language="zh",
    )


def test_time_base_unit_frozen() -> None:
    unit = TimeBaseUnit(text="你", start=0.0, end=0.06, confidence=0.95)
    with pytest.raises(FrozenInstanceError):
        unit.text = "我"


def test_time_base_package_contract_version() -> None:
    pkg = _make_time_base_package()
    assert pkg.contract_version == "1.0"


def test_acoustic_candidate_score_range() -> None:
    cand = AcousticCandidate(text="呀", score=0.53)
    assert 0.0 <= cand.score <= 1.0


@pytest.mark.parametrize("status", ["direct", "phonetic", "estimated", "interpolated", "failed"])
def test_alignment_item_status_enum(status: str) -> None:
    item = AlignmentItem(text="你", start=0.0, end=0.1, status=status, source="text_align")
    assert item.status == status


def test_alignment_item_status_rejects_invalid_value() -> None:
    with pytest.raises(ValueError):
        AlignmentItem(text="你", start=0.0, end=0.1, status="unknown", source="text_align")


@pytest.mark.parametrize("route", ["text", "phonetic", "mixed", "fast", "slow", "error"])
def test_final_alignment_result_route_enum(route: str) -> None:
    result = FinalAlignmentResult(
        items=(),
        route=route,
        metrics=AlignmentMetrics(),
    )
    assert result.route == route


def test_final_alignment_result_route_rejects_invalid_value() -> None:
    with pytest.raises(ValueError):
        FinalAlignmentResult(items=(), route="invalid", metrics=AlignmentMetrics())


def test_text_truth_package_frozen() -> None:
    pkg = TextTruthPackage(
        units=(TextTruthUnit(text="hello", normalized_text="hello", confidence=0.9, language="en"),),
        quality=TextTruthQuality(hallucination_risk=0.1, repetition_ratio=0.0, length_ratio=1.0),
        language="en",
    )
    with pytest.raises(FrozenInstanceError):
        pkg.language = "zh"


def test_phonetic_package_frozen() -> None:
    pkg = PhoneticPackage(
        units=(PhoneticUnit(token="你", reading_key="ni3", language="zh", confidence=0.9),),
        language="zh",
    )
    with pytest.raises(FrozenInstanceError):
        pkg.language = "en"


def test_slow_inference_window_frozen() -> None:
    window = SlowInferenceWindow(window_id="w-1", start=0.0, end=2.0)
    with pytest.raises(FrozenInstanceError):
        window.window_id = "w-2"


def test_protected_span_frozen() -> None:
    span = ProtectedSpan(start=0, end=2, kind="number")
    with pytest.raises(FrozenInstanceError):
        span.kind = "date"


def test_slow_inference_window_envelope_contract() -> None:
    envelope = SlowInferenceWindowEnvelope(
        window=SlowInferenceWindow(window_id="w-1", start=0.0, end=2.0),
        aggregated_time_base=_make_time_base_package(),
        source_contexts=("chunk-0",),
    )
    assert envelope.window.window_id == "w-1"
    assert envelope.aggregated_time_base is not None


def test_pipeline_report_contract_version() -> None:
    report = PipelineReport()
    assert report.contract_version == "1.0"
