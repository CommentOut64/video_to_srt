from __future__ import annotations

from app.services.timeanchored_alignment.contracts import (
    TextTruthPackage,
    TextTruthQuality,
    TextTruthUnit,
    TimeBasePackage,
    TimeBaseQuality,
    TimeBaseUnit,
)
from app.services.timeanchored_alignment.hallucination_gate import (
    GateDecision,
    HallucinationGate,
)


def _build_time_base(*, blank_ratio: float, avg_max_prob: float, low_prob_ratio: float) -> TimeBasePackage:
    units = (
        TimeBaseUnit(text="你", start=0.0, end=0.1, confidence=0.9, token_type="word"),
        TimeBaseUnit(text="好", start=0.1, end=0.2, confidence=0.9, token_type="word"),
    )
    return TimeBasePackage(
        raw_units=units,
        word_units=units,
        quality=TimeBaseQuality(
            blank_ratio=blank_ratio,
            avg_max_prob=avg_max_prob,
            low_prob_ratio=low_prob_ratio,
            unit_count=len(units),
            word_count=len(units),
        ),
        language="zh",
    )


def _build_text_truth(*, is_hallucination: bool, hallucination_risk: float) -> TextTruthPackage:
    units = (
        TextTruthUnit(
            text="你",
            normalized_text="你",
            confidence=0.9,
            language="zh",
            start=0.0,
            end=0.1,
        ),
        TextTruthUnit(
            text="好",
            normalized_text="好",
            confidence=0.9,
            language="zh",
            start=0.1,
            end=0.2,
        ),
    )
    return TextTruthPackage(
        units=units,
        quality=TextTruthQuality(
            hallucination_risk=hallucination_risk,
            repetition_ratio=0.0,
            length_ratio=1.0,
        ),
        language="zh",
        raw_text="你好",
        normalized_text="你好",
        is_hallucination=is_hallucination,
    )


def test_slow_hallucination_selects_fast() -> None:
    gate = HallucinationGate()
    decision = gate.evaluate(
        time_base=_build_time_base(blank_ratio=0.1, avg_max_prob=0.85, low_prob_ratio=0.1),
        text_truth=_build_text_truth(is_hallucination=True, hallucination_risk=0.95),
    )
    assert decision.decision is GateDecision.USE_FAST
    assert decision.reason == "slow_hallucination"


def test_fast_abnormal_selects_slow() -> None:
    gate = HallucinationGate()
    decision = gate.evaluate(
        time_base=_build_time_base(blank_ratio=0.75, avg_max_prob=0.15, low_prob_ratio=0.8),
        text_truth=_build_text_truth(is_hallucination=False, hallucination_risk=0.05),
    )
    assert decision.decision is GateDecision.USE_SLOW
    assert decision.reason == "fast_abnormal"


def test_both_abnormal_reports_error() -> None:
    gate = HallucinationGate()
    decision = gate.evaluate(
        time_base=_build_time_base(blank_ratio=0.8, avg_max_prob=0.1, low_prob_ratio=0.9),
        text_truth=_build_text_truth(is_hallucination=True, hallucination_risk=0.95),
    )
    assert decision.decision is GateDecision.ERROR
    assert decision.error_code == "EDGE_SELECTOR_BOTH_SIDES_UNAVAILABLE"


def test_both_normal_proceeds() -> None:
    gate = HallucinationGate()
    decision = gate.evaluate(
        time_base=_build_time_base(blank_ratio=0.1, avg_max_prob=0.9, low_prob_ratio=0.1),
        text_truth=_build_text_truth(is_hallucination=False, hallucination_risk=0.05),
    )
    assert decision.decision is GateDecision.PROCEED
    assert decision.reason == "proceed"
