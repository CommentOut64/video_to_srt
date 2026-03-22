from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.services.timeanchored_alignment.contracts import (
    TextTruthPackage,
    TextTruthQuality,
    TextTruthUnit,
    TimeBasePackage,
    TimeBaseQuality,
    TimeBaseUnit,
)
from app.services.timeanchored_alignment.edge_selector import EdgeSelector, FailedSpan
from app.services.timeanchored_alignment.hallucination_gate import GateDecision, GateEvaluation


def _build_time_base(
    tokens: list[str],
    *,
    avg_max_prob: float = 0.88,
    blank_ratio: float = 0.1,
    low_prob_ratio: float = 0.1,
) -> TimeBasePackage:
    units = tuple(
        TimeBaseUnit(
            text=token,
            start=index * 0.2,
            end=index * 0.2 + 0.16,
            confidence=0.9,
            token_type="word",
        )
        for index, token in enumerate(tokens)
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


def _build_text_truth(
    tokens: list[str],
    *,
    is_hallucination: bool = False,
    hallucination_risk: float = 0.05,
    with_timestamps: bool = True,
    raw_text: str | None = None,
) -> TextTruthPackage:
    units = []
    for index, token in enumerate(tokens):
        if with_timestamps:
            start = index * 0.2
            end = index * 0.2 + 0.16
        else:
            start = None
            end = None
        units.append(
            TextTruthUnit(
                text=token,
                normalized_text=token,
                confidence=0.9,
                language="zh",
                start=start,
                end=end,
            )
        )
    content = raw_text if raw_text is not None else "".join(tokens)
    return TextTruthPackage(
        units=tuple(units),
        quality=TextTruthQuality(
            hallucination_risk=hallucination_risk,
            repetition_ratio=0.0,
            length_ratio=1.0,
        ),
        language="zh",
        raw_text=content,
        normalized_text=content,
        is_hallucination=is_hallucination,
    )


def test_select_fast_when_slow_hallucinated() -> None:
    selector = EdgeSelector()
    result = selector.select(
        time_base=_build_time_base(["你", "好"]),
        text_truth=_build_text_truth(["你", "好"], is_hallucination=True, hallucination_risk=0.95),
        failed_spans=(),
        edge_selection_mode="auto",
    )
    assert result.route == "fast"
    assert result.error_code is None
    assert all(item.source == "edge_selector_fast" for item in result.items)


def test_select_slow_when_fast_unavailable() -> None:
    selector = EdgeSelector()
    result = selector.select(
        time_base=_build_time_base([], avg_max_prob=0.2, blank_ratio=0.8, low_prob_ratio=0.9),
        text_truth=_build_text_truth(["慢", "流"], with_timestamps=True),
        failed_spans=(),
        edge_selection_mode="auto",
    )
    assert result.route == "slow"
    assert result.error_code is None
    assert all(item.source == "edge_selector_slow" for item in result.items)


def test_error_when_both_fail() -> None:
    selector = EdgeSelector()
    result = selector.select(
        time_base=_build_time_base([], avg_max_prob=0.2, blank_ratio=0.8, low_prob_ratio=0.9),
        text_truth=_build_text_truth(
            ["坏", "样本"],
            is_hallucination=True,
            hallucination_risk=0.95,
            with_timestamps=False,
        ),
        failed_spans=(),
        edge_selection_mode="auto",
    )
    assert result.route == "error"
    assert result.error_code == "EDGE_SELECTOR_BOTH_SIDES_UNAVAILABLE"


def test_interpolated_vs_estimated() -> None:
    selector = EdgeSelector()
    text_truth = TextTruthPackage(
        units=(
            TextTruthUnit(
                text="有锚点",
                normalized_text="有锚点",
                confidence=0.9,
                language="zh",
                start=0.0,
                end=0.4,
            ),
            TextTruthUnit(
                text="无锚点",
                normalized_text="无锚点",
                confidence=0.9,
                language="zh",
            ),
        ),
        quality=TextTruthQuality(hallucination_risk=0.05, repetition_ratio=0.0, length_ratio=1.0),
        language="zh",
        raw_text="有锚点无锚点",
        normalized_text="有锚点无锚点",
    )
    result = selector.select_slow(time_base=_build_time_base(["有", "锚", "点", "无", "锚", "点"]), text_truth=text_truth)
    assert result.route == "slow"
    assert [item.status for item in result.items] == ["estimated", "interpolated"]


@dataclass
class _SpyGate:
    calls: int = 0

    def evaluate(self, *, time_base: TimeBasePackage, text_truth: TextTruthPackage) -> GateEvaluation:
        self.calls += 1
        return GateEvaluation(decision=GateDecision.USE_SLOW, reason="spy")


@pytest.mark.parametrize(
    ("mode", "expected_route"),
    [("force_fast", "fast"), ("force_slow", "slow")],
)
def test_force_mode_bypasses_hallucination_gate(mode: str, expected_route: str) -> None:
    spy_gate = _SpyGate()
    selector = EdgeSelector(hallucination_gate=spy_gate)
    result = selector.select(
        time_base=_build_time_base(["快", "流"]),
        text_truth=_build_text_truth(["慢", "流"], with_timestamps=True),
        failed_spans=(FailedSpan(start=0, end=0),),
        edge_selection_mode=mode,
    )
    assert result.route == expected_route
    assert spy_gate.calls == 0
