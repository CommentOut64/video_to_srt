"""EdgeSelector：文本/发音层后的受约束选边兜底。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from app.services.timeanchored_alignment.contracts import (
    AlignmentItem,
    AlignmentMetrics,
    FinalAlignmentResult,
    TextTruthPackage,
    TimeBasePackage,
)
from app.services.timeanchored_alignment.hallucination_gate import (
    BOTH_SIDES_UNAVAILABLE_ERROR,
    GateDecision,
    HallucinationGate,
)


@dataclass(frozen=True)
class FailedSpan:
    """失败片段索引范围（闭区间）。"""

    start: int
    end: int

    def __post_init__(self) -> None:
        if int(self.start) < 0:
            raise ValueError("FailedSpan.start 必须 >= 0")
        if int(self.end) < int(self.start):
            raise ValueError("FailedSpan.end 必须 >= start")


@dataclass(frozen=True)
class EdgeSelectorConfig:
    """选边层配置。"""

    slow_hallucination_risk_max: float = 0.85
    fallback_duration_sec: float = 0.12


class EdgeSelector:
    """选边兜底层。"""

    def __init__(
        self,
        *,
        hallucination_gate: HallucinationGate | None = None,
        config: EdgeSelectorConfig | Mapping[str, float] | None = None,
    ) -> None:
        self._gate = hallucination_gate or HallucinationGate()
        if config is None:
            self._config = EdgeSelectorConfig()
        elif isinstance(config, EdgeSelectorConfig):
            self._config = config
        else:
            self._config = EdgeSelectorConfig(
                slow_hallucination_risk_max=float(
                    config.get("slow_hallucination_risk_max", EdgeSelectorConfig.slow_hallucination_risk_max)
                ),
                fallback_duration_sec=float(config.get("fallback_duration_sec", EdgeSelectorConfig.fallback_duration_sec)),
            )

    def select(
        self,
        *,
        time_base: TimeBasePackage,
        text_truth: TextTruthPackage,
        failed_spans: Sequence[FailedSpan] = (),
        edge_selection_mode: str = "auto",
    ) -> FinalAlignmentResult:
        mode = self._normalize_mode(edge_selection_mode)
        if mode == "force_fast":
            return self.select_fast(time_base=time_base)
        if mode == "force_slow":
            return self.select_slow(time_base=time_base, text_truth=text_truth)

        gate = self._gate.evaluate(time_base=time_base, text_truth=text_truth)
        if gate.decision is GateDecision.USE_FAST:
            return self.select_fast(time_base=time_base)
        if gate.decision is GateDecision.USE_SLOW:
            return self.select_slow(time_base=time_base, text_truth=text_truth)
        if gate.decision is GateDecision.ERROR:
            return self._error_result(gate.error_code or BOTH_SIDES_UNAVAILABLE_ERROR)

        if not failed_spans:
            return self.select_slow(time_base=time_base, text_truth=text_truth)
        return self._select_mixed(time_base=time_base, text_truth=text_truth, failed_spans=failed_spans)

    def select_fast(self, *, time_base: TimeBasePackage) -> FinalAlignmentResult:
        units = tuple(time_base.word_units or time_base.raw_units)
        if not units:
            return self._error_result("EDGE_SELECTOR_FAST_UNAVAILABLE")

        items = tuple(
            AlignmentItem(
                text=str(unit.text or ""),
                start=float(unit.start),
                end=float(unit.end),
                status="direct",
                source="edge_selector_fast",
                confidence=float(unit.confidence),
                reason="selected_fast",
            )
            for unit in units
        )
        return FinalAlignmentResult(
            items=items,
            route="fast",
            metrics=self._build_metrics(items=items, route_confidence=1.0),
        )

    def select_slow(self, *, time_base: TimeBasePackage, text_truth: TextTruthPackage) -> FinalAlignmentResult:
        if not text_truth.units:
            return self._error_result("EDGE_SELECTOR_SLOW_UNAVAILABLE")
        if not self._is_slow_trusted(text_truth):
            return self._error_result("EDGE_SELECTOR_SLOW_UNTRUSTED")
        if not self._has_any_timestamp(text_truth):
            return self._error_result("EDGE_SELECTOR_SLOW_NO_TIMESTAMPS")

        start_ref, end_ref = self._resolve_reference_range(time_base=time_base, text_truth=text_truth)
        unit_duration = max(
            float(self._config.fallback_duration_sec),
            (end_ref - start_ref) / float(max(len(text_truth.units), 1)),
        )
        cursor = float(start_ref)
        items: list[AlignmentItem] = []
        for index, unit in enumerate(text_truth.units):
            text = str(unit.text or "")
            if unit.start is not None and unit.end is not None:
                start = float(unit.start)
                end = float(unit.end)
                status = "estimated"
                reason = "slow_timestamp_anchor"
                cursor = max(cursor, end)
            else:
                start = max(cursor, float(start_ref) + float(index) * unit_duration)
                end = start + unit_duration
                status = "interpolated"
                reason = "ratio_interpolated"
                cursor = end

            items.append(
                AlignmentItem(
                    text=text,
                    start=start,
                    end=end,
                    status=status,
                    source="edge_selector_slow",
                    confidence=float(unit.confidence),
                    reason=reason,
                )
            )

        tuple_items = tuple(items)
        return FinalAlignmentResult(
            items=tuple_items,
            route="slow",
            metrics=self._build_metrics(items=tuple_items, route_confidence=0.8),
        )

    def _select_mixed(
        self,
        *,
        time_base: TimeBasePackage,
        text_truth: TextTruthPackage,
        failed_spans: Sequence[FailedSpan],
    ) -> FinalAlignmentResult:
        slow_result = self.select_slow(time_base=time_base, text_truth=text_truth)
        if slow_result.route == "error":
            fast_result = self.select_fast(time_base=time_base)
            if fast_result.route == "error":
                return self._error_result(BOTH_SIDES_UNAVAILABLE_ERROR)
            return fast_result

        fast_result = self.select_fast(time_base=time_base)
        if fast_result.route == "error":
            return slow_result

        mixed_items = list(slow_result.items)
        replace_indices = self._flatten_spans(failed_spans, max_index=len(mixed_items) - 1)
        if not replace_indices:
            return slow_result
        for idx in replace_indices:
            if idx >= len(mixed_items) or idx >= len(fast_result.items):
                continue
            fast_item = fast_result.items[idx]
            mixed_items[idx] = AlignmentItem(
                text=fast_item.text,
                start=fast_item.start,
                end=fast_item.end,
                status="estimated",
                source="edge_selector_mixed",
                confidence=fast_item.confidence,
                reason="failed_span_fallback_to_fast",
            )
        tuple_items = tuple(mixed_items)
        return FinalAlignmentResult(
            items=tuple_items,
            route="mixed",
            metrics=self._build_metrics(items=tuple_items, route_confidence=0.75),
        )

    @staticmethod
    def _normalize_mode(mode: str | None) -> str:
        raw = str(mode or "auto").strip().lower()
        if raw in {"force_fast", "force_slow"}:
            return raw
        return "auto"

    def _is_slow_trusted(self, text_truth: TextTruthPackage) -> bool:
        if bool(text_truth.is_hallucination):
            return False
        return float(text_truth.quality.hallucination_risk) <= float(self._config.slow_hallucination_risk_max)

    @staticmethod
    def _has_any_timestamp(text_truth: TextTruthPackage) -> bool:
        for unit in text_truth.units:
            if unit.start is None or unit.end is None:
                continue
            return True
        return False

    @staticmethod
    def _resolve_reference_range(*, time_base: TimeBasePackage, text_truth: TextTruthPackage) -> tuple[float, float]:
        fast_units = tuple(time_base.word_units or time_base.raw_units)
        if fast_units:
            start = float(fast_units[0].start)
            end = float(fast_units[-1].end)
            if end > start:
                return start, end

        anchored = [unit for unit in text_truth.units if unit.start is not None and unit.end is not None]
        if anchored:
            start = float(anchored[0].start)
            end = float(anchored[-1].end)
            if end > start:
                return start, end

        span = float(max(len(text_truth.units), 1)) * 0.12
        return 0.0, span

    @staticmethod
    def _flatten_spans(spans: Sequence[FailedSpan], *, max_index: int) -> set[int]:
        if max_index < 0:
            return set()
        indices: set[int] = set()
        for span in spans:
            start = max(0, int(span.start))
            end = min(max_index, int(span.end))
            for idx in range(start, end + 1):
                indices.add(idx)
        return indices

    @staticmethod
    def _build_metrics(*, items: Sequence[AlignmentItem], route_confidence: float) -> AlignmentMetrics:
        if not items:
            return AlignmentMetrics(coverage=0.0, duration_ratio=1.0, failed_count=0, route_confidence=0.0)
        failed_count = sum(1 for item in items if item.status == "failed")
        coverage = float(len(items) - failed_count) / float(max(len(items), 1))
        duration = float(items[-1].end) - float(items[0].start)
        duration_ratio = max(duration, 1e-3)
        return AlignmentMetrics(
            coverage=max(0.0, min(1.0, coverage)),
            duration_ratio=duration_ratio,
            failed_count=failed_count,
            route_confidence=max(0.0, min(1.0, float(route_confidence))),
        )

    @staticmethod
    def _error_result(error_code: str) -> FinalAlignmentResult:
        return FinalAlignmentResult(
            items=tuple(),
            route="error",
            metrics=AlignmentMetrics(coverage=0.0, duration_ratio=1.0, failed_count=1, route_confidence=0.0),
            error_code=error_code,
        )


__all__ = [
    "EdgeSelector",
    "EdgeSelectorConfig",
    "FailedSpan",
]
