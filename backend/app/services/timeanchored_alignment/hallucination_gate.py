"""HallucinationGate：在文本/发音层前做双流健康门控。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from app.services.timeanchored_alignment.contracts import TextTruthPackage, TimeBasePackage


BOTH_SIDES_UNAVAILABLE_ERROR = "EDGE_SELECTOR_BOTH_SIDES_UNAVAILABLE"


class GateDecision(str, Enum):
    """门控决策类型。"""

    PROCEED = "proceed"
    USE_FAST = "use_fast"
    USE_SLOW = "use_slow"
    ERROR = "error"


@dataclass(frozen=True)
class HallucinationGateConfig:
    """门控阈值配置。"""

    slow_hallucination_risk_min: float = 0.85
    fast_blank_ratio_max: float = 0.7
    fast_avg_max_prob_min: float = 0.25
    fast_low_prob_ratio_max: float = 0.65


@dataclass(frozen=True)
class GateEvaluation:
    """门控结果。"""

    decision: GateDecision
    reason: str
    error_code: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


class HallucinationGate:
    """双流门控器。"""

    def __init__(self, *, config: HallucinationGateConfig | Mapping[str, float] | None = None) -> None:
        if config is None:
            self._config = HallucinationGateConfig()
        elif isinstance(config, HallucinationGateConfig):
            self._config = config
        else:
            self._config = HallucinationGateConfig(
                slow_hallucination_risk_min=float(
                    config.get("slow_hallucination_risk_min", HallucinationGateConfig.slow_hallucination_risk_min)
                ),
                fast_blank_ratio_max=float(config.get("fast_blank_ratio_max", HallucinationGateConfig.fast_blank_ratio_max)),
                fast_avg_max_prob_min=float(
                    config.get("fast_avg_max_prob_min", HallucinationGateConfig.fast_avg_max_prob_min)
                ),
                fast_low_prob_ratio_max=float(
                    config.get("fast_low_prob_ratio_max", HallucinationGateConfig.fast_low_prob_ratio_max)
                ),
            )

    def evaluate(self, *, time_base: TimeBasePackage, text_truth: TextTruthPackage) -> GateEvaluation:
        """评估双流健康状态，给出是否直接选边的决策。"""
        quality = time_base.quality
        metrics = {
            "slow_hallucination_risk": float(text_truth.quality.hallucination_risk),
            "fast_blank_ratio": float(quality.blank_ratio),
            "fast_avg_max_prob": float(quality.avg_max_prob),
            "fast_low_prob_ratio": float(quality.low_prob_ratio),
        }
        slow_hallucinated = self._is_slow_hallucinated(text_truth)
        fast_abnormal = self._is_fast_abnormal(time_base)

        if slow_hallucinated and fast_abnormal:
            return GateEvaluation(
                decision=GateDecision.ERROR,
                reason="both_abnormal",
                error_code=BOTH_SIDES_UNAVAILABLE_ERROR,
                metrics=metrics,
            )
        if slow_hallucinated:
            return GateEvaluation(decision=GateDecision.USE_FAST, reason="slow_hallucination", metrics=metrics)
        if fast_abnormal:
            return GateEvaluation(decision=GateDecision.USE_SLOW, reason="fast_abnormal", metrics=metrics)
        return GateEvaluation(decision=GateDecision.PROCEED, reason="proceed", metrics=metrics)

    def _is_slow_hallucinated(self, text_truth: TextTruthPackage) -> bool:
        if bool(text_truth.is_hallucination):
            return True
        return float(text_truth.quality.hallucination_risk) >= float(self._config.slow_hallucination_risk_min)

    def _is_fast_abnormal(self, time_base: TimeBasePackage) -> bool:
        units = time_base.word_units or time_base.raw_units
        if not units:
            return True
        quality = time_base.quality
        if float(quality.blank_ratio) >= float(self._config.fast_blank_ratio_max):
            return True
        if float(quality.avg_max_prob) <= float(self._config.fast_avg_max_prob_min):
            return True
        if float(quality.low_prob_ratio) >= float(self._config.fast_low_prob_ratio_max):
            return True
        return False


__all__ = [
    "BOTH_SIDES_UNAVAILABLE_ERROR",
    "GateDecision",
    "GateEvaluation",
    "HallucinationGate",
    "HallucinationGateConfig",
]
