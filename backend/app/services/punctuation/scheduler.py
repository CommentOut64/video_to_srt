"""
标点调度层数据结构与接口骨架。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from app.services.punctuation.base import PunctuationResult


class PunctuationMode(str, Enum):
    """标点调度模式。"""

    FAST_ONLY = "fast_only"
    DUAL = "dual"
    SMART_REVIEW = "smart_review"


@dataclass
class PunctuationTrigger:
    """触发慢流标点的原因。"""

    reason: str
    level: str = "info"
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PunctuationPolicy:
    """调度策略配置。"""

    mode: PunctuationMode = PunctuationMode.FAST_ONLY
    fast_confidence_threshold: float = 0.55
    alignment_coverage_threshold: float = 0.7
    arbiter_conflict_threshold: float = 0.35
    max_slow_retries: int = 1

    @property
    def is_slow_allowed(self) -> bool:
        return self.mode in {PunctuationMode.DUAL, PunctuationMode.SMART_REVIEW}

    @classmethod
    def from_config(cls, data: Optional[Dict[str, Any]]) -> "PunctuationPolicy":
        if not data:
            return cls()

        scheduler = data.get("scheduler", data)
        mode_value = str(scheduler.get("mode", cls.mode.value)).lower()
        try:
            mode = PunctuationMode(mode_value)
        except ValueError:
            mode = PunctuationMode.FAST_ONLY

        return cls(
            mode=mode,
            fast_confidence_threshold=float(
                scheduler.get("fast_confidence_threshold", cls.fast_confidence_threshold)
            ),
            alignment_coverage_threshold=float(
                scheduler.get("alignment_coverage_threshold", cls.alignment_coverage_threshold)
            ),
            arbiter_conflict_threshold=float(
                scheduler.get("arbiter_conflict_threshold", cls.arbiter_conflict_threshold)
            ),
            max_slow_retries=int(scheduler.get("max_slow_retries", cls.max_slow_retries)),
        )


@dataclass
class PunctuationDecision:
    """调度决策结果。"""

    is_slow_requested: bool
    reason: str
    level: str
    policy_mode: str
    triggers: List[PunctuationTrigger] = field(default_factory=list)

    @classmethod
    def skip(
        cls,
        reason: str,
        level: str,
        policy_mode: str,
        triggers: Optional[List[PunctuationTrigger]] = None,
    ) -> "PunctuationDecision":
        return cls(
            is_slow_requested=False,
            reason=reason,
            level=level,
            policy_mode=policy_mode,
            triggers=list(triggers or []),
        )

    @classmethod
    def request_slow(
        cls,
        reason: str,
        level: str,
        policy_mode: str,
        triggers: Optional[List[PunctuationTrigger]] = None,
    ) -> "PunctuationDecision":
        return cls(
            is_slow_requested=True,
            reason=reason,
            level=level,
            policy_mode=policy_mode,
            triggers=list(triggers or []),
        )

    def to_log_fields(self) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            "level": self.level,
            "policy_mode": self.policy_mode,
            "is_slow_requested": self.is_slow_requested,
            "trigger_reasons": [trigger.reason for trigger in self.triggers],
        }


class PunctuationScheduler:
    """标点调度层（Fast/Slow 决策骨架）。"""

    def __init__(
        self,
        policy: Optional[PunctuationPolicy] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._policy = policy or PunctuationPolicy()
        self._logger = logger or logging.getLogger(__name__)

    @property
    def policy(self) -> PunctuationPolicy:
        return self._policy

    def evaluate_fast(
        self,
        punct_result: PunctuationResult,
        sv_confidence: Optional[float] = None,
        alignment_coverage: Optional[float] = None,
        arbiter_conflict: Optional[float] = None,
    ) -> PunctuationDecision:
        triggers = self._collect_triggers(
            sv_confidence=sv_confidence,
            alignment_coverage=alignment_coverage,
            arbiter_conflict=arbiter_conflict,
            punct_confidence=punct_result.confidence,
        )

        if not self._policy.is_slow_allowed:
            decision = PunctuationDecision.skip(
                reason="fast_only",
                level="info",
                policy_mode=self._policy.mode.value,
                triggers=triggers,
            )
            self._log_decision(decision)
            return decision

        if triggers:
            decision = PunctuationDecision.request_slow(
                reason=triggers[0].reason,
                level=triggers[0].level,
                policy_mode=self._policy.mode.value,
                triggers=triggers,
            )
            self._log_decision(decision)
            return decision

        decision = PunctuationDecision.skip(
            reason="no_trigger",
            level="info",
            policy_mode=self._policy.mode.value,
        )
        self._log_decision(decision)
        return decision

    def evaluate_slow(
        self,
        triggers: Optional[List[PunctuationTrigger]] = None,
        reason: str = "slow_review",
    ) -> PunctuationDecision:
        if not self._policy.is_slow_allowed:
            decision = PunctuationDecision.skip(
                reason="fast_only",
                level="info",
                policy_mode=self._policy.mode.value,
            )
            self._log_decision(decision)
            return decision

        decision = PunctuationDecision.request_slow(
            reason=reason,
            level="info",
            policy_mode=self._policy.mode.value,
            triggers=triggers,
        )
        self._log_decision(decision)
        return decision

    def _collect_triggers(
        self,
        sv_confidence: Optional[float],
        alignment_coverage: Optional[float],
        arbiter_conflict: Optional[float],
        punct_confidence: Optional[float],
    ) -> List[PunctuationTrigger]:
        triggers: List[PunctuationTrigger] = []

        if sv_confidence is not None and sv_confidence < self._policy.fast_confidence_threshold:
            triggers.append(
                PunctuationTrigger(
                    reason="low_confidence",
                    level="warn",
                    detail={
                        "sv_confidence": sv_confidence,
                        "threshold": self._policy.fast_confidence_threshold,
                    },
                )
            )

        if alignment_coverage is not None and alignment_coverage < self._policy.alignment_coverage_threshold:
            triggers.append(
                PunctuationTrigger(
                    reason="low_alignment_coverage",
                    level="warn",
                    detail={
                        "alignment_coverage": alignment_coverage,
                        "threshold": self._policy.alignment_coverage_threshold,
                    },
                )
            )

        if arbiter_conflict is not None and arbiter_conflict > self._policy.arbiter_conflict_threshold:
            triggers.append(
                PunctuationTrigger(
                    reason="arbiter_conflict",
                    level="warn",
                    detail={
                        "arbiter_conflict": arbiter_conflict,
                        "threshold": self._policy.arbiter_conflict_threshold,
                    },
                )
            )

        if (
            punct_confidence is not None
            and punct_confidence < self._policy.fast_confidence_threshold
            and sv_confidence is None
        ):
            triggers.append(
                PunctuationTrigger(
                    reason="low_punctuation_confidence",
                    level="info",
                    detail={
                        "punct_confidence": punct_confidence,
                        "threshold": self._policy.fast_confidence_threshold,
                    },
                )
            )

        return triggers

    def _log_decision(self, decision: PunctuationDecision) -> None:
        self._logger.debug("标点调度决策: %s", decision.to_log_fields())
