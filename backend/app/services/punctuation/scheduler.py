"""
标点调度器（快慢流触发判定）。
V3.2.0+dev.20260129.02
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.services.punctuation.base import PunctuationResult
from app.services.punctuation.config import get_punctuation_config
from app.services.text_pipeline_config import PunctuationRuntimeOverrides


class PunctuationMode(str, Enum):
    """调度模式枚举。"""

    FAST_ONLY = "fast_only"
    DUAL = "dual"
    SMART_REVIEW = "smart_review"


@dataclass
class PunctuationPolicy:
    """调度策略配置。"""

    mode: PunctuationMode = PunctuationMode.FAST_ONLY
    fast_confidence_threshold: float = 0.55
    alignment_coverage_threshold: float = 0.7
    arbiter_conflict_threshold: float = 0.35
    max_slow_retries: int = 1

    @classmethod
    def from_config(cls, config: dict) -> "PunctuationPolicy":
        scheduler = config.get("scheduler", {}) if isinstance(config, dict) else {}
        mode_raw = str(scheduler.get("mode", cls.mode.value)).lower()
        try:
            mode = PunctuationMode(mode_raw)
        except ValueError:
            mode = cls.mode
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
class PunctuationScheduleDecision:
    """调度决策结果。"""

    is_slow_requested: bool
    reason: str = ""


class PunctuationScheduler:
    """根据快流结果决定是否触发慢流标点补跑。"""

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
        fast_result: Optional[PunctuationResult],
        sv_confidence: Optional[float] = None,
    ) -> PunctuationScheduleDecision:
        """基于快流标点结果做初步决策。"""
        if self._policy.mode == PunctuationMode.FAST_ONLY:
            return PunctuationScheduleDecision(False, reason="fast_only")

        if not fast_result:
            return PunctuationScheduleDecision(True, reason="fast_result_missing")

        threshold = self._policy.fast_confidence_threshold
        if fast_result.confidence < threshold:
            self._logger.debug(
                "快流标点置信度不足: %.2f < %.2f",
                fast_result.confidence,
                threshold,
            )
            return PunctuationScheduleDecision(True, reason="low_confidence")

        if sv_confidence is not None and sv_confidence < threshold:
            self._logger.debug(
                "SenseVoice 置信度不足: %.2f < %.2f",
                sv_confidence,
                threshold,
            )
            return PunctuationScheduleDecision(True, reason="low_confidence")

        return PunctuationScheduleDecision(False, reason="fast_confidence_ok")

    def allow_retry(self, retry_count: int) -> bool:
        """检查是否允许慢流补跑重试。"""
        return retry_count < self._policy.max_slow_retries


_scheduler: Optional[PunctuationScheduler] = None


def get_punctuation_scheduler() -> PunctuationScheduler:
    """获取调度器单例。"""
    global _scheduler
    config = get_punctuation_config()
    policy = PunctuationPolicy.from_config(config)

    # V3.2.0+dev.20260204.09: 运行参数显式覆盖（仅覆盖 override 中出现的键）
    overrides = PunctuationRuntimeOverrides.from_runtime()
    sched = overrides.scheduler
    if sched.mode is not None:
        try:
            policy.mode = PunctuationMode(str(sched.mode))
        except ValueError:
            pass
    if sched.fast_confidence_threshold is not None:
        policy.fast_confidence_threshold = float(sched.fast_confidence_threshold)
    if sched.alignment_coverage_threshold is not None:
        policy.alignment_coverage_threshold = float(sched.alignment_coverage_threshold)
    if sched.arbiter_conflict_threshold is not None:
        policy.arbiter_conflict_threshold = float(sched.arbiter_conflict_threshold)
    if sched.max_slow_retries is not None:
        policy.max_slow_retries = int(sched.max_slow_retries)

    if _scheduler is None:
        _scheduler = PunctuationScheduler(policy=policy)
    else:
        # 动态更新策略，避免热更新时仍使用旧阈值
        _scheduler._policy = policy  # type: ignore[attr-defined]
    return _scheduler
