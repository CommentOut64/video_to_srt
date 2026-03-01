"""
TurnGroup Flush 策略。

设计模式：Strategy Pattern。
原因：把 flush 触发条件从组批器中解耦，便于独立测试和后续参数扩展。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FlushPolicyConfig:
    """Flush 策略配置。"""

    min_audio_sec: float = 6.0
    min_token_count: int = 40
    max_wait_sec: float = 5.0
    tail_idle_sec: float = 1.0
    long_pause_cut_sec: float = 1.8


@dataclass(frozen=True)
class FlushDecision:
    """Flush 判定结果。"""

    should_flush: bool
    reason: str = ""


class FlushPolicy:
    """TurnGroup flush 策略实现。"""

    def __init__(self, config: Optional[FlushPolicyConfig] = None) -> None:
        self.config = config or FlushPolicyConfig()

    def evaluate_before_append(
        self,
        *,
        is_language_changed: bool,
        pause_gap_sec: float,
    ) -> FlushDecision:
        """追加新片段前判定是否必须先出包。"""
        if is_language_changed:
            return FlushDecision(True, "language_change")
        if pause_gap_sec >= self.config.long_pause_cut_sec:
            return FlushDecision(True, "long_pause_cut")
        return FlushDecision(False)

    def evaluate_after_append(
        self,
        *,
        audio_sec: float,
        token_count: int,
        wait_sec: float,
    ) -> FlushDecision:
        """追加新片段后判定是否可出包。"""
        if audio_sec >= self.config.min_audio_sec:
            return FlushDecision(True, "min_audio_sec")
        if token_count >= self.config.min_token_count:
            return FlushDecision(True, "min_token_count")
        if wait_sec >= self.config.max_wait_sec:
            return FlushDecision(True, "max_wait_sec")
        return FlushDecision(False)

    def evaluate_idle(self, *, idle_sec: float) -> FlushDecision:
        """空闲状态下判定是否强制出包。"""
        if idle_sec >= self.config.tail_idle_sec:
            return FlushDecision(True, "tail_idle")
        return FlushDecision(False)

