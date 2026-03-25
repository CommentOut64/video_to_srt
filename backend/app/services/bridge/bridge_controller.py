"""Bridge 控制器（Phase 3 精简版）。

设计说明：
- 当前默认慢流主链已切换为 `SlowWindowBuilder + ReadySlowWindow`。
- 当前类仅保留轻量控制面能力（空闲 flush 判定与仲裁结果记录），避免上层依赖断裂。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from app.core.logging import resolve_loguru_logger
from app.services.bridge.config import BridgeConfig
from app.services.punctuation.semantic_buffer import PunctuationDecision


class BridgeController:
    """Window-first 主链下的 Bridge 轻量控制器。"""

    def __init__(
        self,
        config: Optional[BridgeConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config or BridgeConfig()
        self._logger = resolve_loguru_logger(logger, __name__, layer="L0")
        self._last_input_time = time.time()
        self._last_arbitration: Optional[Any] = None
        self._arbiter_feedback: Optional[PunctuationDecision] = None

    def is_backpressure_active(self) -> bool:
        """Phase 3 起不再维护 Bridge 内部批次队列。"""
        return False

    async def wait_for_capacity(self) -> None:
        """兼容旧接口：无内部队列时立即返回。"""
        return None

    def get_queue_size(self) -> int:
        """兼容旧接口：无内部队列时固定返回 0。"""
        return 0

    def mark_input_activity(self, now: Optional[float] = None) -> None:
        """记录最近输入时间，用于空闲 flush 判定。"""
        self._last_input_time = float(now or time.time())

    def should_flush_on_idle(self, queue_inter_size: int, now: Optional[float] = None) -> bool:
        """根据等待时长判定是否触发空闲 flush。"""
        if not self._config.force_flush_on_gpu_idle:
            return False
        if queue_inter_size > 0:
            return False
        current_time = float(now or time.time())
        waited_ms = max(0.0, (current_time - self._last_input_time) * 1000.0)
        return waited_ms >= float(self._config.dynamic_batch_max_wait_ms)

    def update_arbiter_feedback(self, decision: Optional[PunctuationDecision]) -> None:
        """记录仲裁反馈（保留给后续策略扩展）。"""
        self._arbiter_feedback = decision

    def record_arbitration_result(self, result: Any) -> None:
        """记录最近一次仲裁结果。"""
        self._last_arbitration = result

    def get_last_arbitration_result(self) -> Optional[Any]:
        """返回最近一次仲裁结果。"""
        return self._last_arbitration
