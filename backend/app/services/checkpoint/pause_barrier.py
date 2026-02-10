"""
PauseBarrier - 单元边界暂停/取消屏障。

设计模式：门面模式（Facade Pattern）
原因：统一协调 CancellationToken 与 RuntimeCheckpointService，
将 pause/cancel 生效点约束到单元边界。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from app.services.checkpoint.runtime_checkpoint_service import RuntimeCheckpointService
from app.utils.cancellation_token import CancellationToken


@dataclass(frozen=True)
class PauseBarrierDecision:
    """屏障判定结果。"""

    should_stop: bool
    stop_reason: str | None = None


class PauseBarrier:
    """单元边界控制屏障。"""

    def __init__(
        self,
        token: Optional[CancellationToken],
        runtime_checkpoint_service: Optional[RuntimeCheckpointService],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.token = token
        self.runtime_checkpoint_service = runtime_checkpoint_service
        self.logger = logger or logging.getLogger(__name__)

    def on_unit_start(self, *, stage: str, unit_id: str, payload: dict[str, Any] | None = None) -> None:
        """进入单元前写入 started 日志。"""
        if self.runtime_checkpoint_service:
            self.runtime_checkpoint_service.record_unit_started(
                stage=stage,
                unit_id=unit_id,
                payload=payload,
            )

    def on_unit_end(self, *, stage: str, unit_id: str, payload: dict[str, Any] | None = None) -> PauseBarrierDecision:
        """单元结束后提交并检查是否应停止。"""
        if self.runtime_checkpoint_service:
            self.runtime_checkpoint_service.record_unit_committed(
                stage=stage,
                unit_id=unit_id,
                payload=payload,
            )

        if self.token and self.token.is_canceled:
            return PauseBarrierDecision(should_stop=True, stop_reason="canceled")
        if self.token and self.token.is_paused:
            return PauseBarrierDecision(should_stop=True, stop_reason="paused")

        if self.runtime_checkpoint_service and self.runtime_checkpoint_service.is_cancel_requested():
            return PauseBarrierDecision(should_stop=True, stop_reason="canceled")
        if self.runtime_checkpoint_service and self.runtime_checkpoint_service.is_pause_requested():
            return PauseBarrierDecision(should_stop=True, stop_reason="paused")

        return PauseBarrierDecision(should_stop=False)

