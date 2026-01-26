"""
任务事件总线

使用 Event Bus 模式统一落库与广播，保证事件可追踪。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.services.task_state_repository import TaskStateRepository

try:
    from app.services.sse_service import SSEManager
except Exception:  # pragma: no cover - 避免测试环境未加载 FastAPI 时失败
    SSEManager = None  # type: ignore


class TaskEventBus:
    """
    任务事件总线（Event Bus 模式）

    负责记录状态迁移事件，并可选广播到 SSE。
    """

    def __init__(
        self,
        state_repo: TaskStateRepository,
        sse_manager: Optional["SSEManager"] = None,
        logger: Optional[logging.Logger] = None
    ) -> None:
        self.state_repo = state_repo
        self.sse_manager = sse_manager
        self.logger = logger or logging.getLogger(__name__)

    def emit_status_event(
        self,
        job_id: str,
        from_status: Optional[str],
        to_status: Optional[str],
        reason: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        conn: Optional[Any] = None
    ) -> None:
        event_payload = payload or {}
        self.state_repo.record_event(
            job_id=job_id,
            event_type="status_transition",
            from_status=from_status,
            to_status=to_status,
            reason=reason,
            payload=event_payload,
            conn=conn,
        )
        if self.sse_manager:
            try:
                self.sse_manager.broadcast_sync(
                    "global",
                    "task_event",
                    {
                        "job_id": job_id,
                        "from_status": from_status,
                        "to_status": to_status,
                        "reason": reason,
                        "payload": event_payload,
                    },
                )
            except Exception as exc:
                self.logger.debug(f"任务事件广播失败: {exc}")
