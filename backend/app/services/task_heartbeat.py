"""
任务心跳服务

提供租约与心跳的统一入口，避免多处重复实现。
"""
from __future__ import annotations

import logging
from typing import Optional

from app.services.task_state_repository import TaskStateRepository


class TaskHeartbeatService:
    """任务心跳服务"""

    def __init__(
        self,
        state_repo: TaskStateRepository,
        logger: Optional[logging.Logger] = None
    ) -> None:
        self.state_repo = state_repo
        self.logger = logger or logging.getLogger(__name__)

    def acquire_lease(self, job_id: str, lease_owner: str, ttl_seconds: float) -> bool:
        return self.state_repo.acquire_lease(job_id, lease_owner, ttl_seconds)

    def refresh(self, job_id: str, lease_owner: str, ttl_seconds: float) -> bool:
        return self.state_repo.refresh_heartbeat(job_id, lease_owner, ttl_seconds)

    def release(self, job_id: str, lease_owner: str) -> None:
        self.state_repo.release_lease(job_id, lease_owner)

    def list_expired_leases(self) -> list[str]:
        return self.state_repo.list_expired_leases()

    def list_heartbeat_timeouts(self, timeout_seconds: float) -> list[str]:
        return self.state_repo.list_heartbeat_timeouts(timeout_seconds)
