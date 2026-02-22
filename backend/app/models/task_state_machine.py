"""
任务状态机 - 校验层实现。

设计说明：
- 采用 Guard 校验模式，不替代现有业务流程；
- 仅负责迁移合法性校验与 `state_seq` 序号管理；
- 非法迁移仅拒绝并记录日志，不抛异常。
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """任务状态枚举（单一事实源）。"""

    CREATED = "created"
    QUEUED = "queued"
    PROCESSING = "processing"
    PAUSING = "pausing"
    PAUSED = "paused"
    CANCELING = "canceling"
    CANCELED = "canceled"
    FORCE_CANCELED = "force_canceled"
    FINISHED = "finished"
    FAILED = "failed"
    REMOVED = "removed"


# 终态集合（不可逆）
TERMINAL_STATES: Set[str] = {
    TaskStatus.CANCELED.value,
    TaskStatus.FORCE_CANCELED.value,
    TaskStatus.FINISHED.value,
    TaskStatus.FAILED.value,
    TaskStatus.REMOVED.value,
}

# 合法迁移白名单
# 注：
# - 为兼容重启纠偏与历史状态，允许 created/processing/canceling -> paused；
# - 仅允许 removed 作为最终删除状态。
VALID_TRANSITIONS: Dict[str, Set[str]] = {
    TaskStatus.CREATED.value: {
        TaskStatus.QUEUED.value,
        TaskStatus.PAUSED.value,
        TaskStatus.CANCELED.value,
    },
    TaskStatus.QUEUED.value: {
        TaskStatus.PROCESSING.value,
        TaskStatus.PAUSED.value,
        TaskStatus.CANCELED.value,
        TaskStatus.REMOVED.value,
    },
    TaskStatus.PROCESSING.value: {
        TaskStatus.PAUSING.value,
        TaskStatus.PAUSED.value,
        TaskStatus.CANCELING.value,
        TaskStatus.CANCELED.value,
        TaskStatus.FINISHED.value,
        TaskStatus.FAILED.value,
    },
    TaskStatus.PAUSING.value: {
        TaskStatus.PROCESSING.value,
        TaskStatus.PAUSED.value,
        TaskStatus.CANCELING.value,
        TaskStatus.CANCELED.value,
        TaskStatus.FAILED.value,
    },
    TaskStatus.PAUSED.value: {
        TaskStatus.QUEUED.value,
        TaskStatus.CANCELING.value,
        TaskStatus.CANCELED.value,
        TaskStatus.REMOVED.value,
    },
    TaskStatus.CANCELING.value: {
        TaskStatus.PAUSED.value,
        TaskStatus.CANCELED.value,
        TaskStatus.FORCE_CANCELED.value,
    },
    TaskStatus.CANCELED.value: {TaskStatus.REMOVED.value},
    TaskStatus.FORCE_CANCELED.value: {TaskStatus.REMOVED.value},
    TaskStatus.FINISHED.value: {TaskStatus.REMOVED.value},
    TaskStatus.FAILED.value: {TaskStatus.REMOVED.value},
    TaskStatus.REMOVED.value: set(),
}

# 历史状态别名映射（读取兼容）
STATUS_ALIASES: Dict[str, str] = {
    "completed": TaskStatus.FINISHED.value,
    "uploaded": TaskStatus.CREATED.value,
    "running": TaskStatus.PROCESSING.value,
    "transcribing": TaskStatus.PROCESSING.value,
    # 兼容旧冻结语义事件：统一映射到 canceled
    "freezing": TaskStatus.CANCELED.value,
    "frozen": TaskStatus.CANCELED.value,
}


@dataclass(frozen=True)
class TransitionResult:
    """状态迁移结果。"""

    success: bool
    from_status: str
    to_status: str
    reason: str
    state_seq: int
    timestamp: float


class TaskStateGuard:
    """
    状态迁移守卫（Guard 模式）。

    职责：
    1. 校验迁移合法性；
    2. 维护任务 `state_seq` 单调递增；
    3. 输出可审计日志。
    """

    def __init__(self) -> None:
        self._seq_counter: Dict[str, int] = {}
        self._lock = threading.Lock()

    def normalize_status(self, status: str) -> str:
        """归一化状态名（处理历史别名）。"""
        if status is None:
            return ""
        normalized_status = str(status).strip()
        return STATUS_ALIASES.get(normalized_status, normalized_status)

    def validate_transition(self, from_status: str, to_status: str) -> bool:
        """校验迁移是否合法。"""
        from_normalized = self.normalize_status(from_status)
        to_normalized = self.normalize_status(to_status)
        allowed_targets = VALID_TRANSITIONS.get(from_normalized, set())
        return to_normalized in allowed_targets

    def sync_seq(self, job_id: str, state_seq: int) -> None:
        """将守卫内序号与外部持久化序号对齐（取较大值）。"""
        safe_seq = max(0, int(state_seq or 0))
        with self._lock:
            current_seq = self._seq_counter.get(job_id, 0)
            if safe_seq > current_seq:
                self._seq_counter[job_id] = safe_seq

    def transition(
        self,
        job_id: str,
        current_status: str,
        target_status: str,
        reason: str = "",
    ) -> TransitionResult:
        """
        执行状态迁移（校验 + 序号）。

        返回：
        - 成功：`success=True`，并返回新的 `state_seq`；
        - 失败：`success=False`，保留原序号。
        """
        from_normalized = self.normalize_status(current_status)
        to_normalized = self.normalize_status(target_status)
        now = time.time()

        with self._lock:
            current_seq = self._seq_counter.get(job_id, 0)
            is_valid_transition = self.validate_transition(from_normalized, to_normalized)
            if not is_valid_transition:
                logger.error(
                    "[StateGuard] 非法状态迁移被拒绝: job=%s, %s -> %s, reason=%s",
                    job_id,
                    from_normalized,
                    to_normalized,
                    reason,
                )
                return TransitionResult(
                    success=False,
                    from_status=from_normalized,
                    to_status=to_normalized,
                    reason=f"非法迁移: {from_normalized} -> {to_normalized}",
                    state_seq=current_seq,
                    timestamp=now,
                )

            next_seq = current_seq + 1
            self._seq_counter[job_id] = next_seq

        logger.info(
            "[StateGuard] 状态迁移: job=%s, %s -> %s, seq=%d, reason=%s",
            job_id,
            from_normalized,
            to_normalized,
            next_seq,
            reason,
        )
        return TransitionResult(
            success=True,
            from_status=from_normalized,
            to_status=to_normalized,
            reason=reason,
            state_seq=next_seq,
            timestamp=now,
        )

    def get_seq(self, job_id: str) -> int:
        """获取任务当前序号。"""
        with self._lock:
            return self._seq_counter.get(job_id, 0)

    def cleanup(self, job_id: str) -> None:
        """任务删除后清理序号缓存。"""
        with self._lock:
            self._seq_counter.pop(job_id, None)


_guard: Optional[TaskStateGuard] = None


def get_state_guard() -> TaskStateGuard:
    """获取状态守卫单例。"""
    global _guard
    if _guard is None:
        _guard = TaskStateGuard()
    return _guard
