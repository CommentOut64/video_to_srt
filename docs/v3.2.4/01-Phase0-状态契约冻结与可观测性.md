# Phase 0: 状态契约冻结与可观测性

> Type: Implementation Plan | Status: Ready
> Phase: 0 (P0 止血)
> 目标: 不改变业务行为，仅增强状态治理和可观测性
> 灰度开关: `STATE_MACHINE_GUARD_ENABLED`

## 1. 目标

将"看不见、说不清、复现难"的状态问题变成可追踪、可审计、可拦截的问题。

## 2. 新增文件

### 2.1 `backend/app/models/task_state_machine.py`（~120行）

```python
"""
任务状态机 - 校验层实现。

设计决策：
- 作为校验层嵌入，不替代现有流程
- 所有状态迁移必须经过此模块校验
- 非法迁移记录错误日志并拒绝（不抛异常，仅拒绝+告警）
"""
from enum import Enum
from typing import Optional, Dict, Set, Tuple
from dataclasses import dataclass
import logging
import time
import threading

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """任务状态枚举 - 单一事实源。"""
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


# 终态集合 - 不可逆
TERMINAL_STATES: Set[str] = {
    TaskStatus.FINISHED,
    TaskStatus.FAILED,
    TaskStatus.CANCELED,
    TaskStatus.FORCE_CANCELED,
    TaskStatus.REMOVED,
}

# 合法迁移白名单
VALID_TRANSITIONS: Dict[str, Set[str]] = {
    TaskStatus.CREATED: {TaskStatus.QUEUED},
    TaskStatus.QUEUED: {
        TaskStatus.PROCESSING,
        TaskStatus.PAUSED,      # 排队时暂停
        TaskStatus.CANCELED,    # 排队时取消
    },
    TaskStatus.PROCESSING: {
        TaskStatus.PAUSING,
        TaskStatus.CANCELING,
        TaskStatus.FINISHED,
        TaskStatus.FAILED,
    },
    TaskStatus.PAUSING: {
        TaskStatus.PAUSED,
        TaskStatus.CANCELING,   # 暂停过程中取消
        TaskStatus.FAILED,      # 暂停保存失败
    },
    TaskStatus.PAUSED: {
        TaskStatus.QUEUED,      # 恢复 -> 重新入队
        TaskStatus.CANCELING,   # 暂停中取消
        TaskStatus.CANCELED,    # 暂停中直接取消
    },
    TaskStatus.CANCELING: {
        TaskStatus.CANCELED,
        TaskStatus.FORCE_CANCELED,
    },
    # 终态 -> 仅允许 removed
    TaskStatus.CANCELED: {TaskStatus.REMOVED},
    TaskStatus.FORCE_CANCELED: {TaskStatus.REMOVED},
    TaskStatus.FINISHED: {TaskStatus.REMOVED},
    TaskStatus.FAILED: {TaskStatus.REMOVED},
    # removed 不可迁移
    TaskStatus.REMOVED: set(),
}

# 状态别名映射（统一历史遗留）
STATUS_ALIASES: Dict[str, str] = {
    "completed": TaskStatus.FINISHED,  # 兼容旧代码中的 completed
    "uploaded": TaskStatus.CREATED,    # 兼容上传态
}


@dataclass
class TransitionResult:
    """迁移结果。"""
    success: bool
    from_status: str
    to_status: str
    reason: str
    state_seq: int
    timestamp: float


class TaskStateGuard:
    """
    状态迁移守卫 - 校验 + 序号管理。

    职责：
    1. 校验迁移合法性
    2. 管理 state_seq 自增
    3. 记录迁移日志（成功和拒绝）
    """

    def __init__(self):
        self._seq_counter: Dict[str, int] = {}  # job_id -> 当前序号
        self._lock = threading.Lock()

    def normalize_status(self, status: str) -> str:
        """归一化状态名（处理别名）。"""
        return STATUS_ALIASES.get(status, status)

    def validate_transition(self, from_status: str, to_status: str) -> bool:
        """校验迁移是否合法。"""
        from_normalized = self.normalize_status(from_status)
        to_normalized = self.normalize_status(to_status)
        allowed = VALID_TRANSITIONS.get(from_normalized, set())
        return to_normalized in allowed

    def transition(
        self,
        job_id: str,
        current_status: str,
        target_status: str,
        reason: str = "",
    ) -> TransitionResult:
        """
        执行状态迁移（校验 + 序号递增）。

        返回 TransitionResult，调用方根据 success 决定是否执行实际赋值。
        不直接修改 job 对象，保持校验层的纯粹性。
        """
        from_normalized = self.normalize_status(current_status)
        to_normalized = self.normalize_status(target_status)
        now = time.time()

        with self._lock:
            seq = self._seq_counter.get(job_id, 0) + 1

            if not self.validate_transition(from_normalized, to_normalized):
                logger.error(
                    "[StateGuard] 非法状态迁移被拒绝: "
                    "job=%s, %s -> %s, reason=%s",
                    job_id, from_normalized, to_normalized, reason,
                )
                return TransitionResult(
                    success=False,
                    from_status=from_normalized,
                    to_status=to_normalized,
                    reason=f"非法迁移: {from_normalized} -> {to_normalized}",
                    state_seq=self._seq_counter.get(job_id, 0),
                    timestamp=now,
                )

            self._seq_counter[job_id] = seq

        logger.info(
            "[StateGuard] 状态迁移: job=%s, %s -> %s, seq=%d, reason=%s",
            job_id, from_normalized, to_normalized, seq, reason,
        )

        return TransitionResult(
            success=True,
            from_status=from_normalized,
            to_status=to_normalized,
            reason=reason,
            state_seq=seq,
            timestamp=now,
        )

    def get_seq(self, job_id: str) -> int:
        """获取当前序号。"""
        with self._lock:
            return self._seq_counter.get(job_id, 0)

    def cleanup(self, job_id: str) -> None:
        """清理已删除任务的序号。"""
        with self._lock:
            self._seq_counter.pop(job_id, None)


# 全局单例
_guard: Optional[TaskStateGuard] = None

def get_state_guard() -> TaskStateGuard:
    global _guard
    if _guard is None:
        _guard = TaskStateGuard()
    return _guard
```

## 3. 修改文件

### 3.1 `backend/app/models/job_models.py`

**改动点**: `JobState` 增加 `state_seq` 字段

```python
# 在 JobState 类中增加：
state_seq: int = 0  # 状态迁移序号，单调递增
```

**改动点**: 统一状态命名

```python
# 搜索所有 job.status == "completed" 或 job.status = "completed"
# 统一替换为 "finished"
# 保留 STATUS_ALIASES 做兼容读取
```

### 3.2 `backend/app/services/job_queue_service.py`

**改动策略**: 逐步替换所有 `job.status = "xxx"` 为经过状态机校验的迁移。

**改动模式**（所有涉及状态赋值的位置）:

```python
# 旧代码
job.status = "queued"

# 新代码
from app.models.task_state_machine import get_state_guard

guard = get_state_guard()
result = guard.transition(job.job_id, job.status, "queued", reason="queue_add")
if result.success:
    job.status = result.to_status
    job.state_seq = result.state_seq
else:
    logger.error("状态迁移被拒绝: %s", result.reason)
    # 视具体场景决定是否继续或返回错误
```

**需要替换的位置清单**:

| 代码位置 | 当前赋值 | 迁移 reason |
|----------|---------|-------------|
| `add_job` (~L225) | `job.status = "queued"` | `"queue_add"` |
| `pause_job` (~L260) | `job.status = "pausing"` | `"pause_request_running"` |
| `pause_job` (~L267) | `job.status = "paused"` | `"pause_request_queued"` |
| `resume_job` (~L350) | `job.status = "queued"` | `"resume"` |
| `cancel_job` (~L440) | `job.status = "canceled"` | `"cancel_queued"` |
| `cancel_job` (~L455) | `job.status = "canceling"` | `"cancel_running"` |
| Worker 循环 (~L620) | `job.status = "processing"` | `"worker_start"` |
| Worker 完成 (~L680) | `job.status = "finished"` | `"pipeline_complete"` |
| Worker CancelledException (~L690) | `job.status = "canceled"` | `"pipeline_canceled"` |
| Worker PausedException (~L695) | `job.status = "paused"` | `"pipeline_paused"` |
| Worker Exception (~L700) | `job.status = "failed"` | `"pipeline_error"` |
| `_force_cancel_timeout_job` (~L800) | `job.status = "force_canceled"` | `"cancel_timeout"` |

### 3.3 `backend/app/services/task_event_bus.py`

**改动点**: 事件附加 `state_seq`

```python
def emit_status_event(
    self,
    job_id: str,
    from_status: Optional[str],
    to_status: Optional[str],
    reason: Optional[str] = None,
    state_seq: int = 0,          # 新增
    payload: Optional[Dict[str, Any]] = None,
    conn: Optional[Any] = None
) -> None:
    # 落库事件时包含 state_seq
    self.state_repo.record_event(
        job_id=job_id,
        event_type="status_transition",
        from_status=from_status,
        to_status=to_status,
        reason=reason,
        state_seq=state_seq,       # 新增
        payload=payload,
        conn=conn,
    )

    # SSE 广播时包含 state_seq
    if self.sse_manager:
        self.sse_manager.broadcast_sync(
            "global",
            "task_event",
            {
                "job_id": job_id,
                "from_status": from_status,
                "to_status": to_status,
                "reason": reason,
                "state_seq": state_seq,  # 新增
                "payload": payload,
            },
        )
```

### 3.4 `backend/app/services/task_state_repository.py`

**改动点**: `task_events` 表增加 `state_seq` 列

```sql
ALTER TABLE task_events ADD COLUMN state_seq INTEGER DEFAULT 0;
```

在 `_ensure_schema` 中增加列迁移逻辑（SQLite 兼容）：

```python
def _ensure_schema(self) -> None:
    # ... 现有建表逻辑 ...

    # 增量迁移: 增加 state_seq 列
    try:
        conn.execute("ALTER TABLE task_events ADD COLUMN state_seq INTEGER DEFAULT 0")
    except Exception:
        pass  # 列已存在，忽略
```

### 3.5 `backend/app/services/job_lifecycle_service.py`

**改动点**: 状态写入经过状态机校验

所有涉及 `job.status` 赋值的位置，同 3.2 的替换模式。重点位置：

| 代码位置 | 场景 | 迁移 reason |
|----------|------|-------------|
| 重启纠偏 (~L409) | `processing -> paused` | `"restart_correction"` |
| 恢复入口 | `paused -> queued` | `"lifecycle_resume"` |

### 3.6 关键日志补齐

在 `job_queue_service.py` 的取消链路增加阶段耗时日志：

```python
# cancel_job 入口记录请求时间
logger.info(
    "[Lifecycle] 取消请求: job=%s, current_status=%s, delete_data=%s, "
    "request_time=%s",
    job_id, job.status, delete_data, time.time()
)

# canceling 状态记录进入时间
logger.info(
    "[Lifecycle] 进入 canceling: job=%s, enter_time=%s",
    job_id, time.time()
)

# 终态记录收敛时间
logger.info(
    "[Lifecycle] 取消终态达成: job=%s, status=%s, "
    "total_cancel_duration=%.1fs",
    job_id, job.status, time.time() - cancel_request_time
)
```

## 4. 验收标准

1. **合法性校验**: 任意状态直接赋值均经过 `TaskStateGuard.transition()` 校验
2. **序号追踪**: 每次状态迁移事件包含递增的 `state_seq`
3. **日志完整**: 取消链路可从日志还原完整状态迁移链（请求时间 -> canceling -> 终态）
4. **零业务变更**: 所有迁移校验仅为"校验+日志"，不改变现有业务流程
5. **回滚安全**: 关闭 `STATE_MACHINE_GUARD_ENABLED` 后回退到直接赋值

## 5. 测试要点

```python
# test_task_state_machine.py

def test_valid_transitions():
    """所有白名单内迁移应成功。"""
    guard = TaskStateGuard()
    for from_status, allowed_targets in VALID_TRANSITIONS.items():
        for target in allowed_targets:
            result = guard.transition("test-job", from_status, target)
            assert result.success

def test_invalid_transitions():
    """所有白名单外迁移应被拒绝。"""
    guard = TaskStateGuard()
    # 例如: finished -> processing 应被拒绝
    result = guard.transition("test-job", "finished", "processing")
    assert not result.success

def test_terminal_state_absorption():
    """终态不可逆（除 removed）。"""
    guard = TaskStateGuard()
    for terminal in TERMINAL_STATES - {TaskStatus.REMOVED}:
        for target in TaskStatus:
            if target == TaskStatus.REMOVED:
                continue
            result = guard.transition("test-job", terminal, target)
            assert not result.success

def test_state_seq_monotonic():
    """state_seq 严格递增。"""
    guard = TaskStateGuard()
    r1 = guard.transition("job-1", "created", "queued")
    r2 = guard.transition("job-1", "queued", "processing")
    assert r2.state_seq > r1.state_seq

def test_status_alias():
    """别名 completed -> finished 正确映射。"""
    guard = TaskStateGuard()
    assert guard.normalize_status("completed") == "finished"
```
