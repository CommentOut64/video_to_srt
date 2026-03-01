# Phase 3: 执行模型风险闸门

> Type: Implementation Plan | Status: Ready
> Phase: 3 (P1 安全网)
> 目标: 修正超时语义、加孤儿执行观测、禁止误判空闲触发重 GPU 任务
> 灰度开关: `RUNNER_GATE_ENABLED`
> 注意: 本轮**不做进程级隔离**，仅做风险闸门

## 1. 策略选择

Codex 评估建议：本轮不做 Phase 3 完整的进程级 JobRunner 隔离（改动面过大），改为"P2.5 风险闸门"方案。

**做什么**:
1. 修正超时放行语义：增加孤儿执行观测
2. 禁止误判空闲：`running_job_id = None` 不等于"GPU 空闲"
3. 为未来进程隔离预留接口

**不做什么**:
1. 不引入独立 JobRunner 进程
2. 不改变 Worker 线程模型
3. 不做跨进程通信

## 2. 修改文件

### 2.1 `backend/app/services/job_queue_service.py` - 超时放行语义修正

**当前问题**: `_force_cancel_timeout_job` 清除 `running_job_id = None` 后，队列认为 GPU 空闲，启动下一个任务。但旧流水线可能仍在后台运行占用 GPU。

**改动 1**: 增加孤儿执行跟踪

```python
class JobQueueService:
    def __init__(self):
        # ... 现有字段 ...

        # Phase 3: 孤儿执行跟踪
        self._orphan_executions: Dict[str, float] = {}  # job_id -> 放行时间
        self._gpu_busy_override: bool = False  # GPU 忙碌覆盖标志
```

**改动 2**: `_force_cancel_timeout_job` 增加孤儿标记

```python
def _force_cancel_timeout_job(self, job_id: str, elapsed: float):
    # ... 现有逻辑 ...

    # Phase 3: 标记为孤儿执行
    with self.lock:
        self._orphan_executions[job_id] = time.time()
        # 不立即清除 running_job_id，而是设置 GPU 忙碌覆盖
        if RUNNER_GATE_ENABLED:
            self._gpu_busy_override = True
            logger.warning(
                "[Phase3] GPU 忙碌覆盖已启用: 孤儿任务 %s 可能仍在运行",
                job_id,
            )
```

**改动 3**: 队列取任务时检查 GPU 忙碌覆盖

```python
def _try_start_next_job(self):
    """尝试启动下一个任务。"""
    with self.lock:
        # Phase 3: GPU 忙碌覆盖检查
        if RUNNER_GATE_ENABLED and self._gpu_busy_override:
            # 检查孤儿执行是否已退出
            if self._check_orphan_cleanup():
                self._gpu_busy_override = False
                logger.info("[Phase3] 孤儿执行已清理，GPU 解除忙碌覆盖")
            else:
                logger.warning("[Phase3] GPU 忙碌覆盖生效，暂不启动新任务")
                return False

        # ... 现有的取任务逻辑 ...
```

**改动 4**: 孤儿清理检查

```python
def _check_orphan_cleanup(self) -> bool:
    """检查孤儿执行是否已退出。"""
    if not self._orphan_executions:
        return True

    now = time.time()
    # 超过 2 分钟的孤儿视为已退出（保守估计）
    ORPHAN_TIMEOUT = 120
    expired = [
        jid for jid, t in self._orphan_executions.items()
        if now - t > ORPHAN_TIMEOUT
    ]
    for jid in expired:
        self._orphan_executions.pop(jid, None)
        logger.info("[Phase3] 孤儿执行超时清理: %s", jid)

    return len(self._orphan_executions) == 0
```

### 2.2 Worker finally 块增加孤儿清理

```python
# Worker 循环 finally 块
finally:
    finished_job_id = self.running_job_id or self._current_executing_job_id

    if finished_job_id:
        # Phase 3: 清除孤儿标记
        with self.lock:
            if finished_job_id in self._orphan_executions:
                self._orphan_executions.pop(finished_job_id, None)
                self._gpu_busy_override = False
                logger.info(
                    "[Phase3] 孤儿任务实际退出，清除覆盖标志: %s",
                    finished_job_id,
                )

        # ... 现有的清理逻辑 ...
```

### 2.3 720p 调度器联动

**当前问题**: 720p 转码依赖"队列空闲"检测，但超时放行后误判空闲。

```python
# 在 720p 空闲检测中增加 GPU 忙碌覆盖检查
def _is_gpu_idle(self) -> bool:
    """检查 GPU 是否真正空闲。"""
    queue_service = get_job_queue_service()

    # 基础检查：队列为空且无运行任务
    if queue_service.running_job_id is not None:
        return False
    if len(queue_service.queue) > 0:
        return False

    # Phase 3: 孤儿执行检查
    if RUNNER_GATE_ENABLED and queue_service._gpu_busy_override:
        return False

    return True
```

## 3. 验收标准

1. **不 GPU 争抢**: 超时放行后，不会立即启动新的 GPU 密集任务
2. **孤儿可观测**: 孤儿执行有日志可追踪（开始时间、超时清理时间）
3. **最终收敛**: 孤儿任务最终退出后，队列恢复正常调度
4. **720p 安全**: 720p 转码不在孤儿执行期间启动

## 4. 测试要点

```python
def test_orphan_blocks_next_job():
    """孤儿执行期间不启动新任务。"""
    # 模拟超时放行
    service._force_cancel_timeout_job("job-1", 60.0)
    # 尝试启动下一个任务
    started = service._try_start_next_job()
    assert not started  # 被 GPU 忙碌覆盖阻止

def test_orphan_cleanup_unblocks():
    """孤儿清理后恢复调度。"""
    # 模拟孤儿超时清理
    service._orphan_executions["job-1"] = time.time() - 200
    assert service._check_orphan_cleanup() is True

def test_orphan_actual_exit_unblocks():
    """孤儿任务实际退出后立即恢复。"""
    # Worker finally 块清除孤儿标记
    # 下一次调度应该成功
```
