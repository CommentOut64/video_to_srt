# Phase 1: 取消/删除生命周期收敛

> Type: Implementation Plan | Status: Ready
> Phase: 1 (P0 用户最痛点)
> 目标: 彻底解决"正在取消卡住""删除后状态不收敛""后续任务不启动"
> 灰度开关: `CANCEL_V2_ENABLED`
> 依赖: Phase 0 (状态机校验层)

## 1. 目标

消除用户最痛的三个问题：
1. 前端长期卡在"正在取消..."
2. 运行中删除后任务状态不收敛
3. 取消链路误发 `job_failed` 信号

## 2. 后端改动

### 2.1 `backend/app/pipelines/orchestrator.py` - 异常语义分层

**当前问题**: `orchestrator.py:295` 的 `except Exception` 会将 `CancelledException`/`PausedException` 误标为 `failed`。

**修改方案**: 拆分为三级异常捕获。

```python
# 当前代码 (orchestrator.py:293-300)
except Exception as exc:
    self.logger.error(f"Pipeline 执行失败: {exc}", exc_info=True)
    job.status = "failed"
    job.error = str(exc)
    job.message = f"失败: {str(exc)}"
    raise

# 修改为:
except CancelledException:
    # 取消异常：透传到上层，不标记 failed
    self.logger.info(f"Pipeline 取消: {job.job_id}")
    raise  # 透传，由 Worker 循环处理终态

except PausedException:
    # 暂停异常：透传到上层，不标记 failed
    self.logger.info(f"Pipeline 暂停: {job.job_id}")
    raise  # 透传，由 Worker 循环处理终态

except Exception as exc:
    # 真正的业务异常才标记 failed
    self.logger.error(f"Pipeline 执行失败: {exc}", exc_info=True)
    job.status = "failed"
    job.error = str(exc)
    job.message = f"失败: {str(exc)}"
    raise
```

**同步检查**: 搜索 `orchestrator.py` 中所有 `except Exception` 和 `except BaseException`，确保 `CancelledException`/`PausedException` 均在前面被单独捕获。

### 2.2 `backend/app/services/job_queue_service.py` - 上层异常映射修复

**当前问题**: `job_queue_service.py:947` 附近的上层封装也会先打 `failed`。

**修改位置**: `_run_dual_alignment_pipeline` / `_run_pipeline_v2` 中的异常处理。

```python
# 修改 Worker 循环中的异常处理（约 L680-710）
# 确保 CancelledException/PausedException 在 except Exception 之前被捕获

try:
    self._run_pipeline_v2(job)
    # 正常完成
    result = guard.transition(job.job_id, job.status, "finished", "pipeline_complete")
    if result.success:
        job.status = result.to_status
        job.state_seq = result.state_seq
        job.message = "转录完成"

except CancelledException as exc:
    self.logger.info(f"任务取消完成: {exc.job_id}")
    result = guard.transition(job.job_id, job.status, "canceled", "pipeline_canceled")
    if result.success:
        job.status = result.to_status
        job.state_seq = result.state_seq
        job.message = "已取消"

except PausedException as exc:
    self.logger.info(f"任务暂停完成: {exc.job_id}")
    result = guard.transition(job.job_id, job.status, "paused", "pipeline_paused")
    if result.success:
        job.status = result.to_status
        job.state_seq = result.state_seq
        job.message = "已暂停"

except Exception as exc:
    self.logger.error(f"任务执行异常: {exc}", exc_info=True)
    result = guard.transition(job.job_id, job.status, "failed", f"error: {str(exc)[:100]}")
    if result.success:
        job.status = result.to_status
        job.state_seq = result.state_seq
        job.error = str(exc)
        job.message = f"失败: {str(exc)[:200]}"
```

### 2.3 `backend/app/services/job_queue_service.py` - 取消返回结构化

**当前问题**: `cancel_job` 返回不统一，前端难以判断取消结果。

**改动**:

```python
@dataclass
class CancelResult:
    """取消操作结果。"""
    success: bool
    status: str            # 当前状态
    reason_code: str       # 结构化原因码
    message: str           # 人类可读消息
    pending_delete: bool   # 是否待删除
    state_seq: int = 0

# 原因码枚举
CANCEL_REASON_CODES = {
    "cancel_queued": "排队中任务直接取消",
    "cancel_running": "运行中任务请求取消",
    "cancel_already": "任务已取消或已完成",
    "cancel_not_found": "任务不存在",
    "delete_blocked": "删除被阻塞（任务仍在运行）",
}
```

### 2.4 `backend/app/services/job_queue_service.py` - 删除语义标准化

**当前问题**: 运行中删除可能回落 `paused`，语义混乱。

**改动**:

```python
def cancel_job(self, job_id: str, delete_data: bool = False) -> CancelResult:
    """
    统一取消/删除入口。

    运行中删除 = canceling + pending_delete=true
    不再使用 paused 作为删除失败的回退状态。
    """
    # ...

    elif self.running_job_id == job_id:
        is_running = True
        result = guard.transition(job.job_id, job.status, "canceling", "cancel_running")
        if result.success:
            job.status = result.to_status
            job.state_seq = result.state_seq
            job.message = "正在取消..."
            self._pending_cancel_requests[job_id] = time.time()
            if delete_data:
                self._pending_delete_after_cancel.add(job_id)

        return CancelResult(
            success=True,
            status=job.status,
            reason_code="cancel_running",
            message="任务正在执行，已请求取消" + ("并将在结束后删除" if delete_data else ""),
            pending_delete=delete_data,
            state_seq=job.state_seq,
        )
```

### 2.5 `backend/app/api/routes/transcription_routes.py` - API 返回契约

**改动**: cancel/delete 端点返回结构化 `CancelResult`。

```python
@router.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, delete_data: bool = False):
    result = job_queue_service.cancel_job(job_id, delete_data)
    return {
        "success": result.success,
        "status": result.status,
        "reason_code": result.reason_code,
        "message": result.message,
        "pending_delete": result.pending_delete,
        "state_seq": result.state_seq,
    }
```

## 3. 前端改动

### 3.1 `frontend/src/services/sseChannelManager.js` - 信号补齐

**当前问题**: 单任务频道 `signal.*` 事件注册缺少 `job_canceling` 和 `job_force_canceled`。

**改动**: 在 `subscribeJob` 的事件映射表中补齐。

```javascript
// sseChannelManager.js 约 L211-216
// 现有:
'signal.job_start': handleSignal,
'signal.job_complete': handleSignal,
'signal.job_failed': handleSignal,
'signal.job_paused': handleSignal,
'signal.job_canceled': handleSignal,
'signal.job_resumed': handleSignal,

// 新增:
'signal.job_canceling': handleSignal,        // 补齐
'signal.job_force_canceled': handleSignal,    // 补齐
'signal.pause_pending': handleSignal,         // 补齐（暂停请求已发出）
'signal.pause_ack': handleSignal,             // 补齐（暂停快照已保存）
```

**同步修改** `handleSignal` 内部分发（约 L136-155）:

```javascript
function handleSignal(data) {
    const signal = data.signal || data.type || ''
    handlers.onSignal?.(signal, data)

    if (signal === 'job_complete') {
        handlers.onComplete?.(data)
    } else if (signal === 'job_failed') {
        handlers.onFailed?.(data)
    } else if (signal === 'job_paused') {
        handlers.onPaused?.(data)
    } else if (signal === 'job_canceled') {
        handlers.onCanceled?.(data)
    } else if (signal === 'job_resumed') {
        handlers.onResumed?.(data)
    }
    // 新增:
    else if (signal === 'job_canceling') {
        handlers.onCanceling?.(data)           // 新增回调
    } else if (signal === 'job_force_canceled') {
        handlers.onForceCanceled?.(data)       // 新增回调
    } else if (signal === 'pause_pending') {
        handlers.onPausePending?.(data)        // 新增回调
    } else if (signal === 'pause_ack') {
        handlers.onPauseAck?.(data)            // 新增回调
    }
}
```

### 3.2 `frontend/src/views/EditorView.vue` - 终态后断连

**当前问题**: `EditorView.vue:1506` 取消后立即调用 `cleanupSSE()`，导致收不到终态信号。

**修改方案**: 取消后不立即断连，等收到终态信号后再清理。

```javascript
// 修改 cancelTranscription（约 L1492-1511）
async function cancelTranscription() {
    if (!confirm('确定要取消当前转录任务吗?')) return
    try {
        const result = await transcriptionApi.cancelJob(props.jobId, /* deleteData */ false)

        // V3.2.0+dev: 不立即断 SSE，等待终态信号
        // cleanupSSE()  ← 移除此行
        // stopProgressPolling()  ← 移除此行

        // 改为：设置取消等待标志，SSE 收到终态后自动清理
        isCancelPending.value = true
        console.log('[EditorView] 取消请求已发送，等待终态信号...')

    } catch (error) {
        console.error('[EditorView] 取消失败:', error)
    }
}

// 在 subscribeSSE 的 onCanceled/onForceCanceled 回调中清理
onCanceled(data) {
    console.log('[EditorView] 收到取消终态信号')
    isCancelPending.value = false
    cleanupSSE()
    stopProgressPolling()
},
onForceCanceled(data) {
    console.log('[EditorView] 收到强制取消终态信号')
    isCancelPending.value = false
    cleanupSSE()
    stopProgressPolling()
},
```

**新增**: `canceling` 超时轮询兜底

```javascript
// 在 subscribeSSE 的 onCanceling 回调中启动超时轮询
onCanceling(data) {
    console.log('[EditorView] 收到 canceling 信号，启动超时轮询')
    startCancelTimeoutPolling()
},

// 超时轮询函数
const CANCEL_TIMEOUT_MS = 30000  // 30秒超时
let cancelTimeoutTimer = null

function startCancelTimeoutPolling() {
    cancelTimeoutTimer = setTimeout(async () => {
        console.log('[EditorView] canceling 超时，主动拉取状态')
        try {
            const status = await transcriptionApi.getJobStatus(props.jobId)
            if (['canceled', 'force_canceled', 'failed', 'finished'].includes(status.status)) {
                // 终态已达成，清理
                isCancelPending.value = false
                cleanupSSE()
                stopProgressPolling()
            } else {
                // 仍在 canceling，再等一轮
                startCancelTimeoutPolling()
            }
        } catch (error) {
            console.error('[EditorView] 状态查询失败:', error)
        }
    }, CANCEL_TIMEOUT_MS)
}
```

## 4. 验收标准

1. **取消不落 failed**: 取消链路不再发送 `job_failed` 信号（PBT INV-4 通过）
2. **前端不卡 canceling**: 运行中取消后，前端最迟 30s 内收敛到终态
3. **删除语义明确**: 运行中删除返回 `{ status: "canceling", pending_delete: true, reason_code: "cancel_running" }`
4. **信号完整**: 前端正确处理 `job_canceling`、`job_force_canceled`、`pause_pending`、`pause_ack`
5. **终态后断连**: SSE 连接在收到终态信号后才清理，不提前断开

## 5. 测试要点

```python
# test_cancel_lifecycle.py

async def test_cancel_running_does_not_emit_failed():
    """运行中取消不应产生 failed 事件。"""
    # 启动任务 -> 等待 processing -> 取消
    # 收集所有 SSE 事件
    # 断言: 事件序列中无 job_failed
    # 断言: 终态为 canceled 或 force_canceled

async def test_cancel_returns_structured_result():
    """取消返回结构化结果。"""
    result = cancel_job(job_id, delete_data=True)
    assert result["reason_code"] == "cancel_running"
    assert result["pending_delete"] is True

async def test_cancel_timeout_converges():
    """canceling 超时后收敛到 force_canceled。"""
    # 模拟流水线卡住
    # 等待超时阈值
    # 断言: 状态从 canceling -> force_canceled

async def test_delete_running_does_not_fallback_paused():
    """运行中删除不回落 paused。"""
    result = cancel_job(job_id, delete_data=True)
    assert result["status"] != "paused"
```

```javascript
// 前端测试要点
// test_editor_cancel_sse.spec.js

it('取消后不立即断 SSE', async () => {
    await cancelTranscription()
    expect(sseConnection.readyState).not.toBe(EventSource.CLOSED)
})

it('收到终态信号后断 SSE', async () => {
    await cancelTranscription()
    sseEmit('signal.job_canceled', { job_id: 'xxx' })
    await nextTick()
    expect(sseConnection.readyState).toBe(EventSource.CLOSED)
})

it('canceling 超时轮询兜底', async () => {
    jest.useFakeTimers()
    sseEmit('signal.job_canceling', { job_id: 'xxx' })
    jest.advanceTimersByTime(30000)
    // 应触发 HTTP 状态查询
    expect(transcriptionApi.getJobStatus).toHaveBeenCalled()
})
```
