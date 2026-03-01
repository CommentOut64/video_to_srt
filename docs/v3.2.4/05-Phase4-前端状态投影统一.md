# Phase 4: 前端状态投影统一

> Type: Implementation Plan | Status: Ready
> Phase: 4 (P1 前端收口)
> 目标: 消除 progressStore 与 unifiedTaskStore 双源分叉，统一终态处理
> 依赖: Phase 0 (state_seq), Phase 1 (信号补齐)

## 1. 问题分析

### 当前双源架构

```
后端 SSE 事件
  ├─ 全局频道 job_status    → App.vue → unifiedTaskStore.updateTaskStatus()
  ├─ 全局频道 job_progress  → App.vue → unifiedTaskStore.updateTaskProgress()
  ├─ 单任务频道 progress.*  → EditorView → progressStore.applySseProgress()
  └─ 单任务频道 signal.*    → EditorView → 各种回调

问题：
1. unifiedTaskStore 和 progressStore 独立更新进度，可能产生分歧
2. canceled 状态在 unifiedTaskStore 中无显式处理（缺少 canceled_at 时间戳）
3. App.vue 的 syncTasksFromBackend 与 SSE 初始状态存在竞态
4. progressStore 的容差逻辑（差值<=3保持原值）有歧义
```

## 2. 修改文件

### 2.1 `frontend/src/stores/unifiedTaskStore.js` - 状态序号校验 + canceled 处理

**改动 1**: `state_seq` 递增校验

```javascript
// updateTaskStatus 函数中增加序号校验
function updateTaskStatus(jobId, status, message, meta = {}) {
    const task = tasksMap.value.get(jobId)
    if (!task) return

    // Phase 4: state_seq 递增校验
    const incomingSeq = meta.state_seq ?? 0
    const currentSeq = task.state_seq ?? 0
    if (incomingSeq > 0 && incomingSeq <= currentSeq) {
        console.warn(
            `[TaskStore] 拒绝旧序号事件: job=${jobId}, ` +
            `incoming_seq=${incomingSeq}, current_seq=${currentSeq}`
        )
        return
    }

    // 更新序号
    if (incomingSeq > 0) {
        task.state_seq = incomingSeq
    }

    // ... 原有状态更新逻辑 ...
}
```

**改动 2**: `canceled` 状态显式处理

```javascript
// updateTask 函数中增加 canceled 处理（约 L339-382）
function updateTask(jobId, updates) {
    // ... 现有逻辑 ...

    // 现有: finished 处理
    if (updates.status === 'finished' && task.status !== 'finished') {
        task.isNewlyFinished = true
        updates.completed_at = Date.now()
    }

    // 现有: paused 处理
    if (updates.status === 'paused' && task.status !== 'paused') {
        updates.paused_at = Date.now()
    }

    // 现有: failed 处理
    if (updates.status === 'failed' && task.status !== 'failed') {
        updates.failed_at = Date.now()
    }

    // Phase 4 新增: canceled/force_canceled 处理
    if (['canceled', 'force_canceled'].includes(updates.status)
        && !['canceled', 'force_canceled'].includes(task.status)) {
        updates.canceled_at = Date.now()
        // 清理进度轮询等资源
        progressStore.markStatus(jobId, updates.status)
    }

    // Phase 4 新增: removed 处理
    if (updates.status === 'removed') {
        // 从 Map 中移除
        tasksMap.value.delete(jobId)
        progressStore.cleanupJobState(jobId)
        return
    }

    // ... 应用更新 ...
}
```

**改动 3**: 初始状态增加 `state_seq` 字段

```javascript
// applyTaskSnapshot 中初始化 state_seq
function applyTaskSnapshot(backendTask) {
    const task = tasksMap.value.get(backendTask.id || backendTask.job_id)
    if (task) {
        // Phase 4: 保持 state_seq 不倒退
        const incomingSeq = backendTask.state_seq ?? 0
        if (incomingSeq > 0 && incomingSeq < (task.state_seq ?? 0)) {
            return // 拒绝旧快照
        }
        // ... 更新逻辑 ...
    }
}
```

### 2.2 `frontend/src/stores/progressStore.js` - 修复容差逻辑 + 引用 state_seq

**改动 1**: 修复容差逻辑歧义

```javascript
// shouldAcceptPercent 函数修正（约 L101-121）
function shouldAcceptPercent(state, nextPercent, source, nextStatus) {
    const normalized = clampPercent(nextPercent)
    const now = Date.now()

    // 首次设置，无条件接受
    if (state.lastUpdate === 0) {
        return { accept: true, value: normalized }
    }

    // 递增或持平，接受
    if (normalized >= state.percent) {
        return { accept: true, value: normalized }
    }

    // Phase 4 修正: 区分容差内合法变化和真正倒退
    const diff = state.percent - normalized
    if (diff <= 3) {
        // 小幅下降（<=3%）：SSE 来源接受新值，HTTP 来源保持原值
        if (source === 'sse') {
            return { accept: true, value: normalized }
        }
        return { accept: true, value: state.percent }
    }

    // 大幅倒退：SSE 新鲜时拒绝 HTTP
    if (source !== 'sse' && now - state.lastSseAt < 10000) {
        return { accept: false, value: state.percent }
    }

    // 其他大幅倒退
    console.warn(
        `[ProgressStore] 进度大幅倒退: job=${state.jobId}, ` +
        `${state.percent} -> ${normalized}, source=${source}`
    )
    return { accept: false, value: state.percent }
}
```

**改动 2**: `markStatus` 增加终态清理

```javascript
// markStatus 增加终态处理
function markStatus(jobId, status) {
    const state = progressMap.get(jobId)
    if (!state) return

    state.status = status

    // 终态处理
    const terminalStatuses = ['finished', 'failed', 'canceled', 'force_canceled', 'removed']
    if (terminalStatuses.includes(status)) {
        if (status === 'finished') {
            state.percent = 100
        }
        // 注册延迟清理
        scheduleCleanup(jobId)
    }
}
```

### 2.3 `frontend/src/App.vue` - 修复竞态 + 低频同步

**改动 1**: 修复 syncTasksFromBackend 与 SSE 竞态

```javascript
onMounted(async () => {
    // Phase 4: 先订阅 SSE，再同步任务列表
    // 这样 SSE 初始状态会先到达，HTTP 同步作为兜底
    console.log('[App] 步骤 1: 订阅全局 SSE 事件流...')
    unsubscribeGlobal = sseChannelManager.subscribeGlobal({
        onInitialState(state) {
            if (state.jobs && Array.isArray(state.jobs)) {
                state.jobs.forEach(job => {
                    taskStore.applyTaskSnapshot(job)
                })
            }
            initialStateReceived = true
        },
        // ... 其他回调 ...
    })

    // 步骤 2: HTTP 同步（作为兜底，带 state_seq 校验不会覆盖新数据）
    console.log('[App] 步骤 2: HTTP 同步任务列表...')
    await taskStore.syncTasksFromBackend()
})
```

**改动 2**: 增加低频同步触发

```javascript
// 周期性轻量同步（60s），仅在异常时触发
let syncTimer = null

function startPeriodicSync() {
    syncTimer = setInterval(async () => {
        // 仅在以下情况触发同步：
        // 1. 存在 canceling 状态超过 30s 的任务
        // 2. SSE 连接不健康
        const needsSync = taskStore.hasStaleState('canceling', 30000)
            || !sseChannelManager.isGlobalHealthy()

        if (needsSync) {
            console.log('[App] 检测到异常状态，触发兜底同步')
            await taskStore.syncTasksFromBackend()
        }
    }, 60000)
}

onMounted(() => {
    startPeriodicSync()
})

onUnmounted(() => {
    if (syncTimer) clearInterval(syncTimer)
})
```

### 2.4 统一终态处理

**设计原则**: 所有组件对终态的反应保持一致。

```
终态事件到达 →
  1. unifiedTaskStore.updateTaskStatus() 更新状态和时间戳
  2. progressStore.markStatus() 注册清理
  3. EditorView（如果打开）收到终态后清理 SSE
  4. TaskList 组件更新显示

不再允许：
  - 列表已取消但编辑器仍显示"正在取消"
  - job_removed 后任务"复活"
  - 断线重连后状态不收敛
```

## 3. 验收标准

1. **双源一致**: unifiedTaskStore 和 progressStore 对同一任务的状态始终一致
2. **序号校验**: 旧序号的 SSE 事件被正确拒绝
3. **canceled 显式处理**: 取消后 `canceled_at` 正确设置，进度清理正确执行
4. **无竞态**: App.vue 初始化不再因 HTTP/SSE 顺序导致状态覆盖
5. **断线收敛**: SSE 断线重连后，状态通过 state_seq 自动收敛
6. **job_removed 不复活**: 任务删除后不再出现在列表中

## 4. 测试要点

```javascript
// test_unified_task_store.spec.js

it('拒绝旧序号的状态更新', () => {
    taskStore.updateTaskStatus('job-1', 'processing', '', { state_seq: 5 })
    taskStore.updateTaskStatus('job-1', 'queued', '', { state_seq: 3 })
    expect(taskStore.getTask('job-1').status).toBe('processing')
})

it('canceled 设置 canceled_at', () => {
    taskStore.updateTask('job-1', { status: 'canceled' })
    expect(taskStore.getTask('job-1').canceled_at).toBeDefined()
})

it('removed 从 Map 中删除', () => {
    taskStore.updateTask('job-1', { status: 'removed' })
    expect(taskStore.getTask('job-1')).toBeUndefined()
})

it('进度容差修复: SSE 小幅下降接受', () => {
    progressStore.applySseProgress('job-1', 50)
    progressStore.applySseProgress('job-1', 48)  // 小幅下降
    expect(progressStore.getPercent('job-1')).toBe(48)  // SSE 来源接受
})
```
