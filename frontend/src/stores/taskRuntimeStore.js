/**
 * TaskRuntimeStore - 任务运行时单一真源
 *
 * Phase 2：合并 unifiedTaskStore + progressStore
 * - 任务列表、队列、SSE 健康状态
 * - 编辑器任务进度运行时（percent/phase/detail/dualStream）
 * - state_seq 防倒退与时间戳门控
 */
import { defineStore } from 'pinia'
import { ref, computed, watch, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { useProjectStore } from './projectStore'
import { normalizeTimestamp } from '@/utils/timestamp'
import { navigateToEditor } from '@/utils/editorNavigation'

const CLEANUP_CONFIG = {
  CLEANUP_DELAY: 30000,
  MAX_COMPLETED_STATES: 10,
}
const TASK_MODE = {
  TRANSCRIBE: 'transcribe',
  SUBTITLE_EDIT: 'subtitle_edit',
}

function clampPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 0
  const normalized = Math.max(0, Math.min(100, Number(value)))
  return Math.round(normalized * 10) / 10
}

function normalizeTaskMode(value, options = {}) {
  const fallback = options.fallback || TASK_MODE.TRANSCRIBE
  const normalized = String(value || '').trim().toLowerCase()
  if (normalized === TASK_MODE.TRANSCRIBE || normalized === TASK_MODE.SUBTITLE_EDIT) {
    return normalized
  }
  return fallback
}

function createRuntimeState(jobId) {
  return {
    jobId,
    percent: 0,
    status: 'idle',
    phase: 'pending',
    phasePercent: 0,
    message: '',
    processed: 0,
    total: 0,
    language: null,
    detail: {
      preprocess: 0,
      fast: 0,
      slow: 0,
      align: 0,
    },
    dualStream: {
      fastStream: 0,
      slowStream: 0,
      totalChunks: 0,
      mode: 'unknown',
    },
    lastSource: 'init',
    stateSeq: 0,
    lastUpdate: 0,
    lastSseAt: 0,
    lastServerAt: 0,
    lastRejected: null,
    lastProjection: null,
  }
}

export const useTaskRuntimeStore = defineStore('taskRuntime', () => {
  const router = useRouter()

  // ========== 任务阶段枚举 ==========
  const TaskPhase = {
    UPLOADING: 'uploading',      // 上传中
    TRANSCRIBING: 'transcribing', // 转录中
    EDITING: 'editing',          // 编辑中
    EXPORTING: 'exporting',      // 导出中
    COMPLETED: 'completed'       // 已完成
  }

  // 任务状态枚举（与后端一致）
  const TaskStatus = {
    CREATED: 'created',
    QUEUED: 'queued',
    PROCESSING: 'processing',
    CANCELING: 'canceling',
    PAUSED: 'paused',
    FINISHED: 'finished',
    FAILED: 'failed',
    CANCELED: 'canceled',
    FORCE_CANCELED: 'force_canceled',
    REMOVED: 'removed'
  }

  // ========== 核心数据 ==========
  // 使用 Map 数据结构提高查找性能
  const tasksMap = ref(new Map())
  const activeTaskId = ref(null)
  const currentTask = ref(null)

  // 队列顺序索引（单一事实来源）
  const queueOrder = ref([])
  const queueUpdatedAt = ref(0)

  // SSE 连接状态
  const sseConnected = ref(false)
  const lastHeartbeat = ref(Date.now())

  // 任务运行时进度状态（原 progressStore）
  const jobStates = reactive({})
  const cleanupTimers = new Map()
  const completedQueue = []

  // ========== 计算属性 ==========
  // 将 Map 转换为数组供组件使用
  const tasks = computed(() => Array.from(tasksMap.value.values()))

  // Processing 任务（单例，最多1个）
  const processingTask = computed(() => {
    const processing = tasks.value.filter(t => t.status === 'processing')
    return processing.length > 0 ? processing[0] : null
  })

  // 排队任务（严格按 queueOrder 排序）
  const queuedTasks = computed(() => {
    return queueOrder.value
      .map(id => tasksMap.value.get(id))
      .filter(t => t && t.status === 'queued')
  })

  // 失败任务（按失败时间倒序）
  const failedTasks = computed(() =>
    tasks.value
      .filter(t => t.status === 'failed')
      .sort((a, b) => (b.failed_at || b.updatedAt) - (a.failed_at || a.updatedAt))
  )

  // 暂停任务（按暂停时间倒序）
  const pausedTasks = computed(() =>
    tasks.value
      .filter(t => t.status === 'paused')
      .sort((a, b) => (b.paused_at || b.updatedAt) - (a.paused_at || a.updatedAt))
  )

  // 最近完成任务（最多 20 条）
  const recentFinishedTasks = computed(() =>
    tasks.value
      .filter(t => t.status === 'finished')
      .sort((a, b) => (b.completed_at || b.updatedAt) - (a.completed_at || a.updatedAt))
      .slice(0, 20)
  )

  // 活跃任务数量
  const activeCount = computed(() =>
    tasks.value.filter(t =>
      ['queued', 'processing'].includes(t.status)
    ).length
  )

  // 是否有正在运行的任务
  const hasRunningTask = computed(() =>
    tasks.value.some(t => t.status === 'processing')
  )

  // 最近任务列表（最多8个）
  const recentTasks = computed(() => {
    return tasks.value
      .slice()
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .slice(0, 8)
  })

  // ========== 工具函数 ==========
  function normalizeProgress(value) {
    if (value === null || value === undefined) return null
    const num = Number(value)
    if (Number.isNaN(num)) return null
    return Math.round(Math.max(0, Math.min(100, num)) * 10) / 10
  }

  function normalizeStateSeq(value) {
    const numericValue = Number(value)
    if (!Number.isFinite(numericValue) || numericValue <= 0) return 0
    return Math.floor(numericValue)
  }

  function applyStateSeqGuard(task, incomingSeq, { allowEqual = false } = {}) {
    if (incomingSeq <= 0) return true
    const currentSeq = normalizeStateSeq(task.state_seq)
    const isRejected = allowEqual ? incomingSeq < currentSeq : incomingSeq <= currentSeq
    if (isRejected) {
      console.warn(
        `[TaskRuntimeStore] 拒绝旧序号事件: job=${task.job_id}, incoming_seq=${incomingSeq}, current_seq=${currentSeq}`
      )
      return false
    }
    task.state_seq = incomingSeq
    return true
  }

  function shouldApplyServerUpdate(task, incomingAt) {
    if (!incomingAt) return true
    if (!task.serverUpdatedAt) return true
    return incomingAt >= task.serverUpdatedAt
  }

  function applyUpdateTimestamp(task, incomingAt, isServer = true) {
    if (isServer) {
      if (incomingAt && task.serverUpdatedAt && incomingAt < task.serverUpdatedAt) {
        return false
      }
      if (incomingAt) {
        task.serverUpdatedAt = incomingAt
        task.updatedAt = incomingAt
        return true
      }
    }
    task.updatedAt = Date.now()
    return true
  }

  /**
   * V3.1.0: 应用进度字段（单调递增保护）
   *
   * 核心规则：
   * 1. 只接受递增的进度值，绝不允许进度下降
   * 2. 特殊情况：首次设置进度（current === null）时允许任何值
   * 3. 特殊情况：任务完成（status === finished）时强制设为 100
   *
   * @param {Object} task - 任务对象
   * @param {number|null} incomingProgress - 传入的进度值
   * @param {string|null} statusHint - 状态提示（用于特殊处理）
   */
  function applyProgressField(task, incomingProgress, statusHint) {
    const normalized = normalizeProgress(incomingProgress)

    // 如果传入值无效，不做任何修改
    if (normalized === null) return

    const effectiveStatus = statusHint || task.status
    const current =
      typeof task.progress === 'number' && !Number.isNaN(task.progress)
        ? task.progress
        : null

    // 任务完成时，强制设置为 100%
    if (effectiveStatus === TaskStatus.FINISHED) {
      task.progress = 100
      return
    }

    // 首次设置进度（之前没有有效进度值）
    if (current === null) {
      task.progress = normalized
      return
    }

    // 核心规则：单调递增，取最大值
    // 即使后端返回 0（如恢复任务时），也保持前端已有的高进度
    task.progress = Math.max(current, normalized)
  }

  function ensureRuntimeState(jobId) {
    if (!jobStates[jobId]) {
      const runtimeState = createRuntimeState(jobId)
      const task = tasksMap.value.get(jobId)

      // 运行态被清理后再次访问时，需从任务主状态回填，避免退回 idle 基线。
      if (task) {
        runtimeState.percent = clampPercent(task.progress ?? runtimeState.percent)
        runtimeState.status = task.status || runtimeState.status
        runtimeState.phase = task.phase || runtimeState.phase
        runtimeState.phasePercent = clampPercent(task.phase_percent ?? runtimeState.phasePercent)
        runtimeState.message = task.message ?? runtimeState.message
        runtimeState.processed = task.processed ?? runtimeState.processed
        runtimeState.total = task.total ?? runtimeState.total
        runtimeState.language = task.language ?? runtimeState.language
        runtimeState.stateSeq = Math.max(
          runtimeState.stateSeq,
          normalizeStateSeq(task.state_seq)
        )
        runtimeState.lastServerAt = normalizeTimestamp(
          task.serverUpdatedAt ?? task.updatedAt
        ) || runtimeState.lastServerAt
        runtimeState.lastSource = 'task_restore'
        runtimeState.lastUpdate = Date.now()
      }

      jobStates[jobId] = runtimeState
      console.log(`[TaskRuntimeStore] 创建任务运行态: ${jobId}`)
    }
    return jobStates[jobId]
  }

  function setDualStream(jobId, progress = {}, source = 'sse') {
    const state = ensureRuntimeState(jobId)
    state.dualStream = {
      fastStream: clampPercent(progress.fastStream ?? progress.fast ?? state.dualStream.fastStream),
      slowStream: clampPercent(progress.slowStream ?? progress.slow ?? state.dualStream.slowStream),
      totalChunks: progress.totalChunks ?? progress.total ?? state.dualStream.totalChunks,
      draftChunks: progress.draftChunks ?? state.dualStream.draftChunks,
      finalizedChunks: progress.finalizedChunks ?? state.dualStream.finalizedChunks,
      mode: source,
    }
  }

  function hydrateRuntimeStateFromTask(task, source = 'task') {
    if (!task?.job_id) return
    const state = ensureRuntimeState(task.job_id)
    state.percent = clampPercent(task.progress ?? state.percent)
    state.status = task.status || state.status
    state.phase = task.phase || state.phase
    state.phasePercent = clampPercent(task.phase_percent ?? state.phasePercent)
    state.message = task.message ?? state.message
    state.processed = task.processed ?? state.processed
    state.total = task.total ?? state.total
    state.language = task.language ?? state.language
    state.stateSeq = Math.max(state.stateSeq || 0, normalizeStateSeq(task.state_seq))
    state.lastServerAt = normalizeTimestamp(task.serverUpdatedAt ?? task.updatedAt) || state.lastServerAt
    state.lastSource = source
    state.lastUpdate = Date.now()
  }

  function shouldAcceptPercent(state, nextPercent, source, nextStatus) {
    if (nextPercent === undefined || nextPercent === null || Number.isNaN(Number(nextPercent))) {
      return false
    }
    const normalized = clampPercent(nextPercent)
    const now = Date.now()

    if (state.lastUpdate === 0) {
      return { accept: true, value: normalized }
    }
    if (normalized >= state.percent) {
      return { accept: true, value: normalized }
    }
    const diff = state.percent - normalized
    if (diff <= 3) {
      if (source === 'sse') {
        return { accept: true, value: normalized }
      }
      return { accept: true, value: state.percent }
    }
    if (source !== 'sse' && now - state.lastSseAt < 10000) {
      return { accept: false, value: state.percent }
    }
    if (nextStatus === TaskStatus.FINISHED) {
      return { accept: true, value: 100 }
    }
    console.warn(
      `[TaskRuntimeStore] 进度大幅倒退: job=${state.jobId}, ${state.percent} -> ${normalized}, source=${source}`
    )
    return { accept: false, value: state.percent }
  }

  function syncTaskProjection(jobId, state) {
    updateTaskProgress(jobId, state.percent, state.status, {
      phase: state.phase,
      phase_percent: state.phasePercent,
      message: state.message,
      processed: state.processed,
      total: state.total,
      language: state.language,
    }, {
      updated_at: state.lastServerAt,
      state_seq: state.stateSeq,
      isServer: state.lastSource !== 'signal' && state.lastSource !== 'estimated',
      mirror: true,
    })
  }

  function applyProgressProjection(jobId, payload = {}, source = 'unknown') {
    const state = ensureRuntimeState(jobId)
    const now = Date.now()
    const stateSeq = payload.state_seq ?? null
    const sourcePriority = source === 'sse' ? 2 : 1

    const markProjection = (accepted, reason = null) => {
      state.lastProjection = {
        accepted,
        reason,
        source,
        stateSeq,
        at: now,
      }
    }

    const incomingAt = normalizeTimestamp(payload.updated_at ?? payload.timestamp)
    const incomingSeq = normalizeStateSeq(payload.state_seq)
    const currentSeq = normalizeStateSeq(state.stateSeq)

    if (incomingSeq > 0 && currentSeq > 0 && incomingSeq < currentSeq) {
      state.lastRejected = {
        percent: clampPercent(payload.percent ?? payload.progress),
        source,
        at: now,
      }
      markProjection(false, 'stale_state_seq')
      return state
    }

    if (
      incomingSeq > 0 &&
      incomingSeq === currentSeq &&
      incomingAt &&
      state.lastServerAt &&
      incomingAt === state.lastServerAt &&
      sourcePriority < (state.lastSource === 'sse' ? 2 : 1)
    ) {
      markProjection(false, 'same_seq_lower_priority')
      return state
    }

    if (incomingAt && state.lastServerAt && incomingAt < state.lastServerAt) {
      state.lastRejected = {
        percent: clampPercent(payload.percent ?? payload.progress),
        source,
        at: now,
      }
      markProjection(false, 'stale_server_timestamp')
      return state
    }
    if (incomingAt) {
      state.lastServerAt = incomingAt
    }
    if (incomingSeq > 0) {
      state.stateSeq = Math.max(state.stateSeq || 0, incomingSeq)
    }

    const nextStatus = payload.status || state.status
    let dirty = false

    if (payload.status && payload.status !== state.status) {
      state.status = payload.status
      dirty = true
    }
    if (payload.phase) {
      state.phase = payload.phase
      dirty = true
    }
    if (payload.phase_percent !== undefined) {
      state.phasePercent = clampPercent(payload.phase_percent)
      dirty = true
    }
    if (payload.message !== undefined) {
      state.message = payload.message
      dirty = true
    }
    if (payload.processed !== undefined) {
      state.processed = payload.processed
      dirty = true
    }
    if (payload.total !== undefined) {
      state.total = payload.total
      dirty = true
    }
    if (payload.language !== undefined) {
      state.language = payload.language
      dirty = true
    }

    if (payload.detail && typeof payload.detail === 'object') {
      state.detail = {
        ...state.detail,
        ...payload.detail,
      }
      dirty = true
    }

    if (payload.detail || payload.dualStream) {
      setDualStream(jobId, payload.detail || payload.dualStream, source === 'sse' ? 'sse' : 'http')
      dirty = true
    }

    const percentPayload = payload.percent ?? payload.progress
    if (percentPayload !== undefined) {
      const decision = shouldAcceptPercent(state, percentPayload, source, nextStatus)
      if (decision.accept) {
        state.percent = decision.value
        dirty = true
      } else {
        state.lastRejected = {
          percent: clampPercent(percentPayload),
          source,
          at: now,
        }
        markProjection(false, 'percent_regression_rejected')
      }
    }

    if (!dirty) {
      markProjection(true, 'noop')
      return state
    }

    state.lastSource = source
    state.lastUpdate = now
    if (source === 'sse') {
      state.lastSseAt = now
    }
    markProjection(true, 'applied')
    syncTaskProjection(jobId, state)
    return state
  }

  function scheduleCleanup(jobId) {
    if (cleanupTimers.has(jobId)) {
      clearTimeout(cleanupTimers.get(jobId))
    }
    const idx = completedQueue.indexOf(jobId)
    if (idx !== -1) {
      completedQueue.splice(idx, 1)
    }
    completedQueue.push(jobId)

    const timer = setTimeout(() => {
      cleanupJobState(jobId)
    }, CLEANUP_CONFIG.CLEANUP_DELAY)
    cleanupTimers.set(jobId, timer)

    while (completedQueue.length > CLEANUP_CONFIG.MAX_COMPLETED_STATES) {
      const oldestJobId = completedQueue.shift()
      if (oldestJobId && oldestJobId !== jobId) {
        cleanupJobState(oldestJobId)
      }
    }
  }

  function cleanupJobState(jobId) {
    if (cleanupTimers.has(jobId)) {
      clearTimeout(cleanupTimers.get(jobId))
      cleanupTimers.delete(jobId)
    }
    const idx = completedQueue.indexOf(jobId)
    if (idx !== -1) {
      completedQueue.splice(idx, 1)
    }
    if (jobStates[jobId]) {
      delete jobStates[jobId]
    }
  }

  // ========== Actions ==========

  /**
   * 添加新任务
   */
  function addTask(taskData) {
    const serverUpdatedAt = normalizeTimestamp(
      taskData.updated_at ?? taskData.updatedAt ?? taskData.timestamp
    )
    const createdAt = normalizeTimestamp(
      taskData.createdAt ?? taskData.created_time ?? taskData.created_at
    )
    const completedAt = normalizeTimestamp(taskData.completed_at ?? taskData.completedAt)
    const pausedAt = normalizeTimestamp(taskData.paused_at ?? taskData.pausedAt)
    const failedAt = normalizeTimestamp(taskData.failed_at ?? taskData.failedAt)
    const canceledAt = normalizeTimestamp(taskData.canceled_at ?? taskData.canceledAt)
    const normalizedTaskMode = normalizeTaskMode(taskData.task_mode, {
      fallback: Boolean(taskData.is_project_only)
        ? TASK_MODE.SUBTITLE_EDIT
        : TASK_MODE.TRANSCRIBE,
    })
    const task = {
      job_id: taskData.job_id,
      project_id: taskData.project_id || taskData.job_id,
      filename: taskData.filename,
      file_path: taskData.file_path || null,
      status: taskData.status || TaskStatus.CREATED,
      phase: taskData.phase || TaskPhase.UPLOADING,
      progress: normalizeProgress(taskData.progress) ?? 0,
      phase_percent: taskData.phase_percent || 0,  // 阶段内进度 (0-100)
      message: taskData.message || '',
      settings: taskData.settings || null,
      language: taskData.language || null,
      processed: taskData.processed || 0,
      total: taskData.total || 0,
      createdAt: createdAt || Date.now(),
      updatedAt: serverUpdatedAt || Date.now(),
      serverUpdatedAt: serverUpdatedAt || 0,
      completed_at: completedAt,                    // 完成时间
      paused_at: pausedAt,                          // 暂停时间
      failed_at: failedAt,                          // 失败时间
      canceled_at: canceledAt,                      // 取消时间
      task_mode: normalizedTaskMode,
      is_project_only:
        taskData.is_project_only !== undefined
          ? Boolean(taskData.is_project_only)
          : normalizedTaskMode === TASK_MODE.SUBTITLE_EDIT,
      source_type: taskData.source_type || (normalizedTaskMode === TASK_MODE.SUBTITLE_EDIT ? 'import' : null),
      state_seq: normalizeStateSeq(taskData.state_seq),
      isDirty: false,
      sseConnected: false,
      lastError: null,
      isNewlyFinished: false  // 刚完成标记（用于高亮）
    }

    tasksMap.value.set(task.job_id, task)
    hydrateRuntimeStateFromTask(task, 'task_add')
    saveTasks()
    console.log(`[TaskRuntimeStore] 任务已添加: ${task.job_id}`)
  }

  /**
   * 获取任务
   */
  function getTask(jobId) {
    return tasksMap.value.get(jobId)
  }

  /**
   * 更新任务状态（不更新进度）
   * V3.1.0: 专门用于状态变更，避免进度被覆盖
   */
  function updateTaskStatus(jobId, status, message = null, meta = {}) {
    const task = tasksMap.value.get(jobId)
    if (!task) {
      return false
    }
    const incomingSeq = normalizeStateSeq(meta.state_seq)
    if (!applyStateSeqGuard(task, incomingSeq, { allowEqual: false })) {
      return false
    }
    const incomingAt = normalizeTimestamp(
      meta.updated_at ?? meta.updatedAt ?? meta.timestamp
    )
    const isServer = meta.isServer !== false
    if (!applyUpdateTimestamp(task, incomingAt, isServer)) {
      return false
    }
    task.status = status
    if (message !== null) {
      task.message = message
    }
    if (
      [TaskStatus.CANCELED, TaskStatus.FORCE_CANCELED].includes(status) &&
      !task.canceled_at
    ) {
      task.canceled_at = Date.now()
    }
    if (status === TaskStatus.REMOVED) {
      deleteTask(jobId)
      return true
    }
    hydrateRuntimeStateFromTask(task, 'task_status')
    saveTasks()
    console.log(`[TaskRuntimeStore] 任务状态已更新: ${jobId} -> ${status}`)
    return true
  }

  /**
   * 更新任务进度
   */
  function updateTaskProgress(jobId, percent, status, extraData = {}, meta = {}) {
    const task = tasksMap.value.get(jobId)
    if (!task) {
      ensureRuntimeState(jobId)
      return false
    }

    const incomingSeq = normalizeStateSeq(meta.state_seq)
    const currentSeq = normalizeStateSeq(task.state_seq)
    if (incomingSeq > 0 && incomingSeq < currentSeq) {
      console.warn(
        `[TaskRuntimeStore] 忽略旧序号进度更新: job=${jobId}, incoming_seq=${incomingSeq}, current_seq=${currentSeq}`
      )
      return false
    }
    if (incomingSeq > currentSeq) {
      task.state_seq = incomingSeq
    }

    const incomingAt = normalizeTimestamp(
      meta.updated_at ?? meta.updatedAt ?? meta.timestamp
    )
    const isServer = meta.isServer !== false
    if (!applyUpdateTimestamp(task, incomingAt, isServer)) {
      return false
    }

    applyProgressField(task, percent, status)

    if (status) task.status = status

    // 更新额外字段
    if (extraData.phase) task.phase = extraData.phase
    if (extraData.phase_percent !== undefined) {
      task.phase_percent = Math.round(extraData.phase_percent * 10) / 10
    }
    if (extraData.message) task.message = extraData.message
    if (extraData.processed !== undefined) task.processed = extraData.processed
    if (extraData.total !== undefined) task.total = extraData.total
    if (extraData.language) task.language = extraData.language

    // 收到进度更新说明 SSE 连接正常
    task.sseConnected = true
    task.lastError = null  // 清除错误信息
    if (!meta.mirror) {
      const runtimeState = ensureRuntimeState(jobId)
      runtimeState.lastSource = meta.source || (meta.isServer === false ? 'task_local' : 'sse')
      hydrateRuntimeStateFromTask(task, runtimeState.lastSource)
    }

    if (extraData.detail && typeof extraData.detail === 'object') {
      const state = ensureRuntimeState(jobId)
      state.detail = {
        ...state.detail,
        ...extraData.detail,
      }
      setDualStream(jobId, extraData.detail, 'estimated')
    }

    // 进度更新频繁，不立即保存到 localStorage
    return true
  }

  /**
   * 更新任务 SSE 连接状态
   */
  function updateTaskSSEStatus(jobId, connected, error = null, meta = {}) {
    const task = tasksMap.value.get(jobId)
    if (!task) {
      return false
    }
    const incomingAt = normalizeTimestamp(
      meta.updated_at ?? meta.updatedAt ?? meta.timestamp
    )
    const isServer = meta.isServer === true
    if (isServer && !applyUpdateTimestamp(task, incomingAt, true)) {
      return false
    }
    task.sseConnected = connected
    // 如果是被踢下线的情况，记录错误提示，供 UI 展示弹窗/提示
    if (error) task.lastError = error
    else if (connected) task.lastError = null
    if (!isServer) {
      task.updatedAt = Date.now()
    }
    return true
  }

  /**
   * 检查 SSE 连接状态
   */
  function checkSSEConnection() {
    if (Date.now() - lastHeartbeat.value > 30000) {  // 30 秒无心跳
      sseConnected.value = false
      console.warn('[TaskRuntimeStore] SSE 连接超时')
    }
  }

  /**
   * 更新 SSE 心跳
   */
  function updateSSEHeartbeat() {
    lastHeartbeat.value = Date.now()
    sseConnected.value = true
  }

  /**
   * 更新任务消息
   */
  function updateTaskMessage(jobId, message) {
    const task = tasksMap.value.get(jobId)
    if (task) {
      task.message = message
      task.updatedAt = Date.now()
    }
  }

  /**
   * 通用更新任务方法（更新任意字段）
   */
  function updateTask(jobId, updates, meta = {}) {
    const task = tasksMap.value.get(jobId)
    if (!task) return false
    const incomingSeq = normalizeStateSeq(meta.state_seq ?? updates.state_seq)
    if (updates.status !== undefined) {
      if (!applyStateSeqGuard(task, incomingSeq, { allowEqual: false })) {
        return false
      }
    } else {
      const currentSeq = normalizeStateSeq(task.state_seq)
      if (incomingSeq > 0 && incomingSeq < currentSeq) {
        console.warn(
          `[TaskRuntimeStore] 忽略旧序号任务更新: job=${jobId}, incoming_seq=${incomingSeq}, current_seq=${currentSeq}`
        )
        return false
      }
      if (incomingSeq > currentSeq) {
        task.state_seq = incomingSeq
      }
    }
    const incomingAt = normalizeTimestamp(
      meta.updated_at ?? meta.updatedAt ?? meta.timestamp
    )
    const isServer = meta.isServer !== false
    if (!applyUpdateTimestamp(task, incomingAt, isServer)) {
      return false
    }

    // 检测任务是否刚完成
    if (updates.status === 'finished' && task.status !== 'finished') {
      updates.isNewlyFinished = true
      updates.completed_at = Date.now()

      // 2 秒后移除高亮
      setTimeout(() => {
        const t = tasksMap.value.get(jobId)
        if (t) {
          t.isNewlyFinished = false
        }
      }, 2000)
    }

    // 更新暂停时间戳
    if (updates.status === 'paused' && task.status !== 'paused') {
      updates.paused_at = Date.now()
    }

    // 更新失败时间戳
    if (updates.status === 'failed' && task.status !== 'failed') {
      updates.failed_at = Date.now()
    }
    if (
      ['canceled', 'force_canceled'].includes(updates.status) &&
      !['canceled', 'force_canceled'].includes(task.status)
    ) {
      updates.canceled_at = Date.now()
    }
    if (updates.status === 'removed') {
      deleteTask(jobId)
      return true
    }

    if (updates.progress !== undefined) {
      applyProgressField(task, updates.progress, updates.status)
      delete updates.progress
    }

    Object.assign(task, updates)
    hydrateRuntimeStateFromTask(task, 'task_patch')
    saveTasks()
    console.log(`[TaskRuntimeStore] 任务已更新: ${jobId}`, updates)
    return true
  }

  function applyTaskSnapshot(snapshot, meta = {}) {
    if (!snapshot) return false
    const jobId = snapshot.id || snapshot.job_id
    if (!jobId) return false
    const incomingSeq = normalizeStateSeq(snapshot.state_seq ?? meta.state_seq)
    if (snapshot.status === 'removed') {
      deleteTask(jobId)
      return true
    }

    const incomingAt = normalizeTimestamp(
      snapshot.updated_at ?? snapshot.updatedAt ?? snapshot.timestamp ?? meta.timestamp
    )
    const task = tasksMap.value.get(jobId)
    if (task) {
      if (!applyStateSeqGuard(task, incomingSeq, { allowEqual: true })) {
        return false
      }
      if (!shouldApplyServerUpdate(task, incomingAt)) {
        return false
      }

      if (!applyUpdateTimestamp(task, incomingAt, true)) {
        return false
      }

      const previousStatus = task.status
      if (snapshot.status) task.status = snapshot.status
      if (snapshot.message !== undefined) task.message = snapshot.message
      if (snapshot.filename) task.filename = snapshot.filename
      if (snapshot.title !== undefined) task.title = snapshot.title
      if (snapshot.project_id !== undefined) {
        task.project_id = snapshot.project_id || task.project_id || task.job_id
      }
      const incomingTaskMode = normalizeTaskMode(snapshot.task_mode, {
        fallback:
          snapshot.is_project_only !== undefined
            ? (Boolean(snapshot.is_project_only) ? TASK_MODE.SUBTITLE_EDIT : TASK_MODE.TRANSCRIBE)
            : normalizeTaskMode(task.task_mode, {
              fallback: Boolean(task.is_project_only)
                ? TASK_MODE.SUBTITLE_EDIT
                : TASK_MODE.TRANSCRIBE,
            }),
      })
      task.task_mode = incomingTaskMode
      if (snapshot.phase) task.phase = snapshot.phase
      if (snapshot.phase_percent !== undefined) {
        task.phase_percent = Math.round(snapshot.phase_percent * 10) / 10
      }
      if (snapshot.processed !== undefined) task.processed = snapshot.processed
      if (snapshot.total !== undefined) task.total = snapshot.total
      if (snapshot.language !== undefined) task.language = snapshot.language
      const snapshotCreatedAt = normalizeTimestamp(
        snapshot.created_time ?? snapshot.createdAt ?? snapshot.created_at
      )
      if (snapshotCreatedAt) task.createdAt = snapshotCreatedAt
      const snapshotCompletedAt = normalizeTimestamp(snapshot.completed_at)
      if (snapshotCompletedAt) task.completed_at = snapshotCompletedAt
      const snapshotPausedAt = normalizeTimestamp(snapshot.paused_at)
      if (snapshotPausedAt) task.paused_at = snapshotPausedAt
      const snapshotFailedAt = normalizeTimestamp(snapshot.failed_at)
      if (snapshotFailedAt) task.failed_at = snapshotFailedAt
      const snapshotCanceledAt = normalizeTimestamp(snapshot.canceled_at)
      if (snapshotCanceledAt) task.canceled_at = snapshotCanceledAt
      task.is_project_only =
        snapshot.is_project_only !== undefined
          ? Boolean(snapshot.is_project_only)
          : incomingTaskMode === TASK_MODE.SUBTITLE_EDIT
      if (snapshot.source_type !== undefined) {
        task.source_type = snapshot.source_type || (incomingTaskMode === TASK_MODE.SUBTITLE_EDIT ? 'import' : null)
      } else if (!task.source_type && incomingTaskMode === TASK_MODE.SUBTITLE_EDIT) {
        task.source_type = 'import'
      }
      if (snapshot.progress !== undefined) {
        applyProgressField(task, snapshot.progress, snapshot.status)
      }
      if (task.status === 'finished' && previousStatus !== 'finished') {
        task.completed_at = task.completed_at || Date.now()
      }
      if (task.status === 'paused' && previousStatus !== 'paused') {
        task.paused_at = task.paused_at || Date.now()
      }
      if (task.status === 'failed' && previousStatus !== 'failed') {
        task.failed_at = task.failed_at || Date.now()
      }
      if (
        ['canceled', 'force_canceled'].includes(task.status) &&
        !['canceled', 'force_canceled'].includes(previousStatus)
      ) {
        task.canceled_at = task.canceled_at || Date.now()
      }
      hydrateRuntimeStateFromTask(task, 'snapshot')
      saveTasks()
      return true
    }

    addTask({
      job_id: jobId,
      project_id: snapshot.project_id || jobId,
      filename: snapshot.filename,
      status: snapshot.status,
      progress: snapshot.progress,
      phase_percent: snapshot.phase_percent || 0,
      message: snapshot.message,
      phase: snapshot.phase,
      language: snapshot.language,
      processed: snapshot.processed || 0,
      total: snapshot.total || 0,
      createdAt: normalizeTimestamp(
        snapshot.created_time ?? snapshot.createdAt ?? snapshot.created_at
      ) || Date.now(),
      updated_at: incomingAt,
      completed_at: snapshot.completed_at,
      paused_at: snapshot.paused_at,
      failed_at: snapshot.failed_at,
      canceled_at: snapshot.canceled_at,
      task_mode: snapshot.task_mode,
      is_project_only: snapshot.is_project_only,
      source_type: snapshot.source_type,
      state_seq: incomingSeq
    })
    const newTask = tasksMap.value.get(jobId)
    if (newTask) {
      hydrateRuntimeStateFromTask(newTask, 'snapshot')
    }
    return true
  }

  function applyQueueOrder(queue, meta = {}) {
    if (!Array.isArray(queue)) {
      return false
    }
    const incomingAt = normalizeTimestamp(meta.updated_at ?? meta.updatedAt ?? meta.timestamp)
    if (incomingAt && queueUpdatedAt.value && incomingAt < queueUpdatedAt.value) {
      return false
    }
    queueOrder.value = queue
    queueUpdatedAt.value = incomingAt || Date.now()
    return true
  }

  /**
   * 加载任务到编辑器
   */
  async function loadTask(jobId) {
    const task = tasksMap.value.get(jobId)
    if (!task) {
      console.error(`[TaskRuntimeStore] 任务未找到: ${jobId}`)
      return false
    }

    if (task.status !== TaskStatus.FINISHED) {
      console.warn(`[TaskRuntimeStore] 任务状态异常: ${task.status}`)
      return false
    }

    // 设置当前任务
    currentTask.value = task
    activeTaskId.value = jobId

    // 仅在用户触发加载时进入编辑器，并统一到 project 语义。
    try {
      await navigateToEditor(router, { jobId })
    } catch (error) {
      console.error(`[TaskRuntimeStore] 任务跳转失败: ${jobId}`, error)
      return false
    }

    console.log(`[TaskRuntimeStore] 任务已加载: ${jobId}`)
    return true
  }

  /**
   * 保存当前任务
   */
  async function saveCurrentTask() {
    if (!currentTask.value) {
      console.warn('[TaskRuntimeStore] 无当前任务')
      return
    }

    // 调用 ProjectStore 的保存逻辑
    const projectStore = useProjectStore()
    await projectStore.saveProject()

    currentTask.value.isDirty = false
    console.log(`[TaskRuntimeStore] 当前任务已保存: ${currentTask.value.job_id}`)
  }

  /**
   * 删除任务
   */
  function deleteTask(jobId) {
    tasksMap.value.delete(jobId)
    cleanupJobState(jobId)
    // 同时从队列顺序中删除
    queueOrder.value = queueOrder.value.filter(id => id !== jobId)
    saveTasks()
    console.log(`[TaskRuntimeStore] 任务已删除: ${jobId}`)
  }

  function hasStaleState(statuses, staleMs = 30000) {
    const statusList = Array.isArray(statuses) ? statuses : [statuses]
    const now = Date.now()
    return tasks.value.some(task => {
      if (!statusList.includes(task.status)) return false
      const lastKnownAt = normalizeTimestamp(task.serverUpdatedAt ?? task.updatedAt) || 0
      return now - lastKnownAt > staleMs
    })
  }

  /**
   * 重新排序队列
   */
  async function reorderQueue(newOrder) {
    const oldOrder = [...queueOrder.value]

    // 乐观更新
    queueOrder.value = newOrder

    try {
      const transcriptionApi = (await import('@/services/api/transcriptionApi')).default
      const result = await transcriptionApi.reorderQueue(newOrder)

      if (!result.reordered) {
        // 恢复原顺序
        queueOrder.value = oldOrder
        console.error('[TaskRuntimeStore] 队列重排失败')
      } else {
        // 持久化新顺序
        saveTasks()
      }
    } catch (error) {
      // 恢复原顺序
      queueOrder.value = oldOrder
      console.error('[TaskRuntimeStore] 队列重排请求失败:', error)
      throw error
    }
  }

  /**
   * 从后端同步任务列表（第一阶段修复：数据同步）
   *
   * 从后端获取所有实际存在的任务列表，用于修复幽灵任务问题
   * 这是前端 localStorage 的真实源
   */
  async function syncTasksFromBackend() {
    try {
      const transcriptionApi = (await import('@/services/api/transcriptionApi')).default
      const response = await transcriptionApi.syncTasks()

      if (!response.success) {
        console.warn('[TaskRuntimeStore] 任务同步失败:', response)
        return false
      }

      const backendTasksRaw = response.tasks || []
      const dedupedTasksById = new Map()
      for (const task of backendTasksRaw) {
        const taskId = String(task?.id || task?.job_id || '').trim()
        if (!taskId) continue
        const existing = dedupedTasksById.get(taskId)
        if (!existing) {
          dedupedTasksById.set(taskId, task)
          continue
        }
        const existingUpdatedAt = normalizeTimestamp(
          existing?.updated_at ?? existing?.updatedAt ?? existing?.timestamp
        ) || 0
        const incomingUpdatedAt = normalizeTimestamp(
          task?.updated_at ?? task?.updatedAt ?? task?.timestamp
        ) || 0
        if (incomingUpdatedAt >= existingUpdatedAt) {
          dedupedTasksById.set(taskId, task)
        }
      }
      const backendTasks = Array.from(dedupedTasksById.values())
      const dedupedCount = backendTasksRaw.length - backendTasks.length
      if (dedupedCount > 0) {
        console.log(`[TaskRuntimeStore] 语义去重完成: 合并 ${dedupedCount} 条重复任务快照`)
      }
      console.log(`[TaskRuntimeStore] 从后端同步了 ${backendTasks.length} 个任务`)

      // 1. 获取后端任务ID集合
      const backendTaskIds = new Set(backendTasks.map(t => t.id))

      // 2. 删除前端有但后端没有的任务（幽灵任务清理）
      const localTaskIds = Array.from(tasksMap.value.keys())
      let deletedCount = 0
      for (const localId of localTaskIds) {
        if (!backendTaskIds.has(localId)) {
          console.log(`[TaskRuntimeStore] 删除幽灵任务: ${localId}`)
          tasksMap.value.delete(localId)
          deletedCount++
        }
      }
      if (deletedCount > 0) {
        console.log(`[TaskRuntimeStore] 共清理了 ${deletedCount} 个幽灵任务`)
      }

      // 3. 更新或添加后端任务
      let updatedCount = 0
      let addedCount = 0
      let skippedCount = 0
      for (const backendTask of backendTasks) {
        // V3.1.0: 过滤掉 filename 为空的任务，避免显示"未知任务"
        if (!backendTask.filename || backendTask.filename.trim() === '') {
          console.warn(`[TaskRuntimeStore] 跳过 filename 为空的任务: ${backendTask.id}`)
          skippedCount++
          continue
        }

        const existingTask = tasksMap.value.get(backendTask.id)
        const applied = applyTaskSnapshot(backendTask)
        if (existingTask) {
          if (applied && backendTask.status === 'processing') {
            existingTask.sseConnected = false
          }
          if (applied) {
            updatedCount++
          }
        } else if (applied) {
          addedCount++
        }
      }

      // 4. 更新队列顺序
      if (response.queue) {
        applyQueueOrder(response.queue, { timestamp: response.queue_updated_at })
        console.log(`[TaskRuntimeStore] 队列顺序已同步: ${queueOrder.value.length} 个任务`)
      }

      console.log(
        `[TaskRuntimeStore] 任务同步完成: ${updatedCount} 个更新, ${addedCount} 个新增, ${deletedCount} 个删除` +
        (skippedCount > 0 ? `, ${skippedCount} 个跳过（filename为空）` : '')
      )
      saveTasks()
      return true
    } catch (error) {
      console.error('[TaskRuntimeStore] 任务同步失败:', error)
      return false
    }
  }

  /**
   * 清空所有任务
   */
  function clearAllTasks() {
    tasksMap.value.clear()
    Object.keys(jobStates).forEach((jobId) => cleanupJobState(jobId))
    saveTasks()
    console.log('[TaskRuntimeStore] 所有任务已清空')
  }

  /**
   * 清理过期任务（超过7天）
   */
  function cleanupOldTasks() {
    const SEVEN_DAYS = 7 * 24 * 60 * 60 * 1000
    const now = Date.now()
    let cleanedCount = 0

    for (const [jobId, task] of tasksMap.value) {
      if (now - task.updatedAt > SEVEN_DAYS) {
        tasksMap.value.delete(jobId)
        cleanupJobState(jobId)
        cleanedCount++
      }
    }

    if (cleanedCount > 0) {
      saveTasks()
      console.log(`[TaskRuntimeStore] 已清理 ${cleanedCount} 个过期任务`)
    }
  }

  // ========== 持久化 ==========

  /**
   * 保存任务列表到 localStorage
   */
  function saveTasks() {
    try {
      const tasksArray = Array.from(tasksMap.value.values())
      localStorage.setItem('task-list', JSON.stringify(tasksArray))
      // 保存队列顺序
      localStorage.setItem('queue-order', JSON.stringify(queueOrder.value))
    } catch (error) {
      console.error('[TaskRuntimeStore] 保存任务列表失败:', error)
    }
  }

  /**
   * 从 localStorage 恢复任务列表
   */
  function restoreTasks() {
    try {
      const saved = localStorage.getItem('task-list')
      if (saved) {
        const tasksArray = JSON.parse(saved)
        tasksMap.value = new Map(
          tasksArray.map((t) => {
            const createdAt = normalizeTimestamp(t.createdAt ?? t.created_time ?? t.created_at)
            const updatedAt = normalizeTimestamp(t.updatedAt ?? t.updated_at)
            const serverUpdatedAt = normalizeTimestamp(t.serverUpdatedAt) || updatedAt || 0
            return [
              t.job_id,
              {
                ...t,
                project_id: t.project_id || t.job_id,
                createdAt: createdAt || Date.now(),
                updatedAt: updatedAt || Date.now(),
                completed_at: normalizeTimestamp(t.completed_at),
                paused_at: normalizeTimestamp(t.paused_at),
                failed_at: normalizeTimestamp(t.failed_at),
                canceled_at: normalizeTimestamp(t.canceled_at),
                state_seq: normalizeStateSeq(t.state_seq),
                serverUpdatedAt,
                task_mode: normalizeTaskMode(t.task_mode, {
                  fallback: Boolean(t.is_project_only)
                    ? TASK_MODE.SUBTITLE_EDIT
                    : TASK_MODE.TRANSCRIBE,
                }),
                is_project_only: Boolean(t.is_project_only),
                source_type:
                  t.source_type ||
                  (normalizeTaskMode(t.task_mode, {
                    fallback: Boolean(t.is_project_only)
                      ? TASK_MODE.SUBTITLE_EDIT
                      : TASK_MODE.TRANSCRIBE,
                  }) === TASK_MODE.SUBTITLE_EDIT
                    ? 'import'
                    : null),
              }
            ]
          })
        )
        Array.from(tasksMap.value.values()).forEach((task) => {
          hydrateRuntimeStateFromTask(task, 'restore')
        })
        console.log(`[TaskRuntimeStore] 已恢复 ${tasksArray.length} 个任务`)
      }

      // 恢复队列顺序
      const savedOrder = localStorage.getItem('queue-order')
      if (savedOrder) {
        queueOrder.value = JSON.parse(savedOrder)
        console.log(`[TaskRuntimeStore] 已恢复队列顺序: ${queueOrder.value.length} 个任务`)
      }
    } catch (error) {
      console.error('[TaskRuntimeStore] 恢复任务列表失败:', error)
    }
  }

  // ========== 批量操作 ==========

  /**
   * 批量更新任务
   */
  function updateMultipleTasks(updates) {
    updates.forEach(({ jobId, ...data }) => {
      const task = tasksMap.value.get(jobId)
      if (task) {
        Object.assign(task, data, { updatedAt: Date.now() })
      }
    })
    saveTasks()
  }

  /**
   * 获取指定状态的任务
   */
  function getTasksByStatus(status) {
    return tasks.value.filter(t => t.status === status)
  }

  /**
   * 获取指定阶段的任务
   */
  function getTasksByPhase(phase) {
    return tasks.value.filter(t => t.phase === phase)
  }

  // ========== Progress 兼容 API ==========
  function applySseProgress(jobId, payload = {}) {
    return applyProgressProjection(jobId, payload, 'sse')
  }

  function applySnapshot(jobId, payload = {}, source = 'http') {
    const state = ensureRuntimeState(jobId)
    const now = Date.now()
    const sseFresh = now - state.lastSseAt < 10000
    const percentPayload = payload.percent ?? payload.progress

    if (source !== 'sse' && sseFresh && percentPayload !== undefined && percentPayload < state.percent) {
      return applyProgressProjection(jobId, { ...payload, percent: state.percent }, source)
    }
    return applyProgressProjection(jobId, payload, source)
  }

  function markStatus(jobId, status, extra = {}) {
    const terminalStatuses = ['finished', 'failed', 'canceled', 'force_canceled', 'removed']
    const { percent, progress, ...safeExtra } = extra
    const payload = { status, ...safeExtra }

    if (status === 'finished') {
      payload.percent = 100
    }

    const result = applyProgressProjection(jobId, payload, 'signal')
    if (terminalStatuses.includes(status)) {
      scheduleCleanup(jobId)
    }
    return result
  }

  function applyDualStreamEstimate(jobId, progress = {}) {
    const state = ensureRuntimeState(jobId)
    const sseFresh =
      state.dualStream.mode === 'sse' && Date.now() - state.lastSseAt < 10000

    if (sseFresh) {
      if (
        (state.dualStream.totalChunks || 0) <= 0 &&
        (progress.totalChunks || progress.total)
      ) {
        state.dualStream = {
          ...state.dualStream,
          totalChunks: progress.totalChunks ?? progress.total ?? 0,
          draftChunks: progress.draftChunks ?? state.dualStream.draftChunks,
          finalizedChunks: progress.finalizedChunks ?? state.dualStream.finalizedChunks,
        }
      }
      return
    }

    setDualStream(jobId, progress, 'estimated')
  }

  function getJobProgress(jobId) {
    return computed(() => ensureRuntimeState(jobId))
  }

  function getRawState(jobId) {
    return ensureRuntimeState(jobId)
  }

  function selectTaskStatus(jobId) {
    return computed(() => {
      if (!jobId) return 'idle'
      const task = tasksMap.value.get(jobId)
      if (task?.status) return task.status
      const runtime = jobStates[jobId]
      return runtime?.status || 'idle'
    })
  }

  function clearJobState(jobId) {
    cleanupJobState(jobId)
  }

  function clearAllCompletedStates() {
    for (const jobId of [...completedQueue]) {
      cleanupJobState(jobId)
    }
  }

  // ========== 初始化 ==========
  // 页面加载时恢复任务列表
  restoreTasks()

  // 自动清理过期任务（启动时执行一次）
  cleanupOldTasks()

  // 监听任务变化，自动保存（防抖）
  watch(tasksMap, () => {
    saveTasks()
  }, { deep: true })

  return {
    // 枚举
    TaskPhase,
    TaskStatus,

    // 状态
    tasks,
    tasksMap,
    activeTaskId,
    currentTask,
    queueOrder,
    queueUpdatedAt,
    sseConnected,
    lastHeartbeat,
    jobStates,

    // 计算属性
    processingTask,
    queuedTasks,
    failedTasks,
    pausedTasks,
    recentFinishedTasks,
    activeCount,
    hasRunningTask,
    recentTasks,

    // 操作方法
    addTask,
    getTask,
    updateTaskStatus,
    updateTaskProgress,
    updateTaskSSEStatus,
    updateTaskMessage,
    updateTask,
    applyTaskSnapshot,
    loadTask,
    saveCurrentTask,
    deleteTask,
    hasStaleState,
    applyQueueOrder,
    reorderQueue,
    syncTasksFromBackend,
    clearAllTasks,
    cleanupOldTasks,

    // SSE 相关
    checkSSEConnection,
    updateSSEHeartbeat,

    // 批量操作
    updateMultipleTasks,
    getTasksByStatus,
    getTasksByPhase,

    // Progress 兼容 API
    getJobProgress,
    getRawState,
    selectTaskStatus,
    applySseProgress,
    applySnapshot,
    markStatus,
    applyDualStreamEstimate,
    clearJobState,
    clearAllCompletedStates,

    // 持久化
    saveTasks,
    restoreTasks
  }
})
