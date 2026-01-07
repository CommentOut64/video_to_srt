/**
 * ProgressStore - 统一的任务进度状态管理
 *
 * 作用：
 * - 所有进度、阶段和双流状态都由此集中处理
 * - 自动治理倒退/抖动，优先使用 SSE，HTTP 仅在断链时兜底
 * - 与 unifiedTaskStore 同步，确保任务列表、监控器与编辑器显示一致
 * - [V3.1.0] 自动清理已完成任务的状态，防止内存泄漏
 */
import { defineStore } from 'pinia'
import { reactive, computed } from 'vue'
import { useUnifiedTaskStore } from './unifiedTaskStore'

// [V3.1.0] 清理配置
const CLEANUP_CONFIG = {
  CLEANUP_DELAY: 30000, // 任务完成后 30 秒清理状态
  MAX_COMPLETED_STATES: 10, // 最多保留的已完成任务状态数
}

function clampPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 0
  const normalized = Math.max(0, Math.min(100, Number(value)))
  return Math.round(normalized * 10) / 10
}

function createState(jobId) {
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
    dualStream: {
      fastStream: 0,
      slowStream: 0,
      totalChunks: 0,
      mode: 'unknown'
    },
    lastSource: 'init',
    lastUpdate: 0,
    lastSseAt: 0,
    lastRejected: null
  }
}

export const useProgressStore = defineStore('progress', () => {
  const taskStore = useUnifiedTaskStore()
  const jobStates = reactive({})

  // [V3.1.0] 待清理的任务定时器 { jobId: timeoutId }
  const cleanupTimers = new Map()
  // [V3.1.0] 已完成任务队列（用于 LRU 淘汰）
  const completedQueue = []

  function ensureState(jobId) {
    if (!jobStates[jobId]) {
      jobStates[jobId] = createState(jobId)

      // V3.1.0: 首次创建状态时，从 unifiedTaskStore 同步初始进度
      // 避免页面刷新后进度显示为 0%
      const task = taskStore.getTask(jobId)
      if (task && typeof task.progress === 'number' && !Number.isNaN(task.progress)) {
        jobStates[jobId].percent = task.progress
        jobStates[jobId].status = task.status || 'idle'
        jobStates[jobId].phase = task.phase || 'pending'
        jobStates[jobId].message = task.message || ''
        console.log(`[ProgressStore] 从 unifiedTaskStore 同步初始进度: ${jobId} -> ${task.progress}%`)
      }
    }
    return jobStates[jobId]
  }

  function syncTask(jobId, state) {
    taskStore.updateTaskProgress(jobId, state.percent, state.status, {
      phase: state.phase,
      phase_percent: state.phasePercent,
      message: state.message,
      processed: state.processed,
      total: state.total,
      language: state.language
    })
  }

  function setDualStream(jobId, progress = {}, source = 'sse') {
    const state = ensureState(jobId)
    state.dualStream = {
      fastStream: clampPercent(progress.fastStream ?? progress.fast ?? state.dualStream.fastStream),
      slowStream: clampPercent(progress.slowStream ?? progress.slow ?? state.dualStream.slowStream),
      totalChunks: progress.totalChunks ?? progress.total ?? state.dualStream.totalChunks,
      mode: source
    }
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
    if (state.percent - normalized <= 3) {
      return { accept: true, value: state.percent }
    }
    if (source !== 'sse' && now - state.lastSseAt < 10000) {
      return { accept: false, value: state.percent }
    }
    return { accept: false, value: state.percent }
  }

  function apply(jobId, payload = {}, source = 'unknown') {
    const state = ensureState(jobId)
    const now = Date.now()
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

    if (payload.detail || payload.dualStream) {
      setDualStream(jobId, payload.detail || payload.dualStream, source === 'sse' ? 'sse' : 'http')
      dirty = true
    }

    const percentPayload = payload.percent ?? payload.progress
    if (percentPayload !== undefined) {
      const decision = shouldAcceptPercent(state, percentPayload, source, nextStatus)
      if (decision.accept) {
        const nextValue = decision.value
        state.percent = Math.max(state.percent, nextValue)
        dirty = true
      } else {
        state.lastRejected = {
          percent: clampPercent(percentPayload),
          source,
          at: now
        }
      }
    }

    if (!dirty) {
      return state
    }

    state.lastSource = source
    state.lastUpdate = now
    if (source === 'sse') {
      state.lastSseAt = now
    }

    syncTask(jobId, state)
    return state
  }

  function applySseProgress(jobId, payload = {}) {
    return apply(jobId, payload, 'sse')
  }

  function applySnapshot(jobId, payload = {}, source = 'http') {
    const state = ensureState(jobId)
    const now = Date.now()
    const sseFresh = now - state.lastSseAt < 10000
    const percentPayload = payload.percent ?? payload.progress

    if (source !== 'sse' && sseFresh && percentPayload !== undefined && percentPayload < state.percent) {
      // SSE 仍在更新，忽略 HTTP 导致的倒退，只同步附加信息
      return apply(jobId, { ...payload, percent: state.percent }, source)
    }
    return apply(jobId, payload, source)
  }

  /**
   * V3.1.0: 标记任务状态变更
   * 专门用于状态切换（如 paused, queued, resumed），不会修改进度
   * 除非任务完成（finished），此时强制设为 100%
   * [V3.1.0] 任务完成/取消/失败时，注册延迟清理
   */
  function markStatus(jobId, status, extra = {}) {
    const state = ensureState(jobId)

    // 特殊处理：任务完成时强制设为 100%
    if (status === 'finished') {
      const result = apply(jobId, { status, ...extra, percent: 100 }, 'signal')
      // [V3.1.0] 注册延迟清理
      scheduleCleanup(jobId)
      return result
    }

    // [V3.1.0] 任务取消或失败时也注册清理
    if (status === 'canceled' || status === 'failed') {
      scheduleCleanup(jobId)
    }

    // 其他状态变更时，移除 extra 中的 percent 字段，避免归零
    // 只保留 status 和其他非进度字段
    const { percent, progress, ...safeExtra } = extra
    return apply(jobId, { status, ...safeExtra }, 'signal')
  }

  /**
   * [V3.1.0] 注册延迟清理任务状态
   */
  function scheduleCleanup(jobId) {
    // 取消已有的清理定时器
    if (cleanupTimers.has(jobId)) {
      clearTimeout(cleanupTimers.get(jobId))
    }

    // 添加到已完成队列
    const idx = completedQueue.indexOf(jobId)
    if (idx !== -1) {
      completedQueue.splice(idx, 1)
    }
    completedQueue.push(jobId)

    // 设置延迟清理定时器
    const timer = setTimeout(() => {
      cleanupJobState(jobId)
    }, CLEANUP_CONFIG.CLEANUP_DELAY)

    cleanupTimers.set(jobId, timer)

    // 如果已完成任务数超过上限，立即清理最旧的
    while (completedQueue.length > CLEANUP_CONFIG.MAX_COMPLETED_STATES) {
      const oldestJobId = completedQueue.shift()
      if (oldestJobId && oldestJobId !== jobId) {
        cleanupJobState(oldestJobId)
      }
    }
  }

  /**
   * [V3.1.0] 清理指定任务的状态
   */
  function cleanupJobState(jobId) {
    // 取消定时器
    if (cleanupTimers.has(jobId)) {
      clearTimeout(cleanupTimers.get(jobId))
      cleanupTimers.delete(jobId)
    }

    // 从已完成队列中移除
    const idx = completedQueue.indexOf(jobId)
    if (idx !== -1) {
      completedQueue.splice(idx, 1)
    }

    // 删除状态
    if (jobStates[jobId]) {
      console.log(`[ProgressStore] 清理任务状态: ${jobId}`)
      delete jobStates[jobId]
    }
  }

  /**
   * [V3.1.0] 手动清理任务状态（供外部调用）
   */
  function clearJobState(jobId) {
    cleanupJobState(jobId)
  }

  /**
   * [V3.1.0] 清理所有已完成的任务状态
   */
  function clearAllCompletedStates() {
    for (const jobId of [...completedQueue]) {
      cleanupJobState(jobId)
    }
    console.log('[ProgressStore] 已清理所有已完成任务状态')
  }

  function applyDualStreamEstimate(jobId, progress = {}) {
    const state = ensureState(jobId)
    const sseFresh =
      state.dualStream.mode === 'sse' && Date.now() - state.lastSseAt < 10000

    if (sseFresh) {
      // 若 SSE 已提供真实 fast/slow 但未包含总 Chunk，则补齐总量信息
      if (
        (state.dualStream.totalChunks || 0) <= 0 &&
        (progress.totalChunks || progress.total)
      ) {
        state.dualStream = {
          ...state.dualStream,
          totalChunks: progress.totalChunks ?? progress.total ?? 0,
          draftChunks: progress.draftChunks ?? state.dualStream.draftChunks,
          finalizedChunks:
            progress.finalizedChunks ?? state.dualStream.finalizedChunks,
        }
      }
      return
    }

    setDualStream(jobId, progress, 'estimated')
  }

  function getJobProgress(jobId) {
    return computed(() => ensureState(jobId))
  }

  function getRawState(jobId) {
    return ensureState(jobId)
  }

  return {
    getJobProgress,
    getRawState,
    applySseProgress,
    applySnapshot,
    markStatus,
    applyDualStreamEstimate,
    // [V3.1.0] 新增清理方法
    clearJobState,
    clearAllCompletedStates
  }
})
