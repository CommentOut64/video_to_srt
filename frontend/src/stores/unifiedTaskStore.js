/**
 * UnifiedTaskStore - 统一任务状态管理
 *
 * 负责管理所有任务的生命周期，从上传、转录到编辑、导出的完整流程
 * 实现了任务状态的统一管理，确保转录和编辑工作流的无缝衔接
 */
import { defineStore } from 'pinia'
import { ref, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useProjectStore } from './projectStore'
import { normalizeTimestamp } from '@/utils/timestamp'

export const useUnifiedTaskStore = defineStore('unifiedTask', () => {
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
        `[UnifiedTaskStore] 拒绝旧序号事件: job=${task.job_id}, incoming_seq=${incomingSeq}, current_seq=${currentSeq}`
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

  // ========== Actions ==========

  /**
   * 添加新任务
   */
  function addTask(taskData) {
    const serverUpdatedAt = normalizeTimestamp(
      taskData.updated_at ?? taskData.updatedAt ?? taskData.timestamp
    )
    const task = {
      job_id: taskData.job_id,
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
      createdAt: taskData.createdAt || Date.now(),
      updatedAt: serverUpdatedAt || Date.now(),
      serverUpdatedAt: serverUpdatedAt || 0,
      completed_at: taskData.completed_at || null,  // 完成时间
      paused_at: taskData.paused_at || null,        // 暂停时间
      failed_at: taskData.failed_at || null,        // 失败时间
      canceled_at: taskData.canceled_at || null,    // 取消时间
      state_seq: normalizeStateSeq(taskData.state_seq),
      isDirty: false,
      sseConnected: false,
      lastError: null,
      isNewlyFinished: false  // 刚完成标记（用于高亮）
    }

    tasksMap.value.set(task.job_id, task)
    saveTasks()
    console.log(`[UnifiedTaskStore] 任务已添加: ${task.job_id}`)
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
    if (task) {
      const incomingSeq = normalizeStateSeq(meta.state_seq)
      if (!applyStateSeqGuard(task, incomingSeq, { allowEqual: false })) {
        return
      }
      const incomingAt = normalizeTimestamp(
        meta.updated_at ?? meta.updatedAt ?? meta.timestamp
      )
      const isServer = meta.isServer !== false
      if (!applyUpdateTimestamp(task, incomingAt, isServer)) {
        return
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
        return
      }
      saveTasks()
      console.log(`[UnifiedTaskStore] 任务状态已更新: ${jobId} -> ${status}`)
    }
  }

  /**
   * 更新任务进度
   */
  function updateTaskProgress(jobId, percent, status, extraData = {}, meta = {}) {
    const task = tasksMap.value.get(jobId)
    if (!task) return

    const incomingSeq = normalizeStateSeq(meta.state_seq)
    const currentSeq = normalizeStateSeq(task.state_seq)
    if (incomingSeq > 0 && incomingSeq < currentSeq) {
      console.warn(
        `[UnifiedTaskStore] 忽略旧序号进度更新: job=${jobId}, incoming_seq=${incomingSeq}, current_seq=${currentSeq}`
      )
      return
    }
    if (incomingSeq > currentSeq) {
      task.state_seq = incomingSeq
    }

    const incomingAt = normalizeTimestamp(
      meta.updated_at ?? meta.updatedAt ?? meta.timestamp
    )
    const isServer = meta.isServer !== false
    if (!applyUpdateTimestamp(task, incomingAt, isServer)) {
      return
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

    // 进度更新频繁，不立即保存到 localStorage
  }

  /**
   * 更新任务 SSE 连接状态
   */
  function updateTaskSSEStatus(jobId, connected, error = null, meta = {}) {
    const task = tasksMap.value.get(jobId)
    if (task) {
      const incomingAt = normalizeTimestamp(
        meta.updated_at ?? meta.updatedAt ?? meta.timestamp
      )
      const isServer = meta.isServer === true
      if (isServer && !applyUpdateTimestamp(task, incomingAt, true)) {
        return
      }
      task.sseConnected = connected
      // 如果是被踢下线的情况，记录错误提示，供 UI 展示弹窗/提示
      if (error) task.lastError = error
      else if (connected) task.lastError = null
      if (!isServer) {
        task.updatedAt = Date.now()
      }
    }
  }

  /**
   * 检查 SSE 连接状态
   */
  function checkSSEConnection() {
    if (Date.now() - lastHeartbeat.value > 30000) {  // 30 秒无心跳
      sseConnected.value = false
      console.warn('[UnifiedTaskStore] SSE 连接超时')
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
    if (!task) return
    const incomingSeq = normalizeStateSeq(meta.state_seq ?? updates.state_seq)
    if (updates.status !== undefined) {
      if (!applyStateSeqGuard(task, incomingSeq, { allowEqual: false })) {
        return
      }
    } else {
      const currentSeq = normalizeStateSeq(task.state_seq)
      if (incomingSeq > 0 && incomingSeq < currentSeq) {
        console.warn(
          `[UnifiedTaskStore] 忽略旧序号任务更新: job=${jobId}, incoming_seq=${incomingSeq}, current_seq=${currentSeq}`
        )
        return
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
      return
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
      return
    }

    if (updates.progress !== undefined) {
      applyProgressField(task, updates.progress, updates.status)
      delete updates.progress
    }

    Object.assign(task, updates)
    saveTasks()
    console.log(`[UnifiedTaskStore] 任务已更新: ${jobId}`, updates)
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
      if (snapshot.phase) task.phase = snapshot.phase
      if (snapshot.phase_percent !== undefined) {
        task.phase_percent = Math.round(snapshot.phase_percent * 10) / 10
      }
      if (snapshot.processed !== undefined) task.processed = snapshot.processed
      if (snapshot.total !== undefined) task.total = snapshot.total
      if (snapshot.language !== undefined) task.language = snapshot.language
      if (snapshot.created_time) task.createdAt = snapshot.created_time
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
      saveTasks()
      return true
    }

    addTask({
      job_id: jobId,
      filename: snapshot.filename,
      status: snapshot.status,
      progress: snapshot.progress,
      phase_percent: snapshot.phase_percent || 0,
      message: snapshot.message,
      phase: snapshot.phase,
      language: snapshot.language,
      processed: snapshot.processed || 0,
      total: snapshot.total || 0,
      createdAt: snapshot.created_time || Date.now(),
      updated_at: incomingAt,
      canceled_at: snapshot.canceled_at,
      state_seq: incomingSeq
    })
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
      console.error(`[UnifiedTaskStore] 任务未找到: ${jobId}`)
      return false
    }

    if (task.status !== TaskStatus.FINISHED) {
      console.warn(`[UnifiedTaskStore] 任务状态异常: ${task.status}`)
      return false
    }

    // 设置当前任务
    currentTask.value = task
    activeTaskId.value = jobId

    // 仅在用户触发加载时进入编辑器
    if (router.currentRoute.value.path !== `/editor/${jobId}`) {
      router.push(`/editor/${jobId}`)
    }

    console.log(`[UnifiedTaskStore] 任务已加载: ${jobId}`)
    return true
  }

  /**
   * 保存当前任务
   */
  async function saveCurrentTask() {
    if (!currentTask.value) {
      console.warn('[UnifiedTaskStore] 无当前任务')
      return
    }

    // 调用 ProjectStore 的保存逻辑
    const projectStore = useProjectStore()
    await projectStore.saveProject()

    currentTask.value.isDirty = false
    console.log(`[UnifiedTaskStore] 当前任务已保存: ${currentTask.value.job_id}`)
  }

  /**
   * 删除任务
   */
  function deleteTask(jobId) {
    tasksMap.value.delete(jobId)
    // 同时从队列顺序中删除
    queueOrder.value = queueOrder.value.filter(id => id !== jobId)
    saveTasks()
    console.log(`[UnifiedTaskStore] 任务已删除: ${jobId}`)
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
        console.error('[UnifiedTaskStore] 队列重排失败')
      } else {
        // 持久化新顺序
        saveTasks()
      }
    } catch (error) {
      // 恢复原顺序
      queueOrder.value = oldOrder
      console.error('[UnifiedTaskStore] 队列重排请求失败:', error)
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
        console.warn('[UnifiedTaskStore] 任务同步失败:', response)
        return false
      }

      const backendTasks = response.tasks || []
      console.log(`[UnifiedTaskStore] 从后端同步了 ${backendTasks.length} 个任务`)

      // 1. 获取后端任务ID集合
      const backendTaskIds = new Set(backendTasks.map(t => t.id))

      // 2. 删除前端有但后端没有的任务（幽灵任务清理）
      const localTaskIds = Array.from(tasksMap.value.keys())
      let deletedCount = 0
      for (const localId of localTaskIds) {
        if (!backendTaskIds.has(localId)) {
          console.log(`[UnifiedTaskStore] 删除幽灵任务: ${localId}`)
          tasksMap.value.delete(localId)
          deletedCount++
        }
      }
      if (deletedCount > 0) {
        console.log(`[UnifiedTaskStore] 共清理了 ${deletedCount} 个幽灵任务`)
      }

      // 3. 更新或添加后端任务
      let updatedCount = 0
      let addedCount = 0
      let skippedCount = 0
      for (const backendTask of backendTasks) {
        // V3.1.0: 过滤掉 filename 为空的任务，避免显示"未知任务"
        if (!backendTask.filename || backendTask.filename.trim() === '') {
          console.warn(`[UnifiedTaskStore] 跳过 filename 为空的任务: ${backendTask.id}`)
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
        console.log(`[UnifiedTaskStore] 队列顺序已同步: ${queueOrder.value.length} 个任务`)
      }

      console.log(
        `[UnifiedTaskStore] 任务同步完成: ${updatedCount} 个更新, ${addedCount} 个新增, ${deletedCount} 个删除` +
        (skippedCount > 0 ? `, ${skippedCount} 个跳过（filename为空）` : '')
      )
      saveTasks()
      return true
    } catch (error) {
      console.error('[UnifiedTaskStore] 任务同步失败:', error)
      return false
    }
  }

  /**
   * 清空所有任务
   */
  function clearAllTasks() {
    tasksMap.value.clear()
    saveTasks()
    console.log('[UnifiedTaskStore] 所有任务已清空')
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
        cleanedCount++
      }
    }

    if (cleanedCount > 0) {
      saveTasks()
      console.log(`[UnifiedTaskStore] 已清理 ${cleanedCount} 个过期任务`)
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
      console.error('[UnifiedTaskStore] 保存任务列表失败:', error)
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
          tasksArray.map(t => [
            t.job_id,
            {
              ...t,
              canceled_at: t.canceled_at || null,
              state_seq: normalizeStateSeq(t.state_seq),
              serverUpdatedAt: normalizeTimestamp(t.serverUpdatedAt) || 0
            }
          ])
        )
        console.log(`[UnifiedTaskStore] 已恢复 ${tasksArray.length} 个任务`)
      }

      // 恢复队列顺序
      const savedOrder = localStorage.getItem('queue-order')
      if (savedOrder) {
        queueOrder.value = JSON.parse(savedOrder)
        console.log(`[UnifiedTaskStore] 已恢复队列顺序: ${queueOrder.value.length} 个任务`)
      }
    } catch (error) {
      console.error('[UnifiedTaskStore] 恢复任务列表失败:', error)
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

    // 持久化
    saveTasks,
    restoreTasks
  }
})
