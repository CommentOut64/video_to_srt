<script setup>
/**
 * App.vue - 应用根组件
 *
 * 职责：
 * - 启动时同步后端任务列表（修复幽灵任务）
 * - 全局 SSE 事件监听
 * - 全局任务状态同步
 * - 启动心跳服务（标签页重用机制）
 * - V3.1.1+dev.20260105.01: 启动时自动检查更新
 * - V3.1.1+dev.20260106.01: 使用 sessionStorage 防止会话内重复检查更新
 */
import { ref, onMounted, onUnmounted } from 'vue'
import { useUnifiedTaskStore } from '@/stores/unifiedTaskStore'
import { useProgressStore } from '@/stores/progressStore'
import { useUpdateChecker } from '@/composables'
import sseChannelManager from '@/services/sseChannelManager'
import { heartbeatService } from '@/services/heartbeat'
import UpdateDialog from '@/components/UpdateDialog.vue'

const taskStore = useUnifiedTaskStore()
const progressStore = useProgressStore()

// V3.1.1+dev.20260105.01: 更新相关状态
const showUpdateDialog = ref(false)
const pendingUpdateInfo = ref(null)
const { checkForUpdate } = useUpdateChecker()

let unsubscribeGlobal = null
let syncTimer = null

// V3.1.1+dev.20260106.01: sessionStorage key，用于防止会话内重复检查更新
const UPDATE_CHECK_SESSION_KEY = 'anchorflux_update_checked_this_session'

// V3.1.1+dev.20260106.01: 启动时检查更新（每个会话只检查一次）
async function checkUpdateOnStartup() {
  // 检查本次会话是否已经检查过更新
  try {
    if (sessionStorage.getItem(UPDATE_CHECK_SESSION_KEY)) {
      console.log('[App] 本次会话已检查过更新，跳过')
      return
    }
  } catch {
    // sessionStorage 不可用时继续执行
  }

  console.log('[App] 步骤 0.5: 检查更新...')

  try {
    // 标记本次会话已检查（在请求前标记，避免并发问题）
    try {
      sessionStorage.setItem(UPDATE_CHECK_SESSION_KEY, '1')
    } catch {
      // 忽略
    }

    // 自动检查时会遵守忽略版本规则
    const result = await checkForUpdate(false)

    if (result.hasUpdate && result.updateInfo) {
      console.log('[App] 发现新版本:', result.updateInfo.latestVersion)
      pendingUpdateInfo.value = result.updateInfo

      // 等待 2 秒后自动弹出更新窗口
      setTimeout(() => {
        showUpdateDialog.value = true
      }, 2000)
    } else {
      console.log('[App] 当前已是最新版本或该版本已被忽略')
    }
  } catch (e) {
    console.warn('[App] 检查更新失败:', e)
    // 静默失败，不影响用户使用
  }
}

function startPeriodicSync() {
  if (syncTimer) {
    clearInterval(syncTimer)
  }
  syncTimer = setInterval(async () => {
    const needsSync =
      taskStore.hasStaleState('canceling', 30000) ||
      !sseChannelManager.isGlobalHealthy()
    if (!needsSync) return

    console.log('[App] 检测到异常状态，触发兜底同步')
    try {
      await taskStore.syncTasksFromBackend()
    } catch (error) {
      console.warn('[App] 兜底同步失败:', error)
    }
  }, 60000)
}

onMounted(async () => {
  console.log('[App] 应用已挂载，执行初始化')

  // 第零步：启动心跳服务
  console.log('[App] 步骤 0: 启动心跳服务...')
  try {
    await heartbeatService.start()
    console.log('[App] 心跳服务已启动')
  } catch (error) {
    console.error('[App] 心跳服务启动失败:', error)
  }

  // V3.1.1+dev.20260106.01: 启动时检查更新（每个会话只检查一次，不阻塞其他初始化）
  checkUpdateOnStartup()

  // 第一步：订阅全局 SSE 事件流（先订阅避免和 HTTP 同步竞态）
  console.log('[App] 步骤 1: 订阅全局 SSE 事件流...')
  unsubscribeGlobal = sseChannelManager.subscribeGlobal({
    onInitialState(state) {
      console.log('[App] 全局初始状态:', state)

      // 更新心跳（收到任何事件都说明连接正常）
      taskStore.updateSSEHeartbeat()

      // 同步队列顺序
      if (state.queue && Array.isArray(state.queue)) {
        taskStore.applyQueueOrder(state.queue, { timestamp: state.queue_updated_at })
        console.log(`[App] 初始队列顺序已同步: ${state.queue.length} 个任务`)
      }

      // 同步任务列表到 store（第二阶段修复：实时更新）
      if (state.jobs && Array.isArray(state.jobs)) {
        state.jobs.forEach(job => {
          // V3.1.0: 过滤掉 filename 为空的任务，避免显示"未知任务"
          if (!job.filename || job.filename.trim() === '') {
            console.warn(`[App] 跳过 filename 为空的任务: ${job.id}`)
            return
          }

          taskStore.applyTaskSnapshot(job)
        })
      }
    },

    onQueueUpdate(queue, data) {
      console.log('[App] 队列更新:', queue)

      // 更新心跳
      taskStore.updateSSEHeartbeat()

      // 更新队列顺序到 store
      if (Array.isArray(queue)) {
        taskStore.applyQueueOrder(queue, {
          updated_at: data?.updated_at ?? data?.timestamp
        })
        console.log(`[App] 队列顺序已更新: ${queue.length} 个任务`)
      }
    },

    onJobStatus(jobId, status, data) {
      console.log(`[App] 任务 ${jobId} 状态变化:`, status, data)

      // 更新心跳
      taskStore.updateSSEHeartbeat()

      // 更新 store 中的任务状态
      const task = taskStore.getTask(jobId)
      if (task) {
        // V3.1.0: onJobStatus 只更新 status 和 message，不更新 progress
        // 避免后端推送的低进度（如恢复时的 0）覆盖前端已有的高进度
        // progress 的更新由 onJobProgress 专门负责
        taskStore.updateTaskStatus(jobId, status, data.message || '', {
          updated_at: data.updated_at ?? data.timestamp,
          state_seq: data.state_seq
        })
        progressStore.markStatus(jobId, status, {
          updated_at: data.updated_at ?? data.timestamp,
          state_seq: data.state_seq,
          message: data.message,
          phase: data.phase,
          phase_percent: data.phase_percent
        })

      }
    },

    onJobProgress(jobId, percent, data) {
      console.log(`[App] 任务 ${jobId} 进度:`, percent)

      // 更新心跳
      taskStore.updateSSEHeartbeat()

      // 更新 store 中的任务进度（实时更新卡片），传递完整数据
      taskStore.updateTaskProgress(jobId, percent, data.status, {
        phase: data.phase,
        phase_percent: data.phase_percent,
        message: data.message,
        processed: data.processed,
        total: data.total,
        language: data.language
      }, {
        updated_at: data.updated_at ?? data.timestamp,
        state_seq: data.state_seq
      })
    },
    onJobRenamed(data) {
      console.log('[App] 任务重命名:', data.job_id, data.title)

      const updates = {}
      if (data.title !== undefined) updates.title = data.title
      if (data.filename !== undefined) updates.filename = data.filename

      taskStore.updateTask(data.job_id, updates, {
        updated_at: data.updated_at ?? data.timestamp
      })
    },

    // [V3.1.0] 新增：任务删除事件处理，解决幽灵任务问题
    onJobRemoved(jobId) {
      console.log(`[App] 收到任务删除事件: ${jobId}`)

      // 更新心跳
      taskStore.updateSSEHeartbeat()

      // 从 store 中彻底移除任务
      taskStore.deleteTask(jobId)
      progressStore.clearJobState(jobId)

    },

    onConnected(data) {
      console.log('[App] 全局 SSE 连接成功:', data)

      // 更新心跳
      taskStore.updateSSEHeartbeat()

      // 标记所有 processing 状态的任务为 SSE 已连接
      taskStore.tasks.forEach(task => {
        if (task.status === 'processing') {
          taskStore.updateTaskSSEStatus(task.job_id, true)
        }
      })
    },

    onPing() {
      // 收到心跳 ping，更新心跳时间戳
      taskStore.updateSSEHeartbeat()
    }
  })

  // 第二步：HTTP 同步任务列表（兜底，避免漏事件）
  console.log('[App] 步骤 2: 从后端同步任务列表...')
  try {
    const syncSuccess = await taskStore.syncTasksFromBackend()
    if (syncSuccess) {
      console.log('[App] 任务列表同步成功')
    } else {
      console.warn('[App] 任务列表同步失败，将使用本地 localStorage 数据')
    }
  } catch (error) {
    console.error('[App] 任务列表同步异常:', error)
  }

  // 第三步：启动低频兜底同步
  startPeriodicSync()
})

onUnmounted(() => {
  console.log('[App] 应用卸载，关闭 SSE 连接')

  // 取消全局订阅
  if (unsubscribeGlobal) {
    unsubscribeGlobal()
  }
  if (syncTimer) {
    clearInterval(syncTimer)
    syncTimer = null
  }

  // 停止心跳服务
  heartbeatService.stop()
})
</script>

<template>
  <router-view />

  <!-- V3.1.1+dev.20260105.01: 全局更新窗口 -->
  <UpdateDialog
    v-model="showUpdateDialog"
    :update-info="pendingUpdateInfo"
  />
</template>

<style scoped>
/* 无作用域样式 */
</style>
