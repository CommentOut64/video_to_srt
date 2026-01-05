<script setup>
/**
 * App.vue - 应用根组件
 *
 * 职责：
 * - 启动时同步后端任务列表（修复幽灵任务）
 * - 全局 SSE 事件监听
 * - 自动跳转到编辑器（转录完成时）
 * - 全局任务状态同步
 * - 启动心跳服务（标签页重用机制）
 * - V3.1.1+dev.20260105.01: 启动时自动检查更新
 */
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useUnifiedTaskStore } from '@/stores/unifiedTaskStore'
import { useUpdateChecker } from '@/composables'
import sseChannelManager from '@/services/sseChannelManager'
import { heartbeatService } from '@/services/heartbeat'
import UpdateDialog from '@/components/UpdateDialog.vue'

const router = useRouter()
const route = useRoute()
const taskStore = useUnifiedTaskStore()

// V3.1.1+dev.20260105.01: 更新相关状态
const showUpdateDialog = ref(false)
const pendingUpdateInfo = ref(null)
const { checkForUpdate } = useUpdateChecker()

let unsubscribeGlobal = null

// V3.1.1+dev.20260105.01: 启动时检查更新
async function checkUpdateOnStartup() {
  console.log('[App] 步骤 0.5: 检查更新...')

  try {
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

  // V3.1.1+dev.20260105.01: 启动时检查更新（不阻塞其他初始化）
  checkUpdateOnStartup()

  // 第一步：从后端同步任务列表（第一阶段修复：数据同步）
  console.log('[App] 步骤 1: 从后端同步任务列表...')
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

  // 第二步：订阅全局 SSE 事件流（用于实时更新）
  console.log('[App] 步骤 2: 订阅全局 SSE 事件流...')
  unsubscribeGlobal = sseChannelManager.subscribeGlobal({
    onInitialState(state) {
      console.log('[App] 全局初始状态:', state)

      // 更新心跳（收到任何事件都说明连接正常）
      taskStore.updateSSEHeartbeat()

      // 同步队列顺序
      if (state.queue && Array.isArray(state.queue)) {
        taskStore.queueOrder = state.queue
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

          // 检查 store 中是否已有此任务
          const existingTask = taskStore.getTask(job.id)
          if (!existingTask) {
            // 添加新任务（包含完整信息）
            taskStore.addTask({
              job_id: job.id,
              filename: job.filename,
              status: job.status,
              progress: job.progress,
              message: job.message,
              phase: job.phase || (job.status === 'finished' ? 'editing' : 'transcribing'),
              createdAt: job.created_time || Date.now()
            })
          } else {
            // 更新现有任务
            taskStore.updateTask(job.id, {
              status: job.status,
              progress: job.progress,
              message: job.message,
              phase: job.phase || existingTask.phase,
              createdAt: job.created_time || existingTask.createdAt
            })
          }
        })
      }
    },

    onQueueUpdate(queue) {
      console.log('[App] 队列更新:', queue)

      // 更新心跳
      taskStore.updateSSEHeartbeat()

      // 更新队列顺序到 store
      if (Array.isArray(queue)) {
        taskStore.queueOrder = queue
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
        taskStore.updateTaskStatus(jobId, status, data.message || '')

        // 转录完成自动跳转到编辑器
        if (status === 'finished' && task.phase === 'transcribing') {
          console.log(`[App] 任务 ${jobId} 转录完成，准备跳转到编辑器`)

          // 更新阶段为编辑
          taskStore.updateTask(jobId, {
            phase: 'editing'
          })

          // 自动跳转到编辑器（仅在当前不在编辑器页面时）
          if (!route.path.startsWith('/editor/')) {
            console.log(`[App] 自动跳转到编辑器: /editor/${jobId}`)
            router.push(`/editor/${jobId}`)
          }
        }
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
      })
    },

    // [V3.1.0] 新增：任务删除事件处理，解决幽灵任务问题
    onJobRemoved(jobId) {
      console.log(`[App] 收到任务删除事件: ${jobId}`)

      // 更新心跳
      taskStore.updateSSEHeartbeat()

      // 从 store 中彻底移除任务
      taskStore.deleteTask(jobId)

      // 如果当前在该任务的编辑器页面，跳转回任务列表
      if (route.path === `/editor/${jobId}`) {
        console.log(`[App] 当前任务已删除，跳转回任务列表`)
        router.push('/tasks')
      }
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
})

onUnmounted(() => {
  console.log('[App] 应用卸载，关闭 SSE 连接')

  // 取消全局订阅
  if (unsubscribeGlobal) {
    unsubscribeGlobal()
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
