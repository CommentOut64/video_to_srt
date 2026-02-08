<template>
  <div class="task-list-view">
    <TaskListHeader
      @open-about="showAboutDialog = true"
      @open-upload="showUploadDialog = true"
      @exit-system="handleExit"
    />

    <TaskCardGrid
      :tasks="tasks"
      :thumbnail-cache="thumbnailCache"
      :editing-task-id="editingTaskId"
      :editing-title="editingTitle"
      :get-status-text="getStatusText"
      :format-date="formatDate"
      :get-task-display-name="getTaskDisplayName"
      @open-upload="showUploadDialog = true"
      @retry-thumbnail="(jobId) => getThumbnailUrl(jobId, true)"
      @update:editing-title="(value) => (editingTitle = value)"
      @finish-edit-title="finishEditTitle"
      @cancel-edit-title="cancelEditTitle"
      @title-click="handleTitleClick"
      @start-edit-title="startEditTitle"
      @open-editor="openEditor"
      @delete-task="deleteTask"
    />

    <TaskCreateDialog
      :show-upload-dialog="showUploadDialog"
      :upload-mode="uploadMode"
      :upload-files="uploadFiles"
      :input-files="inputFiles"
      :selected-files="selectedFiles"
      :loading-files="loadingFiles"
      :creating-batch="creatingBatch"
      :uploading="uploading"
      :show-advanced-settings="showAdvancedSettings"
      :task-config="taskConfig"
      :set-upload-ref="setUploadRef"
      :set-file-table-ref="setFileTableRef"
      :handle-file-change="handleFileChange"
      :remove-upload-file="removeUploadFile"
      :handle-file-selection-change="handleFileSelectionChange"
      :handle-row-click="handleRowClick"
      :format-file-size="formatFileSize"
      @update:show-upload-dialog="(value) => (showUploadDialog = value)"
      @update:upload-mode="(value) => (uploadMode = value)"
      @update:task-config="(value) => (taskConfig = value)"
      @open-input-folder="handleOpenInputFolder"
      @toggle-advanced-settings="showAdvancedSettings = !showAdvancedSettings"
      @preset-change="handlePresetChange"
      @close-upload-dialog="closeUploadDialog"
      @upload="handleUpload"
      @batch-create="handleBatchCreate"
    />

    <!-- 关于对话框 -->
    <AboutDialog v-model="showAboutDialog" />
  </div>
</template>

<script setup>
import { ref, computed, nextTick } from 'vue'
import { storeToRefs } from 'pinia'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox, ElLoading } from 'element-plus'
import { useUnifiedTaskStore } from '@/stores/unifiedTaskStore'
import { useTranscriptionConfigStore } from '@/stores/transcriptionConfigStore'
import { transcriptionApi, systemApi } from '@/services/api'
// V3.1.0: 移除 sseChannelManager 导入，SSE 订阅由 App.vue 统一管理
import AboutDialog from '@/components/AboutDialog.vue' // 关于对话框
import TaskListHeader from '@/components/task/TaskListHeader.vue'
import TaskCardGrid from '@/components/task/TaskCardGrid.vue'
import TaskCreateDialog from '@/components/task/TaskCreateDialog.vue'
import { useTaskUpload } from '@/composables/task-list/useTaskUpload'
import { useTaskThumbnail } from '@/composables/task-list/useTaskThumbnail'

const router = useRouter()
const taskStore = useUnifiedTaskStore()
const transcriptionConfigStore = useTranscriptionConfigStore()
const { taskConfig } = storeToRefs(transcriptionConfigStore)

const showAboutDialog = ref(false)
// 内联重命名相关
const editingTaskId = ref(null) // 当前正在编辑的任务ID
const editingTitle = ref('') // 编辑中的标题
const originalTitle = ref('') // 原始标题（用于恢复）

// 转录设置相关 - v3.5 预设模式
const showAdvancedSettings = ref(false) // 是否显示高级设置

// 计算属性 - 使用 computed 包装确保响应式
const tasks = computed(() => taskStore.tasks)

// V3.1.0: 移除 unsubscribeGlobalSSE，SSE 订阅由 App.vue 统一管理

const { thumbnailCache, getThumbnailUrl } = useTaskThumbnail({
  tasks,
  taskStore,
})

const {
  showUploadDialog,
  uploadMode,
  uploading,
  uploadRef,
  uploadFiles,
  inputFiles,
  selectedFiles,
  loadingFiles,
  creatingBatch,
  fileTableRef,
  loadInputFiles,
  handleOpenInputFolder,
  handleFileSelectionChange,
  handleBatchCreate,
  formatFileSize,
  handleFileChange,
  removeUploadFile,
  handleRowClick,
  closeUploadDialog,
  handleUpload,
} = useTaskUpload({
  taskStore,
  taskConfig,
  getThumbnailUrl,
})

function setUploadRef(element) {
  uploadRef.value = element
}

function setFileTableRef(element) {
  fileTableRef.value = element
}

// 打开编辑器
function openEditor(jobId) {
  router.push(`/editor/${jobId}`)
}

// 处理标题单击 - 用于区分单击（跳转）和双击（重命名）
let clickTimer = null
function handleTitleClick(task) {
  if (clickTimer) {
    // 双击时清除单击定时器，让 dblclick 事件处理
    clearTimeout(clickTimer)
    clickTimer = null
    return
  }

  // 延迟执行单击操作，给双击留出时间
  clickTimer = setTimeout(() => {
    clickTimer = null
    openEditor(task.job_id)
  }, 200)
}

// 删除任务
async function deleteTask(jobId) {
  try {
    await ElMessageBox.confirm('确定要删除这个任务吗？此操作无法撤销。', '确认删除', {
      confirmButtonText: '删除',
      cancelButtonText: '取消',
      type: 'warning',
    })

    // 调用后端 API 删除任务数据
    try {
      const res = await transcriptionApi.cancelJob(jobId, true)
      // 后端现在区分“正在执行，等待取消后删除”与“已立即删除”
      if (res.pending_delete) {
        if (res.task) {
          taskStore.applyTaskSnapshot(res.task)
        } else {
          taskStore.updateTaskStatus(
            jobId,
            'canceling',
            res.message || '已请求取消并将在结束后删除',
            { isServer: false }
          )
        }
        ElMessage.info(res.message || '任务正在执行，已请求取消，结束后将删除')
        // 任务会在原子段结束后自动删除，此处不删卡片
      } else {
        // 非运行中，已立即删除
        taskStore.deleteTask(jobId)
        ElMessage.success('任务已删除')
      }
      // 无论哪种情况，刷新 input 列表以反映文件变化
      await loadInputFiles()
      setTimeout(() => {
        taskStore.syncTasksFromBackend()
      }, 500)
    } catch (error) {
      // 处理占用/正在执行的快速失败
      const status = error?.response?.status
      const detail = error?.response?.data?.detail || error?.message
      if (status === 409) {
        ElMessage.info(detail || '任务正在取消中，稍后再试删除')
        taskStore.updateTaskStatus(jobId, 'canceling', detail, { isServer: false })
        return
      }
      if (status === 423) {
        ElMessage.warning(detail || '当前有进程占用，请稍后再试')
        return
      }
      console.error('删除任务失败:', error)
      ElMessage.error(`删除失败: ${detail || '未知错误'}`)
    }
  } catch (error) {
    if (error !== 'cancel') {
      console.error('删除任务失败:', error)
      ElMessage.error(`删除失败: ${error.message}`)
    }
  }
}

// 格式化日期 - 显示为 YYYY-MM-DD HH:mm 格式（第二阶段修复：实时更新）
function formatDate(timestamp) {
  if (!timestamp) return ''
  const date = new Date(timestamp)

  // 格式化为 YYYY-MM-DD HH:mm
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  const hours = String(date.getHours()).padStart(2, '0')
  const minutes = String(date.getMinutes()).padStart(2, '0')

  return `${year}-${month}-${day} ${hours}:${minutes}`
}

// 获取状态文本（与后端状态枚举保持一致）
function getStatusText(status) {
  const statusMap = {
    created: '已创建',
    queued: '排队中',
    processing: '转录中',
    pausing: '正在暂停...',
    paused: '已暂停',
    canceling: '正在取消...', // V3.1.0
    force_canceled: '已强制取消', // V3.1.0
    finished: '已完成',
    failed: '失败',
    canceled: '已取消',
  }
  return statusMap[status] || status
}

// 去除文件扩展名（优先显示 title，否则显示 filename）
function getTaskDisplayName(task) {
  // 优先显示用户自定义的 title
  if (task.title) {
    return task.title
  }

  // 否则显示文件名（去除扩展名）
  const filename = task.filename || ''
  if (!filename) return ''

  // 去除文件扩展名
  const lastDotIndex = filename.lastIndexOf('.')
  if (lastDotIndex > 0) {
    return filename.substring(0, lastDotIndex)
  }
  return filename
}

// 开始编辑任务标题
function startEditTitle(task) {
  editingTaskId.value = task.job_id
  editingTitle.value = task.title || getTaskDisplayName(task)
  originalTitle.value = editingTitle.value

  // 等待 DOM 更新后聚焦输入框
  nextTick(() => {
    const inputs = document.querySelectorAll('.task-title-input')
    const input = Array.from(inputs).find(
      (el) => el.closest('.task-card')?.querySelector('.task-title-input') === el
    )
    if (input) {
      input.focus()
      input.select()
    }
  })
}

// 完成编辑任务标题
async function finishEditTitle(task) {
  if (editingTaskId.value !== task.job_id) return

  const newTitle = editingTitle.value.trim()

  // 如果标题为空，提示并恢复原名称
  if (!newTitle) {
    ElMessage.warning('任务名称不能为空')
    editingTitle.value = originalTitle.value
    editingTaskId.value = null
    return
  }

  // 如果没有变化，直接关闭编辑
  if (newTitle === originalTitle.value) {
    editingTaskId.value = null
    return
  }

  try {
    // 调用 API 重命名任务
    const result = await transcriptionApi.renameJob(task.job_id, newTitle)

    // 更新本地 store
    if (result?.task) {
      taskStore.applyTaskSnapshot(result.task, {
        updated_at: result.task.updated_at ?? result.updated_at,
      })
    } else {
      taskStore.updateTask(task.job_id, { title: newTitle }, { updated_at: result?.updated_at })
    }

    ElMessage.success('重命名成功')
  } catch (error) {
    console.error('重命名任务失败:', error)
    ElMessage.error(`重命名失败: ${error.message || '未知错误'}`)
    // 恢复原名称
    editingTitle.value = originalTitle.value
  } finally {
    editingTaskId.value = null
  }
}

// 取消编辑
function cancelEditTitle() {
  editingTitle.value = originalTitle.value
  editingTaskId.value = null
}

// 处理退出系统
async function handleExit() {
  try {
    await ElMessageBox.confirm('确定要退出系统吗？所有更改都会自动保存', '确认退出', {
      confirmButtonText: '确定退出',
      cancelButtonText: '取消',
      type: 'warning',
    })

    // 显示关闭进度
    const loading = ElLoading.service({
      text: '正在保存断点并关闭系统...',
      background: 'rgba(0, 0, 0, 0.7)',
    })

    let shutdownSuccess = false
    let cleanupReport = null

    try {
      // 调用后端 shutdown API，设置超时
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 10000) // 10秒超时

      try {
        const response = await systemApi.shutdownSystem({
          cleanup_temp: false, // 不清理临时文件，保留断点数据
          force: false,
        })

        clearTimeout(timeoutId)
        shutdownSuccess = response?.success || false
        cleanupReport = response?.cleanup_report || null

        if (cleanupReport) {
          console.log('[Exit] 清理报告:', cleanupReport)
        }
      } catch (e) {
        clearTimeout(timeoutId)
        // 请求可能因后端关闭而失败，这是预期行为
        console.log('[Exit] 后端已关闭或请求超时:', e.message || e)
        shutdownSuccess = true // 如果后端已关闭，认为成功
      }
    } catch (e) {
      console.log('[Exit] 关闭请求异常:', e)
      shutdownSuccess = true // 即使异常也认为成功（后端可能已关闭）
    }

    loading.close()

    // 显示关闭完成提示
    const message = shutdownSuccess
      ? '系统已安全关闭，请手动关闭此浏览器标签页。'
      : '系统关闭可能未完全成功，请手动检查后台进程。'

    await ElMessageBox.alert(message, '关闭完成', { type: shutdownSuccess ? 'success' : 'warning' })

    // 尝试关闭当前窗口（部分浏览器可能阻止）
    try {
      window.close()
    } catch (e) {
      // 忽略关闭窗口失败
    }
  } catch (error) {
    // 用户取消退出
    if (error !== 'cancel') {
      console.error('[Exit] 退出失败:', error)
    }
  }
}
</script>

<style>
.task-list-view {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background: var(--af-bg-primary);
}

:deep(.el-message-box) .el-message-box__btns .el-button {
  background: transparent;
  border: 1px solid transparent;
  color: var(--af-text-secondary);
}

:deep(.el-message-box) .el-message-box__btns .el-button:hover {
  background: var(--af-bg-tertiary);
  border-color: var(--af-border-default);
  color: var(--af-text-primary);
}

:deep(.el-message-box) .el-message-box__btns .el-button:active {
  background: var(--af-bg-quaternary);
}

:deep(.el-message-box) .el-message-box__btns .el-button--primary {
  background: transparent;
  border-color: transparent;
  color: var(--af-accent-primary);
}

:deep(.el-message-box) .el-message-box__btns .el-button--primary:hover {
  background: var(--af-bg-tertiary);
  border-color: var(--af-accent-primary);
}

:deep(.el-message-box) {
  background: var(--af-bg-secondary);
  border: 1px solid var(--af-border-default);
}

:deep(.el-message-box) .el-message-box__header {
  padding: 16px 20px 12px;
}

:deep(.el-message-box) .el-message-box__title {
  padding-left: 8px;
  color: var(--af-text-primary);
}

:deep(.el-message-box) .el-message-box__content {
  padding: 12px 20px;
  color: var(--af-text-secondary);
}

:deep(.el-message-box) .el-message-box__btns {
  padding: 12px 20px 16px;
}
</style>
