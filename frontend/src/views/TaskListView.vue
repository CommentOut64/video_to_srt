<template>
  <div class="task-list-view">
    <TaskListHeader
      @open-about="showAboutDialog = true"
      @open-upload="showUploadDialog = true"
      @open-import="showImportDialog = true"
      @exit-system="handleExit"
    />

    <div class="task-workspace">
      <div
        class="sidebar-edge-zone"
        @mouseenter="isEdgeHovered = true"
        @mouseleave="isEdgeHovered = false"
      >
        <button
          v-if="!isSidebarOpen"
          class="sidebar-edge-btn"
          :class="{ visible: isEdgeHovered }"
          title="展开任务侧栏"
          @click="isSidebarOpen = true"
        >
          <el-icon><ArrowRightBold /></el-icon>
        </button>
      </div>

      <aside class="task-sidebar" :class="{ open: isSidebarOpen }">
        <button
          v-if="isSidebarOpen"
          class="sidebar-collapse-btn"
          title="折叠任务侧栏"
          @click="isSidebarOpen = false"
        >
          <el-icon><ArrowLeftBold /></el-icon>
        </button>

        <div class="sidebar-inner">
          <section class="sidebar-block">
            <div class="block-title">
              <el-icon><Grid /></el-icon>
              <span>展示模式</span>
            </div>
            <div class="mode-switch">
              <button
                class="mode-btn"
                :class="{ active: displayMode === 'card' }"
                @click="setDisplayMode('card')"
              >
                卡片
              </button>
              <button
                class="mode-btn"
                :class="{ active: displayMode === 'list' }"
                @click="setDisplayMode('list')"
              >
                列表
              </button>
            </div>
          </section>

          <section class="sidebar-block">
            <div class="block-title">
              <el-icon><Operation /></el-icon>
              <span>排序模式</span>
            </div>
            <div class="mode-switch">
              <button
                class="mode-btn"
                :class="{ active: sortMode === 'grouped' }"
                @click="setSortMode('grouped')"
              >
                分组
              </button>
              <button
                class="mode-btn"
                :class="{ active: sortMode === 'edited_time' }"
                @click="setSortMode('edited_time')"
              >
                编辑时间
              </button>
            </div>
          </section>

          <section class="sidebar-block">
            <div class="block-title">
              <el-icon><CollectionTag /></el-icon>
              <span>重要设置</span>
            </div>

            <label class="setting-row">
              <span>显示已取消任务</span>
              <el-switch v-model="showCanceledTasks" />
            </label>

            <label class="setting-row">
              <span>完成区默认折叠</span>
              <el-switch v-model="collapseCompletedByDefault" />
            </label>

            <label class="setting-row">
              <span>中断区自动展开</span>
              <el-switch v-model="autoExpandInterrupted" />
            </label>
          </section>

          <section class="sidebar-block sidebar-block--stats">
            <div class="stats-item">
              <span class="stats-label">工作区</span>
              <span class="stats-value">{{ workspaceCount }}</span>
            </div>
            <div class="stats-item">
              <span class="stats-label">中断区</span>
              <span class="stats-value">{{ interruptedCount }}</span>
            </div>
            <div class="stats-item">
              <span class="stats-label">完成区</span>
              <span class="stats-value">{{ completedCount }}</span>
            </div>
          </section>

          <section class="sidebar-footer">
            <el-button class="more-settings-btn" type="primary" plain @click="openAdvancedSettingsDialog">
              <el-icon><Setting /></el-icon>
              更多设置
            </el-button>
          </section>
        </div>
      </aside>

      <TaskCardGrid
        :tasks="tasks"
        :grouped-sections="groupedSections"
        :sorted-tasks="tasksByEditedTime"
        :display-mode="displayMode"
        :sort-mode="sortMode"
        :group-collapse-state="groupCollapseState"
        :thumbnail-cache="thumbnailCache"
        :editing-task-id="editingTaskId"
        :editing-title="editingTitle"
        :get-status-text="getStatusText"
        :format-date="formatDate"
        :get-task-display-name="getTaskDisplayName"
        @open-upload="showUploadDialog = true"
        @open-import="showImportDialog = true"
        @retry-thumbnail="(jobId) => getThumbnailUrl(jobId, true)"
        @update:editing-title="(value) => (editingTitle = value)"
        @finish-edit-title="finishEditTitle"
        @cancel-edit-title="cancelEditTitle"
        @title-click="handleTitleClick"
        @start-edit-title="startEditTitle"
        @open-editor="openEditor"
        @delete-task="deleteTask"
        @toggle-group="toggleGroupCollapse"
      />
    </div>

    <TaskCreateDialog
      :show-upload-dialog="showUploadDialog"
      :upload-mode="uploadMode"
      :upload-files="uploadFiles"
      :input-files="inputFiles"
      :selected-files="selectedFiles"
      :loading-files="loadingFiles"
      :creating-batch="creatingBatch"
      :uploading="uploading"
      :task-config="taskConfig"
      :custom-presets="customPresets"
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
      @save-preset="handleSavePreset"
      @delete-preset="handleDeletePreset"
      @overwrite-preset="handleOverwritePreset"
      @close-upload-dialog="closeUploadDialog"
      @upload="handleUpload"
      @batch-create="handleBatchCreate"
    />

    <ImportDialog
      v-model:show-dialog="showImportDialog"
      @import-success="handleImportSuccess"
    />

    <el-dialog
      v-model="showAdvancedSettings"
      title="高级设置"
      width="600px"
      :close-on-click-modal="false"
      :close-on-press-escape="true"
      @close="handleCloseAdvancedSettings"
    >
      <AdvancedSettings v-model="advancedConfig" />
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="handleCancelAdvancedSettings">取消</el-button>
          <el-button type="primary" @click="handleSaveAdvancedSettings">保存设置</el-button>
        </span>
      </template>
    </el-dialog>

    <AboutDialog v-model="showAboutDialog" />
  </div>
</template>

<script setup>
import { ref, computed, nextTick, onMounted, watch } from 'vue'
import { storeToRefs } from 'pinia'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage, ElMessageBox, ElLoading } from 'element-plus'
import {
  ArrowLeftBold,
  ArrowRightBold,
  CollectionTag,
  Grid,
  Operation,
  Setting,
} from '@element-plus/icons-vue'
import { useTaskRuntimeStore } from '@/stores/taskRuntimeStore'
import { useTranscriptionPresetStore } from '@/stores/transcriptionPresetStore'
import { selectCapabilities } from '@/state/capabilities/capabilitySelector'
import { transcriptionApi, systemApi, presetsApi } from '@/services/api'
import AboutDialog from '@/components/AboutDialog.vue'
import TaskListHeader from '@/components/task/TaskListHeader.vue'
import TaskCardGrid from '@/components/task/TaskCardGrid.vue'
import TaskCreateDialog from '@/components/task/TaskCreateDialog.vue'
import ImportDialog from '@/components/task/ImportDialog.vue'
import AdvancedSettings from '@/components/editor/AdvancedSettings.vue'
import { useTaskUpload } from '@/composables/task-list/useTaskUpload'
import { useTaskThumbnail } from '@/composables/task-list/useTaskThumbnail'
import { navigateToEditor } from '@/utils/editorNavigation'

const DISPLAY_MODE_KEY = 'task-ui-display-mode'
const SORT_MODE_KEY = 'task-ui-sort-mode'
const GROUP_COLLAPSE_KEY = 'task-ui-group-collapse'
const SIDEBAR_SETTINGS_KEY = 'task-ui-sidebar-settings'
const ADVANCED_GENERAL_KEY = 'task-advanced-general'
const CUSTOM_PRESETS_KEY = 'user-custom-presets'

const router = useRouter()
const route = useRoute()
const taskStore = useTaskRuntimeStore()
const transcriptionPresetStore = useTranscriptionPresetStore()
const { taskConfig } = storeToRefs(transcriptionPresetStore)
const capabilities = selectCapabilities()

const showAboutDialog = ref(false)
const showImportDialog = ref(false)
const showAdvancedSettings = ref(false)
const isSidebarOpen = ref(false)
const isEdgeHovered = ref(false)

const displayMode = ref('card')
const sortMode = ref('grouped')
const groupCollapseState = ref({
  workspace: false,
  interrupted: false,
  completed: true,
})

const showCanceledTasks = ref(true)
const collapseCompletedByDefault = ref(true)
const autoExpandInterrupted = ref(true)

// 内联重命名相关
const editingTaskId = ref(null) // 当前正在编辑的任务ID
const editingTitle = ref('') // 编辑中的标题
const originalTitle = ref('') // 原始标题（用于恢复）

// V3.2.4: 自定义预设（Tab 化重构后，showAdvancedSettings 已移除）
const customPresets = ref([])
const advancedGeneral = ref({
  global_time_offset: 0,
  duration_adjust: 0,
  auto_save_interval: 60,
  preview_font_size: 24,
  enable_shortcuts: true,
})
const advancedConfig = ref(buildAdvancedConfig())

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

function parseJsonStorage(key, fallbackValue) {
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return fallbackValue
    return JSON.parse(raw)
  } catch {
    return fallbackValue
  }
}

function saveJsonStorage(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch (error) {
    console.warn(`保存 ${key} 失败:`, error)
  }
}

function normalizeTime(value) {
  const num = Number(value)
  if (!Number.isFinite(num) || num <= 0) return 0
  return num
}

function getActivityTime(task) {
  return (
    normalizeTime(task.serverUpdatedAt) ||
    normalizeTime(task.updatedAt) ||
    normalizeTime(task.createdAt)
  )
}

function getPauseTime(task) {
  return normalizeTime(task.paused_at) || getActivityTime(task)
}

function getFailedTime(task) {
  return normalizeTime(task.failed_at) || getActivityTime(task)
}

function getEndTime(task) {
  return (
    normalizeTime(task.completed_at) ||
    normalizeTime(task.canceled_at) ||
    getActivityTime(task)
  )
}

function shouldIncludeTask(task) {
  if (!task || task.status === 'removed') return false
  if (!showCanceledTasks.value && ['canceled', 'force_canceled'].includes(task.status)) {
    return false
  }
  return true
}

const visibleTasks = computed(() => tasks.value.filter(shouldIncludeTask))

const queueOrderMap = computed(() => {
  const map = new Map()
  taskStore.queueOrder.forEach((jobId, index) => {
    map.set(jobId, index)
  })
  return map
})

function compareQueuedTask(a, b) {
  const orderA = queueOrderMap.value.has(a.job_id)
    ? queueOrderMap.value.get(a.job_id)
    : Number.MAX_SAFE_INTEGER
  const orderB = queueOrderMap.value.has(b.job_id)
    ? queueOrderMap.value.get(b.job_id)
    : Number.MAX_SAFE_INTEGER

  if (orderA !== orderB) {
    return orderA - orderB
  }
  return getActivityTime(b) - getActivityTime(a)
}

const groupedSections = computed(() => {
  const processingTasks = visibleTasks.value
    .filter((task) => task.status === 'processing')
    .sort((a, b) => getActivityTime(b) - getActivityTime(a))

  const queuedTasks = visibleTasks.value
    .filter((task) => task.status === 'queued')
    .sort(compareQueuedTask)

  const pausedCreatedTasks = visibleTasks.value
    .filter((task) => ['paused', 'created'].includes(task.status))
    .sort((a, b) => getPauseTime(b) - getPauseTime(a))

  const failedTasks = visibleTasks.value
    .filter((task) => task.status === 'failed')
    .sort((a, b) => getFailedTime(b) - getFailedTime(a))

  const completedTasks = visibleTasks.value
    .filter((task) => ['finished', 'canceled', 'force_canceled'].includes(task.status))
    .sort((a, b) => getEndTime(b) - getEndTime(a))

  return [
    {
      key: 'workspace',
      title: '工作区',
      description: '运行 / 排队 / 暂停',
      variant: 'primary',
      tasks: [...processingTasks, ...queuedTasks, ...pausedCreatedTasks],
    },
    {
      key: 'interrupted',
      title: '中断区',
      description: '失败',
      variant: 'danger',
      tasks: failedTasks,
    },
    {
      key: 'completed',
      title: '完成区',
      description: showCanceledTasks.value ? '完成 / 取消' : '完成',
      variant: 'success',
      tasks: completedTasks,
    },
  ].filter((section) => section.tasks.length > 0)
})

const tasksByEditedTime = computed(() =>
  visibleTasks.value
    .slice()
    .sort((a, b) => getActivityTime(b) - getActivityTime(a))
)

const workspaceCount = computed(
  () => groupedSections.value.find((section) => section.key === 'workspace')?.tasks.length || 0
)
const interruptedCount = computed(
  () => groupedSections.value.find((section) => section.key === 'interrupted')?.tasks.length || 0
)
const completedCount = computed(
  () => groupedSections.value.find((section) => section.key === 'completed')?.tasks.length || 0
)

function toggleGroupCollapse(groupKey) {
  if (!groupKey) return
  const currentValue = !!groupCollapseState.value[groupKey]
  groupCollapseState.value = {
    ...groupCollapseState.value,
    [groupKey]: !currentValue,
  }
}

function setDisplayMode(mode) {
  if (!['card', 'list'].includes(mode)) return
  displayMode.value = mode
}

function setSortMode(mode) {
  if (!['grouped', 'edited_time'].includes(mode)) return
  sortMode.value = mode
  if (mode === 'grouped' && collapseCompletedByDefault.value) {
    groupCollapseState.value = {
      ...groupCollapseState.value,
      completed: true,
    }
  }
}

function loadTaskUiPreferences() {
  const savedDisplay = localStorage.getItem(DISPLAY_MODE_KEY)
  const savedSort = localStorage.getItem(SORT_MODE_KEY)
  const savedCollapse = parseJsonStorage(GROUP_COLLAPSE_KEY, null)
  const savedSidebarSettings = parseJsonStorage(SIDEBAR_SETTINGS_KEY, null)
  const savedAdvancedGeneral = parseJsonStorage(ADVANCED_GENERAL_KEY, null)

  if (savedDisplay === 'card' || savedDisplay === 'list') {
    displayMode.value = savedDisplay
  }
  if (savedSort === 'grouped' || savedSort === 'edited_time') {
    sortMode.value = savedSort
  }
  if (savedCollapse && typeof savedCollapse === 'object') {
    groupCollapseState.value = {
      workspace: !!savedCollapse.workspace,
      interrupted: !!savedCollapse.interrupted,
      completed: !!savedCollapse.completed,
    }
  }
  if (savedSidebarSettings && typeof savedSidebarSettings === 'object') {
    showCanceledTasks.value = savedSidebarSettings.showCanceledTasks !== false
    collapseCompletedByDefault.value = savedSidebarSettings.collapseCompletedByDefault !== false
    autoExpandInterrupted.value = savedSidebarSettings.autoExpandInterrupted !== false
  }
  if (savedAdvancedGeneral && typeof savedAdvancedGeneral === 'object') {
    advancedGeneral.value = {
      ...advancedGeneral.value,
      ...savedAdvancedGeneral,
    }
  }
}

watch(displayMode, (value) => {
  localStorage.setItem(DISPLAY_MODE_KEY, value)
})

watch(sortMode, (value) => {
  localStorage.setItem(SORT_MODE_KEY, value)
})

watch(
  groupCollapseState,
  (value) => {
    saveJsonStorage(GROUP_COLLAPSE_KEY, value)
  },
  { deep: true }
)

watch(
  [showCanceledTasks, collapseCompletedByDefault, autoExpandInterrupted],
  ([showCanceled, collapseCompleted, autoExpand]) => {
    saveJsonStorage(SIDEBAR_SETTINGS_KEY, {
      showCanceledTasks: showCanceled,
      collapseCompletedByDefault: collapseCompleted,
      autoExpandInterrupted: autoExpand,
    })
  }
)

watch(collapseCompletedByDefault, (value) => {
  if (value) {
    groupCollapseState.value = {
      ...groupCollapseState.value,
      completed: true,
    }
  }
})

watch(interruptedCount, (newCount, oldCount) => {
  if (!autoExpandInterrupted.value) return
  if (newCount > 0 && newCount > oldCount) {
    groupCollapseState.value = {
      ...groupCollapseState.value,
      interrupted: false,
    }
  }
})

// V3.2.4: 保存自定义预设 (Write-Through: localStorage + 后端)
async function handleSavePreset() {
  const name = prompt('请输入预设名称')
  if (!name || !name.trim()) return

  const presetConfig = {
    preprocessing: { ...taskConfig.value.preprocessing },
    transcription: { ...taskConfig.value.transcription },
    refinement: { ...taskConfig.value.refinement },
    compute: { ...taskConfig.value.compute },
  }

  try {
    const res = await presetsApi.createCustomPreset(name.trim(), presetConfig)
    if (res?.success && res.preset) {
      customPresets.value = [...customPresets.value, res.preset]
      saveCustomPresetsToStorage()
      ElMessage.success(`预设"${name.trim()}"已保存`)
    }
  } catch {
    // 后端失败时 fallback 到纯本地保存
    const preset = {
      id: 'custom_' + Date.now(),
      name: name.trim(),
      created_at: new Date().toISOString(),
      config: presetConfig,
    }
    customPresets.value = [...customPresets.value, preset]
    saveCustomPresetsToStorage()
    ElMessage.success(`预设"${name.trim()}"已保存（本地）`)
  }
}

// V3.2.4: 删除自定义预设 (Write-Through)
async function handleDeletePreset(presetId) {
  customPresets.value = customPresets.value.filter((p) => p.id !== presetId)
  saveCustomPresetsToStorage()
  ElMessage.success('预设已删除')

  // 异步同步到后端，不阻塞 UI
  presetsApi.deleteCustomPreset(presetId).catch(() => {})
}

// V3.2.4: 覆盖更新自定义预设（用当前配置覆写）
function handleOverwritePreset(presetId) {
  const idx = customPresets.value.findIndex((p) => p.id === presetId)
  if (idx === -1) return

  const updated = [...customPresets.value]
  updated[idx] = {
    ...updated[idx],
    config: {
      preprocessing: { ...taskConfig.value.preprocessing },
      transcription: { ...taskConfig.value.transcription },
      refinement: { ...taskConfig.value.refinement },
      compute: { ...taskConfig.value.compute },
    },
  }
  customPresets.value = updated
  saveCustomPresetsToStorage()
  ElMessage.success(`预设"${updated[idx].name}"已更新`)
}

// V3.2.4: localStorage 读写自定义预设
function saveCustomPresetsToStorage() {
  try {
    localStorage.setItem(CUSTOM_PRESETS_KEY, JSON.stringify(customPresets.value))
  } catch (error) {
    console.warn('保存自定义预设失败:', error)
  }
}

// V3.2.4: 初始化加载 (Read-Fallback: 优先后端，降级 localStorage)
async function loadCustomPresets() {
  try {
    const res = await presetsApi.getCustomPresets()
    if (res?.success && Array.isArray(res.presets)) {
      customPresets.value = res.presets
      saveCustomPresetsToStorage()
      return
    }
  } catch {
    // 后端不可用，降级读 localStorage
  }

  try {
    const raw = localStorage.getItem(CUSTOM_PRESETS_KEY)
    if (raw) {
      customPresets.value = JSON.parse(raw)
    }
  } catch (error) {
    console.warn('读取自定义预设失败:', error)
  }
}

loadCustomPresets()

onMounted(() => {
  loadTaskUiPreferences()
  if (!capabilities.canTranscribe || route.query.action === 'import') {
    showImportDialog.value = true
  }
})

function buildAdvancedConfig() {
  return {
    general: { ...advancedGeneral.value },
    preprocessing: { ...taskConfig.value.preprocessing },
    transcription: { ...taskConfig.value.transcription },
    refinement: {
      llm_provider: 'openai_compatible',
      ...taskConfig.value.refinement,
    },
    compute: { ...taskConfig.value.compute },
    preset_id: taskConfig.value.preset_id || 'balanced',
  }
}

function openAdvancedSettingsDialog() {
  advancedConfig.value = buildAdvancedConfig()
  showAdvancedSettings.value = true
}

// 导入成功后跳转编辑器
function handleImportSuccess(projectId) {
  showImportDialog.value = false
  router.push(`/editor/project/${projectId}`)
}

// 打开编辑器
async function openEditor(jobId) {
  await navigateToEditor(router, { jobId })
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
  clickTimer = setTimeout(async () => {
    clickTimer = null
    await openEditor(task.job_id)
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
      // 新语义：删除请求成功后立即逻辑删除，不等待后台物理清理
      taskStore.deleteTask(jobId)
      ElMessage.success(res?.message || '任务已删除')
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
    removed: '已删除',
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

async function handleSaveAdvancedSettings() {
  try {
    const rawOffset = advancedConfig.value.general.global_time_offset
    let normalizedOffset = Number(rawOffset)
    if (!Number.isFinite(normalizedOffset)) {
      normalizedOffset = 0
    }

    advancedGeneral.value = {
      ...advancedGeneral.value,
      ...advancedConfig.value.general,
      global_time_offset: normalizedOffset,
    }
    saveJsonStorage(ADVANCED_GENERAL_KEY, advancedGeneral.value)

    transcriptionPresetStore.applyTaskConfig({
      preset_id: advancedConfig.value.preset_id || taskConfig.value.preset_id,
      preprocessing: { ...advancedConfig.value.preprocessing },
      transcription: { ...advancedConfig.value.transcription },
      refinement: { ...advancedConfig.value.refinement },
      compute: { ...advancedConfig.value.compute },
    })

    ElMessage.success('高级设置已保存')
    showAdvancedSettings.value = false
  } catch (error) {
    console.error('保存高级设置失败:', error)
    ElMessage.error('保存高级设置失败: ' + (error.message || '未知错误'))
  }
}

function handleCancelAdvancedSettings() {
  showAdvancedSettings.value = false
}

function handleCloseAdvancedSettings() {
  showAdvancedSettings.value = false
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
      background: 'var(--af-bg-overlay)',
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

<style scoped>
.task-list-view {
  position: relative;
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
  background: var(--af-bg-primary);
}

.task-workspace {
  position: relative;
  display: flex;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.sidebar-edge-zone {
  position: absolute;
  top: 0;
  bottom: 0;
  left: 0;
  width: 18px;
  z-index: 120;
}

.sidebar-edge-btn {
  position: absolute;
  top: 50%;
  left: 4px;
  display: flex;
  justify-content: center;
  align-items: center;
  width: 22px;
  height: 48px;
  background: var(--af-bg-elevated);
  border: 1px solid var(--af-border-default);
  border-radius: var(--af-radius-md);
  color: var(--af-text-secondary);
  transform: translate(-120%, -50%);
  opacity: 0;
  cursor: pointer;
  transition: transform var(--af-transition-normal), opacity var(--af-transition-normal), color var(--af-transition-fast);
}

.sidebar-edge-btn.visible {
  transform: translate(0, -50%);
  opacity: 1;
}

.sidebar-edge-btn:hover {
  color: var(--af-accent-primary);
}

.task-sidebar {
  position: absolute;
  top: 16px;
  bottom: 16px;
  left: 0;
  width: 238px;
  background: var(--af-bg-secondary);
  border: 1px solid var(--af-border-default);
  border-radius: var(--af-radius-lg);
  box-shadow: var(--af-shadow-lg);
  transform: translateX(calc(-100% - 12px));
  opacity: 0;
  pointer-events: none;
  transition: transform var(--af-transition-normal), opacity var(--af-transition-normal);
  z-index: 130;
  backdrop-filter: blur(8px);
}

.task-sidebar.open {
  transform: translateX(10px);
  opacity: 1;
  pointer-events: auto;
}

.sidebar-collapse-btn {
  position: absolute;
  top: 50%;
  right: -14px;
  display: flex;
  justify-content: center;
  align-items: center;
  width: 24px;
  height: 48px;
  background: var(--af-bg-elevated);
  border: 1px solid var(--af-border-default);
  border-radius: var(--af-radius-md);
  color: var(--af-text-secondary);
  transform: translateY(-50%);
  cursor: pointer;
  transition: color var(--af-transition-fast);
}

.sidebar-collapse-btn:hover {
  color: var(--af-accent-primary);
}

.sidebar-inner {
  display: flex;
  flex-direction: column;
  height: 100%;
  gap: 12px;
  padding: 14px 12px 12px;
}

.sidebar-block {
  padding: 10px;
  background: var(--af-bg-tertiary);
  border: 1px solid var(--af-border-default);
  border-radius: var(--af-radius-md);
}

.sidebar-block--stats {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.block-title {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
  color: var(--af-text-normal);
  font-size: 12px;
  font-weight: 600;
}

.mode-switch {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
}

.mode-btn {
  padding: 6px 8px;
  background: var(--af-bg-elevated);
  border: 1px solid var(--af-border-default);
  border-radius: var(--af-radius-sm);
  color: var(--af-text-secondary);
  font-size: 12px;
  cursor: pointer;
  transition: all var(--af-transition-fast);
}

.mode-btn.active {
  border-color: var(--af-accent-primary);
  color: var(--af-accent-primary);
  box-shadow: var(--af-shadow-sm);
}

.mode-btn:hover {
  color: var(--af-text-primary);
}

.setting-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 0;
  padding: 7px 0;
  color: var(--af-text-secondary);
  font-size: 12px;
}

.setting-row + .setting-row {
  border-top: 1px dashed var(--af-border-muted);
}

.stats-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 0;
}

.stats-label {
  color: var(--af-text-muted);
  font-size: 12px;
}

.stats-value {
  color: var(--af-text-primary);
  font-size: 13px;
  font-weight: 600;
}

.sidebar-footer {
  margin-top: auto;
}

.more-settings-btn {
  width: 100%;
}

/* --- el-switch 样式定制 ---
 * 原因：统一任务侧栏中的开关尺寸和主题观感
 */
:deep(.setting-row .el-switch__core) {
  min-width: 34px;
  height: 18px;
}

/* --- el-message-box 样式定制 ---
 * 原因：复用任务页既有弹窗视觉体系，保持页面风格一致
 */
:deep(.el-message-box) {
  background: var(--af-bg-secondary);
  border: 1px solid var(--af-border-default);
}

:deep(.el-message-box .el-message-box__title) {
  color: var(--af-text-primary);
}

:deep(.el-message-box .el-message-box__content) {
  color: var(--af-text-secondary);
}

:deep(.el-message-box .el-message-box__btns .el-button) {
  background: transparent;
  border: 1px solid transparent;
  color: var(--af-text-secondary);
}

:deep(.el-message-box .el-message-box__btns .el-button:hover) {
  background: var(--af-bg-tertiary);
  border-color: var(--af-border-default);
  color: var(--af-text-primary);
}

:deep(.el-message-box .el-message-box__btns .el-button--primary) {
  background: transparent;
  border-color: transparent;
  color: var(--af-accent-primary);
}

:deep(.el-message-box .el-message-box__btns .el-button--primary:hover) {
  border-color: var(--af-accent-primary);
}

@media (max-width: 900px) {
  .task-sidebar {
    width: 220px;
  }

  .sidebar-edge-zone {
    width: 14px;
  }
}
</style>
