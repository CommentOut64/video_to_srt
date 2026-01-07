<template>
  <header class="editor-header">
    <!-- 左侧：返回 + 任务信息堆叠 -->
    <div class="header-left">
      <el-tooltip content="返回任务列表" placement="bottom" :show-after="500">
        <router-link to="/tasks" class="nav-back">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"/>
          </svg>
        </router-link>
      </el-tooltip>

      <div class="divider-vertical"></div>

      <div class="task-info-stack">
        <!-- 任务名称：双击可编辑 -->
        <div class="task-name-wrapper">
          <input
            v-if="isEditingTitle"
            ref="titleInputRef"
            v-model="editingTitleValue"
            class="task-name-input"
            @blur="finishEditTitle"
            @keydown.enter="finishEditTitle"
            @keydown.escape="cancelEditTitle"
          />
          <el-tooltip
            v-else
            :content="taskName + ' (双击重命名)'"
            placement="bottom"
            :show-after="500"
          >
            <h1
              class="task-name"
              @dblclick="startEditTitle"
            >{{ taskName }}</h1>
          </el-tooltip>
        </div>
        <div class="task-meta">
          <span class="status-dot" :class="statusClass"></span>
          <span class="meta-text">{{ metaText }}</span>
        </div>
      </div>
    </div>

    <!-- 中间：动态进度区 -->
    <div class="header-center">
      <!-- 场景1: 当前任务转录中或已暂停 -->
      <el-popover
        v-if="showCurrentTaskProgress"
        trigger="hover"
        :width="240"
        popper-class="control-popover-dark"
        :show-after="200"
        placement="bottom"
      >
        <template #reference>
          <div class="progress-capsule" :class="{ paused: isPaused }">
            <!-- Phase 5: 双流进度条 -->
            <div v-if="showDualStreamProgress" class="dual-progress-track">
              <div class="progress-layer fast" :style="{ width: dualStreamProgress.fastStream + '%' }"></div>
              <div class="progress-layer slow" :style="{ width: dualStreamProgress.slowStream + '%' }"></div>
            </div>
            <!-- 单流进度条 -->
            <div v-else class="progress-track">
              <div class="progress-fill" :style="{ width: currentTaskProgress + '%' }"></div>
            </div>
            <!-- 显示阶段标签和进度 -->
            <span class="capsule-text">
              <span
                class="phase-tag"
                :style="{
                  background: phaseStyle.bgColor,
                  color: phaseStyle.color
                }"
              >
                {{ phaseLabel }}
              </span>
              <!-- Phase 5: 双流进度显示 -->
              <span v-if="showDualStreamProgress" class="progress-percent dual">
                <span class="fast-label">S</span>{{ (dualStreamProgress.fastStream || 0).toFixed(1) }}%
                <span class="slow-label">W</span>{{ (dualStreamProgress.slowStream || 0).toFixed(1) }}%
              </span>
              <span v-else class="progress-percent">{{ formatProgress(currentTaskProgress) }}%</span>
            </span>
          </div>
        </template>

        <div class="hover-controls">
          <div class="label">当前任务控制</div>
          <div class="btn-group">
            <!-- 暂停/恢复按钮 -->
            <el-button v-if="!isPaused" circle size="small" @click="$emit('pause')">
              <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
                <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
              </svg>
            </el-button>
            <el-button v-else circle size="small" type="success" @click="$emit('resume')">
              <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
                <path d="M8 5v14l11-7z"/>
              </svg>
            </el-button>
            <!-- 取消按钮 -->
            <el-button circle size="small" type="danger" plain @click="$emit('cancel')">
              <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
                <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
              </svg>
            </el-button>
          </div>
        </div>
      </el-popover>

      <!-- 场景2: 当前任务完成，显示队列总进度 -->
      <div v-else class="queue-progress">
        <div class="progress-track">
          <div class="progress-fill complete" :style="{ width: queueProgressPercent + '%' }"></div>
        </div>
        <span class="progress-text">
          {{ queueCompleted }}/{{ queueTotal }} 任务完成
          <span v-if="queueProgressPercent === 100" class="complete-check">
            <svg viewBox="0 0 24 24" fill="currentColor" width="12" height="12">
              <path d="M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z"/>
            </svg>
          </span>
        </span>
      </div>
    </div>

    <!-- 右侧：操作按钮 -->
    <div class="header-right">
      <!-- 任务监控器 -->
      <el-popover
        trigger="click"
        :width="400"
        popper-class="task-monitor-popover"
        placement="bottom-end"
      >
        <template #reference>
          <div class="monitor-trigger">
            <el-tooltip content="任务监控" placement="bottom" :show-after="500">
              <button class="icon-btn">
                <svg viewBox="0 0 24 24" fill="currentColor">
                  <path d="M20 19V8H4v11h16m0-14a2 2 0 012 2v12a2 2 0 01-2 2H4a2 2 0 01-2-2V7a2 2 0 012-2h16M6 10h2v6H6v-6m4-1h2v7h-2V9m4 4h2v3h-2v-3z"/>
                </svg>
              </button>
            </el-tooltip>
            <span v-if="activeTasks > 0" class="badge">{{ activeTasks }}</span>
          </div>
        </template>
        <TaskMonitor :current-job-id="jobId" />
      </el-popover>

      <div class="divider-vertical"></div>

      <!-- 撤销/重做 -->
      <el-tooltip content="撤销 (Ctrl+Z)" placement="bottom">
        <button class="icon-btn" :class="{ disabled: !canUndo }" @click="$emit('undo')">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M12.5 8c-2.65 0-5.05.99-6.9 2.6L2 7v9h9l-3.62-3.62c1.39-1.16 3.16-1.88 5.12-1.88 3.54 0 6.55 2.31 7.6 5.5l2.37-.78C21.08 11.03 17.15 8 12.5 8z"/>
          </svg>
        </button>
      </el-tooltip>
      <el-tooltip content="重做 (Ctrl+Y)" placement="bottom">
        <button class="icon-btn" :class="{ disabled: !canRedo }" @click="$emit('redo')">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M18.4 10.6C16.55 8.99 14.15 8 11.5 8c-4.65 0-8.58 3.03-9.96 7.22L3.9 16c1.05-3.19 4.05-5.5 7.6-5.5 1.95 0 3.73.72 5.12 1.88L13 16h9V7l-3.6 3.6z"/>
          </svg>
        </button>
      </el-tooltip>

      <div class="divider-vertical"></div>

      <!-- 导出按钮 -->
      <el-dropdown trigger="click" @command="handleExport">
        <button class="export-btn">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>
          </svg>
          <span>导出</span>
          <svg class="arrow" viewBox="0 0 24 24" fill="currentColor">
            <path d="M7 10l5 5 5-5z"/>
          </svg>
        </button>
        <template #dropdown>
          <el-dropdown-menu>
            <el-dropdown-item command="srt">SRT 格式</el-dropdown-item>
            <el-dropdown-item command="ass">ASS 格式</el-dropdown-item>
            <el-dropdown-item command="vtt">WebVTT 格式</el-dropdown-item>
            <el-dropdown-item command="txt">纯文本</el-dropdown-item>
            <el-dropdown-item command="json">JSON 格式</el-dropdown-item>
          </el-dropdown-menu>
        </template>
      </el-dropdown>
    </div>
  </header>
</template>

<script setup>
/**
 * EditorHeader - 编辑器顶栏组件
 *
 * 职责：
 * - 导航控制（返回按钮）
 * - 任务信息展示（名称、状态）
 * - 动态进度显示（当前任务 / 队列总进度）
 * - 全局操作入口（任务监控、撤销/重做、导出）
 * - 双击重命名任务
 */
import { computed, ref, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import TaskMonitor from './TaskMonitor/index.vue'
import { PHASE_CONFIG, STATUS_CONFIG, formatProgress } from '@/constants/taskPhases'
import transcriptionApi from '@/services/api/transcriptionApi'
import { useUnifiedTaskStore } from '@/stores/unifiedTaskStore'
import { useProjectStore } from '@/stores/projectStore'

const taskStore = useUnifiedTaskStore()
const projectStore = useProjectStore()

const props = defineProps({
  jobId: { type: String, required: true },
  taskName: { type: String, default: '未命名项目' },
  currentTaskStatus: { type: String, default: 'idle' },      // 'processing', 'queued', 'paused', 'finished', etc.
  currentTaskPhase: { type: String, default: 'pending' },    // 任务阶段（transcribe, align, etc.）
  currentTaskProgress: { type: Number, default: 0 },         // 0-100
  queueCompleted: { type: Number, default: 0 },              // 已完成任务数
  queueTotal: { type: Number, default: 0 },                  // 总任务数
  canUndo: { type: Boolean, default: false },
  canRedo: { type: Boolean, default: false },
  activeTasks: { type: Number, default: 0 },                 // 正在进行的任务数
  lastSaved: { type: [Number, null], default: null },        // 上次保存时间戳
  // Phase 5: 双流进度
  dualStreamProgress: {
    type: Object,
    default: () => ({ fastStream: 0, slowStream: 0, totalChunks: 0 })
  }
})

const emit = defineEmits(['undo', 'redo', 'export', 'pause', 'resume', 'cancel', 'rename'])

// ========== 任务名称编辑 ==========
const isEditingTitle = ref(false)
const editingTitleValue = ref('')
const titleInputRef = ref(null)

// 开始编辑任务名称
function startEditTitle() {
  isEditingTitle.value = true
  editingTitleValue.value = props.taskName
  
  nextTick(() => {
    if (titleInputRef.value) {
      titleInputRef.value.focus()
      titleInputRef.value.select()
    }
  })
}

// 完成编辑任务名称
async function finishEditTitle() {
  if (!isEditingTitle.value) return
  
  const newTitle = editingTitleValue.value.trim()
  
  // 如果标题为空，提示并恢复原名称
  if (!newTitle) {
    ElMessage.warning('任务名称不能为空')
    isEditingTitle.value = false
    return
  }
  
  // 如果没有变化，直接关闭编辑
  if (newTitle === props.taskName) {
    isEditingTitle.value = false
    return
  }
  
  try {
    // 调用 API 重命名任务
    await transcriptionApi.renameJob(props.jobId, newTitle)
    
    // 更新 unifiedTaskStore
    taskStore.updateTask(props.jobId, { title: newTitle })
    
    // 更新 projectStore.meta.title
    projectStore.meta.title = newTitle
    
    // 通知父组件
    emit('rename', newTitle)
    
    ElMessage.success('重命名成功')
  } catch (error) {
    console.error('重命名任务失败:', error)
    ElMessage.error(`重命名失败: ${error.message || '未知错误'}`)
  } finally {
    isEditingTitle.value = false
  }
}

// 取消编辑
function cancelEditTitle() {
  isEditingTitle.value = false
  editingTitleValue.value = props.taskName
}

// 是否暂停状态（包含正在暂停和已暂停两种状态）
// V3.1.0: 'pausing' 状态表示正在等待当前原子操作完成，用户可以点击恢复取消暂停
const isPaused = computed(() => ['pausing', 'paused'].includes(props.currentTaskStatus))

// 是否显示当前任务进度（转录中、排队中、正在暂停、或已暂停）
// V3.1.0: 新增 'pausing' 状态，表示任务正在暂停中（等待当前原子操作完成）
const showCurrentTaskProgress = computed(() =>
  ['processing', 'queued', 'pausing', 'paused'].includes(props.currentTaskStatus)
)

// Phase 5: 是否显示双流进度（双模态对齐模式）
const showDualStreamProgress = computed(() =>
  showCurrentTaskProgress.value && props.dualStreamProgress.totalChunks > 0
)

// 队列进度百分比
const queueProgressPercent = computed(() =>
  props.queueTotal > 0 ? Math.round((props.queueCompleted / props.queueTotal) * 100) : 0
)

// 状态点样式
const statusClass = computed(() => {
  if (showCurrentTaskProgress.value) return 'processing'
  if (queueProgressPercent.value === 100 && props.queueTotal > 0) return 'complete'
  return 'idle'
})

// 元信息文字
const metaText = computed(() => {
  // V3.1.0: 区分"正在暂停"和"已暂停"状态
  if (props.currentTaskStatus === 'pausing') {
    return `正在暂停... ${props.currentTaskProgress}%`
  }
  if (props.currentTaskStatus === 'paused') {
    return `已暂停 ${props.currentTaskProgress}%`
  }
  if (showCurrentTaskProgress.value) {
    return `转录中 ${props.currentTaskProgress}%`
  }
  if (props.lastSaved) {
    const date = new Date(props.lastSaved)
    const time = date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
    return `自动保存于 ${time}`
  }
  return '准备就绪'
})

// 获取阶段样式（与TaskMonitor保持一致）
const phaseStyle = computed(() => {
  // 如果任务暂停，使用暂停状态样式
  if (isPaused.value) {
    return STATUS_CONFIG.paused || {
      bgColor: 'rgba(210, 153, 34, 0.15)',
      color: '#d29922',
      label: '已暂停'
    }
  }
  // 如果任务正在处理且有阶段信息，使用阶段样式
  if (props.currentTaskStatus === 'processing' && props.currentTaskPhase) {
    return PHASE_CONFIG[props.currentTaskPhase] || PHASE_CONFIG.pending
  }
  // 其他情况使用状态样式
  return STATUS_CONFIG[props.currentTaskStatus] || STATUS_CONFIG.created
})

// 阶段标签
const phaseLabel = computed(() => {
  // V3.1.0: 区分"正在暂停"和"已暂停"状态
  if (props.currentTaskStatus === 'pausing') {
    return '正在暂停...'
  }
  if (props.currentTaskStatus === 'paused') {
    return '已暂停'
  }
  // 如果任务正在处理且有阶段信息，显示阶段标签
  if (props.currentTaskStatus === 'processing' && props.currentTaskPhase) {
    return PHASE_CONFIG[props.currentTaskPhase]?.label || '处理中'
  }
  // 其他情况显示状态标签
  return STATUS_CONFIG[props.currentTaskStatus]?.label || props.currentTaskStatus
})

// 处理导出
function handleExport(format) {
  // 向父组件发送 export 事件
  // 由父组件处理实际导出逻辑
  const event = new CustomEvent('header-export', { detail: format })
  window.dispatchEvent(event)
}
</script>

<style lang="scss" scoped>
$header-h: 56px;

.editor-header {
  height: $header-h;
  background: var(--bg-primary);
  border-bottom: 1px solid var(--border-default);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 16px;
  position: relative;
  flex-shrink: 0;
}

// 左侧堆叠布局
.header-left {
  display: flex;
  align-items: center;
  gap: 12px;

  .nav-back {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    transition: all 0.2s;

    svg {
      width: 20px;
      height: 20px;
    }

    &:hover {
      background: var(--bg-tertiary);
      color: var(--text-primary);
    }
  }

  .task-info-stack {
    display: flex;
    flex-direction: column;
    gap: 2px;

    .task-name-wrapper {
      display: flex;
      align-items: center;
      max-width: 300px;
    }

    .task-name {
      margin: 0;
      font-size: 14px;
      font-weight: 500;
      color: var(--text-primary);
      max-width: 300px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      cursor: pointer;
      padding: 2px 4px;
      border-radius: var(--radius-sm);
      transition: background 0.2s;

      &:hover {
        background: var(--bg-tertiary);
      }
    }

    .task-name-input {
      font-size: 14px;
      font-weight: 500;
      color: var(--text-primary);
      background: var(--bg-tertiary);
      border: 1px solid var(--primary);
      border-radius: var(--radius-sm);
      padding: 2px 6px;
      width: 200px;
      max-width: 300px;
      outline: none;

      &:focus {
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.3);
      }
    }

    .task-meta {
      display: flex;
      align-items: center;
      gap: 5px;

      .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;

        &.processing {
          background: var(--primary);
          animation: pulse 1.5s infinite;
        }
        &.complete { background: var(--success); }
        &.idle { background: var(--text-muted); }
      }

      .meta-text {
        font-size: 11px;
        color: var(--text-muted);
      }
    }
  }
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

// 中间绝对定位居中
.header-center {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
}

// 当前任务进度胶囊
.progress-capsule {
  background: var(--bg-elevated);
  border-radius: 12px;
  padding: 4px 14px;
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  transition: all 0.2s;

  &:hover {
    background: var(--bg-tertiary);
    transform: scale(1.02);
  }

  &.paused {
    opacity: 0.85;
  }

  .progress-track {
    width: 120px;
    height: 4px;
    background: var(--border-muted);
    border-radius: 2px;
    overflow: hidden;

    .progress-fill {
      height: 100%;
      background: var(--primary);
      transition: width 0.3s ease;
    }
  }

  // Phase 5: 双流进度条
  .dual-progress-track {
    width: 120px;
    height: 8px;
    background: var(--border-muted);
    border-radius: 4px;
    overflow: hidden;
    position: relative;

    .progress-layer {
      position: absolute;
      left: 0;
      height: 4px;
      transition: width 0.3s ease;

      &.fast {
        top: 0;
        background: #58a6ff;  // SenseVoice - 蓝色
      }

      &.slow {
        bottom: 0;
        background: #3fb950;  // Whisper - 绿色
      }
    }
  }

  .capsule-text {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    white-space: nowrap;

    .phase-tag {
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 600;
      white-space: nowrap;
    }

    .progress-percent {
      color: var(--text-secondary);
      font-family: var(--font-mono);

      // Phase 5: 双流进度文字
      &.dual {
        display: flex;
        align-items: center;
        gap: 4px;
        font-size: 11px;

        .fast-label {
          color: #58a6ff;
          font-weight: 600;
          margin-right: 1px;
        }

        .slow-label {
          color: #3fb950;
          font-weight: 600;
          margin-left: 6px;
          margin-right: 1px;
        }
      }
    }
  }
}

// 队列总进度
.queue-progress {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 16px;
  background: var(--bg-elevated);
  border-radius: 10px;

  .progress-track {
    width: 150px;
    height: 4px;
    background: var(--border-muted);
    border-radius: 2px;
    overflow: hidden;

    .progress-fill {
      height: 100%;
      background: var(--primary);
      transition: width 0.5s ease;

      &.complete {
        background: var(--success);
      }
    }
  }

  .progress-text {
    font-size: 12px;
    color: var(--text-secondary);
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: 6px;

    .complete-check {
      color: var(--success);
      display: flex;
      align-items: center;
    }
  }
}

// 悬浮控制面板
.hover-controls {
  .label {
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 10px;
  }

  .btn-group {
    display: flex;
    gap: 8px;
    justify-content: center;
  }
}

// 右侧按钮组
.header-right {
  display: flex;
  align-items: center;
  gap: 8px;

  .monitor-trigger {
    position: relative;
    display: flex;
    align-items: center;

    .badge {
      position: absolute;
      top: 0;
      right: 0;
      background: var(--primary);
      color: #fff;
      font-size: 10px;
      padding: 0 4px;
      min-width: 16px;
      height: 16px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 600;
      transform: translate(25%, -25%);
    }
  }

  .icon-btn {
    width: 36px;
    height: 36px;
    padding: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    border: none;
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.2s;

    svg {
      width: 18px;
      height: 18px;
    }

    &:hover:not(.disabled) {
      color: var(--text-primary);
      background: var(--bg-tertiary);
    }

    &.disabled {
      opacity: 0.4;
      cursor: not-allowed;
    }
  }

  .export-btn {
    height: 32px;
    padding: 0 14px;
    display: flex;
    align-items: center;
    gap: 6px;
    background: var(--primary);
    color: white;
    border: none;
    border-radius: var(--radius-md);
    font-size: 13px;
    cursor: pointer;
    transition: background 0.2s;

    svg {
      width: 16px;
      height: 16px;
    }

    .arrow {
      width: 14px;
      height: 14px;
      margin-left: -2px;
    }

    &:hover {
      background: var(--primary-hover);
    }
  }
}

.divider-vertical {
  width: 1px;
  height: 20px;
  background: var(--border-muted);
  margin: 0 4px;
}
</style>

<style lang="scss">
// 全局样式：Popover 深色主题
.control-popover-dark {
  background: var(--bg-elevated) !important;
  border-color: var(--border-default) !important;

  .el-popper__arrow::before {
    background: var(--bg-elevated) !important;
    border-color: var(--border-default) !important;
  }
}

.task-monitor-popover {
  background: var(--bg-secondary) !important;
  border-color: var(--border-default) !important;
  padding: 0 !important;

  .el-popper__arrow::before {
    background: var(--bg-secondary) !important;
    border-color: var(--border-default) !important;
  }
}

// 导出下拉菜单样式 - 与任务监控配色一致
.el-dropdown__popper.el-popper {
  background: var(--bg-secondary) !important;
  border: 1px solid var(--border-default) !important;
  border-radius: 6px !important;
  box-shadow: var(--shadow-lg) !important;
  padding: 4px 0 !important;

  .el-popper__arrow::before {
    background: var(--bg-secondary) !important;
    border-color: var(--border-default) !important;
  }

  .el-dropdown-menu {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;

    .el-dropdown-menu__item {
      padding: 8px 16px !important;
      font-size: 13px !important;
      color: var(--text-primary) !important;
      background: transparent !important;
      transition: all 0.2s !important;

      &:hover {
        background: rgba(255, 255, 255, 0.03) !important;
        color: var(--text-primary) !important;
      }

      &:focus {
        background: rgba(255, 255, 255, 0.03) !important;
        color: var(--text-primary) !important;
      }
    }
  }
}
</style>

