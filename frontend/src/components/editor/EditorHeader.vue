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
          <span v-if="lastSavedText" class="save-text">{{ lastSavedText }}</span>
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
            <!-- 阶段标签 -->
            <!-- <span
              class="phase-tag"
              :style="{
                background: phaseStyle.bgColor,
                color: phaseStyle.color
              }"
            >
              {{ phaseLabel }}
            </span> -->

            <!-- V3.2.0: 双锚点进度条 -->
            <DualAnchorProgress
              :fast-progress="mappedProgress.fast"
              :slow-progress="mappedProgress.slow"
              :status="progressStatus"
              size="lg"
            />

            <!-- V3.2.0+dev.20260122.02: 显示总进度百分比 -->
            <span class="progress-percent">
              {{ mappedProgress.slow.toFixed(1) }}%
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
        <!-- V3.2.0: 简单进度条 -->
        <SimpleProgress
          :progress="queueProgressPercent"
          :status="queueProgressPercent === 100 ? 'complete' : 'normal'"
          size="md"
        />
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
import DualAnchorProgress from '@/components/common/DualAnchorProgress.vue'
import SimpleProgress from '@/components/common/SimpleProgress.vue'
import { PHASE_CONFIG, STATUS_CONFIG, formatProgress } from '@/constants/taskPhases'
import { PROGRESS_ALLOCATION } from '@/components/common/progress/constants'
import transcriptionApi from '@/services/api/transcriptionApi'
import { useUnifiedTaskStore } from '@/stores/unifiedTaskStore'
import { useProjectStore } from '@/stores/projectStore'
import { useProgressStore } from '@/stores/progressStore'

const taskStore = useUnifiedTaskStore()
const projectStore = useProjectStore()
const progressStore = useProgressStore()

const props = defineProps({
  jobId: { type: String, required: true },
  taskName: { type: String, default: '未命名项目' },
  currentTaskStatus: { type: String, default: 'idle' },      // 'processing', 'queued', 'paused', 'finished', etc.
  currentTaskPhase: { type: String, default: 'pending' },    // 任务阶段（transcribe, align, etc.）
  currentTaskProgress: { type: Number, default: 0 },         // 0-100
  progressDetail: { type: Object, default: null },           // SSE detail 细节
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

// V3.2.0+dev.20260122.02: 从 progressStore 获取当前任务的进度状态
const jobProgress = computed(() => progressStore.getJobProgress(props.jobId).value)

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
const isPaused = computed(() => ['pausing', 'paused'].includes(jobProgress.value.status))

// V3.2.0+dev.20260122.02: 始终显示当前任务进度条，不再切换到队列进度条
const showCurrentTaskProgress = computed(() => {
  return ['processing', 'queued', 'pausing', 'paused', 'finished'].includes(jobProgress.value.status)
})

function clampPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 0
  return Math.max(0, Math.min(100, Number(value)))
}

// V3.2.0+dev.20260122.02: 以下函数和常量保留供未来双流进度功能使用
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function mapToRange(value, start, end) {
  const normalized = clampPercent(value) / 100
  return start + (end - start) * normalized
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const PREPROCESS_PHASES = new Set([
  'extract',
  'vad',
  'spectral',
  'spectrum_analysis',
  'separation',
  'bgm_detect',
  'demucs'
])

// eslint-disable-next-line @typescript-eslint/no-unused-vars, no-unused-vars
const TRANSCRIPTION_PHASES = new Set(['sensevoice', 'whisper'])

// V3.2.0+dev.20260125.04: 修复双流模式和智能复核模式进度映射逻辑
// 关键理解：
// - 双流模式：fast 和 slow 处理同一批 chunk，都映射到 20%-75%
// - 智能复核模式：快流 70%（20%-58.5%），慢流 30%（58.5%-75%）
// - 转录完成后，后端总进度从 75% 跳到 95%（跳过精修），再到 100%（导出完成）
// - 对齐阶段废除，不计入进度
const mappedProgress = computed(() => {
  // 任务完成：强制 100%
  if (jobProgress.value.status === 'finished') {
    return { fast: 100, slow: 100 }
  }

  // 从 SSE 获取详细进度（后端推送的 detail 字段）
  const detail = jobProgress.value.detail || {}
  const mode = jobProgress.value.mode || 'dual_stream'
  const preprocess = clampPercent(detail.preprocess || 0)
  const fast = clampPercent(detail.fast || 0)
  const slow = clampPercent(detail.slow || 0)

  // 阶段分配常量（与设计文档保持一致）
  const PREPROCESS_RANGE = 20    // 前处理 0-20%
  const TRANSCRIPTION_RANGE = 55 // 转录占 55%（20%-75%）

  // 1. 前处理阶段（0-20%）
  const preprocessProgress = (preprocess / 100) * PREPROCESS_RANGE

  // V3.2.0+dev.20260125.04: 智能复核模式特殊处理
  if (mode === 'whisper_patch') {
    // 智能复核模式：快流 70%，慢流 30%
    const FAST_RANGE = TRANSCRIPTION_RANGE * 0.7   // 38.5%
    const SLOW_RANGE = TRANSCRIPTION_RANGE * 0.3   // 16.5%
    const FAST_END = PREPROCESS_RANGE + FAST_RANGE // 58.5%

    // 快流映射到 20%-58.5%
    const fastProgress = (fast / 100) * FAST_RANGE
    const mappedFast = Math.min(FAST_END, preprocessProgress + fastProgress)

    // 慢流映射到 58.5%-75%（在快流完成后才开始）
    // 注意：慢流进度需要叠加在快流结束点之上
    const slowProgress = (slow / 100) * SLOW_RANGE
    // 快流完成后，慢流从 58.5% 开始
    const mappedSlow = fast >= 100
      ? Math.min(75, FAST_END + slowProgress)
      : mappedFast  // 快流未完成时，慢流跟随快流

    return {
      fast: mappedFast,
      slow: mappedSlow
    }
  }

  // 2. 双流模式和极速模式：快慢流都映射到 20%-75%
  // 快流：用于前端追赶动画
  const fastTranscriptionProgress = (fast / 100) * TRANSCRIPTION_RANGE
  // 慢流：代表真实转录进度
  const slowTranscriptionProgress = (slow / 100) * TRANSCRIPTION_RANGE

  // 快流显示进度 = 前处理 + 快流转录（最大到 75%）
  const mappedFast = Math.min(75, preprocessProgress + fastTranscriptionProgress)

  // 慢流显示进度 = 前处理 + 慢流转录（最大到 75%）
  const mappedSlow = Math.min(75, preprocessProgress + slowTranscriptionProgress)

  return {
    fast: mappedFast,
    slow: mappedSlow
  }
})

// V3.2.0+dev.20260122.02: 进度条状态映射，从 progressStore 获取
const progressStatus = computed(() => {
  if (jobProgress.value.status === 'failed') return 'error'
  if (['pausing', 'paused'].includes(jobProgress.value.status)) return 'paused'
  if (jobProgress.value.status === 'finished') return 'completed'
  return 'running'
})

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

// V3.2.0+dev.20260122.02: 元信息文字，从 progressStore 获取状态
const metaText = computed(() => {
  const status = jobProgress.value.status
  if (status === 'pausing') return '正在暂停...'
  if (status === 'paused') return '已暂停'
  if (status === 'queued') return '排队中'
  if (status === 'processing') return '转录中'
  if (status === 'canceling') return '正在取消...'
  if (status === 'force_canceled') return '已强制取消'
  if (status === 'canceled') return '已取消'
  if (status === 'failed') return '任务失败'
  if (status === 'finished') return '已完成'
  return '准备就绪'
})

// 保存提示在每次自动保存后刷新
const lastSavedText = computed(() => {
  if (!props.lastSaved) return ''
  const date = new Date(props.lastSaved)
  const time = date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  return `最后保存 ${time}`
})

// V3.2.0+dev.20260122.02: 获取阶段样式，从 progressStore 获取状态
const phaseStyle = computed(() => {
  const status = jobProgress.value.status
  const phase = jobProgress.value.phase

  // 如果任务暂停，使用暂停状态样式
  if (isPaused.value) {
    return STATUS_CONFIG.paused || {
      bgColor: 'rgba(210, 153, 34, 0.15)',
      color: '#d29922',
      label: '已暂停'
    }
  }
  // 如果任务正在处理且有阶段信息，使用阶段样式
  if (status === 'processing' && phase) {
    return PHASE_CONFIG[phase] || PHASE_CONFIG.pending
  }
  // 其他情况使用状态样式
  return STATUS_CONFIG[status] || STATUS_CONFIG.created
})

// V3.2.0+dev.20260122.02: 阶段标签，从 progressStore 获取状态
const phaseLabel = computed(() => {
  const status = jobProgress.value.status
  const phase = jobProgress.value.phase

  // V3.1.0: 区分"正在暂停"和"已暂停"状态
  if (status === 'pausing') {
    return '正在暂停...'
  }
  if (status === 'paused') {
    return '已暂停'
  }
  // 如果任务正在处理且有阶段信息，显示阶段标签
  if (status === 'processing' && phase) {
    return PHASE_CONFIG[phase]?.label || '处理中'
  }
  // 其他情况显示状态标签
  return STATUS_CONFIG[status]?.label || status
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
      gap: 6px;
      flex-wrap: wrap;

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

      .save-text {
        font-size: 11px;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 4px;

        &::before {
          content: '-';
          color: var(--text-muted);
        }
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
  overflow: visible;
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
    box-shadow: 0 0 0 1px rgba(88, 166, 255, 0.3);
  }

  &.paused {
    opacity: 0.85;
  }

  .phase-tag {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    white-space: nowrap;
  }

  .progress-percent {
    color: var(--text-secondary);
    font-family: var(--font-mono);
    font-size: 11px;
    white-space: nowrap;

    &.dual {
      display: flex;
      align-items: center;
      gap: 4px;

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

// 队列总进度
.queue-progress {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 16px;
  background: var(--bg-elevated);
  border-radius: 16px;

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

