<template>
  <div
    class="task-card"
    :class="[
      `variant-${variant}`,
      {
        'is-draggable': draggable,
        'is-clickable': true,  // 所有状态的卡片都可点击
        'is-current': task.job_id === currentJobId  // 当前正在编辑器打开的任务
      }
    ]"
    @click="handleCardClick"
  >
    <!-- 拖动手柄 -->
    <el-tooltip content="拖动排序" placement="right" :show-after="500">
      <div v-if="draggable" class="drag-handle" @click.stop>
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M11 18c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2zm-2-8c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0-6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm6 4c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/>
        </svg>
      </div>
    </el-tooltip>

    <!-- 任务信息 -->
    <div class="task-info">
      <div class="task-header">
        <el-tooltip :content="task.title || task.filename" placement="top" :show-after="500">
          <span class="task-name">
            {{ task.title || task.filename }}
          </span>
        </el-tooltip>
        <span
          class="task-phase"
          :style="{
            background: getPhaseStyle(task).bgColor,
            color: getPhaseStyle(task).color
          }"
        >
          {{ getPhaseLabel(task) }}
        </span>
      </div>

      <!-- 进度条 -->
      <div v-if="showProgress" class="task-progress">
        <div class="progress-bar">
          <div
            class="progress-fill"
            :style="{ width: task.progress + '%' }"
          ></div>
        </div>
        <span class="progress-text">{{ formatProgress(task.progress) }}%</span>
      </div>

      <!-- 错误信息 -->
      <div v-if="task.status === 'failed'" class="task-error">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
        </svg>
        <span>{{ task.message || '转录失败' }}</span>
      </div>

      <!-- 完成时间 -->
      <div v-if="task.status === 'finished'" class="task-meta">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z"/>
        </svg>
        <span>{{ formatTime(task.completed_at || task.updatedAt) }}</span>
      </div>
    </div>

    <!-- 操作按钮 -->
    <div class="task-actions" @click.stop>
      <el-tooltip content="暂停" placement="left" :show-after="500">
        <button
          v-if="task.status === 'processing'"
          class="action-btn"
          @click="pauseTask"
        >
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
          </svg>
        </button>
      </el-tooltip>

      <el-tooltip content="恢复" placement="left" :show-after="500">
        <button
          v-if="task.status === 'paused'"
          class="action-btn action-btn--success"
          @click="resumeTask"
        >
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M8 5v14l11-7z"/>
          </svg>
        </button>
      </el-tooltip>

      <el-tooltip content="取消" placement="left" :show-after="500">
        <button
          v-if="['processing', 'queued', 'paused'].includes(task.status) && task.status !== 'canceling'"
          class="action-btn action-btn--danger"
          @click="cancelTask"
        >
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
          </svg>
        </button>
      </el-tooltip>

      <!-- V3.1.0: 正在取消状态显示加载动画 -->
      <el-tooltip content="正在取消..." placement="left" :show-after="500">
        <button
          v-if="task.status === 'canceling'"
          class="action-btn action-btn--danger"
          disabled
        >
          <svg viewBox="0 0 24 24" fill="currentColor" class="spin">
            <path d="M12 4V2A10 10 0 0 0 2 12h2a8 8 0 0 1 8-8z"/>
          </svg>
        </button>
      </el-tooltip>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { transcriptionApi } from '@/services/api'
import { PHASE_CONFIG, STATUS_CONFIG, formatProgress } from '@/constants/taskPhases'

const props = defineProps({
  task: { type: Object, required: true },
  variant: { type: String, default: 'default' },
  draggable: { type: Boolean, default: false },
  currentJobId: { type: String, default: '' }  // 当前正在编辑器打开的任务ID
})

const router = useRouter()

// V3.1.0: 防抖控制 - 防止频繁切换暂停/恢复
const lastOperationTime = ref(0)
const DEBOUNCE_DELAY = 2000 // 2秒防抖

const showProgress = computed(() =>
  ['processing', 'queued', 'paused', 'canceling'].includes(props.task.status)
)

// 处理卡片点击事件
function handleCardClick() {
  // 所有状态的任务都可以点击跳转到编辑器
  openEditor()
}

// 获取阶段样式
function getPhaseStyle(task) {
  if (task.status === 'failed') return STATUS_CONFIG.failed
  if (task.status === 'processing' && task.phase) {
    return PHASE_CONFIG[task.phase] || PHASE_CONFIG.pending
  }
  return STATUS_CONFIG[task.status] || STATUS_CONFIG.created
}

// 获取阶段标签
function getPhaseLabel(task) {
  if (task.status === 'failed') return '失败'
  if (task.status === 'processing' && task.phase) {
    return PHASE_CONFIG[task.phase]?.label || '处理中'
  }
  return STATUS_CONFIG[task.status]?.label || task.status
}

// 暂停任务
async function pauseTask() {
  // V3.1.0: 防抖检查
  const now = Date.now()
  if (now - lastOperationTime.value < DEBOUNCE_DELAY) {
    ElMessage.warning('操作过于频繁，请稍后再试')
    return
  }

  lastOperationTime.value = now
  await transcriptionApi.pauseJob(props.task.job_id)
}

// 恢复任务
async function resumeTask() {
  // V3.1.0: 防抖检查
  const now = Date.now()
  if (now - lastOperationTime.value < DEBOUNCE_DELAY) {
    ElMessage.warning('操作过于频繁，请稍后再试')
    return
  }

  lastOperationTime.value = now
  await transcriptionApi.resumeJob(props.task.job_id)
}

// 取消任务
async function cancelTask() {
  if (!confirm('确定要取消这个任务吗?')) return
  await transcriptionApi.cancelJob(props.task.job_id, false)
}

// 打开编辑器
function openEditor() {
  router.push(`/editor/${props.task.job_id}`)
}

// 格式化时间
function formatTime(timestamp) {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  const now = new Date()
  const diff = now - date

  if (diff < 60000) return '刚刚'
  if (diff < 3600000) return `${Math.floor(diff / 60000)} 分钟前`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)} 小时前`
  return date.toLocaleDateString('zh-CN')
}
</script>

<style lang="scss" scoped>
.task-card {
  display: flex;
  gap: 12px;
  padding: 12px;
  background: rgba(255, 255, 255, 0.03);  // 更浅的灰色，能与背景区分
  border: 1px solid transparent;
  border-radius: 6px;
  margin-bottom: 8px;
  transition: border-color 0.2s;
  user-select: none;  // 禁止文本选择

  // hover 时只改变边框颜色
  &:hover {
    border-color: #028AC5;
  }

  // 正在运行的任务使用浅绿色边框
  &.variant-processing {
    border-color: rgba(63, 185, 80, 0.5);
  }

  &.is-draggable {
    cursor: move;
  }

  // 可点击的卡片（已完成任务）
  &.is-clickable {
    cursor: pointer;

    .task-info {
      cursor: pointer;
    }
  }

  // 当前正在编辑器打开的任务 - 呼吸灯效果
  &.is-current {
    animation: breathing-border 3s ease-in-out infinite;
  }
}

// 呼吸灯动画 - 边框颜色从透明到 #028AC5 再到透明
@keyframes breathing-border {
  0%, 100% {
    border-color: transparent;
  }
  50% {
    border-color: #028AC5;
    box-shadow: 0 0 8px rgba(2, 138, 197, 0.4);
  }
}

.drag-handle {
  width: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  cursor: grab;
  opacity: 0.5;
  user-select: none;  // 禁止文本选择

  &:hover {
    opacity: 1;
  }

  &:active {
    cursor: grabbing;
  }

  svg {
    width: 16px;
    height: 16px;
    pointer-events: none;  // 防止 SVG 阻止拖拽
  }
}

.task-info {
  flex: 1;
  min-width: 0;
}

.task-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
  gap: 8px;
}

.task-name {
  flex: 1;
  font-size: 13px;
  font-weight: 500;
  color: var(--text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.task-phase {
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 600;
  white-space: nowrap;
}

.task-progress {
  display: flex;
  align-items: center;
  gap: 8px;
}

.progress-bar {
  flex: 1;
  height: 4px;
  background: var(--border-muted);
  border-radius: 2px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--primary);
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--text-muted);
  min-width: 35px;
  text-align: right;
}

.task-error {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--danger);

  svg {
    width: 14px;
    height: 14px;
  }
}

.task-meta {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--text-muted);

  svg {
    width: 14px;
    height: 14px;
    color: var(--success);
  }
}

.task-actions {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.action-btn {
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  border: none;
  border-radius: 4px;
  color: var(--text-muted);
  cursor: pointer;
  transition: all 0.2s;

  svg {
    width: 16px;
    height: 16px;
  }

  &:hover {
    background: var(--bg-tertiary);
    color: var(--text-secondary);
  }

  &--primary {
    color: var(--primary);
    &:hover {
      background: rgba(88, 166, 255, 0.15);
    }
  }

  &--success {
    color: var(--success);
    &:hover {
      background: rgba(63, 185, 80, 0.15);
    }
  }

  &--danger:hover {
    background: rgba(248, 81, 73, 0.15);
    color: var(--danger);
  }

  // V3.1.0: 禁用状态
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
}

// V3.1.0: 旋转动画
.spin {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

// 拖动占位符样式
.task-ghost {
  opacity: 0.5;
  background: var(--bg-tertiary);
  border: 2px dashed var(--border-default);
}
</style>
