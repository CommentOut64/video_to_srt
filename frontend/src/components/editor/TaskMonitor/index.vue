<template>
  <div class="task-monitor">
    <!-- 固定区：Processing 任务 -->
    <div class="monitor-header">
      <TransitionGroup name="list">
        <TaskCard
          v-if="processingTask"
          :key="processingTask.job_id"
          :task="processingTask"
          :currentJobId="currentJobId"
          variant="processing"
        />
      </TransitionGroup>

      <!-- 无任务时的空状态 -->
      <div v-if="!processingTask && !hasAnyTasks" class="empty-state">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path
            d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"
          />
        </svg>
        <p>暂无后台任务</p>
      </div>
    </div>

    <!-- 滚动区：任务分组 -->
    <div class="monitor-body" ref="scrollContainer">
      <!-- SSE 断开警告 - 已禁用，避免误报干扰用户 -->
      <!-- <div v-if="!sseConnected" class="sse-alert">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z" />
        </svg>
        <span>实时连接已断开，正在尝试重连...</span>
      </div> -->

      <!-- 失败任务 -->
      <TaskGroup
        v-if="failedTasks.length > 0"
        title="失败"
        :count="failedTasks.length"
        variant="danger"
        :defaultCollapsed="true"
      >
        <TransitionGroup name="list">
          <TaskCard
            v-for="task in failedTasks"
            :key="task.job_id"
            :task="task"
            :currentJobId="currentJobId"
          />
        </TransitionGroup>
      </TaskGroup>

      <!-- 排队中（可拖动） -->
      <TaskGroup
        v-if="queuedTasks.length > 0"
        title="排队中"
        :count="queuedTasks.length"
        variant="primary"
        :defaultCollapsed="false"
      >
        <Draggable
          v-model="queuedTasksLocal"
          item-key="job_id"
          :animation="200"
          ghost-class="task-ghost"
          handle=".drag-handle"
          :scroll-sensitivity="80"
          :scroll-speed="20"
          @end="handleDragEnd"
        >
          <template #item="{ element }">
            <TaskCard :task="element" :draggable="true" :currentJobId="currentJobId" />
          </template>
        </Draggable>
      </TaskGroup>

      <!-- 已暂停 -->
      <TaskGroup
        v-if="pausedTasks.length > 0"
        title="已暂停"
        :count="pausedTasks.length"
        variant="warning"
        :defaultCollapsed="true"
      >
        <TransitionGroup name="list">
          <TaskCard
            v-for="task in pausedTasks"
            :key="task.job_id"
            :task="task"
            :currentJobId="currentJobId"
          />
        </TransitionGroup>
      </TaskGroup>

      <!-- 最近完成 -->
      <TaskGroup
        v-if="recentFinishedTasks.length > 0"
        title="最近完成"
        :count="recentFinishedTasks.length"
        variant="success"
        :defaultCollapsed="true"
      >
        <TransitionGroup name="list">
          <TaskCard
            v-for="task in recentFinishedTasks"
            :key="task.job_id"
            :task="task"
            :currentJobId="currentJobId"
            :class="{ 'newly-finished': task.isNewlyFinished }"
          />
        </TransitionGroup>

        <div class="view-more" @click="openHistoryPage">查看全部历史记录 ></div>
      </TaskGroup>
    </div>

    <!-- 底部渐变遮罩 -->
    <div class="gradient-mask"></div>
  </div>
</template>

<script setup>
/**
 * TaskMonitor - 任务监控器组件（重构版）
 *
 * 基于单任务运行机制的任务监控系统
 * 功能：
 * - Processing 区域（固定顶部，最多1个任务）
 * - 任务分组（失败/排队/暂停/完成）
 * - 拖动排序（仅队列任务）
 * - SSE 心跳检测
 */
import { computed, ref, onMounted, onUnmounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useTaskRuntimeStore } from '@/stores/taskRuntimeStore'
import Draggable from 'vuedraggable'
import TaskGroup from './TaskGroup.vue'
import TaskCard from './TaskCard.vue'

const props = defineProps({
  currentJobId: { type: String, default: '' },
})

const router = useRouter()
const taskStore = useTaskRuntimeStore()

// 从 store 获取数据
const processingTask = computed(() => taskStore.processingTask)
const queuedTasks = computed(() => taskStore.queuedTasks)
const failedTasks = computed(() => taskStore.failedTasks)
const pausedTasks = computed(() => taskStore.pausedTasks)
const recentFinishedTasks = computed(() => taskStore.recentFinishedTasks)
const sseConnected = computed(() => taskStore.sseConnected)

const hasAnyTasks = computed(() => taskStore.tasks.length > 0)

// 本地可拖动任务列表
const queuedTasksLocal = ref([])

// 监听 store 的 queuedTasks 变化，更新本地列表
watch(
  () => queuedTasks.value,
  (newTasks) => {
    queuedTasksLocal.value = [...newTasks]
  },
  { immediate: true, deep: true }
)

// 拖动结束处理
async function handleDragEnd(event) {
  if (event.oldIndex === event.newIndex) return

  const newOrder = queuedTasksLocal.value.map((t) => t.job_id)
  try {
    await taskStore.reorderQueue(newOrder)
  } catch (error) {
    console.error('[TaskMonitor] 队列重排失败:', error)
    // 恢复原顺序
    queuedTasksLocal.value = [...queuedTasks.value]
  }
}

// 打开历史记录页面
function openHistoryPage() {
  // TODO: 实现历史记录页面
  console.log('[TaskMonitor] 打开历史记录页面')
}

// SSE 心跳检测
let heartbeatTimer = null
onMounted(() => {
  heartbeatTimer = setInterval(() => {
    taskStore.checkSSEConnection()
  }, 5000)
})

onUnmounted(() => {
  if (heartbeatTimer) {
    clearInterval(heartbeatTimer)
  }
})
</script>

<style scoped>
.task-monitor {
  position: relative;
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--af-bg-secondary);
  user-select: none;
  cursor: default;
}

.monitor-header {
  z-index: 10;
  padding: 5px;
  background: var(--af-bg-secondary);
  flex-shrink: 0;
  max-height: 300px;
  overflow-y: auto;
}

/* 当有 processing 任务时，添加底部边距 */
.monitor-header:has(.task-card) {
  padding: 12px;
  padding-bottom: 8px;
}

/* 空状态时的 padding */
.monitor-header:has(.empty-state) {
  padding: 12px;
}

.monitor-body {
  flex: 0 1 auto;
  max-height: 500px;
  overflow-y: auto;
  padding: 12px;
  padding-top: 4px;
  padding-bottom: 24px;
  scrollbar-gutter: stable;
}

.monitor-body::-webkit-scrollbar {
  width: 8px;
}

.monitor-body::-webkit-scrollbar-thumb {
  background: var(--af-border-muted);
  border-radius: 4px;
}

.gradient-mask {
  position: absolute;
  right: 0;
  bottom: 0;
  left: 0;
  height: 24px;
  background: linear-gradient(to top, var(--af-bg-secondary), transparent);
  transition: opacity 0.3s;
  pointer-events: none;
  opacity: 0;
}

/* 当 body 有滚动时显示渐变遮罩 */
.monitor-body[data-has-scroll='true'] + .gradient-mask,
.task-monitor:has(.monitor-body::-webkit-scrollbar-thumb) .gradient-mask {
  opacity: 1;
}

.sse-alert {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px;
  background: rgb(var(--af-accent-danger-rgb), 0.1);
  border-radius: 6px;
  color: var(--af-accent-danger);
  font-size: 13px;
  margin-bottom: 12px;
}

.sse-alert svg {
  width: 20px;
  height: 20px;
  flex-shrink: 0;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 48px 24px;
  color: var(--af-text-muted);
}

.empty-state svg {
  width: 48px;
  height: 48px;
  opacity: 0.5;
  margin-bottom: 12px;
}

.empty-state p {
  margin: 0;
  font-size: 13px;
}

.view-more {
  padding: 8px;
  color: var(--af-text-muted);
  font-size: 12px;
  text-align: center;
  cursor: pointer;
}

.view-more:hover {
  color: var(--af-accent-primary);
}

/* Vue TransitionGroup 动画 */
.list-enter-active {
  transition: all 0.6s ease;
}

.list-leave-active {
  transition:
    opacity 0.35s ease-out,
    transform 0.5s ease-out;
}

.list-enter-from {
  opacity: 0;
  transform: translateY(-10px);
}

.list-leave-to {
  opacity: 0;
  transform: translateY(-30px) scale(0.95);
}

.list-move {
  transition: transform 0.5s ease-out;
}

/* 新完成任务高亮 */
.newly-finished {
  animation: highlight 2s ease-out;
}

@keyframes highlight {
  0%,
  100% {
    background: transparent;
  }

  50% {
    background: rgb(var(--af-accent-success-rgb), 0.15);
  }
}

/* 拖动占位符样式 */
.task-ghost {
  opacity: 0.5;
  background: var(--af-bg-tertiary);
  border: 2px dashed var(--af-border-default);
}
</style>
