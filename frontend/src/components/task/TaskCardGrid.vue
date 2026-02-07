<template>
  <main class="task-main">
    <div v-if="tasks.length === 0" class="empty-state">
      <svg class="empty-icon" viewBox="0 0 24 24" fill="currentColor">
        <path
          d="M21 3H3c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H3V5h18v14zM5 10h9v2H5zm0-3h9v2H5zm0 6h6v2H5z"
        />
      </svg>
      <h2 class="empty-title">还没有任务</h2>
      <p class="empty-desc">点击上方"上传视频"按钮开始创建字幕任务</p>
      <el-button type="primary" size="large" @click="$emit('open-upload')">
        <el-icon><Upload /></el-icon>
        上传视频
      </el-button>
    </div>

    <div v-else class="task-grid">
      <div
        v-for="task in tasks"
        :key="task.job_id"
        class="task-card"
        :class="`status-${task.status}`"
      >
        <div class="task-thumbnail">
          <img
            v-if="thumbnailCache[task.job_id] && thumbnailCache[task.job_id] !== null"
            :src="thumbnailCache[task.job_id]"
            class="thumbnail-image"
            alt="Video thumbnail"
          />
          <div
            v-else
            class="thumbnail-placeholder"
            :class="{ clickable: thumbnailCache[task.job_id] === null }"
            @click.stop="thumbnailCache[task.job_id] === null && $emit('retry-thumbnail', task.job_id)"
            :title="thumbnailCache[task.job_id] === null ? '点击重试' : ''"
          >
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M8 5v14l11-7z" />
            </svg>
          </div>
          <div v-if="thumbnailCache[task.job_id] === undefined" class="thumbnail-loading">
            <svg class="loading-spinner" viewBox="0 0 24 24">
              <circle
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                stroke-width="2"
                fill="none"
              />
            </svg>
          </div>
          <div v-if="task.status !== 'finished'" class="status-overlay">
            <span class="status-text">{{ getStatusText(task.status) }}</span>
          </div>
        </div>

        <div class="task-info">
          <div class="task-title-wrapper">
            <input
              v-if="editingTaskId === task.job_id"
              :value="editingTitle"
              class="task-title-input"
              @input="emit('update:editing-title', $event.target.value)"
              @blur="$emit('finish-edit-title', task)"
              @keyup.enter="$emit('finish-edit-title', task)"
              @keyup.esc="$emit('cancel-edit-title')"
            />
            <h3
              v-else
              class="task-title task-title-link"
              :title="getTaskDisplayName(task) + ' (点击查看，双击重命名)'"
              @click="$emit('title-click', task)"
              @dblclick.prevent="$emit('start-edit-title', task)"
            >
              {{ getTaskDisplayName(task) }}
            </h3>
          </div>
          <div class="task-meta">
            <span class="meta-item">
              <el-icon><Clock /></el-icon>
              {{ formatDate(task.createdAt) }}
            </span>
            <span v-if="task.status === 'processing' || task.status === 'queued'" class="meta-item">
              <el-icon><Loading /></el-icon>
              {{ (task.progress || 0).toFixed(1) }}%
            </span>
          </div>

          <el-progress
            v-if="task.status === 'processing' || task.status === 'queued'"
            :percentage="task.progress"
            :show-text="false"
            :stroke-width="4"
          />
        </div>

        <div class="task-actions">
          <el-button type="primary" size="small" @click="$emit('open-editor', task.job_id)">
            <el-icon><Edit /></el-icon>
            {{ task.status === "finished" ? "编辑" : "查看" }}
          </el-button>
          <el-button size="small" @click="$emit('start-edit-title', task)">重命名</el-button>
          <el-button size="small" @click="$emit('delete-task', task.job_id)">
            <el-icon><Delete /></el-icon>
            删除
          </el-button>
        </div>
      </div>
    </div>
  </main>
</template>

<script setup>
import { Upload, Edit, Delete, Clock, Loading } from "@element-plus/icons-vue";

defineProps({
  tasks: {
    type: Array,
    default: () => [],
  },
  thumbnailCache: {
    type: Object,
    default: () => ({}),
  },
  editingTaskId: {
    type: [String, Number],
    default: null,
  },
  editingTitle: {
    type: String,
    default: "",
  },
  getStatusText: {
    type: Function,
    required: true,
  },
  formatDate: {
    type: Function,
    required: true,
  },
  getTaskDisplayName: {
    type: Function,
    required: true,
  },
});

const emit = defineEmits([
  "open-upload",
  "retry-thumbnail",
  "update:editing-title",
  "finish-edit-title",
  "cancel-edit-title",
  "title-click",
  "start-edit-title",
  "open-editor",
  "delete-task",
]);
</script>

<style scoped>
@keyframes spin {
  from {
    transform: rotate(0deg);
  }

  to {
    transform: rotate(360deg);
  }
}

.task-main {
  flex: 1;
  width: 100%;
  padding: 32px 24px;
  margin: 0 auto;
  max-width: 1400px;
}

.empty-state {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 80px 24px;
  text-align: center;
}

.empty-state .empty-icon {
  width: 120px;
  height: 120px;
  margin-bottom: 24px;
  color: var(--af-text-disabled);
  opacity: 0.5;
}

.empty-state .empty-title {
  margin: 0 0 12px;
  color: var(--af-text-primary);
  font-size: 24px;
}

.empty-state .empty-desc {
  margin: 0 0 32px;
  color: var(--af-text-secondary);
  font-size: 14px;
}

.task-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 24px;
}

.task-card {
  overflow: hidden;
  background: var(--af-bg-secondary);
  border: 1px solid var(--af-border-default);
  border-radius: var(--af-radius-lg);
}

.task-card .task-thumbnail {
  position: relative;
  width: 100%;
  padding-top: 56.25%;
  background: var(--af-bg-tertiary);
}

.task-card .task-thumbnail .thumbnail-image {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: var(--af-bg-tertiary);
  object-fit: cover;
}

.task-card .task-thumbnail .thumbnail-placeholder {
  position: absolute;
  top: 0;
  left: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
}

.task-card .task-thumbnail .thumbnail-placeholder svg {
  width: 64px;
  height: 64px;
  color: var(--af-text-disabled);
  opacity: 0.3;
}

.task-card .task-thumbnail .thumbnail-placeholder.clickable {
  cursor: pointer;
  transition: background var(--af-transition-fast);
}

.task-card .task-thumbnail .thumbnail-placeholder.clickable:hover {
  background: rgb(0 0 0 / 5%);
}

.task-card .task-thumbnail .thumbnail-placeholder.clickable:hover svg {
  opacity: 0.5;
}

.task-card .task-thumbnail .thumbnail-loading {
  position: absolute;
  top: 0;
  left: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
  background: rgb(0 0 0 / 10%);
  cursor: pointer;
}

.task-card .task-thumbnail .thumbnail-loading .loading-spinner {
  width: 40px;
  height: 40px;
  color: var(--af-accent-primary);
  opacity: 0.6;
  animation: spin 2s linear infinite;
}

.task-card .task-thumbnail .status-overlay {
  position: absolute;
  top: 0;
  left: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
  background: rgb(0 0 0 / 60%);
}

.task-card .task-thumbnail .status-overlay .status-text {
  color: var(--af-text-on-dark);
  font-size: 14px;
  font-weight: 500;
}

.task-card .task-info {
  padding: 16px;
}

.task-card .task-info .task-title-wrapper {
  margin-bottom: 8px;
}

.task-card .task-info .task-title {
  margin: 0;
  overflow: hidden;
  color: var(--af-text-primary);
  font-size: 15px;
  font-weight: 500;
  text-overflow: ellipsis;
  white-space: nowrap;
  cursor: pointer;
}

.task-card .task-info .task-title.task-title-link:hover {
  color: var(--af-accent-primary);
  text-decoration: underline;
}

.task-card .task-info .task-title-input {
  width: 100%;
  padding: 4px 8px;
  background: var(--af-bg-primary);
  border: 1px solid var(--af-accent-primary);
  border-radius: var(--af-radius-sm);
  color: var(--af-text-primary);
  font-size: 15px;
  font-weight: 500;
  outline: none;
  box-sizing: border-box;
}

.task-card .task-info .task-title-input:focus {
  box-shadow: 0 0 0 2px rgb(var(--af-accent-primary-rgb), 0.2);
}

.task-card .task-info .task-meta {
  display: flex;
  gap: 16px;
  margin-bottom: 12px;
}

.task-card .task-info .task-meta .meta-item {
  display: flex;
  align-items: center;
  gap: 4px;
  color: var(--af-text-secondary);
  font-size: 12px;
}

.task-card .task-info .task-meta .meta-item .el-icon {
  font-size: 14px;
}

.task-card .task-actions {
  display: flex;
  gap: 8px;
  padding: 0 16px 16px;
}

.task-card .task-actions .el-button {
  flex: 1;
}

@media (max-width: 768px) {
  .task-main {
    padding: 20px 16px;
  }

  .task-grid {
    grid-template-columns: 1fr;
    gap: 16px;
  }

  .task-card .task-actions {
    flex-direction: column;
  }
}
</style>
