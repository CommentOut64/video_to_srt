<template>
  <main class="task-main" :class="[`sort-${sortMode}`, `display-${displayMode}`]">
    <div v-if="!hasTasks" class="empty-state">
      <svg class="empty-icon" viewBox="0 0 24 24" fill="currentColor">
        <path
          d="M21 3H3c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H3V5h18v14zM5 10h9v2H5zm0-3h9v2H5zm0 6h6v2H5z"
        />
      </svg>
      <h2 class="empty-title">还没有任务</h2>
      <p class="empty-desc">导入字幕或上传视频开始创建任务</p>
      <div class="empty-actions">
        <el-button v-if="canRenderProjectCreateAction" size="large" @click="$emit('open-import')">
          <el-icon><Document /></el-icon>
          导入字幕
        </el-button>
        <el-button
          v-if="canRenderTranscribeAction"
          type="primary"
          size="large"
          @click="$emit('open-upload')"
        >
          <el-icon><Upload /></el-icon>
          上传视频
        </el-button>
      </div>
    </div>

    <div v-else class="task-content">
      <template v-if="sortMode === 'grouped'">
        <section
          v-for="section in groupedSections"
          :key="section.key"
          class="task-group"
          :class="`task-group-${section.variant || 'default'}`"
        >
          <button
            class="group-header"
            :class="{ collapsed: isGroupCollapsed(section.key) }"
            @click="$emit('toggle-group', section.key)"
          >
            <div class="group-header-left">
              <h3 class="group-title">{{ section.title }}</h3>
              <span v-if="section.description" class="group-desc">{{ section.description }}</span>
            </div>
            <div class="group-header-right">
              <span class="group-count">{{ section.tasks.length }}</span>
              <el-icon class="collapse-icon" :class="{ collapsed: isGroupCollapsed(section.key) }">
                <ArrowDown />
              </el-icon>
            </div>
          </button>

          <div v-show="!isGroupCollapsed(section.key)" class="group-body">
            <div v-if="displayMode === 'card'" class="task-grid">
              <article
                v-for="task in section.tasks"
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
              </article>
            </div>

            <div v-else class="task-list">
              <div class="task-list-head">
                <span>任务</span>
                <span>状态</span>
                <span>更新时间</span>
                <span>进度</span>
                <span>操作</span>
              </div>
              <div
                v-for="task in section.tasks"
                :key="task.job_id"
                class="task-list-row"
                :class="`status-${task.status}`"
              >
                <div class="row-title">
                  <input
                    v-if="editingTaskId === task.job_id"
                    :value="editingTitle"
                    class="task-title-input"
                    @input="emit('update:editing-title', $event.target.value)"
                    @blur="$emit('finish-edit-title', task)"
                    @keyup.enter="$emit('finish-edit-title', task)"
                    @keyup.esc="$emit('cancel-edit-title')"
                  />
                  <el-popover
                    v-else
                    :content="getTaskDisplayName(task) + ' (点击查看，双击重命名)'"
                    placement="top"
                    trigger="hover"
                    popper-class="hint-popover-compact"
                    :show-after="500"
                  >
                    <template #reference>
                      <button
                        class="task-title task-title-link"
                        @click="$emit('title-click', task)"
                        @dblclick.prevent="$emit('start-edit-title', task)"
                      >
                        {{ getTaskDisplayName(task) }}
                      </button>
                    </template>
                  </el-popover>
                </div>
                <span class="row-status">{{ getStatusText(task.status) }}</span>
                <span class="row-time">{{ formatDate(getTaskTime(task)) }}</span>
                <span class="row-progress">
                  {{ task.status === 'processing' || task.status === 'queued' ? `${(task.progress || 0).toFixed(1)}%` : '-' }}
                </span>
                <div class="row-actions">
                  <el-button type="primary" link @click="$emit('open-editor', task.job_id)">
                    {{ task.status === "finished" ? "编辑" : "查看" }}
                  </el-button>
                  <el-button link @click="$emit('start-edit-title', task)">重命名</el-button>
                  <el-button link @click="$emit('delete-task', task.job_id)">删除</el-button>
                </div>
              </div>
            </div>
          </div>
        </section>
      </template>

      <template v-else>
        <div v-if="displayMode === 'card'" class="task-grid">
          <article
            v-for="task in sortedTasks"
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
          </article>
        </div>

        <div v-else class="task-list">
          <div class="task-list-head">
            <span>任务</span>
            <span>状态</span>
            <span>更新时间</span>
            <span>进度</span>
            <span>操作</span>
          </div>
          <div
            v-for="task in sortedTasks"
            :key="task.job_id"
            class="task-list-row"
            :class="`status-${task.status}`"
          >
            <div class="row-title">
              <input
                v-if="editingTaskId === task.job_id"
                :value="editingTitle"
                class="task-title-input"
                @input="emit('update:editing-title', $event.target.value)"
                @blur="$emit('finish-edit-title', task)"
                @keyup.enter="$emit('finish-edit-title', task)"
                @keyup.esc="$emit('cancel-edit-title')"
              />
              <el-popover
                v-else
                :content="getTaskDisplayName(task) + ' (点击查看，双击重命名)'"
                placement="top"
                trigger="hover"
                popper-class="hint-popover-compact"
                :show-after="500"
              >
                <template #reference>
                  <button
                    class="task-title task-title-link"
                    @click="$emit('title-click', task)"
                    @dblclick.prevent="$emit('start-edit-title', task)"
                  >
                    {{ getTaskDisplayName(task) }}
                  </button>
                </template>
              </el-popover>
            </div>
            <span class="row-status">{{ getStatusText(task.status) }}</span>
            <span class="row-time">{{ formatDate(getTaskTime(task)) }}</span>
            <span class="row-progress">
              {{ task.status === 'processing' || task.status === 'queued' ? `${(task.progress || 0).toFixed(1)}%` : '-' }}
            </span>
            <div class="row-actions">
              <el-button type="primary" link @click="$emit('open-editor', task.job_id)">
                {{ task.status === "finished" ? "编辑" : "查看" }}
              </el-button>
              <el-button link @click="$emit('start-edit-title', task)">重命名</el-button>
              <el-button link @click="$emit('delete-task', task.job_id)">删除</el-button>
            </div>
          </div>
        </div>
      </template>
    </div>
  </main>
</template>

<script setup>
import { computed } from "vue";
import { Upload, Edit, Delete, Clock, Loading, Document, ArrowDown } from "@element-plus/icons-vue";
import { selectRouteVisibility } from "@/state/capabilities/capabilitySelector";

const routeVisibility = selectRouteVisibility();
const canRenderTranscribeAction = routeVisibility.transcribeCreate;
const canRenderProjectCreateAction = routeVisibility.projectCreate;

const props = defineProps({
  tasks: {
    type: Array,
    default: () => [],
  },
  groupedSections: {
    type: Array,
    default: () => [],
  },
  sortedTasks: {
    type: Array,
    default: () => [],
  },
  displayMode: {
    type: String,
    default: "card",
  },
  sortMode: {
    type: String,
    default: "grouped",
  },
  groupCollapseState: {
    type: Object,
    default: () => ({}),
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

const hasTasks = computed(() => {
  if (props.sortMode === "grouped") {
    return props.groupedSections.some((section) => section.tasks && section.tasks.length > 0);
  }
  return props.sortedTasks.length > 0;
});

function isGroupCollapsed(groupKey) {
  return !!props.groupCollapseState?.[groupKey];
}

function getTaskTime(task) {
  return task.serverUpdatedAt || task.updatedAt || task.createdAt || 0;
}

const emit = defineEmits([
  "open-upload",
  "open-import",
  "retry-thumbnail",
  "update:editing-title",
  "finish-edit-title",
  "cancel-edit-title",
  "title-click",
  "start-edit-title",
  "open-editor",
  "delete-task",
  "toggle-group",
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
  height: 100%;
  min-height: 0;
  overflow-y: auto;
  scrollbar-width: none;
  width: 100%;
  padding: 32px 24px;
  margin: 0 auto;
  max-width: 1400px;
}

.task-main.sort-grouped {
  width: min(1860px, calc(100vw - 40px));
  max-width: 1860px;
}

.task-main::-webkit-scrollbar {
  display: none;
}

.task-content {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.task-group {
  padding: 10px 12px 12px;
  background: var(--af-bg-secondary);
  border: 1px solid var(--af-border-default);
  border-radius: var(--af-radius-lg);
}

.task-group-primary {
  border-color: rgba(var(--af-accent-primary-rgb), 0.35);
}

.task-group-danger {
  border-color: rgba(var(--af-accent-danger-rgb), 0.35);
}

.task-group-success {
  border-color: rgba(var(--af-accent-success-rgb), 0.35);
}

.group-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  padding: 0;
  background: transparent;
  border: none;
  color: var(--af-text-primary);
  cursor: pointer;
}

.group-header-left {
  display: flex;
  align-items: baseline;
  gap: 8px;
}

.group-title {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
}

.group-desc {
  color: var(--af-text-secondary);
  font-size: 12px;
}

.group-header-right {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--af-text-secondary);
}

.group-count {
  min-width: 24px;
  padding: 2px 8px;
  background: var(--af-bg-tertiary);
  border-radius: var(--af-radius-full);
  color: var(--af-text-normal);
  font-size: 12px;
  font-weight: 600;
  text-align: center;
}

.collapse-icon {
  transition: transform var(--af-transition-fast);
}

.collapse-icon.collapsed {
  transform: rotate(-90deg);
}

.group-body {
  margin-top: 10px;
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

.empty-state .empty-actions {
  display: flex;
  gap: 12px;
}

.task-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 24px;
}

.task-main.sort-grouped.display-card .group-body > .task-grid {
  grid-template-columns: repeat(5, minmax(0, 1fr));
  justify-content: stretch;
}

.task-list {
  overflow: hidden;
  border: 1px solid var(--af-border-default);
  border-radius: var(--af-radius-md);
}

.task-list-head {
  display: grid;
  grid-template-columns: minmax(220px, 2fr) 120px 160px 100px 180px;
  align-items: center;
  padding: 10px 14px;
  background: var(--af-bg-tertiary);
  color: var(--af-text-normal);
  font-size: 12px;
  font-weight: 600;
}

.task-list-row {
  display: grid;
  grid-template-columns: minmax(220px, 2fr) 120px 160px 100px 180px;
  align-items: center;
  padding: 10px 14px;
  border-top: 1px solid var(--af-border-muted);
}

.task-list-row:hover {
  background: var(--af-bg-tertiary);
}

.row-title {
  min-width: 0;
}

.row-title .task-title {
  overflow: hidden;
  background: transparent;
  border: none;
  color: var(--af-text-primary);
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 14px;
  text-align: left;
}

.row-title .task-title.task-title-link:hover {
  color: var(--af-accent-primary);
  text-decoration: underline;
}

.row-status {
  color: var(--af-text-normal);
  font-size: 12px;
  font-weight: 500;
}

.row-time,
.row-progress {
  color: var(--af-text-secondary);
  font-size: 12px;
}

.row-actions {
  display: flex;
  align-items: center;
  gap: 6px;
}

/* --- el-button 链接态样式定制 ---
 * 原因：列表模式下按钮文字在部分状态下会继承异常颜色，显式收口主题色
 */
:deep(.task-list .row-actions .el-button.is-link:not(.el-button--primary)) {
  color: var(--af-text-secondary);
}

:deep(.task-list .row-actions .el-button.is-link.el-button--primary) {
  color: var(--af-accent-primary);
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
  background: var(--af-bg-elevated);
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
  background: rgba(var(--af-text-muted-rgb), 0.08);
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
  background: rgba(var(--af-text-muted-rgb), 0.55);
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
  box-shadow: 0 0 0 2px rgba(var(--af-accent-primary-rgb), 0.2);
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

  .task-content {
    gap: 12px;
  }

  .task-group {
    padding: 10px;
  }

  .task-list-head {
    display: none;
  }

  .task-list-row {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }

  .row-actions {
    width: 100%;
    justify-content: flex-start;
  }

  .task-grid {
    grid-template-columns: 1fr;
    gap: 16px;
  }

  .task-main.sort-grouped.display-card .group-body > .task-grid {
    grid-template-columns: 1fr;
  }

  .task-card .task-actions {
    flex-direction: column;
  }
}

@media (max-width: 1760px) {
  .task-main.sort-grouped.display-card .group-body > .task-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }
}

@media (max-width: 1460px) {
  .task-main.sort-grouped.display-card .group-body > .task-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

@media (max-width: 1160px) {
  .task-main.sort-grouped.display-card .group-body > .task-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
