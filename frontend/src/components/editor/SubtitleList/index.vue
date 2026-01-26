<template>
  <div class="subtitle-list">
    <!-- 工具栏 -->
    <div class="list-toolbar">
      <div class="toolbar-left">
        <span class="subtitle-count">{{ totalSubtitles }} 条字幕</span>
        <!-- Phase 5: 草稿/定稿计数 -->
        <span v-if="draftCount > 0" class="draft-count">({{ draftCount }} 草稿)</span>
      </div>

      <div class="toolbar-center">
        <div class="search-box">
          <svg class="search-icon" viewBox="0 0 24 24" fill="currentColor">
            <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
          </svg>
          <input
            v-model="searchText"
            type="text"
            placeholder="搜索字幕..."
            class="search-input"
          />
          <button v-if="searchText" class="search-clear" @click="searchText = ''">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
            </svg>
          </button>
        </div>
      </div>

      <div class="toolbar-right">
        <el-tooltip content="添加字幕" placement="bottom" :show-after="500">
          <button class="toolbar-btn" @click="addNewSubtitle">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/>
            </svg>
          </button>
        </el-tooltip>
      </div>
    </div>

    <!-- 字幕列表 (Phase 5: 使用 SubtitleItem 组件) -->
    <div class="list-container" ref="listRef">
      <div v-if="filteredSubtitles.length === 0" class="empty-state">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M20 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zM4 18V6h16v12H4zm2-1h2v-2H6v2zm0-3h8v-2H6v2zm10 3h2v-5h-2v5zm-4 0h2v-2h-2v2zm0-3h4v-2h-4v2z"/>
        </svg>
        <p>暂无字幕</p>
        <button class="add-first-btn" @click="addNewSubtitle">添加第一条字幕</button>
      </div>

      <!-- Phase 5: 使用 SubtitleItem 组件替代内联渲染 -->
      <!-- 添加 TransitionGroup 实现切分动画，批量更新时禁用 -->
      <TransitionGroup :name="animationEnabled ? 'subtitle-list' : ''" tag="div">
        <SubtitleItem
          v-for="(subtitle, index) in filteredSubtitles"
          :key="subtitle.id"
          :subtitle="subtitle"
          :index="index"
          :is-active="activeSubtitleId === subtitle.id"
          :is-current="currentSubtitleId === subtitle.id"
          :editable="props.editable"
          @click="onSubtitleClick"
          @update-time="updateTime"
          @update-text="updateText"
          @delete="deleteSubtitle"
          @insert-before="insertBefore(index)"
          @insert-after="insertAfter(index)"
        />
      </TransitionGroup>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import { useProjectStore } from '@/stores/projectStore'
import { usePlaybackManager } from '@/services/PlaybackManager'
import { useSubtitleSync } from '@/composables'  // V3.2.0+dev.20260124.01: 导入字幕同步
import transcriptionApi from '@/services/api/transcriptionApi'
// Phase 5: 导入 SubtitleItem 组件
import SubtitleItem from './SubtitleItem.vue'

// Props
const props = defineProps({
  autoScroll: { type: Boolean, default: true },
  editable: { type: Boolean, default: true }
})

const emit = defineEmits(['subtitle-click', 'subtitle-edit', 'subtitle-delete', 'subtitle-add'])

// Store
const projectStore = useProjectStore()

// 全局播放管理器
const playbackManager = usePlaybackManager()

const jobId = computed(() => projectStore.meta.jobId)
// V3.2.0+dev.20260124.02: 字幕同步（防止 AI 覆盖用户编辑）
const { onSubtitleEdit, applyPendingEditsToStore, forceSyncNow, pendingCount } = useSubtitleSync(jobId)

// Refs
const listRef = ref(null)

// State
const searchText = ref('')
const animationEnabled = ref(true)  // 批量更新时禁用动画
let previousSubtitleCount = 0  // 上一次字幕数量
const hasAppliedPending = ref(false)

// Computed
const subtitles = computed(() => projectStore.subtitles)
const totalSubtitles = computed(() => projectStore.totalSubtitles)
const currentSubtitleId = computed(() => projectStore.currentSubtitle?.id)
const activeSubtitleId = computed(() => projectStore.view.selectedSubtitleId)
// Phase 5: 草稿计数
const draftCount = computed(() => projectStore.draftSubtitleCount)

const filteredSubtitles = computed(() => {
  if (!searchText.value) return subtitles.value
  const search = searchText.value.toLowerCase()
  return subtitles.value.filter(sub => sub.text.toLowerCase().includes(search))
})

watch(
  () => jobId.value,
  async () => {
    hasAppliedPending.value = false
    if (pendingCount() > 0) {
      await forceSyncNow()
    }
  }
)

watch(
  () => subtitles.value.length,
  async (length) => {
    if (!jobId.value || length === 0 || hasAppliedPending.value) return
    const applied = applyPendingEditsToStore(projectStore)
    if (applied > 0) {
      await forceSyncNow()
    }
    hasAppliedPending.value = true
  },
  { immediate: true }
)

// Methods
function onSubtitleClick(subtitle) {
  projectStore.view.selectedSubtitleId = subtitle.id
  // 使用 PlaybackManager 进行跳转，确保视频和波形同步
  playbackManager.seekTo(subtitle.start)
  emit('subtitle-click', subtitle)
}

function updateTime(id, field, value) {
  if (isNaN(value)) return

  // 1. 乐观更新本地状态
  projectStore.updateSubtitle(id, { [field]: value }, { isUserEdit: true })
  emit('subtitle-edit', id, field, value)

  // 2. V3.2.0+dev.20260124.01: 同步到后端（防抖）
  const subtitle = projectStore.subtitles.find(s => s.id === id)
  if (subtitle && subtitle.sentenceIndex !== undefined) {
    onSubtitleEdit(subtitle.sentenceIndex, {
      [field]: value
    })
  }
}

function updateText(id, text) {
  // 1. 乐观更新本地状态
  projectStore.updateSubtitle(id, { text }, { isUserEdit: true })
  emit('subtitle-edit', id, 'text', text)

  // 2. V3.2.0+dev.20260124.01: 同步到后端（防抖）
  const subtitle = projectStore.subtitles.find(s => s.id === id)
  if (subtitle && subtitle.sentenceIndex !== undefined) {
    onSubtitleEdit(subtitle.sentenceIndex, {
      text
    })
  }
}

async function deleteSubtitle(id) {
  const subtitle = projectStore.subtitles.find(s => s.id === id)
  projectStore.removeSubtitle(id, { isUserEdit: true })
  emit('subtitle-delete', id)

  if (!subtitle || subtitle.sentenceIndex === undefined) {
    return
  }

  try {
    await transcriptionApi.deleteSubtitle(projectStore.meta.jobId, subtitle.sentenceIndex)
  } catch (error) {
    console.warn('[SubtitleList] 删除字幕同步失败:', error)
  }
}

async function addNewSubtitle() {
  const lastSubtitle = subtitles.value[subtitles.value.length - 1]
  const newStart = lastSubtitle ? lastSubtitle.end : 0
  const insertIndex = subtitles.value.length
  projectStore.addSubtitle(insertIndex, {
    start: newStart,
    end: newStart + 3,
    text: '',
    isModified: true,
    source: 'manual'
  })
  nextTick(() => {
    scrollToBottom()
  })
  emit('subtitle-add', subtitles.value.length - 1)

  try {
    const response = await transcriptionApi.createSubtitle(projectStore.meta.jobId, {
      text: '',
      start: newStart,
      end: newStart + 3
    })
    const data = response?.data?.data || response?.data
    if (data?.index !== undefined) {
      const newSubtitle = projectStore.subtitles[insertIndex]
      if (newSubtitle) {
        projectStore.updateSubtitle(newSubtitle.id, {
          sentenceIndex: data.index,
          isModified: true,
          source: data.source || 'manual'
        }, { isUserEdit: true })
      }
    }
  } catch (error) {
    console.warn('[SubtitleList] 新增字幕同步失败:', error)
  }
}

function insertBefore(index) {
  const current = subtitles.value[index]
  const prev = subtitles.value[index - 1]
  const start = prev ? prev.end : Math.max(0, current.start - 3)
  const end = current.start
  projectStore.addSubtitle(index, { start, end, text: '', isModified: true, source: 'manual' })
  syncInsertedSubtitle(index, start, end, '')
}

function insertAfter(index) {
  const current = subtitles.value[index]
  const next = subtitles.value[index + 1]
  const start = current.end
  const end = next ? next.start : current.end + 3
  projectStore.addSubtitle(index + 1, { start, end, text: '', isModified: true, source: 'manual' })
  syncInsertedSubtitle(index + 1, start, end, '')
}

async function syncInsertedSubtitle(insertIndex, start, end, text) {
  try {
    const response = await transcriptionApi.createSubtitle(projectStore.meta.jobId, {
      text,
      start,
      end
    })
    const data = response?.data?.data || response?.data
    if (data?.index !== undefined) {
      const newSubtitle = projectStore.subtitles[insertIndex]
      if (newSubtitle) {
        projectStore.updateSubtitle(newSubtitle.id, {
          sentenceIndex: data.index,
          isModified: true,
          source: data.source || 'manual'
        }, { isUserEdit: true })
      }
    }
  } catch (error) {
    console.warn('[SubtitleList] 插入字幕同步失败:', error)
  }
}

function scrollToBottom() {
  if (listRef.value) {
    listRef.value.scrollTop = listRef.value.scrollHeight
  }
}

function scrollToItem(index) {
  const items = listRef.value?.querySelectorAll('.subtitle-item')
  if (items && items[index]) {
    items[index].scrollIntoView({ behavior: 'smooth', block: 'center' })
  }
}

// 自动滚动跟随当前播放
watch(currentSubtitleId, (id) => {
  if (!props.autoScroll || !id) return
  const index = filteredSubtitles.value.findIndex(s => s.id === id)
  if (index !== -1) {
    nextTick(() => scrollToItem(index))
  }
})

// 批量更新检测：超过阈值时禁用动画，避免重叠闪烁
const BATCH_UPDATE_THRESHOLD = 5
watch(subtitles, (newList) => {
  const newCount = newList.length
  const diff = Math.abs(newCount - previousSubtitleCount)

  if (diff > BATCH_UPDATE_THRESHOLD) {
    // 批量更新，禁用动画
    animationEnabled.value = false
    // 下一帧恢复动画（确保本次渲染完成）
    nextTick(() => {
      animationEnabled.value = true
    })
  }

  previousSubtitleCount = newCount
}, { flush: 'pre' })  // pre: 在 DOM 更新前触发
</script>

<style lang="scss" scoped>
.subtitle-list {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--bg-primary);
}

// 工具栏 - 针对 350px 宽度优化
.list-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;  // 减少内边距
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-default);
  gap: 12px;

  .toolbar-left {
    .subtitle-count {
      font-size: 12px;
      color: var(--text-secondary);
      white-space: nowrap;
    }
    // Phase 5: 草稿计数样式
    .draft-count {
      font-size: 12px;
      color: var(--warning);
      margin-left: 4px;
    }
  }

  .toolbar-center {
    flex: 1;
    max-width: 180px;  // 缩小搜索框
    min-width: 100px;
  }

  .search-box {
    display: flex;
    align-items: center;
    background: var(--bg-tertiary);
    border-radius: var(--radius-md);
    padding: 5px 10px;
    gap: 6px;

    .search-icon {
      width: 14px;
      height: 14px;
      color: var(--text-muted);
      flex-shrink: 0;
    }

    .search-input {
      flex: 1;
      min-width: 0;
      background: transparent;
      border: none;
      color: var(--text-normal);
      font-size: 12px;

      &::placeholder { color: var(--text-muted); }
    }

    .search-clear {
      width: 16px;
      height: 16px;
      color: var(--text-muted);
      flex-shrink: 0;
      svg { width: 100%; height: 100%; }
      &:hover { color: var(--text-normal); }
    }
  }

  .toolbar-btn {
    width: 30px;
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    transition: all var(--transition-fast);
    flex-shrink: 0;

    svg { width: 18px; height: 18px; }

    &:hover {
      background: var(--bg-tertiary);
      color: var(--primary);
    }
  }
}

// 列表容器
.list-container {
  flex: 1;
  overflow-y: auto;
  padding: 6px;
  position: relative;

  &::-webkit-scrollbar {
    width: 6px;
  }
  &::-webkit-scrollbar-track {
    background: transparent;
  }
  &::-webkit-scrollbar-thumb {
    background: var(--border-default);
    border-radius: 3px;
    &:hover { background: var(--text-muted); }
  }
}

// 空状态
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 32px 16px;
  color: var(--text-muted);

  svg {
    width: 48px;
    height: 48px;
    margin-bottom: 12px;
    opacity: 0.5;
  }

  p {
    font-size: 13px;
    margin-bottom: 12px;
  }

  .add-first-btn {
    padding: 6px 16px;
    background: var(--primary);
    color: white;
    border-radius: var(--radius-md);
    font-size: 13px;
    transition: background var(--transition-fast);
    &:hover { background: var(--primary-hover); }
  }
}

// 字幕项 - 紧凑布局
.subtitle-item {
  display: flex;
  gap: 10px;
  padding: 10px;  // 减少内边距
  margin-bottom: 6px;
  background: var(--bg-secondary);
  border: 1px solid transparent;
  border-radius: var(--radius-md);
  transition: all var(--transition-fast);
  cursor: pointer;

  &:hover {
    background: var(--bg-tertiary);

    .item-actions { opacity: 1; }
  }

  &.is-active {
    border-color: var(--primary);
    background: rgba(88, 166, 255, 0.08);
  }

  &.is-current {
    border-color: var(--success);
    background: rgba(63, 185, 80, 0.08);

    .item-index { background: var(--success); color: white; }
  }

  // 置信度警告高亮样式
  &.warning-low-confidence {
    border-color: var(--warning);
    background: rgba(210, 153, 34, 0.06);

    .item-index {
      background: var(--warning);
      color: white;
    }
  }

  &.warning-high-perplexity {
    border-color: #e67700;
    background: rgba(230, 119, 0, 0.06);

    .item-index {
      background: #e67700;
      color: white;
    }
  }

  &.warning-both {
    border-color: var(--danger);
    background: rgba(248, 81, 73, 0.08);
    border-width: 2px;

    .item-index {
      background: var(--danger);
      color: white;
    }
  }
}

// 序号 - 缩小尺寸
.item-index {
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-tertiary);
  border-radius: var(--radius-sm);
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  flex-shrink: 0;
}

// 内容区
.item-content {
  flex: 1;
  min-width: 0;
}

// 时间行 - 优化间距
.time-row {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
  flex-wrap: wrap;

  .time-input {
    width: 75px;  // 缩小宽度
    padding: 3px 6px;
    background: var(--bg-tertiary);
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text-normal);
    text-align: center;

    &:focus {
      border-color: var(--primary);
      outline: none;
    }
  }

  .time-arrow {
    color: var(--text-muted);
    svg { width: 14px; height: 14px; }
  }

  .duration-tag {
    padding: 2px 6px;
    background: var(--bg-tertiary);
    border-radius: var(--radius-full);
    font-size: 10px;
    font-family: var(--font-mono);
    color: var(--text-muted);
  }
}

// 文本行 - 优化尺寸（支持 Toggle Mode）
.text-row {
  position: relative;

  // 只读高亮视图
  .text-display {
    width: 100%;
    min-height: 45px;
    padding: 6px 35px 6px 8px;
    background: var(--bg-tertiary);
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    font-size: 12px;
    color: var(--text-normal);
    line-height: 1.4;
    white-space: pre-wrap;
    word-break: break-word;
    cursor: default;

    &.can-edit {
      cursor: text;
      &:hover {
        border-color: var(--primary);
        background: var(--bg-secondary);
      }
    }

    // 字级警告高亮样式
    :deep(.word-warning) {
      background-color: rgba(255, 193, 7, 0.25);
      border-bottom: 2px solid var(--warning, #ffc107);
      padding: 0 2px;
      border-radius: 2px;
    }

    :deep(.word-critical) {
      background-color: rgba(244, 67, 54, 0.25);
      border-bottom: 2px solid var(--error, #f44336);
      padding: 0 2px;
      border-radius: 2px;
      font-weight: 500;
    }
  }

  .text-input {
    width: 100%;
    padding: 6px 8px;
    padding-right: 35px;
    background: var(--bg-tertiary);
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    font-size: 12px;
    color: var(--text-normal);
    resize: none;
    line-height: 1.4;

    &:focus {
      border-color: var(--primary);
      outline: none;
    }

    &::placeholder { color: var(--text-muted); }
  }

  .char-count {
    position: absolute;
    right: 6px;
    bottom: 6px;
    font-size: 10px;
    font-family: var(--font-mono);
    color: var(--text-muted);

  }
}

// 操作按钮 - 始终可见，更小尺寸
.item-actions {
  display: flex;
  flex-direction: column;
  gap: 2px;
  opacity: 1;  // 始终显示

  .action-btn {
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--radius-sm);
    color: var(--text-muted);
    transition: all var(--transition-fast);

    svg { width: 14px; height: 14px; }

    &:hover {
      background: var(--bg-tertiary);
      color: var(--text-normal);
    }

    &--danger:hover {
      background: rgba(248, 81, 73, 0.15);
      color: var(--danger);
    }
  }
}

// 字幕切分动画
.subtitle-list-move {
  transition: transform 0.3s ease;
}

.subtitle-list-enter-active,
.subtitle-list-leave-active {
  transition: all 0.2s ease;
}

.subtitle-list-enter-from,
.subtitle-list-leave-to {
  opacity: 0;
  transform: scaleY(0.3);
  margin-top: 0;
  margin-bottom: 0;
}

.subtitle-list-leave-active {
  position: absolute;
  width: calc(100% - 32px);
}
</style>
