<template>
  <div class="subtitle-list tw-flex tw-flex-col tw-h-full tw-bg-bg-primary">
    <!-- 搜索工具栏（整合原 list-toolbar） -->
    <SearchToolbar
      v-model:search-mode="homophoneSearch.searchMode"
      v-model:sort-mode="homophoneSearch.sortMode"
      v-model:search-text="homophoneSearch.searchText"
      v-model:search-reading="homophoneSearch.searchReading"
      v-model:is-ignore-punctuation="homophoneSearch.isIgnorePunctuation"
      v-model:replace-text="homophoneSearch.replaceText"
      :total-subtitles="totalSubtitles"
      :draft-count="draftCount"
      :is-searching="homophoneSearch.isSearching"
      :can-search="homophoneSearch.canSearch"
      :is-search-active="homophoneSearch.isSearchActive"
      :index-status="homophoneSearch.indexStatus"
      :match-count="homophoneSearch.matchCount"
      :selected-count="homophoneSearch.selectedCount"
      :is-all-selected="homophoneSearch.isAllSelected"
      :is-indeterminate="homophoneSearch.isIndeterminate"
      @search="handleHomophoneSearch"
      @reset="handleHomophoneReset"
      @batch-replace="handleBatchReplace"
      @toggle-select-all="handleToggleSelectAll"
      @add-subtitle="addNewSubtitle"
      @quick-search="handleQuickSearch"
    />

    <!-- 字幕列表 (使用 SubtitleItem 组件) -->
    <div class="list-container tw-flex-1 tw-overflow-y-auto tw-p-1.5 tw-relative" ref="listRef">
      <div v-if="filteredSubtitles.length === 0 && !homophoneSearch.isSearchActive" class="empty-state tw-flex tw-flex-col tw-items-center tw-justify-center tw-px-4 tw-py-8 tw-text-text-muted">
        <svg viewBox="0 0 24 24" fill="currentColor" class="tw-w-12 tw-h-12 tw-mb-3 tw-opacity-50">
          <path d="M20 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 14H4V6h16v12zM6 10h2v2H6v-2zm0 4h8v2H6v-2zm10 0h2v2h-2v-2zm0-4h2v2h-2v-2zm-4 4h2v2h-2v-2zm0-4h2v2h-2v-2z"/>
        </svg>
        <p class="tw-text-[13px] tw-mb-3">暂无字幕</p>
        <button class="add-first-btn" @click="addNewSubtitle">添加第一条字幕</button>
      </div>

      <!-- 搜索无结果提示 -->
      <div v-else-if="homophoneSearch.isSearchActive && homophoneSearch.matchCount === 0" class="empty-state tw-flex tw-flex-col tw-items-center tw-justify-center tw-px-4 tw-py-8 tw-text-text-muted">
        <p class="tw-text-[13px]">未找到匹配结果</p>
      </div>

      <!-- 分组模式渲染 -->
      <template v-else-if="homophoneSearch.isSearchActive && homophoneSearch.sortMode === SortMode.GROUPED">
        <template v-for="group in homophoneSearch.groupedViewItems" :key="group.clusterId">
          <GroupHeader
            :reading-label="group.readingLabel"
            :color="group.color"
            :item-count="group.items.length"
            :is-collapsed="collapsedGroups.has(group.clusterId)"
            :is-group-all-selected="isGroupAllSelected(group)"
            :is-group-indeterminate="isGroupIndeterminate(group)"
            @toggle-collapse="toggleGroupCollapse(group.clusterId)"
            @group-select-change="(checked) => handleGroupSelectChange(group, checked)"
          />
          <template v-if="!collapsedGroups.has(group.clusterId)">
            <SubtitleItem
              v-for="item in group.items"
              :key="item.subtitle.id"
              :subtitle="item.subtitle"
              :index="item.index"
              :is-active="activeSubtitleId === item.subtitle.id"
              :is-current="currentSubtitleId === item.subtitle.id"
              :editable="props.editable"
              :match-spans="item.matchSpans"
              :is-selected="item.isSelected"
              :is-selectable="true"
              @click="onSubtitleClick"
              @update-time="updateTime"
              @update-text="updateText"
              @delete="deleteSubtitle"
              @insert-before="insertBefore(item.index)"
              @insert-after="insertAfter(item.index)"
              @select-change="handleItemSelectChange"
            />
          </template>
        </template>
      </template>

      <!-- 时间线模式渲染（搜索激活） -->
      <template v-else-if="homophoneSearch.isSearchActive && homophoneSearch.sortMode === SortMode.TIMELINE">
        <SubtitleItem
          v-for="item in homophoneSearch.timelineViewItems"
          :key="item.subtitle.id"
          :subtitle="item.subtitle"
          :index="item.index"
          :is-active="activeSubtitleId === item.subtitle.id"
          :is-current="currentSubtitleId === item.subtitle.id"
          :editable="props.editable"
          :match-spans="item.matchSpans"
          :is-selected="item.isSelected"
          :is-selectable="true"
          :cluster-color="item.clusterColor"
          @click="onSubtitleClick"
          @update-time="updateTime"
          @update-text="updateText"
          @delete="deleteSubtitle"
          @insert-before="insertBefore(item.index)"
          @insert-after="insertAfter(item.index)"
          @select-change="handleItemSelectChange"
        />
      </template>

      <!-- 默认模式渲染（无搜索） -->
      <TransitionGroup v-else :name="animationEnabled ? 'subtitle-list' : ''" tag="div">
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
import { ref, reactive, computed, watch, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import { useProjectStore } from '@/stores/projectStore'
import { usePlaybackManager } from '@/services/PlaybackManager'
import { useSubtitleSync, useHomophoneSearch, SortMode } from '@/composables'
import transcriptionApi from '@/services/api/transcriptionApi'
// 导入组件
import SubtitleItem from './SubtitleItem.vue'
import SearchToolbar from './SearchToolbar.vue'
import GroupHeader from './GroupHeader.vue'

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

// 同音搜索 composable（用 reactive 包裹，使模板 v-model 能正确写入 ref.value）
const homophoneSearch = reactive(useHomophoneSearch({
  jobId,
  subtitles: computed(() => projectStore.subtitles),
}))

// 分组折叠状态
const collapsedGroups = ref(new Set())

// Refs
const listRef = ref(null)

// State
const quickSearchText = ref('')     // 快速搜索文本（简单过滤）
const animationEnabled = ref(true)  // 批量更新时禁用动画
let previousSubtitleCount = 0  // 上一次字幕数量
const hasAppliedPending = ref(false)

// Computed
const subtitles = computed(() => projectStore.subtitles)
const totalSubtitles = computed(() => projectStore.totalSubtitles)
const currentSubtitleId = computed(() => projectStore.currentSubtitle?.id)
const activeSubtitleId = computed(() => projectStore.view.selectedSubtitleId)
// 草稿计数
const draftCount = computed(() => projectStore.draftSubtitleCount)

const filteredSubtitles = computed(() => {
  if (!quickSearchText.value) return subtitles.value
  const search = quickSearchText.value.toLowerCase()
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
    const baseStart = projectStore.toBaseTime(newStart)
    const baseEnd = projectStore.toBaseTime(newStart + 3)
    const response = await transcriptionApi.createSubtitle(projectStore.meta.jobId, {
      text: '',
      start: baseStart,
      end: baseEnd
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
    const baseStart = projectStore.toBaseTime(start)
    const baseEnd = projectStore.toBaseTime(end)
    const response = await transcriptionApi.createSubtitle(projectStore.meta.jobId, {
      text,
      start: baseStart,
      end: baseEnd
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

// ========================
// 同音搜索相关方法
// ========================

// 执行搜索
async function handleHomophoneSearch() {
  await homophoneSearch.executeSearch()
}

// 重置搜索
function handleHomophoneReset() {
  homophoneSearch.resetSearch()
  quickSearchText.value = ''  // 同时清除快速搜索
}

// 快速搜索（简单文本过滤）
function handleQuickSearch(text) {
  quickSearchText.value = text
}

// 批量替换
async function handleBatchReplace() {
  const result = await homophoneSearch.executeBatchReplace()
  if (result.success) {
    ElMessage.success(`成功替换 ${result.count} 处`)
    // 仅增量回填被替换字幕，避免整表 import 覆写快流草稿/切分状态。
    // 设计说明：此前整量 importSegments 会将 isDraft/chunk_id 等运行态信息重置，
    // 在 processing 阶段可能让快流表现为“未切分”。
    if (jobId.value) {
      try {
        await forceSyncNow()
        const textData = await transcriptionApi.getTranscriptionText(jobId.value)
        if (Array.isArray(textData?.segments)) {
          const segmentMap = new Map(
            textData.segments.map((segment) => [Number(segment.id), segment])
          )

          let patchedCount = 0
          const updatedIndices = Array.isArray(result?.indices)
            ? result.indices
            : []

          for (const sentenceIndex of updatedIndices) {
            const idx = Number(sentenceIndex)
            const segment = segmentMap.get(idx)
            if (!segment) continue

            const subtitle = projectStore.subtitles.find(
              (item) => Number(item.sentenceIndex) === idx
            )
            if (!subtitle) continue

            projectStore.updateSubtitle(subtitle.id, {
              text: String(segment.text ?? subtitle.text ?? ''),
              isModified: Boolean(segment.is_modified ?? subtitle.isModified),
              originalText: segment.original_text ?? subtitle.originalText,
            })
            patchedCount += 1
          }

          // 兜底：若增量回填未命中本地字幕（如刚打开页面本地列表为空），再执行整量导入。
          if (patchedCount === 0 && projectStore.subtitles.length === 0) {
            projectStore.importSegments(textData.segments, {
              jobId: jobId.value,
              filename: projectStore.meta.filename,
              duration: projectStore.meta.duration,
              videoPath: projectStore.meta.videoPath,
              audioPath: projectStore.meta.audioPath,
              subtitleOffset: projectStore.subtitleOffset,
            })
          }
        }
      } catch (error) {
        console.error('刷新字幕失败:', error)
        ElMessage.warning('替换成功，但刷新失败，请手动刷新页面')
      }
    }
  } else {
    ElMessage.error('替换失败')
  }
}

// 切换全选
function handleToggleSelectAll() {
  homophoneSearch.toggleSelectAll()
}

// 切换分组折叠
function toggleGroupCollapse(clusterId) {
  if (collapsedGroups.value.has(clusterId)) {
    collapsedGroups.value.delete(clusterId)
  } else {
    collapsedGroups.value.add(clusterId)
  }
  collapsedGroups.value = new Set(collapsedGroups.value)
}

// 组内全选变化
function handleGroupSelectChange(group, checked) {
  for (const item of group.items) {
    if (checked) {
      homophoneSearch.selectedSubtitleIds.add(item.index)
    } else {
      homophoneSearch.selectedSubtitleIds.delete(item.index)
    }
  }
  homophoneSearch.selectedSubtitleIds = new Set(homophoneSearch.selectedSubtitleIds)
}

// 单条选择变化
function handleItemSelectChange(index, checked) {
  if (checked) {
    homophoneSearch.selectedSubtitleIds.add(index)
  } else {
    homophoneSearch.selectedSubtitleIds.delete(index)
  }
  homophoneSearch.selectedSubtitleIds = new Set(homophoneSearch.selectedSubtitleIds)
}

// 检查组是否全选
function isGroupAllSelected(group) {
  return group.items.every(item => homophoneSearch.selectedSubtitleIds.has(item.index))
}

// 检查组是否部分选中
function isGroupIndeterminate(group) {
  const selectedCount = group.items.filter(item => homophoneSearch.selectedSubtitleIds.has(item.index)).length
  return selectedCount > 0 && selectedCount < group.items.length
}

// 暴露给父组件（EditorView tab-nav 视图切换按钮需要）
defineExpose({
  sortMode: computed(() => homophoneSearch.sortMode),
  isSearchActive: computed(() => homophoneSearch.isSearchActive),
  toggleSortMode() {
    const current = homophoneSearch.sortMode
    homophoneSearch.sortMode = current === SortMode.GROUPED ? SortMode.TIMELINE : SortMode.GROUPED
  },
})
</script>

<style scoped>
/* 主容器 */
.subtitle-list {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--af-bg-primary);
}

/* SVG 选择器 - 按特异性从低到高排列 */
svg {
  width: 14px;
  height: 14px;
}

.empty-state svg {
  width: 48px;
  height: 48px;
  margin-bottom: 12px;
  opacity: 0.5;
}

.time-row .time-arrow svg {
  width: 14px;
  height: 14px;
}

.item-actions .action-btn svg {
  width: 14px;
  height: 14px;
}

/* 列表容器 */
.list-container {
  position: relative;
  flex: 1;
  padding: 6px;
  overflow-y: auto;
}

.list-container::-webkit-scrollbar {
  width: 6px;
}

.list-container::-webkit-scrollbar-track {
  background: transparent;
}

.list-container::-webkit-scrollbar-thumb {
  background: var(--af-border-default);
  border-radius: 3px;
}

.list-container::-webkit-scrollbar-thumb:hover {
  background: var(--af-text-muted);
}

/* 空状态 */
.empty-state {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 32px 16px;
  color: var(--af-text-muted);
}

.empty-state p {
  font-size: 13px;
  margin-bottom: 12px;
}

.empty-state .add-first-btn {
  padding: 6px 16px;
  background: var(--af-accent-primary);
  border: none;
  border-radius: var(--af-radius-md);
  color: var(--af-text-inverse);
  font-size: 13px;
  cursor: pointer;
  transition: background var(--af-transition-fast);
}

.empty-state .add-first-btn:hover {
  background: var(--af-accent-primary-hover);
}

/* 字幕项 - 紧凑布局 */
.subtitle-item {
  display: flex;
  gap: 10px;
  padding: 10px;
  margin-bottom: 6px;
  background: var(--af-bg-secondary);
  border: 1px solid transparent;
  border-radius: var(--af-radius-md);
  transition: all var(--af-transition-fast);
  cursor: pointer;
}

.subtitle-item:hover {
  background: var(--af-bg-tertiary);
}

.subtitle-item.is-active {
  border-color: var(--af-accent-primary);
  background: rgb(var(--af-accent-primary-rgb), 0.08);
}

.subtitle-item.is-current {
  border-color: var(--af-accent-success);
  background: rgb(var(--af-accent-success-rgb), 0.08);
}

/* 置信度警告高亮样式 */
.subtitle-item.warning-low-confidence {
  border-color: var(--af-accent-warning);
  background: rgb(var(--af-accent-warning-rgb), 0.06);
}

.subtitle-item.warning-high-perplexity {
  border-color: var(--af-status-warning);
  background: rgb(var(--af-status-warning-rgb), 0.06);
}

.subtitle-item.warning-both {
  border-color: var(--af-accent-danger);
  background: rgb(var(--af-accent-danger-rgb), 0.08);
  border-width: 2px;
}

/* 序号 - 缩小尺寸 */
.item-index {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 28px;
  height: 28px;
  background: var(--af-bg-tertiary);
  border-radius: var(--af-radius-sm);
  color: var(--af-text-secondary);
  font-size: 11px;
  font-weight: 600;
  flex-shrink: 0;
}

.subtitle-item.is-current .item-index {
  background: var(--af-accent-success);
  color: var(--af-text-inverse);
}

.subtitle-item.warning-low-confidence .item-index {
  background: var(--af-accent-warning);
  color: var(--af-text-inverse);
}

.subtitle-item.warning-high-perplexity .item-index {
  background: var(--af-status-warning);
  color: var(--af-text-inverse);
}

.subtitle-item.warning-both .item-index {
  background: var(--af-accent-danger);
  color: var(--af-text-inverse);
}

/* 内容区 */
.item-content {
  flex: 1;
  min-width: 0;
}

/* 时间行 - 优化间距 */
.time-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
}

.time-row .time-input {
  width: 75px;
  padding: 3px 6px;
  background: var(--af-bg-tertiary);
  border: 1px solid transparent;
  border-radius: var(--af-radius-sm);
  color: var(--af-text-normal);
  font-size: 11px;
  font-family: var(--af-font-mono);
  text-align: center;
}

.time-row .time-input:focus {
  border-color: var(--af-accent-primary);
  outline: none;
}

.time-row .time-arrow {
  color: var(--af-text-muted);
}

.time-row .duration-tag {
  padding: 2px 6px;
  background: var(--af-bg-tertiary);
  border-radius: var(--af-radius-full);
  color: var(--af-text-muted);
  font-size: 10px;
  font-family: var(--af-font-mono);
}

/* 文本行 - 优化尺寸（支持 Toggle Mode） */
.text-row {
  position: relative;
}

/* 只读高亮视图 */
.text-row .text-display {
  width: 100%;
  padding: 6px 35px 6px 8px;
  background: var(--af-bg-tertiary);
  border: 1px solid transparent;
  border-radius: var(--af-radius-sm);
  color: var(--af-text-normal);
  font-size: 12px;
  min-height: 45px;
  line-height: 1.4;
  white-space: pre-wrap;
  overflow-wrap: break-word;
  cursor: default;
}

.text-row .text-display.can-edit {
  cursor: text;
}

.text-row .text-display.can-edit:hover {
  border-color: var(--af-accent-primary);
  background: var(--af-bg-secondary);
}

/* 字级警告高亮样式 */
.text-row .text-display :deep(.word-warning) {
  background-color: rgb(var(--af-accent-warning-rgb), 0.25);
  border-bottom: 2px solid var(--af-accent-warning);
  padding: 0 2px;
  border-radius: 2px;
}

.text-row .text-display :deep(.word-critical) {
  background-color: rgb(var(--af-accent-danger-rgb), 0.25);
  border-bottom: 2px solid var(--af-accent-danger);
  padding: 0 2px;
  border-radius: 2px;
  font-weight: 500;
}

.text-row .text-input {
  width: 100%;
  padding: 6px 8px;
  background: var(--af-bg-tertiary);
  border: 1px solid transparent;
  border-radius: var(--af-radius-sm);
  color: var(--af-text-normal);
  font-size: 12px;
  padding-right: 35px;
  resize: none;
  line-height: 1.4;
}

.text-row .text-input:focus {
  border-color: var(--af-accent-primary);
  outline: none;
}

.text-row .text-input::placeholder {
  color: var(--af-text-muted);
}

.text-row .char-count {
  position: absolute;
  right: 6px;
  bottom: 6px;
  color: var(--af-text-muted);
  font-size: 10px;
  font-family: var(--af-font-mono);
}

/* 操作按钮 - 始终可见，更小尺寸 */
.item-actions {
  display: flex;
  flex-direction: column;
  gap: 2px;
  opacity: 1;
}

.subtitle-item:hover .item-actions {
  opacity: 1;
}

.item-actions .action-btn {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 24px;
  height: 24px;
  border-radius: var(--af-radius-sm);
  color: var(--af-text-muted);
  transition: all var(--af-transition-fast);
}

.item-actions .action-btn:hover {
  background: var(--af-bg-tertiary);
  color: var(--af-text-normal);
}

.item-actions .action-btn-danger:hover {
  background: rgb(var(--af-accent-danger-rgb), 0.15);
  color: var(--af-accent-danger);
}

/* 字幕切分动画 */
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
