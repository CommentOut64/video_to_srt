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
    <div class="list-host tw-flex-1 tw-min-h-0 tw-relative">
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

      <DynamicScroller
        v-else-if="isGroupedSearchMode"
        ref="listRef"
        class="list-container tw-flex-1 tw-p-1.5"
        :items="groupedVirtualItems"
        key-field="key"
        :min-item-size="48"
        :buffer="600"
      >
        <template #default="{ item, index, active }">
          <DynamicScrollerItem
            :item="item"
            :active="active"
            :size-dependencies="item.type === 'group-header'
              ? [item.clusterId, item.itemCount, item.isCollapsed]
              : [item.subtitle.text, item.subtitle.start, item.subtitle.end, item.subtitle.isDraft, item.subtitle.warning_type, item.isSelected, item.isCurrent, item.isActive, item.clusterColor]"
            :data-index="index"
          >
            <GroupHeader
              v-if="item.type === 'group-header'"
              :reading-label="item.readingLabel"
              :color="item.color"
              :item-count="item.itemCount"
              :is-collapsed="item.isCollapsed"
              :is-group-all-selected="isGroupAllSelected(item.group)"
              :is-group-indeterminate="isGroupIndeterminate(item.group)"
              @toggle-collapse="toggleGroupCollapse(item.clusterId)"
              @group-select-change="(checked) => handleGroupSelectChange(item.group, checked)"
            />
            <SubtitleItem
              v-else
              :subtitle="item.subtitle"
              :index="item.index"
              :is-active="item.isActive"
              :is-current="item.isCurrent"
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
          </DynamicScrollerItem>
        </template>
      </DynamicScroller>

      <DynamicScroller
        v-else-if="isTimelineSearchMode"
        ref="listRef"
        class="list-container tw-flex-1 tw-p-1.5"
        :items="homophoneSearch.timelineViewItems"
        key-field="index"
        :min-item-size="96"
        :buffer="800"
      >
        <template #default="{ item, index, active }">
          <DynamicScrollerItem
            :item="item"
            :active="active"
            :size-dependencies="[item.subtitle.text, item.subtitle.start, item.subtitle.end, item.subtitle.isDraft, item.subtitle.warning_type, item.isSelected, item.clusterColor]"
            :data-index="index"
          >
            <SubtitleItem
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
          </DynamicScrollerItem>
        </template>
      </DynamicScroller>

      <DynamicScroller
        v-else
        ref="listRef"
        class="list-container tw-flex-1 tw-p-1.5"
        :items="filteredSubtitles"
        key-field="id"
        :min-item-size="96"
        :buffer="800"
      >
        <template #default="{ item, index, active }">
          <DynamicScrollerItem
            :item="item"
            :active="active"
            :size-dependencies="[item.text, item.start, item.end, item.isDraft, item.warning_type, activeSubtitleId === item.id, currentSubtitleId === item.id]"
            :data-index="index"
          >
            <SubtitleItem
              :subtitle="item"
              :index="index"
              :is-active="activeSubtitleId === item.id"
              :is-current="currentSubtitleId === item.id"
              :editable="props.editable"
              @click="onSubtitleClick"
              @update-time="updateTime"
              @update-text="updateText"
              @delete="deleteSubtitle"
              @insert-before="insertBefore(index)"
              @insert-after="insertAfter(index)"
            />
          </DynamicScrollerItem>
        </template>
      </DynamicScroller>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, watch, nextTick, onMounted, onUnmounted } from 'vue'
import { DynamicScroller, DynamicScrollerItem } from 'vue-virtual-scroller'
import { ElMessage } from 'element-plus'
import { useProjectStore } from '@/stores/projectStore'
import { useSubtitleDocumentStore } from '@/stores/subtitleDocumentStore'
import { usePlaybackManager } from '@/services/PlaybackManager'
import { useHomophoneSearch, SortMode } from '@/composables'
import { useEditBufferStore } from '@/core/editor/editBufferStore'
import projectApi from '@/services/api/projectApi'
import { useStructuralSyncStore } from '@/stores/structuralSyncStore'
// 导入组件
import SubtitleItem from './SubtitleItem.vue'
import SearchToolbar from './SearchToolbar.vue'
import GroupHeader from './GroupHeader.vue'

// Props
const props = defineProps({
  autoScroll: { type: Boolean, default: true },
  editable: { type: Boolean, default: true },
  // 手动滚动后是否允许“温和自动恢复跟随”。
  // true：6 秒无滚动后自动恢复；false：用户手动滚动后保持暂停跟随。
  enableAutoResumeFollow: { type: Boolean, default: true },
})

const emit = defineEmits(['subtitle-click', 'subtitle-edit', 'subtitle-delete', 'subtitle-add'])

// Store
const projectStore = useProjectStore()
const subtitleDocumentStore = useSubtitleDocumentStore()
const structuralSyncStore = useStructuralSyncStore()
const editBufferStore = useEditBufferStore()

// 全局播放管理器
const playbackManager = usePlaybackManager()

const identityId = computed(() => projectStore.primaryId)
// 统一使用 project 主身份，兼容旧任务时可回退到 legacy job_id。
const projectId = computed(() => projectStore.primaryId)
const onSubtitleEdit = subtitleDocumentStore.onSubtitleEdit
const applyPendingEditsToStore = subtitleDocumentStore.applyPendingEditsToStore
const forceSyncNow = subtitleDocumentStore.forceSyncNow
const pendingCount = subtitleDocumentStore.pendingCount

// 同音搜索 composable（用 reactive 包裹，使模板 v-model 能正确写入 ref.value）
const homophoneSearch = reactive(useHomophoneSearch({
  projectId,
  subtitles: computed(() => projectStore.subtitles),
}))

// 分组折叠状态
const collapsedGroups = ref(new Set())

// Refs
const listRef = ref(null)
const isFollowPausedByUser = ref(false)
const USER_SCROLL_AUTO_RESUME_DELAY_MS = 6000
const PROGRAMMATIC_SCROLL_GUARD_MS = 900
let followResumeTimer = null
let programmaticScrollGuardUntil = 0
let boundListElement = null

// State
const quickSearchText = ref('')     // 快速搜索文本（简单过滤）
const animationEnabled = ref(true)  // 批量更新时禁用动画
let previousSubtitleCount = 0  // 上一次字幕数量
const hasAppliedPending = ref(false)

// Computed
const subtitles = computed(() => projectStore.subtitles)
const totalSubtitles = computed(() => projectStore.totalSubtitles)
const currentSubtitleId = computed(() => projectStore.currentSubtitle?.id)
const activeSubtitleId = computed(() => subtitleDocumentStore.selectedSubtitleId)
// 草稿计数
const draftCount = computed(() => projectStore.draftSubtitleCount)

const isGroupedSearchMode = computed(
  () => homophoneSearch.isSearchActive && homophoneSearch.sortMode === SortMode.GROUPED
)
const isTimelineSearchMode = computed(
  () => homophoneSearch.isSearchActive && homophoneSearch.sortMode === SortMode.TIMELINE
)
const groupedVirtualItems = computed(() => {
  if (!isGroupedSearchMode.value) return []

  return homophoneSearch.groupedViewItems.flatMap((group) => {
    const items = [{
      type: 'group-header',
      key: `group-${group.clusterId}`,
      clusterId: group.clusterId,
      readingLabel: group.readingLabel,
      color: group.color,
      itemCount: group.items.length,
      isCollapsed: collapsedGroups.value.has(group.clusterId),
      group,
    }]

    if (collapsedGroups.value.has(group.clusterId)) {
      return items
    }

    group.items.forEach((item) => {
      items.push({
        type: 'subtitle',
        key: `subtitle-${item.subtitle.id}`,
        subtitle: item.subtitle,
        index: item.index,
        matchSpans: item.matchSpans,
        isSelected: item.isSelected,
        clusterColor: item.clusterColor,
        isActive: activeSubtitleId.value === item.subtitle.id,
        isCurrent: currentSubtitleId.value === item.subtitle.id,
      }, { isUserEdit: false })
    })

    return items
  })
})

const filteredSubtitles = computed(() => {
  if (!quickSearchText.value) return subtitles.value
  const search = quickSearchText.value.toLowerCase()
  return subtitles.value.filter(sub => sub.text.toLowerCase().includes(search))
})

function getListScrollElement() {
  return listRef.value?.$el ?? listRef.value ?? null
}

function bindListInteractionListeners() {
  const nextElement = getListScrollElement()
  if (boundListElement === nextElement) {
    return
  }

  if (boundListElement) {
    boundListElement.removeEventListener('scroll', handleListScroll)
    boundListElement.removeEventListener('wheel', handleListWheel)
    boundListElement.removeEventListener('touchmove', handleListTouchMove)
  }

  boundListElement = nextElement
  if (!boundListElement) {
    return
  }

  boundListElement.addEventListener('scroll', handleListScroll, { passive: true })
  boundListElement.addEventListener('wheel', handleListWheel, { passive: true })
  boundListElement.addEventListener('touchmove', handleListTouchMove, { passive: true })
}

function unbindListInteractionListeners() {
  if (!boundListElement) {
    return
  }

  boundListElement.removeEventListener('scroll', handleListScroll)
  boundListElement.removeEventListener('wheel', handleListWheel)
  boundListElement.removeEventListener('touchmove', handleListTouchMove)
  boundListElement = null
}

function findSubtitleVirtualIndex(subtitleId) {
  if (!subtitleId) return -1

  if (isGroupedSearchMode.value) {
    return groupedVirtualItems.value.findIndex(
      (item) => item.type === 'subtitle' && item.subtitle.id === subtitleId
    )
  }

  if (isTimelineSearchMode.value) {
    return homophoneSearch.timelineViewItems.findIndex(
      (item) => item.subtitle.id === subtitleId
    )
  }

  return filteredSubtitles.value.findIndex((subtitle) => subtitle.id === subtitleId)
}

function scrollToSubtitleId(subtitleId) {
  const index = findSubtitleVirtualIndex(subtitleId)
  if (index !== -1) {
    scrollToItem(index)
  }
}

watch(
  () => identityId.value,
  async (identity) => {
    await subtitleDocumentStore.bindSyncIdentity(identity)
    hasAppliedPending.value = false
    if (pendingCount() > 0) {
      await forceSyncNow()
    }
  },
  { immediate: true }
)

watch(
  () => subtitles.value.length,
  async (length) => {
    if (!identityId.value || length === 0 || hasAppliedPending.value) return
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
  subtitleDocumentStore.setSelectedSubtitleId(subtitle.id)
  // 使用 PlaybackManager 进行跳转，确保视频和波形同步
  playbackManager.seekTo(subtitle.start)
  emit('subtitle-click', subtitle)
}

function updateTime(id, field, value) {
  if (isNaN(value)) return

  // 1. 乐观更新本地状态
  projectStore.updateSubtitle(id, { [field]: value }, { isUserEdit: true })
  const subtitle = projectStore.subtitles.find(s => s.id === id)
  if (subtitle) {
    editBufferStore.syncCommittedTime(id, {
      start: subtitle.start,
      end: subtitle.end,
    })
  }
  emit('subtitle-edit', id, field, value)

  // 2. V3.2.0+dev.20260124.01: 同步到后端（防抖）
  const syncKey = subtitle?.segment_id ?? subtitle?.sentenceIndex ?? subtitle?.id
  if (syncKey !== undefined && syncKey !== null) {
    onSubtitleEdit(syncKey, {
      [field]: value
    })
  }
}

function updateText(id, text) {
  // 1. 乐观更新本地状态
  projectStore.updateSubtitle(id, { text }, { isUserEdit: true })
  editBufferStore.syncCommittedText(id, text)
  emit('subtitle-edit', id, 'text', text)

  // 2. V3.2.0+dev.20260124.01: 同步到后端（防抖）
  const subtitle = projectStore.subtitles.find(s => s.id === id)
  const syncKey = subtitle?.segment_id ?? subtitle?.sentenceIndex ?? subtitle?.id
  if (syncKey !== undefined && syncKey !== null) {
    onSubtitleEdit(syncKey, {
      text
    })
  }
}

async function deleteSubtitle(id) {
  const subtitle = projectStore.subtitles.find(s => s.id === id)
  projectStore.removeSubtitle(id, { isUserEdit: true })
  emit('subtitle-delete', id)

  if (!subtitle) {
    return
  }

  // V3.2.4+dev.20260304.01: 结构性操作追踪
  const syncPromise = (async () => {
    if (!projectStore.meta.projectId) {
      throw new Error('缺少 project_id，禁止走 job 字幕删除分支')
    }
    if (!subtitle.segment_id) {
      throw new Error(`缺少 segment_id，无法删除字幕（id=${id}）`)
    }
    await projectApi.deleteSubtitle(projectStore.meta.projectId, subtitle.segment_id)
  })()
  structuralSyncStore.trackOperation('delete', syncPromise)

  try {
    await syncPromise
  } catch (error) {
    console.warn('[SubtitleList] 删除字幕同步失败:', error)
  }
}

async function addNewSubtitle() {
  const lastSubtitle = subtitles.value[subtitles.value.length - 1]
  const newStart = lastSubtitle ? lastSubtitle.end : 0
  const insertIndex = subtitles.value.length
  const newSubtitle = projectStore.addSubtitle(insertIndex, {
    start: newStart,
    end: newStart + 3,
    text: '',
    isModified: true,
    source: 'manual'
  }, { isUserEdit: true })
  const localSubtitleId = newSubtitle?.id
  nextTick(() => {
    scrollToBottom()
  })
  emit('subtitle-add', subtitles.value.length - 1)

  // V3.2.4+dev.20260304.01: 结构性操作追踪
  const syncPromise = (async () => {
    const baseStart = projectStore.toBaseTime(newStart)
    const baseEnd = projectStore.toBaseTime(newStart + 3)
    if (!projectStore.meta.projectId) {
      throw new Error('缺少 project_id，禁止走 job 字幕新增分支')
    }
    const data = await projectApi.createSubtitle(projectStore.meta.projectId, {
      text: '',
      start: baseStart,
      end: baseEnd
    })
    const newSubtitle = projectStore.subtitles.find((item) => item.id === localSubtitleId)
    if (newSubtitle) {
      projectStore.updateSubtitle(newSubtitle.id, {
        sentenceIndex: data?.index ?? data?.legacy_index ?? newSubtitle.sentenceIndex,
        segment_id: data?.segment_id ?? newSubtitle.segment_id,
        isModified: true,
        source: data?.source || data?.source_type || 'manual'
      }, { isUserEdit: false })
    }
  })()
  const opId = structuralSyncStore.trackOperation('insert', syncPromise)

  try {
    await syncPromise
  } catch (error) {
    const rollbackTarget = projectStore.subtitles.find((item) => item.id === localSubtitleId)
    if (rollbackTarget) {
      projectStore.removeSubtitle(rollbackTarget.id, { isUserEdit: true })
    }
    // 回滚后本地与后端一致，清除错误
    structuralSyncStore.clearError(opId)
    console.warn('[SubtitleList] 新增字幕同步失败:', error)
  }
}

function insertBefore(index) {
  const current = subtitles.value[index]
  const prev = subtitles.value[index - 1]
  const start = prev ? prev.end : Math.max(0, current.start - 3)
  const end = current.start
  const newSubtitle = projectStore.addSubtitle(index, { start, end, text: '', isModified: true, source: 'manual' }, { isUserEdit: true })
  const localSubtitleId = newSubtitle?.id
  syncInsertedSubtitle(localSubtitleId, start, end, '')
}

function insertAfter(index) {
  const current = subtitles.value[index]
  const next = subtitles.value[index + 1]
  const start = current.end
  const end = next ? next.start : current.end + 3
  const newSubtitle = projectStore.addSubtitle(index + 1, { start, end, text: '', isModified: true, source: 'manual' }, { isUserEdit: true })
  const localSubtitleId = newSubtitle?.id
  syncInsertedSubtitle(localSubtitleId, start, end, '')
}

async function syncInsertedSubtitle(localSubtitleId, start, end, text) {
  // V3.2.4+dev.20260304.01: 结构性操作追踪
  const syncPromise = (async () => {
    const baseStart = projectStore.toBaseTime(start)
    const baseEnd = projectStore.toBaseTime(end)
    if (!projectStore.meta.projectId) {
      throw new Error('缺少 project_id，禁止走 job 字幕新增分支')
    }
    const data = await projectApi.createSubtitle(projectStore.meta.projectId, {
      text,
      start: baseStart,
      end: baseEnd
    })
    const newSubtitle = projectStore.subtitles.find((item) => item.id === localSubtitleId)
    if (newSubtitle) {
      projectStore.updateSubtitle(newSubtitle.id, {
        sentenceIndex: data?.index ?? data?.legacy_index ?? newSubtitle.sentenceIndex,
        segment_id: data?.segment_id ?? newSubtitle.segment_id,
        isModified: true,
        source: data?.source || data?.source_type || 'manual'
      }, { isUserEdit: false })
    }
  })()
  const opId = structuralSyncStore.trackOperation('insert', syncPromise)

  try {
    await syncPromise
  } catch (error) {
    const rollbackTarget = projectStore.subtitles.find((item) => item.id === localSubtitleId)
    if (rollbackTarget) {
      projectStore.removeSubtitle(rollbackTarget.id, { isUserEdit: true })
    }
    // 回滚后本地与后端一致，清除错误
    structuralSyncStore.clearError(opId)
    console.warn('[SubtitleList] 新增字幕同步失败:', error)
  }
}

function scrollToBottom() {
  const element = getListScrollElement()
  if (!element) return

  markProgrammaticScroll()
  element.scrollTop = element.scrollHeight
}

function scrollToItem(index) {
  if (!Number.isInteger(index) || index < 0) return

  if (typeof listRef.value?.scrollToItem === 'function') {
    markProgrammaticScroll()
    listRef.value.scrollToItem(index)
    return
  }

  const items = getListScrollElement()?.querySelectorAll('.subtitle-item')
  if (items && items[index]) {
    markProgrammaticScroll()
    items[index].scrollIntoView({ behavior: 'smooth', block: 'center' })
  }
}

function clearFollowResumeTimer() {
  if (followResumeTimer) {
    clearTimeout(followResumeTimer)
    followResumeTimer = null
  }
}

function markProgrammaticScroll() {
  programmaticScrollGuardUntil = Date.now() + PROGRAMMATIC_SCROLL_GUARD_MS
}

function isInProgrammaticScrollGuard() {
  return Date.now() < programmaticScrollGuardUntil
}

function scheduleFollowAutoResume() {
  clearFollowResumeTimer()
  if (!props.enableAutoResumeFollow) return
  followResumeTimer = setTimeout(() => {
    isFollowPausedByUser.value = false
    followResumeTimer = null
  }, USER_SCROLL_AUTO_RESUME_DELAY_MS)
}

function pauseFollowByUserScroll() {
  if (!props.autoScroll) return
  if (isInProgrammaticScrollGuard()) return
  isFollowPausedByUser.value = true
  scheduleFollowAutoResume()
}

function handleListScroll() {
  pauseFollowByUserScroll()
}

function handleListWheel() {
  pauseFollowByUserScroll()
}

function handleListTouchMove() {
  pauseFollowByUserScroll()
}

// 自动滚动跟随当前播放
watch(currentSubtitleId, (id) => {
  if (!props.autoScroll || !id) return
  if (isFollowPausedByUser.value) return
  nextTick(() => scrollToSubtitleId(id))
})

watch(
  () => props.enableAutoResumeFollow,
  (enabled) => {
    // 用户重新启用自动恢复时，如果当前已暂停，则从当前时刻重新计时恢复。
    if (enabled && isFollowPausedByUser.value) {
      scheduleFollowAutoResume()
      return
    }
    if (!enabled) {
      clearFollowResumeTimer()
    }
  }
)

watch(
  () => props.autoScroll,
  (enabled) => {
    if (!enabled) {
      clearFollowResumeTimer()
      isFollowPausedByUser.value = false
    }
  }
)

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

watch(
  () => `${isGroupedSearchMode.value}:${isTimelineSearchMode.value}:${filteredSubtitles.value.length}:${groupedVirtualItems.value.length}:${homophoneSearch.matchCount}`,
  () => {
    nextTick(() => bindListInteractionListeners())
  },
  { flush: 'post' }
)

onMounted(() => {
  nextTick(() => bindListInteractionListeners())
})

onUnmounted(() => {
  clearFollowResumeTimer()
  unbindListInteractionListeners()
})

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
    if (projectStore.meta.projectId) {
      try {
        await forceSyncNow()
        let segments = []
        const projectSegments = await projectApi.getSubtitles(projectStore.meta.projectId)
        if (Array.isArray(projectSegments)) {
          segments = projectSegments.map((segment, index) => ({
            id: segment.legacy_index ?? segment.sentence_index ?? index,
            start: segment.start,
            end: segment.end,
            text: segment.text,
            is_modified: segment.is_modified,
            original_text: segment.original_text,
          }))
        }
        if (Array.isArray(segments) && segments.length > 0) {
          const segmentMap = new Map(
            segments.map((segment) => [Number(segment.id), segment])
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
            projectStore.importSegments(segments, {
              jobId: projectId.value,
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
    } else {
      ElMessage.error('缺少 project_id，无法执行批量替换回写；请从任务列表重新进入项目编辑器')
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
