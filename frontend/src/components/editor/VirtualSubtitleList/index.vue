<!-- V3.2.5+dev.20260316.02: VirtualSubtitleList - 接入搜索/分组/批量替换 -->
<template>
  <div class="virtual-subtitle-list">
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
      :multi-selected-count="selectionStore.multiSelectedCount"
      @search="handleHomophoneSearch"
      @reset="handleHomophoneReset"
      @batch-replace="handleBatchReplace"
      @batch-delete="handleBatchDelete"
      @toggle-select-all="handleToggleSelectAll"
      @add-subtitle="handleAddSubtitle"
      @quick-search="handleQuickSearch"
    />

    <div v-if="showNoSubtitles" class="empty-state">
      <svg class="empty-icon" viewBox="0 0 24 24" fill="currentColor">
        <path d="M20 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 14H4V6h16v12zM6 10h2v2H6v-2zm0 4h8v2H6v-2zm10 0h2v2h-2v-2zm0-4h2v2h-2v-2zm-4 4h2v2h-2v-2zm0-4h2v2h-2v-2z"/>
      </svg>
      <p>暂无字幕</p>
      <button class="empty-action-btn" @click="handleAddSubtitle">添加第一条字幕</button>
    </div>

    <div v-else-if="showNoMatches" class="empty-state">
      <p>未找到匹配结果</p>
    </div>

    <div
      v-else-if="isGroupedMode"
      ref="listContainerRef"
      class="scroller grouped-container"
      tabindex="0"
      @keydown="handleKeydown"
    >
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
        <div v-if="!collapsedGroups.has(group.clusterId)" class="group-items">
          <div
            v-for="item in group.items"
            :key="item.subtitle.id"
            class="row-shell"
            :data-local-id="item.subtitle.localId"
          >
            <SubtitleRow
              :local-id="item.subtitle.localId"
              :is-selected="selectionStore.isMultiSelected(item.subtitle.localId)"
              :is-active="activeId === item.subtitle.localId"
              :is-current="currentSubtitleId === item.subtitle.localId"
              :match-spans="item.matchSpans"
              :is-match-selected="item.isSelected"
              :is-selectable="true"
              :cluster-color="item.clusterColor"
              @click="handleRowClick"
              @insert-before="handleInsertBefore(item.subtitle.localId)"
              @insert-after="handleInsertAfter(item.subtitle.localId)"
              @delete="handleDelete(item.subtitle.localId)"
              @focus-row="handleRowFocus"
              @open-context-menu="handleOpenContextMenu"
              @select-change="(checked) => handleItemSelectChange(item.index, checked)"
            />
          </div>
        </div>
      </template>
    </div>

    <DynamicScroller
      v-else
      ref="scrollerRef"
      :items="virtualVisibleRows"
      :min-item-size="98"
      class="scroller"
      tabindex="0"
      @keydown="handleKeydown"
    >
      <template #default="{ item, index, active }">
        <DynamicScrollerItem
          :item="item"
          :index="index"
          :active="active"
          :size-dependencies="[item.sizeKey]"
        >
          <div class="row-shell" :data-local-id="item.localId">
            <SubtitleRow
              :local-id="item.localId"
              :is-selected="selectionStore.isMultiSelected(item.localId)"
              :is-active="activeId === item.localId"
              :is-current="currentSubtitleId === item.localId"
              :match-spans="item.matchSpans"
              :is-match-selected="item.isMatchSelected"
              :is-selectable="item.isSelectable"
              :cluster-color="item.clusterColor"
              @click="handleRowClick"
              @insert-before="handleInsertBefore(item.localId)"
              @insert-after="handleInsertAfter(item.localId)"
              @delete="handleDelete(item.localId)"
              @focus-row="handleRowFocus"
              @open-context-menu="handleOpenContextMenu"
              @select-change="(checked) => handleItemSelectChange(item.selectionIndex, checked)"
            />
          </div>
        </DynamicScrollerItem>
      </template>
    </DynamicScroller>

    <ContextMenu
      ref="sharedContextMenuRef"
      :items="sharedContextMenuItems"
      @close="handleSharedContextMenuClose"
      @select="handleSharedContextMenuSelect"
    />
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import { ElMessage, ElLoading } from 'element-plus'
import { DynamicScroller, DynamicScrollerItem } from 'vue-virtual-scroller'
import 'vue-virtual-scroller/dist/vue-virtual-scroller.css'
import ContextMenu from '@/components/editor/ContextMenu.vue'
import GroupHeader from '@/components/editor/SubtitleList/GroupHeader.vue'
import SearchToolbar from '@/components/editor/SubtitleList/SearchToolbar.vue'
import { SortMode, useHomophoneSearch } from '@/composables/useHomophoneSearch'
import SubtitleRow from './SubtitleRow.vue'
import { createRowModelCache } from './rowModelCache'
import { useSharedContextMenu } from './useSharedContextMenu'
import { buildBatchReplaceReplacements } from './searchBatchReplace'
import { shouldIgnoreStructuralShortcut } from './keyboardSafety'
import projectApi from '@/services/api/projectApi'
import { useEditorPlayback } from '@/composables/editor/useEditorPlayback'
import { useProjectStore } from '@/stores/projectStore'
import { useSubtitleDocumentStore } from '@/stores/subtitleDocumentStore'
import { useEditorCommandBus } from '@/stores/editor/editorCommandBus'
import {
  createBatchReplaceCommand,
  createDeleteSubtitleCommand,
  createInsertSubtitleCommand,
  createMergeSubtitlesCommand,
  createSplitSubtitleCommand,
} from '@/stores/editor/editorCommandFactory'
import { useEditorDocumentStore } from '@/stores/editor/editorDocumentStore'
import { useEditorProjectionBridge } from '@/stores/editor/editorProjectionBridge'
import { useEditorSelectionStore } from '@/stores/editor/editorSelectionStore'
import { useEditorSessionStore } from '@/stores/editor/editorSessionStore'
import { useEditorSyncEngine } from '@/stores/editor/editorSyncEngine'
import {
  getMergeSeparator,
  normalizeProjectionWordsToCold,
  splitSubtitleByCursor,
} from './structuralEdit'
import { resolveMergeSideSnapshot } from './mergeSnapshot'
import {
  buildVisibleFlipPlan,
  captureVisibleSnapshot,
  playVisibleFlip,
  shouldSkipVisibleFlip,
  waitForStableLayout,
} from './visibleWindowFlip'
import { focusRowForEditing, resolveDeleteFallbackId } from './rowFocusController'

const props = defineProps({
  autoScroll: { type: Boolean, default: true },
  editable: { type: Boolean, default: true },
  enableAutoResumeFollow: { type: Boolean, default: true },
  isResizing: { type: Boolean, default: false },
})

const docStore = useEditorDocumentStore()
const commandBus = useEditorCommandBus()
const editorProjectionBridge = useEditorProjectionBridge()
const selectionStore = useEditorSelectionStore()
const projectStore = useProjectStore()
const sessionStore = useEditorSessionStore()
const subtitleDocumentStore = useSubtitleDocumentStore()
const syncEngine = useEditorSyncEngine()
const { currentSubtitleId, seekToSubtitle, followCurrentSubtitle } = useEditorPlayback()

const identityId = computed(() => projectStore.primaryId)
const projectionSubtitles = computed(() => editorProjectionBridge.subtitles)
const totalSubtitles = computed(() => editorProjectionBridge.totalSubtitles)
const draftCount = computed(() => editorProjectionBridge.draftCount)

const homophoneSearch = reactive(useHomophoneSearch({
  projectId: identityId,
  subtitles: projectionSubtitles,
}))

const scrollerRef = ref(null)
const sharedContextMenuRef = ref(null)
const listContainerRef = ref(null)
const quickSearchText = ref('')
const collapsedGroups = ref(new Set())
const selectedLocalId = computed(() => subtitleDocumentStore.selectedSubtitleId)
const activeId = computed(() => selectionStore.activeLocalId)
const autoScrollEnabled = computed(() => props.autoScroll && projectStore.view.autoScroll !== false)
const isGroupedMode = computed(() => {
  return homophoneSearch.isSearchActive && homophoneSearch.sortMode === SortMode.GROUPED
})

const isFollowPausedByUser = ref(false)
const USER_SCROLL_AUTO_RESUME_DELAY_MS = 6000
const PROGRAMMATIC_SCROLL_GUARD_MS = 900
let followResumeTimer = null
let programmaticScrollGuardUntil = 0
let programmaticScrollReleaseTimer = null
let attachedScrollerElement = null
let followRequestToken = 0

// FLIP 动画运行时常量与引用
const FLIP_DURATION_MS = 160
const FLIP_MAX_ITEMS = 12
const FLIP_ENTER_OFFSET_PX = 8
const USER_SCROLL_FLIP_GUARD_MS = 120
const lastManualListInteractionAt = ref(0)
let cancelActiveFlipAnimation = null
let flipRunToken = 0

const {
  handleMenuClose: finalizeSharedContextMenuClose,
  isOpen: isSharedContextMenuOpen,
  items: sharedContextMenuItems,
  openMenu: openSharedContextMenu,
  selectMenuItem: selectSharedContextMenuItem,
} = useSharedContextMenu({
  onSelect: handleSharedContextMenuSelection,
})
const rowModelCache = createRowModelCache()

const filteredSubtitles = computed(() => {
  if (!quickSearchText.value) {
    return projectionSubtitles.value
  }

  const normalizedQuery = quickSearchText.value.trim().toLowerCase()
  if (!normalizedQuery) {
    return projectionSubtitles.value
  }

  return projectionSubtitles.value.filter((subtitle) => {
    return String(subtitle?.text || '').toLowerCase().includes(normalizedQuery)
  })
})

const groupedVisibleIds = computed(() => {
  return homophoneSearch.groupedViewItems.flatMap((group) => {
    if (collapsedGroups.value.has(group.clusterId)) {
      return []
    }
    return group.items.map((item) => item.subtitle.localId)
  })
})

const timelineRowMetaByLocalId = computed(() => {
  return new Map(
    homophoneSearch.timelineViewItems.map((item) => [
      item.subtitle.localId,
      {
        rowIndex: item.index,
        matchSpans: item.matchSpans,
        isMatchSelected: item.isSelected,
        isSelectable: true,
        clusterColor: item.clusterColor,
        selectionIndex: item.index,
      },
    ])
  )
})

const rowMeasurementKeyByLocalId = computed(() => {
  const measurementMap = new Map()
  projectionSubtitles.value.forEach((subtitle) => {
    measurementMap.set(localIdKey(subtitle), [
      subtitle.text ?? '',
      subtitle.warning_type ?? 'none',
      subtitle.isDraft ? 'draft' : 'final',
      Array.isArray(subtitle.words) ? subtitle.words.length : 0,
      subtitle.startMs ?? 0,
      subtitle.endMs ?? 0,
    ].join('|'))
  })
  return measurementMap
})

const defaultVisibleRowEntries = computed(() => {
  return filteredSubtitles.value.map((subtitle, index) => buildVirtualRowEntry(subtitle.localId, index))
})

const defaultVisibleRows = computed(() => {
  return rowModelCache.rebuildVisibleRows(defaultVisibleRowEntries.value)
})

const defaultVisibleIds = computed(() => {
  return defaultVisibleRows.value.map((row) => row.localId)
})

const timelineVisibleRowEntries = computed(() => {
  return homophoneSearch.timelineViewItems.map((item) => buildVirtualRowEntry(item.subtitle.localId, item.index))
})

const timelineVisibleRows = computed(() => {
  return rowModelCache.rebuildVisibleRows(timelineVisibleRowEntries.value)
})

const virtualVisibleRows = computed(() => {
  if (homophoneSearch.isSearchActive && homophoneSearch.sortMode === SortMode.TIMELINE) {
    return timelineVisibleRows.value
  }
  return defaultVisibleRows.value
})

const orderedVisibleIds = computed(() => {
  return isGroupedMode.value
    ? groupedVisibleIds.value
    : virtualVisibleRows.value.map((row) => row.localId)
})

const showNoSubtitles = computed(() => {
  return !homophoneSearch.isSearchActive && defaultVisibleRows.value.length === 0
})

const showNoMatches = computed(() => {
  return homophoneSearch.isSearchActive && homophoneSearch.matchCount === 0
})

function resolveRowMeta(localId, fallbackIndex) {
  const searchMeta = timelineRowMetaByLocalId.value.get(localId)
  if (searchMeta) {
    return searchMeta
  }

  return {
    rowIndex: fallbackIndex,
    matchSpans: [],
    isMatchSelected: false,
    isSelectable: false,
    clusterColor: null,
    selectionIndex: null,
  }
}

function localIdKey(subtitle) {
  return subtitle?.localId ?? subtitle?.id ?? ''
}

function buildMatchSpanSignature(matchSpans) {
  if (!Array.isArray(matchSpans) || matchSpans.length === 0) {
    return 'no-match'
  }

  return matchSpans
    .map((span) => `${span.start ?? 0}:${span.end ?? 0}`)
    .join(',')
}

function buildVirtualRowEntry(localId, fallbackIndex) {
  const rowMeta = resolveRowMeta(localId, fallbackIndex)
  const matchSpans = Array.isArray(rowMeta.matchSpans) ? rowMeta.matchSpans : []
  const payload = {
    id: localId,
    localId,
    matchSpans,
    isMatchSelected: Boolean(rowMeta.isMatchSelected),
    isSelectable: Boolean(rowMeta.isSelectable),
    clusterColor: rowMeta.clusterColor ?? null,
    selectionIndex: Number.isFinite(rowMeta.selectionIndex) ? rowMeta.selectionIndex : null,
    sizeKey: [
      rowMeasurementKeyByLocalId.value.get(localId) || '',
      buildMatchSpanSignature(matchSpans),
      rowMeta.isSelectable ? 'search' : 'normal',
    ].join('|'),
  }

  return {
    localId,
    payload,
    signature: [
      payload.sizeKey,
      payload.isMatchSelected ? 'selected' : 'not-selected',
      payload.clusterColor ?? 'no-cluster',
      payload.selectionIndex ?? 'no-selection-index',
    ].join('|'),
  }
}

function handleRowClick(localId, event) {
  const appendRange = Boolean(event?.ctrlKey || event?.metaKey)
  if (event?.shiftKey) {
    selectionStore.selectRange(localId, orderedVisibleIds.value, { append: appendRange })
  } else if (appendRange) {
    selectionStore.toggleMultiSelection(localId, {
      makeActive: true,
      updateAnchor: true,
    })
  } else {
    selectionStore.selectOnly(localId)
  }

  subtitleDocumentStore.setSelectedSubtitleId(localId)
  seekToSubtitle(localId)
}

function handleRowFocus(localId) {
  focusRowForEditing(localId, {
    selectionStore,
    subtitleDocumentStore,
  })
}

function handleOpenContextMenu(payload) {
  if (!payload || !Array.isArray(payload.items) || payload.items.length === 0) {
    return
  }

  sharedContextMenuRef.value?.hide()
  openSharedContextMenu(payload)

  nextTick(() => {
    if (!isSharedContextMenuOpen.value) {
      return
    }
    sharedContextMenuRef.value?.show(payload.x, payload.y)
  })
}

function handleSharedContextMenuClose() {
  finalizeSharedContextMenuClose()
}

function handleSharedContextMenuSelect(key) {
  selectSharedContextMenuItem(key)
}

function handleSharedContextMenuSelection({ key, context }) {
  if (!context?.localId) {
    return
  }

  if (key === 'split') {
    if (!context.canSplitAtCursor) {
      return
    }
    handleSplitFromCursor({
      localId: context.localId,
      cursorPosition: context.cursorPosition,
      text: context.text,
    })
    return
  }

  if (key === 'merge-prev') {
    handleMerge(context.localId, 'prev', context)
    return
  }

  if (key === 'merge-next') {
    handleMerge(context.localId, 'next', context)
  }
}

function handleInsertBefore(localId) {
  runStructuralMutationWithFlip(() => {
    const currentIndex = defaultVisibleIds.value.indexOf(localId)
    const previousLocalId = currentIndex > 0 ? defaultVisibleIds.value[currentIndex - 1] : null
    const currentEntity = docStore.getEntity(localId)
    const previousEntity = previousLocalId ? docStore.getEntity(previousLocalId) : null
    const startMs = previousEntity?.endMs ?? Math.max(0, (currentEntity?.startMs ?? 3000) - 3000)
    const candidateEndMs = currentEntity?.startMs ?? (startMs + 3000)
    const endMs = candidateEndMs > startMs ? candidateEndMs : (startMs + 500)

    return commandBus.dispatch(createInsertSubtitleCommand({
      afterLocalId: previousLocalId,
      entity: {
        text: '',
        startMs,
        endMs,
        isDraft: false,
      },
      source: 'user',
    }))
  })
}

function handleInsertAfter(localId) {
  runStructuralMutationWithFlip(() => {
    const currentEntity = docStore.getEntity(localId)
    const nextLocalId = docStore.getNeighbors(localId).next
    const nextEntity = nextLocalId ? docStore.getEntity(nextLocalId) : null
    const startMs = currentEntity?.endMs ?? 0
    const candidateEndMs = nextEntity?.startMs ?? (startMs + 3000)
    const endMs = candidateEndMs > startMs ? candidateEndMs : (startMs + 500)

    return commandBus.dispatch(createInsertSubtitleCommand({
      afterLocalId: localId,
      entity: {
        text: '',
        startMs,
        endMs,
        isDraft: false,
      },
      source: 'user',
    }))
  })
}

function handleDelete(localId) {
  runStructuralMutationWithFlip(() => {
    const result = commandBus.dispatch(createDeleteSubtitleCommand({
      localId,
      source: 'user',
    }))

    if (!result?.success) {
      return result
    }

    if (selectionStore.isMultiSelected(localId)) {
      selectionStore.toggleMultiSelection(localId, {
        makeActive: false,
        updateAnchor: false,
      })
    }

    const fallbackId = resolveDeleteFallbackId(localId, orderedVisibleIds.value)

    if (selectedLocalId.value === localId) {
      subtitleDocumentStore.setSelectedSubtitleId(fallbackId)
    }

    if (activeId.value === localId) {
      selectionStore.setActive(fallbackId, { updateAnchor: false })
    }

    return result
  })
}

function handleBatchDelete() {
  const selectedIds = Array.from(selectionStore.multiSelectedLocalIds)
    .filter((localId) => {
      const entity = docStore.getEntity(localId)
      return entity && !entity.isDraft
    })
    .sort((leftId, rightId) => {
      return docStore.getOrderIndex(rightId) - docStore.getOrderIndex(leftId)
    })

  if (selectedIds.length === 0) {
    ElMessage.warning('请先选择要删除的字幕')
    return
  }

  const minIndex = Math.min(...selectedIds.map((localId) => docStore.getOrderIndex(localId)))

  runStructuralMutationWithFlip(() => {
    const commands = selectedIds.map((localId) => createDeleteSubtitleCommand({
      localId,
      source: 'user',
    }))
    const result = commandBus.dispatchTransaction(commands, {
      historyType: 'batch_delete',
    })
    if (!result?.success) {
      ElMessage.error('批量删除失败')
      return result
    }

    selectionStore.clearMultiSelection()
    const fallbackId = docStore.order[minIndex] ?? docStore.order[minIndex - 1] ?? null
    selectionStore.setActive(fallbackId, { updateAnchor: false })
    subtitleDocumentStore.setSelectedSubtitleId(fallbackId)
    return result
  })
}

function handleKeydown(event) {
  if (shouldIgnoreStructuralShortcut(event)) {
    return
  }

  if (!activeId.value) {
    return
  }

  const currentIndex = orderedVisibleIds.value.indexOf(activeId.value)

  switch (event.key) {
    case 'ArrowUp': {
      event.preventDefault()
      if (currentIndex > 0) {
        const nextId = orderedVisibleIds.value[currentIndex - 1]
        if (event.shiftKey) {
          selectionStore.selectRange(nextId, orderedVisibleIds.value)
        } else {
          selectionStore.selectOnly(nextId)
        }
        subtitleDocumentStore.setSelectedSubtitleId(nextId)
      }
      break
    }
    case 'ArrowDown': {
      event.preventDefault()
      if (currentIndex < orderedVisibleIds.value.length - 1) {
        const nextId = orderedVisibleIds.value[currentIndex + 1]
        if (event.shiftKey) {
          selectionStore.selectRange(nextId, orderedVisibleIds.value)
        } else {
          selectionStore.selectOnly(nextId)
        }
        subtitleDocumentStore.setSelectedSubtitleId(nextId)
      }
      break
    }
    case 'Delete':
      event.preventDefault()
      if (selectionStore.multiSelectedCount > 0) {
        handleBatchDelete()
      } else {
        handleDelete(activeId.value)
      }
      break
    case 'Enter':
      event.preventDefault()
      handleInsertAfter(activeId.value)
      break
    case 'a':
    case 'A':
      if (event.ctrlKey || event.metaKey) {
        event.preventDefault()
        selectionStore.replaceMultiSelection(orderedVisibleIds.value, {
          activeLocalId: activeId.value || orderedVisibleIds.value[0] || null,
          updateAnchor: true,
        })
      }
      break
    default:
      break
  }
}

function handleAddSubtitle() {
  const afterLocalId = defaultVisibleIds.value[defaultVisibleIds.value.length - 1] ?? null
  const anchorEntity = afterLocalId ? docStore.getEntity(afterLocalId) : null
  const startMs = anchorEntity?.endMs ?? 0

  commandBus.dispatch(createInsertSubtitleCommand({
    afterLocalId,
    entity: {
      text: '',
      startMs,
      endMs: startMs + 3000,
      isDraft: false,
    },
    source: 'user',
  }))
}

function handleQuickSearch(text) {
  quickSearchText.value = text
}

async function handleHomophoneSearch() {
  // 近音模式需要后端检索，耗时较长，显示全局加载提示
  const needsLoading = homophoneSearch.isHomophoneMode
  const loading = needsLoading
    ? ElLoading.service({ text: '搜索中...', background: 'rgba(0, 0, 0, 0.4)' })
    : null
  try {
    await homophoneSearch.executeSearch()
    selectionStore.setSearchSelection(Array.from(homophoneSearch.selectedSubtitleIds))
  } finally {
    loading?.close()
  }
}

function handleHomophoneReset() {
  homophoneSearch.resetSearch()
  selectionStore.clearSearchSelection()
  quickSearchText.value = ''
}

function handleToggleSelectAll() {
  homophoneSearch.toggleSelectAll()
  selectionStore.setSearchSelection(Array.from(homophoneSearch.selectedSubtitleIds))
}

function handleItemSelectChange(sentenceIndex, checked) {
  if (sentenceIndex === null || sentenceIndex === undefined) {
    return
  }

  selectionStore.toggleSearchSelection(sentenceIndex, checked)
  if (checked) homophoneSearch.selectedSubtitleIds.add(sentenceIndex)
  else homophoneSearch.selectedSubtitleIds.delete(sentenceIndex)
  homophoneSearch.selectedSubtitleIds = new Set(homophoneSearch.selectedSubtitleIds)
}

function toggleGroupCollapse(clusterId) {
  if (collapsedGroups.value.has(clusterId)) {
    collapsedGroups.value.delete(clusterId)
  } else {
    collapsedGroups.value.add(clusterId)
  }
  collapsedGroups.value = new Set(collapsedGroups.value)
}

function handleGroupSelectChange(group, checked) {
  for (const item of group.items) {
    selectionStore.toggleSearchSelection(item.index, checked)
    if (checked) homophoneSearch.selectedSubtitleIds.add(item.index)
    else homophoneSearch.selectedSubtitleIds.delete(item.index)
  }
  homophoneSearch.selectedSubtitleIds = new Set(homophoneSearch.selectedSubtitleIds)
}

function isGroupAllSelected(group) {
  return group.items.every((item) => homophoneSearch.selectedSubtitleIds.has(item.index))
}

function isGroupIndeterminate(group) {
  const selectedCount = group.items.filter((item) => {
    return homophoneSearch.selectedSubtitleIds.has(item.index)
  }).length
  return selectedCount > 0 && selectedCount < group.items.length
}

async function handleBatchReplace() {
  if (!identityId.value) {
    ElMessage.error('缺少项目标识，无法执行批量替换')
    return
  }

  try {
    await syncEngine.flush()
    // 先冻结一次“替换前”视图快照。
    // 原因：后端 batch replace 会同步推送 subtitle.edited SSE，
    // 如果等接口返回后再读取 projectionSubtitles，当前文本可能已经被 SSE 改成新值，
    // 进而让 replacements 误判为“无变化”，导致 history 不入栈、批量替换不可撤销。
    const subtitlesBeforeBatchReplace = projectionSubtitles.value.map((subtitle) => ({
      localId: subtitle.localId,
      segment_id: subtitle.segment_id ?? null,
      sentenceIndex: subtitle.sentenceIndex ?? null,
      legacy_index: subtitle.legacy_index ?? subtitle.sentenceIndex ?? null,
      text: subtitle.text ?? '',
    }))

    const result = await homophoneSearch.executeBatchReplace()
    if (!result.success) {
      ElMessage.error('批量替换失败')
      return
    }

    if (result.count <= 0) {
      ElMessage.warning('未产生可替换内容')
      return
    }

    const serverSegments = await projectApi.getSubtitles(identityId.value)
    const replacements = buildBatchReplaceReplacements({
      subtitles: subtitlesBeforeBatchReplace,
      serverSegments,
      updatedIndices: result.indices,
    })

    if (replacements.length > 0) {
      commandBus.dispatch(createBatchReplaceCommand({
        replacements,
        source: 'user',
        skipSync: true,
      }))
    }

    const needsAuthoritativeReload = replacements.length === 0
      || replacements.length < result.indices.length
    if (needsAuthoritativeReload) {
      const { loadSubtitlesFromBackend } = await import('@/stores/editor/editorCoreLoader')
      await loadSubtitlesFromBackend(identityId.value)
    }

    ElMessage.success(`成功替换 ${result.count} 处`)
  } catch (error) {
    console.error('[VirtualSubtitleList] 批量替换失败:', error)
    ElMessage.error('批量替换失败')
  }
}

function handleSplitFromCursor({ localId, cursorPosition, text }) {
  const subtitle = editorProjectionBridge.findSubtitleById(localId)
  const entity = docStore.getEntity(localId)
  const cold = docStore.getCold(localId)
  if (!subtitle || !entity) {
    ElMessage.error('字幕不存在，无法切分')
    return
  }

  const hasDraftText = typeof text === 'string' && text !== entity.text
  const splitSourceSubtitle = hasDraftText
    ? {
        ...subtitle,
        text,
        words: [],
      }
    : subtitle

  const splitResult = splitSubtitleByCursor(splitSourceSubtitle, cursorPosition)
  if (!splitResult.success) {
    ElMessage.warning(splitResult.error || '切分失败')
    return
  }

  const createdLocalId = sessionStore.nextLocalId()
  const leftWords = normalizeProjectionWordsToCold(splitResult.left.words, projectStore)
  const rightWords = normalizeProjectionWordsToCold(splitResult.right.words, projectStore)

  runStructuralMutationWithFlip(() => {
    const dispatchResult = commandBus.dispatch(createSplitSubtitleCommand({
      sourceLocalId: localId,
      sourceSegmentId: cold?.segmentId ?? null,
      createdLocalId,
      splitAtMs: Math.round(projectStore.toBaseTime(splitResult.splitAtTime) * 1000),
      splitAtTextOffset: splitResult.splitAtTextOffset,
      before: {
        text: splitSourceSubtitle.text,
        startMs: entity.startMs,
        endMs: entity.endMs,
      },
      afterKept: {
        text: splitResult.left.text,
        startMs: Math.round(projectStore.toBaseTime(splitResult.left.start) * 1000),
        endMs: Math.round(projectStore.toBaseTime(splitResult.left.end) * 1000),
      },
      afterCreated: {
        text: splitResult.right.text,
        startMs: Math.round(projectStore.toBaseTime(splitResult.right.start) * 1000),
        endMs: Math.round(projectStore.toBaseTime(splitResult.right.end) * 1000),
      },
      keptColdPatch: {
        words: leftWords,
      },
      createdColdInit: cold
        ? {
            ...cold,
            localId: createdLocalId,
            segmentId: null,
            sentenceIndex: null,
            originalText: null,
            words: rightWords,
          }
        : {
            words: rightWords,
          },
      source: 'user',
    }))

    if (!dispatchResult?.success) {
      ElMessage.error('切分失败')
      return dispatchResult
    }

    selectionStore.selectOnly(createdLocalId)
    subtitleDocumentStore.setSelectedSubtitleId(createdLocalId)
    return dispatchResult
  })
}

function handleMerge(localId, direction, mergePayload = null) {
  const neighbors = docStore.getNeighbors(localId)
  const keptLocalId = direction === 'prev' ? neighbors.prev : localId
  const removedLocalId = direction === 'prev' ? localId : neighbors.next
  if (!keptLocalId || !removedLocalId) {
    ElMessage.warning('没有可合并的相邻字幕')
    return
  }

  const keptSubtitle = editorProjectionBridge.findSubtitleById(keptLocalId)
  const removedSubtitle = editorProjectionBridge.findSubtitleById(removedLocalId)
  const keptEntity = docStore.getEntity(keptLocalId)
  const removedEntity = docStore.getEntity(removedLocalId)
  const keptCold = docStore.getCold(keptLocalId)
  const removedCold = docStore.getCold(removedLocalId)
  if (!keptSubtitle || !removedSubtitle || !keptEntity || !removedEntity) {
    ElMessage.error('字幕不存在，无法合并')
    return
  }

  const keptSnapshot = resolveMergeSideSnapshot(keptLocalId, keptEntity, mergePayload)
  const removedSnapshot = resolveMergeSideSnapshot(removedLocalId, removedEntity, mergePayload)
  const separator = getMergeSeparator()

  runStructuralMutationWithFlip(() => {
    const dispatchResult = commandBus.dispatch(createMergeSubtitlesCommand({
      keptLocalId,
      removedLocalId,
      keptSegmentId: keptCold?.segmentId ?? null,
      removedSegmentId: removedCold?.segmentId ?? null,
      separator,
      beforeKept: keptSnapshot,
      beforeRemoved: removedSnapshot,
      after: {
        text: `${keptSnapshot.text}${separator}${removedSnapshot.text}`,
        startMs: Math.min(keptSnapshot.startMs, removedSnapshot.startMs),
        endMs: Math.max(keptSnapshot.endMs, removedSnapshot.endMs),
      },
      keptColdPatch: {
        words: [],
        originalText: null,
        confidence: null,
        displayConfidence: null,
        confidenceSource: 'manual',
        warningType: 'none',
        sourceType: 'merge',
      },
      source: 'user',
    }))

    if (!dispatchResult?.success) {
      ElMessage.error('合并失败')
      return dispatchResult
    }

    selectionStore.selectOnly(keptLocalId)
    subtitleDocumentStore.setSelectedSubtitleId(keptLocalId)
    return dispatchResult
  })
}

function getScrollerElement() {
  if (isGroupedMode.value) {
    return listContainerRef.value
  }
  return scrollerRef.value?.$el ?? scrollerRef.value ?? null
}

function bindScrollerListeners() {
  const nextScrollerElement = getScrollerElement()
  if (attachedScrollerElement === nextScrollerElement) {
    return
  }

  if (attachedScrollerElement) {
    attachedScrollerElement.removeEventListener('scroll', handleListScroll)
    attachedScrollerElement.removeEventListener('wheel', handleListWheel)
    attachedScrollerElement.removeEventListener('touchmove', handleListTouchMove)
  }

  attachedScrollerElement = nextScrollerElement
  if (!attachedScrollerElement) {
    return
  }

  attachedScrollerElement.addEventListener('scroll', handleListScroll, { passive: true })
  attachedScrollerElement.addEventListener('wheel', handleListWheel, { passive: true })
  attachedScrollerElement.addEventListener('touchmove', handleListTouchMove, { passive: true })
}

function clearFollowResumeTimer() {
  if (!followResumeTimer) {
    return
  }
  clearTimeout(followResumeTimer)
  followResumeTimer = null
}

function clearProgrammaticScrollReleaseTimer() {
  if (!programmaticScrollReleaseTimer) {
    return
  }
  clearTimeout(programmaticScrollReleaseTimer)
  programmaticScrollReleaseTimer = null
}

function beginProgrammaticScrollLock() {
  clearProgrammaticScrollReleaseTimer()
  programmaticScrollGuardUntil = Number.POSITIVE_INFINITY
}

function extendProgrammaticScrollGuard(durationMs = PROGRAMMATIC_SCROLL_GUARD_MS) {
  clearProgrammaticScrollReleaseTimer()
  programmaticScrollGuardUntil = Date.now() + durationMs
  programmaticScrollReleaseTimer = setTimeout(() => {
    programmaticScrollGuardUntil = 0
    programmaticScrollReleaseTimer = null
  }, durationMs)
}

function releaseProgrammaticScrollLock() {
  clearProgrammaticScrollReleaseTimer()
  programmaticScrollGuardUntil = 0
}

function isInProgrammaticScrollGuard() {
  return Date.now() < programmaticScrollGuardUntil
}

function scheduleFollowAutoResume() {
  clearFollowResumeTimer()
  if (!props.enableAutoResumeFollow) {
    return
  }
  followResumeTimer = setTimeout(() => {
    isFollowPausedByUser.value = false
    followResumeTimer = null
  }, USER_SCROLL_AUTO_RESUME_DELAY_MS)
}

function pauseFollowByUserScroll() {
  if (!autoScrollEnabled.value || isInProgrammaticScrollGuard()) {
    return
  }
  isFollowPausedByUser.value = true
  scheduleFollowAutoResume()
}

function markRecentUserListInteraction() {
  lastManualListInteractionAt.value = Date.now()
}

function handleListScroll() {
  markRecentUserListInteraction()
  pauseFollowByUserScroll()
}

function handleListWheel() {
  markRecentUserListInteraction()
  pauseFollowByUserScroll()
}

function handleListTouchMove() {
  markRecentUserListInteraction()
  pauseFollowByUserScroll()
}

function hasRecentUserScroll() {
  return Date.now() - lastManualListInteractionAt.value < USER_SCROLL_FLIP_GUARD_MS
}

function resolveReducedMotionPreference() {
  return window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false
}

// 结构性编辑事务包装器：采集前快照 -> 执行命令 -> 等布局稳定 -> 播放 FLIP
async function runStructuralMutationWithFlip(mutate) {
  // 新事务开始前先取消未完成的上一个动画
  cancelActiveFlipAnimation?.()
  cancelActiveFlipAnimation = null

  const container = getScrollerElement()
  const skipDecision = shouldSkipVisibleFlip({
    isGroupedMode: isGroupedMode.value,
    isResizing: props.isResizing,
    isInProgrammaticScrollGuard: isInProgrammaticScrollGuard(),
    hasRecentUserScroll: hasRecentUserScroll(),
    prefersReducedMotion: resolveReducedMotionPreference(),
  })

  if (skipDecision.skip || !container) {
    return await mutate()
  }

  const before = captureVisibleSnapshot(container)
  const result = await mutate()

  // FLIP 阶段与命令执行解耦：即使动画出错也不影响命令结果
  try {
    if (result?.success === false) {
      return result
    }

    const runToken = ++flipRunToken
    await nextTick()
    const after = await waitForStableLayout({
      capture: () => captureVisibleSnapshot(container),
    })

    // 如果在等待期间有新事务进入，放弃本次动画
    if (runToken !== flipRunToken) {
      return result
    }

    const plan = buildVisibleFlipPlan(before, after, {
      maxAnimatedItems: FLIP_MAX_ITEMS,
      enterOffsetPx: FLIP_ENTER_OFFSET_PX,
    })

    cancelActiveFlipAnimation = playVisibleFlip(plan, {
      durationMs: FLIP_DURATION_MS,
    })
  } catch {
    // Why: FLIP 是纯视觉增强，失败时静默降级，不阻断命令执行
  }

  return result
}

function scrollToLocalId(localId) {
  if (!localId) {
    return
  }

  const visibleIndex = defaultVisibleIds.value.indexOf(localId)
  if (visibleIndex === -1) {
    return
  }

  const requestToken = ++followRequestToken
  beginProgrammaticScrollLock()
  const didStartFollow = followCurrentSubtitle(scrollerRef.value, {
    localId,
    index: visibleIndex,
    shouldAbort: () => requestToken !== followRequestToken,
    onCentered: () => {
      if (requestToken !== followRequestToken) {
        return
      }
      extendProgrammaticScrollGuard()
    },
    onFailed: () => {
      if (requestToken !== followRequestToken) {
        return
      }
      releaseProgrammaticScrollLock()
    },
  })

  if (!didStartFollow) {
    releaseProgrammaticScrollLock()
  }
}

watch(selectedLocalId, (localId) => {
  if (!localId) {
    if (activeId.value && !orderedVisibleIds.value.includes(activeId.value)) {
      selectionStore.setActive(null, { updateAnchor: false })
    }
    return
  }
  selectionStore.setActive(localId, { updateAnchor: false })
})

watch(orderedVisibleIds, (ids) => {
  if (!ids.includes(activeId.value)) {
    selectionStore.setActive(ids[0] ?? null, { updateAnchor: false })
  }
}, { immediate: true })

watch(currentSubtitleId, (localId) => {
  if (!autoScrollEnabled.value || homophoneSearch.isSearchActive || !localId || isFollowPausedByUser.value) {
    return
  }

  nextTick(() => {
    scrollToLocalId(localId)
  })
})

watch(autoScrollEnabled, (enabled) => {
  if (!enabled) {
    followRequestToken += 1
    clearFollowResumeTimer()
    releaseProgrammaticScrollLock()
    isFollowPausedByUser.value = false
  }
})

watch(
  () => props.enableAutoResumeFollow,
  (enabled) => {
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
  () => homophoneSearch.isSearchActive,
  (isActive) => {
    if (!isActive) {
      collapsedGroups.value = new Set()
      selectionStore.clearSearchSelection()
    }
  }
)

onMounted(() => {
  nextTick(() => {
    bindScrollerListeners()
  })
})

onUnmounted(() => {
  clearFollowResumeTimer()
  clearProgrammaticScrollReleaseTimer()
  followRequestToken += 1
  cancelActiveFlipAnimation?.()
  cancelActiveFlipAnimation = null
  if (!attachedScrollerElement) {
    return
  }

  attachedScrollerElement.removeEventListener('scroll', handleListScroll)
  attachedScrollerElement.removeEventListener('wheel', handleListWheel)
  attachedScrollerElement.removeEventListener('touchmove', handleListTouchMove)
  attachedScrollerElement = null
})

watch(
  [scrollerRef, listContainerRef, isGroupedMode, () => virtualVisibleRows.value.length],
  () => {
    nextTick(() => {
      bindScrollerListeners()
    })
  },
  { flush: 'post' }
)

defineExpose({
  sortMode: computed(() => homophoneSearch.sortMode),
  isSearchActive: computed(() => homophoneSearch.isSearchActive),
  toggleSortMode() {
    homophoneSearch.sortMode = homophoneSearch.sortMode === SortMode.GROUPED
      ? SortMode.TIMELINE
      : SortMode.GROUPED
  },
  // 供外部（如 EditorView 的 undo/redo）包裹结构性变更以触发 FLIP 动画
  // 约束：FLIP 仅做视觉采快照-播动画，不干涉命令执行；异常时降级跳过
  runWithFlip(mutate) {
    return runStructuralMutationWithFlip(mutate)
  },
})
</script>

<style scoped>
.virtual-subtitle-list {
  height: 100%;
  display: flex;
  flex-direction: column;
  background: var(--af-bg-primary);
}

.scroller {
  flex: 1;
  overflow-y: auto;
  scrollbar-gutter: stable;
  padding: 6px;
}

.scroller :deep(.vue-recycle-scroller__item-wrapper) {
  overflow: visible;
}

.grouped-container {
  position: relative;
}

.group-items {
  padding-bottom: 6px;
}

.row-shell {
  padding-bottom: 6px;
  box-sizing: border-box;
}

.scroller::-webkit-scrollbar {
  width: 6px;
}

.scroller::-webkit-scrollbar-track {
  background: transparent;
}

.scroller::-webkit-scrollbar-thumb {
  background: var(--af-border-default);
  border-radius: 3px;
}

.scroller::-webkit-scrollbar-thumb:hover {
  background: var(--af-text-muted);
}

.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 32px 16px;
  color: var(--af-text-muted);
  font-size: 13px;
}

.empty-icon {
  width: 48px;
  height: 48px;
  opacity: 0.5;
}

.empty-action-btn {
  padding: 6px 16px;
  background: var(--af-accent-primary);
  border: none;
  border-radius: var(--af-radius-md);
  color: var(--af-text-inverse);
  font-size: 13px;
  cursor: pointer;
  transition: background-color var(--af-transition-fast);
}

.empty-action-btn:hover {
  background: var(--af-accent-primary-hover);
}
</style>
