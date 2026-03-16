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
              :row-index="item.index"
              :is-selected="selectionStore.isMultiSelected(item.subtitle.localId)"
              :is-active="activeId === item.subtitle.localId"
              :is-current="currentSubtitleId === item.subtitle.localId"
              :match-spans="item.matchSpans"
              :is-match-selected="item.isSelected"
              :is-selectable="true"
              :cluster-color="item.clusterColor"
              :can-merge-prev="canMergePrev(item.subtitle.localId)"
              :can-merge-next="canMergeNext(item.subtitle.localId)"
              @click="handleRowClick"
              @split="handleSplitFromCursor"
              @merge-prev="handleMerge(item.subtitle.localId, 'prev')"
              @merge-next="handleMerge(item.subtitle.localId, 'next')"
              @insert-before="handleInsertBefore(item.subtitle.localId)"
              @insert-after="handleInsertAfter(item.subtitle.localId)"
              @delete="handleDelete(item.subtitle.localId)"
              @select-change="(checked) => handleItemSelectChange(item.index, checked)"
            />
          </div>
        </div>
      </template>
    </div>

    <DynamicScroller
      v-else
      ref="scrollerRef"
      :items="virtualVisibleIds"
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
          :size-dependencies="resolveRowSizeDependencies(item, index)"
        >
          <div class="row-shell" :data-local-id="item">
            <SubtitleRow
              :local-id="item"
              :row-index="resolveRowMeta(item, index).rowIndex"
              :is-selected="selectionStore.isMultiSelected(item)"
              :is-active="activeId === item"
              :is-current="currentSubtitleId === item"
              :match-spans="resolveRowMeta(item, index).matchSpans"
              :is-match-selected="resolveRowMeta(item, index).isMatchSelected"
              :is-selectable="resolveRowMeta(item, index).isSelectable"
              :cluster-color="resolveRowMeta(item, index).clusterColor"
              :can-merge-prev="canMergePrev(item)"
              :can-merge-next="canMergeNext(item)"
              @click="handleRowClick"
              @split="handleSplitFromCursor"
              @merge-prev="handleMerge(item, 'prev')"
              @merge-next="handleMerge(item, 'next')"
              @insert-before="handleInsertBefore(item)"
              @insert-after="handleInsertAfter(item)"
              @delete="handleDelete(item)"
              @select-change="(checked) => handleItemSelectChange(resolveRowMeta(item, index).selectionIndex, checked)"
            />
          </div>
        </DynamicScrollerItem>
      </template>
    </DynamicScroller>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { DynamicScroller, DynamicScrollerItem } from 'vue-virtual-scroller'
import 'vue-virtual-scroller/dist/vue-virtual-scroller.css'
import GroupHeader from '@/components/editor/SubtitleList/GroupHeader.vue'
import SearchToolbar from '@/components/editor/SubtitleList/SearchToolbar.vue'
import { SortMode, useHomophoneSearch } from '@/composables/useHomophoneSearch'
import SubtitleRow from './SubtitleRow.vue'
import { buildBatchReplaceReplacements } from './searchBatchReplace'
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

const props = defineProps({
  autoScroll: { type: Boolean, default: true },
  editable: { type: Boolean, default: true },
  enableAutoResumeFollow: { type: Boolean, default: true },
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

const defaultVisibleIds = computed(() => {
  return filteredSubtitles.value.map((subtitle) => subtitle.localId)
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

const virtualVisibleIds = computed(() => {
  if (homophoneSearch.isSearchActive && homophoneSearch.sortMode === SortMode.TIMELINE) {
    return homophoneSearch.timelineViewItems.map((item) => item.subtitle.localId)
  }
  return defaultVisibleIds.value
})

const orderedVisibleIds = computed(() => {
  return isGroupedMode.value
    ? groupedVisibleIds.value
    : virtualVisibleIds.value
})

const showNoSubtitles = computed(() => {
  return !homophoneSearch.isSearchActive && defaultVisibleIds.value.length === 0
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

function resolveRowSizeDependencies(localId, fallbackIndex) {
  const rowMeta = resolveRowMeta(localId, fallbackIndex)
  return [
    rowMeasurementKeyByLocalId.value.get(localId) || '',
    rowMeta.matchSpans.length,
    rowMeta.isSelectable ? 'search' : 'normal',
  ]
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

function handleInsertBefore(localId) {
  const currentIndex = defaultVisibleIds.value.indexOf(localId)
  const previousLocalId = currentIndex > 0 ? defaultVisibleIds.value[currentIndex - 1] : null
  const currentEntity = docStore.getEntity(localId)
  const previousEntity = previousLocalId ? docStore.getEntity(previousLocalId) : null
  const startMs = previousEntity?.endMs ?? Math.max(0, (currentEntity?.startMs ?? 3000) - 3000)
  const candidateEndMs = currentEntity?.startMs ?? (startMs + 3000)
  const endMs = candidateEndMs > startMs ? candidateEndMs : (startMs + 500)

  commandBus.dispatch(createInsertSubtitleCommand({
    afterLocalId: previousLocalId,
    entity: {
      text: '',
      startMs,
      endMs,
      isDraft: false,
    },
    source: 'user',
  }))
}

function handleInsertAfter(localId) {
  const currentEntity = docStore.getEntity(localId)
  const nextLocalId = docStore.getNeighbors(localId).next
  const nextEntity = nextLocalId ? docStore.getEntity(nextLocalId) : null
  const startMs = currentEntity?.endMs ?? 0
  const candidateEndMs = nextEntity?.startMs ?? (startMs + 3000)
  const endMs = candidateEndMs > startMs ? candidateEndMs : (startMs + 500)

  commandBus.dispatch(createInsertSubtitleCommand({
    afterLocalId: localId,
    entity: {
      text: '',
      startMs,
      endMs,
      isDraft: false,
    },
    source: 'user',
  }))
}

function handleDelete(localId) {
  const result = commandBus.dispatch(createDeleteSubtitleCommand({
    localId,
    source: 'user',
  }))

  if (!result?.success) {
    return
  }

  if (selectionStore.isMultiSelected(localId)) {
    selectionStore.toggleMultiSelection(localId, {
      makeActive: false,
      updateAnchor: false,
    })
  }

  if (selectedLocalId.value === localId) {
    subtitleDocumentStore.setSelectedSubtitleId(null)
  }

  if (activeId.value === localId) {
    const fallbackId = defaultVisibleIds.value.find((id) => id !== localId) ?? null
    selectionStore.setActive(fallbackId, { updateAnchor: false })
  }
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
  const commands = selectedIds.map((localId) => createDeleteSubtitleCommand({
    localId,
    source: 'user',
  }))
  const result = commandBus.dispatchTransaction(commands, {
    historyType: 'batch_delete',
  })
  if (!result?.success) {
    ElMessage.error('批量删除失败')
    return
  }

  selectionStore.clearMultiSelection()
  const fallbackId = docStore.order[minIndex] ?? docStore.order[minIndex - 1] ?? null
  selectionStore.setActive(fallbackId, { updateAnchor: false })
  subtitleDocumentStore.setSelectedSubtitleId(fallbackId)
}

function handleKeydown(event) {
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
  await homophoneSearch.executeSearch()
  selectionStore.setSearchSelection(Array.from(homophoneSearch.selectedSubtitleIds))
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

function canMergePrev(localId) {
  const entity = docStore.getEntity(localId)
  const prevLocalId = docStore.getNeighbors(localId).prev
  const prevEntity = prevLocalId ? docStore.getEntity(prevLocalId) : null
  return Boolean(entity && !entity.isDraft && prevEntity && !prevEntity.isDraft)
}

function canMergeNext(localId) {
  const entity = docStore.getEntity(localId)
  const nextLocalId = docStore.getNeighbors(localId).next
  const nextEntity = nextLocalId ? docStore.getEntity(nextLocalId) : null
  return Boolean(entity && !entity.isDraft && nextEntity && !nextEntity.isDraft)
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
  const dispatchResult = commandBus.dispatch(createSplitSubtitleCommand({
    sourceLocalId: localId,
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
    return
  }

  selectionStore.selectOnly(createdLocalId)
  subtitleDocumentStore.setSelectedSubtitleId(createdLocalId)
}

function handleMerge(localId, direction) {
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
  if (!keptSubtitle || !removedSubtitle || !keptEntity || !removedEntity) {
    ElMessage.error('字幕不存在，无法合并')
    return
  }

  const separator = getMergeSeparator()
  const dispatchResult = commandBus.dispatch(createMergeSubtitlesCommand({
    keptLocalId,
    removedLocalId,
    separator,
    beforeKept: {
      text: keptEntity.text,
      startMs: keptEntity.startMs,
      endMs: keptEntity.endMs,
    },
    beforeRemoved: {
      text: removedEntity.text,
      startMs: removedEntity.startMs,
      endMs: removedEntity.endMs,
    },
    after: {
      text: `${keptEntity.text}${separator}${removedEntity.text}`,
      startMs: Math.min(keptEntity.startMs, removedEntity.startMs),
      endMs: Math.max(keptEntity.endMs, removedEntity.endMs),
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
    return
  }

  selectionStore.selectOnly(keptLocalId)
  subtitleDocumentStore.setSelectedSubtitleId(keptLocalId)
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

function handleListScroll() {
  pauseFollowByUserScroll()
}

function handleListWheel() {
  pauseFollowByUserScroll()
}

function handleListTouchMove() {
  pauseFollowByUserScroll()
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
  if (!attachedScrollerElement) {
    return
  }

  attachedScrollerElement.removeEventListener('scroll', handleListScroll)
  attachedScrollerElement.removeEventListener('wheel', handleListWheel)
  attachedScrollerElement.removeEventListener('touchmove', handleListTouchMove)
  attachedScrollerElement = null
})

watch(
  [scrollerRef, listContainerRef, isGroupedMode, () => virtualVisibleIds.value.length],
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
  color: var(--af-text-muted);
  font-size: 13px;
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
