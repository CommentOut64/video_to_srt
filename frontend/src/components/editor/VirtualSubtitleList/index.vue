<!-- V3.2.5+dev.20260315.11: VirtualSubtitleList - 虚拟滚动字幕列表 -->
<template>
  <div class="virtual-subtitle-list">
    <!-- 工具栏 -->
    <SearchToolbar
      :total-subtitles="visibleIds.length"
      :draft-count="0"
      @add-subtitle="handleAddSubtitle"
    />

    <DynamicScroller
      v-if="visibleIds.length > 0"
      ref="scrollerRef"
      :items="visibleIds"
      :min-item-size="98"
      class="scroller"
      @keydown="handleKeydown"
      tabindex="0"
    >
      <template #default="{ item, index, active }">
        <DynamicScrollerItem
          :item="item"
          :index="index"
          :active="active"
        >
          <div class="row-shell" :data-local-id="item">
            <SubtitleRow
              :local-id="item"
              :row-index="index"
              :is-selected="selectedLocalId === item"
              :is-active="activeId === item"
              :is-current="currentSubtitleId === item"
              @click="handleRowClick"
              @insert-before="handleInsertBefore(item)"
              @insert-after="handleInsertAfter(item)"
              @delete="handleDelete(item)"
            />
          </div>
        </DynamicScrollerItem>
      </template>
    </DynamicScroller>

    <div v-else class="empty-state">
      <p>暂无字幕</p>
    </div>
  </div>
</template>

<script setup>
// V3.2.5+dev.20260315.11: simpleArray 模式必须传 index，间距也必须计入被测量高度
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { DynamicScroller, DynamicScrollerItem } from 'vue-virtual-scroller'
import 'vue-virtual-scroller/dist/vue-virtual-scroller.css'
import SubtitleRow from './SubtitleRow.vue'
import SearchToolbar from '@/components/editor/SubtitleList/SearchToolbar.vue'
import { useEditorDocumentStore } from '@/stores/editor/editorDocumentStore'
import { useEditorCommandBus } from '@/stores/editor/editorCommandBus'
import { useProjectStore } from '@/stores/projectStore'
import { useSubtitleDocumentStore } from '@/stores/subtitleDocumentStore'
import { useEditorPlayback } from '@/composables/editor/useEditorPlayback'
import { createDeleteSubtitleCommand, createInsertSubtitleCommand } from '@/stores/editor/editorCommandFactory'

const props = defineProps({
  autoScroll: { type: Boolean, default: true },
  editable: { type: Boolean, default: true },
  enableAutoResumeFollow: { type: Boolean, default: true },
})

const docStore = useEditorDocumentStore()
const commandBus = useEditorCommandBus()
const projectStore = useProjectStore()
const subtitleDocumentStore = useSubtitleDocumentStore()
const { currentSubtitleId, seekToSubtitle, followCurrentSubtitle } = useEditorPlayback()

const scrollerRef = ref(null)
const activeId = ref(null)
const selectedLocalId = computed(() => subtitleDocumentStore.selectedSubtitleId)
const autoScrollEnabled = computed(() => props.autoScroll && projectStore.view.autoScroll !== false)

const isFollowPausedByUser = ref(false)
const USER_SCROLL_AUTO_RESUME_DELAY_MS = 6000
const PROGRAMMATIC_SCROLL_GUARD_MS = 900
let followResumeTimer = null
let programmaticScrollGuardUntil = 0
let programmaticScrollReleaseTimer = null
let attachedScrollerElement = null
let followRequestToken = 0

// 只传 localId[]，不传整个 entity
const visibleIds = computed(() => {
  if (!Array.isArray(docStore.order)) {
    return []
  }
  return docStore.order.filter(id => {
    const entity = docStore.entities.get(id)
    return entity && !entity.isDeleted
  })
})

function handleRowClick(localId) {
  activeId.value = localId
  subtitleDocumentStore.setSelectedSubtitleId(localId)
  seekToSubtitle(localId)
}

function handleInsertBefore(localId) {
  const currentIndex = visibleIds.value.indexOf(localId)
  const previousLocalId = currentIndex > 0 ? visibleIds.value[currentIndex - 1] : null
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

  if (selectedLocalId.value === localId) {
    subtitleDocumentStore.setSelectedSubtitleId(null)
  }

  if (activeId.value === localId) {
    const fallbackId = visibleIds.value.find((id) => id !== localId) ?? null
    activeId.value = fallbackId
  }
}

function handleKeydown(e) {
  if (!activeId.value) return

  const currentIndex = visibleIds.value.indexOf(activeId.value)

  switch (e.key) {
    case 'ArrowUp':
      e.preventDefault()
      if (currentIndex > 0) {
        const nextId = visibleIds.value[currentIndex - 1]
        activeId.value = nextId
        subtitleDocumentStore.setSelectedSubtitleId(nextId)
      }
      break
    case 'ArrowDown':
      e.preventDefault()
      if (currentIndex < visibleIds.value.length - 1) {
        const nextId = visibleIds.value[currentIndex + 1]
        activeId.value = nextId
        subtitleDocumentStore.setSelectedSubtitleId(nextId)
      }
      break
    case 'Delete':
      e.preventDefault()
      handleDelete(activeId.value)
      break
    case 'Enter':
      e.preventDefault()
      handleInsertAfter(activeId.value)
      break
  }
}

function handleAddSubtitle() {
  const afterLocalId = visibleIds.value[visibleIds.value.length - 1] ?? null
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

function getScrollerElement() {
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

  const visibleIndex = visibleIds.value.indexOf(localId)
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
    if (activeId.value && !visibleIds.value.includes(activeId.value)) {
      activeId.value = null
    }
    return
  }
  activeId.value = localId
})

watch(visibleIds, (ids) => {
  if (!ids.includes(activeId.value)) {
    activeId.value = ids[0] ?? null
  }
}, { immediate: true })

watch(currentSubtitleId, (localId) => {
  if (!autoScrollEnabled.value || !localId || isFollowPausedByUser.value) {
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

watch([scrollerRef, () => visibleIds.value.length], () => {
  nextTick(() => {
    bindScrollerListeners()
  })
}, { flush: 'post' })
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

.row-shell {
  padding-bottom: 6px;
  box-sizing: border-box;
}

/* 暗色滚动条 */
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
  align-items: center;
  justify-content: center;
  color: var(--af-text-muted);
  font-size: 13px;
}
</style>
