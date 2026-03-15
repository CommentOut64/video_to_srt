<!-- V3.2.5+dev.20260315.04: VirtualSubtitleList - 虚拟滚动字幕列表 -->
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
          <div class="row-shell">
            <SubtitleRow
              :local-id="item"
              :row-index="index"
              :is-selected="selection.has(item)"
              :is-active="activeId === item"
              @click="handleRowClick"
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
// V3.2.5+dev.20260315.04: simpleArray 模式必须传 index，间距也必须计入被测量高度
import { computed, ref } from 'vue'
import { DynamicScroller, DynamicScrollerItem } from 'vue-virtual-scroller'
import 'vue-virtual-scroller/dist/vue-virtual-scroller.css'
import SubtitleRow from './SubtitleRow.vue'
import SearchToolbar from '@/components/editor/SubtitleList/SearchToolbar.vue'
import { useEditorDocumentStore } from '@/stores/editor/editorDocumentStore'
import { useEditorCommandBus } from '@/stores/editor/editorCommandBus'
import { createDeleteSubtitleCommand, createInsertSubtitleCommand } from '@/stores/editor/editorCommandFactory'

const docStore = useEditorDocumentStore()
const commandBus = useEditorCommandBus()

const activeId = ref(null)
const selection = ref(new Set())

// 只传 localId[]，不传整个 entity
const visibleIds = computed(() => {
  console.log('[VirtualSubtitleList] 计算 visibleIds, docStore.order:', docStore.order)
  if (!Array.isArray(docStore.order)) {
    console.log('[VirtualSubtitleList] order 不是数组，返回空')
    return []
  }
  const filtered = docStore.order.filter(id => {
    const entity = docStore.entities.get(id)
    return entity && !entity.isDeleted
  })
  console.log('[VirtualSubtitleList] 过滤后数量:', filtered.length)
  return filtered
})

function handleRowClick(localId) {
  activeId.value = localId
}

function handleKeydown(e) {
  if (!activeId.value) return

  const currentIndex = docStore.order.indexOf(activeId.value)

  switch (e.key) {
    case 'ArrowUp':
      e.preventDefault()
      if (currentIndex > 0) {
        activeId.value = docStore.order[currentIndex - 1]
      }
      break
    case 'ArrowDown':
      e.preventDefault()
      if (currentIndex < docStore.order.length - 1) {
        activeId.value = docStore.order[currentIndex + 1]
      }
      break
    case 'Delete':
      e.preventDefault()
      commandBus.dispatch(createDeleteSubtitleCommand({
        localId: activeId.value,
        source: 'user',
      }))
      break
    case 'Enter':
      e.preventDefault()
      commandBus.dispatch(createInsertSubtitleCommand({
        afterLocalId: activeId.value,
        source: 'user',
      }))
      break
  }
}

function handleAddSubtitle() {
  commandBus.dispatch(createInsertSubtitleCommand({
    afterLocalId: null,
    source: 'user',
  }))
}
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
