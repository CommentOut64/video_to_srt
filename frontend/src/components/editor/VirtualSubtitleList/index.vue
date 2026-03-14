<!-- V3.2.5+dev.20260314.03: VirtualSubtitleList - 虚拟滚动字幕列表 -->
<template>
  <div class="virtual-subtitle-list">
    <RecycleScroller
      v-if="visibleIds.length > 0"
      :items="visibleIds"
      :item-size="60"
      key-field="id"
      class="scroller"
      @keydown="handleKeydown"
      tabindex="0"
    >
      <template #default="{ item }">
        <SubtitleRow
          :local-id="item"
          :is-selected="selection.has(item)"
          :is-active="activeId === item"
          @click="handleRowClick"
        />
      </template>
    </RecycleScroller>

    <div v-else class="empty-state">
      <p>暂无字幕</p>
    </div>
  </div>
</template>

<script setup>
// V3.2.5+dev.20260314.03: 只传 localId[]，避免每次 order 变化都重建整表对象数组
import { computed, ref } from 'vue'
import { RecycleScroller } from 'vue-virtual-scroller'
import 'vue-virtual-scroller/dist/vue-virtual-scroller.css'
import SubtitleRow from './SubtitleRow.vue'
import { useEditorDocumentStore } from '@/stores/editor/editorDocumentStore'
import { useEditorCommandBus } from '@/stores/editor/editorCommandBus'

const docStore = useEditorDocumentStore()
const commandBus = useEditorCommandBus()

const activeId = ref(null)
const selection = ref(new Set())

// 只传 localId[]，不传整个 entity
const visibleIds = computed(() => {
  if (!docStore.order.value) return []
  return docStore.order.value.filter(id => {
    const entity = docStore.entities.get(id)
    return entity && !entity.isDeleted
  })
})

function handleRowClick(localId) {
  activeId.value = localId
}

function handleKeydown(e) {
  if (!activeId.value) return

  const currentIndex = docStore.order.value.indexOf(activeId.value)

  switch (e.key) {
    case 'ArrowUp':
      e.preventDefault()
      if (currentIndex > 0) {
        activeId.value = docStore.order.value[currentIndex - 1]
      }
      break
    case 'ArrowDown':
      e.preventDefault()
      if (currentIndex < docStore.order.value.length - 1) {
        activeId.value = docStore.order.value[currentIndex + 1]
      }
      break
    case 'Delete':
      e.preventDefault()
      commandBus.dispatch({
        type: 'delete_subtitle',
        localId: activeId.value,
        source: 'user'
      })
      break
    case 'Enter':
      e.preventDefault()
      commandBus.dispatch({
        type: 'insert_subtitle',
        afterLocalId: activeId.value,
        source: 'user'
      })
      break
  }
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
}

/* 重置 vue-virtual-scroller 默认样式 */
.scroller :deep(.vue-recycle-scroller__item-wrapper) {
  overflow: visible;
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
