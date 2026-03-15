<!-- V3.2.5+dev.20260314.03: TimelineViewport - 时间轴窗口化渲染 -->
<template>
  <div class="timeline-viewport" ref="containerRef">
    <div class="timeline-regions" :style="containerStyle">
      <div
        v-for="entity in visibleRegions"
        :key="entity.localId"
        class="region"
        :class="{ 'is-active': activeId === entity.localId }"
        :style="getRegionStyle(entity)"
        @mousedown="handleMouseDown(entity, $event)"
      >
        <span class="region-text">{{ entity.text }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
// V3.2.5+dev.20260314.03: 二分查找窗口化渲染
import { computed, ref, shallowRef } from 'vue'
import { useEditorDocumentStore } from '@/stores/editor/editorDocumentStore'

const props = defineProps({
  viewportStartMs: { type: Number, default: 0 },
  viewportEndMs: { type: Number, default: 60000 },
  pixelsPerMs: { type: Number, default: 0.1 }
})

const emit = defineEmits(['region-click'])

const docStore = useEditorDocumentStore()
const containerRef = ref(null)
const activeId = ref(null)

// 二分查找可见字幕
const visibleRegions = computed(() => {
  const order = docStore.order
  const entities = docStore.entities

  const bufferMs = 5000
  const searchStart = props.viewportStartMs - bufferMs
  const searchEnd = props.viewportEndMs + bufferMs

  let left = 0, right = order.length
  while (left < right) {
    const mid = Math.floor((left + right) / 2)
    const entity = entities.get(order[mid])
    if (entity && entity.endMs < searchStart) {
      left = mid + 1
    } else {
      right = mid
    }
  }

  const visible = []
  for (let i = left; i < order.length; i++) {
    const entity = entities.get(order[i])
    if (!entity) continue
    if (entity.startMs > searchEnd) break
    if (entity.endMs >= searchStart && entity.startMs <= searchEnd && !entity.isDeleted) {
      visible.push(entity)
    }
  }

  return visible
})

const containerStyle = computed(() => ({
  width: `${(props.viewportEndMs - props.viewportStartMs) * props.pixelsPerMs}px`,
  height: '100px'
}))

function getRegionStyle(entity) {
  const left = (entity.startMs - props.viewportStartMs) * props.pixelsPerMs
  const width = (entity.endMs - entity.startMs) * props.pixelsPerMs
  return {
    left: `${left}px`,
    width: `${width}px`
  }
}

function handleMouseDown(entity, event) {
  activeId.value = entity.localId
  emit('region-click', entity.localId)
}
</script>

<style scoped>
.timeline-viewport {
  position: relative;
  overflow-x: auto;
  overflow-y: hidden;
  background: var(--af-bg-secondary);
  border: 1px solid var(--af-border-primary);
}

.timeline-regions {
  position: relative;
  height: 100%;
}

.region {
  position: absolute;
  top: 10px;
  height: 40px;
  background: var(--af-accent-primary);
  border: 1px solid var(--af-border-primary);
  border-radius: var(--af-radius-sm);
  cursor: pointer;
  display: flex;
  align-items: center;
  padding: 0 8px;
  transition: background var(--af-transition-fast);
  overflow: hidden;
}

.region:hover {
  background: var(--af-accent-secondary);
}

.region.is-active {
  border: 2px solid var(--af-accent-primary);
  background: rgb(var(--af-accent-primary-rgb), 0.3);
}

.region-text {
  font-size: 11px;
  color: var(--af-text-normal);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
</style>
