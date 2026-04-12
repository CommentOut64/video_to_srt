<!-- V3.2.5+dev.20260314.03: DragOverlay - 草稿拖拽覆盖层 -->
<template>
  <div
    v-if="isDragging"
    class="drag-overlay"
    @pointermove="handlePointerMove"
    @pointerup="handlePointerUp"
  >
    <div class="draft-preview" :style="previewStyle">
      {{ draftText }}
    </div>
  </div>
</template>

<script setup>
// V3.2.5+dev.20260314.03: 草稿拖拽 + pointerup 提交
import { ref, computed } from 'vue'
import { useEditorDraftStore } from '@/stores/editor/editorDraftStore'
import { useEditorCommandBus } from '@/stores/editor/editorCommandBus'
import { createUpdateTimingCommand } from '@/stores/editor/editorCommandFactory'

const props = defineProps({
  localId: { type: String, required: true },
  initialX: { type: Number, required: true },
  initialY: { type: Number, required: true }
})

const emit = defineEmits(['drag-end'])

const draftStore = useEditorDraftStore()
const commandBus = useEditorCommandBus()

const isDragging = ref(true)
const currentX = ref(props.initialX)
const currentY = ref(props.initialY)

const draftText = computed(() => draftStore.activeTimingDraft?.text || '')

const previewStyle = computed(() => ({
  left: `${currentX.value}px`,
  top: `${currentY.value}px`
}))

function handlePointerMove(e) {
  currentX.value = e.clientX
  currentY.value = e.clientY
}

function handlePointerUp(e) {
  isDragging.value = false

  const draft = draftStore.activeTimingDraft
  if (draft) {
    commandBus.dispatch(createUpdateTimingCommand({
      localId: props.localId,
      before: { startMs: draft.originalStartMs, endMs: draft.originalEndMs },
      after: { startMs: draft.startMs, endMs: draft.endMs },
      source: 'user',
    }))
  }

  emit('drag-end')
}
</script>

<style scoped>
.drag-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 9999;
  cursor: grabbing;
}

.draft-preview {
  position: absolute;
  padding: 8px 12px;
  background: var(--af-accent-primary);
  color: white;
  border-radius: var(--af-radius-sm);
  font-size: 12px;
  pointer-events: none;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  max-width: 200px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
</style>
