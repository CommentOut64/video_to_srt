<template>
  <div class="custom-scrollbar" @wheel="$emit('wheel', $event)">
    <div class="scrollbar-track" ref="trackRef" @mousedown="$emit('mousedown', $event)">
      <div class="scrollbar-thumb" :style="thumbStyle"></div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

defineProps({
  thumbStyle: { type: Object, required: true },
})

defineEmits(['wheel', 'mousedown'])

const trackRef = ref(null)

defineExpose({ trackRef })
</script>

<style scoped>
.custom-scrollbar {
  height: 14px;
  padding: 3px 16px;
  background: var(--af-bg-tertiary);
  flex-shrink: 0;
}

/* stylelint-disable color-function-alias-notation -- 使用 rgba 传统语法保证兼容性 */
.scrollbar-track {
  position: relative;
  width: 100%;
  height: 8px;
  background: rgba(var(--af-text-on-dark-rgb), 0.06);
  border-radius: 4px;
  cursor: pointer;
  transition: background 0.2s ease;
}

.scrollbar-thumb {
  position: absolute;
  top: 0;
  height: 100%;
  background: rgba(var(--af-text-secondary-rgb), 0.3);
  border-radius: 4px;
  transition:
    background 0.15s ease,
    left 0.05s linear,
    width 0.05s linear;
  cursor: grab;
  min-width: 20px;
}

.scrollbar-thumb:hover {
  background: rgba(var(--af-text-secondary-rgb), 0.5);
}

.scrollbar-thumb:active {
  cursor: grabbing;
  background: rgba(var(--af-text-secondary-rgb), 0.65);
}

.custom-scrollbar:hover .scrollbar-thumb {
  background: rgba(var(--af-text-secondary-rgb), 0.25);
}

.scrollbar-track:hover {
  background: rgba(var(--af-text-on-dark-rgb), 0.05);
}

.scrollbar-track:hover .scrollbar-thumb {
  background: rgba(var(--af-text-secondary-rgb), 0.55);
}
/* stylelint-enable color-function-alias-notation */
</style>
