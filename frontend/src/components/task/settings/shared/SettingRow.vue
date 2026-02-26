<template>
  <div class="setting-row" :class="{ disabled }">
    <div class="setting-label">
      <span class="label-text">{{ label }}</span>
      <span v-if="hint" class="label-hint">{{ hint }}</span>
    </div>
    <div class="setting-control">
      <slot />
    </div>
  </div>
</template>

<script setup>
defineProps({
  label: {
    type: String,
    required: true,
  },
  hint: {
    type: String,
    default: '',
  },
  disabled: {
    type: Boolean,
    default: false,
  },
})
</script>

<style scoped>
.setting-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  padding: 4px 0;
}

.setting-row + .setting-row {
  border-top: 1px solid var(--af-border-muted);
}

.setting-row.disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.setting-row.disabled * {
  pointer-events: none;
}

.setting-label {
  display: flex;
  flex-direction: column;
  gap: 1px;
  flex-shrink: 0;
  min-width: 100px;
}

.label-text {
  color: var(--af-text-normal);
  font-size: 12px;
  font-weight: 500;
}

.label-hint {
  color: var(--af-text-muted);
  font-size: 10px;
}

.setting-control {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
}
</style>
