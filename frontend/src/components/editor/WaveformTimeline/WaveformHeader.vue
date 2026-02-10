<template>
  <div class="timeline-header">
    <div class="zoom-controls">
      <button class="zoom-btn" @click="$emit('zoom-out')" title="缩小">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M19 13H5v-2h14v2z" />
        </svg>
      </button>
      <div class="zoom-slider">
        <input
          type="range"
          :value="zoomLevel"
          :min="ZOOM_MIN"
          :max="ZOOM_MAX"
          :step="ZOOM_STEP"
          @input="$emit('zoom-input', $event)"
        />
      </div>
      <button class="zoom-btn" @click="$emit('zoom-in')" title="放大">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z" />
        </svg>
      </button>
      <span class="zoom-label">{{ zoomLevel }}%</span>
      <button class="fit-btn" @click="$emit('fit-screen')" title="适应屏幕">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path
            d="M3 5v4h2V5h4V3H5c-1.1 0-2 .9-2 2zm2 10H3v4c0 1.1.9 2 2 2h4v-2H5v-4zm14 4h-4v2h4c1.1 0 2-.9 2-2v-4h-2v4zm0-16h-4v2h4v4h2V5c0-1.1-.9-2-2-2z"
          />
        </svg>
      </button>
    </div>

    <div class="time-indicator">
      <span class="current-time">{{ formatTime(currentTime) }}</span>
      <span class="separator">/</span>
      <span class="total-time">{{ formatTime(duration) }}</span>
    </div>
  </div>
</template>

<script setup>
import { ZOOM_MIN, ZOOM_MAX, ZOOM_STEP } from '@/composables'

defineProps({
  zoomLevel: { type: Number, required: true },
  currentTime: { type: Number, default: 0 },
  duration: { type: Number, default: 0 },
})

defineEmits(['zoom-in', 'zoom-out', 'zoom-input', 'fit-screen'])

function formatTime(seconds) {
  if (!seconds || isNaN(seconds)) return '0:00'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}
</script>

<style scoped>
.timeline-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 16px;
  background: var(--af-bg-tertiary);
  border-bottom: 1px solid var(--af-border-default);
}

.zoom-controls {
  display: flex;
  align-items: center;
  gap: 8px;
}

.zoom-controls .zoom-btn,
.zoom-controls .fit-btn {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 28px;
  height: 28px;
  border-radius: var(--af-radius-sm);
  color: var(--af-text-secondary);
  transition: all var(--af-transition-fast);
}

.zoom-controls .zoom-btn svg,
.zoom-controls .fit-btn svg {
  width: 16px;
  height: 16px;
}

.zoom-controls .zoom-btn:hover,
.zoom-controls .fit-btn:hover {
  background: var(--af-bg-elevated);
  color: var(--af-text-primary);
}

.zoom-controls .zoom-slider {
  width: 100px;
}

.zoom-controls .zoom-slider input[type='range'] {
  width: 100%;
  height: 4px;
  background: var(--af-bg-elevated);
  border-radius: 2px;
  appearance: none;
  cursor: pointer;
}

.zoom-controls .zoom-slider input[type='range']::-webkit-slider-thumb {
  width: 12px;
  height: 12px;
  background: var(--af-accent-primary);
  border-radius: 50%;
  appearance: none;
  cursor: pointer;
}

.zoom-controls .zoom-label {
  color: var(--af-text-muted);
  font-size: 12px;
  min-width: 50px;
  font-family: var(--af-font-mono);
  text-align: center;
}

.time-indicator {
  display: flex;
  align-items: center;
  gap: 6px;
  font-family: var(--af-font-mono);
  font-size: 13px;
}

.time-indicator .current-time {
  color: var(--af-accent-primary);
  font-weight: 600;
}

.time-indicator .separator {
  color: var(--af-text-muted);
}

.time-indicator .total-time {
  color: var(--af-text-secondary);
}
</style>
