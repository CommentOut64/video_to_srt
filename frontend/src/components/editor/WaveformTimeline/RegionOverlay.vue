<template>
  <div ref="overlayRef" class="region-overlay" :style="overlayStyle">
    <div
      v-for="region in regions"
      :key="region.localId"
      class="region-overlay__item"
      :class="{
        'is-selected': region.selected,
        'is-overlapping': region.overlapping,
        'is-dragging': activeSession?.localId === region.localId && activeSession?.mode === 'body',
        'is-resizing': activeSession?.localId === region.localId && activeSession?.mode !== 'body',
        'is-marker': isRegionMarker(region),
        'is-playback-active': props.playbackActiveIds.has(region.localId),
      }"
      :style="getRegionStyle(region)"
      :part="getRegionPart(region)"
      :data-region-id="region.localId"
      @pointerdown="handleRegionPointerDown(region, 'body', $event)"
      @dblclick="handleRegionDblClick(region, $event)"
    >
      <div
        v-if="shouldShowStartHandle(region)"
        class="region-overlay__handle region-overlay__handle--start"
        part="region-handle region-handle-left"
        @pointerdown.stop="handleRegionPointerDown(region, 'start', $event)"
      ></div>
      <div class="region-overlay__body">
        <div v-if="hasRegionContent(region)" class="region-overlay__content" part="region-content">
          {{ region.content }}
        </div>
      </div>
      <div
        v-if="shouldShowEndHandle(region)"
        class="region-overlay__handle region-overlay__handle--end"
        part="region-handle region-handle-right"
        @pointerdown.stop="handleRegionPointerDown(region, 'end', $event)"
      ></div>
    </div>
  </div>
</template>

<script setup>
// V3.2.5+dev.20260320.01: 补齐与 wavesurfer Region 插件的交互/样式差距
import { computed, ref, onBeforeUnmount, onMounted, onUpdated } from 'vue'
import {
  buildRegionInlineStyle,
  getRegionPointerMoveThreshold,
  isMarkerRegion,
} from './regionOverlayContract.js'

const props = defineProps({
  regions: {
    type: Array,
    default: () => [],
  },
  contentWidth: {
    type: Number,
    default: 0,
  },
  pixelsPerMs: {
    type: Number,
    default: 0,
  },
  durationMs: {
    type: Number,
    default: 0,
  },
  minLengthMs: {
    type: Number,
    default: 50,
  },
  maxLengthMs: {
    type: Number,
    default: Infinity,
  },
  dragEnabled: {
    type: Boolean,
    default: true,
  },
  resizeEnabled: {
    type: Boolean,
    default: true,
  },
  resizeStart: {
    type: Boolean,
    default: true,
  },
  resizeEnd: {
    type: Boolean,
    default: true,
  },
  playbackActiveIds: {
    type: Set,
    default: () => new Set(),
  },
})

const emit = defineEmits(['region-click', 'region-commit', 'region-dblclick'])

const overlayRef = ref(null)
const activeSession = ref(null)
let pointerMoveHandler = null
let pointerUpHandler = null
let pointerCancelHandler = null
let pointerOutHandler = null
let touchMoveHandler = null
let blurHandler = null

// 活跃指针跟踪（多指安全）
const activePointers = new Set()

// coarse pointer 设备检测（触屏），参照 wavesurfer createDragDetector
const isCoarsePointer =
  typeof matchMedia === 'function' && matchMedia('(pointer: coarse)').matches
const TOUCH_DELAY_MS = 100
const REGION_OVERLAY_SHADOW_STYLE_SELECTOR = 'style[data-region-overlay-shadow-style="true"]'
const REGION_OVERLAY_SHADOW_STYLE_TEXT = `
.region-overlay {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 5;
  height: 100%;
  pointer-events: none;
}

.region-overlay__item {
  position: absolute;
  top: 0;
  height: 100%;
  box-sizing: border-box;
  border-radius: 2px;
  transition: background-color 0.2s ease, box-shadow 0.15s ease;
  pointer-events: all;
  user-select: none;
  touch-action: none;
}

.region-overlay__item.is-dragging,
.region-overlay__item.is-resizing {
  transition: none;
}

.region-overlay__item.is-dragging {
  cursor: grabbing;
}

.region-overlay__item:not(.is-dragging):not(.is-resizing):hover {
  box-shadow: inset 0 0 0 9999px rgba(255, 255, 255, 0.12);
}

.region-overlay__item.is-playback-active:not(.is-dragging):not(.is-resizing) {
  box-shadow: inset 0 0 0 9999px rgba(255, 255, 255, 0.15);
}

.region-overlay__item.is-playback-active:not(.is-dragging):not(.is-resizing):hover {
  box-shadow: inset 0 0 0 9999px rgba(255, 255, 255, 0.22);
}

.region-overlay__body {
  position: relative;
  width: 100%;
  height: 100%;
}

.region-overlay__content {
  display: inline-block;
  padding: 0.2em 0.4em;
  pointer-events: none;
}

.region-overlay__item.is-marker .region-overlay__content {
  padding: 0.2em 0.2em;
}

.region-overlay__handle {
  position: absolute;
  z-index: 2;
  top: 0;
  width: 6px;
  height: 100%;
  box-sizing: content-box;
  background: transparent;
  touch-action: none;
  cursor: ew-resize;
  word-break: keep-all;
}

.region-overlay__handle--start {
  left: 0;
  border-left: 2px solid rgba(0, 0, 0, 0.5);
  border-radius: 2px 0 0 2px;
}

.region-overlay__handle--end {
  right: 0;
  border-right: 2px solid rgba(0, 0, 0, 0.5);
  border-radius: 0 2px 2px 0;
}
`

const overlayStyle = computed(() => ({
  width: `${Math.max(0, Number(props.contentWidth) || 0)}px`,
}))

function ensureShadowRootStyles() {
  const overlayElement = overlayRef.value
  const rootNode = overlayElement?.getRootNode?.()
  if (!(rootNode instanceof ShadowRoot)) return

  let styleElement = rootNode.querySelector(REGION_OVERLAY_SHADOW_STYLE_SELECTOR)
  if (!styleElement) {
    styleElement = document.createElement('style')
    styleElement.setAttribute('data-region-overlay-shadow-style', 'true')
    styleElement.textContent = REGION_OVERLAY_SHADOW_STYLE_TEXT
    rootNode.appendChild(styleElement)
    return
  }

  if (styleElement.textContent !== REGION_OVERLAY_SHADOW_STYLE_TEXT) {
    styleElement.textContent = REGION_OVERLAY_SHADOW_STYLE_TEXT
  }
}

// ============ 视图模型 ============

function getDisplayRange(region) {
  if (activeSession.value?.localId === region.localId) {
    return {
      startMs: activeSession.value.previewStartMs,
      endMs: activeSession.value.previewEndMs,
    }
  }
  return {
    startMs: Number(region.startMs ?? 0),
    endMs: Number(region.endMs ?? region.startMs ?? 0),
  }
}

function getRegionStyle(region) {
  const displayRange = getDisplayRange(region)
  return buildRegionInlineStyle({
    startMs: displayRange.startMs,
    endMs: displayRange.endMs,
    durationMs: props.durationMs,
    color: region.color,
    dragEnabled: props.dragEnabled,
    isDragging:
      activeSession.value?.localId === region.localId &&
      activeSession.value?.mode === 'body',
  })
}

function isRegionMarker(region) {
  const displayRange = getDisplayRange(region)
  return isMarkerRegion(displayRange.startMs, displayRange.endMs)
}

function getRegionPart(region) {
  return `${isRegionMarker(region) ? 'marker' : 'region'} ${region.localId}`
}

function shouldShowStartHandle(region) {
  return props.resizeEnabled && props.resizeStart && !isRegionMarker(region)
}

function shouldShowEndHandle(region) {
  return props.resizeEnabled && props.resizeEnd && !isRegionMarker(region)
}

function hasRegionContent(region) {
  return Boolean(region?.content)
}

// ============ 拖拽时容器滚动跟随 ============

function findScrollableAncestor() {
  let el = overlayRef.value?.parentElement
  while (el) {
    if (el.scrollWidth > el.clientWidth) return el
    el = el.parentElement
  }
  return null
}

function adjustScrollDuringDrag(event) {
  const container = findScrollableAncestor()
  if (!container) return
  const { clientWidth, scrollWidth } = container
  if (scrollWidth <= clientWidth) return

  const containerRect = container.getBoundingClientRect()
  const pointerX = event.clientX - containerRect.left
  const edgeZone = 40

  // 距边缘越近滚动越快
  if (pointerX < edgeZone) {
    const ratio = 1 - pointerX / edgeZone
    container.scrollLeft = Math.max(0, container.scrollLeft - Math.ceil(ratio * 12))
  } else if (pointerX > clientWidth - edgeZone) {
    const ratio = (pointerX - (clientWidth - edgeZone)) / edgeZone
    container.scrollLeft = Math.min(
      scrollWidth - clientWidth,
      container.scrollLeft + Math.ceil(ratio * 12),
    )
  }
}

// ============ 指针会话管理 ============

function cleanupPointerSession() {
  if (pointerMoveHandler) {
    document.removeEventListener('pointermove', pointerMoveHandler, true)
    pointerMoveHandler = null
  }
  if (pointerUpHandler) {
    document.removeEventListener('pointerup', pointerUpHandler, true)
    pointerUpHandler = null
  }
  if (pointerCancelHandler) {
    document.removeEventListener('pointercancel', pointerCancelHandler, true)
    pointerCancelHandler = null
  }
  if (pointerOutHandler) {
    document.removeEventListener('pointerout', pointerOutHandler, true)
    pointerOutHandler = null
  }
  if (touchMoveHandler) {
    document.removeEventListener('touchmove', touchMoveHandler)
    touchMoveHandler = null
  }
  if (blurHandler) {
    window.removeEventListener('blur', blurHandler, true)
    blurHandler = null
  }
  activePointers.clear()
}

/**
 * 拖拽结束后阻止紧随的 click 事件冒泡到父级（如 waveform seek）。
 * 与 wavesurfer createDragDetector 中 preventClick 逻辑等价。
 */
function suppressNextClick() {
  let removed = false
  const handler = (e) => {
    e.stopPropagation()
    e.preventDefault()
    remove()
  }
  const remove = () => {
    if (removed) return
    removed = true
    document.removeEventListener('click', handler, true)
  }
  document.addEventListener('click', handler, true)
  // 兜底移除，防止 click 未触发导致监听器泄漏
  setTimeout(remove, 20)
}

function finalizePointerSession(event, cancelled = false) {
  const session = activeSession.value
  if (!session) {
    cleanupPointerSession()
    return
  }

  if (event?.pointerId != null) {
    activePointers.delete(event.pointerId)
  }

  cleanupPointerSession()
  activeSession.value = null

  if (cancelled) {
    return
  }

  const hasMoved = session.moved === true
  const payload = {
    localId: session.localId,
    startMs: session.previewStartMs,
    endMs: session.previewEndMs,
    side: session.mode === 'body' ? null : session.mode,
  }

  if (hasMoved) {
    suppressNextClick()
    emit('region-commit', payload)
    return
  }

  emit('region-click', payload)
}

/**
 * 指针离开浏览器视口时终止拖拽。
 * relatedTarget 为 null 或 documentElement 表示指针已离开页面。
 */
function handlePointerOut(event) {
  if (!activeSession.value) return
  if (event.pointerId !== activeSession.value.pointerId) return
  if (!event.relatedTarget || event.relatedTarget === document.documentElement) {
    finalizePointerSession(event, false)
  }
}

function attachPointerSession() {
  cleanupPointerSession()
  pointerMoveHandler = (event) => handlePointerMove(event)
  pointerUpHandler = (event) => finalizePointerSession(event, false)
  pointerCancelHandler = (event) => finalizePointerSession(event, true)
  pointerOutHandler = (event) => handlePointerOut(event)
  blurHandler = () => finalizePointerSession(null, true)
  // 拖拽期间阻止触摸滚动（与 wavesurfer createDragDetector 一致）
  touchMoveHandler = (event) => {
    if (activeSession.value?.moved) {
      event.preventDefault()
    }
  }

  document.addEventListener('pointermove', pointerMoveHandler, true)
  document.addEventListener('pointerup', pointerUpHandler, true)
  document.addEventListener('pointercancel', pointerCancelHandler, true)
  document.addEventListener('pointerout', pointerOutHandler, true)
  document.addEventListener('touchmove', touchMoveHandler, { passive: false })
  window.addEventListener('blur', blurHandler, true)
}

// ============ 范围约束 ============

function clampPreviewRange(session, nextStartMs, nextEndMs) {
  const durationMs = Math.max(0, Number(props.durationMs) || 0)
  const minLengthMs = Math.max(1, Number(props.minLengthMs) || 1)
  const maxLengthMs = Number.isFinite(props.maxLengthMs)
    ? Math.max(minLengthMs, props.maxLengthMs)
    : Infinity

  if (session.mode === 'body') {
    const lengthMs = session.originEndMs - session.originStartMs
    const maxStartMs = Math.max(0, durationMs - lengthMs)
    const clampedStartMs = Math.min(maxStartMs, Math.max(0, nextStartMs))
    return {
      startMs: clampedStartMs,
      endMs: clampedStartMs + lengthMs,
    }
  }

  if (session.mode === 'start') {
    const maxStartMs = Math.max(0, nextEndMs - minLengthMs)
    const minStartMs =
      maxLengthMs < Infinity ? Math.max(0, nextEndMs - maxLengthMs) : 0
    return {
      startMs: Math.max(minStartMs, Math.min(maxStartMs, Math.max(0, nextStartMs))),
      endMs: nextEndMs,
    }
  }

  // mode === 'end'
  const minEndMs = nextStartMs + minLengthMs
  const maxEndMs =
    maxLengthMs < Infinity
      ? nextStartMs + maxLengthMs
      : durationMs || Number.MAX_SAFE_INTEGER
  return {
    startMs: nextStartMs,
    endMs: Math.max(minEndMs, Math.min(maxEndMs, nextEndMs)),
  }
}

// ============ 指针事件处理 ============

function handlePointerMove(event) {
  const session = activeSession.value
  if (!session || session.pointerId !== event.pointerId) {
    return
  }

  // 多指触摸保护：超过 1 个活跃指针时忽略（与 wavesurfer 一致）
  if (activePointers.size > 1) {
    return
  }

  // coarse pointer 设备延迟保护：防止滚动手势误触发拖拽
  if (
    isCoarsePointer &&
    session.pointerDownTime &&
    Date.now() - session.pointerDownTime < TOUCH_DELAY_MS
  ) {
    return
  }

  if (props.pixelsPerMs <= 0) {
    return
  }

  const deltaPx = event.clientX - session.originClientX
  const deltaMs = Math.round(deltaPx / props.pixelsPerMs)
  let nextStartMs = session.originStartMs
  let nextEndMs = session.originEndMs

  if (session.mode === 'body') {
    nextStartMs += deltaMs
    nextEndMs += deltaMs
  } else if (session.mode === 'start') {
    nextStartMs += deltaMs
  } else {
    nextEndMs += deltaMs
  }

  const normalizedRange = clampPreviewRange(session, nextStartMs, nextEndMs)
  const moveThreshold = getRegionPointerMoveThreshold(session.mode)
  activeSession.value = {
    ...session,
    previewStartMs: normalizedRange.startMs,
    previewEndMs: normalizedRange.endMs,
    moved: session.moved || Math.abs(deltaPx) >= moveThreshold,
  }

  // 拖拽时自动滚动容器
  if (activeSession.value.moved) {
    adjustScrollDuringDrag(event)
  }
}

function handleRegionPointerDown(region, mode, event) {
  if (event.pointerType === 'mouse' && event.button !== 0) {
    return
  }

  // 多指安全：记录活跃指针
  activePointers.add(event.pointerId)
  if (activePointers.size > 1) {
    // 多指时取消当前会话
    if (activeSession.value) {
      finalizePointerSession(null, true)
    }
    return
  }

  if (mode === 'body' && !props.dragEnabled) {
    emit('region-click', {
      localId: region.localId,
      startMs: region.startMs,
      endMs: region.endMs,
      side: null,
    })
    return
  }
  if (mode === 'start' && (!props.resizeEnabled || !props.resizeStart)) {
    return
  }
  if (mode === 'end' && (!props.resizeEnabled || !props.resizeEnd)) {
    return
  }

  event.preventDefault()
  event.stopPropagation()

  activeSession.value = {
    localId: region.localId,
    mode,
    pointerId: event.pointerId,
    originClientX: event.clientX,
    originStartMs: Number(region.startMs ?? 0),
    originEndMs: Number(region.endMs ?? region.startMs ?? 0),
    previewStartMs: Number(region.startMs ?? 0),
    previewEndMs: Number(region.endMs ?? region.startMs ?? 0),
    moved: false,
    pointerDownTime: Date.now(),
  }

  if (typeof event.currentTarget?.setPointerCapture === 'function') {
    try {
      event.currentTarget.setPointerCapture(event.pointerId)
    } catch {
      // 指针捕获失败时退回 document 级监听即可
    }
  }

  attachPointerSession()
}

function handleRegionDblClick(region, event) {
  event.stopPropagation()
  emit('region-dblclick', {
    localId: region.localId,
    startMs: Number(region.startMs ?? 0),
    endMs: Number(region.endMs ?? region.startMs ?? 0),
  })
}

// 组件卸载时清理残留会话
onMounted(() => {
  ensureShadowRootStyles()
})

onUpdated(() => {
  ensureShadowRootStyles()
})

onBeforeUnmount(() => {
  cleanupPointerSession()
  activeSession.value = null
})
</script>

<style scoped>
.region-overlay {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 5;
  height: 100%;
  pointer-events: none;
}

.region-overlay__item {
  position: absolute;
  top: 0;
  height: 100%;
  box-sizing: border-box;
  border-radius: 2px;
  transition: background-color 0.2s ease, box-shadow 0.15s ease;
  pointer-events: all;
  user-select: none;
  touch-action: none;
}

/* 拖拽/缩放中禁用过渡，确保流畅跟手 */
.region-overlay__item.is-dragging,
.region-overlay__item.is-resizing {
  transition: none;
}

.region-overlay__item.is-dragging {
  cursor: grabbing;
}

/*
 * hover 使用 inset box-shadow 叠加半透明白色层，
 * 不覆盖 background-color，保留重叠(红)/选中(紫)等状态色。
 * 与 wavesurfer 事件驱动的 hover 效果视觉等价。
 */
.region-overlay__item:not(.is-dragging):not(.is-resizing):hover {
  box-shadow: inset 0 0 0 9999px rgba(255, 255, 255, 0.12);
}

.region-overlay__body {
  position: relative;
  width: 100%;
  height: 100%;
}

.region-overlay__content {
  display: inline-block;
  padding: 0.2em 0.4em;
  pointer-events: none;
}

/* Marker 内容使用更窄的水平 padding（与 wavesurfer 一致） */
.region-overlay__item.is-marker .region-overlay__content {
  padding: 0.2em 0.2em;
}

.region-overlay__handle {
  position: absolute;
  z-index: 2;
  top: 0;
  width: 6px;
  height: 100%;
  box-sizing: content-box;
  background: transparent;
  touch-action: none;
  cursor: ew-resize;
  word-break: keep-all;
}

.region-overlay__handle--start {
  left: 0;
  border-left: 2px solid rgba(0, 0, 0, 0.5);
  border-radius: 2px 0 0 2px;
}

.region-overlay__handle--end {
  right: 0;
  border-right: 2px solid rgba(0, 0, 0, 0.5);
  border-radius: 0 2px 2px 0;
}
</style>
