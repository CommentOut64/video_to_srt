<template>
  <div class="waveform-timeline" ref="containerRef" tabindex="-1">
    <!-- 缩放控制栏（子组件） -->
    <WaveformHeader
      :zoom-level="zoomLevel"
      :current-time="currentTime"
      :duration="duration"
      @zoom-in="zoomIn"
      @zoom-out="zoomOut"
      @zoom-input="handleZoomInput"
      @fit-screen="fitToScreen"
    />

    <!-- 波形容器 -->
    <div
      class="waveform-wrapper"
      ref="waveformWrapperRef"
      :style="{ cursor: currentMouseCursor }"
      @contextmenu="handleWaveformContextMenu"
    >
      <!-- 上半部分交互层 -->
      <div
        class="waveform-upper-zone"
        :class="{ 'is-region-dragging': isRegionPointerDragging }"
        @pointerdown="handleUpperZonePointerDown"
        @pointermove="handleUpperZonePointerMove"
        @pointerleave="handleWaveformPointerLeave"
      ></div>

      <!-- WaveSurfer 波形 -->
      <div id="waveform" ref="waveformRef"></div>

      <!-- 加载状态 -->
      <div v-if="isLoading" class="waveform-loading">
        <div class="loading-spinner"></div>
        <span>加载波形中...</span>
      </div>

      <!-- 错误状态 -->
      <div v-if="hasError" class="waveform-error">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path
            d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"
          />
        </svg>
        <span>{{ errorMessage }}</span>
        <button @click="retryLoad">重试</button>
      </div>
    </div>

    <!-- 自定义滚动条（子组件） -->
    <WaveformScrollbar
      ref="scrollbarRef"
      :thumb-style="scrollbarThumbStyle"
      @wheel="handleScrollbarWheel"
      @mousedown="handleScrollbarMouseDown"
    />

    <!-- 右键菜单 -->
    <ContextMenu ref="contextMenuRef" :items="contextMenuItems" @select="handleContextMenuSelect" />
  </div>
</template>

<script setup>
/**
 * WaveformTimeline 组件 - 重构版
 *
 * 重构内容：
 * 1. 逻辑提取到 5 个 composables（缩放、滚动、拖拽、Region、右键菜单）
 * 2. UI 拆分为子组件（WaveformHeader、WaveformScrollbar）
 * 3. 样式迁移为纯 CSS + CSS Variables
 *
 * 原文件行数：2252 行 → 重构后：~550 行
 */
import { ref, computed, watch, onMounted, onUnmounted, nextTick, inject } from 'vue'
import { useProjectStore } from '@/stores/projectStore'
import { usePlaybackStore } from '@/stores/playbackStore'
import { useSubtitleDocumentStore } from '@/stores/subtitleDocumentStore'
import { usePlaybackManager } from '@/services/PlaybackManager'
import ContextMenu from '@/components/editor/ContextMenu.vue'
import WaveformHeader from './WaveformHeader.vue'
import WaveformScrollbar from './WaveformScrollbar.vue'
import {
  useSubtitleSync,
  useWaveformZoom,
  useWaveformScroll,
  useWaveformCursorDrag,
  useWaveformRegions,
  useWaveformContextMenu,
  calculateWaveformConfig,
  ZOOM_MIN,
  ZOOM_MAX,
  ZOOM_WHEEL_STEP,
  ZOOM_BASE_PX_PER_SEC,
} from '@/composables'

// ============ Props & Emits ============
const props = defineProps({
  audioUrl: String,
  peaksUrl: String,
  mediaId: String,
  waveColor: { type: String, default: '#58a6ff' },
  progressColor: { type: String, default: '#238636' },
  cursorColor: { type: String, default: '#f85149' },
  height: { type: Number, default: 128 },
  regionColor: { type: String, default: 'rgba(88, 166, 255, 0.25)' },
  dragEnabled: { type: Boolean, default: true },
  resizeEnabled: { type: Boolean, default: true },
})

const emit = defineEmits(['ready', 'region-update', 'region-click', 'seek', 'zoom'])

// ============ Store & Services ============
const projectStore = useProjectStore()
const playbackStore = usePlaybackStore()
const subtitleDocumentStore = useSubtitleDocumentStore()
const playbackManager = usePlaybackManager()
const identityRef = computed(() => props.mediaId || projectStore.primaryId)
const { onSubtitleEdit } = useSubtitleSync(identityRef)

// 编辑器上下文
const editorContext = inject('editorContext', {
  isMediaReady: computed(() => true),
  isVideoReady: computed(() => true),
  hasVideoSource: computed(() => true),
})
const isMediaReady = computed(
  () => editorContext.isMediaReady?.value ?? editorContext.isVideoReady?.value ?? true
)
const hasVideoSource = computed(
  () => editorContext.hasVideoSource?.value ?? Boolean(projectStore.meta.videoPath)
)

// ============ DOM Refs ============
const containerRef = ref(null)
const waveformRef = ref(null)
const waveformWrapperRef = ref(null)
const scrollbarRef = ref(null)

// ============ 基础状态 ============
const isLoading = ref(true)
const hasError = ref(false)
const errorMessage = ref('')
const isReady = ref(false)
const retryCount = ref(0)
const maxRetries = 3

// WaveSurfer 实例
const wavesurferRef = ref(null)
const regionsPluginRef = ref(null)

// ============ Computed ============
const audioSource = computed(() => {
  if (props.audioUrl) return props.audioUrl
  if (props.mediaId) return `/api/media/${props.mediaId}/audio`
  return projectStore.meta.audioPath || ''
})

const peaksSource = computed(() => {
  if (props.peaksUrl) return props.peaksUrl
  if (props.mediaId) return `/api/media/${props.mediaId}/peaks?samples=0`
  return projectStore.meta.peaksPath || ''
})

const currentTime = computed(() => playbackStore.currentTime)
const duration = computed(() => projectStore.meta.duration || 0)

// 滚动条轨道 ref（从子组件获取）
const scrollbarTrackRef = computed(() => scrollbarRef.value?.trackRef)

// ============ Composables 集成 ============

// 缩放逻辑
const {
  zoomLevel,
  zoomIn,
  zoomOut,
  fitToScreen,
  handleZoomInput,
  handleZoomWithSmartAnchor,
  cleanup: cleanupZoom,
} = useWaveformZoom(
  wavesurferRef,
  containerRef,
  projectStore,
  playbackStore,
  () => updateScrollbarThumb()
)

// 滚动逻辑
const {
  scrollbarThumbStyle,
  updateScrollbarThumb,
  handleScrollbarMouseDown,
  handleScrollbarWheel,
  startSmartFollow,
  stopSmartFollow,
  cleanup: cleanupScroll,
} = useWaveformScroll(
  wavesurferRef,
  scrollbarTrackRef,
  zoomLevel,
  playbackStore,
  isReady
)

// 光标拖拽逻辑
const {
  isRegionPointerDragging,
  currentMouseCursor,
  handleUpperZonePointerDown,
  handleUpperZonePointerMove,
  handleWaveformPointerLeave,
  getTimeFromClientX,
  setupRegionPointerGuards,
  teardownRegionPointerGuards,
  cleanup: cleanupCursorDrag,
} = useWaveformCursorDrag(
  wavesurferRef,
  zoomLevel,
  playbackManager,
  playbackStore,
  isReady,
  isMediaReady,
  emit
)

// Region 管理逻辑
const {
  isUpdatingRegions,
  setupRegionEvents,
  renderSubtitleRegions,
  cleanup: cleanupRegions,
} = useWaveformRegions(
  regionsPluginRef,
  projectStore,
  props,
  isReady,
  onSubtitleEdit,
  playbackManager,
  subtitleDocumentStore,
  emit
)

// 右键菜单逻辑
const {
  contextMenuRef,
  contextMenuItems,
  handleWaveformContextMenu: onContextMenu,
  handleContextMenuSelect,
} = useWaveformContextMenu(projectStore, identityRef)

// 包装右键菜单处理（需要传递额外参数）
function handleWaveformContextMenu(e) {
  onContextMenu(e, getTimeFromClientX)
}

// ============ WaveSurfer 初始化 ============
let peaksCheckTimer = null

async function initWavesurfer() {
  if (!waveformRef.value) return

  try {
    const WaveSurfer = (await import('wavesurfer.js')).default
    const RegionsPlugin = (await import('wavesurfer.js/dist/plugins/regions.js')).default
    const TimelinePlugin = (await import('wavesurfer.js/dist/plugins/timeline.js')).default

    regionsPluginRef.value = RegionsPlugin.create()

    // TimelinePlugin 嵌入波形顶部，自动跟随滚动
    const timelinePlugin = TimelinePlugin.create({
      height: 20,
      insertPosition: 'beforebegin', // 插入到波形容器之前（顶部）
      timeInterval: 0.5,
      primaryLabelInterval: 10,
      secondaryLabelInterval: 5,
      primaryColor: 'var(--af-text-muted)',
      secondaryColor: 'var(--af-border-default)',
      primaryFontColor: 'var(--af-text-secondary)',
      secondaryFontColor: 'var(--af-text-muted)',
      style: { fontSize: '10px', fontFamily: 'var(--af-font-mono)' },
    })

    const containerWidth = containerRef.value?.offsetWidth || 800
    const estimatedDuration = projectStore.meta.duration || 60
    const { basePxPerSec, suggestedZoom, barConfig } = calculateWaveformConfig(
      estimatedDuration,
      containerWidth
    )

    wavesurferRef.value = WaveSurfer.create({
      container: waveformRef.value,
      waveColor: props.waveColor,
      progressColor: props.progressColor,
      cursorColor: props.cursorColor,
      cursorWidth: 2,
      height: props.height,
      normalize: true,
      backend: 'MediaElement',
      plugins: [regionsPluginRef.value, timelinePlugin],
      minPxPerSec: basePxPerSec,
      scrollParent: true,
      fillParent: false,
      dragToSeek: false,
      interact: false,
      autoScroll: false,
      autoCenter: false,
      hideScrollbar: true,
      ...barConfig,
      media: document.createElement('audio'),
    })

    // 有视频源时由 VideoStage 提供声音；纯音频模式下由 WaveSurfer 输出声音。
    wavesurferRef.value.setMuted(hasVideoSource.value)
    zoomLevel.value = suggestedZoom

    setupWavesurferEvents()
    setupRegionEvents(wavesurferRef)
    await loadAudioData()
  } catch (error) {
    console.error('初始化波形失败:', error)
    hasError.value = false
    isLoading.value = true
    startPeaksPolling()
  }
}

function setupWavesurferEvents() {
  const ws = wavesurferRef.value
  if (!ws) return

  ws.on('ready', () => {
    isLoading.value = false
    isReady.value = true
    retryCount.value = 0

    // 防止 WaveSurfer audio 元素响应空格键
    const audioElement = ws.getMediaElement()
    if (audioElement) {
      audioElement.setAttribute('tabindex', '-1')
      audioElement.addEventListener('keydown', (e) => {
        if (e.code === 'Space') {
          e.preventDefault()
          e.stopPropagation()
        }
      })
    }
    applyWaveformMediaState()

    // 根据实际时长重新调整配置
    const actualDuration = ws.getDuration()
    const containerWidth = containerRef.value?.offsetWidth || 800
    if (actualDuration > 0) {
      // 纯音频场景下无视频元数据，需以波形真实时长作为全局时间上限。
      if (!projectStore.meta.duration || Math.abs(projectStore.meta.duration - actualDuration) > 0.01) {
        projectStore.setProjectDuration(actualDuration)
      }

      const { basePxPerSec, suggestedZoom, barConfig } = calculateWaveformConfig(
        actualDuration,
        containerWidth
      )
      zoomLevel.value = suggestedZoom
      ws.zoom(basePxPerSec * (suggestedZoom / 100))
      ws.setOptions(barConfig)
    }

    renderSubtitleRegions()
    emit('ready')
    playbackManager.registerWaveSurfer(ws)

    nextTick(() => {
      updateScrollbarThumb()
      const wrapper = ws.getWrapper()
      const scrollContainer = wrapper?.parentElement
      if (scrollContainer) {
        scrollContainer.addEventListener('scroll', updateScrollbarThumb)
      }
    })
  })

  ws.on('zoom', (minPxPerSec) => {
    const newZoom = Math.round((minPxPerSec / ZOOM_BASE_PX_PER_SEC) * 100)
    zoomLevel.value = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, newZoom))
    emit('zoom', zoomLevel.value)
    nextTick(() => updateScrollbarThumb())
  })

  ws.on('error', (error) => {
    console.error('Wavesurfer error:', error)
    hasError.value = false
    isLoading.value = true
    if (retryCount.value < maxRetries) {
      retryCount.value++
      setTimeout(() => loadAudioData(), 1000)
    } else {
      startPeaksPolling()
    }
  })
}

function applyWaveformMediaState() {
  const ws = wavesurferRef.value
  if (!ws) return

  const targetRate = Number(playbackStore.playbackRate) || 1
  const rawVolume = Number(playbackStore.volume)
  const targetVolume = Math.max(0, Math.min(1, Number.isFinite(rawVolume) ? rawVolume : 1))

  if (typeof ws.setPlaybackRate === 'function') {
    ws.setPlaybackRate(targetRate)
  }

  const mediaElement = ws.getMediaElement?.()
  if (mediaElement) {
    mediaElement.playbackRate = targetRate
    mediaElement.volume = targetVolume
  }
}

async function loadAudioData() {
  if (!audioSource.value) {
    isLoading.value = false
    return
  }

  try {
    if (peaksSource.value) {
      const response = await fetch(peaksSource.value)
      if (response.ok) {
        const data = await response.json()
        wavesurferRef.value.load(audioSource.value, data.peaks, data.duration)
        return
      }
    }
    wavesurferRef.value.load(audioSource.value)
  } catch (error) {
    console.error('加载音频失败:', error)
    wavesurferRef.value.load(audioSource.value)
  }
}

function startPeaksPolling() {
  if (peaksCheckTimer) return
  peaksCheckTimer = setInterval(async () => {
    try {
      if (peaksSource.value) {
        const response = await fetch(peaksSource.value)
        if (response.ok) {
          stopPeaksPolling()
          retryCount.value = 0
          hasError.value = false
          isLoading.value = true
          await loadAudioData()
        }
      }
    } catch {
      // 继续等待
    }
  }, 2000)
}

function stopPeaksPolling() {
  if (peaksCheckTimer) {
    clearInterval(peaksCheckTimer)
    peaksCheckTimer = null
  }
}

function retryLoad() {
  hasError.value = false
  errorMessage.value = ''
  isLoading.value = true
  retryCount.value = 0
  loadAudioData()
}

// ============ 滚轮缩放 ============
let zoomRafId = null
let pendingZoomDelta = 0

function smoothZoom() {
  if (pendingZoomDelta === 0) {
    zoomRafId = null
    return
  }
  const newZoom = zoomLevel.value + pendingZoomDelta
  pendingZoomDelta = 0
  handleZoomWithSmartAnchor(newZoom)
  zoomRafId = null
}

function handleWheel(e) {
  if (!e.ctrlKey) return
  e.preventDefault()
  const delta = e.deltaY < 0 ? ZOOM_WHEEL_STEP : -ZOOM_WHEEL_STEP
  pendingZoomDelta += delta
  if (!zoomRafId) {
    zoomRafId = requestAnimationFrame(smoothZoom)
  }
}

// ============ Watchers ============
let regionUpdateTimer = null
let lastSyncTime = 0

watch(
  () => projectStore.subtitles,
  () => {
    if (isReady.value && !isUpdatingRegions.value) {
      clearTimeout(regionUpdateTimer)
      regionUpdateTimer = setTimeout(() => renderSubtitleRegions(), 100)
    } else if (!isReady.value) {
      setTimeout(() => {
        if (isReady.value && projectStore.subtitles.length > 0) {
          renderSubtitleRegions()
        }
      }, 500)
    }
  },
  { deep: true }
)

watch(
  () => playbackStore.isPlaying,
  (playing) => {
    const ws = wavesurferRef.value
    if (!ws || !isReady.value) return
    if (playing) {
      ws.play()
      startSmartFollow()
    } else {
      ws.pause()
      stopSmartFollow()
    }
  }
)

watch(
  () => playbackStore.currentTime,
  (newTime) => {
    const ws = wavesurferRef.value
    if (!ws || !isReady.value) return

    const isPlaying = playbackStore.isPlaying
    const isSeeking = playbackManager.isLocked()
    if (isPlaying && !isSeeking) return

    const now = Date.now()
    if (now - lastSyncTime < 50) return
    lastSyncTime = now

    const currentWsTime = ws.getCurrentTime()
    const timeDiff = Math.abs(currentWsTime - newTime)
    if (timeDiff > 0.1) {
      const wsDuration = ws.getDuration()
      if (wsDuration > 0) ws.seekTo(newTime / wsDuration)
    }
  }
)

watch(
  () => subtitleDocumentStore.selectedSubtitleId,
  () => {
    if (isReady.value) renderSubtitleRegions()
  }
)

watch(hasVideoSource, (nextHasVideo) => {
  const ws = wavesurferRef.value
  if (!ws) return
  ws.setMuted(nextHasVideo)
  applyWaveformMediaState()
})

watch(
  () => playbackStore.playbackRate,
  () => {
    if (!isReady.value) return
    applyWaveformMediaState()
  }
)

watch(
  () => playbackStore.volume,
  () => {
    if (!isReady.value) return
    applyWaveformMediaState()
  }
)

// ============ 生命周期 ============
onMounted(async () => {
  await nextTick()
  setupRegionPointerGuards(waveformRef)
  await initWavesurfer()
  containerRef.value?.addEventListener('wheel', handleWheel, { passive: false })
})

onUnmounted(() => {
  containerRef.value?.removeEventListener('wheel', handleWheel)
  if (zoomRafId) cancelAnimationFrame(zoomRafId)
  clearTimeout(regionUpdateTimer)
  stopPeaksPolling()

  // 清理所有 composables
  cleanupZoom()
  cleanupScroll()
  cleanupCursorDrag()
  cleanupRegions()
  teardownRegionPointerGuards()

  playbackManager.unregisterWaveSurfer()

  if (wavesurferRef.value) {
    wavesurferRef.value.destroy()
    wavesurferRef.value = null
  }
})
</script>

<style scoped>
/* WaveformTimeline 组件样式 - 已重构为纯 CSS */

.waveform-timeline {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--af-bg-secondary);
  overflow: hidden;
}

/* 错误状态 */
.waveform-error {
  position: absolute;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  gap: 12px;
  background: var(--af-bg-secondary);
  color: var(--af-text-muted);
  inset: 0;
}

.waveform-error svg {
  width: 40px;
  height: 40px;
  color: var(--af-accent-danger);
}

.waveform-error button {
  padding: 6px 16px;
  background: var(--af-accent-primary);
  border-radius: var(--af-radius-md);
  color: var(--af-text-on-dark);
  font-size: 13px;
}

.waveform-error button:hover {
  background: var(--af-accent-primary-hover);
}

/* 波形容器 */
.waveform-wrapper {
  position: relative;
  flex: 1;
  min-height: 80px;
  overflow: auto hidden;
  -ms-overflow-style: none;
  scrollbar-width: none;
}

.waveform-wrapper::-webkit-scrollbar {
  display: none;
}

/* 上半部分交互遮罩层 */
.waveform-wrapper .waveform-upper-zone {
  position: absolute;
  top: 0;
  right: 0;
  left: 0;
  z-index: 10;
  height: 50%;
  user-select: none;
  touch-action: none;
}

.waveform-wrapper .waveform-upper-zone.is-region-dragging {
  pointer-events: none;
  cursor: inherit;
}

.waveform-wrapper #waveform {
  height: 100%;
}

/* WaveSurfer 样式覆盖 */
.waveform-wrapper #waveform :deep(.wavesurfer-cursor) {
  filter: drop-shadow(0 0 1px rgb(0 0 0 / 80%)) drop-shadow(0 0 2px rgb(255 255 255 / 50%))
    drop-shadow(0 0 4px rgb(var(--af-accent-danger-rgb) / 60%));
}

.waveform-wrapper #waveform :deep(> div) {
  -ms-overflow-style: none;
  scrollbar-width: none;
}

.waveform-wrapper #waveform :deep(> div)::-webkit-scrollbar {
  display: none;
}

.waveform-wrapper #waveform :deep(.wavesurfer-region) {
  border-radius: 2px;
  transition: background-color 0.2s;
  user-select: none;
  touch-action: none;
}

.waveform-wrapper #waveform :deep(.wavesurfer-region):hover {
  background-color: rgb(var(--af-accent-primary-rgb) / 40%) !important;
}

.waveform-wrapper #waveform :deep(.wavesurfer-handle) {
  width: 4px !important;
  background: var(--af-accent-primary) !important;
  border-radius: 2px;
  box-sizing: content-box;
  touch-action: none;
}

/* 加载状态 */
.waveform-loading {
  position: absolute;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  gap: 12px;
  background: var(--af-bg-secondary);
  color: var(--af-text-muted);
  inset: 0;
}

.waveform-loading .loading-spinner {
  width: 32px;
  height: 32px;
  border: 3px solid var(--af-border-default);
  border-top-color: var(--af-accent-primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
