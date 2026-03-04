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
import { mediaApi } from '@/services/api'
import ContextMenu from '@/components/editor/ContextMenu.vue'
import WaveformHeader from './WaveformHeader.vue'
import WaveformScrollbar from './WaveformScrollbar.vue'
import { logWaveformDragDiagnostics } from '@/composables/waveformDragDiagnostics.js'
import {
  createRuntimeHealthSampler,
  recordRuntimeHealthCounter,
} from '@/composables/runtimeHealthDiagnostics.js'
import {
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
  regionMinLength: { type: Number, default: 0.05 },
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
const onSubtitleEdit = subtitleDocumentStore.onSubtitleEdit

// 编辑器上下文
const editorContext = inject('editorContext', {
  isMediaReady: computed(() => true),
  isVideoReady: computed(() => true),
  hasVideoSource: computed(() => true),
  hideTimelineScale: ref(false),
})
const isMediaReady = computed(
  () => editorContext.isMediaReady?.value ?? editorContext.isVideoReady?.value ?? true
)
const hideTimelineScale = computed(
  () => editorContext.hideTimelineScale?.value ?? false
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
const timelinePluginRef = ref(null)

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

function normalizeRegionTimeForSignature(time) {
  const value = Number(time)
  if (!Number.isFinite(value)) return '0.000'
  return value.toFixed(3)
}

const regionRenderSignature = computed(() => {
  const selectedId = subtitleDocumentStore.selectedSubtitleId ?? ''
  const subtitlesSnapshot = projectStore.subtitles
    .map(
      (subtitle) =>
        `${subtitle.id}:${normalizeRegionTimeForSignature(subtitle.start)}:${normalizeRegionTimeForSignature(subtitle.end)}`
    )
    .join('|')
  return `${subtitlesSnapshot}#selected:${selectedId}#color:${props.regionColor}`
})

// 滚动条轨道 ref（从子组件获取）
const scrollbarTrackRef = computed(() => scrollbarRef.value?.trackRef)
const silentAudioUrl = ref(null)
const mediaCapability = ref('unknown')
const shouldPollPeaks = ref(false)
const isVirtualTimelineMode = ref(false)
const isTimelineInteractive = computed(() => isMediaReady.value || isVirtualTimelineMode.value)
let virtualClockRafId = null
let virtualClockLastTs = 0
let mediaKeydownElement = null
let scrollListenerElement = null
let stopRuntimeHealthSampler = null

function handleMediaElementKeydown(e) {
  if (e.code === 'Space') {
    e.preventDefault()
    e.stopPropagation()
  }
}

function detachWaveformDomListeners() {
  if (mediaKeydownElement) {
    mediaKeydownElement.removeEventListener('keydown', handleMediaElementKeydown)
    mediaKeydownElement = null
    recordRuntimeHealthCounter('waveform.media_keydown.unbind')
  }
  if (scrollListenerElement) {
    scrollListenerElement.removeEventListener('scroll', updateScrollbarThumb)
    scrollListenerElement = null
    recordRuntimeHealthCounter('waveform.scroll_listener.unbind')
  }
}

function bindWaveformDomListeners(ws) {
  detachWaveformDomListeners()

  const audioElement = ws.getMediaElement?.()
  if (audioElement) {
    audioElement.setAttribute('tabindex', '-1')
    audioElement.addEventListener('keydown', handleMediaElementKeydown)
    mediaKeydownElement = audioElement
    recordRuntimeHealthCounter('waveform.media_keydown.bind')
  }

  const wrapper = ws.getWrapper?.()
  const scrollContainer = wrapper?.parentElement
  if (scrollContainer) {
    scrollContainer.addEventListener('scroll', updateScrollbarThumb)
    scrollListenerElement = scrollContainer
    recordRuntimeHealthCounter('waveform.scroll_listener.bind')
  }
}

function resolveFallbackDuration() {
  const subtitleMaxEnd = projectStore.subtitles.reduce((maxEnd, subtitle) => {
    const end = Number(subtitle?.end)
    return Number.isFinite(end) ? Math.max(maxEnd, end) : maxEnd
  }, 0)
  const metaDuration = Number(projectStore.meta.duration) || 0
  return Math.max(metaDuration, subtitleMaxEnd, 1)
}

function canUseFallbackDuration() {
  // 无视频或音频时钟不可用时，必须使用字幕/项目时长兜底，避免 0/1 秒时钟污染 seek。
  return (
    isVirtualTimelineMode.value
    || !hasVideoSource.value
    || mediaCapability.value !== 'audio_ready'
  )
}

function resolveWaveformDuration(ws) {
  const wsDuration = Number(ws?.getDuration?.()) || 0
  if (!canUseFallbackDuration()) return wsDuration
  return Math.max(wsDuration, resolveFallbackDuration())
}

function buildFlatPeaks(targetDuration) {
  const sampleCount = Math.max(4000, Math.min(Math.ceil(targetDuration * 20), 100000))
  return new Array(sampleCount * 2).fill(0)
}

// V3.2.4+dev.20260228.01: 生成与 targetDuration 匹配的静音 WAV
// 旧实现固定 1 秒，导致 WaveSurfer getDuration() 返回 1，Region 定位/Seek 全部错乱
function buildSilentAudioUrl(targetDuration) {
  // 释放旧 URL 避免内存泄漏
  if (silentAudioUrl.value) {
    URL.revokeObjectURL(silentAudioUrl.value)
    silentAudioUrl.value = null
  }

  const ceilDuration = Math.max(1, Math.ceil(targetDuration))
  // V3.2.4+dev.20260228.02: 固定 8000Hz（浏览器 MediaElement 普遍支持的最低标准采样率）。
  // 原自适应方案在时长 >30s 时会降到 <8000Hz，导致 Chromium FFmpegDemuxer 拒绝解码
  // （DEMUXER_ERROR_NO_SUPPORTED_STREAMS）。
  // 内存开销：16-bit mono → duration_sec * 16KB/s（30 分钟 ≈ 28.8MB，桌面端可接受）。
  const sampleRate = 8000
  const frameCount = ceilDuration * sampleRate
  const dataSize = frameCount * 2
  const buffer = new ArrayBuffer(44 + dataSize)
  const view = new DataView(buffer)

  const writeString = (offset, value) => {
    for (let i = 0; i < value.length; i++) {
      view.setUint8(offset + i, value.charCodeAt(i))
    }
  }

  writeString(0, 'RIFF')
  view.setUint32(4, 36 + dataSize, true)
  writeString(8, 'WAVE')
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeString(36, 'data')
  view.setUint32(40, dataSize, true)
  // 所有采样保持 0（静音）

  const blob = new Blob([buffer], { type: 'audio/wav' })
  silentAudioUrl.value = URL.createObjectURL(blob)
  return silentAudioUrl.value
}

function setVirtualTimelineMode(enabled) {
  const nextEnabled = Boolean(enabled)
  if (isVirtualTimelineMode.value === nextEnabled) return
  isVirtualTimelineMode.value = nextEnabled
  // V3.2.4+dev.20260228.01: 通知 PlaybackManager 屏蔽/恢复 WaveSurfer 时间事件
  playbackManager.setVirtualTimelineActive(nextEnabled)
  if (!nextEnabled) {
    stopVirtualClock()
  }
}

function stopVirtualClock() {
  if (virtualClockRafId !== null) {
    cancelAnimationFrame(virtualClockRafId)
    virtualClockRafId = null
  }
  virtualClockLastTs = 0
}

function runVirtualClockFrame(timestamp) {
  if (!isVirtualTimelineMode.value || !playbackStore.isPlaying) {
    stopVirtualClock()
    return
  }

  if (!virtualClockLastTs) {
    virtualClockLastTs = timestamp
  }

  const deltaSec = Math.max(0, (timestamp - virtualClockLastTs) / 1000)
  virtualClockLastTs = timestamp

  const playbackRate = Number(playbackStore.playbackRate) || 1
  const currentStoreTime = Number(playbackStore.currentTime) || 0
  const maxDuration = resolveFallbackDuration()
  const nextTime = Math.min(maxDuration, currentStoreTime + deltaSec * playbackRate)

  playbackStore.updateCurrentTimeRaw(nextTime)
  playbackStore.commitCurrentTime(nextTime)
  const ws = wavesurferRef.value
  if (ws && isReady.value) {
    const wsDuration = resolveWaveformDuration(ws)
    if (wsDuration > 0) {
      const progress = Math.max(0, Math.min(1, nextTime / wsDuration))
      ws.seekTo(progress)
    }
  }

  if (nextTime >= maxDuration) {
    playbackStore.setPlaying(false)
    stopVirtualClock()
    return
  }

  virtualClockRafId = requestAnimationFrame(runVirtualClockFrame)
}

function startVirtualClock() {
  if (!isVirtualTimelineMode.value || virtualClockRafId !== null) return
  virtualClockLastTs = 0
  virtualClockRafId = requestAnimationFrame(runVirtualClockFrame)
}

async function refreshMediaCapability() {
  const identityId = props.mediaId || projectStore.primaryId || ''
  const hasLocalAudioPath = Boolean(projectStore.meta.audioPath)

  if (!identityId) {
    mediaCapability.value = hasLocalAudioPath ? 'audio_ready' : 'no_media'
    shouldPollPeaks.value = false
    return mediaCapability.value
  }

  try {
    const info = await mediaApi.getMediaInfo(identityId)
    const audioState = String(info?.audio?.state || '').trim().toLowerCase()
    const audioExists = Boolean(info?.audio?.exists) || audioState === 'ready'
    const audioExtractable = Boolean(info?.audio?.extractable) || audioState === 'derivable'
    const hasPeaks = Boolean(info?.peaks?.exists)

    if (audioExists || hasLocalAudioPath) {
      mediaCapability.value = 'audio_ready'
      shouldPollPeaks.value = !hasPeaks && Boolean(peaksSource.value)
      return mediaCapability.value
    }

    if (audioExtractable) {
      mediaCapability.value = 'audio_derivable'
      shouldPollPeaks.value = true
      return mediaCapability.value
    }

    mediaCapability.value = 'no_media'
    shouldPollPeaks.value = false
    return mediaCapability.value
  } catch (error) {
    const hasPotentialAudio = hasLocalAudioPath || hasVideoSource.value
    mediaCapability.value = hasPotentialAudio ? 'audio_derivable' : 'no_media'
    shouldPollPeaks.value = mediaCapability.value === 'audio_derivable'
    return mediaCapability.value
  }
}

function loadFallbackBaseline(reason = '', options = {}) {
  const { virtualTimeline = false } = options
  const ws = wavesurferRef.value
  if (!ws) return

  const fallbackDuration = resolveFallbackDuration()
  const baselinePeaks = buildFlatPeaks(fallbackDuration)
  // V3.2.4+dev.20260228.01: 静音 WAV 时长与 fallbackDuration 匹配
  const silentUrl = buildSilentAudioUrl(fallbackDuration)

  console.warn('[WaveformTimeline] 音频不可用，启用无波形基线模式:', reason || 'unknown')
  hasError.value = false
  errorMessage.value = ''
  // 重置就绪状态，防止 region 在加载中途被渲染到旧时长坐标系
  isReady.value = false
  isLoading.value = true
  projectStore.setProjectDuration(fallbackDuration)
  setVirtualTimelineMode(virtualTimeline)
  ws.load(silentUrl, baselinePeaks, fallbackDuration)
}

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
  clearUserScrollOverride,
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
  isTimelineInteractive,
  () => {
    const ws = wavesurferRef.value
    if (!ws) return resolveFallbackDuration()
    return resolveWaveformDuration(ws)
  },
  emit
)

// Region 管理逻辑
const {
  isUpdatingRegions,
  setupRegionEvents,
  renderSubtitleRegions,
  flushPendingRegionCommits,
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
} = useWaveformContextMenu(projectStore)

// 包装右键菜单处理（需要传递额外参数）
function handleWaveformContextMenu(e) {
  onContextMenu(e, getTimeFromClientX)
}

// ============ WaveSurfer 初始化 ============
let peaksCheckTimer = null
let loadRetryTimer = null
let lateReadyRenderTimer = null

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
    timelinePluginRef.value = timelinePlugin

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

    applyWaveformMediaState()

    // 根据实际时长重新调整配置
    const actualDuration = ws.getDuration()
    const effectiveDuration = resolveWaveformDuration(ws)
    const containerWidth = containerRef.value?.offsetWidth || 800
    if (effectiveDuration > 0) {
      // 无媒体基线场景中，actualDuration 可能只有 1s（静音占位）；应以有效时长为准。
      if (!projectStore.meta.duration || Math.abs(projectStore.meta.duration - effectiveDuration) > 0.01) {
        projectStore.setProjectDuration(effectiveDuration)
      }

      const { basePxPerSec, suggestedZoom, barConfig } = calculateWaveformConfig(
        effectiveDuration,
        containerWidth
      )
      zoomLevel.value = suggestedZoom
      ws.zoom(basePxPerSec * (suggestedZoom / 100))
      ws.setOptions(barConfig)
    }

    renderSubtitleRegions()
    emit('ready')
    playbackManager.registerWaveSurfer(ws, identityRef.value)

    if (isVirtualTimelineMode.value) {
      ws.pause()
      if (playbackStore.isPlaying) {
        startVirtualClock()
      }
    }

    nextTick(() => {
      updateScrollbarThumb()
      applyTimelineVisibility(hideTimelineScale.value)
      bindWaveformDomListeners(ws)
    })
  })

  ws.on('zoom', (minPxPerSec) => {
    const newZoom = Math.round((minPxPerSec / ZOOM_BASE_PX_PER_SEC) * 100)
    zoomLevel.value = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, newZoom))
    emit('zoom', zoomLevel.value)
    nextTick(() => updateScrollbarThumb())
  })

  ws.on('error', async (error) => {
    console.error('Wavesurfer error:', error)

    // V3.2.4+dev.20260228.02: 虚拟时间轴模式下，静音占位音频解码失败是预期行为。
    // 波形数据（peaks）已通过 load() 的第二参数提供，直接标记就绪即可。
    // 不再调用 loadFallbackBaseline 以避免 error → load → error 无限循环。
    if (isVirtualTimelineMode.value) {
      console.warn('[WaveformTimeline] 虚拟时间轴模式：音频解码失败（预期），直接进入就绪状态')
      isLoading.value = false
      hasError.value = false
      if (!isReady.value) {
        isReady.value = true
        applyWaveformMediaState()
        renderSubtitleRegions()
        playbackManager.registerWaveSurfer(ws, identityRef.value)
        emit('ready')

        if (playbackStore.isPlaying) {
          startVirtualClock()
        }

        nextTick(() => {
          updateScrollbarThumb()
          applyTimelineVisibility(hideTimelineScale.value)
          bindWaveformDomListeners(ws)
        })
      }
      return
    }

    hasError.value = false
    isLoading.value = true

    await refreshMediaCapability()
    if (mediaCapability.value === 'no_media') {
      stopPeaksPolling()
      loadFallbackBaseline('wavesurfer_no_media', { virtualTimeline: true })
      return
    }

    if (retryCount.value < maxRetries) {
      retryCount.value++
      if (loadRetryTimer) {
        clearTimeout(loadRetryTimer)
        recordRuntimeHealthCounter('waveform.retry_timer.clear_before_reset')
      }
      recordRuntimeHealthCounter('waveform.retry_timer.set')
      loadRetryTimer = setTimeout(() => {
        loadRetryTimer = null
        loadAudioData()
      }, 1000)
    } else {
      // 错误兜底时仍需可编辑、可 seek、可模拟播放，统一进入虚拟时钟。
      loadFallbackBaseline('wavesurfer_error_max_retries', { virtualTimeline: true })
      if (shouldPollPeaks.value) {
        startPeaksPolling()
      }
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
  const capability = await refreshMediaCapability()

  if (capability === 'no_media') {
    stopPeaksPolling()
    loadFallbackBaseline('no_media_source', { virtualTimeline: true })
    return
  }

  if (!audioSource.value) {
    loadFallbackBaseline('no_audio_source', { virtualTimeline: true })
    return
  }

  try {
    if (peaksSource.value) {
      const response = await fetch(peaksSource.value)
      if (response.ok) {
        const data = await response.json()
        setVirtualTimelineMode(false)
        wavesurferRef.value.load(audioSource.value, data.peaks, data.duration)
        return
      }

      if (response.status === 404 && capability === 'audio_derivable') {
        shouldPollPeaks.value = true
      }
    }
    setVirtualTimelineMode(false)
    wavesurferRef.value.load(audioSource.value)
  } catch (error) {
    console.error('加载音频失败:', error)
    if (mediaCapability.value === 'no_media') {
      loadFallbackBaseline('audio_load_exception_no_media', { virtualTimeline: true })
      return
    }
    loadFallbackBaseline('audio_load_exception', { virtualTimeline: true })
  }
}

function startPeaksPolling() {
  if (peaksCheckTimer || !shouldPollPeaks.value) return
  peaksCheckTimer = setInterval(async () => {
    try {
      await refreshMediaCapability()
      if (mediaCapability.value === 'no_media' || !shouldPollPeaks.value) {
        stopPeaksPolling()
        return
      }

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
  if (loadRetryTimer) {
    clearTimeout(loadRetryTimer)
    loadRetryTimer = null
    recordRuntimeHealthCounter('waveform.retry_timer.clear_on_retry')
  }
  stopPeaksPolling()
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
let hasDeferredRegionRender = false

function clearScheduledRegionRender() {
  if (regionUpdateTimer) {
    clearTimeout(regionUpdateTimer)
    regionUpdateTimer = null
  }
}

function scheduleRegionRender(delay = 80, reason = 'unknown') {
  if (!isReady.value) return
  clearScheduledRegionRender()
  regionUpdateTimer = setTimeout(() => {
    regionUpdateTimer = null
    flushPendingRegionCommits({ force: true })
    logWaveformDragDiagnostics('timeline-region-render-commit', {
      reason,
      subtitlesCount: projectStore.subtitles.length,
    })
    renderSubtitleRegions()
    hasDeferredRegionRender = false
  }, Math.max(0, delay))
}

watch(
  () => identityRef.value,
  async (identityId, oldIdentityId) => {
    subtitleDocumentStore.bindSyncIdentity(identityId)
    playbackManager.bindSession(identityId, { force: true, resetPosition: false })

    if (identityId === oldIdentityId || !wavesurferRef.value) {
      return
    }

    stopPeaksPolling()
    stopVirtualClock()
    hasError.value = false
    errorMessage.value = ''
    retryCount.value = 0
    isLoading.value = true
    await loadAudioData()
  },
  { immediate: true }
)

watch(
  () => regionRenderSignature.value,
  () => {
    if (isRegionPointerDragging.value) {
      clearScheduledRegionRender()
      hasDeferredRegionRender = true
      logWaveformDragDiagnostics('timeline-skip-render-while-region-dragging', {
        subtitlesCount: projectStore.subtitles.length,
        reason: 'builtin-guard',
      })
      return
    }

    if (!isReady.value) {
      if (lateReadyRenderTimer) {
        clearTimeout(lateReadyRenderTimer)
        recordRuntimeHealthCounter('waveform.late_ready_timer.clear_before_reset')
      }
      recordRuntimeHealthCounter('waveform.late_ready_timer.set')
      lateReadyRenderTimer = setTimeout(() => {
        lateReadyRenderTimer = null
        if (isReady.value && projectStore.subtitles.length > 0) {
          scheduleRegionRender(0, 'late-ready')
        }
      }, 500)
      return
    }

    if (isUpdatingRegions.value) {
      hasDeferredRegionRender = true
      return
    }

    scheduleRegionRender(80, 'signature-change')
  },
  { flush: 'post' }
)

watch(
  () => isRegionPointerDragging.value,
  (isDragging, wasDragging) => {
    if (isDragging) {
      clearScheduledRegionRender()
      hasDeferredRegionRender = true
      return
    }

    if (!wasDragging) return
    if (!isReady.value) return

    flushPendingRegionCommits({ force: true })
    scheduleRegionRender(0, 'drag-end-replay')
  }
)

watch(
  () => isUpdatingRegions.value,
  (isUpdating, wasUpdating) => {
    if (isUpdating || !wasUpdating) return
    if (!hasDeferredRegionRender) return
    if (isRegionPointerDragging.value || !isReady.value) return
    scheduleRegionRender(0, 'regions-lock-released')
  }
)

watch(
  () => playbackStore.isPlaying,
  (playing) => {
    const ws = wavesurferRef.value
    if (!ws || !isReady.value) return

    if (isVirtualTimelineMode.value) {
      ws.pause()
      if (playing) {
        startVirtualClock()
        startSmartFollow()
      } else {
        stopVirtualClock()
        stopSmartFollow()
      }
      return
    }

    stopVirtualClock()
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

    // V3.2.4+dev.20260228.01: 虚拟模式播放中，时间由 RAF 直接驱动 seekTo，watch 不介入
    if (isVirtualTimelineMode.value && playbackStore.isPlaying) return

    const isPlaying = playbackStore.isPlaying
    const isSeeking = playbackManager.isLocked()
    if (!isVirtualTimelineMode.value && isPlaying && !isSeeking) return

    const now = Date.now()
    if (now - lastSyncTime < 50) return
    lastSyncTime = now

    const currentWsTime = ws.getCurrentTime()
    const timeDiff = Math.abs(currentWsTime - newTime)
    if (timeDiff > 0.1) {
      // 时间跳变说明用户 seek，清除滚动覆盖以恢复智能跟随
      clearUserScrollOverride()
      const wsDuration = resolveWaveformDuration(ws)
      if (wsDuration > 0) {
        const progress = Math.max(0, Math.min(1, newTime / wsDuration))
        ws.seekTo(progress)
      }
    }
  }
)

watch(
  () => isVirtualTimelineMode.value,
  (enabled) => {
    const ws = wavesurferRef.value
    if (!ws || !isReady.value) return

    if (enabled) {
      ws.pause()
      if (playbackStore.isPlaying) {
        startVirtualClock()
      }
      return
    }

    stopVirtualClock()
    if (playbackStore.isPlaying) {
      ws.play()
    }
  }
)

// V3.2.4+dev.20260228.01: 虚拟时间轴时长扩展监测
// 字幕异步到达（SSE/恢复）时 fallbackDuration 可能远超初始化时的值，
// 需要用匹配时长的静音音频重新加载基线，否则 Region 定位超出 WaveSurfer 内部 duration 范围
let durationReloadTimer = null
watch(
  () => isVirtualTimelineMode.value ? resolveFallbackDuration() : 0,
  (newDuration) => {
    if (newDuration <= 0) return
    const ws = wavesurferRef.value
    if (!ws || !isReady.value) return
    const currentWsDuration = Number(ws.getDuration()) || 0
    // 新时长显著超过当前 WaveSurfer 时长（>20% 或 >1s）时重新加载
    if (newDuration > currentWsDuration * 1.2 + 1) {
      clearTimeout(durationReloadTimer)
      durationReloadTimer = setTimeout(() => {
        loadFallbackBaseline('duration_expanded', { virtualTimeline: true })
      }, 300)
    }
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

// 波形刻度显隐控制
function applyTimelineVisibility(hidden) {
  const plugin = timelinePluginRef.value
  if (!plugin) return
  // timelineWrapper 是 Timeline 插件的根 DOM 元素
  const wrapper = plugin.timelineWrapper || plugin.wrapper
  if (wrapper) {
    wrapper.style.display = hidden ? 'none' : ''
  }
}

watch(hideTimelineScale, (hidden) => {
  applyTimelineVisibility(hidden)
})

// ============ 生命周期 ============
onMounted(async () => {
  await nextTick()
  setupRegionPointerGuards(waveformRef)
  await initWavesurfer()
  stopRuntimeHealthSampler = createRuntimeHealthSampler(
    'WaveformTimeline',
    () => ({
      subtitlesCount: projectStore.subtitles.length,
      isReady: Boolean(isReady.value),
      isLoading: Boolean(isLoading.value),
      hasError: Boolean(hasError.value),
      hasLoadRetryTimer: Boolean(loadRetryTimer),
      hasLateReadyRenderTimer: Boolean(lateReadyRenderTimer),
      hasPeaksPolling: Boolean(peaksCheckTimer),
      hasMediaKeydownListener: Boolean(mediaKeydownElement),
      hasScrollListener: Boolean(scrollListenerElement),
      isVirtualTimelineMode: Boolean(isVirtualTimelineMode.value),
      currentTime: Number(playbackStore.currentTime || 0),
      duration: Number(projectStore.meta.duration || 0),
    })
  )
  containerRef.value?.addEventListener('wheel', handleWheel, { passive: false })
})

onUnmounted(() => {
  containerRef.value?.removeEventListener('wheel', handleWheel)
  if (zoomRafId) cancelAnimationFrame(zoomRafId)
  if (loadRetryTimer) {
    clearTimeout(loadRetryTimer)
    loadRetryTimer = null
    recordRuntimeHealthCounter('waveform.retry_timer.clear_on_unmount')
  }
  if (lateReadyRenderTimer) {
    clearTimeout(lateReadyRenderTimer)
    lateReadyRenderTimer = null
    recordRuntimeHealthCounter('waveform.late_ready_timer.clear_on_unmount')
  }
  if (stopRuntimeHealthSampler) {
    stopRuntimeHealthSampler()
    stopRuntimeHealthSampler = null
  }
  stopVirtualClock()
  clearScheduledRegionRender()
  clearTimeout(durationReloadTimer)
  stopPeaksPolling()

  // 清理所有 composables
  cleanupZoom()
  cleanupScroll()
  cleanupCursorDrag()
  cleanupRegions()
  teardownRegionPointerGuards()
  detachWaveformDomListeners()

  playbackManager.setVirtualTimelineActive(false)
  playbackManager.unregisterWaveSurfer()

  if (wavesurferRef.value) {
    wavesurferRef.value.destroy()
    wavesurferRef.value = null
  }
  if (silentAudioUrl.value) {
    URL.revokeObjectURL(silentAudioUrl.value)
    silentAudioUrl.value = null
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
  will-change: transform;
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
