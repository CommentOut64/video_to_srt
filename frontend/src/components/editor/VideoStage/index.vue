<template>
  <div class="video-stage" :class="{ 'is-fullscreen': isFullscreen }">
    <!-- 视频容器 -->
    <div class="video-container" ref="containerRef" @click="handleContainerClick" @dblclick="toggleFullscreen">
      <!-- 视频转码中的占位符 -->
      <transition name="fade">
        <div v-if="showTranscodingPlaceholder" class="video-overlay transcoding-overlay">
          <div class="transcoding-spinner"></div>
          <h3>视频画面正在解码</h3>
          <p>{{ transcodingMessage }}</p>
          <div v-if="isProcessing" class="progress-bar">
            <div v-if="transcodingProgress > 0 && transcodingProgress < 100" class="progress-fill" :style="{ width: transcodingProgress + '%' }"></div>
            <div v-else class="progress-fill indeterminate"></div>
            <span v-if="transcodingProgress > 0">{{ transcodingProgress.toFixed(1) }}%</span>
            <span v-else>准备中...</span>
          </div>
        </div>
      </transition>

      <!-- Proxy 错误覆盖层（新增） -->
      <transition name="fade">
        <div v-if="showProxyError" class="video-overlay proxy-error-overlay">
          <svg class="error-icon" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
          </svg>
          <h3>视频处理失败</h3>
          <p class="error-message">{{ props.proxyError }}</p>
          <button class="retry-btn" @click="$emit('retry')">重试</button>
        </div>
      </transition>

      <!-- 纯音频模式占位（无视频时保持纯黑背景） -->
      <div v-if="showAudioOnlyPlaceholder" class="audio-only-placeholder"></div>

      <!-- HTML5 视频元素 -->
      <video
        ref="videoRef"
        v-show="hasVideoSource"
        :src="hasVideoSource ? effectiveVideoSource : null"
        :muted="muted"
        :preload="preloadStrategy"
        @loadedmetadata="onMetadataLoaded"
        @timeupdate="onTimeUpdate"
        @play="onPlay"
        @pause="onPause"
        @ended="onEnded"
        @error="onError"
        @seeking="onSeeking"
        @seeked="onSeeked"
        @waiting="isBuffering = true"
        @canplay="isBuffering = false"
        @progress="onProgress"
      />

      <!-- 字幕覆盖层 - 可拖动版本 -->
      <div
        v-if="showSubtitle && currentSubtitleText"
        ref="subtitleRef"
        class="subtitle-overlay"
        :class="{ 'is-vertical': isSubtitleVertical, 'is-dragging': isDraggingSubtitle }"
        :style="subtitleStyle"
        @mousedown="handleSubtitleMouseDown"
      >
        <!-- 左上角：方向切换按钮 -->
        <button class="subtitle-control-btn direction-btn" @click.stop="toggleSubtitleDirection" title="切换方向">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M9 4v3h5v12h3V7h5V4H9zm-6 8h3v7h3v-7h3V9H3v3z"/>
          </svg>
        </button>

        <!-- 右上角：重置按钮 -->
        <button class="subtitle-control-btn reset-btn" @click.stop="resetSubtitlePosition" title="重置位置">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z"/>
          </svg>
        </button>

        <span class="subtitle-text">{{ currentSubtitleText }}</span>
      </div>

      <!-- V3.1.2+dev.20260113.01: 分辨率标志/按钮一体化 -->
      <!-- 热区容器：始终存在以接收hover事件 -->
      <div
        class="resolution-hotspot"
        @mouseenter="handleResolutionHover(true)"
        @mouseleave="handleResolutionHover(false)"
      >
        <transition name="fade-smooth">
          <div v-show="showResolutionBadge" class="resolution-indicator">
            <div
              class="resolution-badge"
              :class="resolutionClass"
            >
              {{ currentResolutionLabel }}
            </div>
            <span v-if="isUpgrading" class="upgrade-progress">
              HD {{ Math.round(upgradeProgress) }}%
            </span>

            <!-- 升级气泡 -->
            <transition name="pop">
              <div
                v-if="showUpgradeBubble"
                class="upgrade-bubble"
                @click.stop="handleResolutionClick"
              >
                <span>升级为720P</span>
              </div>
            </transition>
          </div>
        </transition>
      </div>

      <!-- 加载指示器 -->
      <transition name="fade">
        <div v-if="isBuffering" class="video-overlay loading-overlay">
          <div class="loading-spinner"></div>
          <span>加载中...</span>
        </div>
      </transition>

      <!-- 错误提示（转码中不显示） -->
      <transition name="fade">
        <div v-if="hasError && !isUpgrading" class="video-overlay error-overlay">
          <svg class="error-icon" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
          </svg>
          <p class="error-message">{{ errorMessage }}</p>
          <button v-if="canRetry" class="retry-btn" @click="retryLoad">重试</button>
        </div>
      </transition>

      <!-- 播放/暂停状态提示（短暂显示） -->
      <transition name="pop">
        <div v-if="showStateHint" class="state-hint">
          <svg v-if="stateHintType === 'play'" viewBox="0 0 24 24" fill="currentColor">
            <path d="M8 5v14l11-7z"/>
          </svg>
          <svg v-else-if="stateHintType === 'pause'" viewBox="0 0 24 24" fill="currentColor">
            <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
          </svg>
          <svg v-else-if="stateHintType === 'volume'" viewBox="0 0 24 24" fill="currentColor">
            <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/>
          </svg>
          <span v-if="stateHintText">{{ stateHintText }}</span>
        </div>
      </transition>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted, nextTick, inject } from 'vue'
import { useProjectStore } from '@/stores/projectStore'
import { usePlaybackManager } from '@/services/PlaybackManager'
import { ProxyState } from '@/composables/useProxyVideo'

// Props
const props = defineProps({
  videoUrl: String,
  mediaId: String,
  autoPlay: { type: Boolean, default: false },
  muted: { type: Boolean, default: false },
  showSubtitle: { type: Boolean, default: true },
  enableKeyboard: { type: Boolean, default: true },
  seekStep: { type: Number, default: 5 },
  // 渐进式加载相关
  progressiveUrl: String,           // 从外部传入的渐进式 URL
  currentResolution: String,        // 当前分辨率 ('360p', '720p', 'source')
  isUpgrading: { type: Boolean, default: false },  // 是否正在升级
  upgradeProgress: { type: Number, default: 0 },    // 升级进度
  // Proxy 状态（来自 useProxyVideo）
  proxyState: { type: String, default: null },      // ProxyState 枚举值
  proxyError: { type: String, default: null },      // Proxy 错误信息
  autoTrigger720p: { type: Boolean, default: true } // V3.1.2+dev.20260113.02: 是否启用自动触发720p
})

// V3.1.2+dev.20260113.02: 添加 upgrade-started 和 upgrade-failed 事件
const emit = defineEmits(['loaded', 'error', 'play', 'pause', 'timeupdate', 'ended', 'resolution-change', 'retry', 'upgrade-started', 'upgrade-failed'])

// Store
const projectStore = useProjectStore()

// 全局播放管理器（单例）
const playbackManager = usePlaybackManager()

// 编辑器上下文（用于纯音频场景放开播放控制）
const editorContext = inject('editorContext', {
  isMediaReady: computed(() => true),
  isVideoReady: computed(() => true),
})

// Refs
const videoRef = ref(null)
const containerRef = ref(null)
const subtitleRef = ref(null)

// State
const isBuffering = ref(false)
const hasError = ref(false)
const errorMessage = ref('')
const canRetry = ref(false)
const isFullscreen = ref(false)
const retryCount = ref(0)
const maxRetries = 3
const showProgressiveHint = ref(false)
let progressiveHintTimer = null

// V3.1.2+dev.20260113.01: 分辨率标志/按钮状态
const showResolutionBadge = ref(false)
const showUpgradeBubble = ref(false)
const isHoveringResolution = ref(false)
let resolutionBadgeTimer = null
let resolutionHideTimer = null

// 状态提示（短暂显示播放/暂停图标）
const showStateHint = ref(false)
const stateHintType = ref('')
const stateHintText = ref('')
let stateHintTimer = null

// 字幕拖动相关状态
const subtitlePosition = ref({ x: 0, y: 0 })  // 字幕位置偏移（相对于默认位置）
const isSubtitleVertical = ref(false)  // 是否竖向显示
const isDraggingSubtitle = ref(false)  // 是否正在拖动
const dragStartPos = ref({ x: 0, y: 0 })  // 拖动起始位置
const dragStartSubtitlePos = ref({ x: 0, y: 0 })  // 拖动开始时的字幕位置

// Computed
const mediaId = computed(() => props.mediaId || projectStore.primaryId || null)

const videoSource = computed(() => {
  return props.videoUrl || null
})

// 标记当前是否启用了渐进式模式（Task6 编辑器始终由 proxy 状态受控）。
const isProgressiveMode = computed(() => {
  return props.progressiveUrl !== undefined || props.proxyState !== null
})

// 实际使用的视频源（支持渐进式加载）
const effectiveVideoSource = computed(() => {
  if (isProgressiveMode.value) {
    // 渐进式模式下，如果正在转码中，返回 null 避免加载不兼容的源视频
    if (isProcessing.value) {
      return null
    }
    // 当后端仍未提供 URL（通常是转码中）时不再退回原始 H265，让界面回到提示状态
    return props.progressiveUrl || null
  }
  return videoSource.value
})

const hasVideoSource = computed(() => !!effectiveVideoSource.value)

// 动态 preload 策略（根据视频时长决定）
const preloadStrategy = computed(() => {
  const duration = projectStore.meta.duration
  if (duration < 180) return 'auto'       // 3分钟内：自动预加载（避免过度预加载）
  if (duration < 1800) return 'metadata'  // 30分钟内：仅元数据
  return 'metadata'                       // 超长视频：也使用metadata而不是none，保证基本性能
})

// 分辨率标签
const currentResolutionLabel = computed(() => {
  switch (props.currentResolution) {
    case '360p': return '360P'
    case '720p': return '720P'
    case '1080p': return '1080P'
    case 'source':
      // 根据实际分辨率判断是否为1080P+
      const height = projectStore.meta.videoHeight || 0
      return height > 1080 ? '1080P+' : 'SRC'
    default: return ''
  }
})

// 分辨率样式类
const resolutionClass = computed(() => {
  const height = projectStore.meta.videoHeight || 0
  return {
    'preview': props.currentResolution === '360p',
    'hd': props.currentResolution === '720p',
    'full-hd': props.currentResolution === '1080p',
    'ultra-hd': props.currentResolution === 'source' && height > 1080,
    'source': props.currentResolution === 'source' && height <= 1080
  }
})

// V3.1.2+dev.20260113.02: 是否可以升级到720P
// 只有在360p状态、未转码中、且未启用自动触发时才允许手动升级
const canUpgrade = computed(() => {
  return props.currentResolution === '360p' && !props.isUpgrading && !props.autoTrigger720p
})

const currentSubtitleText = computed(() => projectStore.currentSubtitle?.text || '')
const isPlaying = computed(() => projectStore.player.isPlaying)

// 字幕样式（控制位置）
const subtitleStyle = computed(() => {
  return {
    transform: `translate(calc(-50% + ${subtitlePosition.value.x}px), ${subtitlePosition.value.y}px)`,
    cursor: isDraggingSubtitle.value ? 'grabbing' : 'grab'
  }
})

// 视频是否就绪（用于控件拦截）
// 只有在这些状态下，用户才能操作播放控件
const isVideoReady = computed(() => {
  // 如果传入了 proxyState，使用它来判断
  if (props.proxyState) {
    return [
      ProxyState.READY_360P,
      ProxyState.READY_720P,
      ProxyState.DIRECT_PLAY
    ].includes(props.proxyState)
  }
  // 兼容旧逻辑：有视频源且未在升级中
  return !!effectiveVideoSource.value && !props.isUpgrading
})

const isMediaReady = computed(
  () => editorContext.isMediaReady?.value ?? editorContext.isVideoReady?.value ?? isVideoReady.value
)

// 是否处于转码/处理中
const isProcessing = computed(() => {
  if (props.proxyState) {
    return [
      ProxyState.ANALYZING,
      ProxyState.TRANSCODING_360,
      ProxyState.TRANSCODING_720,
      ProxyState.REMUXING
    ].includes(props.proxyState)
  }
  return props.isUpgrading
})

// 是否显示 Proxy 错误
const showProxyError = computed(() => {
  return props.proxyState === ProxyState.ERROR && props.proxyError
})

const showAudioOnlyPlaceholder = computed(() => {
  if (hasVideoSource.value) {
    return false
  }
  if (isProcessing.value || showProxyError.value || hasError.value) {
    return false
  }
  return true
})

// 转码占位符相关
const showTranscodingPlaceholder = computed(() => {
  // 如果有 proxyState，使用新逻辑
  if (props.proxyState) {
    // 错误状态不显示转码占位符（显示错误覆盖层）
    if (props.proxyState === ProxyState.ERROR) {
      return false
    }
    // 正在处理中，显示转码占位符
    return isProcessing.value
  }
  // 兼容旧逻辑
  if (props.isUpgrading) {
    return true
  }
  return !effectiveVideoSource.value && !hasError.value
})

const transcodingMessage = computed(() => {
  // 使用 proxyState 提供更精确的消息
  if (props.proxyState) {
    switch (props.proxyState) {
      case ProxyState.ANALYZING:
        return '分析视频中...'
      case ProxyState.REMUXING:
        return '容器重封装中（极速完成）...'
      case ProxyState.TRANSCODING_360:
        return '正在生成 360p 预览...'
      case ProxyState.TRANSCODING_720:
        return '正在生成 720p 高清...'
      default:
        return '处理中...'
    }
  }
  // 兼容旧逻辑
  if (props.currentResolution === '360p') {
    return '正在生成高清视频 (720p)...'
  } else {
    return '正在优化视频以提升拖动性能...'
  }
})

const transcodingProgress = computed(() => {
  return props.upgradeProgress || 0
})

// 显示状态提示
function showHint(type, text = '') {
  stateHintType.value = type
  stateHintText.value = text
  showStateHint.value = true
  clearTimeout(stateHintTimer)
  stateHintTimer = setTimeout(() => {
    showStateHint.value = false
  }, 800)
}

// 显示分辨率提示（视频源变更时）
function showResolutionHint() {
  showProgressiveHint.value = true
  clearTimeout(progressiveHintTimer)
  progressiveHintTimer = setTimeout(() => {
    showProgressiveHint.value = false
  }, 3000)
}

// V3.1.2+dev.20260113.01: 显示分辨率标志（5秒后自动隐藏）
function showResolutionBadgeTemporarily() {
  showResolutionBadge.value = true
  clearTimeout(resolutionBadgeTimer)
  resolutionBadgeTimer = setTimeout(() => {
    // 如果没有hover，则隐藏
    if (!isHoveringResolution.value) {
      showResolutionBadge.value = false
    }
  }, 5000)
}

// V3.1.2+dev.20260113.01: 处理分辨率标志hover
function handleResolutionHover(isHovering) {
  isHoveringResolution.value = isHovering

  if (isHovering) {
    // hover时立即显示
    showResolutionBadge.value = true
    clearTimeout(resolutionBadgeTimer)
    clearTimeout(resolutionHideTimer)

    // 如果可以升级，显示气泡
    if (canUpgrade.value) {
      showUpgradeBubble.value = true
    }
  } else {
    // hover取消后延迟3秒隐藏
    showUpgradeBubble.value = false
    clearTimeout(resolutionHideTimer)
    resolutionHideTimer = setTimeout(() => {
      showResolutionBadge.value = false
    }, 3000)
  }
}

// V3.1.2+dev.20260113.01: 处理分辨率标志点击
async function handleResolutionClick() {
  if (!canUpgrade.value || !mediaId.value) return

  // 隐藏气泡
  showUpgradeBubble.value = false

  try {
    // 调用手动触发720p的API
    const response = await fetch(`/api/media/${mediaId.value}/upgrade-720p`, {
      method: 'POST'
    })

    const result = await response.json()

    if (result.success) {
      // 成功：触发全局Toast提示（由父组件处理）
      emit('upgrade-started')
    } else {
      // 失败：显示错误提示
      const message = result.message || '启动失败'
      console.error('[VideoStage] 720p转码启动失败:', message)
      emit('upgrade-failed', message)
    }
  } catch (error) {
    console.error('[VideoStage] 720p转码请求异常:', error)
    emit('upgrade-failed', '网络错误，请稍后再试')
  }
}

// 追踪当前的 play() Promise，用于避免 AbortError 导致的状态不一致
let currentPlayPromise = null

// 监听 Store 播放状态（单向：Store → Video）
watch(() => projectStore.player.isPlaying, async (playing) => {
  if (!videoRef.value || !hasVideoSource.value) return

  const video = videoRef.value
  const isPaused = video.paused

  if (playing && isPaused) {
    try {
      // 保存 play() 返回的 Promise
      currentPlayPromise = video.play()
      await currentPlayPromise
    } catch (error) {
      // AbortError 是由于 play() 被 pause() 中断，这是正常行为，不需要处理
      if (error.name === 'AbortError') {
        console.debug('[VideoStage] 播放被中断（用户快速切换播放/暂停）')
        // 不需要重置 isPlaying，因为 pause 事件处理器会处理
        return
      }
      // 其他错误才是真正的播放失败
      console.error('[VideoStage] 播放失败:', error)
      playbackManager.pause()
    } finally {
      currentPlayPromise = null
    }
  } else if (!playing && !isPaused) {
    // 在调用 pause() 之前，等待当前 play() 完成或失败
    // 这样可以避免 AbortError
    if (currentPlayPromise) {
      try {
        await currentPlayPromise
      } catch {
        // 忽略错误，我们只是等待 Promise 完成
      }
    }
    video.pause()
  }
})

// 【重要】监听 Store 时间变化：由 PlaybackManager 统一处理
// VideoStage 不需要在这里做额外的时间同步，因为 PlaybackManager.seekTo() 会直接操作 videoElement

// 【重要】监听 videoRef 变化，确保 Video 元素注册到 PlaybackManager
watch(videoRef, (video) => {
  if (video) {
    playbackManager.registerVideo(video)
  }
}, { immediate: true })

// 监听播放速度
watch(() => projectStore.player.playbackRate, (rate) => {
  if (videoRef.value) videoRef.value.playbackRate = rate
})

// 监听音量
watch(() => projectStore.player.volume, (volume) => {
  if (videoRef.value) videoRef.value.volume = volume
})

// 监听 effectiveVideoSource 变化
watch(effectiveVideoSource, (newUrl, oldUrl) => {
  if (!newUrl) {
    hasError.value = false
    errorMessage.value = ''
    canRetry.value = false
    retryCount.value = 0
  }
})

// 监听转码状态变化（刷新后恢复时清除错误状态）
watch(() => props.isUpgrading, (isUpgrading) => {
  if (isUpgrading) {
    // 正在转码时，清除错误状态，显示转码提示
    hasError.value = false
    errorMessage.value = ''
    retryCount.value = 0
  }
})

// V3.1.2+dev.20260113.01: 监听分辨率变化，触发badge显示
watch(() => props.currentResolution, (newRes, oldRes) => {
  if (newRes && newRes !== oldRes) {
    showResolutionBadgeTemporarily()
  }
})

// 监听 Proxy 状态变化（转码完成时自动加载视频）
watch(() => props.proxyState, async (newState, oldState) => {
  // 当从转码状态切换到就绪状态时，触发视频加载
  const transcodingStates = [
    ProxyState.ANALYZING,
    ProxyState.TRANSCODING_360,
    ProxyState.TRANSCODING_720,
    ProxyState.REMUXING
  ]

  const readyStates = [
    ProxyState.READY_360P,
    ProxyState.READY_720P,
    ProxyState.DIRECT_PLAY
  ]

  const wasTranscoding = transcodingStates.includes(oldState)
  const isNowReady = readyStates.includes(newState)

  if (wasTranscoding && isNowReady) {
    // 等待下一帧确保 effectiveVideoSource 已更新
    await nextTick()

    const video = videoRef.value
    if (video && effectiveVideoSource.value) {
      // 清除错误状态
      hasError.value = false
      errorMessage.value = ''
      retryCount.value = 0

      // 显示分辨率提示
      showResolutionHint()
      emit('resolution-change', props.currentResolution)

      try {
        // 强制重新加载视频
        video.load()
      } catch (error) {
        console.error('[VideoStage] 视频加载触发失败:', error)
      }
    }
  }
})

// 监听视频源变化（渐进式加载升级时）
watch(() => props.progressiveUrl, async (newUrl, oldUrl) => {
  if (newUrl && newUrl !== oldUrl) {
    const video = videoRef.value
    if (!video) {
      console.warn('[VideoStage] 视频元素不存在，跳过加载')
      return
    }

    // 保存当前播放状态
    const currentTime = video.currentTime || 0
    const wasPlaying = !video.paused
    const currentVolume = video.volume
    const currentRate = video.playbackRate

    // 清除错误状态
    hasError.value = false
    errorMessage.value = ''
    retryCount.value = 0

    // 显示分辨率提示
    showResolutionHint()
    emit('resolution-change', props.currentResolution)

    // 等待下一帧，确保src已更新
    await nextTick()

    try {
      // 强制重新加载视频
      video.load()

      // 等待元数据加载
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('加载超时')), 10000)

        const onLoaded = () => {
          clearTimeout(timeout)
          video.removeEventListener('loadedmetadata', onLoaded)
          video.removeEventListener('error', onError)
          resolve()
        }

        const onError = () => {
          clearTimeout(timeout)
          video.removeEventListener('loadedmetadata', onLoaded)
          video.removeEventListener('error', onError)
          reject(new Error('加载失败'))
        }

        video.addEventListener('loadedmetadata', onLoaded, { once: true })
        video.addEventListener('error', onError, { once: true })
      })

      // 恢复播放状态
      video.currentTime = currentTime
      video.volume = currentVolume
      video.playbackRate = currentRate

      if (wasPlaying) {
        await video.play()
      }
    } catch (error) {
      console.error('[VideoStage] 视频源切换失败:', error)
    }
  }
})

// ========== 事件处理 ==========

function onMetadataLoaded() {
  const video = videoRef.value
  projectStore.meta.duration = video.duration
  video.playbackRate = projectStore.player.playbackRate
  video.volume = projectStore.player.volume
  retryCount.value = 0
  emit('loaded', video.duration)
  if (props.autoPlay) playbackManager.togglePlay()
}

function onTimeUpdate() {
  const video = videoRef.value
  // 【重要】时间更新由 PlaybackManager 内部通过事件监听处理
  // 这里只负责发射事件通知外部
  emit('timeupdate', video.currentTime)
}

function onPlay() {
  showHint('play')
  emit('play')
}

function onPause() {
  showHint('pause')
  emit('pause')
}

function onEnded() {
  playbackManager.pause()
  emit('ended')
}

function onProgress() {
  // 可以计算缓冲进度
}

function onSeeking() {
  // PlaybackManager 内部处理
}

function onSeeked() {
  // PlaybackManager 内部处理
}

function onError() {
  const video = videoRef.value
  const error = video?.error

  // 无视频源或转码中：不进入错误重试流程，避免纯音频场景被误判。
  if (!hasVideoSource.value || isProcessing.value || showAudioOnlyPlaceholder.value) {
    hasError.value = false
    canRetry.value = false
    return
  }

  hasError.value = true

  if (error) {
    switch (error.code) {
      case 1: errorMessage.value = '视频加载被中止'; canRetry.value = true; break
      case 2: errorMessage.value = '网络错误'; canRetry.value = true; break
      case 3: errorMessage.value = '视频解码失败'; canRetry.value = true; break
      case 4: errorMessage.value = '视频加载失败'; canRetry.value = true; break  // 改为可重试，因为可能是转码中
      default: errorMessage.value = '未知错误'; canRetry.value = true
    }
  }

  console.error('[VideoStage] 视频加载错误:', error?.code, errorMessage.value)

  // 自动重试机制（但先检查是否是转码导致的 404）
  if (canRetry.value && retryCount.value < maxRetries) {
    retryCount.value++
    errorMessage.value = `${errorMessage.value}，正在检查视频状态...`

    // 触发父组件刷新视频状态（检查是否正在转码）
    emit('check-status')

    setTimeout(() => {
      hasError.value = false
      videoRef.value?.load()
    }, 2000)
  } else if (retryCount.value >= maxRetries) {
    console.error('[VideoStage] 达到最大重试次数')
    errorMessage.value = `${errorMessage.value}，请手动重试`
  }

  emit('error', new Error(errorMessage.value))
}

function retryLoad() {
  if (!hasVideoSource.value) {
    return
  }
  hasError.value = false
  errorMessage.value = ''
  retryCount.value = 0

  // 触发父组件刷新视频状态（检查是否正在转码）
  emit('check-status')

  // 短暂延迟后重新加载，给父组件时间更新状态
  setTimeout(() => {
    videoRef.value?.load()
  }, 500)
}

// 控制方法（带拦截）
function togglePlay() {
  // 媒体未就绪时拦截操作（纯音频场景允许播放）
  if (!isMediaReady.value) {
    console.warn('[VideoStage] 媒体未就绪，播放操作被拦截')
    return
  }
  playbackManager.togglePlay()
}

function seek(seconds) {
  // 媒体未就绪时拦截操作（纯音频场景允许跳转）
  if (!isMediaReady.value) {
    console.warn('[VideoStage] 媒体未就绪，跳转操作被拦截')
    return
  }
  const video = videoRef.value
  const baseTime = video ? video.currentTime : projectStore.player.currentTime
  const newTime = Math.max(0, baseTime + seconds)
  playbackManager.seekTo(newTime)
}

function toggleFullscreen() {
  if (!containerRef.value) return
  if (!document.fullscreenElement) {
    containerRef.value.requestFullscreen()
    isFullscreen.value = true
  } else {
    document.exitFullscreen()
    isFullscreen.value = false
  }
}

// ========== 字幕拖动相关 ==========

// 处理字幕鼠标按下事件（开始拖动）
function handleSubtitleMouseDown(e) {
  // 只有按住 Ctrl 键时才能拖动
  if (!e.ctrlKey) return

  e.preventDefault()
  e.stopPropagation()

  isDraggingSubtitle.value = true
  dragStartPos.value = { x: e.clientX, y: e.clientY }
  dragStartSubtitlePos.value = { ...subtitlePosition.value }

  // 添加全局鼠标事件监听
  document.addEventListener('mousemove', handleSubtitleMouseMove)
  document.addEventListener('mouseup', handleSubtitleMouseUp)
}

// 处理字幕鼠标移动事件(拖动中)
function handleSubtitleMouseMove(e) {
  if (!isDraggingSubtitle.value) return

  e.preventDefault()

  const deltaX = e.clientX - dragStartPos.value.x
  const deltaY = e.clientY - dragStartPos.value.y

  // 计算新位置
  let newX = dragStartSubtitlePos.value.x + deltaX
  let newY = dragStartSubtitlePos.value.y + deltaY

  // 边界限制：确保字幕不超出视频容器
  if (containerRef.value && subtitleRef.value) {
    const container = containerRef.value.getBoundingClientRect()
    const subtitle = subtitleRef.value.getBoundingClientRect()

    // 字幕默认位置是 left: 50%, bottom: 48px
    // transform: translate(calc(-50% + x), y)
    // 所以字幕中心点的实际位置是 container.width / 2 + newX

    // 计算字幕的半宽和半高
    const subtitleHalfWidth = subtitle.width / 2
    const subtitleHeight = subtitle.height

    // 计算容器的边界(相对于字幕的默认中心位置)
    const containerCenterX = container.width / 2
    const maxLeft = -containerCenterX + subtitleHalfWidth  // 左边界(贴边)
    const maxRight = containerCenterX - subtitleHalfWidth  // 右边界(贴边)

    // Y 轴边界(字幕默认 bottom: 48px)
    // newY > 0 向下移动，newY < 0 向上移动
    const maxTop = -(container.height - 48 - subtitleHeight)  // 上边界(贴顶)
    const maxBottom = 48  // 下边界(允许移动到 bottom: 0,即贴底)

    // 限制 X 轴
    newX = Math.max(maxLeft, Math.min(maxRight, newX))

    // 限制 Y 轴
    newY = Math.max(maxTop, Math.min(maxBottom, newY))
  }

  subtitlePosition.value = {
    x: newX,
    y: newY
  }
}

// 处理字幕鼠标释放事件（结束拖动）
function handleSubtitleMouseUp() {
  if (!isDraggingSubtitle.value) return

  isDraggingSubtitle.value = false

  // 移除全局鼠标事件监听
  document.removeEventListener('mousemove', handleSubtitleMouseMove)
  document.removeEventListener('mouseup', handleSubtitleMouseUp)

  // 保存位置到 localStorage
  saveSubtitlePreferences()
}

// 切换字幕方向
function toggleSubtitleDirection() {
  isSubtitleVertical.value = !isSubtitleVertical.value
  saveSubtitlePreferences()
}

// 重置字幕位置和方向
function resetSubtitlePosition() {
  subtitlePosition.value = { x: 0, y: 0 }
  isSubtitleVertical.value = false
  saveSubtitlePreferences()
}

// 保存字幕偏好设置
function saveSubtitlePreferences() {
  const preferences = {
    position: subtitlePosition.value,
    isVertical: isSubtitleVertical.value
  }
  localStorage.setItem('subtitle-preferences', JSON.stringify(preferences))
}

// 加载字幕偏好设置
function loadSubtitlePreferences() {
  try {
    const saved = localStorage.getItem('subtitle-preferences')
    if (saved) {
      const preferences = JSON.parse(saved)
      subtitlePosition.value = preferences.position || { x: 0, y: 0 }
      isSubtitleVertical.value = preferences.isVertical || false
    }
  } catch (error) {
    console.error('[VideoStage] 加载字幕偏好设置失败:', error)
  }
}

// 键盘快捷键（带拦截）
function handleKeyboard(e) {
  if (!props.enableKeyboard || !videoRef.value) return
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return

  switch (e.code) {
    case 'Space':
      e.preventDefault()
      togglePlay()  // 已内置拦截
      break
    case 'ArrowLeft':
      e.preventDefault()
      seek(-props.seekStep)  // 已内置拦截
      break
    case 'ArrowRight':
      e.preventDefault()
      seek(props.seekStep)  // 已内置拦截
      break
    case 'KeyF':
      if (!e.ctrlKey && !e.metaKey) {
        e.preventDefault()
        toggleFullscreen()
      }
      break
  }
}

// 全屏变化监听
function handleFullscreenChange() {
  isFullscreen.value = !!document.fullscreenElement
}

// 处理视频容器点击（切换播放暂停）
let clickTimer = null
function handleContainerClick(e) {
  // V3.1.1+dev.20260106.02: 修饰键点击时不触发暂停切换
  // 防止 Shift/Ctrl/Alt 等修饰键与字幕移动操作冲突
  if (e.shiftKey || e.ctrlKey || e.altKey || e.metaKey) {
    return
  }

  if (clickTimer) {
    clearTimeout(clickTimer)
    clickTimer = null
    return
  }

  clickTimer = setTimeout(() => {
    clickTimer = null
    togglePlay()
  }, 200)
}

onMounted(() => {
  document.addEventListener('keydown', handleKeyboard)
  document.addEventListener('fullscreenchange', handleFullscreenChange)

  // 加载字幕偏好设置
  loadSubtitlePreferences()

  // 【关键】注册 Video 元素到 PlaybackManager
  if (videoRef.value) {
    playbackManager.registerVideo(videoRef.value)
  }

  // V3.1.2+dev.20260113.01: 初始化时显示分辨率标志
  if (props.currentResolution) {
    showResolutionBadgeTemporarily()
  }
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeyboard)
  document.removeEventListener('fullscreenchange', handleFullscreenChange)
  clearTimeout(stateHintTimer)
  clearTimeout(progressiveHintTimer)
  if (clickTimer) clearTimeout(clickTimer)

  // V3.1.2+dev.20260113.01: 清理分辨率标志相关定时器
  clearTimeout(resolutionBadgeTimer)
  clearTimeout(resolutionHideTimer)

  // 清理字幕拖动事件监听器
  document.removeEventListener('mousemove', handleSubtitleMouseMove)
  document.removeEventListener('mouseup', handleSubtitleMouseUp)

  // 【关键】注销 Video 元素
  playbackManager.unregisterVideo()
})
</script>

<style scoped>
.video-stage {
  width: 100%;
  height: 100%;
  background: var(--af-bg-primary);
  overflow: hidden;
}

.video-container {
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
  background: var(--af-video-bg);
  cursor: pointer;
}

.video-container video {
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: auto;
}

.audio-only-placeholder {
  position: absolute;
  inset: 0;
  z-index: 0;
  background: #000;
}

/* 字幕覆盖层 - 可拖动版本 */
.subtitle-overlay {
  position: absolute;
  bottom: 48px;
  left: 50%;

  /* transform 由 subtitleStyle 计算属性控制 */
  max-width: 80%;
  z-index: 10;
  pointer-events: auto;
  user-select: none;
  transition: opacity 0.2s;
}

/* 状态提示（短暂显示播放/暂停图标） */
.state-hint {
  position: absolute;
  top: 50%;
  left: 50%;
  z-index: 30;
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 8px;
  padding: 16px 24px;
  background: var(--af-functional-video-overlay-hint);
  border-radius: var(--af-radius-lg);
  color: var(--af-text-primary);
  transform: translate(-50%, -50%);
  pointer-events: none;
}

.state-hint svg {
  width: 32px;
  height: 32px;
}

.state-hint span {
  font-size: 16px;
  font-weight: 500;
}

/* 控制按钮容器（默认隐藏） */
.subtitle-overlay .subtitle-control-btn {
  position: absolute;
  top: -8px;
  display: flex;
  justify-content: center;
  align-items: center;
  width: 24px;
  height: 24px;
  background: var(--af-functional-video-control-bg);
  border: 1px solid var(--af-functional-video-control-border);
  border-radius: 50%;
  color: var(--af-functional-video-control-text);
  transition: all 0.2s;
  cursor: pointer;
  opacity: 0;
  pointer-events: auto;
}

.subtitle-overlay .subtitle-control-btn.direction-btn {
  left: -8px;
}

.subtitle-overlay .subtitle-control-btn.reset-btn {
  right: -8px;
}

.subtitle-overlay .subtitle-control-btn svg {
  width: 14px;
  height: 14px;
}

.subtitle-overlay .subtitle-control-btn:hover {
  background: var(--af-functional-video-control-bg-hover);
  color: var(--af-text-primary);
  border-color: var(--af-functional-video-control-border-hover);
  transform: scale(1.1);
}

/* hover 时显示控制按钮 */
.subtitle-overlay:hover .subtitle-control-btn {
  opacity: 1;
}

.subtitle-overlay .subtitle-text {
  display: inline-block;
  padding: 8px 20px;
  background: var(--af-functional-video-overlay-medium);
  border-radius: var(--af-radius-sm);
  color: var(--af-functional-video-text);
  font-size: 20px;
  transition: box-shadow 0.2s;
  line-height: 1.4;
  text-align: center;
  text-shadow: var(--af-functional-video-text-shadow);
  pointer-events: none;
}

/* 拖动时的样式 */
.subtitle-overlay.is-dragging {
  opacity: 0.8;
}

.subtitle-overlay.is-dragging .subtitle-text {
  box-shadow: var(--af-functional-video-shadow-lg);
}

/* 竖向显示模式 */
.subtitle-overlay.is-vertical .subtitle-text {
  writing-mode: vertical-rl;
  text-orientation: upright;
  padding: 20px 8px;
  max-height: 60vh;
  overflow-y: auto;
}

/* 自定义滚动条 */
.subtitle-overlay.is-vertical .subtitle-text::-webkit-scrollbar {
  width: 4px;
}

.subtitle-overlay.is-vertical .subtitle-text::-webkit-scrollbar-thumb {
  background: var(--af-functional-video-scrollbar-thumb);
  border-radius: 2px;
}

/* 通用覆盖层 */
.video-overlay {
  position: absolute;
  z-index: 20;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  inset: 0;
}

/* 加载状态 */
.loading-overlay {
  gap: 12px;
  background: var(--af-functional-video-overlay);
  color: var(--af-text-secondary);
}

.loading-overlay .loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid var(--af-border-subtle);
  border-top-color: var(--af-accent-primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* 错误状态 */
.error-overlay {
  gap: 16px;
  background: var(--af-functional-video-overlay-dark);
  color: var(--af-text-secondary);
}

.error-overlay .error-icon {
  width: 48px;
  height: 48px;
  color: var(--af-accent-danger);
}

.error-overlay .error-message {
  font-size: 14px;
}

.error-overlay .retry-btn {
  padding: 8px 24px;
  background: var(--af-accent-primary);
  border-radius: var(--af-radius-md);
  color: var(--af-text-primary);
  font-size: 14px;
  transition: background var(--af-transition-fast);
}

/* Proxy 错误状态 */
.proxy-error-overlay {
  gap: 16px;
  background: var(--af-functional-video-overlay-darker);
  color: var(--af-text-secondary);
}

.proxy-error-overlay .error-icon {
  width: 48px;
  height: 48px;
  color: var(--af-accent-warning);
}

.proxy-error-overlay h3 {
  margin: 0;
  color: var(--af-text-normal);
  font-size: 18px;
  font-weight: 600;
}

.proxy-error-overlay .error-message {
  color: var(--af-text-muted);
  font-size: 14px;
  max-width: 300px;
  text-align: center;
}

.proxy-error-overlay .retry-btn {
  padding: 8px 24px;
  background: var(--af-accent-primary);
  border-radius: var(--af-radius-md);
  color: var(--af-text-primary);
  font-size: 14px;
  transition: background var(--af-transition-fast);
}

.proxy-error-overlay .retry-btn:hover {
  background: var(--af-accent-primary-hover);
}

.error-overlay .retry-btn:hover {
  background: var(--af-accent-primary-hover);
}

/* 转码中状态 */
.transcoding-overlay {
  gap: 16px;
  background: var(--af-functional-video-overlay-darker);
  color: var(--af-text-secondary);
}

.transcoding-overlay .transcoding-spinner {
  width: 48px;
  height: 48px;
  border: 4px solid var(--af-border-subtle);
  border-top-color: var(--af-accent-primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.transcoding-overlay h3 {
  margin: 0;
  color: var(--af-text-normal);
  font-size: 18px;
  font-weight: 600;
}

.transcoding-overlay p {
  margin: 0;
  color: var(--af-text-muted);
  font-size: 14px;
}

.transcoding-overlay .progress-bar {
  position: relative;
  width: 300px;
  height: 24px;
  background: var(--af-bg-tertiary);
  border-radius: var(--af-radius-md);
  overflow: hidden;
}

.transcoding-overlay .progress-bar .progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--af-accent-primary), var(--af-accent-primary-hover));
  transition: width 0.3s ease;
}

.transcoding-overlay .progress-bar .progress-fill.indeterminate {
  width: 40%;
  animation: indeterminate 1.5s infinite ease-in-out;
}

@keyframes indeterminate {
  0% {
    transform: translateX(-100%);
  }

  50% {
    transform: translateX(250%);
  }

  100% {
    transform: translateX(-100%);
  }
}

.transcoding-overlay .progress-bar span {
  position: absolute;
  top: 50%;
  left: 50%;
  color: var(--af-text-primary);
  font-size: 12px;
  font-weight: 600;
  transform: translate(-50%, -50%);
  text-shadow: var(--af-functional-video-text-shadow-sm);
}

/* V3.1.2+dev.20260113.01: 分辨率标志/按钮一体化样式 */

/* 热区容器：始终存在以接收hover事件 */
.resolution-hotspot {
  position: absolute;
  top: 16px;
  right: 16px;
  min-width: 80px;
  min-height: 40px;
  z-index: 25;
  pointer-events: auto;
}

.resolution-indicator {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 8px;
  pointer-events: auto;
}

.resolution-indicator .resolution-badge {
  padding: 4px 12px;
  border: none;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.5px;
  user-select: none;
}

.resolution-indicator .resolution-badge.preview {
  background: var(--af-functional-status-badge-warning-bg);
  color: var(--af-functional-status-badge-warning-text);
}

.resolution-indicator .resolution-badge.hd {
  background: var(--af-functional-status-badge-success-bg);
  color: var(--af-functional-status-badge-success-text);
}

.resolution-indicator .resolution-badge.full-hd {
  background: var(--af-functional-status-badge-info-bg);
  color: var(--af-functional-status-badge-info-text);
}

.resolution-indicator .resolution-badge.ultra-hd {
  background: var(--af-functional-status-badge-premium-bg);
  color: var(--af-functional-status-badge-premium-text);
}

.resolution-indicator .resolution-badge.source {
  background: var(--af-functional-status-badge-info-bg);
  color: var(--af-functional-status-badge-info-text);
}

.resolution-indicator .upgrade-progress {
  padding: 4px 10px;
  background: var(--af-functional-video-overlay-dark);
  border-radius: 4px;
  color: var(--af-accent-success);
  font-size: 11px;
  font-weight: 500;
}

/* V3.1.2+dev.20260113.01: 升级气泡样式 */
.resolution-indicator .upgrade-bubble {
  padding: 8px 14px;
  background: var(--af-functional-video-overlay-dark);
  border: 1px solid var(--af-border-subtle);
  border-radius: 6px;
  color: var(--af-text-primary);
  font-size: 12px;
  white-space: nowrap;
  cursor: pointer;
  box-shadow: var(--af-functional-video-shadow-lg);
}

.resolution-indicator .upgrade-bubble:hover {
  background: var(--af-functional-video-overlay-darker);
}

/* 全屏模式 */
.is-fullscreen .subtitle-overlay {
  bottom: 80px;
}

.is-fullscreen .subtitle-overlay .subtitle-text {
  padding: 12px 28px;
  font-size: 28px;
}

/* 动画 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

/* V3.1.2+dev.20260113.01: 平滑渐变动画（用于分辨率标志） */
.fade-smooth-enter-active {
  transition: opacity 0.3s ease;
}

.fade-smooth-leave-active {
  transition: opacity 0.5s ease;
}

.fade-smooth-enter-from,
.fade-smooth-leave-to {
  opacity: 0;
}

.pop-enter-active {
  animation: pop-in 0.3s ease;
}

.pop-leave-active {
  animation: pop-out 0.2s ease;
}

/* V3.1.2+dev.20260113.01: 气泡专用过渡（用于升级气泡） */
.resolution-indicator .pop-enter-active {
  animation: bubble-pop-in 0.2s ease;
}

.resolution-indicator .pop-leave-active {
  animation: bubble-pop-out 0.15s ease;
}

@keyframes pop-in {
  0% {
    transform: translate(-50%, -50%) scale(0.5);
    opacity: 0;
  }

  100% {
    transform: translate(-50%, -50%) scale(1);
    opacity: 1;
  }
}

@keyframes pop-out {
  0% {
    transform: translate(-50%, -50%) scale(1);
    opacity: 1;
  }

  100% {
    transform: translate(-50%, -50%) scale(1.2);
    opacity: 0;
  }
}

/* V3.1.2+dev.20260113.01: 气泡弹出动画 */
@keyframes bubble-pop-in {
  0% {
    transform: scale(0.8) translateY(-4px);
    opacity: 0;
  }

  100% {
    transform: scale(1) translateY(0);
    opacity: 1;
  }
}

@keyframes bubble-pop-out {
  0% {
    transform: scale(1) translateY(0);
    opacity: 1;
  }

  100% {
    transform: scale(0.9) translateY(-4px);
    opacity: 0;
  }
}
</style>
