<template>
  <div class="video-stage" :class="{ 'is-fullscreen': isFullscreen, 'video-not-ready': !isVideoReady }">
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

      <!-- HTML5 视频元素 -->
      <video
        ref="videoRef"
        :src="effectiveVideoSource"
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

      <!-- 渐进式加载状态指示器 -->
      <transition name="fade">
        <div v-if="showProgressiveHint" class="progressive-indicator">
          <span class="resolution-badge" :class="resolutionClass">
            {{ currentResolutionLabel }}
          </span>
          <span v-if="isUpgrading" class="upgrade-progress">
            HD {{ Math.round(upgradeProgress) }}%
          </span>
        </div>
      </transition>

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
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { useProjectStore } from '@/stores/projectStore'
import { usePlaybackManager } from '@/services/PlaybackManager'
import { ProxyState } from '@/composables/useProxyVideo'

// Props
const props = defineProps({
  videoUrl: String,
  jobId: String,
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
  proxyError: { type: String, default: null }       // Proxy 错误信息
})

const emit = defineEmits(['loaded', 'error', 'play', 'pause', 'timeupdate', 'ended', 'resolution-change', 'retry'])

// Store
const projectStore = useProjectStore()

// 全局播放管理器（单例）
const playbackManager = usePlaybackManager()

// Refs
const videoRef = ref(null)
const containerRef = ref(null)

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
const videoSource = computed(() => {
  if (props.videoUrl) return props.videoUrl
  if (props.jobId) return `/api/media/${props.jobId}/video`
  return projectStore.meta.videoPath || ''
})

// 标记当前是否启用了渐进式模式（父组件传入 progressiveUrl 即表示受控模式）
const isProgressiveMode = computed(() => props.progressiveUrl !== undefined)

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
    case '720p': return 'HD'
    case 'source': return 'SRC'
    default: return ''
  }
})

// 分辨率样式类
const resolutionClass = computed(() => {
  return {
    'preview': props.currentResolution === '360p',
    'hd': props.currentResolution === '720p',
    'source': props.currentResolution === 'source'
  }
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

// 追踪当前的 play() Promise，用于避免 AbortError 导致的状态不一致
let currentPlayPromise = null

// 监听 Store 播放状态（单向：Store → Video）
watch(() => projectStore.player.isPlaying, async (playing) => {
  if (!videoRef.value) return

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
      projectStore.player.isPlaying = false
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

// 调试：监听 effectiveVideoSource 变化
watch(effectiveVideoSource, (newUrl, oldUrl) => {
  console.log('[VideoStage] effectiveVideoSource 变化:', {
    oldUrl,
    newUrl,
    progressiveUrl: props.progressiveUrl,
    proxyState: props.proxyState,
    isProcessing: isProcessing.value,
    isProgressiveMode: isProgressiveMode.value
  })
})

// 监听转码状态变化（刷新后恢复时清除错误状态）
watch(() => props.isUpgrading, (isUpgrading) => {
  if (isUpgrading) {
    // 正在转码时，清除错误状态，显示转码提示
    console.log('[VideoStage] 检测到转码状态，清除错误提示')
    hasError.value = false
    errorMessage.value = ''
    retryCount.value = 0
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
    console.log('[VideoStage] 转码完成，自动加载视频:', {
      oldState,
      newState,
      url: effectiveVideoSource.value
    })

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
        console.log('[VideoStage] 视频加载触发成功')
      } catch (error) {
        console.error('[VideoStage] 视频加载触发失败:', error)
      }
    }
  }
})

// 监听视频源变化（渐进式加载升级时）
watch(() => props.progressiveUrl, async (newUrl, oldUrl) => {
  if (newUrl && newUrl !== oldUrl) {
    console.log('[VideoStage] 检测到视频源变更:', {
      oldUrl,
      newUrl,
      resolution: props.currentResolution
    })

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

      console.log('[VideoStage] 视频源切换成功，已恢复播放状态')
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
  projectStore.player.isPlaying = false
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

  // 如果视频正在转码中，不显示错误（显示转码占位符）
  if (isProcessing.value || !effectiveVideoSource.value) {
    console.log('[VideoStage] 视频正在转码或源为空，跳过错误提示')
    hasError.value = false
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
    console.log(`[VideoStage] 视频加载失败，将检查转码状态并重试 ${retryCount.value}/${maxRetries}`)
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
  console.log('[VideoStage] 手动重试，先刷新视频状态')
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
  // 视频未就绪时拦截操作
  if (!isVideoReady.value) {
    console.warn('[VideoStage] 视频未就绪，播放操作被拦截')
    return
  }
  playbackManager.togglePlay()
}

function seek(seconds) {
  // 视频未就绪时拦截操作
  if (!isVideoReady.value) {
    console.warn('[VideoStage] 视频未就绪，跳转操作被拦截')
    return
  }
  const video = videoRef.value
  if (!video) return
  const newTime = Math.max(0, Math.min(video.duration, video.currentTime + seconds))
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

// 处理字幕鼠标移动事件（拖动中）
function handleSubtitleMouseMove(e) {
  if (!isDraggingSubtitle.value) return

  e.preventDefault()

  const deltaX = e.clientX - dragStartPos.value.x
  const deltaY = e.clientY - dragStartPos.value.y

  subtitlePosition.value = {
    x: dragStartSubtitlePos.value.x + deltaX,
    y: dragStartSubtitlePos.value.y + deltaY
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
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeyboard)
  document.removeEventListener('fullscreenchange', handleFullscreenChange)
  clearTimeout(stateHintTimer)
  clearTimeout(progressiveHintTimer)
  if (clickTimer) clearTimeout(clickTimer)

  // 清理字幕拖动事件监听器
  document.removeEventListener('mousemove', handleSubtitleMouseMove)
  document.removeEventListener('mouseup', handleSubtitleMouseUp)

  // 【关键】注销 Video 元素
  playbackManager.unregisterVideo()
})
</script>

<style lang="scss" scoped>
.video-stage {
  width: 100%;
  height: 100%;
  background: var(--bg-base);
  border-radius: var(--radius-lg);
  overflow: hidden;

  // 视频未就绪时的样式（禁用交互提示）
  &.video-not-ready {
    .video-container {
      cursor: not-allowed;
    }
  }
}

.video-container {
  position: relative;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #000;

  video {
    max-width: 100%;
    max-height: 100%;
    width: auto;
    height: auto;
  }
}

// 字幕覆盖层 - 可拖动版本
.subtitle-overlay {
  position: absolute;
  bottom: 48px;
  left: 50%;
  // transform 由 subtitleStyle 计算属性控制
  max-width: 80%;
  z-index: 10;
  pointer-events: auto;  // 允许交互
  user-select: none;  // 禁止文本选择
  transition: opacity 0.2s;

  // 控制按钮容器（默认隐藏）
  .subtitle-control-btn {
    position: absolute;
    top: -8px;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 50%;
    color: rgba(255, 255, 255, 0.7);
    cursor: pointer;
    opacity: 0;
    transition: all 0.2s;
    pointer-events: auto;

    svg {
      width: 14px;
      height: 14px;
    }

    &:hover {
      background: rgba(0, 0, 0, 0.95);
      color: #fff;
      border-color: rgba(255, 255, 255, 0.4);
      transform: scale(1.1);
    }

    &.direction-btn {
      left: -8px;
    }

    &.reset-btn {
      right: -8px;
    }
  }

  // hover 时显示控制按钮
  &:hover .subtitle-control-btn {
    opacity: 1;
  }

  // 拖动时的样式
  &.is-dragging {
    opacity: 0.8;

    .subtitle-text {
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    }
  }

  .subtitle-text {
    display: inline-block;
    padding: 8px 20px;
    font-size: 20px;
    line-height: 1.4;
    color: #fff;
    background: rgba(0, 0, 0, 0.75);
    border-radius: var(--radius-sm);
    text-align: center;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
    transition: box-shadow 0.2s;
    pointer-events: none;  // 文本本身不响应事件
  }

  // 竖向显示模式
  &.is-vertical {
    .subtitle-text {
      writing-mode: vertical-rl;
      text-orientation: upright;
      padding: 20px 8px;
      max-height: 60vh;
      overflow-y: auto;

      // 自定义滚动条
      &::-webkit-scrollbar {
        width: 4px;
      }
      &::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 2px;
      }
    }
  }
}

// 通用覆盖层
.video-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 20;
}

// 加载状态
.loading-overlay {
  background: rgba(0, 0, 0, 0.6);
  color: var(--text-secondary);
  gap: 12px;

  .loading-spinner {
    width: 40px;
    height: 40px;
    border: 3px solid rgba(255, 255, 255, 0.1);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

// 错误状态
.error-overlay {
  background: rgba(0, 0, 0, 0.85);
  color: var(--text-secondary);
  gap: 16px;

  .error-icon {
    width: 48px;
    height: 48px;
    color: var(--danger);
  }

  .error-message {
    font-size: 14px;
  }

  .retry-btn {
    padding: 8px 24px;
    background: var(--primary);
    color: white;
    border-radius: var(--radius-md);
    font-size: 14px;
    transition: background var(--transition-fast);

    &:hover { background: var(--primary-hover); }
  }
}

// Proxy 错误状态（新增）
.proxy-error-overlay {
  background: rgba(0, 0, 0, 0.9);
  color: var(--text-secondary);
  gap: 16px;

  .error-icon {
    width: 48px;
    height: 48px;
    color: var(--warning);
  }

  h3 {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-normal);
    margin: 0;
  }

  .error-message {
    font-size: 14px;
    color: var(--text-muted);
    max-width: 300px;
    text-align: center;
  }

  .retry-btn {
    padding: 8px 24px;
    background: var(--primary);
    color: white;
    border-radius: var(--radius-md);
    font-size: 14px;
    transition: background var(--transition-fast);

    &:hover { background: var(--primary-hover); }
  }
}

// 转码中状态
.transcoding-overlay {
  background: rgba(0, 0, 0, 0.9);
  color: var(--text-secondary);
  gap: 16px;

  .transcoding-spinner {
    width: 48px;
    height: 48px;
    border: 4px solid rgba(255, 255, 255, 0.1);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  h3 {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-normal);
    margin: 0;
  }

  p {
    font-size: 14px;
    color: var(--text-muted);
    margin: 0;
  }

  .progress-bar {
    width: 300px;
    height: 24px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: var(--radius-md);
    overflow: hidden;
    position: relative;

    .progress-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--primary), var(--primary-hover));
      transition: width 0.3s ease;

      &.indeterminate {
        width: 40%;
        animation: indeterminate 1.5s infinite ease-in-out;
      }
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

    span {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      font-size: 12px;
      font-weight: 600;
      color: white;
      text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
    }
  }
}

// 状态提示（短暂显示播放/暂停图标）
.state-hint {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 16px 24px;
  background: rgba(0, 0, 0, 0.7);
  border-radius: var(--radius-lg);
  color: white;
  z-index: 30;
  pointer-events: none;

  svg {
    width: 32px;
    height: 32px;
  }

  span {
    font-size: 16px;
    font-weight: 500;
  }
}

// 渐进式加载指示器
.progressive-indicator {
  position: absolute;
  top: 16px;
  right: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
  z-index: 25;
  pointer-events: none;

  .resolution-badge {
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 600;
    border-radius: 4px;
    letter-spacing: 0.5px;

    &.preview {
      background: rgba(255, 193, 7, 0.9);
      color: #000;
    }

    &.hd {
      background: rgba(76, 175, 80, 0.9);
      color: #fff;
    }

    &.source {
      background: rgba(33, 150, 243, 0.9);
      color: #fff;
    }
  }

  .upgrade-progress {
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 500;
    background: rgba(0, 0, 0, 0.7);
    color: rgba(76, 175, 80, 1);
    border-radius: 4px;
  }
}

// 全屏模式
.is-fullscreen {
  .subtitle-overlay {
    bottom: 80px;

    .subtitle-text {
      font-size: 28px;
      padding: 12px 28px;
    }
  }
}

// 动画
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

.pop-enter-active {
  animation: pop-in 0.3s ease;
}
.pop-leave-active {
  animation: pop-out 0.2s ease;
}

@keyframes pop-in {
  0% { transform: translate(-50%, -50%) scale(0.5); opacity: 0; }
  100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
}

@keyframes pop-out {
  0% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
  100% { transform: translate(-50%, -50%) scale(1.2); opacity: 0; }
}
</style>
