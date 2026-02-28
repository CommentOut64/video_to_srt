/**
 * useWaveformZoom - 波形缩放逻辑 Composable
 *
 * 职责：智能锚点缩放、滑块/按钮/滚轮逻辑
 * 提取自 WaveformTimeline/index.vue L720-920
 */
import { ref, computed } from 'vue'

// ============ 缩放配置常量 ============
export const ZOOM_MIN = 20 // 最小缩放 20%
export const ZOOM_MAX = 800 // 最大缩放 800%
export const ZOOM_STEP = 5 // 滑块精度 5%
export const ZOOM_BUTTON_STEP = 20 // 按钮步进 20%
export const ZOOM_WHEEL_STEP = 10 // 滚轮步进 10%
export const ZOOM_BASE_PX_PER_SEC = 50 // 100%缩放时的基准：每秒50像素

/**
 * 根据像素密度动态计算柱子配置
 * @param {number} minPxPerSec - 当前每秒像素数
 * @returns {object} barConfig - { barWidth, barGap, barRadius }
 */
export function getAdaptiveBarConfig(minPxPerSec) {
  if (minPxPerSec >= 400) {
    // 极度放大（800%+）：较粗柱子
    return { barWidth: 3, barGap: 1.5, barRadius: 1.5 }
  } else if (minPxPerSec >= 200) {
    // 高倍放大（400%+）：中粗柱子
    return { barWidth: 2.5, barGap: 1, barRadius: 1.5 }
  } else if (minPxPerSec >= 100) {
    // 中等放大（200%+）：标准柱子
    return { barWidth: 2, barGap: 1, barRadius: 1 }
  } else if (minPxPerSec >= 50) {
    // 标准缩放（100%+）：较细柱子
    return { barWidth: 1.5, barGap: 0.5, barRadius: 1 }
  } else {
    // 缩小查看全局：细柱子
    return { barWidth: 1, barGap: 0.3, barRadius: 0.5 }
  }
}

/**
 * 根据视频时长计算合适的波形配置（初始化用）
 * @param {number} videoDuration - 视频时长（秒）
 * @param {number} containerWidth - 容器宽度（像素）
 */
export function calculateWaveformConfig(videoDuration, containerWidth) {
  const basePxPerSec = ZOOM_BASE_PX_PER_SEC

  // 计算建议的初始缩放级别（适应屏幕）
  const idealFitZoom = Math.round((containerWidth / videoDuration / basePxPerSec) * 100)
  const suggestedZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, idealFitZoom))

  // 根据初始缩放级别计算初始每秒像素数，然后动态获取柱子配置
  const initialPxPerSec = basePxPerSec * (suggestedZoom / 100)
  const barConfig = getAdaptiveBarConfig(initialPxPerSec)

  return {
    basePxPerSec,
    suggestedZoom,
    barConfig,
  }
}

/**
 * 波形缩放 Composable
 * @param {Ref<object>} wavesurferRef - WaveSurfer 实例引用
 * @param {Ref<HTMLElement>} containerRef - 容器 DOM 引用
 * @param {object} projectStore - Pinia store
 * @param {object} playbackStore - 播放状态 store
 * @param {Function} updateScrollbarThumb - 更新滚动条回调
 */
export function useWaveformZoom(
  wavesurferRef,
  containerRef,
  projectStore,
  playbackStore,
  updateScrollbarThumb
) {
  // ============ 状态 ============
  const zoomLevel = ref(100)

  // ============ DOM 缓存 ============
  let cachedWrapper = null
  let cachedScrollContainer = null
  let scrollbarUpdateTimer = null
  let lastSliderZoomTime = 0

  const SLIDER_THROTTLE_MS = 16 // ~60fps

  function readPlaybackValue(maybeRefValue) {
    if (
      maybeRefValue &&
      typeof maybeRefValue === 'object' &&
      'value' in maybeRefValue
    ) {
      return maybeRefValue.value
    }
    return maybeRefValue
  }

  function getCurrentTimeSec() {
    const value = Number(readPlaybackValue(playbackStore.currentTime))
    return Number.isFinite(value) ? value : 0
  }

  function getIsPlaying() {
    return Boolean(readPlaybackValue(playbackStore.isPlaying))
  }

  /**
   * 获取缓存的滚动容器（避免频繁 DOM 查询）
   */
  function getScrollContainer() {
    if (cachedScrollContainer && cachedScrollContainer.isConnected) {
      return cachedScrollContainer
    }
    const ws = wavesurferRef.value
    if (!ws) return null
    cachedWrapper = ws.getWrapper()
    if (!cachedWrapper) return null
    cachedScrollContainer = cachedWrapper.parentElement
    return cachedScrollContainer
  }

  /**
   * 判断播放头当前是否在可视范围内
   */
  function isPlayheadInViewport() {
    const scrollContainer = getScrollContainer()
    if (!scrollContainer) return false

    const currentPxPerSec = (zoomLevel.value / 100) * ZOOM_BASE_PX_PER_SEC
    const playheadX = getCurrentTimeSec() * currentPxPerSec
    const { scrollLeft, clientWidth } = scrollContainer

    return playheadX >= scrollLeft - 10 && playheadX <= scrollLeft + clientWidth + 10
  }

  /**
   * 获取光标相对于视口的坐标
   */
  function getPlayheadRelativeX() {
    const scrollContainer = getScrollContainer()
    if (!scrollContainer) return 0

    const currentPxPerSec = (zoomLevel.value / 100) * ZOOM_BASE_PX_PER_SEC
    const playheadTotalX = getCurrentTimeSec() * currentPxPerSec
    return playheadTotalX - scrollContainer.scrollLeft
  }

  /**
   * 滚动条更新防抖
   */
  function debouncedUpdateScrollbar() {
    if (scrollbarUpdateTimer) return
    scrollbarUpdateTimer = setTimeout(() => {
      scrollbarUpdateTimer = null
      if (updateScrollbarThumb) updateScrollbarThumb()
    }, 16)
  }

  /**
   * 锚点缩放核心算法
   * @param {number} targetZoom - 目标缩放比例
   * @param {number} anchorPx - 锚点相对于视口左侧的像素位置
   */
  function setZoomWithAnchor(targetZoom, anchorPx) {
    const ws = wavesurferRef.value
    if (!ws || !containerRef.value) return

    const scrollContainer = getScrollContainer()
    if (!scrollContainer) return

    // 1. 记录缩放前的状态
    const oldPxPerSec = (zoomLevel.value / 100) * ZOOM_BASE_PX_PER_SEC
    const oldScroll = scrollContainer.scrollLeft

    // 计算锚点对应的"绝对时间点"
    const anchorTime = (oldScroll + anchorPx) / oldPxPerSec

    // 2. 应用新的缩放
    const clampedZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, targetZoom))

    if (clampedZoom === zoomLevel.value) return

    zoomLevel.value = clampedZoom
    const newPxPerSec = (clampedZoom / 100) * ZOOM_BASE_PX_PER_SEC

    ws.zoom(newPxPerSec)
    projectStore.setZoomLevel(clampedZoom)

    // 动态柱子宽度
    const newBarConfig = getAdaptiveBarConfig(newPxPerSec)
    ws.setOptions(newBarConfig)

    // 3. 计算新的滚动位置
    const newScroll = Math.max(0, anchorTime * newPxPerSec - anchorPx)

    // 4. RAF 设置滚动位置
    requestAnimationFrame(() => {
      scrollContainer.scrollLeft = newScroll
      debouncedUpdateScrollbar()
    })
  }

  /**
   * 智能锚点缩放
   * 策略：播放中跟随光标；暂停且光标可见锚定光标；否则锚定视口中心
   */
  function handleZoomWithSmartAnchor(targetZoom) {
    const scrollContainer = getScrollContainer()
    if (!scrollContainer) return

    let anchorPx

    if (getIsPlaying()) {
      anchorPx = getPlayheadRelativeX()
    } else if (isPlayheadInViewport()) {
      anchorPx = getPlayheadRelativeX()
    } else {
      anchorPx = scrollContainer.clientWidth / 2
    }

    setZoomWithAnchor(targetZoom, anchorPx)
  }

  /**
   * 滑块输入事件处理（带节流）
   */
  function handleZoomInput(e) {
    const now = performance.now()
    if (now - lastSliderZoomTime < SLIDER_THROTTLE_MS) return
    lastSliderZoomTime = now

    const value = parseInt(e.target.value)
    handleZoomWithSmartAnchor(value)
  }

  /**
   * 直接设置缩放（供初始化、fitToScreen 使用）
   */
  function setZoom(value) {
    const ws = wavesurferRef.value
    if (!ws) return
    const clampedValue = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, value))
    zoomLevel.value = clampedValue
    const minPxPerSec = (clampedValue / 100) * ZOOM_BASE_PX_PER_SEC
    ws.zoom(minPxPerSec)
    projectStore.setZoomLevel(clampedValue)
  }

  /**
   * 放大
   */
  function zoomIn() {
    const newValue = Math.min(ZOOM_MAX, zoomLevel.value + ZOOM_BUTTON_STEP)
    handleZoomWithSmartAnchor(newValue)
  }

  /**
   * 缩小
   */
  function zoomOut() {
    const newValue = Math.max(ZOOM_MIN, zoomLevel.value - ZOOM_BUTTON_STEP)
    handleZoomWithSmartAnchor(newValue)
  }

  /**
   * 适应屏幕
   */
  function fitToScreen() {
    const ws = wavesurferRef.value
    if (!ws || !containerRef.value) return

    const containerWidth = containerRef.value.offsetWidth - 32
    const audioDuration = ws.getDuration()

    if (audioDuration > 0) {
      const idealZoom = Math.round((containerWidth / audioDuration / ZOOM_BASE_PX_PER_SEC) * 100)
      const fitZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, idealZoom))

      setZoom(fitZoom)

      const { barConfig } = calculateWaveformConfig(audioDuration, containerWidth)
      ws.setOptions(barConfig)
    }
  }

  /**
   * 清理缓存（组件卸载时调用）
   */
  function cleanup() {
    cachedWrapper = null
    cachedScrollContainer = null
    if (scrollbarUpdateTimer) {
      clearTimeout(scrollbarUpdateTimer)
      scrollbarUpdateTimer = null
    }
  }

  return {
    // 状态
    zoomLevel,
    // 方法
    setZoom,
    zoomIn,
    zoomOut,
    fitToScreen,
    handleZoomInput,
    handleZoomWithSmartAnchor,
    getScrollContainer,
    cleanup,
    // 常量
    ZOOM_MIN,
    ZOOM_MAX,
    ZOOM_STEP,
    ZOOM_BUTTON_STEP,
    ZOOM_WHEEL_STEP,
    ZOOM_BASE_PX_PER_SEC,
  }
}
