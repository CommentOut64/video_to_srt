/**
 * useWaveformZoom - 波形缩放逻辑 Composable
 *
 * 职责：智能锚点缩放、滑块/按钮/滚轮逻辑
 * 提取自 WaveformTimeline/index.vue L720-920
 */
import { ref } from 'vue'

// ============ 缩放配置常量 ============
export const ZOOM_MIN = 20 // 最小缩放 20%
export const ZOOM_MAX = 800 // 最大缩放 800%
export const ZOOM_STEP = 5 // 滑块精度 5%
export const ZOOM_BUTTON_STEP = 20 // 按钮步进 20%
export const ZOOM_WHEEL_STEP = 10 // 滚轮步进 10%
export const ZOOM_BASE_PX_PER_SEC = 50 // 100%缩放时的基准：每秒50像素
const WHEEL_PREVIEW_SETTLE_MS = 96
const BAR_CONFIG_LOW_DETAIL_MAX_PX_PER_SEC = 40
const BAR_CONFIG_HIGH_DETAIL_MIN_PX_PER_SEC = 250

/**
 * 根据像素密度动态计算柱子配置
 * @param {number} minPxPerSec - 当前每秒像素数
 * @returns {object} barConfig - { barWidth, barGap, barRadius }
 */
export function getAdaptiveBarConfig(minPxPerSec) {
  if (minPxPerSec < BAR_CONFIG_LOW_DETAIL_MAX_PX_PER_SEC) {
    // 低倍总览：略细，避免压缩视图里柱体互相挤压。
    return { barWidth: 1.25, barGap: 0.35, barRadius: 0.8 }
  }

  if (minPxPerSec > BAR_CONFIG_HIGH_DETAIL_MIN_PX_PER_SEC) {
    // 极高倍精修：只轻微加粗，避免放大时柱体形态突然跳变。
    return { barWidth: 1.8, barGap: 0.7, barRadius: 1 }
  }

  // 主工作区：覆盖绝大多数编辑缩放范围，尽量保持柱体观感稳定。
  return { barWidth: 1.5, barGap: 0.5, barRadius: 1 }
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
  const isZoomPreviewActive = ref(false)
  const previewZoomLevel = ref(null)

  // ============ DOM 缓存 ============
  let cachedWrapper = null
  let cachedScrollContainer = null
  let scrollbarUpdateTimer = null
  let lastSliderZoomTime = 0
  let previewApplyRafId = null
  let previewCommitTimer = null
  let previewAnchorClientX = null

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

  function clampZoom(targetZoom) {
    return Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, targetZoom))
  }

  function getPxPerSec(zoom) {
    return Number((((zoom / 100) * ZOOM_BASE_PX_PER_SEC)).toFixed(4))
  }

  function areBarConfigsEqual(a, b) {
    return a.barWidth === b.barWidth && a.barGap === b.barGap && a.barRadius === b.barRadius
  }

  function maybeScheduleAnimationFrame(callback) {
    if (typeof requestAnimationFrame === 'function') {
      return requestAnimationFrame(callback)
    }
    return setTimeout(() => callback(Date.now()), 16)
  }

  function maybeCancelAnimationFrame(handle) {
    if (!handle) return
    if (typeof cancelAnimationFrame === 'function') {
      cancelAnimationFrame(handle)
      return
    }
    clearTimeout(handle)
  }

  /**
   * 获取缓存的滚动容器（避免频繁 DOM 查询）
   */
  function getWrapperElement() {
    if (cachedWrapper && cachedWrapper.isConnected) {
      return cachedWrapper
    }
    const ws = wavesurferRef.value
    if (!ws) return null
    cachedWrapper = ws.getWrapper()
    return cachedWrapper
  }

  /**
   * 获取缓存的滚动容器（避免频繁 DOM 查询）
   */
  function getScrollContainer() {
    if (cachedScrollContainer && cachedScrollContainer.isConnected) {
      return cachedScrollContainer
    }
    cachedWrapper = getWrapperElement()
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

    const currentPxPerSec = getPxPerSec(zoomLevel.value)
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

    const currentPxPerSec = getPxPerSec(zoomLevel.value)
    const playheadTotalX = getCurrentTimeSec() * currentPxPerSec
    return playheadTotalX - scrollContainer.scrollLeft
  }

  /**
   * 将 clientX 转成相对于视口左侧的锚点像素
   */
  function resolveAnchorPx(clientX) {
    const scrollContainer = getScrollContainer()
    if (!scrollContainer) return 0

    if (!Number.isFinite(clientX)) {
      return scrollContainer.clientWidth / 2
    }

    const rect = scrollContainer.getBoundingClientRect?.()
    if (!rect) {
      return scrollContainer.clientWidth / 2
    }

    return Math.max(0, Math.min(scrollContainer.clientWidth, clientX - rect.left))
  }

  function clearZoomPreviewStyles() {
    const wrapper = getWrapperElement()
    if (!wrapper?.style) return

    wrapper.style.transform = ''
    wrapper.style.transformOrigin = ''
    wrapper.style.willChange = ''
  }

  function resetZoomPreviewState() {
    if (previewApplyRafId) {
      maybeCancelAnimationFrame(previewApplyRafId)
      previewApplyRafId = null
    }
    if (previewCommitTimer) {
      clearTimeout(previewCommitTimer)
      previewCommitTimer = null
    }
    previewAnchorClientX = null
    isZoomPreviewActive.value = false
    previewZoomLevel.value = null
    clearZoomPreviewStyles()
  }

  function applyZoomPreview() {
    previewApplyRafId = null

    const wrapper = getWrapperElement()
    const scrollContainer = getScrollContainer()
    const targetZoom = previewZoomLevel.value
    if (!wrapper || !scrollContainer || !Number.isFinite(targetZoom)) {
      resetZoomPreviewState()
      return
    }

    const committedZoom = zoomLevel.value
    const anchorPx = resolveAnchorPx(previewAnchorClientX)
    const anchorContentPx = scrollContainer.scrollLeft + anchorPx
    const scale = targetZoom / committedZoom

    wrapper.style.transform = `scaleX(${scale})`
    wrapper.style.transformOrigin = `${anchorContentPx}px 0`
    wrapper.style.willChange = 'transform'
    isZoomPreviewActive.value = true
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

  function commitZoom(targetZoom, anchorPx, options = {}) {
    const { deferPreviewCleanup = false } = options
    const ws = wavesurferRef.value
    if (!ws || !containerRef.value) return

    const scrollContainer = getScrollContainer()
    if (!scrollContainer) return

    const clampedZoom = clampZoom(targetZoom)
    if (clampedZoom === zoomLevel.value) {
      if (deferPreviewCleanup) {
        resetZoomPreviewState()
      }
      return
    }

    const safeAnchorPx = Math.max(0, Math.min(scrollContainer.clientWidth, anchorPx))
    const oldPxPerSec = getPxPerSec(zoomLevel.value)
    const oldScroll = scrollContainer.scrollLeft
    const anchorTime = (oldScroll + safeAnchorPx) / oldPxPerSec
    const newPxPerSec = getPxPerSec(clampedZoom)

    zoomLevel.value = clampedZoom
    ws.zoom(newPxPerSec)
    projectStore.setZoomLevel(clampedZoom)

    const oldBarConfig = getAdaptiveBarConfig(oldPxPerSec)
    const newBarConfig = getAdaptiveBarConfig(newPxPerSec)
    if (!areBarConfigsEqual(oldBarConfig, newBarConfig)) {
      ws.setOptions(newBarConfig)
    }

    const newScroll = Math.max(0, anchorTime * newPxPerSec - safeAnchorPx)
    maybeScheduleAnimationFrame(() => {
      scrollContainer.scrollLeft = newScroll
      debouncedUpdateScrollbar()
      if (deferPreviewCleanup) {
        resetZoomPreviewState()
      }
    })
  }

  /**
   * 锚点缩放核心算法
   * @param {number} targetZoom - 目标缩放比例
   * @param {number} anchorPx - 锚点相对于视口左侧的像素位置
   */
  function setZoomWithAnchor(targetZoom, anchorPx) {
    resetZoomPreviewState()
    commitZoom(targetZoom, anchorPx)
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

  function queueWheelPreviewZoom(delta, clientX) {
    const scrollContainer = getScrollContainer()
    const wrapper = getWrapperElement()
    if (!scrollContainer || !wrapper) return

    const baseZoom = previewZoomLevel.value ?? zoomLevel.value
    const targetZoom = clampZoom(baseZoom + delta)
    if (targetZoom === zoomLevel.value) {
      if (isZoomPreviewActive.value) {
        resetZoomPreviewState()
      }
      return
    }

    previewAnchorClientX = clientX
    previewZoomLevel.value = targetZoom
    isZoomPreviewActive.value = true

    if (!previewApplyRafId) {
      previewApplyRafId = maybeScheduleAnimationFrame(() => {
        applyZoomPreview()
      })
    }

    if (previewCommitTimer) {
      clearTimeout(previewCommitTimer)
    }
    previewCommitTimer = setTimeout(() => {
      previewCommitTimer = null
      const latestPreviewZoom = previewZoomLevel.value
      if (!Number.isFinite(latestPreviewZoom)) {
        resetZoomPreviewState()
        return
      }
      const anchorPx = resolveAnchorPx(previewAnchorClientX)
      commitZoom(latestPreviewZoom, anchorPx, { deferPreviewCleanup: true })
    }, WHEEL_PREVIEW_SETTLE_MS)
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
    resetZoomPreviewState()
    const clampedValue = clampZoom(value)
    if (clampedValue === zoomLevel.value) return
    const oldPxPerSec = getPxPerSec(zoomLevel.value)
    zoomLevel.value = clampedValue
    const minPxPerSec = getPxPerSec(clampedValue)
    ws.zoom(minPxPerSec)
    projectStore.setZoomLevel(clampedValue)

    const oldBarConfig = getAdaptiveBarConfig(oldPxPerSec)
    const newBarConfig = getAdaptiveBarConfig(minPxPerSec)
    if (!areBarConfigsEqual(oldBarConfig, newBarConfig)) {
      ws.setOptions(newBarConfig)
    }
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

    resetZoomPreviewState()
    const containerWidth = containerRef.value.offsetWidth - 32
    const audioDuration = ws.getDuration()

    if (audioDuration > 0) {
      const idealZoom = Math.round((containerWidth / audioDuration / ZOOM_BASE_PX_PER_SEC) * 100)
      const fitZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, idealZoom))

      setZoom(fitZoom)
    }
  }

  /**
   * 清理缓存（组件卸载时调用）
   */
  function cleanup() {
    resetZoomPreviewState()
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
    isZoomPreviewActive,
    previewZoomLevel,
    // 方法
    setZoom,
    zoomIn,
    zoomOut,
    fitToScreen,
    handleZoomInput,
    handleZoomWithSmartAnchor,
    queueWheelPreviewZoom,
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
