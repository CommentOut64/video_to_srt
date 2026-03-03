/**
 * useWaveformScroll - 波形滚动逻辑 Composable
 *
 * 职责：智能跟随滚动、滚动条计算/拖拽/阻尼
 * 提取自 WaveformTimeline/index.vue L932-1007, L1522-1710
 */
import { ref, computed, nextTick } from 'vue'
import { ZOOM_BASE_PX_PER_SEC } from './useWaveformZoom.js'

/**
 * 波形滚动 Composable
 * @param {Ref<object>} wavesurferRef - WaveSurfer 实例引用
 * @param {Ref<HTMLElement>} scrollbarTrackRef - 滚动条轨道 DOM 引用
 * @param {Ref<number>} zoomLevel - 当前缩放级别
 * @param {object} playbackStore - 播放状态 store
 * @param {Ref<boolean>} isReady - 波形是否就绪
 */
export function useWaveformScroll(
  wavesurferRef,
  scrollbarTrackRef,
  zoomLevel,
  playbackStore,
  isReady
) {
  // ============ 滚动条状态 ============
  const scrollbarThumbLeft = ref(0)
  const scrollbarThumbWidth = ref(100)
  const isDraggingScrollbar = ref(false)

  // ============ 私有状态 ============
  let scrollbarDragStartX = 0
  let scrollbarDragStartScroll = 0
  let cachedScrollWidth = 0
  let cachedClientWidth = 0
  let cachedMaxScrollLeft = 0
  let scrollbarRafId = null
  let pendingScrollEvent = null
  let followRafId = null

  // 用户滚动覆盖：播放期间用户主动滚动时暂停跟随
  let userScrollOverride = false
  let userScrollTimeoutId = null
  const USER_SCROLL_RESUME_MS = 5000

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

  // ============ 计算属性 ============
  const scrollbarThumbStyle = computed(() => ({
    left: `${scrollbarThumbLeft.value}%`,
    width: `${scrollbarThumbWidth.value}%`,
  }))

  // ============ 滚动条方法 ============

  /**
   * 更新滚动条位置和宽度
   */
  function updateScrollbarThumb() {
    const ws = wavesurferRef.value
    if (!ws || !isReady.value) return

    const wrapper = ws.getWrapper()
    if (!wrapper) return

    const scrollContainer = wrapper.parentElement
    if (!scrollContainer) return

    const scrollWidth = wrapper.scrollWidth
    const clientWidth = scrollContainer.clientWidth
    const scrollLeft = scrollContainer.scrollLeft

    // 计算新的宽度和位置
    const newThumbWidthPercent = Math.max(5, Math.min(100, (clientWidth / scrollWidth) * 100))

    const maxScrollLeft = scrollWidth - clientWidth
    let newThumbLeftPercent = 0
    if (maxScrollLeft > 0) {
      newThumbLeftPercent = (scrollLeft / maxScrollLeft) * (100 - newThumbWidthPercent)
    }

    // 只在值真正变化时才更新（减少响应式触发）
    if (Math.abs(scrollbarThumbWidth.value - newThumbWidthPercent) > 0.1) {
      scrollbarThumbWidth.value = newThumbWidthPercent
    }

    if (Math.abs(scrollbarThumbLeft.value - newThumbLeftPercent) > 0.1) {
      scrollbarThumbLeft.value = newThumbLeftPercent
    }
  }

  /**
   * 滚动条鼠标按下事件
   */
  function handleScrollbarMouseDown(e) {
    const ws = wavesurferRef.value
    if (!ws || !isReady.value) return

    const wrapper = ws.getWrapper()
    if (!wrapper) return

    const scrollContainer = wrapper.parentElement
    if (!scrollContainer) return

    e.preventDefault()
    activateUserScrollOverride()

    // 缓存容器尺寸
    cachedScrollWidth = wrapper.scrollWidth
    cachedClientWidth = scrollContainer.clientWidth
    cachedMaxScrollLeft = cachedScrollWidth - cachedClientWidth

    const rect = scrollbarTrackRef.value.getBoundingClientRect()
    const clickX = e.clientX - rect.left
    const trackWidth = rect.width

    // 点击位置相对于 track 的百分比
    const clickPercent = clickX / trackWidth

    // 计算应该滚动到的位置
    const targetScrollLeft = clickPercent * cachedMaxScrollLeft
    scrollContainer.scrollLeft = targetScrollLeft

    updateScrollbarThumb()

    // 如果点击的是 thumb 本身，开始拖拽
    const thumbLeft = (scrollbarThumbLeft.value / 100) * trackWidth
    const thumbRight = thumbLeft + (scrollbarThumbWidth.value / 100) * trackWidth

    if (clickX >= thumbLeft && clickX <= thumbRight) {
      isDraggingScrollbar.value = true
      scrollbarDragStartX = clickX
      scrollbarDragStartScroll = scrollContainer.scrollLeft

      document.addEventListener('mousemove', handleScrollbarDragMove, true)
      document.addEventListener('mouseup', handleScrollbarDragEnd, true)
    }
  }

  /**
   * 滚动条拖拽移动（RAF 节流）
   */
  function handleScrollbarDragMove(e) {
    const ws = wavesurferRef.value
    if (!isDraggingScrollbar.value || !ws) return

    pendingScrollEvent = {
      clientX: e.clientX,
      timestamp: Date.now(),
      shiftKey: e.shiftKey,
    }

    if (!scrollbarRafId) {
      scrollbarRafId = requestAnimationFrame(processScrollbarDrag)
    }
  }

  /**
   * 处理滚动条拖拽（在 RAF 中执行）
   */
  function processScrollbarDrag() {
    const ws = wavesurferRef.value
    if (!pendingScrollEvent || !isDraggingScrollbar.value || !ws) {
      scrollbarRafId = null
      return
    }

    const wrapper = ws.getWrapper()
    if (!wrapper) {
      scrollbarRafId = null
      return
    }

    const scrollContainer = wrapper.parentElement
    if (!scrollContainer) {
      scrollbarRafId = null
      return
    }

    const rect = scrollbarTrackRef.value.getBoundingClientRect()
    const trackWidth = rect.width
    const deltaX = pendingScrollEvent.clientX - rect.left - scrollbarDragStartX
    const deltaPercent = deltaX / trackWidth

    // 动态阻尼：高缩放时适度降低灵敏度
    const dampingFactor = 1 + Math.log10(Math.max(1, zoomLevel.value / 100)) * 0.5
    let effectiveDeltaPercent = deltaPercent / dampingFactor

    // Shift 精细模式
    if (pendingScrollEvent.shiftKey) {
      effectiveDeltaPercent *= 0.25
    }

    const newScrollLeft = scrollbarDragStartScroll + effectiveDeltaPercent * cachedMaxScrollLeft
    scrollContainer.scrollLeft = Math.max(0, Math.min(newScrollLeft, cachedMaxScrollLeft))

    updateScrollbarThumb()

    pendingScrollEvent = null
    scrollbarRafId = null
  }

  /**
   * 滚动条拖拽结束
   */
  function handleScrollbarDragEnd() {
    isDraggingScrollbar.value = false

    if (scrollbarRafId) {
      cancelAnimationFrame(scrollbarRafId)
      scrollbarRafId = null
    }

    document.removeEventListener('mousemove', handleScrollbarDragMove, true)
    document.removeEventListener('mouseup', handleScrollbarDragEnd, true)

    nextTick(() => {
      updateScrollbarThumb()
    })
  }

  /**
   * 滚动条区域滚轮事件
   */
  function handleScrollbarWheel(e) {
    const ws = wavesurferRef.value
    if (!ws || !isReady.value) return

    const wrapper = ws.getWrapper()
    if (!wrapper) return

    const scrollContainer = wrapper.parentElement
    if (!scrollContainer) return

    e.preventDefault()
    activateUserScrollOverride()

    // 动态阻尼系数
    const BASE_SPEED = 2
    const dampingFactor = Math.max(1, zoomLevel.value / 100)
    const dynamicSpeed = BASE_SPEED / Math.sqrt(dampingFactor)

    const scrollAmount = e.deltaY * dynamicSpeed
    scrollContainer.scrollLeft += scrollAmount
    updateScrollbarThumb()
  }

  // ============ 用户滚动覆盖 ============

  /**
   * 激活用户滚动覆盖：暂停智能跟随，超时后自动恢复
   */
  function activateUserScrollOverride() {
    userScrollOverride = true
    if (userScrollTimeoutId) clearTimeout(userScrollTimeoutId)
    userScrollTimeoutId = setTimeout(() => {
      userScrollOverride = false
      userScrollTimeoutId = null
    }, USER_SCROLL_RESUME_MS)
  }

  /**
   * 立即清除用户滚动覆盖（seek 跳转时调用，恢复跟随）
   */
  function clearUserScrollOverride() {
    userScrollOverride = false
    if (userScrollTimeoutId) {
      clearTimeout(userScrollTimeoutId)
      userScrollTimeoutId = null
    }
  }

  // ============ 智能跟随方法 ============

  /**
   * 智能跟随滚动：90%边缘触发，翻页式滚动
   * 用户主动滚动期间跳过，避免抢夺滚动控制权
   */
  function smartScrollFollow() {
    if (userScrollOverride) return

    const ws = wavesurferRef.value
    if (!ws || !isReady.value) return

    const wrapper = ws.getWrapper()
    if (!wrapper) return

    const scrollContainer = wrapper.parentElement
    if (!scrollContainer) return

    const currentTime = getCurrentTimeSec()
    const duration = ws.getDuration()
    if (!duration) return

    const pxPerSec = (zoomLevel.value / 100) * ZOOM_BASE_PX_PER_SEC
    const cursorAbsoluteX = currentTime * pxPerSec

    const viewportWidth = scrollContainer.clientWidth
    const scrollLeft = scrollContainer.scrollLeft
    const cursorRelativeX = cursorAbsoluteX - scrollLeft

    const rightEdgeThreshold = viewportWidth * 0.9

    if (cursorRelativeX > rightEdgeThreshold) {
      const newScrollLeft = cursorAbsoluteX - viewportWidth * 0.1
      scrollContainer.scrollLeft = Math.max(0, newScrollLeft)
      updateScrollbarThumb()
    } else if (cursorAbsoluteX < scrollLeft) {
      const newScrollLeft = cursorAbsoluteX - viewportWidth * 0.1
      scrollContainer.scrollLeft = Math.max(0, newScrollLeft)
      updateScrollbarThumb()
    }
  }

  /**
   * 启动智能跟随 RAF 循环
   */
  function startSmartFollow() {
    if (followRafId) return

    const loop = () => {
      if (getIsPlaying() && isReady.value) {
        smartScrollFollow()
        followRafId = requestAnimationFrame(loop)
      } else {
        followRafId = null
      }
    }

    loop()
  }

  /**
   * 停止智能跟随
   */
  function stopSmartFollow() {
    if (followRafId) {
      cancelAnimationFrame(followRafId)
      followRafId = null
    }
  }

  /**
   * 清理资源
   */
  function cleanup() {
    stopSmartFollow()
    clearUserScrollOverride()
    if (scrollbarRafId) {
      cancelAnimationFrame(scrollbarRafId)
      scrollbarRafId = null
    }
    document.removeEventListener('mousemove', handleScrollbarDragMove, true)
    document.removeEventListener('mouseup', handleScrollbarDragEnd, true)
  }

  return {
    // 状态
    scrollbarThumbStyle,
    isDraggingScrollbar,
    // 滚动条方法
    updateScrollbarThumb,
    handleScrollbarMouseDown,
    handleScrollbarWheel,
    // 智能跟随方法
    startSmartFollow,
    stopSmartFollow,
    clearUserScrollOverride,
    // 清理
    cleanup,
  }
}
