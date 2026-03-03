/**
 * useWaveformCursorDrag - 光标拖拽逻辑 Composable
 *
 * 职责：上层拖拽、指针守护、PlaybackManager 联动
 * 提取自 WaveformTimeline/index.vue L1009-1389
 */
import { ref, computed } from 'vue'
import { ZOOM_BASE_PX_PER_SEC } from './useWaveformZoom.js'
import {
  getWaveformDragDiagnosticsConfig,
  logWaveformDragDiagnostics,
} from './waveformDragDiagnostics.js'

/**
 * 光标拖拽 Composable
 * @param {Ref<object>} wavesurferRef - WaveSurfer 实例引用
 * @param {Ref<number>} zoomLevel - 当前缩放级别
 * @param {object} playbackManager - PlaybackManager 实例
 * @param {object} playbackStore - 播放状态 store
 * @param {Ref<boolean>} isReady - 波形是否就绪
 * @param {Ref<boolean>} isMediaReady - 媒体是否就绪（音频或视频可用）
 * @param {Function} resolveEffectiveDuration - 获取时间轴有效时长（秒）
 * @param {Function} emit - 事件发射函数
 */
export function useWaveformCursorDrag(
  wavesurferRef,
  zoomLevel,
  playbackManager,
  playbackStore,
  isReady,
  isMediaReady,
  resolveEffectiveDuration,
  emit
) {
  // ============ 状态 ============
  const cursorDragMode = ref('hover-only') // 'hover-only' | 'anywhere'
  const isDraggingCursor = ref(false)
  const isRegionPointerDragging = ref(false)
  const currentMouseCursor = ref('default')
  const canEditTimeline = computed(() => isMediaReady.value)

  // ============ 私有状态 ============
  let cursorPointerId = null
  let cursorPointerTarget = null
  let cursorDragGuardsAttached = false
  let regionPointerId = null
  let regionPointerTarget = null
  let regionDragGuardsAttached = false
  let regionPointerGuardEl = null
  let previousBodyUserSelect = ''
  let previousBodyWebkitSelect = ''
  let cursorMoveSampleCount = 0

  function getDiagConfig() {
    return getWaveformDragDiagnosticsConfig()
  }

  function logDiag(eventName, payload = {}) {
    logWaveformDragDiagnostics(eventName, {
      ...payload,
      cursorPointerId,
      regionPointerId,
      isDraggingCursor: isDraggingCursor.value,
      isRegionPointerDragging: isRegionPointerDragging.value,
    })
  }

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

  function syncCursorDirectly(newTime) {
    const ws = wavesurferRef.value
    const wsDuration = Number(ws?.getDuration?.()) || 0
    const fallbackDuration = Number(resolveEffectiveDuration?.() || 0)
    const duration = Math.max(wsDuration, fallbackDuration)

    if (ws && duration > 0 && typeof ws.seekTo === 'function') {
      const progress = Math.max(0, Math.min(1, newTime / duration))
      ws.seekTo(progress)
    }

    if (typeof playbackStore.updateCurrentTimeRaw === 'function') {
      playbackStore.updateCurrentTimeRaw(newTime)
    }
    if (typeof playbackStore.commitCurrentTime === 'function') {
      playbackStore.commitCurrentTime(newTime)
    }
  }

  // ============ 工具函数 ============

  /**
   * 获取鼠标点击的时间位置
   */
  function getTimeFromClientX(clientX) {
    const ws = wavesurferRef.value
    if (!ws) return 0

    const wrapper = ws.getWrapper()
    if (!wrapper) return 0

    const scrollContainer = wrapper.parentElement
    if (!scrollContainer) return 0

    const containerRect = scrollContainer.getBoundingClientRect()
    const mouseRelativeX = clientX - containerRect.left
    const absoluteX = mouseRelativeX + scrollContainer.scrollLeft

    const pxPerSec = (zoomLevel.value / 100) * ZOOM_BASE_PX_PER_SEC
    const wsDuration = Number(ws.getDuration()) || 0
    const fallbackDuration = Number(resolveEffectiveDuration?.() || 0)
    const duration = Math.max(wsDuration, fallbackDuration)

    const time = absoluteX / pxPerSec
    // 无媒体或波形尚未就绪时，duration 可能暂时为 0，不能把点击强行裁剪回 0。
    if (duration > 0) {
      return Math.max(0, Math.min(time, duration))
    }
    return Math.max(0, time)
  }

  /**
   * 获取光标当前的 X 位置（像素）
   */
  function getCursorX() {
    const ws = wavesurferRef.value
    if (!ws) return 0

    const currentTime = getCurrentTimeSec()
    const pxPerSec = (zoomLevel.value / 100) * ZOOM_BASE_PX_PER_SEC
    return currentTime * pxPerSec
  }

  /**
   * 检查鼠标是否在光标附近
   */
  function isMouseNearCursor(clientX, threshold = 10) {
    const ws = wavesurferRef.value
    if (!ws) return false

    const wrapper = ws.getWrapper()
    if (!wrapper) return false

    const scrollContainer = wrapper.parentElement
    if (!scrollContainer) return false

    const containerRect = scrollContainer.getBoundingClientRect()
    const mouseAbsoluteX = clientX - containerRect.left + scrollContainer.scrollLeft
    const cursorX = getCursorX()

    return Math.abs(mouseAbsoluteX - cursorX) <= threshold
  }

  // ============ 光标拖拽守护 ============

  function attachCursorDragGuards() {
    if (cursorDragGuardsAttached) return
    cursorDragGuardsAttached = true
    document.addEventListener('pointermove', handleCursorDragMove, true)
    document.addEventListener('pointerup', handleCursorPointerEnd, true)
    document.addEventListener('pointercancel', handleCursorPointerCancel, true)
    window.addEventListener('blur', handleCursorPointerCancel, true)
  }

  function detachCursorDragGuards() {
    if (!cursorDragGuardsAttached) return
    cursorDragGuardsAttached = false
    document.removeEventListener('pointermove', handleCursorDragMove, true)
    document.removeEventListener('pointerup', handleCursorPointerEnd, true)
    document.removeEventListener('pointercancel', handleCursorPointerCancel, true)
    window.removeEventListener('blur', handleCursorPointerCancel, true)
  }

  function handleCursorPointerEnd(e) {
    if (!isDraggingCursor.value) return
    if (
      typeof e?.pointerId === 'number' &&
      cursorPointerId !== null &&
      e.pointerId !== cursorPointerId
    ) {
      logDiag('cursor-pointerup-ignored', { eventPointerId: e.pointerId })
      return
    }
    logDiag('cursor-pointerup', { eventPointerId: e?.pointerId ?? null })
    handleCursorDragEnd()
  }

  function handleCursorPointerCancel(e) {
    if (!isDraggingCursor.value) return
    if (
      typeof e?.pointerId === 'number' &&
      cursorPointerId !== null &&
      e.pointerId !== cursorPointerId
    ) {
      logDiag('cursor-pointercancel-ignored', { eventPointerId: e.pointerId })
      return
    }
    logDiag('cursor-pointercancel', { eventPointerId: e?.pointerId ?? null })
    handleCursorDragEnd()
  }

  function handleCursorDragMove(e) {
    try {
      if (!isDraggingCursor.value) return
      if (
        typeof e.pointerId === 'number' &&
        cursorPointerId !== null &&
        e.pointerId !== cursorPointerId
      ) {
        return
      }

      const newTime = getTimeFromClientX(e.clientX)
      const diagConfig = getDiagConfig()
      const playbackManagerDragging =
        typeof playbackManager?.isDragging === 'function'
          ? playbackManager.isDragging()
          : null
      if (diagConfig.bypassPlaybackManagerDuringCursorDrag) {
        syncCursorDirectly(newTime)
      } else {
        if (playbackManagerDragging === false) {
          // 拖拽会话本身仍在进行中，若 PlaybackManager 状态意外丢失则就地恢复。
          playbackManager.startDragging('waveformCursor')
          logDiag('cursor-drag-manager-recover', {
            eventPointerId: e.pointerId,
            newTime,
          })
        }
        playbackManager.updateDragging(newTime)
      }
      emit('seek', newTime)
      cursorMoveSampleCount += 1
      if (cursorMoveSampleCount % 20 === 0) {
        logDiag('cursor-drag-move-sample', {
          eventPointerId: e.pointerId,
          sample: cursorMoveSampleCount,
          newTime,
          bypassPlaybackManager: diagConfig.bypassPlaybackManagerDuringCursorDrag,
          playbackManagerDragging,
        })
      }
    } catch (error) {
      console.error('[WaveformCursorDrag] 拖拽移动出错:', error)
      handleCursorDragEnd()
    }
  }

  function handleCursorDragEnd() {
    try {
      if (!isDraggingCursor.value) return
      isDraggingCursor.value = false

      if (
        cursorPointerTarget &&
        typeof cursorPointerId === 'number' &&
        typeof cursorPointerTarget.releasePointerCapture === 'function'
      ) {
        try {
          if (cursorPointerTarget.hasPointerCapture?.(cursorPointerId)) {
            cursorPointerTarget.releasePointerCapture(cursorPointerId)
          }
        } catch (error) {
          console.debug('[WaveformCursorDrag] 光标释放指针捕获失败:', error)
        }
      }

      cursorPointerTarget = null
      cursorPointerId = null
      detachCursorDragGuards()

      const diagConfig = getDiagConfig()
      if (!diagConfig.bypassPlaybackManagerDuringCursorDrag) {
        playbackManager.stopDragging()
      }
      logDiag('cursor-drag-end', {
        bypassPlaybackManager: diagConfig.bypassPlaybackManagerDuringCursorDrag,
      })
    } catch (error) {
      console.error('[WaveformCursorDrag] 拖拽结束出错:', error)
    }
  }

  // ============ Region 拖拽守护 ============

  function disableBodySelection() {
    if (typeof document === 'undefined' || !document.body) return
    previousBodyUserSelect = document.body.style.userSelect
    previousBodyWebkitSelect = document.body.style.webkitUserSelect
    document.body.style.userSelect = 'none'
    document.body.style.webkitUserSelect = 'none'
  }

  function restoreBodySelection() {
    if (typeof document === 'undefined' || !document.body) return
    document.body.style.userSelect = previousBodyUserSelect
    document.body.style.webkitUserSelect = previousBodyWebkitSelect
  }

  function attachRegionDragGuards() {
    if (regionDragGuardsAttached) return
    regionDragGuardsAttached = true
    document.addEventListener('pointerup', handleRegionPointerUp, true)
    document.addEventListener('pointercancel', handleRegionPointerCancel, true)
    window.addEventListener('blur', handleRegionPointerCancel, true)
    disableBodySelection()
  }

  function detachRegionDragGuards() {
    if (!regionDragGuardsAttached) return
    regionDragGuardsAttached = false
    document.removeEventListener('pointerup', handleRegionPointerUp, true)
    document.removeEventListener('pointercancel', handleRegionPointerCancel, true)
    window.removeEventListener('blur', handleRegionPointerCancel, true)
    isRegionPointerDragging.value = false
    regionPointerId = null
    regionPointerTarget = null
    restoreBodySelection()
  }

  function handleRegionPointerUp(e) {
    if (
      typeof e?.pointerId === 'number' &&
      regionPointerId !== null &&
      e.pointerId !== regionPointerId
    ) {
      logDiag('region-pointerup-ignored', { eventPointerId: e.pointerId })
      return
    }
    logDiag('region-pointerup', { eventPointerId: e?.pointerId ?? null })
    finalizeRegionPointerDrag()
  }

  function handleRegionPointerCancel(e) {
    if (
      typeof e?.pointerId === 'number' &&
      regionPointerId !== null &&
      e.pointerId !== regionPointerId
    ) {
      logDiag('region-pointercancel-ignored', { eventPointerId: e.pointerId })
      return
    }
    logDiag('region-pointercancel', { eventPointerId: e?.pointerId ?? null })
    finalizeRegionPointerDrag()
  }

  function finalizeRegionPointerDrag() {
    if (!isRegionPointerDragging.value) return
    logDiag('region-drag-finalize-start')

    regionPointerId = null
    regionPointerTarget = null
    isRegionPointerDragging.value = false
    detachRegionDragGuards()
    logDiag('region-drag-finalize-end')
  }

  function handleRegionPointerDown(e) {
    const ws = wavesurferRef.value
    if (!ws || !isReady.value) return
    if (e.pointerType === 'mouse' && e.button !== 0) return

    const target = e.target
    if (!(target instanceof HTMLElement)) return

    const regionEl = target.closest('[part*="region"]')
    if (!(regionEl instanceof HTMLElement)) return

    regionPointerId = e.pointerId
    regionPointerTarget = regionEl
    isRegionPointerDragging.value = true
    logDiag('region-pointerdown', {
      eventPointerId: e.pointerId,
      pointerType: e.pointerType,
      part: regionEl.getAttribute('part') || '',
    })

    attachRegionDragGuards()
  }

  function setupRegionPointerGuards(waveformRef) {
    if (regionPointerGuardEl || !waveformRef.value) return
    regionPointerGuardEl = waveformRef.value
    regionPointerGuardEl.addEventListener('pointerdown', handleRegionPointerDown, true)
  }

  function teardownRegionPointerGuards() {
    if (!regionPointerGuardEl) return
    regionPointerGuardEl.removeEventListener('pointerdown', handleRegionPointerDown, true)
    regionPointerGuardEl = null
  }

  // ============ 上半区域事件处理 ============

  function handleUpperZonePointerDown(e) {
    const ws = wavesurferRef.value
    if (!ws || !isReady.value) return
    if (e.pointerType === 'mouse' && e.button !== 0) return

    if (!canEditTimeline.value) {
      console.warn('[WaveformCursorDrag] 媒体未就绪，时间轴暂不可编辑')
      return
    }

    e.preventDefault()
    e.stopPropagation()

    let canDrag = false

    if (cursorDragMode.value === 'anywhere') {
      canDrag = true
    } else if (cursorDragMode.value === 'hover-only') {
      canDrag = isMouseNearCursor(e.clientX, 10)
    }

    const clickTime = getTimeFromClientX(e.clientX)
    const diagConfig = getDiagConfig()
    logDiag('cursor-pointerdown', {
      eventPointerId: e.pointerId,
      pointerType: e.pointerType,
      canDrag,
      clickTime,
      mode: cursorDragMode.value,
    })

    if (canDrag) {
      isDraggingCursor.value = true
      cursorMoveSampleCount = 0
      cursorPointerId = typeof e.pointerId === 'number' ? e.pointerId : null
      cursorPointerTarget = e.currentTarget instanceof HTMLElement ? e.currentTarget : null

      if (diagConfig.bypassPlaybackManagerDuringCursorDrag) {
        syncCursorDirectly(clickTime)
      } else {
        playbackManager.startDragging('waveformCursor')
        playbackManager.updateDragging(clickTime)
      }
      emit('seek', clickTime)

      if (
        cursorPointerTarget &&
        typeof cursorPointerTarget.setPointerCapture === 'function' &&
        typeof e.pointerId === 'number' &&
        !diagConfig.disablePointerCapture
      ) {
        try {
          cursorPointerTarget.setPointerCapture(e.pointerId)
        } catch (error) {
          console.debug('[WaveformCursorDrag] 光标指针捕获失败:', error)
        }
      } else if (diagConfig.disablePointerCapture) {
        logDiag('cursor-pointercapture-skipped')
      }

      attachCursorDragGuards()
    } else {
      playbackManager.seekTo(clickTime)
      emit('seek', clickTime)
    }
  }

  function handleUpperZonePointerMove(e) {
    const ws = wavesurferRef.value
    if (!ws || !isReady.value) return
    if (isDraggingCursor.value || isRegionPointerDragging.value) return

    const nearCursor = isMouseNearCursor(e.clientX, 10)

    if (nearCursor && cursorDragMode.value === 'hover-only') {
      currentMouseCursor.value = 'ew-resize'
    } else if (cursorDragMode.value === 'anywhere') {
      currentMouseCursor.value = 'col-resize'
    } else {
      currentMouseCursor.value = 'crosshair'
    }
  }

  function handleWaveformPointerLeave() {
    currentMouseCursor.value = 'default'
  }

  // ============ 清理 ============

  function cleanup() {
    if (isDraggingCursor.value) {
      handleCursorDragEnd()
    } else {
      detachCursorDragGuards()
    }

    if (isRegionPointerDragging.value) {
      finalizeRegionPointerDrag()
    } else {
      detachRegionDragGuards()
    }

    teardownRegionPointerGuards()
  }

  return {
    // 状态
    isDraggingCursor,
    isRegionPointerDragging,
    currentMouseCursor,
    cursorDragMode,
    getTimeFromClientX,
    // 上半区域事件
    handleUpperZonePointerDown,
    handleUpperZonePointerMove,
    handleWaveformPointerLeave,
    // Region 守护
    setupRegionPointerGuards,
    teardownRegionPointerGuards,
    // 清理
    cleanup,
  }
}
