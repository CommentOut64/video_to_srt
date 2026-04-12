import { computed, onMounted, onUnmounted, ref, unref, watch } from 'vue'

const DEFAULT_BUFFER_PX = 240

function normalizeFiniteNumber(value, fallback = 0) {
  const numericValue = Number(value)
  return Number.isFinite(numericValue) ? numericValue : fallback
}

function resolvePixelsPerSecondFromWidth(contentWidth, durationSeconds, fallback = 0) {
  const normalizedWidth = normalizeFiniteNumber(contentWidth, 0)
  const normalizedDuration = normalizeFiniteNumber(durationSeconds, 0)
  if (normalizedWidth <= 0 || normalizedDuration <= 0) {
    return normalizeFiniteNumber(fallback, 0)
  }
  return normalizedWidth / normalizedDuration
}

export function resolveViewportContentWidth(options = {}) {
  const contentElement = options.contentElement ?? null
  const scrollElement = options.scrollElement ?? null
  const layoutWidth = normalizeFiniteNumber(
    contentElement?.offsetWidth ?? contentElement?.clientWidth,
    0
  )

  if (layoutWidth > 0) {
    return layoutWidth
  }

  const contentScrollWidth = normalizeFiniteNumber(contentElement?.scrollWidth, 0)
  if (contentScrollWidth > 0) {
    return contentScrollWidth
  }

  return normalizeFiniteNumber(scrollElement?.scrollWidth, 0)
}

export function computeViewportState(options = {}) {
  const scrollLeft = Math.max(0, normalizeFiniteNumber(options.scrollLeft, 0))
  const containerWidth = Math.max(0, normalizeFiniteNumber(options.containerWidth, 0))
  const pixelsPerSecond = Math.max(0, normalizeFiniteNumber(options.pixelsPerSecond, 0))
  const bufferPx = Math.max(0, normalizeFiniteNumber(options.bufferPx, DEFAULT_BUFFER_PX))
  const bufferMs = pixelsPerSecond > 0 ? Math.round((bufferPx / pixelsPerSecond) * 1000) : 0
  const viewportStartMs = pixelsPerSecond > 0
    ? Math.round((scrollLeft / pixelsPerSecond) * 1000)
    : 0
  const viewportEndMs = pixelsPerSecond > 0
    ? Math.round(((scrollLeft + containerWidth) / pixelsPerSecond) * 1000)
    : 0

  return {
    scrollLeft,
    containerWidth,
    pixelsPerSecond,
    pixelsPerMs: pixelsPerSecond / 1000,
    bufferPx,
    bufferMs,
    viewportStartMs,
    viewportEndMs,
  }
}

export function buildVisibleRegionSlice(options = {}) {
  const rangeStartMs = normalizeFiniteNumber(options.rangeStartMs, 0)
  const rangeEndMs = Math.max(rangeStartMs, normalizeFiniteNumber(options.rangeEndMs, rangeStartMs))
  const bufferMs = Math.max(0, normalizeFiniteNumber(options.bufferMs, 0))
  const order = Array.isArray(options.order) ? options.order : []
  const entities = options.entities instanceof Map ? options.entities : new Map()
  const searchStart = Math.max(0, rangeStartMs - bufferMs)
  const searchEnd = rangeEndMs + bufferMs
  const visibleEntries = []

  let left = 0
  let right = order.length

  while (left < right) {
    const middle = Math.floor((left + right) / 2)
    const entity = entities.get(order[middle])
    const entityEndMs = normalizeFiniteNumber(entity?.endMs, 0)
    if (entityEndMs < searchStart) {
      left = middle + 1
    } else {
      right = middle
    }
  }

  for (let index = left; index < order.length; index += 1) {
    const localId = order[index]
    const entity = entities.get(localId)
    if (!entity) {
      continue
    }

    const startMs = normalizeFiniteNumber(entity.startMs, 0)
    const endMs = Math.max(startMs, normalizeFiniteNumber(entity.endMs, startMs))
    if (startMs > searchEnd) {
      break
    }
    if (entity.isDeleted) {
      continue
    }
    if (endMs < searchStart || startMs > searchEnd) {
      continue
    }

    visibleEntries.push(entity)
  }

  return visibleEntries
}

export function useWaveformViewport(options = {}) {
  const viewportState = ref(computeViewportState())
  const scrollElementRef = options.scrollElementRef
  const contentElementRef = options.contentElementRef
  const durationSecondsRef = options.durationSecondsRef
  const bufferPxRef = options.bufferPxRef ?? DEFAULT_BUFFER_PX
  const fallbackPixelsPerSecondRef = options.fallbackPixelsPerSecondRef ?? 0
  let rafId = null
  let resizeHandler = null
  let scrollHandler = null
  let boundScrollElement = null

  function resolveScrollElement() {
    return unref(scrollElementRef) ?? null
  }

  function resolveContentElement() {
    return unref(contentElementRef) ?? resolveScrollElement()
  }

  function readViewportState() {
    const scrollElement = resolveScrollElement()
    const contentElement = resolveContentElement()
    const durationSeconds = normalizeFiniteNumber(unref(durationSecondsRef), 0)
    const bufferPx = normalizeFiniteNumber(unref(bufferPxRef), DEFAULT_BUFFER_PX)
    const scrollLeft = normalizeFiniteNumber(scrollElement?.scrollLeft, 0)
    const containerWidth = normalizeFiniteNumber(
      scrollElement?.clientWidth ?? contentElement?.clientWidth,
      0
    )
    const contentWidth = resolveViewportContentWidth({
      contentElement,
      scrollElement,
    })
    const pixelsPerSecond = resolvePixelsPerSecondFromWidth(
      contentWidth,
      durationSeconds,
      unref(fallbackPixelsPerSecondRef)
    )

    viewportState.value = computeViewportState({
      scrollLeft,
      containerWidth,
      pixelsPerSecond,
      bufferPx,
    })
    return viewportState.value
  }

  function cancelScheduledMeasure() {
    if (rafId === null) {
      return
    }
    const cancel = typeof cancelAnimationFrame === 'function'
      ? cancelAnimationFrame
      : clearTimeout
    cancel(rafId)
    rafId = null
  }

  function scheduleMeasure() {
    if (rafId !== null) {
      return
    }
    const schedule = typeof requestAnimationFrame === 'function'
      ? requestAnimationFrame
      : (callback) => setTimeout(() => callback(Date.now()), 16)
    rafId = schedule(() => {
      rafId = null
      readViewportState()
    })
  }

  function bindScrollListener() {
    const nextScrollElement = resolveScrollElement()
    if (boundScrollElement === nextScrollElement) {
      return
    }

    if (boundScrollElement && scrollHandler) {
      boundScrollElement.removeEventListener('scroll', scrollHandler)
    }

    boundScrollElement = nextScrollElement
    if (!boundScrollElement) {
      return
    }

    scrollHandler = scrollHandler ?? (() => scheduleMeasure())
    boundScrollElement.addEventListener('scroll', scrollHandler, { passive: true })
  }

  onMounted(() => {
    resizeHandler = () => scheduleMeasure()
    if (typeof window !== 'undefined' && window?.addEventListener) {
      window.addEventListener('resize', resizeHandler)
    }
    bindScrollListener()
    scheduleMeasure()
  })

  onUnmounted(() => {
    cancelScheduledMeasure()
    if (boundScrollElement && scrollHandler) {
      boundScrollElement.removeEventListener('scroll', scrollHandler)
    }
    boundScrollElement = null
    if (resizeHandler && typeof window !== 'undefined' && window?.removeEventListener) {
      window.removeEventListener('resize', resizeHandler)
    }
    resizeHandler = null
  })

  watch(
    () => [
      unref(durationSecondsRef),
      unref(bufferPxRef),
      unref(fallbackPixelsPerSecondRef),
      unref(scrollElementRef),
      unref(contentElementRef),
    ],
    () => {
      bindScrollListener()
      scheduleMeasure()
    },
    { flush: 'post' }
  )

  return {
    viewportState,
    viewportStartMs: computed(() => viewportState.value.viewportStartMs),
    viewportEndMs: computed(() => viewportState.value.viewportEndMs),
    bufferMs: computed(() => viewportState.value.bufferMs),
    pixelsPerSecond: computed(() => viewportState.value.pixelsPerSecond),
    pixelsPerMs: computed(() => viewportState.value.pixelsPerMs),
    scheduleMeasure,
    syncViewportNow: readViewportState,
    cancelScheduledMeasure,
  }
}
