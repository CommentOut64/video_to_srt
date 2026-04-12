const DEFAULT_LERP_FACTOR = 0.10
const DISCRETE_WHEEL_DELTA_EPSILON = 0.01
const BOOST_THRESHOLD_PX = 240
const BOOST_RANGE_PX = 260
const MAX_LERP_FACTOR = 0.22

function resolveDeltaY(event) {
  if (!event) {
    return 0
  }
  return event.deltaMode === 1 ? event.deltaY * 32 : event.deltaY
}

function isApproximatelyInteger(value) {
  if (!Number.isFinite(value)) {
    return false
  }
  return Math.abs(value - Math.round(value)) <= DISCRETE_WHEEL_DELTA_EPSILON
}

function resolveAdaptiveLerpFactor(baseLerpFactor, diff) {
  const distance = Math.abs(diff)
  if (distance <= BOOST_THRESHOLD_PX) {
    return baseLerpFactor
  }

  const overflow = Math.min(1, (distance - BOOST_THRESHOLD_PX) / BOOST_RANGE_PX)
  return Math.min(
    MAX_LERP_FACTOR,
    baseLerpFactor + ((MAX_LERP_FACTOR - baseLerpFactor) * overflow)
  )
}

export function shouldUseCustomWheelSmoothing(event) {
  if (!event) {
    return false
  }
  if (event.ctrlKey) {
    return false
  }
  if (Math.abs(event.deltaX ?? 0) > Math.abs(event.deltaY ?? 0)) {
    return false
  }
  if (event.deltaMode === 1) {
    return true
  }
  return isApproximatelyInteger(event.deltaY) && Math.abs(event.deltaY) >= 4
}

export function createWheelSmoother(options) {
  const {
    getScrollerElement,
    requestAnimationFrameFn = globalThis.requestAnimationFrame?.bind(globalThis),
    cancelAnimationFrameFn = globalThis.cancelAnimationFrame?.bind(globalThis),
    lerpFactor = DEFAULT_LERP_FACTOR,
  } = options

  const state = {
    targetScrollTop: null,
    rafId: null,
  }

  function clear(reason = 'manual') {
    if (state.rafId !== null) {
      cancelAnimationFrameFn?.(state.rafId)
    }
    state.rafId = null
    state.targetScrollTop = null
  }

  function isAnimating() {
    return state.rafId !== null || state.targetScrollTop !== null
  }

  function step() {
    const scrollerElement = getScrollerElement?.()
    if (!scrollerElement || state.targetScrollTop === null) {
      clear()
      return
    }

    const diff = state.targetScrollTop - scrollerElement.scrollTop
    if (Math.abs(diff) < 0.5) {
      scrollerElement.scrollTop = Math.round(state.targetScrollTop)
      clear()
      return
    }

    const adaptiveLerpFactor = resolveAdaptiveLerpFactor(lerpFactor, diff)
    scrollerElement.scrollTop += diff * adaptiveLerpFactor
    state.rafId = requestAnimationFrameFn?.(step) ?? null
  }

  function handleWheel(event) {
    const scrollerElement = getScrollerElement?.()
    if (!scrollerElement) {
      return false
    }

    const resolvedDeltaY = resolveDeltaY(event)
    if (state.targetScrollTop === null) {
      state.targetScrollTop = scrollerElement.scrollTop
    }

    state.targetScrollTop = Math.max(
      0,
      Math.min(
        scrollerElement.scrollHeight - scrollerElement.clientHeight,
        state.targetScrollTop + resolvedDeltaY
      )
    )

    if (state.rafId === null) {
      state.rafId = requestAnimationFrameFn?.(step) ?? null
    }

    return true
  }

  return {
    clear,
    isAnimating,
    handleWheel,
  }
}
