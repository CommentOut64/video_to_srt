function scheduleFrame(callback) {
  if (typeof requestAnimationFrame === 'function') {
    return requestAnimationFrame(callback)
  }
  return setTimeout(() => callback(Date.now()), 16)
}

function cancelFrame(handle) {
  if (typeof cancelAnimationFrame === 'function') {
    cancelAnimationFrame(handle)
    return
  }
  clearTimeout(handle)
}

export function createOverlayScheduler(flush) {
  let frameHandle = null
  const pendingReasons = new Set()

  function flushPending(meta = {}) {
    if (pendingReasons.size === 0) {
      return 0
    }

    const reasons = Array.from(pendingReasons)
    pendingReasons.clear()
    flush({
      reasons,
      forced: meta.forced === true,
    })
    return reasons.length
  }

  function runScheduledFlush() {
    frameHandle = null
    flushPending()
  }

  function invalidate(reason = 'unknown') {
    pendingReasons.add(reason)
    if (frameHandle !== null) {
      return
    }
    frameHandle = scheduleFrame(runScheduledFlush)
  }

  function flushNow(meta = {}) {
    if (frameHandle !== null) {
      cancelFrame(frameHandle)
      frameHandle = null
    }
    return flushPending(meta)
  }

  function cancel() {
    if (frameHandle !== null) {
      cancelFrame(frameHandle)
      frameHandle = null
    }
    pendingReasons.clear()
  }

  return {
    invalidate,
    flushNow,
    cancel,
  }
}

export function useWaveformOverlayScheduler(flush) {
  return createOverlayScheduler(flush)
}
