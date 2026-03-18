// V3.2.5+dev.20260318.01: 可视窗口 FLIP 协调器 - 纯逻辑与动画播放

/**
 * 判定当前上下文是否应跳过 FLIP 动画
 * @param {Object} context - 运行时上下文
 * @returns {{ skip: boolean, reason: string|null }}
 */
export function shouldSkipVisibleFlip(context) {
  if (context.isGroupedMode) {
    return { skip: true, reason: 'grouped_mode' }
  }
  if (context.isResizing) {
    return { skip: true, reason: 'sidebar_resizing' }
  }
  if (context.isInProgrammaticScrollGuard) {
    return { skip: true, reason: 'programmatic_scroll_guard' }
  }
  if (context.hasRecentUserScroll) {
    return { skip: true, reason: 'recent_user_scroll' }
  }
  if (context.prefersReducedMotion) {
    return { skip: true, reason: 'reduced_motion' }
  }
  return { skip: false, reason: null }
}

/**
 * 根据前后可见快照构建 FLIP 动画计划
 * @param {Map} before - 编辑前可见项快照
 * @param {Map} after - 编辑后可见项快照
 * @param {Object} options - 配置项
 * @returns {{ skip: boolean, reason: string|null, moves: Array, enters: Array }}
 */
export function buildVisibleFlipPlan(before, after, options = {}) {
  const maxAnimatedItems = options.maxAnimatedItems ?? 12
  const enterOffsetPx = options.enterOffsetPx ?? 8
  const moves = []
  const enters = []

  for (const [localId, next] of after.entries()) {
    const prev = before.get(localId)
    if (prev) {
      const deltaY = prev.top - next.top
      if (deltaY !== 0) {
        moves.push({ localId, deltaY, element: next.element })
      }
      continue
    }
    enters.push({ localId, enterOffsetPx, element: next.element })
  }

  const affectedCount = moves.length + enters.length
  if (affectedCount > maxAnimatedItems) {
    return { skip: true, reason: 'too_many_items', moves: [], enters: [] }
  }

  return { skip: false, reason: null, moves, enters }
}

/**
 * 采集当前滚动容器内可见 row-shell 的几何快照
 * @param {HTMLElement} container - 滚动容器元素
 * @returns {Map<string, Object>}
 */
export function captureVisibleSnapshot(container) {
  const snapshot = new Map()
  if (!container) {
    return snapshot
  }

  const containerRect = container.getBoundingClientRect()
  const rowElements = container.querySelectorAll('.row-shell[data-local-id]')

  rowElements.forEach((element) => {
    if (!element.isConnected) return
    const localId = element.dataset.localId
    if (!localId) return

    const rect = element.getBoundingClientRect()
    // 跳过不在可视区内的项
    if (rect.bottom <= containerRect.top || rect.top >= containerRect.bottom) {
      return
    }

    snapshot.set(localId, {
      localId,
      top: rect.top - containerRect.top,
      left: rect.left - containerRect.left,
      width: rect.width,
      height: rect.height,
      element,
    })
  })

  return snapshot
}

function defaultNextFrame() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()))
}

function snapshotSignature(snapshot) {
  return JSON.stringify(
    Array.from(snapshot.entries()).map(([localId, item]) => [localId, item.top])
  )
}

/**
 * 等待布局稳定：连续 stableFrameCount 帧快照不变后返回
 * @param {Object} options
 * @returns {Promise<Map>}
 */
export async function waitForStableLayout({
  capture,
  nextFrame = defaultNextFrame,
  stableFrameCount = 2,
  maxFrames = 4,
}) {
  let stableCount = 0
  let lastSignature = ''
  let current = capture()

  for (let index = 0; index < maxFrames; index += 1) {
    await nextFrame()
    current = capture()
    const signature = snapshotSignature(current)
    if (signature === lastSignature) {
      stableCount += 1
      if (stableCount >= stableFrameCount) {
        return current
      }
    } else {
      stableCount = 1
      lastSignature = signature
    }
  }

  return current
}

/**
 * 按 FLIP 计划播放动画，返回取消函数
 * @param {Object} plan - buildVisibleFlipPlan 的返回值
 * @param {Object} options - 动画配置
 * @returns {Function} cancel 清理函数
 */
export function playVisibleFlip(plan, options = {}) {
  if (plan.skip) {
    return () => {}
  }

  const durationMs = options.durationMs ?? 160
  const easing = options.easing ?? 'cubic-bezier(0.4, 0, 0.2, 1)'
  const animations = []

  for (const move of plan.moves) {
    if (!move.element?.isConnected || typeof move.element.animate !== 'function') continue
    animations.push(move.element.animate([
      { transform: `translateY(${move.deltaY}px)` },
      { transform: 'translateY(0)' },
    ], { duration: durationMs, easing, fill: 'both' }))
  }

  for (const enter of plan.enters) {
    if (!enter.element?.isConnected || typeof enter.element.animate !== 'function') continue
    animations.push(enter.element.animate([
      { opacity: 0, transform: `translateY(${enter.enterOffsetPx}px)` },
      { opacity: 1, transform: 'translateY(0)' },
    ], { duration: durationMs, easing, fill: 'both' }))
  }

  return () => {
    animations.forEach((animation) => {
      try {
        animation.cancel()
      } catch {
        // Why: 取消函数是幂等清理路径，个别浏览器/测试桩取消失败不应阻断后续 cleanup
      }
    })
  }
}
