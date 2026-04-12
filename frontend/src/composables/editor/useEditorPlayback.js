// V3.2.5+dev.20260321.02: 虚拟列表跟随改为“目标解析”和“滚动执行”解耦
import { computed } from 'vue'
import { usePlaybackManager } from '@/services/PlaybackManager'
import { usePlaybackStore } from '@/stores/playbackStore'
import { useProjectStore } from '@/stores/projectStore'
import { useEditorDocumentStore } from '@/stores/editor/editorDocumentStore'

const VIRTUAL_CENTER_MAX_ATTEMPTS = 8
const FOLLOW_VIEWPORT_PADDING_RATIO = 0.2

function toBaseMs(projectStore, displaySeconds) {
  return Math.round(projectStore.toBaseTime(displaySeconds) * 1000)
}

function resolveSubtitleIdByDisplaySeconds(docStore, projectStore, displaySeconds) {
  if (!Number.isFinite(displaySeconds)) {
    return null
  }

  return findSubtitleIdByTime(docStore, toBaseMs(projectStore, displaySeconds))
}

function findSubtitleIdByTime(docStore, targetMs) {
  const order = docStore.order
  let left = 0
  let right = order.length - 1

  while (left <= right) {
    const middle = Math.floor((left + right) / 2)
    const localId = order[middle]
    const entity = docStore.getEntity(localId)

    if (!entity || entity.isDeleted) {
      left = middle + 1
      continue
    }

    if (targetMs < entity.startMs) {
      right = middle - 1
      continue
    }

    if (targetMs >= entity.endMs) {
      left = middle + 1
      continue
    }

    return localId
  }

  return null
}

function clampScrollTop(rootElement, nextTop) {
  const scrollHeight = rootElement?.scrollHeight
  const clientHeight = rootElement?.clientHeight
  if (!Number.isFinite(scrollHeight) || !Number.isFinite(clientHeight)) {
    return Math.max(0, nextTop)
  }
  const maxScrollTop = Math.max(0, scrollHeight - clientHeight)
  return Math.max(0, Math.min(nextTop, maxScrollTop))
}

function resolveTargetMetrics(rootElement, target) {
  if (!rootElement || !target) {
    return null
  }

  const rootRect = typeof rootElement.getBoundingClientRect === 'function'
    ? rootElement.getBoundingClientRect()
    : null
  const targetRect = typeof target.getBoundingClientRect === 'function'
    ? target.getBoundingClientRect()
    : null
  const targetHeight = targetRect?.height ?? target.offsetHeight ?? 0
  const scrollerHeight = rootRect?.height ?? rootElement.clientHeight ?? 0
  const targetTop = (
    rootRect
      && targetRect
      && Number.isFinite(rootRect.top)
      && Number.isFinite(targetRect.top)
  )
    ? (targetRect.top - rootRect.top + (rootElement.scrollTop ?? 0))
    : target.offsetTop

  return {
    targetHeight,
    scrollerHeight,
    targetTop,
    targetBottom: targetTop + targetHeight,
    scrollTop: rootElement.scrollTop ?? 0,
  }
}

function calculateCenterScrollTop(metrics) {
  return Math.max(
    0,
    metrics.targetTop - metrics.scrollerHeight / 2 + metrics.targetHeight / 2
  )
}

function calculateFollowScrollTop(metrics, alignment = 'band') {
  if (alignment === 'center') {
    return calculateCenterScrollTop(metrics)
  }

  const {
    targetTop,
    targetBottom,
    targetHeight,
    scrollerHeight,
    scrollTop,
  } = metrics
  const viewportPadding = scrollerHeight * FOLLOW_VIEWPORT_PADDING_RATIO
  const followTopBoundary = scrollTop + viewportPadding
  const followBottomBoundary = scrollTop + scrollerHeight - viewportPadding

  if (targetHeight < scrollerHeight) {
    if (targetTop >= followTopBoundary && targetBottom <= followBottomBoundary) {
      return scrollTop
    }
  }

  if (targetTop < followTopBoundary) {
    return Math.max(0, targetTop - viewportPadding)
  }
  if (targetBottom > followBottomBoundary) {
    return Math.max(0, targetBottom - scrollerHeight + viewportPadding)
  }
  return scrollTop
}

function buildResolvedScrollTarget(rootElement, target, alignment = 'band') {
  const metrics = resolveTargetMetrics(rootElement, target)
  if (!metrics) {
    return null
  }

  const top = clampScrollTop(rootElement, calculateFollowScrollTop(metrics, alignment))
  return {
    rootElement,
    target,
    top,
    scrollTop: metrics.scrollTop,
    diff: top - metrics.scrollTop,
    targetTop: metrics.targetTop,
    targetHeight: metrics.targetHeight,
    scrollerHeight: metrics.scrollerHeight,
    alignment,
  }
}

function applyResolvedScrollTarget(payload, behavior = 'auto') {
  if (!payload?.rootElement) {
    return false
  }

  if (Math.abs(payload.diff) < 0.5) {
    return true
  }

  if (typeof payload.rootElement.scrollTo === 'function') {
    payload.rootElement.scrollTo({
      top: payload.top,
      behavior,
    })
    return true
  }

  payload.rootElement.scrollTop = payload.top
  return true
}

function scheduleAnimationFrame(callback) {
  if (typeof requestAnimationFrame === 'function') {
    requestAnimationFrame(callback)
    return
  }

  setTimeout(callback, 16)
}

function retryResolveRenderedTarget(rootElement, selector, options = {}) {
  const {
    attempt = 0,
    maxAttempts = VIRTUAL_CENTER_MAX_ATTEMPTS,
    alignment = 'band',
    shouldAbort,
    onResolved,
    onFailed,
  } = options

  if (shouldAbort?.()) {
    onFailed?.()
    return false
  }

  const target = rootElement?.querySelector?.(selector)
  if (target) {
    const payload = buildResolvedScrollTarget(rootElement, target, alignment)
    if (payload) {
      onResolved?.({
        ...payload,
        afterVirtualScroll: true,
      })
      return true
    }
  }

  if (attempt >= maxAttempts) {
    onFailed?.()
    return false
  }

  scheduleAnimationFrame(() => {
    retryResolveRenderedTarget(rootElement, selector, {
      ...options,
      attempt: attempt + 1,
    })
  })
  return true
}

export function useEditorPlayback() {
  const docStore = useEditorDocumentStore()
  const playbackStore = usePlaybackStore()
  const projectStore = useProjectStore()
  const playbackManager = usePlaybackManager()

  const currentSubtitleId = computed(() => {
    return resolveSubtitleIdByDisplaySeconds(
      docStore,
      projectStore,
      Number(playbackStore.currentTimeRaw)
    )
  })

  const committedSubtitleId = computed(() => {
    return resolveSubtitleIdByDisplaySeconds(
      docStore,
      projectStore,
      Number(playbackStore.currentTime)
    )
  })

  const followSubtitleId = computed(() => {
    return playbackStore.isSeeking
      ? committedSubtitleId.value
      : currentSubtitleId.value
  })

  const currentSubtitle = computed(() => {
    const localId = currentSubtitleId.value
    return localId ? docStore.getEntity(localId) ?? null : null
  })

  function seekToSubtitle(localId, options = {}) {
    const { autoPlay = false } = options
    const entity = docStore.getEntity(localId)
    if (!entity) {
      return false
    }

    const displaySeconds = projectStore.toDisplayTime(entity.startMs / 1000)
    playbackManager.seekTo(displaySeconds)
    if (autoPlay) {
      playbackManager.play()
    }
    return true
  }

  function resolveSubtitleScrollTarget(scroller, options = {}) {
    const targetLocalId = options.localId ?? followSubtitleId.value
    if (!scroller || !targetLocalId) {
      options.onFailed?.()
      return false
    }

    const fallbackIndex = Number.isInteger(options.index)
      ? options.index
      : docStore.getOrderIndex(targetLocalId)
    const rootElement = scroller.$el ?? scroller
    if (!rootElement?.querySelector) {
      options.onFailed?.()
      return false
    }

    const selector = `[data-local-id="${targetLocalId}"]`
    const alignment = options.alignment ?? 'band'
    const target = rootElement.querySelector(selector)
    if (target) {
      const payload = buildResolvedScrollTarget(rootElement, target, alignment)
      if (payload) {
        options.onResolved?.({
          ...payload,
          afterVirtualScroll: false,
        })
        return true
      }
      options.onFailed?.()
      return false
    }

    if (typeof scroller.scrollToItem === 'function' && Number.isInteger(fallbackIndex) && fallbackIndex >= 0) {
      options.onBeforeVirtualScroll?.()
      scroller.scrollToItem(fallbackIndex)
      return retryResolveRenderedTarget(rootElement, selector, {
        maxAttempts: options.maxAttempts,
        alignment,
        shouldAbort: options.shouldAbort,
        onResolved: options.onResolved,
        onFailed: options.onFailed,
      })
    }

    options.onFailed?.()
    return false
  }

  function followCurrentSubtitle(scroller, options = {}) {
    const behavior = options.behavior ?? 'auto'
    const virtualScrollBehavior = options.virtualScrollBehavior ?? 'auto'

    return resolveSubtitleScrollTarget(scroller, {
      ...options,
      onResolved: (payload) => {
        const nextBehavior = payload.afterVirtualScroll ? virtualScrollBehavior : behavior
        const applied = applyResolvedScrollTarget(payload, nextBehavior)
        if (applied) {
          options.onCentered?.(payload)
          return
        }
        options.onFailed?.()
      },
      onFailed: options.onFailed,
    })
  }

  return {
    currentSubtitleId,
    followSubtitleId,
    currentSubtitle,
    seekToSubtitle,
    resolveSubtitleScrollTarget,
    followCurrentSubtitle,
  }
}
