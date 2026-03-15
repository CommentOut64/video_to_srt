// V3.2.5+dev.20260315.11: 虚拟列表跟随补齐多帧渲染后的稳定居中
import { computed } from 'vue'
import { usePlaybackManager } from '@/services/PlaybackManager'
import { usePlaybackStore } from '@/stores/playbackStore'
import { useProjectStore } from '@/stores/projectStore'
import { useEditorDocumentStore } from '@/stores/editor/editorDocumentStore'

const VIRTUAL_CENTER_MAX_ATTEMPTS = 8

function toBaseMs(projectStore, displaySeconds) {
  return Math.round(projectStore.toBaseTime(displaySeconds) * 1000)
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

function centerTargetInScroller(rootElement, target, behavior = 'smooth') {
  if (!rootElement || !target) {
    return false
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
  const nextTop = Math.max(
    0,
    targetTop - ((scrollerHeight - targetHeight) / 2)
  )

  if (typeof rootElement.scrollTo === 'function') {
    rootElement.scrollTo({
      top: nextTop,
      behavior,
    })
    return true
  }

  rootElement.scrollTop = nextTop
  return true
}

function scheduleAnimationFrame(callback) {
  if (typeof requestAnimationFrame === 'function') {
    requestAnimationFrame(callback)
    return
  }

  setTimeout(callback, 16)
}

function retryCenterRenderedTarget(rootElement, selector, options = {}) {
  const {
    attempt = 0,
    maxAttempts = VIRTUAL_CENTER_MAX_ATTEMPTS,
    behavior = 'auto',
    shouldAbort,
    onCentered,
    onFailed,
  } = options

  if (shouldAbort?.()) {
    onFailed?.()
    return false
  }

  const target = rootElement?.querySelector?.(selector)
  if (target) {
    const centered = centerTargetInScroller(rootElement, target, behavior)
    if (centered) {
      onCentered?.()
      return true
    }
  }

  if (attempt >= maxAttempts) {
    onFailed?.()
    return false
  }

  scheduleAnimationFrame(() => {
    retryCenterRenderedTarget(rootElement, selector, {
      ...options,
      attempt: attempt + 1,
    })
  })
  return true
}

function scrollToSubtitle(scroller, localId, fallbackIndex, options = {}) {
  if (!scroller || !localId) {
    options.onFailed?.()
    return false
  }

  const rootElement = scroller.$el ?? scroller
  if (!rootElement?.querySelector) {
    options.onFailed?.()
    return false
  }

  const selector = `[data-local-id="${localId}"]`
  const target = rootElement.querySelector(selector)
  if (target) {
    const centered = centerTargetInScroller(rootElement, target, 'smooth')
    if (centered) {
      options.onCentered?.()
      return true
    }
    options.onFailed?.()
    return false
  }

  if (typeof scroller.scrollToItem === 'function' && Number.isInteger(fallbackIndex) && fallbackIndex >= 0) {
    options.onBeforeVirtualScroll?.()
    scroller.scrollToItem(fallbackIndex)
    return retryCenterRenderedTarget(rootElement, selector, {
      maxAttempts: options.maxAttempts,
      behavior: 'auto',
      shouldAbort: options.shouldAbort,
      onCentered: options.onCentered,
      onFailed: options.onFailed,
    })
  }

  options.onFailed?.()
  return false
}

export function useEditorPlayback() {
  const docStore = useEditorDocumentStore()
  const playbackStore = usePlaybackStore()
  const projectStore = useProjectStore()
  const playbackManager = usePlaybackManager()

  const currentSubtitleId = computed(() => {
    const displaySeconds = Number(playbackStore.currentTimeRaw)
    if (!Number.isFinite(displaySeconds)) {
      return null
    }

    return findSubtitleIdByTime(docStore, toBaseMs(projectStore, displaySeconds))
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

  function followCurrentSubtitle(scroller, options = {}) {
    const targetLocalId = options.localId ?? currentSubtitleId.value
    if (!targetLocalId) {
      return false
    }

    const fallbackIndex = Number.isInteger(options.index)
      ? options.index
      : docStore.getOrderIndex(targetLocalId)

    return scrollToSubtitle(scroller, targetLocalId, fallbackIndex, options)
  }

  return {
    currentSubtitleId,
    currentSubtitle,
    seekToSubtitle,
    followCurrentSubtitle,
  }
}
