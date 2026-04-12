/**
 * @deprecated V3.2.5: 使用 TimelineViewport 替代
 * useWaveformRegions - Region 管理逻辑 Composable
 *
 * 职责：Overlay Region 交互提交、点击跳转、时间同步节流
 * 提取自 WaveformTimeline/index.vue L507-601, L667-729
 */
import { ref } from 'vue'
import { isFeatureEnabled } from '@/config/featureFlags'
import { useEditorDocumentStore } from '@/stores/editor/editorDocumentStore'
import { useEditorCommandBus } from '@/stores/editor/editorCommandBus'
import {
  createMoveBoundaryCommand,
  createUpdateTimingCommand,
} from '@/stores/editor/editorCommandFactory'
import {
  logWaveformDragDiagnostics,
} from './waveformDragDiagnostics.js'

/**
 * 简单防抖工具
 */
function debounce(fn, delay) {
  let timer = null
  const debounced = function (...args) {
    if (timer) {
      clearTimeout(timer)
    }
    timer = setTimeout(() => {
      timer = null
      fn.apply(this, args)
    }, delay)
  }
  debounced.cancel = () => {
    if (timer) {
      clearTimeout(timer)
      timer = null
    }
  }
  return debounced
}

/**
 * Region 管理 Composable
 * @param {object} projectStore - Pinia store
 * @param {Function} onSubtitleEdit - 字幕编辑回调
 * @param {object} playbackManager - PlaybackManager 实例
 * @param {object} subtitleDocumentStore - 字幕文档 store
 * @param {Function} emit - 事件发射函数
 * @param {Ref<Array>|null} regionItemsRef - Region 渲染快照
 */
export function useWaveformRegions(
  projectStore,
  onSubtitleEdit,
  playbackManager,
  subtitleDocumentStore,
  emit,
  regionItemsRef = null
) {
  // ============ 状态 ============
  const isUpdatingRegions = ref(false)
  const useEditorV2 = isFeatureEnabled('USE_EDITOR_V2')
  const docStore = useEditorV2 ? useEditorDocumentStore() : null
  const commandBus = useEditorV2 ? useEditorCommandBus() : null

  // ============ 私有状态 ============
  let pendingRegionCommitTimer = null
  const pendingRegionCommits = new Map()

  function logDiag(eventName, payload = {}) {
    logWaveformDragDiagnostics(eventName, payload)
  }

  function resolveMaybeRefValue(valueOrRef) {
    if (valueOrRef && typeof valueOrRef === 'object' && 'value' in valueOrRef) {
      return valueOrRef.value
    }
    return valueOrRef
  }

  function buildEditorV2SubtitleList() {
    if (!useEditorV2 || !docStore) {
      return []
    }

    return docStore.order
      .map((localId) => {
        const entity = docStore.getEntity(localId)
        if (!entity || entity.isDeleted) {
          return null
        }

        const cold = docStore.getCold(localId)
        const start = Math.max(0, projectStore.toDisplayTime(entity.startMs / 1000))
        const end = Math.max(start, projectStore.toDisplayTime(entity.endMs / 1000))

        return {
          id: localId,
          segment_id: cold?.segmentId ?? null,
          sentenceIndex: cold?.sentenceIndex ?? null,
          start,
          end,
        }
      })
      .filter(Boolean)
  }

  function getSubtitleList() {
    const regionItems = resolveMaybeRefValue(regionItemsRef)
    if (Array.isArray(regionItems)) {
      return regionItems
    }

    if (useEditorV2 && docStore) {
      return buildEditorV2SubtitleList()
    }

    const subtitles = resolveMaybeRefValue(subtitleDocumentStore?.subtitles)
    if (Array.isArray(subtitles)) {
      return subtitles
    }
    return []
  }

  function resolveEditorV2LocalId(regionId) {
    if (!useEditorV2 || !docStore || regionId === undefined || regionId === null) {
      return null
    }

    const normalizedRegionId = String(regionId)

    if (docStore.getEntity(normalizedRegionId)) {
      return normalizedRegionId
    }

    const boundLocalId = docStore.bindingBySegmentId.get(normalizedRegionId)
    if (boundLocalId && docStore.getEntity(boundLocalId)) {
      return boundLocalId
    }

    return docStore.order.find((localId) => String(localId) === normalizedRegionId) ?? null
  }

  function setSelectedSubtitleId(subtitleId) {
    subtitleDocumentStore.setSelectedSubtitleId(subtitleId)
  }

  // 节流的 Region 同步
  const debouncedRegionSync = debounce((syncKey, start, end) => {
    if (syncKey === undefined || syncKey === null) return
    if (typeof syncKey === 'number' && projectStore.isSentenceDeleted?.(syncKey)) return
    onSubtitleEdit(syncKey, { start, end })
  }, 200)

  function resolveFiniteTime(rawValue, fallback = 0) {
    const normalized = Number(rawValue)
    return Number.isFinite(normalized) ? normalized : fallback
  }

  function toBaseMs(displaySeconds) {
    return Math.round(projectStore.toBaseTime(displaySeconds) * 1000)
  }

  function normalizeTimingRange(startMs, endMs) {
    const normalizedStart = Math.max(0, Math.round(startMs))
    const normalizedEnd = Math.max(normalizedStart + 1, Math.round(endMs))
    return {
      startMs: normalizedStart,
      endMs: normalizedEnd,
    }
  }

  function hasSharedBoundary(leftMs, rightMs) {
    return Math.abs(Number(leftMs) - Number(rightMs)) <= 1
  }

  function createBoundaryMoveCommand(localId, side, nextTiming) {
    if (!useEditorV2 || !docStore || !commandBus || !side) {
      return null
    }

    const entity = docStore.getEntity(localId)
    if (!entity) {
      return null
    }

    const neighbors = docStore.getNeighbors(localId)

    if (side === 'start' && neighbors.prev) {
      const upper = docStore.getEntity(neighbors.prev)
      if (!upper || !hasSharedBoundary(upper.endMs, entity.startMs)) {
        return null
      }

      const minBoundaryMs = upper.startMs + 1
      const maxBoundaryMs = entity.endMs - 1
      if (minBoundaryMs > maxBoundaryMs) {
        return null
      }

      const boundaryMs = Math.min(maxBoundaryMs, Math.max(minBoundaryMs, nextTiming.startMs))
      return createMoveBoundaryCommand({
        upperLocalId: neighbors.prev,
        lowerLocalId: localId,
        before: {
          upperEndMs: upper.endMs,
          lowerStartMs: entity.startMs,
        },
        after: {
          upperEndMs: boundaryMs,
          lowerStartMs: boundaryMs,
        },
        source: 'user',
      })
    }

    if (side === 'end' && neighbors.next) {
      const lower = docStore.getEntity(neighbors.next)
      if (!lower || !hasSharedBoundary(entity.endMs, lower.startMs)) {
        return null
      }

      const minBoundaryMs = entity.startMs + 1
      const maxBoundaryMs = lower.endMs - 1
      if (minBoundaryMs > maxBoundaryMs) {
        return null
      }

      const boundaryMs = Math.min(maxBoundaryMs, Math.max(minBoundaryMs, nextTiming.endMs))
      return createMoveBoundaryCommand({
        upperLocalId: localId,
        lowerLocalId: neighbors.next,
        before: {
          upperEndMs: entity.endMs,
          lowerStartMs: lower.startMs,
        },
        after: {
          upperEndMs: boundaryMs,
          lowerStartMs: boundaryMs,
        },
        source: 'user',
      })
    }

    return null
  }

  function scheduleFlushPendingRegionCommits(delay = 16) {
    if (pendingRegionCommitTimer) {
      clearTimeout(pendingRegionCommitTimer)
    }
    pendingRegionCommitTimer = setTimeout(() => {
      pendingRegionCommitTimer = null
      flushPendingRegionCommits()
    }, delay)
  }

  function commitRegionTimeChange(regionSnapshot) {
    const regionId = regionSnapshot?.id
    if (!regionId) return false

    let nextStart = null
    let nextEnd = null
    let syncKey = null
    if (useEditorV2) {
      if (!docStore || !commandBus) {
        throw new Error('[WaveformRegions] Phase F-0 缺少 V2 命令上下文，禁止回退旧时间编辑链')
      }

      const localId = resolveEditorV2LocalId(regionId)
      if (!localId) {
        throw new Error(`[WaveformRegions] 未找到 region 对应的 localId（regionId=${regionId}）`)
      }

      const entity = docStore.getEntity(localId)
      if (!entity) {
        throw new Error(`[WaveformRegions] 未找到 localId 对应实体（localId=${localId}）`)
      }

      const currentStart = Math.max(0, projectStore.toDisplayTime(entity.startMs / 1000))
      const currentEnd = Math.max(currentStart, projectStore.toDisplayTime(entity.endMs / 1000))
      nextStart = resolveFiniteTime(regionSnapshot.start, currentStart)
      nextEnd = resolveFiniteTime(regionSnapshot.end, currentEnd)
      const hasTimeChanged =
        Math.abs(currentStart - nextStart) > 0.0005 || Math.abs(currentEnd - nextEnd) > 0.0005

      if (!hasTimeChanged) {
        return false
      }

      const nextTiming = normalizeTimingRange(toBaseMs(nextStart), toBaseMs(nextEnd))
      const command = createBoundaryMoveCommand(localId, regionSnapshot?.side, nextTiming)
        || createUpdateTimingCommand({
          localId,
          before: {
            startMs: entity.startMs,
            endMs: entity.endMs,
          },
          after: nextTiming,
          source: 'user',
        })

      const result = commandBus.dispatch(command)
      if (!result?.success) {
        throw new Error(
          `[WaveformRegions] 时间命令执行失败（localId=${localId}，reason=${result?.reason || '未知原因'}）`
        )
      }

      setSelectedSubtitleId(localId)
    } else {
      const subtitle = getSubtitleList().find((s) => (
        String(s.id) === String(regionId)
        || String(s.segment_id ?? '') === String(regionId)
      ))
      if (!subtitle) return false

      nextStart = resolveFiniteTime(regionSnapshot.start, resolveFiniteTime(subtitle.start, 0))
      nextEnd = resolveFiniteTime(regionSnapshot.end, resolveFiniteTime(subtitle.end, nextStart))
      const currentStart = resolveFiniteTime(subtitle.start, 0)
      const currentEnd = resolveFiniteTime(subtitle.end, currentStart)
      const hasTimeChanged =
        Math.abs(currentStart - nextStart) > 0.0005 || Math.abs(currentEnd - nextEnd) > 0.0005

      if (!hasTimeChanged) {
        return false
      }

      syncKey = subtitle?.segment_id ?? subtitle?.sentenceIndex ?? subtitle?.id
      if (syncKey !== undefined && syncKey !== null) {
        debouncedRegionSync(syncKey, nextStart, nextEnd)
      }

      setSelectedSubtitleId(subtitle.id)
    }

    const emittedRegion = regionSnapshot.region ?? {
      id: regionId,
      start: nextStart ?? resolveFiniteTime(regionSnapshot?.start, 0),
      end: nextEnd ?? resolveFiniteTime(regionSnapshot?.end, nextStart ?? 0),
    }
    emit('region-update', emittedRegion)
    logDiag('regions-emit-region-update', {
      regionId,
      syncKey: syncKey ?? null,
      deferred: regionSnapshot.deferred === true,
    })

    return true
  }

  function queuePendingRegionCommit(region, side = null) {
    const regionId = region?.id
    if (!regionId) return

    pendingRegionCommits.set(regionId, {
      id: regionId,
      start: region.start,
      end: region.end,
      side: side || null,
      region,
      deferred: true,
    })

    logDiag('regions-region-updated-deferred', {
      regionId,
      side: side || null,
      start: region.start,
      end: region.end,
      pendingCount: pendingRegionCommits.size,
    })
    scheduleFlushPendingRegionCommits(16)
  }

  function flushPendingRegionCommits(options = {}) {
    const { force = false } = options
    if (pendingRegionCommits.size === 0) return 0
    if (!force && isUpdatingRegions.value) {
      scheduleFlushPendingRegionCommits(16)
      return 0
    }

    const snapshots = Array.from(pendingRegionCommits.values())
    pendingRegionCommits.clear()
    let committedCount = 0

    snapshots.forEach((snapshot) => {
      if (commitRegionTimeChange(snapshot)) {
        committedCount += 1
      }
    })

    if (committedCount > 0) {
      logDiag('regions-pending-commit-flushed', {
        committedCount,
        force,
      })
    }
    return committedCount
  }

  function handleRegionClick(region) {
    if (!region?.id) {
      return false
    }

    if (useEditorV2) {
      const localId = resolveEditorV2LocalId(region.id)
      setSelectedSubtitleId(localId ?? region.id)
    } else {
      setSelectedSubtitleId(region.id)
    }

    playbackManager.seekTo(region.start)
    playbackManager.play()
    emit('region-click', region)
    return true
  }

  /**
   * 清理资源
   */
  function cleanup() {
    clearTimeout(pendingRegionCommitTimer)
    pendingRegionCommitTimer = null
    pendingRegionCommits.clear()
    debouncedRegionSync.cancel?.()
  }

  return {
    // 状态
    isUpdatingRegions,
    // 方法
    flushPendingRegionCommits,
    commitRegionTimeChange,
    handleRegionClick,
    // 清理
    cleanup,
  }
}
