/**
 * @deprecated V3.2.5: 使用 TimelineViewport 替代
 * useWaveformRegions - Region 管理逻辑 Composable
 *
 * 职责：Region 渲染、重叠检测、事件绑定、同步节流
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
import { detectOverlappingSubtitles, OVERLAP_COLORS } from '@/utils/subtitleUtils'
import {
  getWaveformDragDiagnosticsConfig,
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
 * @param {Ref<object>} regionsPluginRef - Regions 插件引用
 * @param {object} projectStore - Pinia store
 * @param {object} props - 组件 props
 * @param {Ref<boolean>} isReady - 波形是否就绪
 * @param {Function} onSubtitleEdit - 字幕编辑回调
 * @param {object} playbackManager - PlaybackManager 实例
 * @param {object} subtitleDocumentStore - 字幕文档 store
 * @param {Function} emit - 事件发射函数
 * @param {Ref<Array>|null} regionItemsRef - Region 渲染快照
 */
export function useWaveformRegions(
  regionsPluginRef,
  projectStore,
  props,
  isReady,
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
  let regionUpdateTimer = null
  let pendingRegionCommitTimer = null
  const pendingRegionCommits = new Map()
  let overlapCacheDirty = true
  let overlapCacheIds = new Set()

  function logDiag(eventName, payload = {}) {
    logWaveformDragDiagnostics(eventName, payload)
  }

  function resolveMaybeRefValue(valueOrRef) {
    if (valueOrRef && typeof valueOrRef === 'object' && 'value' in valueOrRef) {
      return valueOrRef.value
    }
    return valueOrRef
  }

  function getSelectedSubtitleId() {
    return resolveMaybeRefValue(subtitleDocumentStore?.selectedSubtitleId) ?? null
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

  function resolveRegionMinLength() {
    const fallbackMinLength = 0.05
    const rawValue = Number(props.regionMinLength)
    if (!Number.isFinite(rawValue)) {
      return fallbackMinLength
    }
    return Math.max(0, rawValue)
  }

  function markOverlapCacheDirty() {
    overlapCacheDirty = true
  }

  function getOverlappingIds(options = {}) {
    const { forceRefresh = false } = options
    if (forceRefresh || overlapCacheDirty) {
      overlapCacheIds = detectOverlappingSubtitles(getSubtitleList())
      overlapCacheDirty = false
    }
    return overlapCacheIds
  }

  function resolveRegionRenderId(subtitle) {
    const segmentId = String(subtitle?.segment_id ?? '').trim()
    if (segmentId) {
      return segmentId
    }
    if (subtitle?.id === undefined || subtitle?.id === null) {
      return null
    }
    return String(subtitle.id)
  }

  function isRegionSelected(regionId) {
    const selectedSubtitleId = getSelectedSubtitleId()
    if (selectedSubtitleId === null || selectedSubtitleId === undefined) {
      return false
    }

    if (String(selectedSubtitleId) === String(regionId)) {
      return true
    }

    if (useEditorV2) {
      const localId = resolveEditorV2LocalId(regionId)
      return Boolean(localId && String(selectedSubtitleId) === String(localId))
    }

    return false
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

    // 拖拽结束后检测并标记重叠区域
    markOverlapCacheDirty()
    checkAndMarkOverlaps({ forceRefresh: true })
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

  // ============ Region 事件 ============

  /**
   * 设置 Region 事件监听
   */
  function setupRegionEvents(wavesurferRef) {
    const regionsPlugin = regionsPluginRef.value
    if (!regionsPlugin) return

    regionsPlugin.on('region-update', (region, side) => {
      const diagConfig = getWaveformDragDiagnosticsConfig()
      if (!diagConfig.logRegionUpdateFlow) return
      logDiag('regions-plugin-region-update', {
        regionId: region.id,
        side: side || null,
        start: region.start,
        end: region.end,
      })
    })

    regionsPlugin.on('region-updated', (region, side) => {
      logDiag('regions-plugin-region-updated', {
        regionId: region.id,
        side: side || null,
        start: region.start,
        end: region.end,
      })
      if (isUpdatingRegions.value) {
        queuePendingRegionCommit(region, side)
        return
      }
      commitRegionTimeChange({
        id: region.id,
        start: region.start,
        end: region.end,
        side: side || null,
        region,
        deferred: false,
      })
    })

    regionsPlugin.on('region-clicked', (region, e) => {
      e.stopPropagation()
      handleRegionClick(region)
    })

    regionsPlugin.on('region-in', (region) => {
      const overlappingIds = getOverlappingIds()
      if (overlappingIds.has(region.id)) {
        region.setOptions({ color: 'rgba(248, 81, 73, 0.5)' })
      } else {
        region.setOptions({ color: OVERLAP_COLORS.hover })
      }
    })

    regionsPlugin.on('region-out', (region) => {
      const overlappingIds = getOverlappingIds()
      const isSelected = isRegionSelected(region.id)

      if (overlappingIds.has(region.id)) {
        region.setOptions({ color: OVERLAP_COLORS.error })
      } else if (isSelected) {
        region.setOptions({ color: OVERLAP_COLORS.selected })
      } else {
        region.setOptions({ color: props.regionColor })
      }
    })
  }

  /**
   * 检测并标记重叠区域
   */
  function checkAndMarkOverlaps(options = {}) {
    const { forceRefresh = false } = options
    const regionsPlugin = regionsPluginRef.value
    if (!regionsPlugin || !isReady.value) return

    const overlappingIds = getOverlappingIds({ forceRefresh })
    const regions = regionsPlugin.getRegions()

    if (!regions || regions.length === 0) return

    regions.forEach((region) => {
      const isOverlapping = overlappingIds.has(region.id)
      const isSelected = isRegionSelected(region.id)

      if (isOverlapping) {
        region.setOptions({ color: OVERLAP_COLORS.error })
      } else if (isSelected) {
        region.setOptions({ color: OVERLAP_COLORS.selected })
      } else {
        region.setOptions({ color: props.regionColor })
      }
    })

    if (overlappingIds.size > 0) {
      console.warn(
        `[WaveformRegions] 检测到 ${overlappingIds.size} 个重叠区域:`,
        Array.from(overlappingIds)
      )
    }
  }

  /**
   * 计算 Region 颜色
   * @param {object} subtitle - 字幕对象
   * @param {Set} overlappingIds - 重叠字幕 ID 集合
   * @returns {string} 颜色值
   */
  function computeRegionColor(subtitle, overlappingIds) {
    const regionId = resolveRegionRenderId(subtitle)
    const isSelected = isRegionSelected(regionId)
    const isOverlapping = overlappingIds.has(regionId)

    if (isOverlapping) return OVERLAP_COLORS.error
    if (isSelected) return OVERLAP_COLORS.selected
    return props.regionColor
  }

  function addSubtitleRegion(regionsPlugin, subtitle, color, regionId) {
    return regionsPlugin.addRegion({
      id: regionId,
      start: subtitle.start,
      end: subtitle.end,
      color,
      minLength: resolveRegionMinLength(),
      drag: props.dragEnabled,
      resize: props.resizeEnabled,
    })
  }

  /**
   * 渲染字幕区域（增量更新算法，避免闪烁）
   *
   * 算法策略：
   * 1. 获取现有 regions 构建 Map
   * 2. 遍历字幕数据，对比现有 region：
   *    - 命中强制重建或时间变化 → remove + addRegion 重建
   *    - 仅颜色变化 → setOptions({ color }) 轻量更新
   *    - 不存在 → addRegion 添加
   * 3. 删除不再存在的 regions
   *
   * 优势：避免 clearRegions() 导致的视觉闪烁
   */
  function renderSubtitleRegions(subtitleListOverride = null, options = {}) {
    const { forceRecreateAll = false } = options
    const regionsPlugin = regionsPluginRef.value
    const subtitleList = Array.isArray(subtitleListOverride)
      ? subtitleListOverride
      : getSubtitleList()

    if (!isReady.value) {
      console.warn('[WaveformRegions] renderSubtitleRegions: 波形未就绪，跳过渲染')
      return
    }
    if (!regionsPlugin) {
      console.warn('[WaveformRegions] renderSubtitleRegions: regions 插件未加载，跳过渲染')
      return
    }

    const normalizedSubtitles = subtitleList
      .map((subtitle) => {
        const regionId = resolveRegionRenderId(subtitle)
        if (!regionId) {
          return null
        }
        return {
          ...subtitle,
          regionId,
        }
      })
      .filter(Boolean)

    const subtitleCount = normalizedSubtitles.length

    // 无字幕时清空所有 regions
    if (subtitleCount === 0) {
      console.log('[WaveformRegions] renderSubtitleRegions: 无字幕数据，清除 regions')
      regionsPlugin.clearRegions()
      return
    }

    // 预先检测重叠区域
    overlapCacheIds = detectOverlappingSubtitles(
      normalizedSubtitles.map((subtitle) => ({
        ...subtitle,
        id: subtitle.regionId,
      }))
    )
    overlapCacheDirty = false
    const overlappingIds = overlapCacheIds

    isUpdatingRegions.value = true
    try {
      // 构建现有 regions 的 Map（id → region）
      const existingRegions = new Map()
      regionsPlugin.getRegions().forEach((region) => {
        existingRegions.set(String(region.id), region)
      })

      // 记录本次需要保留的 region IDs
      const newSubtitleIds = new Set()
      let addedCount = 0
      let updatedCount = 0

      normalizedSubtitles.forEach((subtitle) => {
        if (subtitle.start === undefined || subtitle.end === undefined) {
          console.warn(`[WaveformRegions] 跳过无效字幕: id=${subtitle.regionId}`)
          return
        }

        newSubtitleIds.add(subtitle.regionId)
        const targetColor = computeRegionColor(subtitle, overlappingIds)
        const existing = existingRegions.get(subtitle.regionId)

        if (existing) {
          // 与 addRegion 参数保持一致，避免历史 region 遗留旧的极小宽度阈值。
          existing.minLength = resolveRegionMinLength()
          // 已存在：检查是否需要更新
          const needsTimeUpdate =
            Math.abs(existing.start - subtitle.start) > 0.001 ||
            Math.abs(existing.end - subtitle.end) > 0.001
          const isDomDetached = existing?.element?.isConnected === false
          // 注意：region.color 可能是 undefined，需要通过 element style 获取
          // wavesurfer regions 插件只会在 addRegion/saveRegion 时重跑 virtualAppend；
          // 纯 setOptions 不会重新把已脱挂的 Region DOM 挂回容器。
          // 结构性编辑（merge/split/undo/redo）恰好会让保留字幕的时间范围大幅变化，
          // 继续复用旧 Region 会出现“数据已更新，但 Region DOM 丢失”的假活状态。
          // 因此只要时间变更，或检测到 DOM 已脱挂，就必须 remove + add 完整重建。
          if (forceRecreateAll || needsTimeUpdate || isDomDetached) {
            existing.remove()
            addSubtitleRegion(regionsPlugin, subtitle, targetColor, subtitle.regionId)
            updatedCount++
          } else {
            // 仅更新颜色（选中状态变化等）
            existing.setOptions({ color: targetColor })
          }
        } else {
          // 新增：添加 region
          addSubtitleRegion(regionsPlugin, subtitle, targetColor, subtitle.regionId)
          addedCount++
        }
      })

      // 删除已不存在的 regions
      let removedCount = 0
      existingRegions.forEach((region, id) => {
        if (!newSubtitleIds.has(id)) {
          region.remove()
          removedCount++
        }
      })

      if (addedCount > 0 || updatedCount > 0 || removedCount > 0) {
        console.log(
          `[WaveformRegions] 增量更新: +${addedCount} 新增, ~${updatedCount} 更新, -${removedCount} 删除`
        )
      }
    } finally {
      // 仅保持一个微任务周期的渲染锁，避免快速拖拽时吞掉 region-updated。
      Promise.resolve().then(() => {
        isUpdatingRegions.value = false
        flushPendingRegionCommits()
      })
    }
  }

  /**
   * 调度 Region 更新（带防抖）
   */
  function scheduleRegionUpdate() {
    if (isReady.value && !isUpdatingRegions.value) {
      clearTimeout(regionUpdateTimer)
      regionUpdateTimer = setTimeout(() => {
        renderSubtitleRegions()
      }, 100)
    }
  }

  /**
   * 清理资源
   */
  function cleanup() {
    clearTimeout(regionUpdateTimer)
    clearTimeout(pendingRegionCommitTimer)
    regionUpdateTimer = null
    pendingRegionCommitTimer = null
    pendingRegionCommits.clear()
    overlapCacheIds.clear()
    overlapCacheDirty = true
    debouncedRegionSync.cancel?.()
  }

  return {
    // 状态
    isUpdatingRegions,
    // 方法
    setupRegionEvents,
    renderSubtitleRegions,
    checkAndMarkOverlaps,
    scheduleRegionUpdate,
    flushPendingRegionCommits,
    commitRegionTimeChange,
    handleRegionClick,
    // 清理
    cleanup,
  }
}
