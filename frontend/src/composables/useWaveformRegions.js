/**
 * useWaveformRegions - Region 管理逻辑 Composable
 *
 * 职责：Region 渲染、重叠检测、事件绑定、同步节流
 * 提取自 WaveformTimeline/index.vue L507-601, L667-729
 */
import { ref } from 'vue'
import { detectOverlappingSubtitles, OVERLAP_COLORS } from '@/utils/subtitleUtils'

/**
 * 简单防抖工具
 */
function debounce(fn, delay) {
  let timer = null
  return function (...args) {
    if (timer) {
      clearTimeout(timer)
    }
    timer = setTimeout(() => {
      fn.apply(this, args)
    }, delay)
  }
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
 */
export function useWaveformRegions(
  regionsPluginRef,
  projectStore,
  props,
  isReady,
  onSubtitleEdit,
  playbackManager,
  subtitleDocumentStore,
  emit
) {
  // ============ 状态 ============
  const isUpdatingRegions = ref(false)

  // ============ 私有状态 ============
  let regionUpdateTimer = null

  function resolveMaybeRefValue(valueOrRef) {
    if (valueOrRef && typeof valueOrRef === 'object' && 'value' in valueOrRef) {
      return valueOrRef.value
    }
    return valueOrRef
  }

  function getSelectedSubtitleId() {
    return resolveMaybeRefValue(subtitleDocumentStore?.selectedSubtitleId) ?? null
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

  // ============ Region 事件 ============

  /**
   * 设置 Region 事件监听
   */
  function setupRegionEvents(wavesurferRef) {
    const regionsPlugin = regionsPluginRef.value
    if (!regionsPlugin) return

    regionsPlugin.on('region-updated', (region) => {
      if (isUpdatingRegions.value) return
      projectStore.updateSubtitle(
        region.id,
        {
          start: region.start,
          end: region.end,
        },
        { isUserEdit: true }
      )

      // 波形拖拽同步到后端（节流）
      const subtitle = projectStore.subtitles.find((s) => s.id === region.id)
      const syncKey = subtitle?.segment_id || subtitle?.sentenceIndex
      if (syncKey !== undefined && syncKey !== null) {
        debouncedRegionSync(syncKey, region.start, region.end)
      }
      emit('region-update', region)

      // 拖拽结束后检测并标记重叠区域
      checkAndMarkOverlaps()
    })

    regionsPlugin.on('region-clicked', (region, e) => {
      e.stopPropagation()
      setSelectedSubtitleId(region.id)
      playbackManager.seekTo(region.start)
      // V3.2.4+dev.20260228.01: 通过 PlaybackManager 触发播放，
      // 由 WaveformTimeline 的 isPlaying watcher 决定使用真实播放或虚拟时钟
      playbackManager.play()
      emit('region-click', region)
    })

    regionsPlugin.on('region-in', (region) => {
      const overlappingIds = detectOverlappingSubtitles(projectStore.subtitles)
      if (overlappingIds.has(region.id)) {
        region.setOptions({ color: 'rgba(248, 81, 73, 0.5)' })
      } else {
        region.setOptions({ color: OVERLAP_COLORS.hover })
      }
    })

    regionsPlugin.on('region-out', (region) => {
      const overlappingIds = detectOverlappingSubtitles(projectStore.subtitles)
      const isSelected = region.id === getSelectedSubtitleId()

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
  function checkAndMarkOverlaps() {
    const regionsPlugin = regionsPluginRef.value
    if (!regionsPlugin || !isReady.value) return

    const overlappingIds = detectOverlappingSubtitles(projectStore.subtitles)
    const regions = regionsPlugin.getRegions()

    if (!regions || regions.length === 0) return

    regions.forEach((region) => {
      const isOverlapping = overlappingIds.has(region.id)
      const isSelected = region.id === getSelectedSubtitleId()

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
    const isSelected = subtitle.id === getSelectedSubtitleId()
    const isOverlapping = overlappingIds.has(subtitle.id)

    if (isOverlapping) return OVERLAP_COLORS.error
    if (isSelected) return OVERLAP_COLORS.selected
    return props.regionColor
  }

  /**
   * 渲染字幕区域（增量更新算法，避免闪烁）
   *
   * 算法策略：
   * 1. 获取现有 regions 构建 Map
   * 2. 遍历字幕数据，对比现有 region：
   *    - 存在且时间/颜色变化 → setOptions 更新
   *    - 不存在 → addRegion 添加
   * 3. 删除不再存在的 regions
   *
   * 优势：避免 clearRegions() 导致的视觉闪烁
   */
  function renderSubtitleRegions() {
    const regionsPlugin = regionsPluginRef.value

    if (!isReady.value) {
      console.warn('[WaveformRegions] renderSubtitleRegions: 波形未就绪，跳过渲染')
      return
    }
    if (!regionsPlugin) {
      console.warn('[WaveformRegions] renderSubtitleRegions: regions 插件未加载，跳过渲染')
      return
    }

    const subtitleCount = projectStore.subtitles.length

    // 无字幕时清空所有 regions
    if (subtitleCount === 0) {
      console.log('[WaveformRegions] renderSubtitleRegions: 无字幕数据，清除 regions')
      regionsPlugin.clearRegions()
      return
    }

    // 预先检测重叠区域
    const overlappingIds = detectOverlappingSubtitles(projectStore.subtitles)

    isUpdatingRegions.value = true

    // 构建现有 regions 的 Map（id → region）
    const existingRegions = new Map()
    regionsPlugin.getRegions().forEach((region) => {
      existingRegions.set(region.id, region)
    })

    // 记录本次需要保留的 region IDs
    const newSubtitleIds = new Set()
    let addedCount = 0
    let updatedCount = 0

    projectStore.subtitles.forEach((subtitle) => {
      if (subtitle.start === undefined || subtitle.end === undefined) {
        console.warn(`[WaveformRegions] 跳过无效字幕: id=${subtitle.id}`)
        return
      }

      newSubtitleIds.add(subtitle.id)
      const targetColor = computeRegionColor(subtitle, overlappingIds)
      const existing = existingRegions.get(subtitle.id)

      if (existing) {
        // 已存在：检查是否需要更新
        const needsTimeUpdate =
          Math.abs(existing.start - subtitle.start) > 0.001 ||
          Math.abs(existing.end - subtitle.end) > 0.001
        // 注意：region.color 可能是 undefined，需要通过 element style 获取
        // 简化处理：每次都更新颜色（setOptions 内部会做优化）
        if (needsTimeUpdate) {
          existing.setOptions({
            start: subtitle.start,
            end: subtitle.end,
            color: targetColor,
          })
          updatedCount++
        } else {
          // 仅更新颜色（选中状态变化等）
          existing.setOptions({ color: targetColor })
        }
      } else {
        // 新增：添加 region
        regionsPlugin.addRegion({
          id: subtitle.id,
          start: subtitle.start,
          end: subtitle.end,
          color: targetColor,
          drag: props.dragEnabled,
          resize: props.resizeEnabled,
        })
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

    setTimeout(() => {
      isUpdatingRegions.value = false
    }, 100)
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
    regionUpdateTimer = null
  }

  return {
    // 状态
    isUpdatingRegions,
    // 方法
    setupRegionEvents,
    renderSubtitleRegions,
    checkAndMarkOverlaps,
    scheduleRegionUpdate,
    // 清理
    cleanup,
  }
}
