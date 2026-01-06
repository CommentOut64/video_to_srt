/**
 * 字幕工具函数
 * V3.1.1+dev.20260106.03: 新增重叠检测功能
 */

// 重叠区域颜色配置
export const OVERLAP_COLORS = {
  // 重叠区域警示色（红色半透明）
  error: 'rgba(248, 81, 73, 0.35)',
  // 正常区域颜色（蓝色半透明）
  normal: 'rgba(88, 166, 255, 0.25)',
  // 选中区域颜色（紫色半透明）
  selected: 'rgba(163, 113, 247, 0.35)',
  // hover 颜色
  hover: 'rgba(88, 166, 255, 0.4)'
}

/**
 * 检测字幕是否存在重叠
 * @param {Array} subtitles - 字幕数组，每个元素需有 start 和 end 属性
 * @returns {Set} - 返回有重叠问题的字幕 ID 集合
 */
export function detectOverlappingSubtitles(subtitles) {
  if (!subtitles || subtitles.length < 2) {
    return new Set()
  }

  // 按开始时间排序
  const sorted = [...subtitles].sort((a, b) => a.start - b.start)
  const overlappingIds = new Set()

  // 遍历检测
  for (let i = 0; i < sorted.length - 1; i++) {
    const current = sorted[i]
    const next = sorted[i + 1]

    // 判定重叠: 当前的结束时间 > 下一个的开始时间
    // 增加 0.001 (1ms) 的容差，避免浮点数精度误报
    if (current.end > next.start + 0.001) {
      overlappingIds.add(current.id)
      overlappingIds.add(next.id)
    }
  }

  return overlappingIds
}

/**
 * 检测并标记重叠的 WaveSurfer Regions
 * @param {Object} regionsPlugin - WaveSurfer 的 regions 插件实例
 * @param {Set} overlappingIds - 重叠的字幕 ID 集合
 * @param {string} selectedId - 当前选中的字幕 ID
 * @param {string} normalColor - 正常颜色
 */
export function markOverlappingRegions(regionsPlugin, overlappingIds, selectedId = null, normalColor = OVERLAP_COLORS.normal) {
  if (!regionsPlugin) return

  // 获取所有 regions
  const regions = regionsPlugin.getRegions()
  if (!regions || regions.length === 0) return

  regions.forEach(region => {
    const isOverlapping = overlappingIds.has(region.id)
    const isSelected = region.id === selectedId

    if (isOverlapping) {
      // 重叠区域使用警示色
      region.setOptions({ color: OVERLAP_COLORS.error })
    } else if (isSelected) {
      // 选中区域使用选中色
      region.setOptions({ color: OVERLAP_COLORS.selected })
    } else {
      // 正常区域恢复正常色
      region.setOptions({ color: normalColor })
    }
  })
}

/**
 * V3.1.1+dev.20260106.04: 修复字幕时间戳重叠
 * 策略：如果 前一条.end > 后一条.start，则将 前一条.end 截断为 后一条.start - gapMs
 * 原则：下一句的开始时间是神圣不可侵犯的，因为那是人开始说话的点。
 *
 * @param {Array} subtitles - 字幕数组，每个元素需有 start 和 end 属性
 * @param {number} gapMs - 修复后预留的间隔（毫秒），默认 1ms
 * @returns {Array} - 修复后的字幕数组（返回新数组，不修改原数据）
 */
export function repairSubtitleOverlaps(subtitles, gapMs = 1) {
  if (!subtitles || subtitles.length < 2) {
    return subtitles ? [...subtitles] : []
  }

  // 按开始时间排序
  const sorted = [...subtitles].sort((a, b) => a.start - b.start)
  const gapSec = gapMs / 1000

  // 修复重叠
  for (let i = 0; i < sorted.length - 1; i++) {
    const current = sorted[i]
    const next = sorted[i + 1]

    // 检测重叠
    if (current.end > next.start) {
      // 计算新的结束时间
      let newEnd = next.start - gapSec

      // 确保结束时间不早于开始时间
      if (newEnd < current.start) {
        // 极端情况：至少保留 10ms 的显示时间
        newEnd = current.start + 0.01
      }

      current.end = newEnd
    }
  }

  return sorted
}

/**
 * 获取重叠信息的详细描述
 * @param {Array} subtitles - 字幕数组
 * @param {Set} overlappingIds - 重叠的字幕 ID 集合
 * @returns {Array} - 重叠对的详细信息
 */
export function getOverlapDetails(subtitles, overlappingIds) {
  if (!subtitles || overlappingIds.size === 0) {
    return []
  }

  const sorted = [...subtitles].sort((a, b) => a.start - b.start)
  const details = []

  for (let i = 0; i < sorted.length - 1; i++) {
    const current = sorted[i]
    const next = sorted[i + 1]

    if (current.end > next.start + 0.001) {
      const overlapDuration = current.end - next.start
      details.push({
        firstId: current.id,
        secondId: next.id,
        firstText: current.text?.substring(0, 20) || '',
        secondText: next.text?.substring(0, 20) || '',
        overlapDuration: overlapDuration,
        overlapMs: Math.round(overlapDuration * 1000)
      })
    }
  }

  return details
}
