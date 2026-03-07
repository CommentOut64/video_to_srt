/**
 * 字幕时间窗口工具
 *
 * 目标：
 * - 时间轴渲染仅处理可见窗口附近的字幕。
 * - 依赖字幕按开始时间大体有序这一既有约束，用二分缩小遍历范围。
 */

function normalizeTime(value, fallback = 0) {
  const normalized = Number(value)
  return Number.isFinite(normalized) ? normalized : fallback
}

function findFirstCandidateIndex(subtitles, windowStart) {
  let left = 0
  let right = subtitles.length

  while (left < right) {
    const middle = Math.floor((left + right) / 2)
    const subtitle = subtitles[middle] || {}
    const end = normalizeTime(subtitle.end, normalizeTime(subtitle.start, 0))

    if (end < windowStart) {
      left = middle + 1
    } else {
      right = middle
    }
  }

  return left
}

export function getSubtitlesInTimeWindow(subtitles, windowStart, windowEnd) {
  if (!Array.isArray(subtitles) || subtitles.length === 0) {
    return []
  }

  const safeWindowStart = normalizeTime(windowStart, 0)
  const safeWindowEnd = normalizeTime(windowEnd, Number.POSITIVE_INFINITY)

  if (!Number.isFinite(safeWindowStart) || !Number.isFinite(safeWindowEnd) || safeWindowEnd < safeWindowStart) {
    return subtitles
  }

  const startIndex = findFirstCandidateIndex(subtitles, safeWindowStart)
  const visible = []

  for (let index = startIndex; index < subtitles.length; index += 1) {
    const subtitle = subtitles[index]
    const start = normalizeTime(subtitle?.start, 0)
    const end = normalizeTime(subtitle?.end, start)

    if (start > safeWindowEnd) {
      break
    }

    if (end >= safeWindowStart) {
      visible.push(subtitle)
    }
  }

  return visible
}
