/**
 * 时间戳工具
 *
 * 统一处理秒/毫秒时间戳，避免前端不同模块各自实现产生偏差。
 */
export function normalizeTimestamp(value) {
  if (value === null || value === undefined) return null
  const num = Number(value)
  if (Number.isNaN(num) || num <= 0) return null
  if (num < 1_000_000_000_000) {
    return Math.round(num * 1000)
  }
  return Math.round(num)
}
