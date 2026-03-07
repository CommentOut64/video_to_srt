/**
 * 本地字幕 ID 生成工具
 *
 * 设计目标：
 * 1. 彻底摆脱 `Date.now()` 单点时间戳造成的主键碰撞。
 * 2. 在同一毫秒内连续插入/切分时仍保证唯一。
 * 3. 保持纯前端实现，避免依赖后端往返。
 */

let localSubtitleSequence = 0

function nextSequencePart() {
  localSubtitleSequence = (localSubtitleSequence + 1) % Number.MAX_SAFE_INTEGER
  return localSubtitleSequence.toString(36)
}

function getPerformancePart() {
  if (typeof performance === 'undefined' || typeof performance.now !== 'function') {
    return '0'
  }
  const microTick = Math.floor(performance.now() * 1000)
  return Math.max(0, microTick).toString(36)
}

function getRandomPart() {
  return Math.random().toString(36).slice(2, 8)
}

export function createLocalSubtitleId(prefix = 'subtitle') {
  const timePart = Date.now().toString(36)
  const performancePart = getPerformancePart()
  const sequencePart = nextSequencePart()
  const randomPart = getRandomPart()
  return `${prefix}-${timePart}-${performancePart}-${sequencePart}-${randomPart}`
}

export function createSplitSubtitleIds(baseId = 'subtitle') {
  const splitGroupId = createLocalSubtitleId('split')
  return {
    leftId: `${baseId}__split_left__${splitGroupId}`,
    rightId: `${baseId}__split_right__${splitGroupId}`,
  }
}
