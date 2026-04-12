function normalizeFiniteNumber(value, fallback = 0) {
  const numericValue = Number(value)
  return Number.isFinite(numericValue) ? numericValue : fallback
}

function clampRatio(value) {
  return Math.min(1, Math.max(0, normalizeFiniteNumber(value, 0)))
}

function formatPercent(ratio) {
  const percentage = Math.round(clampRatio(ratio) * 100000) / 1000
  return `${percentage}%`
}

export function isMarkerRegion(startMs, endMs) {
  const normalizedStartMs = normalizeFiniteNumber(startMs, 0)
  const normalizedEndMs = Math.max(normalizedStartMs, normalizeFiniteNumber(endMs, normalizedStartMs))
  return normalizedStartMs === normalizedEndMs
}

export function getRegionPointerMoveThreshold(mode) {
  return mode === 'body' ? 3 : 1
}

export function buildRegionInlineStyle(options = {}) {
  const startMs = Math.max(0, normalizeFiniteNumber(options.startMs, 0))
  const endMs = Math.max(startMs, normalizeFiniteNumber(options.endMs, startMs))
  const durationMs = Math.max(endMs, normalizeFiniteNumber(options.durationMs, 0), 1)
  const color = options.color || 'rgba(88, 166, 255, 0.25)'
  const dragEnabled = options.dragEnabled === true
  const isDragging = options.isDragging === true
  const marker = isMarkerRegion(startMs, endMs)

  return {
    left: formatPercent(startMs / durationMs),
    right: formatPercent((durationMs - endMs) / durationMs),
    backgroundColor: marker ? 'transparent' : color,
    borderLeft: marker ? `2px solid ${color}` : 'none',
    borderRadius: '2px',
    cursor: isDragging ? 'grabbing' : (dragEnabled ? 'grab' : 'default'),
    pointerEvents: 'all',
  }
}
