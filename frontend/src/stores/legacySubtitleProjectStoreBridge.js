function requireLegacyProjectMethod(projectStore, methodName) {
  const method = projectStore?.[methodName]
  if (typeof method !== 'function') {
    throw new Error(
      `[LegacySubtitleProjectStoreBridge] projectStore.${methodName} 不可用，无法执行 legacy 字幕桥接`
    )
  }
  return method.bind(projectStore)
}

// Trade-off: 把 legacy projectStore 方法查找封装到单独桥接层，
// 让 subtitleDocumentStore 主实现只保留新内核主路径与少量兼容入口，不再直接依赖旧字幕写/恢复接口名。
export function patchLegacySubtitleSegmentId(projectStore, subtitleId, segmentId) {
  if (!subtitleId || !segmentId) {
    return false
  }

  const updateSubtitle = requireLegacyProjectMethod(projectStore, 'updateSubtitle')
  updateSubtitle(subtitleId, { segment_id: String(segmentId) })
  return true
}

export function restoreLegacySubtitleSnapshot(projectStore, segments = [], metadata = {}) {
  const loadFromProjectData = requireLegacyProjectMethod(projectStore, 'loadFromProjectData')
  loadFromProjectData(segments, metadata)
  return Array.isArray(segments) ? segments.length : 0
}
