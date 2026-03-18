export function focusRowForEditing(localId, stores) {
  if (!localId) {
    return false
  }

  const selectionStore = stores?.selectionStore
  const subtitleDocumentStore = stores?.subtitleDocumentStore

  selectionStore?.selectOnly?.(localId)
  subtitleDocumentStore?.setSelectedSubtitleId?.(localId)
  return true
}

export function resolveDeleteFallbackId(localId, orderedLocalIds = []) {
  if (!localId || !Array.isArray(orderedLocalIds) || orderedLocalIds.length === 0) {
    return null
  }

  const currentIndex = orderedLocalIds.indexOf(localId)
  if (currentIndex === -1) {
    return orderedLocalIds[0] ?? null
  }

  return orderedLocalIds[currentIndex + 1] ?? orderedLocalIds[currentIndex - 1] ?? null
}
