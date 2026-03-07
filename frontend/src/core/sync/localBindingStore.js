function normalizeQueueKey(rawKey) {
  const normalized = String(rawKey ?? '').trim()
  if (!normalized) return null
  if (/^-?\d+$/.test(normalized)) {
    const numericKey = Number(normalized)
    if (Number.isFinite(numericKey)) {
      return numericKey
    }
  }
  return normalized
}

function findSubtitleByQueueKey(targetStore, rawQueueKey) {
  if (!targetStore?.subtitles) return null
  const queueKey = normalizeQueueKey(rawQueueKey)
  if (queueKey === null) return null

  return targetStore.subtitles.find(
    (item) => item.sentenceIndex === queueKey
      || String(item.segment_id ?? '') === String(queueKey)
      || String(item.id ?? '') === String(queueKey)
  ) || null
}

function resolveProjectSegmentId(projectStore, rawQueueKey) {
  const subtitle = findSubtitleByQueueKey(projectStore, rawQueueKey)
  return subtitle?.segment_id || null
}

async function resolveProjectSegmentIdFromServer({
  projectApi,
  projectId,
  rawQueueKey,
  snapshotCache,
}) {
  const queueKey = normalizeQueueKey(rawQueueKey)
  if (queueKey === null || !projectId) return null

  if (!snapshotCache.value) {
    snapshotCache.value = await projectApi.getSubtitles(projectId)
  }

  const matched = (snapshotCache.value || []).find((segment) => {
    if (String(segment?.segment_id ?? '') === String(queueKey)) {
      return true
    }
    const legacyIndex = segment?.legacy_index ?? segment?.sentenceIndex ?? segment?.id
    return Number.isFinite(Number(legacyIndex)) && Number(legacyIndex) === Number(queueKey)
  })

  return matched?.segment_id || null
}

export {
  normalizeQueueKey,
  findSubtitleByQueueKey,
  resolveProjectSegmentId,
  resolveProjectSegmentIdFromServer,
}