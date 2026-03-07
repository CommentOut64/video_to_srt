export function normalizeBatchReplaceUpdatedSegments(updatedSegments = []) {
  if (!Array.isArray(updatedSegments)) {
    return []
  }

  return updatedSegments
    .map((item) => {
      const sentenceIndex = Number(
        item?.sentence_index ?? item?.legacy_index ?? item?.index
      )
      if (!Number.isFinite(sentenceIndex)) {
        return null
      }

      return {
        sentenceIndex,
        text: String(item?.text ?? ''),
        isModified: item?.is_modified ?? item?.isModified,
        originalText: item?.original_text ?? item?.originalText ?? null,
      }
    })
    .filter(Boolean)
}

export function applyBatchReplaceUpdatedSegments(projectStore, updatedSegments = []) {
  if (!projectStore || typeof projectStore.updateSubtitle !== 'function') {
    return 0
  }

  const normalizedSegments = normalizeBatchReplaceUpdatedSegments(updatedSegments)
  if (normalizedSegments.length === 0) {
    return 0
  }

  if (typeof projectStore.pauseHistory === 'function') {
    projectStore.pauseHistory()
  }

  let patchedCount = 0
  try {
    for (const segment of normalizedSegments) {
      const subtitle = projectStore.subtitles.find(
        (item) => Number(item?.sentenceIndex) === segment.sentenceIndex
      )
      if (!subtitle) {
        continue
      }

      projectStore.updateSubtitle(
        subtitle.id,
        {
          text: segment.text,
          isModified: Boolean(segment.isModified ?? subtitle.isModified),
          originalText: segment.originalText ?? subtitle.originalText,
        },
        { isUserEdit: false }
      )
      patchedCount += 1
    }
  } finally {
    if (typeof projectStore.resumeHistory === 'function') {
      projectStore.resumeHistory()
    }
  }

  return patchedCount
}
