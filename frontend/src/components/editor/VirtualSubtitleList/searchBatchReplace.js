export function normalizeSentenceIndex(segment, fallbackIndex) {
  const sentenceIndex = segment?.legacy_index
    ?? segment?.sentence_index
    ?? segment?.sentenceIndex
    ?? fallbackIndex

  if (sentenceIndex === null || sentenceIndex === undefined) {
    return null
  }

  const numericValue = Number(sentenceIndex)
  return Number.isNaN(numericValue) ? null : numericValue
}

export function buildBatchReplaceReplacements({ subtitles, serverSegments, updatedIndices }) {
  const subtitleBySentenceIndex = new Map()
  const subtitleBySegmentId = new Map()

  for (const subtitle of subtitles || []) {
    const sentenceIndex = normalizeSentenceIndex(subtitle, subtitle?.sentenceIndex)
    if (sentenceIndex !== null) {
      subtitleBySentenceIndex.set(sentenceIndex, subtitle)
    }
    if (subtitle?.segment_id) {
      subtitleBySegmentId.set(String(subtitle.segment_id), subtitle)
    }
  }

  const segmentBySentenceIndex = new Map()
  for (const [index, segment] of (serverSegments || []).entries()) {
    const sentenceIndex = normalizeSentenceIndex(segment, index)
    if (sentenceIndex !== null) {
      segmentBySentenceIndex.set(sentenceIndex, segment)
    }
  }

  const replacements = []
  for (const updatedIndex of updatedIndices || []) {
    const numericIndex = Number(updatedIndex)
    if (Number.isNaN(numericIndex)) {
      continue
    }

    const segment = segmentBySentenceIndex.get(numericIndex)
    if (!segment) {
      continue
    }

    const subtitle = subtitleBySentenceIndex.get(numericIndex)
      || subtitleBySegmentId.get(String(segment?.segment_id || ''))
    if (!subtitle) {
      continue
    }

    const nextText = String(segment?.text ?? '')
    const currentText = String(subtitle?.text ?? '')
    if (nextText === currentText) {
      continue
    }

    replacements.push({
      localId: subtitle.localId,
      before: { text: currentText },
      after: { text: nextText },
    })
  }

  return replacements
}
