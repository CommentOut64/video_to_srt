// V3.2.5+dev.20260315.16: 服务端字幕投影辅助函数，避免 V2 下整表回拉覆盖本地真源

function toMs(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) {
    return 0
  }
  return Math.round(numeric * 1000)
}

function normalizeSegmentId(value) {
  return String(value ?? '').trim()
}

export function shouldIgnoreProjectAckEvent(data, currentSessionId) {
  const normalizedSessionId = String(currentSessionId ?? '').trim()
  if (!normalizedSessionId) {
    return false
  }

  if (data?.source !== 'project_api' || data?.is_update !== true) {
    return false
  }

  const eventSessionId = String(
    data?.editor_session_id
    ?? data?.editorSessionId
    ?? data?.session_id
    ?? ''
  ).trim()

  return Boolean(eventSessionId) && eventSessionId === normalizedSessionId
}

function resolveSentenceIndex(rawSegment) {
  const rawIndex = rawSegment?.legacy_index ?? rawSegment?.sentence_index ?? rawSegment?.index
  const numeric = Number(rawIndex)
  return Number.isFinite(numeric) ? numeric : null
}

function resolvePatchLocalId(docStore, rawSentence) {
  const segmentId = normalizeSegmentId(rawSentence?.segment_id)
  if (segmentId) {
    const localId = docStore.bindingBySegmentId.get(segmentId)
    if (localId) {
      return localId
    }
  }

  const sentenceIndex = resolveSentenceIndex(rawSentence)
  if (sentenceIndex !== null) {
    const localId = docStore.findLocalIdBySentenceIndex(sentenceIndex)
    if (localId) {
      return localId
    }
  }

  return null
}

function normalizeWords(words) {
  if (!Array.isArray(words)) {
    return null
  }

  return words.map((word) => ({
    ...word,
    startMs: word?.start_ms ?? toMs(word?.start ?? 0),
    endMs: word?.end_ms ?? toMs(word?.end ?? word?.start ?? 0),
    text: word?.text ?? word?.word ?? '',
  }))
}

function buildHotEntity(localId, rawSegment) {
  return {
    localId,
    text: String(rawSegment?.text ?? ''),
    startMs: rawSegment?.start_ms ?? toMs(rawSegment?.start ?? 0),
    endMs: rawSegment?.end_ms ?? toMs(rawSegment?.end ?? rawSegment?.start ?? 0),
    isDraft: Boolean(rawSegment?.is_draft),
    isModified: Boolean(rawSegment?.is_modified),
    isDeleted: false,
    revision: 0,
  }
}

function buildColdEntity(localId, rawSegment) {
  return {
    localId,
    segmentId: normalizeSegmentId(rawSegment?.segment_id) || null,
    sentenceIndex: resolveSentenceIndex(rawSegment),
    chunkId: rawSegment?.chunk_id ?? rawSegment?.chunk_uid ?? null,
    words: normalizeWords(rawSegment?.words),
    confidence: rawSegment?.confidence ?? null,
    displayConfidence: rawSegment?.display_confidence ?? null,
    confidenceSource: rawSegment?.confidence_source ?? null,
    speakerId: rawSegment?.speaker_id ?? null,
    speakerLabel: rawSegment?.speaker_label ?? null,
    speakerColorKey: rawSegment?.speaker_color_key ?? null,
    turnId: rawSegment?.turn_id ?? null,
    bindingSource: rawSegment?.binding_source ?? null,
    sourceType: rawSegment?.source_type ?? rawSegment?.source ?? null,
    warningType: rawSegment?.warning_type || 'none',
    originalText: rawSegment?.original_text ?? null,
  }
}

function resolveUpsertLocalId(docStore, rawSegment, explicitLocalId = null) {
  if (explicitLocalId && docStore.getEntity(explicitLocalId)) {
    return explicitLocalId
  }

  const segmentId = normalizeSegmentId(rawSegment?.segment_id)
  if (segmentId) {
    const boundLocalId = docStore.bindingBySegmentId.get(segmentId)
    if (boundLocalId) {
      return boundLocalId
    }
  }

  const sentenceIndex = resolveSentenceIndex(rawSegment)
  if (sentenceIndex !== null) {
    const boundLocalId = docStore.findLocalIdBySentenceIndex(sentenceIndex)
    if (boundLocalId) {
      return boundLocalId
    }
  }

  return explicitLocalId || segmentId || null
}

function syncColdBindings(docStore, localId, currentCold, nextCold) {
  if (!nextCold) {
    return
  }

  if (currentCold?.segmentId && currentCold.segmentId !== nextCold.segmentId) {
    docStore.bindingBySegmentId.delete(currentCold.segmentId)
  }

  if (currentCold) {
    docStore._applyColdUpdate(localId, nextCold)
  }

  if (nextCold.segmentId) {
    docStore.updateColdBinding(localId, nextCold.segmentId)
  }
}

export function upsertServerSegment(docStore, rawSegment, options = {}) {
  if (!docStore || !rawSegment || typeof rawSegment !== 'object') {
    return null
  }

  const localId = resolveUpsertLocalId(docStore, rawSegment, options.localId ?? null)
  if (!localId) {
    return null
  }

  const nextHot = buildHotEntity(localId, rawSegment)
  const nextCold = buildColdEntity(localId, rawSegment)
  const currentEntity = docStore.getEntity(localId)
  const currentCold = docStore.getCold(localId)

  if (!currentEntity) {
    docStore._applyInsert(localId, nextHot, nextCold, options.afterLocalId ?? null)
    docStore.clearDirtyFlags([localId])
    return localId
  }

  docStore._applyUpdate(localId, {
    text: nextHot.text,
    startMs: nextHot.startMs,
    endMs: nextHot.endMs,
    isDraft: nextHot.isDraft,
    isModified: nextHot.isModified,
    isDeleted: false,
  })
  docStore._applyReorder(localId)
  syncColdBindings(docStore, localId, currentCold, nextCold)
  docStore.clearDirtyFlags([localId])
  return localId
}

export function deleteServerSegment(docStore, segmentId) {
  if (!docStore) {
    return false
  }

  const normalizedSegmentId = normalizeSegmentId(segmentId)
  if (!normalizedSegmentId) {
    return false
  }

  const matchedLocalIds = []
  for (const [localId, cold] of docStore.coldEntities.entries()) {
    if (cold?.segmentId === normalizedSegmentId) {
      matchedLocalIds.push(localId)
    }
  }

  if (matchedLocalIds.length === 0) {
    return false
  }

  matchedLocalIds.forEach((localId) => {
    docStore._applyDelete(localId, { trackTombstone: false })
  })
  docStore.clearDirtyFlags(matchedLocalIds)
  return true
}

export function applySentencePatch(docStore, rawSentence) {
  if (!docStore || !rawSentence || typeof rawSentence !== 'object') {
    return false
  }

  const localId = resolvePatchLocalId(docStore, rawSentence)
  if (!localId) {
    return false
  }

  const entity = docStore.getEntity(localId)
  if (!entity) {
    return false
  }

  const hotPatch = {}
  if (rawSentence.text !== undefined) {
    hotPatch.text = String(rawSentence.text ?? '')
  }
  if (rawSentence.start !== undefined || rawSentence.start_ms !== undefined) {
    hotPatch.startMs = rawSentence.start_ms ?? toMs(rawSentence.start ?? 0)
  }
  if (rawSentence.end !== undefined || rawSentence.end_ms !== undefined) {
    hotPatch.endMs = rawSentence.end_ms ?? toMs(rawSentence.end ?? rawSentence.start ?? 0)
  }
  if (rawSentence.is_draft !== undefined) {
    hotPatch.isDraft = Boolean(rawSentence.is_draft)
  }
  if (rawSentence.is_modified !== undefined) {
    hotPatch.isModified = Boolean(rawSentence.is_modified)
  }

  if (Object.keys(hotPatch).length > 0) {
    docStore._applyUpdate(localId, {
      ...hotPatch,
      isDeleted: false,
    })
    if (hotPatch.startMs !== undefined || hotPatch.endMs !== undefined) {
      docStore._applyReorder(localId)
    }
  }

  const currentCold = docStore.getCold(localId)
  if (!currentCold) {
    return Object.keys(hotPatch).length > 0
  }

  const sentenceIndex = resolveSentenceIndex(rawSentence) ?? currentCold.sentenceIndex ?? null
  const coldPatch = {}
  if (rawSentence.segment_id !== undefined) {
    coldPatch.segmentId = normalizeSegmentId(rawSentence.segment_id) || null
  }
  if (rawSentence.chunk_id !== undefined || rawSentence.chunk_uid !== undefined) {
    coldPatch.chunkId = rawSentence.chunk_id ?? rawSentence.chunk_uid ?? null
  }
  if (Array.isArray(rawSentence.words)) {
    coldPatch.words = normalizeWords(rawSentence.words)
  }
  if (rawSentence.confidence !== undefined) {
    coldPatch.confidence = rawSentence.confidence
  }
  if (rawSentence.display_confidence !== undefined) {
    coldPatch.displayConfidence = rawSentence.display_confidence
  }
  if (rawSentence.confidence_source !== undefined) {
    coldPatch.confidenceSource = rawSentence.confidence_source
  }
  if (rawSentence.warning_type !== undefined) {
    coldPatch.warningType = rawSentence.warning_type || 'none'
  }
  if (rawSentence.original_text !== undefined) {
    coldPatch.originalText = rawSentence.original_text
  }
  if (rawSentence.speaker_id !== undefined) {
    coldPatch.speakerId = rawSentence.speaker_id
  }
  if (rawSentence.speaker_label !== undefined) {
    coldPatch.speakerLabel = rawSentence.speaker_label
  }
  if (rawSentence.speaker_color_key !== undefined) {
    coldPatch.speakerColorKey = rawSentence.speaker_color_key
  }
  if (rawSentence.turn_id !== undefined) {
    coldPatch.turnId = rawSentence.turn_id
  }
  if (rawSentence.binding_source !== undefined) {
    coldPatch.bindingSource = rawSentence.binding_source
  }
  if (rawSentence.source !== undefined || rawSentence.source_type !== undefined) {
    coldPatch.sourceType = rawSentence.source_type ?? rawSentence.source
  }

  if (Object.keys(coldPatch).length > 0) {
    syncColdBindings(docStore, localId, currentCold, {
      ...currentCold,
      ...coldPatch,
      localId,
      sentenceIndex,
    })
  }

  docStore.clearDirtyFlags([localId])

  return Object.keys(hotPatch).length > 0 || Object.keys(coldPatch).length > 0
}

export function replaceServerSegments(docStore, segments) {
  if (!docStore) {
    return 0
  }

  docStore.clearDocument()
  const normalizedSegments = Array.isArray(segments) ? segments : []
  normalizedSegments.forEach((segment) => {
    upsertServerSegment(docStore, segment)
  })
  return normalizedSegments.length
}
