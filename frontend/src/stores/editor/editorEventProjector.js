// V3.2.4+dev.20260314.03: 编辑器事件投影器 - SSE → 命令映射 + 微任务批量
import { useEditorSessionStore } from './editorSessionStore'
import { useEditorDocumentStore } from './editorDocumentStore'
import { useEditorCommandBus } from './editorCommandBus'
import { createInsertSubtitleCommand } from './editorCommandFactory'

let pendingCommands = []
let flushScheduled = false

function secondsToMs(sessionStore, value) {
  return sessionStore.secondsToMs(value || 0)
}

function normalizeChunkId(data) {
  return data?.chunk_uid ?? data?.chunk_id ?? data?.chunk_index ?? null
}

function normalizeLogicalChunkKey(chunkId) {
  if (chunkId === undefined || chunkId === null) {
    return null
  }

  const raw = String(chunkId).trim()
  if (!raw) {
    return null
  }

  const withoutDedupSuffix = raw.split('#')[0]?.trim() || raw
  if (!withoutDedupSuffix) {
    return null
  }

  if (withoutDedupSuffix.startsWith('chunk-')) {
    const suffix = withoutDedupSuffix.slice('chunk-'.length).trim()
    if (/^-?\d+$/.test(suffix)) {
      return Number(suffix)
    }
    return suffix || withoutDedupSuffix
  }

  if (/^-?\d+$/.test(withoutDedupSuffix)) {
    return Number(withoutDedupSuffix)
  }

  return withoutDedupSuffix
}

function getLogicalChunkKeyFromData(data) {
  return normalizeLogicalChunkKey(normalizeChunkId(data))
}

function isSameLogicalChunk(left, right) {
  const leftKey = normalizeLogicalChunkKey(left)
  const rightKey = normalizeLogicalChunkKey(right)
  if (leftKey === null || rightKey === null) {
    return false
  }
  return leftKey === rightKey
}

function hasFinalizedEntityForLogicalChunk(docStore, logicalChunkKey) {
  if (logicalChunkKey === null || logicalChunkKey === undefined) {
    return false
  }

  return docStore.order.some((localId) => {
    const entity = docStore.getEntity(localId)
    const cold = docStore.getCold(localId)
    if (!entity || entity.isDraft) {
      return false
    }
    return isSameLogicalChunk(cold?.chunkId, logicalChunkKey)
  })
}

function collectLocalIdsForLogicalChunk(docStore, logicalChunkKey, options = {}) {
  if (logicalChunkKey === null || logicalChunkKey === undefined) {
    return []
  }

  const {
    draftOnly = false,
  } = options

  return docStore.order.filter((localId) => {
    const entity = docStore.getEntity(localId)
    const cold = docStore.getCold(localId)
    if (!entity) {
      return false
    }
    if (draftOnly && !entity.isDraft) {
      return false
    }
    return isSameLogicalChunk(cold?.chunkId, logicalChunkKey)
  })
}

function uniqueLocalIds(localIds) {
  return [...new Set((localIds || []).filter(Boolean))]
}

function takeReusableLocalId(reusableLocalIds, preferredLocalId = null) {
  if (preferredLocalId) {
    const preferredIndex = reusableLocalIds.indexOf(preferredLocalId)
    if (preferredIndex >= 0) {
      reusableLocalIds.splice(preferredIndex, 1)
    }
    return preferredLocalId
  }

  return reusableLocalIds.shift() || null
}

function mapWords(words, sessionStore) {
  if (!Array.isArray(words)) {
    return null
  }

  return words.map((word) => ({
    startMs: secondsToMs(sessionStore, word?.start),
    endMs: secondsToMs(sessionStore, word?.end ?? word?.start),
    text: word?.text || word?.word || '',
  }))
}

function buildServerEntity(sentence, sessionStore, options = {}) {
  const {
    localId,
    sentenceIndex = null,
    chunkId = null,
    isDraft = false,
    fallbackSource = 'unknown',
  } = options

  return {
    localId,
    text: sentence?.text || '',
    startMs: secondsToMs(sessionStore, sentence?.start),
    endMs: secondsToMs(sessionStore, sentence?.end),
    cold: {
      segmentId: sentence?.segment_id ?? null,
      sentenceIndex,
      chunkId: normalizeLogicalChunkKey(chunkId),
      sourceType: sentence?.source || sentence?.source_type || fallbackSource,
      confidence: sentence?.confidence ?? null,
      displayConfidence: sentence?.display_confidence ?? null,
      confidenceSource: sentence?.confidence_source ?? null,
      words: mapWords(sentence?.words, sessionStore),
      warningType: sentence?.warning_type || 'none',
      originalText: sentence?.original_text ?? sentence?.text ?? null,
      speakerId: isDraft ? null : (sentence?.speaker_id ?? null),
      speakerLabel: isDraft ? null : (sentence?.speaker_label ?? null),
      speakerColorKey: isDraft ? null : (sentence?.speaker_color_key ?? null),
      turnId: isDraft ? null : (sentence?.turn_id ?? null),
      bindingSource: isDraft ? null : (sentence?.binding_source ?? null),
    },
  }
}

function normalizeSegmentId(value) {
  const normalized = String(value ?? '').trim()
  return normalized || null
}

function resolveSentenceIndex(payload) {
  const rawIndex =
    payload?.index
    ?? payload?.sentence_index
    ?? payload?.sentenceIndex
    ?? null

  return rawIndex === undefined || rawIndex === null ? null : rawIndex
}

function resolvePrimaryLocalId(docStore, sentence, sentenceIndex, options = {}) {
  const {
    preferDraft = null,
  } = options

  const segmentId = normalizeSegmentId(sentence?.segment_id)
  if (segmentId) {
    const boundLocalId = docStore.bindingBySegmentId.get(segmentId)
    if (boundLocalId) {
      return boundLocalId
    }
  }

  if (sentenceIndex === null || sentenceIndex === undefined) {
    return null
  }

  return docStore.findLocalIdBySentenceIndex(sentenceIndex, {
    preferDraft,
  })
}

function resolveFinalizedIndices(data) {
  if (Array.isArray(data?.indices)) {
    return data.indices.filter((value) => value !== undefined && value !== null)
  }

  const singleIndex =
    data?.index
    ?? data?.sentence_index
    ?? data?.sentence?.index
    ?? data?.sentence?.sentence_index

  return singleIndex === undefined || singleIndex === null
    ? []
    : [singleIndex]
}

function buildFinalizedSentenceMap(data, indices = []) {
  const sentenceMap = new Map()

  if (data?.sentence && typeof data.sentence === 'object') {
    const sentenceIndex =
      data?.index
      ?? data?.sentence_index
      ?? data?.sentence?.index
      ?? data?.sentence?.sentence_index
      ?? indices[0]

    if (sentenceIndex !== undefined && sentenceIndex !== null) {
      sentenceMap.set(sentenceIndex, data.sentence)
    }
  }

  if (Array.isArray(data?.sentences)) {
    data.sentences.forEach((sentence, index) => {
      const sentenceIndex =
        sentence?.index
        ?? sentence?.sentence_index
        ?? sentence?.sentenceIndex
        ?? indices[index]

      if (sentenceIndex !== undefined && sentenceIndex !== null) {
        sentenceMap.set(sentenceIndex, sentence)
      }
    })
  }

  return sentenceMap
}

function resolveReplacementItems(data, mode = 'replace_chunk') {
  if (mode === 'finalized') {
    const indices = resolveFinalizedIndices(data)
    const sentenceMap = buildFinalizedSentenceMap(data, indices)
    return indices.map((sentenceIndex) => ({
      sentenceIndex,
      sentence: sentenceMap.get(sentenceIndex) || null,
    }))
  }

  const sentences = Array.isArray(data?.sentences) ? data.sentences : []
  const newIndices = Array.isArray(data?.new_indices) ? data.new_indices : []

  return sentences.map((sentence, index) => ({
    sentence,
    sentenceIndex: mode === 'replace_chunk'
      ? (newIndices[index] ?? resolveSentenceIndex(sentence))
      : resolveSentenceIndex(sentence),
  }))
}

export function useEditorEventProjector() {
  function flushPendingCommandsNow() {
    if (pendingCommands.length === 0) {
      flushScheduled = false
      return
    }

    const batch = pendingCommands
    pendingCommands = []
    flushScheduled = false
    const commandBus = useEditorCommandBus()
    commandBus.dispatchBatch(batch)
  }

  function scheduleFlush() {
    if (flushScheduled) return
    flushScheduled = true
    queueMicrotask(() => {
      flushPendingCommandsNow()
    })
  }

  function projectDraft(data) {
    const sessionStore = useEditorSessionStore()
    const docStore = useEditorDocumentStore()

    const sentence = data.sentence
    const sentenceIndex = data.index
    const logicalChunkKey = getLogicalChunkKeyFromData(data)

    const existingLocalId = resolvePrimaryLocalId(docStore, sentence, sentenceIndex, {
      preferDraft: true,
    })
    const existingEntity = existingLocalId
      ? docStore.getEntity(existingLocalId)
      : null

    // Why:
    // - 双流中 draft 事件可能晚于 replace_chunk/finalized 到达；
    // - 若该句或该逻辑 chunk 已定稿，再接收 draft 只会把草稿错误投影成“当前真相”。
    if (existingEntity && !existingEntity.isDraft) {
      return false
    }
    if (!existingEntity && hasFinalizedEntityForLogicalChunk(docStore, logicalChunkKey)) {
      return false
    }

    if (existingLocalId) {
      const cmd = {
        type: 'update_text',
        commandId: sessionStore.nextCommandId(),
        source: 'system',
        createdAt: Date.now(),
        localId: existingLocalId,
        before: { text: docStore.getEntity(existingLocalId)?.text || '' },
        after: { text: sentence.text || '' },
        mergeKey: null
      }
      pendingCommands.push(cmd)
    } else {
      const localId = sessionStore.nextLocalId()
      const cmd = createInsertSubtitleCommand({
        source: 'system',
        localId,
        entity: {
          text: sentence.text || '',
          startMs: secondsToMs(sessionStore, sentence.start),
          endMs: secondsToMs(sessionStore, sentence.end),
          isDraft: true
        },
        afterLocalId: null,
        coldInit: buildServerEntity(sentence, sessionStore, {
          localId,
          sentenceIndex,
          chunkId: normalizeChunkId(data),
          isDraft: true,
          fallbackSource: 'sensevoice',
        }).cold,
      })
      pendingCommands.push(cmd)
    }

    scheduleFlush()
    return true
  }

  function projectChunkReplace(data, mode = 'replace_chunk') {
    const sessionStore = useEditorSessionStore()
    const docStore = useEditorDocumentStore()
    flushPendingCommandsNow()
    const chunkId = normalizeChunkId(data)
    const logicalChunkKey = getLogicalChunkKeyFromData(data)
    const replacementItems = resolveReplacementItems(data, mode)

    if (replacementItems.length === 0) {
      return false
    }

    const oldLocalIdsFromIndices = (data.old_indices || [])
      .map((idx) => docStore.findLocalIdBySentenceIndex(idx))
      .filter(Boolean)
    const oldLocalIdsFromPrimaryKeys = replacementItems
      .map(({ sentence, sentenceIndex }) => resolvePrimaryLocalId(docStore, sentence, sentenceIndex))
      .filter(Boolean)
    const oldLocalIds = uniqueLocalIds([
      ...oldLocalIdsFromIndices,
      ...oldLocalIdsFromPrimaryKeys,
      // Why:
      // - 后端 replace_chunk 在极端恢复路径下可能缺失/漂移 old_indices；
      // - 若这里只回收草稿，旧定稿会与新定稿并存，形成“标点差异重复”。
      // 因此 fallback 必须按逻辑 chunk 整体替换，而不是仅替换草稿。
      ...collectLocalIdsForLogicalChunk(docStore, logicalChunkKey),
    ])
    const reusableLocalIds = [...oldLocalIds]

    const newEntities = replacementItems.map(({ sentence, sentenceIndex }) => {
      if (!sentence) {
        return null
      }
      const boundLocalId = resolvePrimaryLocalId(docStore, sentence, sentenceIndex) || null
      const localId = takeReusableLocalId(reusableLocalIds, boundLocalId) || sessionStore.nextLocalId()
      return buildServerEntity(sentence, sessionStore, {
        localId,
        sentenceIndex,
        chunkId,
        isDraft: Boolean(sentence?.is_draft),
        fallbackSource: mode === 'restored'
          ? 'restored'
          : (mode === 'finalized' ? 'finalized' : 'sensevoice'),
      })
    }).filter(Boolean)

    if (newEntities.length !== replacementItems.length) {
      return false
    }

    pendingCommands.push({
      type: 'apply_server_replace',
      commandId: sessionStore.nextCommandId(),
      source: 'system',
      createdAt: Date.now(),
      oldLocalIds,
      newEntities
    })

    scheduleFlush()
    return true
  }

  function projectReplaceChunk(data) {
    return projectChunkReplace(data, 'replace_chunk')
  }

  function projectRestored(data) {
    return projectChunkReplace(data, 'restored')
  }

  function projectFinalized(data) {
    return projectChunkReplace(data, 'finalized')
  }

  function projectRevised(data) {
    const sessionStore = useEditorSessionStore()
    const docStore = useEditorDocumentStore()
    flushPendingCommandsNow()

    const localId = docStore.bindingBySegmentId.get(data.segment_id)
    if (!localId) return

    const entity = docStore.getEntity(localId)
    if (!entity) return

    if (data.text !== undefined && data.text !== entity.text) {
      pendingCommands.push({
        type: 'update_text',
        commandId: sessionStore.nextCommandId(),
        source: 'system',
        createdAt: Date.now(),
        localId,
        before: { text: entity.text },
        after: { text: data.text },
        mergeKey: null
      })
    }

    if (data.start_ms !== undefined || data.end_ms !== undefined) {
      pendingCommands.push({
        type: 'update_timing',
        commandId: sessionStore.nextCommandId(),
        source: 'system',
        createdAt: Date.now(),
        localId,
        before: { startMs: entity.startMs, endMs: entity.endMs },
        after: {
          startMs: data.start_ms ?? entity.startMs,
          endMs: data.end_ms ?? entity.endMs
        }
      })
    }

    scheduleFlush()
  }

  function projectServerSnapshot(segments) {
    const sessionStore = useEditorSessionStore()

    for (const seg of segments) {
      const localId = sessionStore.nextLocalId()
      pendingCommands.push(createInsertSubtitleCommand({
        source: 'rehydrate',
        localId,
        entity: {
          text: seg.text || '',
          startMs: seg.start_ms || 0,
          endMs: seg.end_ms || 0,
          isDraft: false
        },
        afterLocalId: null,
        coldInit: {
          localId,
          segmentId: seg.segment_id,
          sentenceIndex: seg.sentence_index ?? null,
          chunkId: normalizeLogicalChunkKey(seg.chunk_id ?? seg.chunk_uid ?? null),
          sourceType: seg.source_type || 'unknown',
          confidence: seg.confidence ?? null,
          words: null,
          warningType: 'none',
          originalText: null
        }
      }))
    }

    scheduleFlush()
  }

  function reset() {
    pendingCommands = []
    flushScheduled = false
  }

  return {
    projectDraft,
    projectReplaceChunk,
    projectRestored,
    projectFinalized,
    projectRevised,
    projectServerSnapshot,
    reset,
  }
}
