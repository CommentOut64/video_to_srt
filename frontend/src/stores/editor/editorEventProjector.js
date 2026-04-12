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

    const existingLocalId = docStore.bindingBySentenceIndex.get(sentenceIndex)
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

  function projectReplaceChunk(data) {
    const sessionStore = useEditorSessionStore()
    const docStore = useEditorDocumentStore()
    flushPendingCommandsNow()
    const chunkId = normalizeChunkId(data)
    const logicalChunkKey = getLogicalChunkKeyFromData(data)
    const sentences = Array.isArray(data?.sentences) ? data.sentences : []
    const newIndices = Array.isArray(data?.new_indices) ? data.new_indices : []

    if (sentences.length === 0 || newIndices.length !== sentences.length) {
      return false
    }

    const oldLocalIdsFromIndices = (data.old_indices || [])
      .map(idx => docStore.bindingBySentenceIndex.get(idx))
      .filter(Boolean)
    const oldLocalIds = uniqueLocalIds([
      ...oldLocalIdsFromIndices,
      // Why:
      // - 后端 replace_chunk 在极端恢复路径下可能缺失/漂移 old_indices；
      // - 若这里只回收草稿，旧定稿会与新定稿并存，形成“标点差异重复”。
      // 因此 fallback 必须按逻辑 chunk 整体替换，而不是仅替换草稿。
      ...collectLocalIdsForLogicalChunk(docStore, logicalChunkKey),
    ])
    const reusableLocalIds = [...oldLocalIds]

    const newEntities = sentences.map((sentence, i) => {
      const sentenceIndex = newIndices[i]
      if (sentenceIndex === undefined || sentenceIndex === null) {
        return null
      }
      const boundLocalId = docStore.bindingBySentenceIndex.get(sentenceIndex) || null
      const localId = takeReusableLocalId(reusableLocalIds, boundLocalId) || sessionStore.nextLocalId()
      return buildServerEntity(sentence, sessionStore, {
        localId,
        sentenceIndex,
        chunkId,
        fallbackSource: 'sensevoice',
      })
    }).filter(Boolean)

    if (newEntities.length !== sentences.length) {
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

  function projectRestored(data) {
    const sessionStore = useEditorSessionStore()
    const docStore = useEditorDocumentStore()
    flushPendingCommandsNow()
    const sentences = Array.isArray(data?.sentences) ? data.sentences : []
    if (sentences.length === 0) {
      return false
    }

    const chunkId = normalizeChunkId(data)
    const oldLocalIds = chunkId === null
      ? []
      : docStore.order.filter((localId) => docStore.getCold(localId)?.chunkId === chunkId)
    const fallbackReplaceIds = sentences
      .map((sentence) => docStore.bindingBySentenceIndex.get(sentence?.index))
      .filter(Boolean)
    const replaceIds = [...new Set(oldLocalIds.length > 0 ? oldLocalIds : fallbackReplaceIds)]

    const newEntities = sentences.map((sentence) => {
      const sentenceIndex = sentence?.index ?? null
      const boundLocalId = sentenceIndex === null
        ? null
        : docStore.bindingBySentenceIndex.get(sentenceIndex)
      const localId = boundLocalId || sessionStore.nextLocalId()

      return buildServerEntity(sentence, sessionStore, {
        localId,
        sentenceIndex,
        chunkId,
        isDraft: Boolean(sentence?.is_draft),
        fallbackSource: 'restored',
      })
    })

    pendingCommands.push({
      type: 'apply_server_replace',
      commandId: sessionStore.nextCommandId(),
      source: 'system',
      createdAt: Date.now(),
      oldLocalIds: replaceIds,
      newEntities,
    })

    scheduleFlush()
    return true
  }

  function projectFinalized(data) {
    const sessionStore = useEditorSessionStore()
    const docStore = useEditorDocumentStore()
    flushPendingCommandsNow()
    const indices = resolveFinalizedIndices(data)
    if (indices.length === 0) {
      return false
    }
    const sentenceMap = buildFinalizedSentenceMap(data, indices)
    const chunkId = normalizeChunkId(data)
    const logicalChunkKey = getLogicalChunkKeyFromData(data)
    const draftLocalIdsForChunk = uniqueLocalIds(
      collectLocalIdsForLogicalChunk(docStore, logicalChunkKey, { draftOnly: true })
    )

    if (sentenceMap.size === indices.length) {
      const replaceLocalIds = uniqueLocalIds([
        ...draftLocalIdsForChunk,
        ...indices
          .map((sentenceIndex) => docStore.bindingBySentenceIndex.get(sentenceIndex))
          .filter((localId) => {
            if (!localId) {
              return false
            }
            const entity = docStore.getEntity(localId)
            return Boolean(entity?.isDraft)
          }),
      ])
      const reusableLocalIds = [...replaceLocalIds]

      const newEntities = indices.map((sentenceIndex) => {
        const sentence = sentenceMap.get(sentenceIndex)
        const boundLocalId = docStore.bindingBySentenceIndex.get(sentenceIndex) || null
        const localId = takeReusableLocalId(reusableLocalIds, boundLocalId) || sessionStore.nextLocalId()
        return buildServerEntity(sentence, sessionStore, {
          localId,
          sentenceIndex,
          chunkId,
          isDraft: false,
          fallbackSource: 'finalized',
        })
      })

      pendingCommands.push({
        type: 'apply_server_replace',
        commandId: sessionStore.nextCommandId(),
        source: 'system',
        createdAt: Date.now(),
        oldLocalIds: replaceLocalIds,
        newEntities,
      })

      scheduleFlush()
      return true
    }

    const localIds = []
    const updates = []
    for (const sentenceIndex of indices) {
      const localId = docStore.bindingBySentenceIndex.get(sentenceIndex)
      if (!localId) {
        continue
      }

      const entity = docStore.getEntity(localId)
      if (!entity) {
        continue
      }

      const sentence = sentenceMap.get(sentenceIndex)
      if (sentence) {
        const finalizedEntity = buildServerEntity(sentence, sessionStore, {
          localId,
          sentenceIndex,
          chunkId,
          isDraft: false,
          fallbackSource: 'finalized',
        })
        updates.push({
          localId,
          text: finalizedEntity.text,
          startMs: finalizedEntity.startMs,
          endMs: finalizedEntity.endMs,
          cold: finalizedEntity.cold,
        })
        continue
      }

      if (entity.isDraft) {
        localIds.push(localId)
      }
    }

    if (updates.length > 0 || localIds.length > 0) {
      pendingCommands.push({
        type: 'finalize_draft_chunk',
        commandId: sessionStore.nextCommandId(),
        source: 'system',
        createdAt: Date.now(),
        localIds,
        updates,
      })
    }

    scheduleFlush()
    return true
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
          chunkId: null,
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
