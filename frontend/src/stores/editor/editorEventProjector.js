// V3.2.4+dev.20260314.03: 编辑器事件投影器 - SSE → 命令映射 + 微任务批量
import { useEditorSessionStore } from './editorSessionStore'
import { useEditorDocumentStore } from './editorDocumentStore'
import { useEditorCommandBus } from './editorCommandBus'

let pendingCommands = []
let flushScheduled = false

function secondsToMs(sessionStore, value) {
  return sessionStore.secondsToMs(value || 0)
}

function normalizeChunkId(data) {
  return data?.chunk_uid ?? data?.chunk_id ?? data?.chunk_index ?? null
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
      chunkId,
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

export function useEditorEventProjector() {

  function scheduleFlush() {
    if (flushScheduled) return
    flushScheduled = true
    queueMicrotask(() => {
      flushScheduled = false
      if (pendingCommands.length === 0) return
      const batch = pendingCommands
      pendingCommands = []
      const commandBus = useEditorCommandBus()
      commandBus.dispatchBatch(batch)
    })
  }

  function projectDraft(data) {
    const sessionStore = useEditorSessionStore()
    const docStore = useEditorDocumentStore()

    const sentence = data.sentence
    const sentenceIndex = data.index

    const existingLocalId = docStore.bindingBySentenceIndex.get(sentenceIndex)
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
      const cmd = {
        type: 'insert_subtitle',
        commandId: sessionStore.nextCommandId(),
        source: 'system',
        createdAt: Date.now(),
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
      }
      pendingCommands.push(cmd)
    }

    scheduleFlush()
  }

  function projectReplaceChunk(data) {
    const sessionStore = useEditorSessionStore()
    const docStore = useEditorDocumentStore()
    const chunkId = normalizeChunkId(data)

    const oldLocalIds = (data.old_indices || [])
      .map(idx => docStore.bindingBySentenceIndex.get(idx))
      .filter(Boolean)

    const newEntities = data.sentences.map((sentence, i) => {
      const sentenceIndex = data.new_indices[i]
      const localId = docStore.bindingBySentenceIndex.get(sentenceIndex) || sessionStore.nextLocalId()
      return buildServerEntity(sentence, sessionStore, {
        localId,
        sentenceIndex,
        chunkId,
        fallbackSource: 'sensevoice',
      })
    })

    pendingCommands.push({
      type: 'apply_server_replace',
      commandId: sessionStore.nextCommandId(),
      source: 'system',
      createdAt: Date.now(),
      oldLocalIds,
      newEntities
    })

    scheduleFlush()
  }

  function projectRestored(data) {
    const sessionStore = useEditorSessionStore()
    const docStore = useEditorDocumentStore()
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

    const localIds = []
    for (const sentenceIndex of data.indices || []) {
      const localId = docStore.bindingBySentenceIndex.get(sentenceIndex)
      if (localId) {
        const entity = docStore.getEntity(localId)
        if (entity?.isDraft) {
          localIds.push(localId)
        }
      }
    }

    if (localIds.length > 0) {
      pendingCommands.push({
        type: 'finalize_draft_chunk',
        commandId: sessionStore.nextCommandId(),
        source: 'system',
        createdAt: Date.now(),
        localIds
      })
    }

    scheduleFlush()
  }

  function projectRevised(data) {
    const sessionStore = useEditorSessionStore()
    const docStore = useEditorDocumentStore()

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
      pendingCommands.push({
        type: 'insert_subtitle',
        commandId: sessionStore.nextCommandId(),
        source: 'rehydrate',
        createdAt: Date.now(),
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
      })
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
