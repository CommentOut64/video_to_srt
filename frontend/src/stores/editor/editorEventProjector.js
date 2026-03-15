// V3.2.4+dev.20260314.03: 编辑器事件投影器 - SSE → 命令映射 + 微任务批量
import { useEditorSessionStore } from './editorSessionStore'
import { useEditorDocumentStore } from './editorDocumentStore'
import { useEditorCommandBus } from './editorCommandBus'

let pendingCommands = []
let flushScheduled = false

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
          startMs: sessionStore.secondsToMs(sentence.start || 0),
          endMs: sessionStore.secondsToMs(sentence.end || 0),
          isDraft: true
        },
        afterLocalId: null,
        coldInit: {
          localId,
          segmentId: null,
          sentenceIndex,
          chunkId: data.chunk_uid || null,
          sourceType: sentence.source || 'sensevoice',
          confidence: sentence.confidence ?? null,
          words: sentence.words
            ? sentence.words.map(w => ({
                startMs: sessionStore.secondsToMs(w.start || 0),
                endMs: sessionStore.secondsToMs(w.end || 0),
                text: w.text || ''
              }))
            : null,
          warningType: 'none',
          originalText: sentence.text || null
        }
      }
      pendingCommands.push(cmd)
    }

    scheduleFlush()
  }

  function projectReplaceChunk(data) {
    const sessionStore = useEditorSessionStore()
    const docStore = useEditorDocumentStore()

    const oldLocalIds = (data.old_indices || [])
      .map(idx => docStore.bindingBySentenceIndex.get(idx))
      .filter(Boolean)

    const newEntities = data.sentences.map((sentence, i) => {
      const localId = sessionStore.nextLocalId()
      const sentenceIndex = data.new_indices[i]
      return {
        localId,
        text: sentence.text || '',
        startMs: sessionStore.secondsToMs(sentence.start || 0),
        endMs: sessionStore.secondsToMs(sentence.end || 0),
        cold: {
          segmentId: null,
          sentenceIndex,
          chunkId: data.chunk_uid || null,
          sourceType: sentence.source || 'sensevoice',
          confidence: sentence.confidence ?? null,
          words: sentence.words
            ? sentence.words.map(w => ({
                startMs: sessionStore.secondsToMs(w.start || 0),
                endMs: sessionStore.secondsToMs(w.end || 0),
                text: w.text || ''
              }))
            : null,
          warningType: 'none',
          originalText: sentence.text || null
        }
      }
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
    projectFinalized,
    projectRevised,
    projectServerSnapshot,
    reset,
  }
}
