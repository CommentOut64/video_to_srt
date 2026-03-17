// V3.2.5+dev.20260315.16: 编辑器内核初始化入口
import { useEditorDocumentStore } from './editorDocumentStore'
import { useEditorSessionStore } from './editorSessionStore'
import { useEditorCommandBus } from './editorCommandBus'
import { useEditorDraftStore } from './editorDraftStore'
import { useEditorEventProjector } from './editorEventProjector'
import { useEditorHistoryStore } from './editorHistoryStore'
import { useEditorPersistenceClient } from './editorPersistenceClient'
import { useEditorSyncEngine } from './editorSyncEngine'

let activeCoreSessionKey = null
let stopCommandBusBridge = null

export function initEditorCore(projectId, jobId) {
  const nextSessionKey = `${projectId || ''}::${jobId || ''}`
  const docStore = useEditorDocumentStore()
  const sessionStore = useEditorSessionStore()
  const commandBus = useEditorCommandBus()
  const draftStore = useEditorDraftStore()
  const eventProjector = useEditorEventProjector()
  const historyStore = useEditorHistoryStore()
  const persistenceClient = useEditorPersistenceClient()
  const syncEngine = useEditorSyncEngine()

  if (activeCoreSessionKey === nextSessionKey && stopCommandBusBridge) {
    return {
      sessionStore,
      commandBus,
      persistenceClient,
      syncEngine
    }
  }

  if (stopCommandBusBridge) {
    stopCommandBusBridge()
    stopCommandBusBridge = null
  }

  persistenceClient.destroy()
  syncEngine.reset()
  historyStore.reset({ persist: false, unbindPersistence: true })
  draftStore.reset()
  eventProjector.reset()
  docStore.clearDocument()

  sessionStore.openSession(projectId, jobId)
  historyStore.bindPersistenceSession(nextSessionKey)
  persistenceClient.init(projectId, sessionStore.sessionId)

  // 连接命令总线到持久化和同步
  stopCommandBusBridge = commandBus.onCommandApplied((command) => {
    persistenceClient.appendCommands([command])
    syncEngine.enqueue(command)
  })
  activeCoreSessionKey = nextSessionKey

  return {
    sessionStore,
    commandBus,
    persistenceClient,
    syncEngine
  }
}

// V3.2.5+dev.20260315.01: 从旧 projectStore 迁移数据
export function migrateFromProjectStore(projectStore) {
  const docStore = useEditorDocumentStore()

  projectStore.subtitles.forEach((subtitle, index) => {
    const baseStart = projectStore.toBaseTime(subtitle.start ?? 0)
    const baseEnd = projectStore.toBaseTime(subtitle.end ?? 0)
    const hot = {
      localId: subtitle.id,
      text: subtitle.text,
      startMs: Math.round(Number(baseStart) * 1000),
      endMs: Math.round(Number(baseEnd) * 1000),
      isDraft: subtitle.isDraft || false,
      isModified: subtitle.isModified || false,
      isDeleted: false,
      revision: 0
    }

    const cold = {
      localId: subtitle.id,
      segmentId: subtitle.segment_id ?? null,
      sentenceIndex: subtitle.sentenceIndex ?? null,
      chunkId: subtitle.chunk_id ?? null,
      words: Array.isArray(subtitle.words)
        ? subtitle.words.map((word) => ({
            ...word,
            startMs: Math.round(Number(projectStore.toBaseTime(word.start ?? 0)) * 1000),
            endMs: Math.round(Number(projectStore.toBaseTime(word.end ?? word.start ?? 0)) * 1000),
            text: word.text ?? word.word ?? '',
          }))
        : null,
      confidence: subtitle.confidence ?? null,
      displayConfidence: subtitle.display_confidence ?? null,
      confidenceSource: subtitle.confidence_source ?? null,
      speakerId: subtitle.speaker_id ?? null,
      sourceType: subtitle.source ?? null,
      warningType: subtitle.warning_type || 'none',
      originalText: subtitle.originalText ?? subtitle.original_text ?? null,
    }

    docStore._applyInsert(subtitle.id, hot, cold, index > 0 ? projectStore.subtitles[index - 1].id : null)
  })
}

export function destroyEditorCore() {
  if (stopCommandBusBridge) {
    stopCommandBusBridge()
    stopCommandBusBridge = null
  }
  activeCoreSessionKey = null
  const docStore = useEditorDocumentStore()
  const draftStore = useEditorDraftStore()
  const eventProjector = useEditorEventProjector()
  const historyStore = useEditorHistoryStore()
  const persistenceClient = useEditorPersistenceClient()
  const syncEngine = useEditorSyncEngine()
  persistenceClient.destroy()
  syncEngine.reset()
  historyStore.reset({ persist: false, unbindPersistence: true })
  draftStore.reset()
  eventProjector.reset()
  docStore.clearDocument()
}
