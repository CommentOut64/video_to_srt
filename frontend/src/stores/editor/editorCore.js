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
