// V3.2.4+dev.20260314.03: 编辑器内核初始化入口
import { useEditorSessionStore } from './editorSessionStore'
import { useEditorDocumentStore } from './editorDocumentStore'
import { useEditorCommandBus } from './editorCommandBus'
import { useEditorPersistenceClient } from './editorPersistenceClient'
import { useEditorSyncEngine } from './editorSyncEngine'
import { useEditorEventProjector } from './editorEventProjector'

export function initEditorCore(projectId, jobId) {
  const sessionStore = useEditorSessionStore()
  const commandBus = useEditorCommandBus()
  const persistenceClient = useEditorPersistenceClient()
  const syncEngine = useEditorSyncEngine()

  sessionStore.openSession(projectId, jobId)
  persistenceClient.init(projectId, sessionStore.sessionId)

  // 连接命令总线到持久化和同步
  commandBus.onCommandApplied((command) => {
    persistenceClient.appendCommands([command])
    syncEngine.enqueue(command)
  })

  return {
    sessionStore,
    commandBus,
    persistenceClient,
    syncEngine
  }
}

export function destroyEditorCore() {
  const persistenceClient = useEditorPersistenceClient()
  persistenceClient.destroy()
}
