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

// V3.2.5+dev.20260315.01: 从旧 projectStore 迁移数据
export function migrateFromProjectStore(projectStore) {
  const docStore = useEditorDocumentStore()
  const commandBus = useEditorCommandBus()

  projectStore.subtitles.forEach((subtitle, index) => {
    const hot = {
      localId: subtitle.id,
      text: subtitle.text,
      startMs: subtitle.start,
      endMs: subtitle.end,
      isDraft: subtitle.isDraft || false,
      isModified: subtitle.isModified || false,
      isDeleted: false,
      revision: 0
    }

    const cold = {
      segmentId: subtitle.segment_id,
      sentenceIndex: subtitle.sentenceIndex
    }

    docStore._applyInsert(subtitle.id, hot, cold, index > 0 ? projectStore.subtitles[index - 1].id : null)
  })
}

// V3.2.5+dev.20260315.01: 从后端加载字幕数据
export async function loadSubtitlesFromBackend(projectId) {
  console.log('[editorCore] 开始加载字幕，projectId:', projectId)
  const docStore = useEditorDocumentStore()
  const { projectApi } = await import('@/services/api')

  const segments = await projectApi.getSubtitles(projectId)
  console.log('[editorCore] 获取到字幕数量:', segments.length)

  segments.forEach((seg) => {
    const hot = {
      localId: seg.segment_id,
      text: seg.text,
      startMs: seg.start,
      endMs: seg.end,
      isDraft: false,
      isModified: false,
      isDeleted: false,
      revision: 0
    }

    const cold = {
      segmentId: seg.segment_id,
      sentenceIndex: seg.legacy_index
    }

    docStore._applyInsert(hot.localId, hot, cold, null)
  })

  console.log('[editorCore] 字幕加载完成，docStore.order:', docStore.order.value.length)
}

export function destroyEditorCore() {
  const persistenceClient = useEditorPersistenceClient()
  persistenceClient.destroy()
}
