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

// V3.2.5+dev.20260315.01: 从后端加载字幕数据
export async function loadSubtitlesFromBackend(projectId) {
  console.log('[editorCore] 开始加载字幕，projectId:', projectId)
  const docStore = useEditorDocumentStore()
  const { projectApi } = await import('@/services/api')

  // 先清空文档，避免重复插入
  docStore.clearDocument()

  const segments = await projectApi.getSubtitles(projectId)
  console.log('[editorCore] 获取到字幕数量:', segments.length)

  segments.forEach((seg) => {
    const startMs = seg.start_ms ?? Math.round(Number(seg.start ?? 0) * 1000)
    const endMs = seg.end_ms ?? Math.round(Number(seg.end ?? 0) * 1000)
    const hot = {
      localId: seg.segment_id,
      text: seg.text,
      startMs,
      endMs,
      isDraft: Boolean(seg.is_draft),
      isModified: Boolean(seg.is_modified),
      isDeleted: false,
      revision: 0
    }

    const cold = {
      localId: seg.segment_id,
      segmentId: seg.segment_id,
      sentenceIndex: seg.legacy_index ?? seg.sentence_index ?? null,
      chunkId: seg.chunk_id ?? null,
      words: Array.isArray(seg.words)
        ? seg.words.map((word) => ({
            ...word,
            startMs: word.start_ms ?? Math.round(Number(word.start ?? 0) * 1000),
            endMs: word.end_ms ?? Math.round(Number(word.end ?? word.start ?? 0) * 1000),
            text: word.text ?? word.word ?? '',
          }))
        : null,
      confidence: seg.confidence ?? null,
      displayConfidence: seg.display_confidence ?? null,
      confidenceSource: seg.confidence_source ?? null,
      speakerId: seg.speaker_id ?? null,
      sourceType: seg.source_type ?? seg.source ?? null,
      warningType: seg.warning_type || 'none',
      originalText: seg.original_text ?? null,
    }

    docStore._applyInsert(hot.localId, hot, cold, null)
  })

  console.log('[editorCore] 字幕加载完成，docStore.order:', docStore.order.length)
}

export function destroyEditorCore() {
  const persistenceClient = useEditorPersistenceClient()
  persistenceClient.destroy()
}
