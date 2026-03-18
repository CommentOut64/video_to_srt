// V3.2.5+dev.20260314.01: 编辑器内核模块索引
export { useEditorSessionStore } from './editorSessionStore'
export { useEditorTimingStore } from './editorTimingStore'
export { useEditorDocumentStore } from './editorDocumentStore'
export { useEditorCommandBus } from './editorCommandBus'
export { useEditorHistoryStore } from './editorHistoryStore'
export { useEditorSelectionStore } from './editorSelectionStore'
export { useEditorDraftStore } from './editorDraftStore'
export { useEditorPersistenceClient } from './editorPersistenceClient'
export { useEditorSyncEngine } from './editorSyncEngine'
export { useEditorEventProjector } from './editorEventProjector'
export {
  createUpdateTextCommand,
  createUpdateTimingCommand,
  createInsertSubtitleCommand,
  createDeleteSubtitleCommand,
  createSplitSubtitleCommand,
  createMergeSubtitlesCommand,
  createMoveBoundaryCommand,
  createBatchReplaceCommand,
  normalizeEditorCommand,
} from './editorCommandFactory'
export { editorReducer } from './editorReducer'
export { createSubtitleHotEntity, createSubtitleColdEntity } from './models'
export { initEditorCore, destroyEditorCore } from './editorCore'
export { createEditorBridge } from './editorBridge'
export {
  applySentencePatch,
  deleteServerSegment,
  shouldIgnoreProjectAckEvent,
  upsertServerSegment,
} from './editorServerProjection'
