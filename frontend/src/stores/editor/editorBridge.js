// V3.2.4+dev.20260314.03: 新旧编辑器桥接适配器
import { useEditorDocumentStore } from './editorDocumentStore'
import { useEditorCommandBus } from './editorCommandBus'
import { useEditorEventProjector } from './editorEventProjector'
import { createUpdateTextCommand, createUpdateTimingCommand } from './editorCommandFactory'

export function createEditorBridge() {
  const docStore = useEditorDocumentStore()
  const commandBus = useEditorCommandBus()
  const projector = useEditorEventProjector()

  return {
    // 兼容旧 API：获取字幕列表
    getSubtitles() {
      return docStore.order.map(id => {
        const hot = docStore.getEntity(id)
        const cold = docStore.getCold(id)
        return {
          localId: id,
          text: hot.text,
          startMs: hot.startMs,
          endMs: hot.endMs,
          isDraft: hot.isDraft,
          segmentId: cold?.segmentId,
          sentenceIndex: cold?.sentenceIndex
        }
      })
    },

    // 兼容旧 API：更新文本
    updateText(localId, text) {
      commandBus.dispatch(createUpdateTextCommand({
        localId,
        before: { text: docStore.getEntity(localId)?.text || '' },
        after: { text },
        mergeKey: null,
      }))
    },

    // 兼容旧 API：更新时间
    updateTiming(localId, startMs, endMs) {
      const entity = docStore.getEntity(localId)
      commandBus.dispatch(createUpdateTimingCommand({
        localId,
        before: { startMs: entity.startMs, endMs: entity.endMs },
        after: { startMs, endMs },
      }))
    },

    // SSE 事件入口
    handleSSE(eventType, data) {
      switch (eventType) {
        case 'subtitle.draft':
          projector.projectDraft(data)
          break
        case 'subtitle.replace_chunk':
          projector.projectReplaceChunk(data)
          break
        case 'subtitle.restored':
          projector.projectRestored(data)
          break
        case 'subtitle.finalized':
          projector.projectFinalized(data)
          break
        case 'subtitle.revised':
          projector.projectRevised(data)
          break
      }
    }
  }
}
