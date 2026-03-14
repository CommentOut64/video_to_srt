// V3.2.5+dev.20260315.01: projectStore 桥接层（新内核 → 旧消费者）
import { computed } from 'vue'
import { useEditorDocumentStore } from './editor/editorDocumentStore'
import { isFeatureEnabled } from '@/config/featureFlags'

export function createProjectStoreBridge() {
  if (!isFeatureEnabled('USE_EDITOR_V2')) {
    return null
  }

  const docStore = useEditorDocumentStore()

  // 只读投影：将新内核数据映射为旧格式
  const subtitles = computed(() => {
    return docStore.order.value.map(localId => {
      const entity = docStore.entities.get(localId)
      const cold = docStore.coldEntities.get(localId)
      if (!entity) return null

      return {
        id: localId,
        text: entity.text,
        start: entity.startMs,
        end: entity.endMs,
        isDraft: entity.isDraft,
        isModified: entity.isModified,
        segment_id: cold?.segmentId,
        sentenceIndex: cold?.sentenceIndex,
      }
    }).filter(Boolean)
  })

  return { subtitles }
}
