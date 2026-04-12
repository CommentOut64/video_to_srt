// V3.2.5+dev.20260314.03: 分组能力懒装配
import { ref } from 'vue'
import { useEditorDocumentStore } from '@/stores/editor/editorDocumentStore'

let groupIndex = null

export function useEditorGroup() {
  const isActive = ref(false)
  const groups = ref([])

  async function activate() {
    if (isActive.value) return

    const docStore = useEditorDocumentStore()

    groupIndex = {
      entities: docStore.entities,
      order: docStore.order
    }

    isActive.value = true
  }

  function groupBy(field) {
    if (!isActive.value || !groupIndex) return []

    const groupMap = new Map()
    for (const localId of groupIndex.order) {
      const entity = groupIndex.entities.get(localId)
      if (!entity) continue

      const key = entity[field] || 'unknown'
      if (!groupMap.has(key)) {
        groupMap.set(key, [])
      }
      groupMap.get(key).push(localId)
    }

    groups.value = Array.from(groupMap.entries()).map(([key, items]) => ({
      key,
      items
    }))

    return groups.value
  }

  function deactivate() {
    groupIndex = null
    groups.value = []
    isActive.value = false
  }

  return { isActive, groups, activate, groupBy, deactivate }
}
