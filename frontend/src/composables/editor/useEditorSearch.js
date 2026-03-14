// V3.2.5+dev.20260314.03: 搜索能力懒装配
import { ref } from 'vue'
import { useEditorDocumentStore } from '@/stores/editor/editorDocumentStore'

let searchIndex = null
let idleTimer = null

export function useEditorSearch() {
  const isActive = ref(false)
  const searchResults = ref([])

  async function activate() {
    if (isActive.value) return

    const docStore = useEditorDocumentStore()

    // 懒加载搜索索引模块
    searchIndex = {
      entities: docStore.entities,
      order: docStore.order.value
    }

    isActive.value = true
  }

  function search(query) {
    if (!isActive.value || !searchIndex) return []

    const results = []
    for (const localId of searchIndex.order) {
      const entity = searchIndex.entities.get(localId)
      if (entity && entity.text.includes(query)) {
        results.push(localId)
      }
    }

    searchResults.value = results
    return results
  }

  function deactivate() {
    searchIndex = null
    searchResults.value = []
    isActive.value = false
  }

  function scheduleDeactivate() {
    clearTimeout(idleTimer)
    idleTimer = setTimeout(deactivate, 30000)
  }

  return { isActive, searchResults, activate, search, deactivate, scheduleDeactivate }
}
