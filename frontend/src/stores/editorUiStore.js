import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useEditorUiStore = defineStore('editorUi', () => {
  const selectedSubtitleId = ref(null)
  const zoomLevel = ref(100)

  function setSelectedSubtitleId(subtitleId) {
    selectedSubtitleId.value = subtitleId || null
  }

  function clearSelectedSubtitleId() {
    selectedSubtitleId.value = null
  }

  function setZoomLevel(value) {
    const normalized = Number(value)
    if (!Number.isFinite(normalized)) {
      return zoomLevel.value
    }
    zoomLevel.value = normalized
    return zoomLevel.value
  }

  return {
    selectedSubtitleId,
    zoomLevel,
    setSelectedSubtitleId,
    clearSelectedSubtitleId,
    setZoomLevel,
  }
})
