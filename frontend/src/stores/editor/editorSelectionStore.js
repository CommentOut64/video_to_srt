// V3.2.5+dev.20260316.03: 编辑器选择状态真源
import { computed, shallowRef, ref } from 'vue'
import { defineStore } from 'pinia'

function cloneSet(source) {
  return new Set(source || [])
}

export const useEditorSelectionStore = defineStore('editorSelection', () => {
  const activeLocalId = ref(null)
  const rangeAnchorLocalId = ref(null)
  const multiSelectedLocalIds = shallowRef(new Set())
  const searchSelectedSentenceIndices = shallowRef(new Set())

  const multiSelectedCount = computed(() => multiSelectedLocalIds.value.size)
  const hasMultiSelection = computed(() => multiSelectedLocalIds.value.size > 0)
  const searchSelectedCount = computed(() => searchSelectedSentenceIndices.value.size)

  function setActive(localId, options = {}) {
    const { updateAnchor = true } = options
    activeLocalId.value = localId || null
    if (updateAnchor && localId) {
      rangeAnchorLocalId.value = localId
    }
  }

  function replaceMultiSelection(localIds = [], options = {}) {
    const nextSelection = new Set(
      Array.isArray(localIds)
        ? localIds.filter(Boolean)
        : []
    )
    multiSelectedLocalIds.value = nextSelection

    const { activeLocalId: nextActiveLocalId = null, updateAnchor = true } = options
    if (nextActiveLocalId !== undefined) {
      activeLocalId.value = nextActiveLocalId || null
    }
    if (updateAnchor) {
      const fallbackAnchor = nextActiveLocalId || Array.from(nextSelection).at(-1) || null
      rangeAnchorLocalId.value = fallbackAnchor
    }
  }

  function clearMultiSelection(options = {}) {
    multiSelectedLocalIds.value = new Set()
    if (options.clearAnchor) {
      rangeAnchorLocalId.value = null
    }
  }

  function selectOnly(localId) {
    replaceMultiSelection(localId ? [localId] : [], {
      activeLocalId: localId || null,
      updateAnchor: true,
    })
  }

  function toggleMultiSelection(localId, options = {}) {
    if (!localId) return false

    const nextSelection = cloneSet(multiSelectedLocalIds.value)
    const willSelect = !nextSelection.has(localId)
    if (willSelect) {
      nextSelection.add(localId)
    } else {
      nextSelection.delete(localId)
    }
    multiSelectedLocalIds.value = nextSelection

    const { makeActive = true, updateAnchor = willSelect } = options
    if (makeActive) {
      activeLocalId.value = localId
    }
    if (updateAnchor) {
      rangeAnchorLocalId.value = localId
    }
    return willSelect
  }

  function selectRange(targetLocalId, orderedLocalIds = [], options = {}) {
    if (!targetLocalId || !Array.isArray(orderedLocalIds) || orderedLocalIds.length === 0) {
      return []
    }

    const anchorLocalId = rangeAnchorLocalId.value || activeLocalId.value || targetLocalId
    const anchorIndex = orderedLocalIds.indexOf(anchorLocalId)
    const targetIndex = orderedLocalIds.indexOf(targetLocalId)
    if (anchorIndex === -1 || targetIndex === -1) {
      selectOnly(targetLocalId)
      return [targetLocalId]
    }

    const startIndex = Math.min(anchorIndex, targetIndex)
    const endIndex = Math.max(anchorIndex, targetIndex)
    const rangeIds = orderedLocalIds.slice(startIndex, endIndex + 1)
    const { append = false } = options
    const nextSelection = append
      ? cloneSet(multiSelectedLocalIds.value)
      : new Set()

    rangeIds.forEach((localId) => {
      if (localId) {
        nextSelection.add(localId)
      }
    })

    multiSelectedLocalIds.value = nextSelection
    activeLocalId.value = targetLocalId
    if (!rangeAnchorLocalId.value) {
      rangeAnchorLocalId.value = anchorLocalId
    }
    return rangeIds
  }

  function isMultiSelected(localId) {
    return multiSelectedLocalIds.value.has(localId)
  }

  function setSearchSelection(indices = []) {
    searchSelectedSentenceIndices.value = new Set(indices)
  }

  function clearSearchSelection() {
    searchSelectedSentenceIndices.value = new Set()
  }

  function toggleSearchSelection(sentenceIndex, checked) {
    if (sentenceIndex === null || sentenceIndex === undefined) {
      return false
    }

    const normalizedIndex = Number(sentenceIndex)
    const nextSelection = cloneSet(searchSelectedSentenceIndices.value)
    const shouldSelect = checked ?? !nextSelection.has(normalizedIndex)
    if (shouldSelect) {
      nextSelection.add(normalizedIndex)
    } else {
      nextSelection.delete(normalizedIndex)
    }
    searchSelectedSentenceIndices.value = nextSelection
    return shouldSelect
  }

  function isSearchSelected(sentenceIndex) {
    return searchSelectedSentenceIndices.value.has(Number(sentenceIndex))
  }

  function reset() {
    activeLocalId.value = null
    rangeAnchorLocalId.value = null
    multiSelectedLocalIds.value = new Set()
    searchSelectedSentenceIndices.value = new Set()
  }

  return {
    activeLocalId,
    rangeAnchorLocalId,
    multiSelectedLocalIds,
    searchSelectedSentenceIndices,
    multiSelectedCount,
    hasMultiSelection,
    searchSelectedCount,
    setActive,
    replaceMultiSelection,
    clearMultiSelection,
    selectOnly,
    toggleMultiSelection,
    selectRange,
    isMultiSelected,
    setSearchSelection,
    clearSearchSelection,
    toggleSearchSelection,
    isSearchSelected,
    reset,
  }
})
