// V3.2.5+dev.20260314.01: 命令事务式撤销/重做
import { defineStore } from 'pinia'
import { shallowRef, computed } from 'vue'
import { useEditorDocumentStore } from './editorDocumentStore'

export const useEditorHistoryStore = defineStore('editorHistory', () => {
  const LIMIT = 50

  const past = shallowRef([])
  const future = shallowRef([])
  const canUndo = computed(() => past.value.length > 0)
  const canRedo = computed(() => future.value.length > 0)
  let lastMergeKey = null

  function push(doCommand, undoCommand) {
    if (!doCommand || doCommand.source === 'system') return

    const docStore = useEditorDocumentStore()

    if (doCommand.mergeKey && tryMergeWithPrevious(doCommand)) {
      return
    }

    lastMergeKey = doCommand.mergeKey || null

    const entry = {
      entryId: `he_${String(docStore.revision).padStart(5, '0')}`,
      type: doCommand.type,
      doCommands: [doCommand],
      undoCommands: undoCommand ? [undoCommand] : [],
      revisionBefore: docStore.revision - 1,
      revisionAfter: docStore.revision,
      createdAt: Date.now(),
    }

    const newPast = [...past.value, entry]
    if (newPast.length > LIMIT) newPast.shift()
    past.value = newPast
    future.value = []
  }

  function tryMergeWithPrevious(command) {
    if (past.value.length === 0) return false
    const lastEntry = past.value[past.value.length - 1]
    if (lastEntry.type !== 'update_text') return false
    if (lastEntry.doCommands.length !== 1) return false

    const lastDo = lastEntry.doCommands[0]
    if (lastDo.localId !== command.localId) return false
    if (!lastDo.mergeKey || lastDo.mergeKey !== command.mergeKey) return false
    if (lastDo.mergeKey !== lastMergeKey) return false
    if (Date.now() - lastEntry.createdAt > 2000) return false

    const docStore = useEditorDocumentStore()
    const newDo = { ...lastDo, after: command.after }
    const newEntry = {
      ...lastEntry,
      doCommands: [newDo],
      revisionAfter: docStore.revision,
      createdAt: Date.now(),
    }

    const newPast = [...past.value]
    newPast[newPast.length - 1] = newEntry
    past.value = newPast
    lastMergeKey = command.mergeKey

    return true
  }

  function undo() {
    if (past.value.length === 0) return null

    const entry = past.value[past.value.length - 1]
    past.value = past.value.slice(0, -1)
    future.value = [...future.value, entry]

    return entry.undoCommands
  }

  function redo() {
    if (future.value.length === 0) return null

    const entry = future.value[future.value.length - 1]
    future.value = future.value.slice(0, -1)
    past.value = [...past.value, entry]

    return entry.doCommands
  }

  function clear() {
    past.value = []
    future.value = []
  }

  function closeTransaction() {
    lastMergeKey = null
  }

  function reset() {
    clear()
    closeTransaction()
  }

  return { past, future, canUndo, canRedo, push, undo, redo, clear, closeTransaction, reset }
})
