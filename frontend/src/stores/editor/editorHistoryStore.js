// V3.2.5+dev.20260314.01: 命令事务式撤销/重做
import { defineStore } from 'pinia'
import { shallowRef, computed } from 'vue'
import { useEditorDocumentStore } from './editorDocumentStore'

const STORAGE_KEY_PREFIX = 'editor-history:v1:'
const HISTORY_SNAPSHOT_VERSION = 2

function cloneSerializable(value) {
  if (value === undefined) return undefined
  if (value === null) return null
  return JSON.parse(JSON.stringify(value))
}

function cloneHistoryEntry(entry) {
  if (!entry || typeof entry !== 'object') {
    return null
  }
  return {
    entryId: String(entry.entryId || ''),
    type: String(entry.type || 'transaction'),
    doCommands: cloneSerializable(Array.isArray(entry.doCommands) ? entry.doCommands : []),
    undoCommands: cloneSerializable(Array.isArray(entry.undoCommands) ? entry.undoCommands : []),
    revisionBefore: entry.revisionBefore ?? null,
    revisionAfter: entry.revisionAfter ?? null,
    createdAt: Number(entry.createdAt ?? Date.now()),
  }
}

function cloneEntryList(entries) {
  if (!Array.isArray(entries)) {
    return []
  }
  return entries
    .map((entry) => cloneHistoryEntry(entry))
    .filter(Boolean)
}

function normalizeSessionKey(sessionKey) {
  const normalized = String(sessionKey ?? '').trim()
  return normalized || null
}

function normalizeSegmentId(value) {
  const normalized = String(value ?? '').trim()
  return normalized || null
}

function migrateHistoryEntry(entry) {
  const clonedEntry = cloneHistoryEntry(entry)
  if (!clonedEntry) {
    return null
  }

  const doMerge = clonedEntry.doCommands.find((command) => command?.type === 'merge_subtitles')
  const doSplit = clonedEntry.doCommands.find((command) => command?.type === 'split_subtitle')
  const undoMerge = clonedEntry.undoCommands.find((command) => command?.type === 'merge_subtitles')
  const undoSplit = clonedEntry.undoCommands.find((command) => command?.type === 'split_subtitle')

  if (doMerge && undoSplit) {
    if (!doMerge.keptLocalId && undoSplit.sourceLocalId) {
      doMerge.keptLocalId = undoSplit.sourceLocalId
    }
    if (!doMerge.removedLocalId && undoSplit.createdLocalId) {
      doMerge.removedLocalId = undoSplit.createdLocalId
    }
    if (!normalizeSegmentId(doMerge.keptSegmentId)) {
      const sourceSegmentId = normalizeSegmentId(undoSplit.sourceSegmentId)
      if (sourceSegmentId) {
        doMerge.keptSegmentId = sourceSegmentId
      }
    }
    if (!normalizeSegmentId(doMerge.removedSegmentId)) {
      const createdSegmentId = normalizeSegmentId(undoSplit.createdColdInit?.segmentId)
      if (createdSegmentId) {
        doMerge.removedSegmentId = createdSegmentId
      }
    }
  }

  if (doSplit && undoMerge) {
    if (!doSplit.sourceLocalId && undoMerge.keptLocalId) {
      doSplit.sourceLocalId = undoMerge.keptLocalId
    }
    if (!doSplit.createdLocalId && undoMerge.removedLocalId) {
      doSplit.createdLocalId = undoMerge.removedLocalId
    }
    if (!normalizeSegmentId(doSplit.sourceSegmentId)) {
      const keptSegmentId = normalizeSegmentId(undoMerge.keptSegmentId)
      if (keptSegmentId) {
        doSplit.sourceSegmentId = keptSegmentId
      }
    }
    const createdColdInit = doSplit.createdColdInit && typeof doSplit.createdColdInit === 'object'
      ? doSplit.createdColdInit
      : null
    if (createdColdInit && !normalizeSegmentId(createdColdInit.segmentId)) {
      const removedSegmentId = normalizeSegmentId(undoMerge.removedSegmentId)
      if (removedSegmentId) {
        doSplit.createdColdInit = {
          ...createdColdInit,
          segmentId: removedSegmentId,
        }
      }
    }
  }

  return clonedEntry
}

function migrateHistorySnapshot(snapshot) {
  const migratedPast = cloneEntryList(snapshot?.past)
    .map((entry) => migrateHistoryEntry(entry))
    .filter(Boolean)
  const migratedFuture = cloneEntryList(snapshot?.future)
    .map((entry) => migrateHistoryEntry(entry))
    .filter(Boolean)

  return {
    version: HISTORY_SNAPSHOT_VERSION,
    savedAt: Number(snapshot?.savedAt ?? Date.now()),
    lastMergeKey: typeof snapshot?.lastMergeKey === 'string' ? snapshot.lastMergeKey : null,
    past: migratedPast,
    future: migratedFuture,
  }
}

export const useEditorHistoryStore = defineStore('editorHistory', () => {
  const LIMIT = 50

  const past = shallowRef([])
  const future = shallowRef([])
  const canUndo = computed(() => past.value.length > 0)
  const canRedo = computed(() => future.value.length > 0)
  let lastMergeKey = null
  let persistenceSessionKey = null

  function persistSnapshot() {
    if (!persistenceSessionKey || typeof localStorage === 'undefined') {
      return
    }

    try {
      const payload = JSON.stringify({
        version: HISTORY_SNAPSHOT_VERSION,
        savedAt: Date.now(),
        lastMergeKey,
        past: past.value,
        future: future.value,
      })
      localStorage.setItem(`${STORAGE_KEY_PREFIX}${persistenceSessionKey}`, payload)
    } catch (error) {
      console.warn('[EditorHistory] 持久化失败，已忽略:', error)
    }
  }

  function takeSnapshot() {
    return {
      version: HISTORY_SNAPSHOT_VERSION,
      savedAt: Date.now(),
      lastMergeKey,
      past: cloneEntryList(past.value),
      future: cloneEntryList(future.value),
    }
  }

  function restoreFromSnapshot(snapshot, options = {}) {
    const { persist = false } = options
    if (!snapshot || typeof snapshot !== 'object') {
      return false
    }

    const migratedSnapshot = migrateHistorySnapshot(snapshot)
    past.value = migratedSnapshot.past.slice(-LIMIT)
    future.value = migratedSnapshot.future.slice(-LIMIT)
    lastMergeKey = typeof migratedSnapshot.lastMergeKey === 'string'
      ? migratedSnapshot.lastMergeKey
      : null
    if (persist) {
      persistSnapshot()
    }
    return true
  }

  function bindPersistenceSession(sessionKey) {
    persistenceSessionKey = normalizeSessionKey(sessionKey)
    if (!persistenceSessionKey || typeof localStorage === 'undefined') {
      return false
    }

    try {
      const raw = localStorage.getItem(`${STORAGE_KEY_PREFIX}${persistenceSessionKey}`)
      if (!raw) {
        clear({ persist: false })
        return false
      }
      const parsed = JSON.parse(raw)
      return restoreFromSnapshot(parsed, { persist: false })
    } catch (error) {
      console.warn('[EditorHistory] 读取持久化历史失败，已回退为空历史:', error)
      clear({ persist: false })
      return false
    }
  }

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
      doCommands: cloneSerializable([doCommand]),
      undoCommands: undoCommand ? cloneSerializable([undoCommand]) : [],
      revisionBefore: docStore.revision - 1,
      revisionAfter: docStore.revision,
      createdAt: Date.now(),
    }

    const newPast = [...past.value, entry]
    if (newPast.length > LIMIT) newPast.shift()
    past.value = newPast
    future.value = []
    persistSnapshot()
  }

  function pushTransaction({
    type = 'transaction',
    doCommands = [],
    undoCommands = [],
    createdAt = Date.now(),
    revisionBefore = null,
    revisionAfter = null,
  } = {}) {
    const normalizedDoCommands = Array.isArray(doCommands)
      ? doCommands.filter(Boolean)
      : []
    if (normalizedDoCommands.length === 0) {
      return
    }

    const docStore = useEditorDocumentStore()
    lastMergeKey = null

    const entry = {
      entryId: `he_${String(docStore.revision).padStart(5, '0')}`,
      type,
      doCommands: cloneSerializable(normalizedDoCommands),
      undoCommands: cloneSerializable(Array.isArray(undoCommands) ? undoCommands.filter(Boolean) : []),
      revisionBefore: revisionBefore ?? Math.max(0, docStore.revision - normalizedDoCommands.length),
      revisionAfter: revisionAfter ?? docStore.revision,
      createdAt,
    }

    const newPast = [...past.value, entry]
    if (newPast.length > LIMIT) newPast.shift()
    past.value = newPast
    future.value = []
    persistSnapshot()
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
    const newDo = {
      ...cloneSerializable(lastDo),
      after: cloneSerializable(command.after),
    }
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
    persistSnapshot()

    return true
  }

  function peekUndo() {
    if (past.value.length === 0) return null
    return cloneHistoryEntry(past.value[past.value.length - 1])
  }

  function peekRedo() {
    if (future.value.length === 0) return null
    return cloneHistoryEntry(future.value[future.value.length - 1])
  }

  function commitUndo() {
    if (past.value.length === 0) return null
    const entry = past.value[past.value.length - 1]
    past.value = past.value.slice(0, -1)
    future.value = [...future.value, entry]
    persistSnapshot()

    return cloneHistoryEntry(entry)
  }

  function commitRedo() {
    if (future.value.length === 0) return null
    const entry = future.value[future.value.length - 1]
    future.value = future.value.slice(0, -1)
    past.value = [...past.value, entry]
    persistSnapshot()

    return cloneHistoryEntry(entry)
  }

  function undo() {
    return commitUndo()
  }

  function redo() {
    return commitRedo()
  }

  function clear(options = {}) {
    const { persist = true } = options
    past.value = []
    future.value = []
    lastMergeKey = null
    if (persist) {
      persistSnapshot()
    }
  }

  function closeTransaction() {
    lastMergeKey = null
  }

  function reset(options = {}) {
    const {
      persist = false,
      unbindPersistence = false,
    } = options
    clear({ persist })
    closeTransaction()
    if (unbindPersistence) {
      persistenceSessionKey = null
    }
  }

  return {
    past,
    future,
    canUndo,
    canRedo,
    takeSnapshot,
    restoreFromSnapshot,
    bindPersistenceSession,
    push,
    pushTransaction,
    peekUndo,
    peekRedo,
    commitUndo,
    commitRedo,
    undo,
    redo,
    clear,
    closeTransaction,
    reset,
  }
})
