// V3.2.4+dev.20260314.03: 编辑器持久化 Worker - IndexedDB 分片 append-only
import { openDB } from 'idb'

const DB_VERSION = 2
const LEGACY_COMMANDS_STORE = 'commands'
const COMMAND_CHUNKS_STORE = 'command_chunks'
const SNAPSHOTS_STORE = 'snapshots'

let db = null
let projectId = null
let sessionId = null

function hasStore(storeName) {
  return Boolean(db?.objectStoreNames?.contains?.(storeName))
}

function createCommandChunk(commands = []) {
  const normalizedCommands = Array.isArray(commands) ? commands.filter(Boolean) : []
  const createdAtValues = normalizedCommands
    .map((command) => Number(command.createdAt) || Date.now())
    .sort((left, right) => left - right)
  const chunkCreatedAt = createdAtValues[0] ?? Date.now()
  const lastCommandAt = createdAtValues[createdAtValues.length - 1] ?? chunkCreatedAt
  const firstCommandId = normalizedCommands[0]?.commandId ?? `chunk_${chunkCreatedAt}`

  return {
    chunkId: `${projectId}:${sessionId}:${firstCommandId}:${lastCommandAt}`,
    projectId,
    sessionId,
    commands: normalizedCommands,
    createdAt: chunkCreatedAt,
    lastCommandAt,
  }
}

async function initDB() {
  db = await openDB('editor-persistence', DB_VERSION, {
    upgrade(database) {
      if (!database.objectStoreNames.contains(COMMAND_CHUNKS_STORE)) {
        const chunkStore = database.createObjectStore(COMMAND_CHUNKS_STORE, { keyPath: 'chunkId' })
        chunkStore.createIndex('by_project_session', ['projectId', 'sessionId'])
        chunkStore.createIndex('by_last_command_at', 'lastCommandAt')
      }
      if (!database.objectStoreNames.contains(SNAPSHOTS_STORE)) {
        database.createObjectStore(SNAPSHOTS_STORE, { keyPath: 'projectId' })
      }
    }
  })
}

async function readLegacyCommands(targetProjectId, targetSessionId) {
  if (!hasStore(LEGACY_COMMANDS_STORE)) {
    return []
  }

  const tx = db.transaction(LEGACY_COMMANDS_STORE, 'readonly')
  const store = tx.objectStore(LEGACY_COMMANDS_STORE)
  const index = store.index('by_project_session')
  const range = IDBKeyRange.only([targetProjectId, targetSessionId])
  const commands = []

  let cursor = await index.openCursor(range)
  while (cursor) {
    commands.push(cursor.value.command)
    cursor = await cursor.continue()
  }

  return commands
}

async function readChunkCommands(targetProjectId, targetSessionId) {
  if (!hasStore(COMMAND_CHUNKS_STORE)) {
    return []
  }

  const tx = db.transaction(COMMAND_CHUNKS_STORE, 'readonly')
  const store = tx.objectStore(COMMAND_CHUNKS_STORE)
  const index = store.index('by_project_session')
  const range = IDBKeyRange.only([targetProjectId, targetSessionId])
  const commands = []

  let cursor = await index.openCursor(range)
  while (cursor) {
    const chunkCommands = Array.isArray(cursor.value.commands) ? cursor.value.commands : []
    commands.push(...chunkCommands)
    cursor = await cursor.continue()
  }

  return commands
}

async function deleteRangeByCursor(storeName, indexName, range, predicate = null) {
  if (!hasStore(storeName)) {
    return 0
  }

  const tx = db.transaction(storeName, 'readwrite')
  const store = tx.objectStore(storeName)
  const index = store.index(indexName)
  let removedCount = 0

  let cursor = await index.openCursor(range)
  while (cursor) {
    if (!predicate || predicate(cursor.value)) {
      await cursor.delete()
      removedCount += 1
    }
    cursor = await cursor.continue()
  }

  await tx.done
  return removedCount
}

self.onmessage = async (event) => {
  const msg = event.data

  try {
    switch (msg.type) {
      case 'init':
        projectId = msg.projectId
        sessionId = msg.sessionId
        await initDB()
        self.postMessage({ type: 'ready' })
        break

      case 'append_commands': {
        const commands = Array.isArray(msg.commands) ? msg.commands : []
        if (commands.length === 0) {
          self.postMessage({ type: 'append_ok', count: 0 })
          break
        }

        const chunk = createCommandChunk(commands)
        const tx = db.transaction(COMMAND_CHUNKS_STORE, 'readwrite')
        await tx.objectStore(COMMAND_CHUNKS_STORE).put(chunk)
        await tx.done
        self.postMessage({ type: 'append_ok', count: commands.length, chunkId: chunk.chunkId })
        break
      }

      case 'save_snapshot': {
        const tx = db.transaction(SNAPSHOTS_STORE, 'readwrite')
        await tx.objectStore(SNAPSHOTS_STORE).put({
          projectId,
          snapshot: msg.snapshot,
          createdAt: msg.snapshot.createdAt,
          revision: msg.snapshot.revision
        })
        await tx.done
        self.postMessage({ type: 'snapshot_ok' })
        break
      }

      case 'load': {
        const snapshotTx = db.transaction(SNAPSHOTS_STORE, 'readonly')
        const snapshotData = await snapshotTx.objectStore(SNAPSHOTS_STORE).get(msg.projectId)
        const snapshotCreatedAt = Number(snapshotData?.createdAt ?? snapshotData?.snapshot?.createdAt ?? 0)
        const loadedCommands = [
          ...(await readLegacyCommands(msg.projectId, sessionId)),
          ...(await readChunkCommands(msg.projectId, sessionId)),
        ]

        const dedupedCommands = new Map()
        loadedCommands.forEach((command) => {
          if (!command?.commandId) {
            return
          }
          if (snapshotCreatedAt > 0 && Number(command.createdAt) <= snapshotCreatedAt) {
            return
          }
          dedupedCommands.set(command.commandId, command)
        })

        const pendingCommands = Array.from(dedupedCommands.values())
          .sort((left, right) => (Number(left.createdAt) || 0) - (Number(right.createdAt) || 0))

        self.postMessage({
          type: 'loaded',
          snapshot: snapshotData?.snapshot || null,
          pendingCommands,
        })
        break
      }

      case 'compact': {
        const snapshotTx = db.transaction(SNAPSHOTS_STORE, 'readonly')
        const snapshotData = await snapshotTx.objectStore(SNAPSHOTS_STORE).get(projectId)
        const snapshotCreatedAt = Number(snapshotData?.createdAt ?? snapshotData?.snapshot?.createdAt ?? 0)
        const sevenDaysAgo = Date.now() - 7 * 24 * 60 * 60 * 1000
        let removedCount = 0

        removedCount += await deleteRangeByCursor(
          COMMAND_CHUNKS_STORE,
          'by_project_session',
          IDBKeyRange.only([projectId, sessionId]),
          (value) => {
            if (snapshotCreatedAt > 0) {
              return Number(value.lastCommandAt) <= snapshotCreatedAt
            }
            return Number(value.lastCommandAt) <= sevenDaysAgo
          }
        )

        removedCount += await deleteRangeByCursor(
          LEGACY_COMMANDS_STORE,
          'by_project_session',
          IDBKeyRange.only([projectId, sessionId]),
          (value) => {
            if (snapshotCreatedAt > 0) {
              return Number(value.createdAt) <= snapshotCreatedAt
            }
            return Number(value.createdAt) <= sevenDaysAgo
          }
        )

        self.postMessage({ type: 'compact_ok', removedCount })
        break
      }

      case 'clear': {
        let removedCount = 0

        removedCount += await deleteRangeByCursor(
          COMMAND_CHUNKS_STORE,
          'by_project_session',
          IDBKeyRange.only([msg.projectId, sessionId])
        )
        removedCount += await deleteRangeByCursor(
          LEGACY_COMMANDS_STORE,
          'by_project_session',
          IDBKeyRange.only([msg.projectId, sessionId])
        )

        if (hasStore(SNAPSHOTS_STORE)) {
          const tx = db.transaction(SNAPSHOTS_STORE, 'readwrite')
          await tx.objectStore(SNAPSHOTS_STORE).delete(msg.projectId)
          await tx.done
        }

        self.postMessage({ type: 'clear_ok', removedCount })
        break
      }

      default:
        self.postMessage({
          type: 'error',
          requestType: msg.type,
          message: `Unknown message type: ${msg.type}`
        })
    }
  } catch (error) {
    self.postMessage({
      type: 'error',
      requestType: msg.type,
      message: error.message
    })
  }
}
