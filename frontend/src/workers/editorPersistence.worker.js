// V3.2.4+dev.20260314.03: 编辑器持久化 Worker - IndexedDB 分片 append-only
import { openDB } from 'idb'

let db = null
let projectId = null
let sessionId = null

async function initDB() {
  db = await openDB('editor-persistence', 1, {
    upgrade(db) {
      // 命令日志表（分片存储）
      if (!db.objectStoreNames.contains('commands')) {
        const commandStore = db.createObjectStore('commands', { keyPath: 'id', autoIncrement: true })
        commandStore.createIndex('by_project_session', ['projectId', 'sessionId'])
        commandStore.createIndex('by_timestamp', 'createdAt')
      }
      // 快照表
      if (!db.objectStoreNames.contains('snapshots')) {
        db.createObjectStore('snapshots', { keyPath: 'projectId' })
      }
    }
  })
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
        const tx = db.transaction('commands', 'readwrite')
        const store = tx.objectStore('commands')

        for (const cmd of msg.commands) {
          await store.add({
            projectId,
            sessionId,
            commandId: cmd.commandId,
            command: cmd,
            createdAt: cmd.createdAt
          })
        }

        await tx.done
        self.postMessage({ type: 'append_ok', count: msg.commands.length })
        break
      }

      case 'save_snapshot': {
        const tx = db.transaction('snapshots', 'readwrite')
        await tx.objectStore('snapshots').put({
          projectId,
          snapshot: msg.snapshot,
          createdAt: msg.snapshot.createdAt,
          revision: msg.snapshot.revision
        })
        await tx.done

        // 删除快照之前的命令
        const deleteTx = db.transaction('commands', 'readwrite')
        const commandStore = deleteTx.objectStore('commands')
        const index = commandStore.index('by_project_session')
        const range = IDBKeyRange.only([projectId, sessionId])

        let cursor = await index.openCursor(range)
        while (cursor) {
          if (cursor.value.createdAt <= msg.snapshot.createdAt) {
            await cursor.delete()
          }
          cursor = await cursor.continue()
        }

        await deleteTx.done
        self.postMessage({ type: 'snapshot_ok' })
        break
      }

      case 'load': {
        const snapshotTx = db.transaction('snapshots', 'readonly')
        const snapshotData = await snapshotTx.objectStore('snapshots').get(msg.projectId)

        const commandTx = db.transaction('commands', 'readonly')
        const commandStore = commandTx.objectStore('commands')
        const index = commandStore.index('by_project_session')
        const range = IDBKeyRange.only([msg.projectId, sessionId])

        const commands = []
        let cursor = await index.openCursor(range)
        while (cursor) {
          commands.push(cursor.value.command)
          cursor = await cursor.continue()
        }

        self.postMessage({
          type: 'loaded',
          snapshot: snapshotData?.snapshot || null,
          pendingCommands: commands
        })
        break
      }

      case 'compact': {
        const sevenDaysAgo = Date.now() - 7 * 24 * 60 * 60 * 1000
        const tx = db.transaction(['commands', 'snapshots'], 'readwrite')

        const commandStore = tx.objectStore('commands')
        const timeIndex = commandStore.index('by_timestamp')
        const range = IDBKeyRange.upperBound(sevenDaysAgo)

        let removedCount = 0
        let cursor = await timeIndex.openCursor(range)
        while (cursor) {
          await cursor.delete()
          removedCount++
          cursor = await cursor.continue()
        }

        await tx.done
        self.postMessage({ type: 'compact_ok', removedCount })
        break
      }

      case 'clear': {
        const tx = db.transaction(['commands', 'snapshots'], 'readwrite')
        const commandStore = tx.objectStore('commands')
        const index = commandStore.index('by_project_session')
        const range = IDBKeyRange.only([msg.projectId, sessionId])

        let cursor = await index.openCursor(range)
        while (cursor) {
          await cursor.delete()
          cursor = await cursor.continue()
        }

        await tx.objectStore('snapshots').delete(msg.projectId)
        await tx.done
        self.postMessage({ type: 'clear_ok' })
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
