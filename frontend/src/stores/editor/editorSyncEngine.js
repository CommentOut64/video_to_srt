// V3.2.4+dev.20260314.03: 编辑器同步引擎 - 命令队列 + 乐观更新 + 对账
import { defineStore } from 'pinia'
import { ref, shallowRef } from 'vue'
import { useEditorSessionStore } from './editorSessionStore'
import { useEditorDocumentStore } from './editorDocumentStore'

const SYNC_INTERVAL_MS = 2000
const MAX_BATCH_SIZE = 50
const RETRY_DELAY_MS = 5000

export const useEditorSyncEngine = defineStore('editorSyncEngine', () => {
  const pendingCommands = shallowRef([])
  const isSyncing = ref(false)
  const lastSyncTime = ref(0)

  let syncTimer = null
  let retryTimer = null

  function enqueue(command) {
    // 系统命令、恢复命令、对账命令不需要同步回后端
    if (command.source === 'system' || command.source === 'rehydrate' || command.source === 'reconcile_replay') {
      return
    }

    // 仅同步用户操作命令（包括 undo_redo）
    const syncableTypes = [
      'update_text', 'update_timing',
      'insert_subtitle', 'delete_subtitle',
      'split_subtitle', 'merge_subtitles',
      'move_boundary', 'batch_replace'
    ]
    if (!syncableTypes.includes(command.type)) {
      return
    }

    pendingCommands.value = [...pendingCommands.value, command]
    schedulePush()
  }

  function schedulePush() {
    if (syncTimer) return
    syncTimer = setTimeout(() => {
      syncTimer = null
      pushToBackend()
    }, SYNC_INTERVAL_MS)
  }

  async function pushToBackend() {
    if (isSyncing.value || pendingCommands.value.length === 0) return

    isSyncing.value = true
    const batch = pendingCommands.value.slice(0, MAX_BATCH_SIZE)

    try {
      const ops = batch.map(cmd => commandToEditorOp(cmd))
      const sessionStore = useEditorSessionStore()

      const response = await fetch('/api/editor-ops/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionStore.sessionId,
          base_revision: sessionStore.ackedRevision,
          ops
        })
      })

      if (!response.ok) {
        if (response.status === 409) {
          await reconcile()
        } else {
          throw new Error(`HTTP ${response.status}`)
        }
        return
      }

      const data = await response.json()

      if (data.success) {
        pendingCommands.value = pendingCommands.value.slice(batch.length)

        if (data.created_bindings?.length) {
          applyBindings(data.created_bindings)
        }

        sessionStore.ackedRevision = data.server_revision
        lastSyncTime.value = Date.now()
      }
    } catch (error) {
      console.error('[SyncEngine] Push failed:', error)
      scheduleRetry()
    } finally {
      isSyncing.value = false
    }
  }

  function scheduleRetry() {
    if (retryTimer) return
    retryTimer = setTimeout(() => {
      retryTimer = null
      pushToBackend()
    }, RETRY_DELAY_MS)
  }

  async function flush() {
    if (syncTimer) {
      clearTimeout(syncTimer)
      syncTimer = null
    }
    while (pendingCommands.value.length > 0) {
      await pushToBackend()
    }
  }

  function applyBindings(bindings) {
    const docStore = useEditorDocumentStore()
    for (const binding of bindings) {
      docStore.updateColdBinding(binding.client_ref_id, binding.segment_id)
    }
  }

  async function reconcile() {
    console.warn('[SyncEngine] Revision conflict, reconciling...')
    // TODO: 实现对账逻辑
  }

  function commandToEditorOp(command) {
    const docStore = useEditorDocumentStore()

    switch (command.type) {
      case 'update_text': {
        const cold = docStore.getCold(command.localId)
        return {
          op_id: command.commandId,
          type: 'update_text',
          client_ref_id: command.localId,
          segment_id: cold?.segmentId || null,
          after: { text: command.after.text }
        }
      }
      case 'update_timing': {
        const cold = docStore.getCold(command.localId)
        return {
          op_id: command.commandId,
          type: 'update_timing',
          client_ref_id: command.localId,
          segment_id: cold?.segmentId || null,
          after: {
            start_ms: command.after.startMs,
            end_ms: command.after.endMs
          }
        }
      }
      case 'insert_subtitle': {
        return {
          op_id: command.commandId,
          type: 'insert_subtitle',
          client_ref_id: command.localId,
          after: {
            text: command.entity.text,
            start_ms: command.entity.startMs,
            end_ms: command.entity.endMs
          }
        }
      }
      case 'delete_subtitle': {
        const cold = docStore.getCold(command.localId)
        return {
          op_id: command.commandId,
          type: 'delete_subtitle',
          client_ref_id: command.localId,
          segment_id: cold?.segmentId || null
        }
      }
      case 'split_subtitle': {
        const cold = docStore.getCold(command.localId)
        return {
          op_id: command.commandId,
          type: 'split_subtitle',
          client_ref_id: command.localId,
          segment_id: cold?.segmentId || null,
          split_at_ms: command.splitAtMs,
          new_client_ref_id: command.newLocalId
        }
      }
      case 'merge_subtitles': {
        const cold1 = docStore.getCold(command.localIds[0])
        const cold2 = docStore.getCold(command.localIds[1])
        return {
          op_id: command.commandId,
          type: 'merge_subtitles',
          client_ref_ids: command.localIds,
          segment_ids: [cold1?.segmentId || null, cold2?.segmentId || null]
        }
      }
      case 'move_boundary': {
        const cold = docStore.getCold(command.localId)
        return {
          op_id: command.commandId,
          type: 'update_timing',
          client_ref_id: command.localId,
          segment_id: cold?.segmentId || null,
          after: {
            start_ms: command.after.startMs,
            end_ms: command.after.endMs
          }
        }
      }
      case 'batch_replace': {
        return {
          op_id: command.commandId,
          type: 'batch_replace',
          replacements: command.replacements.map(r => ({
            client_ref_id: r.localId,
            segment_id: docStore.getCold(r.localId)?.segmentId || null,
            after: { text: r.after.text }
          }))
        }
      }
      default:
        return null
    }
  }

  return {
    pendingCommands,
    isSyncing,
    lastSyncTime,
    enqueue,
    flush,
    applyBindings
  }
})
