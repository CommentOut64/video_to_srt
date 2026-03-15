// V3.2.5+dev.20260315.05: 编辑器同步引擎 - 命令队列 + 乐观更新 + 最小对账
import { defineStore } from 'pinia'
import { ref, shallowRef } from 'vue'
import projectApi from '@/services/api/projectApi'
import { useEditorSessionStore } from './editorSessionStore'
import { useEditorDocumentStore } from './editorDocumentStore'
import { editorReducer } from './editorReducer'

const SYNC_INTERVAL_MS = 2000
const MAX_BATCH_SIZE = 50
const RETRY_DELAY_MS = 5000

function toMs(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return 0
  return Math.round(numeric * 1000)
}

function preserveOrCreateLocalId(docStore, segmentId) {
  return docStore.bindingBySegmentId.get(segmentId) || segmentId
}

function toServerSegmentSnapshot(segment) {
  const segmentId = String(segment?.segment_id || '')
  return {
    segmentId,
    text: String(segment?.text ?? ''),
    startMs: segment?.start_ms ?? toMs(segment?.start ?? 0),
    endMs: segment?.end_ms ?? toMs(segment?.end ?? 0),
    sentenceIndex: segment?.legacy_index ?? segment?.sentence_index ?? null,
    confidence: segment?.confidence ?? null,
    displayConfidence: segment?.display_confidence ?? null,
    confidenceSource: segment?.confidence_source ?? null,
    warningType: segment?.warning_type || 'none',
    sourceType: segment?.source_type ?? segment?.source ?? null,
    words: Array.isArray(segment?.words)
      ? segment.words.map((word) => ({
          startMs: word?.start_ms ?? toMs(word?.start ?? 0),
          endMs: word?.end_ms ?? toMs(word?.end ?? 0),
          text: word?.text ?? word?.word ?? '',
        }))
      : null,
  }
}

export const useEditorSyncEngine = defineStore('editorSyncEngine', () => {
  const pendingCommands = shallowRef([])
  const isSyncing = ref(false)
  const lastSyncTime = ref(0)

  let syncTimer = null
  let retryTimer = null

  function enqueue(command) {
    if (command.source === 'system' || command.source === 'rehydrate' || command.source === 'reconcile_replay') {
      return
    }

    const syncableTypes = [
      'update_text',
      'update_timing',
      'insert_subtitle',
      'delete_subtitle',
      'split_subtitle',
      'merge_subtitles',
      'move_boundary',
      'batch_replace',
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
      void pushToBackend()
    }, SYNC_INTERVAL_MS)
  }

  async function pushToBackend() {
    if (isSyncing.value || pendingCommands.value.length === 0) return true

    const sessionStore = useEditorSessionStore()
    const projectId = sessionStore.projectId
    if (!projectId) {
      console.warn('[SyncEngine] 缺少 projectId，跳过本轮同步')
      return false
    }

    isSyncing.value = true
    const batch = pendingCommands.value.slice(0, MAX_BATCH_SIZE)

    try {
      const ops = batch
        .map((command) => commandToEditorOp(command))
        .filter(Boolean)

      if (ops.length === 0) {
        pendingCommands.value = pendingCommands.value.slice(batch.length)
        return true
      }

      const response = await fetch(`/api/projects/${projectId}/editor-ops:apply`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Editor-Session-Id': sessionStore.sessionId || '',
        },
        body: JSON.stringify({
          session_id: sessionStore.sessionId,
          base_revision: sessionStore.ackedRevision,
          ops,
        }),
      })

      const payload = await response.json().catch(() => ({}))

      if (!response.ok) {
        if (response.status === 409) {
          await reconcile(payload?.server_revision ?? null)
          return true
        }
        throw new Error(payload?.message || `HTTP ${response.status}`)
      }

      if (payload.success) {
        pendingCommands.value = pendingCommands.value.slice(batch.length)

        if (payload.created_bindings?.length) {
          applyBindings(payload.created_bindings)
        }

        if (payload.updated_entities?.length) {
          applyNormalizedEntities(payload.updated_entities)
        }

        sessionStore.ackedRevision = payload.server_revision ?? sessionStore.ackedRevision
        lastSyncTime.value = Date.now()
      }

      return true
    } catch (error) {
      console.error('[SyncEngine] Push failed:', error)
      scheduleRetry()
      return false
    } finally {
      isSyncing.value = false
    }
  }

  function scheduleRetry() {
    if (retryTimer) return
    retryTimer = setTimeout(() => {
      retryTimer = null
      void pushToBackend()
    }, RETRY_DELAY_MS)
  }

  async function flush() {
    if (syncTimer) {
      clearTimeout(syncTimer)
      syncTimer = null
    }

    let attempts = 0
    while (pendingCommands.value.length > 0) {
      attempts += 1
      if (attempts > pendingCommands.value.length + 5) {
        throw new Error('同步重试次数过多，已停止 flush')
      }
      const success = await pushToBackend()
      if (!success) {
        throw new Error('同步失败，flush 已中止')
      }
    }
  }

  function applyBindings(bindings) {
    const docStore = useEditorDocumentStore()
    for (const binding of bindings) {
      docStore.updateColdBinding(binding.client_ref_id, binding.segment_id)
    }
  }

  function applyNormalizedEntities(entities) {
    const docStore = useEditorDocumentStore()
    for (const entity of entities) {
      const localId = entity.client_ref_id || docStore.bindingBySegmentId.get(entity.segment_id)
      if (!localId) continue

      const patch = {}
      if (entity.text !== undefined) patch.text = entity.text
      if (entity.start_ms !== undefined) patch.startMs = entity.start_ms
      if (entity.end_ms !== undefined) patch.endMs = entity.end_ms
      if (entity.start !== undefined) patch.startMs = toMs(entity.start)
      if (entity.end !== undefined) patch.endMs = toMs(entity.end)
      if (Object.keys(patch).length > 0) {
        docStore._applyUpdate(localId, patch)
        docStore._applyReorder(localId)
      }
      if (entity.segment_id) {
        docStore.updateColdBinding(localId, entity.segment_id)
      }
    }
  }

  async function reconcile(serverRevision = null) {
    const sessionStore = useEditorSessionStore()
    const docStore = useEditorDocumentStore()
    const projectId = sessionStore.projectId
    if (!projectId) {
      throw new Error('缺少 projectId，无法执行对账')
    }

    console.warn('[SyncEngine] Revision conflict, reconciling...')
    const serverSegments = await projectApi.getSubtitles(projectId)
    const replayCommands = pendingCommands.value.map((command) => ({
      ...command,
      source: 'reconcile_replay',
    }))
    const preservedTombstones = [...docStore.tombstones]

    // 1. 移除所有尚未绑定 segment_id 的乐观本地实体，后续靠 replay 恢复。
    for (const localId of [...docStore.order]) {
      const cold = docStore.getCold(localId)
      if (!cold?.segmentId) {
        docStore._applyDelete(localId)
      }
    }

    // 2. 用服务端权威快照覆盖所有已绑定实体。
    const seenSegmentIds = new Set()
    for (const rawSegment of serverSegments) {
      const snapshot = toServerSegmentSnapshot(rawSegment)
      if (!snapshot.segmentId) continue

      seenSegmentIds.add(snapshot.segmentId)
      const localId = preserveOrCreateLocalId(docStore, snapshot.segmentId)
      const existingEntity = docStore.getEntity(localId)
      const existingCold = docStore.getCold(localId)

      if (existingEntity) {
        docStore._applyUpdate(localId, {
          text: snapshot.text,
          startMs: snapshot.startMs,
          endMs: snapshot.endMs,
          isDraft: false,
          isModified: existingEntity.isModified,
        })
        docStore._applyReorder(localId)
        if (existingCold) {
          docStore._applyColdUpdate(localId, {
            segmentId: snapshot.segmentId,
            sentenceIndex: snapshot.sentenceIndex,
            confidence: snapshot.confidence,
            displayConfidence: snapshot.displayConfidence,
            confidenceSource: snapshot.confidenceSource,
            warningType: snapshot.warningType,
            sourceType: snapshot.sourceType,
            words: snapshot.words,
          })
          docStore.updateColdBinding(localId, snapshot.segmentId)
        }
      } else {
        docStore._applyInsert(
          localId,
          {
            localId,
            text: snapshot.text,
            startMs: snapshot.startMs,
            endMs: snapshot.endMs,
            isDraft: false,
            isModified: false,
            isDeleted: false,
            revision: 0,
          },
          {
            localId,
            segmentId: snapshot.segmentId,
            sentenceIndex: snapshot.sentenceIndex,
            chunkId: null,
            words: snapshot.words,
            confidence: snapshot.confidence,
            displayConfidence: snapshot.displayConfidence,
            confidenceSource: snapshot.confidenceSource,
            speakerId: null,
            sourceType: snapshot.sourceType,
            warningType: snapshot.warningType,
            originalText: null,
          },
          null
        )
      }
    }

    // 3. 删除服务端快照中已不存在的旧绑定实体。
    for (const localId of [...docStore.order]) {
      const cold = docStore.getCold(localId)
      if (cold?.segmentId && !seenSegmentIds.has(cold.segmentId)) {
        docStore._applyDelete(localId)
      }
    }

    // 4. 恢复 tombstone，避免对账过程污染删除同步语义。
    docStore.tombstones = preservedTombstones

    // 5. 在权威快照上重放未确认命令，但不进入 history/sync/persistence。
    for (const command of replayCommands) {
      const result = editorReducer(docStore, command)
      if (!result.success) {
        console.warn('[SyncEngine] 对账回放命令失败，已跳过:', command.type, result.reason)
      }
    }

    if (serverRevision !== null && serverRevision !== undefined) {
      sessionStore.ackedRevision = serverRevision
    }
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
          before: { text: command.before?.text ?? '' },
          after: { text: command.after.text },
        }
      }

      case 'update_timing': {
        const cold = docStore.getCold(command.localId)
        return {
          op_id: command.commandId,
          type: 'update_timing',
          client_ref_id: command.localId,
          segment_id: cold?.segmentId || null,
          before: {
            start_ms: command.before?.startMs ?? 0,
            end_ms: command.before?.endMs ?? 0,
          },
          after: {
            start_ms: command.after.startMs,
            end_ms: command.after.endMs,
          },
        }
      }

      case 'insert_subtitle': {
        const nextLocalId = command.afterLocalId
          ? docStore.getNeighbors(command.afterLocalId).next
          : null
        return {
          op_id: command.commandId,
          type: 'insert_subtitle',
          client_ref_id: command.localId,
          anchor: {
            before_client_ref_id: command.afterLocalId ?? null,
            after_client_ref_id: nextLocalId ?? null,
          },
          after: {
            text: command.entity.text,
            start_ms: command.entity.startMs,
            end_ms: command.entity.endMs,
          },
        }
      }

      case 'delete_subtitle': {
        const cold = docStore.getCold(command.localId)
        const entity = docStore.getEntity(command.localId)
        return {
          op_id: command.commandId,
          type: 'delete_subtitle',
          client_ref_id: command.localId,
          segment_id: cold?.segmentId || null,
          before: {
            text: command.snapshot?.text ?? entity?.text ?? '',
            start_ms: command.snapshot?.startMs ?? entity?.startMs ?? 0,
            end_ms: command.snapshot?.endMs ?? entity?.endMs ?? 0,
          },
        }
      }

      case 'split_subtitle': {
        const cold = docStore.getCold(command.sourceLocalId)
        return {
          op_id: command.commandId,
          type: 'split_subtitle',
          client_ref_id: command.sourceLocalId,
          segment_id: cold?.segmentId || null,
          created_client_ref_id: command.createdLocalId,
          split_at_ms: command.splitAtMs,
          split_at_text_offset: command.splitAtTextOffset,
          before: {
            text: command.before.text,
            start_ms: command.before.startMs,
            end_ms: command.before.endMs,
          },
          after_kept: {
            text: command.afterKept.text,
            start_ms: command.afterKept.startMs,
            end_ms: command.afterKept.endMs,
          },
          after_created: {
            text: command.afterCreated.text,
            start_ms: command.afterCreated.startMs,
            end_ms: command.afterCreated.endMs,
          },
        }
      }

      case 'merge_subtitles': {
        const keptCold = docStore.getCold(command.keptLocalId)
        const removedCold = docStore.getCold(command.removedLocalId)
        return {
          op_id: command.commandId,
          type: 'merge_subtitle',
          kept_client_ref_id: command.keptLocalId,
          removed_client_ref_id: command.removedLocalId,
          kept_segment_id: keptCold?.segmentId || null,
          removed_segment_id: removedCold?.segmentId || null,
          before_kept: {
            text: command.beforeKept.text,
            start_ms: command.beforeKept.startMs,
            end_ms: command.beforeKept.endMs,
          },
          before_removed: {
            text: command.beforeRemoved.text,
            start_ms: command.beforeRemoved.startMs,
            end_ms: command.beforeRemoved.endMs,
          },
          after: {
            text: command.after.text,
            start_ms: command.after.startMs,
            end_ms: command.after.endMs,
          },
        }
      }

      case 'move_boundary': {
        const upperCold = docStore.getCold(command.upperLocalId)
        const lowerCold = docStore.getCold(command.lowerLocalId)
        return {
          op_id: command.commandId,
          type: 'move_boundary',
          upper_client_ref_id: command.upperLocalId,
          lower_client_ref_id: command.lowerLocalId,
          upper_segment_id: upperCold?.segmentId || null,
          lower_segment_id: lowerCold?.segmentId || null,
          before: {
            upper_end_ms: command.before.upperEndMs,
            lower_start_ms: command.before.lowerStartMs,
          },
          after: {
            upper_end_ms: command.after.upperEndMs,
            lower_start_ms: command.after.lowerStartMs,
          },
        }
      }

      case 'batch_replace': {
        return {
          op_id: command.commandId,
          type: 'batch_replace',
          replacements: command.replacements.map((replacement) => ({
            client_ref_id: replacement.localId,
            segment_id: docStore.getCold(replacement.localId)?.segmentId || null,
            before: { text: replacement.before.text },
            after: { text: replacement.after.text },
          })),
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
    applyBindings,
    reconcile,
    commandToEditorOp,
  }
})
