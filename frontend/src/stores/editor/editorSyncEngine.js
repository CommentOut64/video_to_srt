// V3.2.5+dev.20260316.21: 编辑器同步引擎（BindingMap + AckShadow + Dirty/Tombstone Diff）
import { defineStore } from 'pinia'
import { ref, shallowRef } from 'vue'
import projectApi from '@/services/api/projectApi'
import { useEditorSessionStore } from './editorSessionStore'
import { useEditorDocumentStore } from './editorDocumentStore'
import { buildPrimitiveDiffEntries } from './editorDiffSyncBuilder'

const SYNC_INTERVAL_MS = 2000
const MAX_BATCH_SIZE = 50
const RETRY_DELAY_MS = 5000

function toMs(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return 0
  return Math.round(numeric * 1000)
}

function toNormalizedSegmentId(value) {
  const normalized = String(value ?? '').trim()
  return normalized || null
}

function toNormalizedChunkId(value) {
  if (value === undefined || value === null) {
    return null
  }

  const raw = String(value).trim()
  if (!raw) {
    return null
  }

  const withoutDedupSuffix = raw.split('#')[0]?.trim() || raw
  if (!withoutDedupSuffix) {
    return null
  }

  if (withoutDedupSuffix.startsWith('chunk-')) {
    const suffix = withoutDedupSuffix.slice('chunk-'.length).trim()
    if (/^-?\d+$/.test(suffix)) {
      return Number(suffix)
    }
    return suffix || withoutDedupSuffix
  }

  if (/^-?\d+$/.test(withoutDedupSuffix)) {
    return Number(withoutDedupSuffix)
  }

  return withoutDedupSuffix
}

function toServerSegmentSnapshot(segment) {
  const segmentId = toNormalizedSegmentId(segment?.segment_id)
  return {
    segmentId,
    chunkId: toNormalizedChunkId(segment?.chunk_id ?? segment?.chunk_uid ?? null),
    text: String(segment?.text ?? ''),
    startMs: segment?.start_ms ?? toMs(segment?.start ?? 0),
    endMs: segment?.end_ms ?? toMs(segment?.end ?? 0),
    isDraft: Boolean(segment?.is_draft),
    isModified: Boolean(segment?.is_modified),
    sentenceIndex: segment?.legacy_index ?? segment?.sentence_index ?? null,
    confidence: segment?.confidence ?? null,
    displayConfidence: segment?.display_confidence ?? null,
    confidenceSource: segment?.confidence_source ?? null,
    warningType: segment?.warning_type || 'none',
    sourceType: segment?.source_type ?? segment?.source ?? null,
    originalText: segment?.original_text ?? null,
    speakerId: segment?.speaker_id ?? null,
    speakerLabel: segment?.speaker_label ?? null,
    speakerColorKey: segment?.speaker_color_key ?? null,
    turnId: segment?.turn_id ?? null,
    bindingSource: segment?.binding_source ?? null,
    words: Array.isArray(segment?.words)
      ? segment.words.map((word) => ({
          startMs: word?.start_ms ?? toMs(word?.start ?? 0),
          endMs: word?.end_ms ?? toMs(word?.end ?? word?.start ?? 0),
          text: word?.text ?? word?.word ?? '',
        }))
      : null,
  }
}

function resolveAuthoritativeLocalId(docStore, snapshot) {
  const boundBySegmentId = docStore.bindingBySegmentId.get(snapshot.segmentId)
  if (boundBySegmentId) {
    return boundBySegmentId
  }

  const sentenceIndex = snapshot.sentenceIndex
  if (sentenceIndex !== null && sentenceIndex !== undefined) {
    const boundBySentenceIndex = docStore.bindingBySentenceIndex.get(sentenceIndex)
    if (boundBySentenceIndex) {
      const existingCold = docStore.getCold(boundBySentenceIndex)
      const existingSegmentId = toNormalizedSegmentId(existingCold?.segmentId)
      if (!existingSegmentId || existingSegmentId === snapshot.segmentId) {
        return boundBySentenceIndex
      }
    }
  }

  return snapshot.segmentId
}

function isSameLogicalChunk(chunkId, chunkKeySet) {
  if (!(chunkKeySet instanceof Set) || chunkKeySet.size === 0) {
    return false
  }

  const normalizedChunkId = toNormalizedChunkId(chunkId)
  if (normalizedChunkId === null) {
    return false
  }

  return chunkKeySet.has(normalizedChunkId)
}

function syncSnapshotColdState(docStore, localId, snapshot, existingCold) {
  if (existingCold?.segmentId && existingCold.segmentId !== snapshot.segmentId) {
    docStore.bindingBySegmentId.delete(existingCold.segmentId)
  }
  if (
    existingCold?.sentenceIndex !== null
    && existingCold?.sentenceIndex !== undefined
    && existingCold.sentenceIndex !== snapshot.sentenceIndex
  ) {
    docStore.bindingBySentenceIndex.delete(existingCold.sentenceIndex)
  }

  if (existingCold) {
    docStore._applyColdUpdate(localId, {
      segmentId: snapshot.segmentId,
      sentenceIndex: snapshot.sentenceIndex,
      chunkId: snapshot.chunkId,
      confidence: snapshot.confidence,
      displayConfidence: snapshot.displayConfidence,
      confidenceSource: snapshot.confidenceSource,
      warningType: snapshot.warningType,
      sourceType: snapshot.sourceType,
      originalText: snapshot.originalText,
      speakerId: snapshot.speakerId,
      speakerLabel: snapshot.speakerLabel,
      speakerColorKey: snapshot.speakerColorKey,
      turnId: snapshot.turnId,
      bindingSource: snapshot.bindingSource,
      words: snapshot.words,
    })
  }

  if (snapshot.segmentId) {
    docStore.updateColdBinding(localId, snapshot.segmentId)
  }
  if (snapshot.sentenceIndex !== null && snapshot.sentenceIndex !== undefined) {
    docStore.bindingBySentenceIndex.set(snapshot.sentenceIndex, localId)
  }
}

function captureUnsyncedLocalState(docStore) {
  const dirtyLocalIds = docStore.getDirtyLocalIds()
  const dirtyLocalSnapshots = new Map()

  for (const localId of dirtyLocalIds) {
    const entity = docStore.getEntity(localId)
    if (!entity) continue

    const cold = docStore.getCold(localId)
    const neighbors = docStore.getNeighbors(localId)
    dirtyLocalSnapshots.set(localId, {
      hot: {
        ...entity,
        localId,
      },
      cold: cold
        ? {
            ...cold,
            localId,
          }
        : null,
      afterLocalId: neighbors.prev ?? null,
    })
  }

  return {
    dirtyLocalIds: new Set(dirtyLocalIds),
    dirtyLocalSnapshots,
    tombstones: docStore.getTombstones(),
  }
}

function applyAuthoritativeSegmentsToDocument(docStore, serverSegments, preserveDirtyLocalIds = new Set()) {
  const normalizedSegments = Array.isArray(serverSegments) ? serverSegments : []
  const seenSegmentIds = new Set()
  const finalizedChunkIds = new Set()
  const finalizedSentenceIndices = new Set()

  for (const rawSegment of normalizedSegments) {
    const snapshot = toServerSegmentSnapshot(rawSegment)
    if (!snapshot.segmentId) continue

    seenSegmentIds.add(snapshot.segmentId)
    if (!snapshot.isDraft && snapshot.chunkId !== null) {
      finalizedChunkIds.add(snapshot.chunkId)
    }
    if (!snapshot.isDraft && snapshot.sentenceIndex !== null && snapshot.sentenceIndex !== undefined) {
      finalizedSentenceIndices.add(snapshot.sentenceIndex)
    }

    const localId = resolveAuthoritativeLocalId(docStore, snapshot)
    const existingEntity = docStore.getEntity(localId)
    const existingCold = docStore.getCold(localId)

    if (existingEntity && existingCold) {
      docStore._applyUpdate(localId, {
        text: snapshot.text,
        startMs: snapshot.startMs,
        endMs: snapshot.endMs,
        isDraft: snapshot.isDraft,
        isModified: snapshot.isModified,
        isDeleted: false,
      })
      docStore._applyReorder(localId)
      syncSnapshotColdState(docStore, localId, snapshot, existingCold)
      continue
    }

    if (existingEntity && !existingCold) {
      docStore._applyDelete(localId, { trackTombstone: false })
    }

    docStore._applyInsert(
      localId,
      {
        localId,
        text: snapshot.text,
        startMs: snapshot.startMs,
        endMs: snapshot.endMs,
        isDraft: snapshot.isDraft,
        isModified: snapshot.isModified,
        isDeleted: false,
        revision: 0,
      },
      {
        localId,
        segmentId: snapshot.segmentId,
        sentenceIndex: snapshot.sentenceIndex,
        chunkId: snapshot.chunkId,
        words: snapshot.words,
        confidence: snapshot.confidence,
        displayConfidence: snapshot.displayConfidence,
        confidenceSource: snapshot.confidenceSource,
        speakerId: snapshot.speakerId,
        speakerLabel: snapshot.speakerLabel,
        speakerColorKey: snapshot.speakerColorKey,
        turnId: snapshot.turnId,
        bindingSource: snapshot.bindingSource,
        sourceType: snapshot.sourceType,
        warningType: snapshot.warningType,
        originalText: snapshot.originalText,
      },
      null
    )
  }

  // 权威定稿一旦回拉成功，同逻辑 chunk 的本地 draft 必须立即淘汰；
  // 否则旧 draft 会与新定稿并存，或在后续 preserve/replay 时再次复活。
  const droppedLocalIds = new Set()
  if (finalizedChunkIds.size > 0) {
    for (const localId of [...docStore.order]) {
      const entity = docStore.getEntity(localId)
      const cold = docStore.getCold(localId)
      if (!entity?.isDraft) {
        continue
      }
      if (!isSameLogicalChunk(cold?.chunkId, finalizedChunkIds)) {
        continue
      }
      docStore._applyDelete(localId, { trackTombstone: false })
      droppedLocalIds.add(localId)
    }
  }

  // 对账兜底：若 SSE replace/finalized 事件缺少 segment_id，前端会产生“无主键定稿”。
  // 当权威快照已确认该句/该 chunk 的 segment_id 后，必须强制清理此类孤儿条目，避免双份字幕并存。
  if (finalizedChunkIds.size > 0 || finalizedSentenceIndices.size > 0) {
    for (const localId of [...docStore.order]) {
      const entity = docStore.getEntity(localId)
      const cold = docStore.getCold(localId)
      if (!entity || entity.isDraft) {
        continue
      }
      if (cold?.segmentId) {
        continue
      }
      if (preserveDirtyLocalIds.has(localId)) {
        continue
      }

      const hitFinalizedChunk = isSameLogicalChunk(cold?.chunkId, finalizedChunkIds)
      const sentenceIndex = cold?.sentenceIndex
      const hitFinalizedSentence = sentenceIndex !== null
        && sentenceIndex !== undefined
        && finalizedSentenceIndices.has(sentenceIndex)
      if (!hitFinalizedChunk && !hitFinalizedSentence) {
        continue
      }

      docStore._applyDelete(localId, { trackTombstone: false })
      droppedLocalIds.add(localId)
    }
  }

  for (const localId of [...docStore.order]) {
    const cold = docStore.getCold(localId)
    if (!cold?.segmentId) {
      continue
    }
    if (seenSegmentIds.has(cold.segmentId)) {
      continue
    }
    if (preserveDirtyLocalIds.has(localId)) {
      continue
    }
    docStore._applyDelete(localId, { trackTombstone: false })
  }

  return {
    appliedCount: normalizedSegments.length,
    finalizedChunkIds,
    droppedLocalIds,
  }
}

function extractCommandLocalIds(command) {
  if (!command) return []

  switch (command.type) {
    case 'update_text':
    case 'update_timing':
    case 'insert_subtitle':
    case 'delete_subtitle':
      return command.localId ? [command.localId] : []
    case 'split_subtitle':
      return [command.sourceLocalId, command.createdLocalId].filter(Boolean)
    case 'merge_subtitles':
      return [command.keptLocalId, command.removedLocalId].filter(Boolean)
    case 'move_boundary':
      return [command.upperLocalId, command.lowerLocalId].filter(Boolean)
    case 'batch_replace':
      return Array.isArray(command.replacements)
        ? command.replacements.map((item) => item.localId).filter(Boolean)
        : []
    case 'bind_segment_id':
      return Array.isArray(command.bindings)
        ? command.bindings.map((item) => item.localId).filter(Boolean)
        : []
    case 'apply_server_replace': {
      const oldLocalIds = Array.isArray(command.oldLocalIds) ? command.oldLocalIds : []
      const newLocalIds = Array.isArray(command.newEntities)
        ? command.newEntities.map((item) => item.localId).filter(Boolean)
        : []
      return [...oldLocalIds, ...newLocalIds].filter(Boolean)
    }
    case 'finalize_draft_chunk': {
      const updateLocalIds = Array.isArray(command.updates)
        ? command.updates.map((item) => item.localId).filter(Boolean)
        : []
      const directLocalIds = Array.isArray(command.localIds) ? command.localIds : []
      return [...updateLocalIds, ...directLocalIds].filter(Boolean)
    }
    default:
      return []
  }
}

function buildAckSnapshot(localId, docStore) {
  const entity = docStore.getEntity(localId)
  const cold = docStore.getCold(localId)
  const segmentId = toNormalizedSegmentId(cold?.segmentId)
  if (!entity || !segmentId) {
    return null
  }

  return {
    localId,
    segmentId,
    text: String(entity.text ?? ''),
    startMs: Number(entity.startMs ?? 0),
    endMs: Number(entity.endMs ?? 0),
  }
}

function buildServerEntityBeforeSnapshot(entity) {
  const startMs = entity?.start_ms ?? toMs(entity?.start ?? 0)
  const endMs = entity?.end_ms ?? toMs(entity?.end ?? entity?.start ?? 0)
  return {
    text: String(entity?.text ?? ''),
    startMs: Number(startMs ?? 0),
    endMs: Number(endMs ?? startMs ?? 0),
    isDraft: Boolean(entity?.is_draft),
  }
}

function collectAuthoritativeSegmentIds(serverSegments) {
  const segmentIdSet = new Set()
  const normalizedSegments = Array.isArray(serverSegments) ? serverSegments : []
  for (const segment of normalizedSegments) {
    const segmentId = toNormalizedSegmentId(segment?.segment_id)
    if (segmentId) {
      segmentIdSet.add(segmentId)
    }
  }
  return segmentIdSet
}

export const useEditorSyncEngine = defineStore('editorSyncEngine', () => {
  // 为兼容旧 UI 统计面保留该字段，内部不再作为“命令队列真源”
  const pendingCommands = shallowRef([])
  const isSyncing = ref(false)
  const lastSyncTime = ref(0)

  const ackShadowByLocalId = new Map()
  const ackLocalIdBySegmentId = new Map()

  let syncTimer = null
  let retryTimer = null
  let inflightPushPromise = null

  function updatePendingPreview() {
    const docStore = useEditorDocumentStore()
    const dirtyPreview = docStore.getDirtyLocalIds().map((localId) => ({
      type: 'dirty',
      localId,
    }))
    const tombstonePreview = docStore.getTombstones().map((item) => ({
      type: 'delete_tombstone',
      localId: item.localId ?? null,
      segmentId: item.segmentId ?? null,
    }))
    pendingCommands.value = [...dirtyPreview, ...tombstonePreview]
  }

  function upsertAckShadowForLocal(localId) {
    if (!localId) return
    const docStore = useEditorDocumentStore()
    const snapshot = buildAckSnapshot(localId, docStore)
    const previous = ackShadowByLocalId.get(localId)
    if (previous?.segmentId && previous.segmentId !== snapshot?.segmentId) {
      ackLocalIdBySegmentId.delete(previous.segmentId)
    }

    if (!snapshot) {
      ackShadowByLocalId.delete(localId)
      return
    }

    ackShadowByLocalId.set(localId, snapshot)
    ackLocalIdBySegmentId.set(snapshot.segmentId, localId)
  }

  function removeAckShadow(localId = null, segmentId = null) {
    const normalizedSegmentId = toNormalizedSegmentId(segmentId)
    if (localId) {
      const snapshot = ackShadowByLocalId.get(localId)
      ackShadowByLocalId.delete(localId)
      if (snapshot?.segmentId) {
        ackLocalIdBySegmentId.delete(snapshot.segmentId)
      }
    }

    if (normalizedSegmentId) {
      const mappedLocalId = ackLocalIdBySegmentId.get(normalizedSegmentId)
      ackLocalIdBySegmentId.delete(normalizedSegmentId)
      if (mappedLocalId) {
        ackShadowByLocalId.delete(mappedLocalId)
      }
    }
  }

  function rebuildAckShadowFromDocument() {
    const docStore = useEditorDocumentStore()
    ackShadowByLocalId.clear()
    ackLocalIdBySegmentId.clear()
    for (const localId of docStore.order) {
      const snapshot = buildAckSnapshot(localId, docStore)
      if (!snapshot) continue
      ackShadowByLocalId.set(localId, snapshot)
      ackLocalIdBySegmentId.set(snapshot.segmentId, localId)
    }
  }

  function hasAckBinding(localId, segmentId) {
    if (!localId) {
      return false
    }
    const normalizedSegmentId = toNormalizedSegmentId(segmentId)
    if (!normalizedSegmentId) {
      return false
    }
    return ackLocalIdBySegmentId.get(normalizedSegmentId) === localId
  }

  function repairUnboundLocalBindingsFromAckShadow(docStore) {
    const repairedLocalIds = []
    for (const localId of docStore.order) {
      const entity = docStore.getEntity(localId)
      if (!entity) {
        continue
      }

      const cold = docStore.getCold(localId)
      const currentSegmentId = toNormalizedSegmentId(cold?.segmentId)
      if (currentSegmentId) {
        continue
      }

      const ackSnapshot = ackShadowByLocalId.get(localId)
      const ackSegmentId = toNormalizedSegmentId(ackSnapshot?.segmentId)
      if (!ackSegmentId) {
        continue
      }

      const boundLocalId = docStore.bindingBySegmentId.get(ackSegmentId)
      if (boundLocalId && boundLocalId !== localId) {
        throw new Error(
          `[SyncEngine] 检测到 segment 绑定冲突：segment_id=${ackSegmentId} 已绑定到 ${boundLocalId}，无法自动修复 ${localId}`
        )
      }

      docStore.updateColdBinding(localId, ackSegmentId)
      repairedLocalIds.push(localId)
    }

    return repairedLocalIds
  }

  function schedulePush() {
    if (syncTimer) return
    syncTimer = setTimeout(() => {
      syncTimer = null
      void pushToBackend()
    }, SYNC_INTERVAL_MS)
  }

  function enqueue(command) {
    if (!command) return

    if (command.skipSync === true) {
      const docStore = useEditorDocumentStore()
      const affectedLocalIds = extractCommandLocalIds(command)
      docStore.clearDirtyFlags(affectedLocalIds)
      docStore.clearTombstones({ localIds: affectedLocalIds })
      updatePendingPreview()
      return
    }

    if (['system', 'rehydrate', 'reconcile_replay'].includes(command.source)) {
      updatePendingPreview()
      return
    }

    updatePendingPreview()
    schedulePush()
  }

  function rewritePendingForHistory() {
    // 硬切到 diff 同步后，历史回放不再驱动同步队列
    return { mode: 'disabled_diff_sync' }
  }

  async function executePushToBackend(options = {}) {
    const { allowConflictRetry = true } = options
    const sessionStore = useEditorSessionStore()
    const docStore = useEditorDocumentStore()
    const projectId = sessionStore.projectId
    if (!projectId) {
      updatePendingPreview()
      return true
    }

    // 防御性自愈：若历史回放导致本地实体短暂丢失 segment 绑定，但 ackShadow 仍保留
    // 已知服务端绑定，则先恢复绑定再计算 diff，避免 flush 误判“已收敛”。
    const repairedLocalIds = repairUnboundLocalBindingsFromAckShadow(docStore)
    if (repairedLocalIds.length > 0) {
      updatePendingPreview()
    }

    const { entries, touchedDirtyLocalIds, touchedTombstones } = buildPrimitiveDiffEntries({
      docStore,
      sessionStore,
      ackShadowByLocalId,
      ackLocalIdBySegmentId,
      maxOps: MAX_BATCH_SIZE,
    })

    const touchedDirtyLocalIdSet = new Set(touchedDirtyLocalIds.filter(Boolean))
    const touchedTombstoneLocalIds = new Set()
    const touchedTombstoneSegmentIds = new Set()
    for (const tombstone of touchedTombstones) {
      if (tombstone?.localId) {
        touchedTombstoneLocalIds.add(tombstone.localId)
      }
      if (tombstone?.segmentId) {
        touchedTombstoneSegmentIds.add(tombstone.segmentId)
      }
    }

    if (entries.length === 0) {
      if (touchedDirtyLocalIdSet.size > 0) {
        const localIds = [...touchedDirtyLocalIdSet]
        docStore.clearDirtyFlags(localIds)
        for (const localId of localIds) {
          upsertAckShadowForLocal(localId)
        }
      }
      if (touchedTombstoneLocalIds.size > 0 || touchedTombstoneSegmentIds.size > 0) {
        docStore.clearTombstones({
          localIds: [...touchedTombstoneLocalIds],
          segmentIds: [...touchedTombstoneSegmentIds],
        })
      }
      for (const localId of touchedTombstoneLocalIds) {
        removeAckShadow(localId, null)
      }
      for (const segmentId of touchedTombstoneSegmentIds) {
        removeAckShadow(null, segmentId)
      }
      updatePendingPreview()
      return true
    }

    const ops = entries.map((entry) => entry.op).filter(Boolean)
    if (ops.length === 0) {
      if (touchedDirtyLocalIdSet.size > 0) {
        const localIds = [...touchedDirtyLocalIdSet]
        docStore.clearDirtyFlags(localIds)
        for (const localId of localIds) {
          upsertAckShadowForLocal(localId)
        }
      }
      if (touchedTombstoneLocalIds.size > 0 || touchedTombstoneSegmentIds.size > 0) {
        docStore.clearTombstones({
          localIds: [...touchedTombstoneLocalIds],
          segmentIds: [...touchedTombstoneSegmentIds],
        })
      }
      for (const localId of touchedTombstoneLocalIds) {
        removeAckShadow(localId, null)
      }
      for (const segmentId of touchedTombstoneSegmentIds) {
        removeAckShadow(null, segmentId)
      }
      updatePendingPreview()
      return true
    }

    // V3.2.5+dev.20260317.01: 推送前冻结实体快照，防止 HTTP 在途期间
    // 撤销/重做修改实体后，响应到达时用当前态（而非推送时态）污染 ackShadow。
    const pushTimeSnapshots = new Map()
    for (const localId of touchedDirtyLocalIdSet) {
      const entity = docStore.getEntity(localId)
      if (entity) {
        pushTimeSnapshots.set(localId, {
          text: String(entity.text ?? ''),
          startMs: Number(entity.startMs ?? 0),
          endMs: Number(entity.endMs ?? 0),
        })
      }
    }

    isSyncing.value = true
    try {
      const response = await fetch(`/api/projects/${projectId}/editor-ops:apply`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Editor-Session-Id': sessionStore.sessionId || '',
        },
        body: JSON.stringify({
          session_id: sessionStore.sessionId,
          base_revision: Number.isInteger(sessionStore.ackedRevision) ? sessionStore.ackedRevision : null,
          ops,
        }),
      })

      let payload = {}
      try {
        payload = await response.json()
      } catch {
        payload = {}
      }

      if (!response.ok) {
        const conflictByHttpCode = response.status === 409
        const conflictByPayload = payload?.error_code === 'REVISION_CONFLICT'
        if (allowConflictRetry && (conflictByHttpCode || conflictByPayload)) {
          await reconcile(payload?.server_revision ?? null)
          return executePushToBackend({ allowConflictRetry: false })
        }
        const message = payload?.message || `同步失败(${response.status})`
        throw new Error(message)
      }

      if (payload?.server_revision !== undefined && payload?.server_revision !== null) {
        sessionStore.ackedRevision = payload.server_revision
      }

      applyBindings(payload?.created_bindings || [])
      applyNormalizedEntities(payload?.updated_entities || [])
      compensateOrphanCreatedBindings(
        payload?.created_bindings || [],
        payload?.updated_entities || []
      )

      const dirtyLocalIdSet = new Set(touchedDirtyLocalIdSet)
      const tombstoneLocalIds = new Set(touchedTombstoneLocalIds)
      const tombstoneSegmentIds = new Set(touchedTombstoneSegmentIds)
      for (const entry of entries) {
        if (entry.kind === 'dirty') {
          dirtyLocalIdSet.add(entry.localId)
        } else if (entry.kind === 'tombstone') {
          if (entry.localId) tombstoneLocalIds.add(entry.localId)
          if (entry.segmentId) tombstoneSegmentIds.add(entry.segmentId)
        }
      }

      // V3.2.5+dev.20260317.01: 仅对"推送后未被并发修改"的实体清除脏标记和更新 ackShadow。
      // 若实体在 HTTP 在途期间被撤销/重做修改，保留脏标记让下轮 diff 捕获变更。
      const dirtyLocalIds = [...dirtyLocalIdSet]
      for (const localId of dirtyLocalIds) {
        const pushSnapshot = pushTimeSnapshots.get(localId)
        if (!pushSnapshot) {
          // 仅 tombstone 路径产生的 localId，无推送快照，直接消化
          docStore.clearDirtyFlags([localId])
          upsertAckShadowForLocal(localId)
          continue
        }

        const currentEntity = docStore.getEntity(localId)
        if (!currentEntity) {
          // 实体在途期间被删除（如 undo 触发 merge）→ 保留脏状态，由 tombstone 路径处理
          continue
        }

        const isStale = (
          String(currentEntity.text ?? '') !== pushSnapshot.text
          || Number(currentEntity.startMs ?? 0) !== pushSnapshot.startMs
          || Number(currentEntity.endMs ?? 0) !== pushSnapshot.endMs
        )

        if (isStale) {
          // 实体在 HTTP 在途期间被修改 → 保留脏标记，下轮 diff 会捕获新变更
          continue
        }

        docStore.clearDirtyFlags([localId])
        upsertAckShadowForLocal(localId)
      }

      if (tombstoneLocalIds.size > 0 || tombstoneSegmentIds.size > 0) {
        docStore.clearTombstones({
          localIds: [...tombstoneLocalIds],
          segmentIds: [...tombstoneSegmentIds],
        })
      }
      for (const localId of tombstoneLocalIds) {
        removeAckShadow(localId, null)
      }
      for (const segmentId of tombstoneSegmentIds) {
        removeAckShadow(null, segmentId)
      }

      lastSyncTime.value = Date.now()
      updatePendingPreview()
      return true
    } finally {
      isSyncing.value = false
    }
  }

  async function pushToBackend(options = {}) {
    if (inflightPushPromise) {
      return inflightPushPromise
    }

    inflightPushPromise = (async () => {
      try {
        return await executePushToBackend(options)
      } finally {
        inflightPushPromise = null
      }
    })()

    return inflightPushPromise
  }

  async function flush() {
    if (syncTimer) {
      clearTimeout(syncTimer)
      syncTimer = null
    }
    if (retryTimer) {
      clearTimeout(retryTimer)
      retryTimer = null
    }

    while (true) {
      const success = await pushToBackend()
      if (!success) {
        return false
      }

      const docStore = useEditorDocumentStore()
      const { entries } = buildPrimitiveDiffEntries({
        docStore,
        sessionStore: useEditorSessionStore(),
        ackShadowByLocalId,
        ackLocalIdBySegmentId,
        maxOps: MAX_BATCH_SIZE,
      })
      if (entries.length === 0) {
        updatePendingPreview()
        return true
      }
    }
  }

  async function flushStrict() {
    const firstPass = await flush()
    if (!firstPass) {
      return false
    }

    const sessionStore = useEditorSessionStore()
    if (!sessionStore.projectId) {
      return true
    }

    await reconcile()
    return flush()
  }

  function reset() {
    if (syncTimer) {
      clearTimeout(syncTimer)
      syncTimer = null
    }
    if (retryTimer) {
      clearTimeout(retryTimer)
      retryTimer = null
    }
    pendingCommands.value = []
    isSyncing.value = false
    lastSyncTime.value = 0
    inflightPushPromise = null
    ackShadowByLocalId.clear()
    ackLocalIdBySegmentId.clear()
  }

  function applyBindings(bindings) {
    const docStore = useEditorDocumentStore()
    for (const binding of bindings) {
      if (!binding?.client_ref_id) continue
      // 实体可能在 HTTP 在途期间被 undo 删除，跳过已不存在的实体
      const currentCold = docStore.getCold(binding.client_ref_id)
      if (!currentCold) continue

      const currentSegmentId = toNormalizedSegmentId(currentCold.segmentId)
      const nextSegmentId = toNormalizedSegmentId(binding.segment_id)
      if (currentSegmentId === nextSegmentId) continue

      docStore.updateColdBinding(binding.client_ref_id, binding.segment_id)
    }
  }

  function applyNormalizedEntities(entities) {
    const docStore = useEditorDocumentStore()
    for (const entity of entities) {
      const localId = entity.client_ref_id || docStore.bindingBySegmentId.get(entity.segment_id)
      if (!localId) continue
      // 实体可能在 HTTP 在途期间被 undo 删除，跳过已不存在的实体
      const currentEntity = docStore.getEntity(localId)
      if (!currentEntity) continue

      const patch = {}
      if (entity.text !== undefined) patch.text = entity.text
      if (entity.start_ms !== undefined) patch.startMs = entity.start_ms
      if (entity.end_ms !== undefined) patch.endMs = entity.end_ms
      if (entity.start !== undefined) patch.startMs = toMs(entity.start)
      if (entity.end !== undefined) patch.endMs = toMs(entity.end)
      if (Object.keys(patch).length > 0) {
        const isTextChanged = patch.text !== undefined
          && String(currentEntity.text ?? '') !== String(patch.text ?? '')
        const isStartChanged = patch.startMs !== undefined
          && Number(currentEntity.startMs ?? 0) !== Number(patch.startMs ?? 0)
        const isEndChanged = patch.endMs !== undefined
          && Number(currentEntity.endMs ?? 0) !== Number(patch.endMs ?? 0)

        if (isTextChanged || isStartChanged || isEndChanged) {
          docStore._applyUpdate(localId, patch)
          if (isStartChanged || isEndChanged) {
            docStore._applyReorder(localId)
          }
        }
      }
      if (entity.segment_id) {
        const currentSegmentId = toNormalizedSegmentId(docStore.getCold(localId)?.segmentId)
        const nextSegmentId = toNormalizedSegmentId(entity.segment_id)
        if (currentSegmentId !== nextSegmentId) {
          docStore.updateColdBinding(localId, entity.segment_id)
        }
      }
    }
  }

  function compensateOrphanCreatedBindings(bindings, updatedEntities) {
    const docStore = useEditorDocumentStore()
    const normalizedBindings = Array.isArray(bindings) ? bindings : []
    if (normalizedBindings.length === 0) {
      return
    }

    const updatedEntityByLocalId = new Map()
    if (Array.isArray(updatedEntities)) {
      for (const entity of updatedEntities) {
        const localId = entity?.client_ref_id ?? null
        if (localId) {
          updatedEntityByLocalId.set(localId, entity)
        }
      }
    }

    for (const binding of normalizedBindings) {
      const localId = binding?.client_ref_id ?? null
      const segmentId = toNormalizedSegmentId(binding?.segment_id)
      if (!localId || !segmentId) {
        continue
      }
      if (docStore.getEntity(localId)) {
        continue
      }

      const updatedEntity = updatedEntityByLocalId.get(localId)
      docStore.upsertTombstone({
        localId,
        segmentId,
        deletedAt: Date.now(),
        before: buildServerEntityBeforeSnapshot(updatedEntity),
      })
    }
  }

  function applyAuthoritativeSegments(serverSegments, options = {}) {
    const {
      serverRevision = null,
      preserveLocalChanges = true,
    } = options
    const sessionStore = useEditorSessionStore()
    const docStore = useEditorDocumentStore()

    const preservedState = captureUnsyncedLocalState(docStore)
    const authoritativeSegmentIds = collectAuthoritativeSegmentIds(serverSegments)
    const {
      appliedCount,
      finalizedChunkIds,
      droppedLocalIds,
    } = applyAuthoritativeSegmentsToDocument(
      docStore,
      serverSegments,
      preserveLocalChanges ? preservedState.dirtyLocalIds : new Set()
    )

    docStore.clearAllDirty()
    rebuildAckShadowFromDocument()

    if (preserveLocalChanges) {
      for (const [localId, snapshot] of preservedState.dirtyLocalSnapshots.entries()) {
        if (droppedLocalIds.has(localId)) {
          continue
        }

        const replayChunkId = snapshot.cold?.chunkId ?? null
        if (Boolean(snapshot.hot?.isDraft) && isSameLogicalChunk(replayChunkId, finalizedChunkIds)) {
          continue
        }

        const replayCold = snapshot.cold
          ? {
              ...snapshot.cold,
            }
          : null
        const replaySegmentId = toNormalizedSegmentId(replayCold?.segmentId)
        if (replayCold && replaySegmentId && !authoritativeSegmentIds.has(replaySegmentId)) {
          replayCold.segmentId = null
          replayCold.sentenceIndex = null
        }

        const existingEntity = docStore.getEntity(localId)
        if (existingEntity) {
          docStore._applyUpdate(localId, {
            text: snapshot.hot.text,
            startMs: snapshot.hot.startMs,
            endMs: snapshot.hot.endMs,
            isDraft: Boolean(snapshot.hot.isDraft),
            isModified: Boolean(snapshot.hot.isModified),
            isDeleted: false,
          })
          docStore._applyReorder(localId)
          if (replayCold) {
            const previousCold = docStore.getCold(localId)
            docStore._applyColdUpdate(localId, replayCold)
            docStore.updateColdBinding(localId, replayCold.segmentId ?? null)

            const previousSentenceIndex = previousCold?.sentenceIndex
            const nextSentenceIndex = replayCold.sentenceIndex
            if (
              previousSentenceIndex !== null
              && previousSentenceIndex !== undefined
              && previousSentenceIndex !== nextSentenceIndex
            ) {
              docStore.bindingBySentenceIndex.delete(previousSentenceIndex)
            }
            if (nextSentenceIndex !== null && nextSentenceIndex !== undefined) {
              docStore.bindingBySentenceIndex.set(nextSentenceIndex, localId)
            }
          }
        } else {
          docStore._applyInsert(
            localId,
            {
              ...snapshot.hot,
              localId,
            },
            replayCold
              ? {
                  ...replayCold,
                  localId,
                }
              : null,
            snapshot.afterLocalId
          )
        }
        docStore.markDirty(localId)
      }
      docStore.replaceTombstones(preservedState.tombstones)
    }

    if (serverRevision !== null && serverRevision !== undefined) {
      sessionStore.ackedRevision = serverRevision
    }

    updatePendingPreview()
    return appliedCount
  }

  async function reconcile(serverRevision = null) {
    const sessionStore = useEditorSessionStore()
    const projectId = sessionStore.projectId
    if (!projectId) {
      throw new Error('缺少 projectId，无法执行对账')
    }

    const serverSegments = await projectApi.getSubtitles(projectId)
    applyAuthoritativeSegments(serverSegments, {
      serverRevision,
      preserveLocalChanges: true,
    })
  }

  function commandToEditorOp(command) {
    // 兼容旧调用方：硬切后仅保留 primitive op 映射
    if (!command) return null

    if (command.type === 'insert_subtitle') {
      return {
        op_id: command.commandId,
        type: 'insert_subtitle',
        client_ref_id: command.localId,
        anchor: {
          before_client_ref_id: command.beforeClientRefId ?? null,
          after_client_ref_id: command.afterClientRefId ?? null,
        },
        after: {
          text: command.entity?.text ?? '',
          start_ms: command.entity?.startMs ?? 0,
          end_ms: command.entity?.endMs ?? 0,
        },
      }
    }

    if (command.type === 'delete_subtitle') {
      return {
        op_id: command.commandId,
        type: 'delete_subtitle',
        client_ref_id: command.localId,
        segment_id: command.segmentId ?? null,
        before: {
          text: command.snapshot?.text ?? '',
          start_ms: command.snapshot?.startMs ?? 0,
          end_ms: command.snapshot?.endMs ?? 0,
        },
      }
    }

    if (command.type === 'update_text') {
      return {
        op_id: command.commandId,
        type: 'update_text',
        client_ref_id: command.localId,
        segment_id: command.segmentId ?? null,
        before: { text: command.before?.text ?? '' },
        after: { text: command.after?.text ?? '' },
      }
    }

    if (command.type === 'update_timing') {
      return {
        op_id: command.commandId,
        type: 'update_timing',
        client_ref_id: command.localId,
        segment_id: command.segmentId ?? null,
        before: {
          start_ms: command.before?.startMs ?? 0,
          end_ms: command.before?.endMs ?? 0,
        },
        after: {
          start_ms: command.after?.startMs ?? 0,
          end_ms: command.after?.endMs ?? 0,
        },
      }
    }

    return null
  }

  return {
    pendingCommands,
    isSyncing,
    lastSyncTime,
    enqueue,
    rewritePendingForHistory,
    flush,
    flushStrict,
    reset,
    applyBindings,
    applyAuthoritativeSegments,
    reconcile,
    commandToEditorOp,
    hasAckBinding,
  }
})
