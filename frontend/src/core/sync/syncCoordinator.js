import { defineStore } from 'pinia'
import { computed, ref, toRaw } from 'vue'
import { legacyApi } from '@/services/api'
import projectApi from '@/services/api/projectApi'
import { useProjectStore } from '@/stores/projectStore'
import { useEditorHistoryStore } from '@/core/editor/historyStore'
import { useEditorTimingStore } from '@/stores/editorTimingStore'
import persistenceClient from '@/core/persistence/persistenceClient'
import {
  findSubtitleByQueueKey,
  normalizeQueueKey,
  resolveProjectSegmentId,
  resolveProjectSegmentIdFromServer,
} from './localBindingStore'
import { buildEditorCommandsFromDiff, computeSubtitleSnapshotDiff } from './editorOpDiff'

const QUEUE_PERSIST_DELAY_MS = 1200
const EDIT_SYNC_DEBOUNCE_MS = 800
const UNDO_REDO_SYNC_DEBOUNCE_MS = 300
let structuralOpSeq = 0
let editorCommandSeq = 0
let browserGuardsRegistered = false
// local_id -> segment_id 缓存，用于 undo/redo 和 merge 补偿缺失的 segment_id
const createdSegmentIdCache = new Map()

export const useSyncCoordinatorStore = defineStore('syncCoordinator', () => {
  const projectStore = useProjectStore()
  const editorHistoryStore = useEditorHistoryStore()
  const editorTimingStore = useEditorTimingStore()

  const currentSyncIdentityId = ref(null)
  const pendingUpdates = ref(new Map())
  const editSyncErrors = ref(new Map())
  const isEditSyncing = ref(false)
  const structuralSyncErrors = ref(new Map())
  const undoRedoLastError = ref(null)
  const isUndoRedoSyncing = ref(false)

  const pendingCount = computed(() => pendingUpdates.value.size)
  const structuralErrorCount = computed(() => structuralSyncErrors.value.size)
  const hasStructuralErrors = computed(() => structuralSyncErrors.value.size > 0)
  const inflightStructuralCount = computed(() => inflightStructural.size)

  const inflightStructural = new Map()
  let persistQueueTimer = null
  let processQueueTimer = null
  let inflightEditSyncPromise = null
  let initialUndoRedoSnapshot = null
  let undoRedoDebounceTimer = null
  let inflightUndoRedoPromise = null
  let undoRedoApplyChain = Promise.resolve()
  let editFlushBarrier = Promise.resolve()
  let structuralBarrier = Promise.resolve()

  function getActiveIdentity(candidateIdentityId = null) {
    return candidateIdentityId || currentSyncIdentityId.value || projectStore.primaryId || null
  }

  function clearEditTimers() {
    if (persistQueueTimer) {
      clearTimeout(persistQueueTimer)
      persistQueueTimer = null
    }
    if (processQueueTimer) {
      clearTimeout(processQueueTimer)
      processQueueTimer = null
    }
  }

  function mergePendingUpdate(queueKey, nextUpdate) {
    const normalizedKey = normalizeQueueKey(queueKey)
    if (normalizedKey === null || !nextUpdate || typeof nextUpdate !== 'object') {
      return
    }
    const previous = pendingUpdates.value.get(normalizedKey) || {}
    pendingUpdates.value.set(normalizedKey, {
      ...previous,
      ...nextUpdate,
    })
  }

  function writeEmergencyBuffer(identityId = getActiveIdentity()) {
    if (!identityId || pendingUpdates.value.size === 0) return
    persistenceClient.writeEmergencyQueue(identityId, pendingUpdates.value)
  }

  function registerBrowserGuards() {
    const hasWindow = typeof window !== 'undefined' && typeof window.addEventListener === 'function'
    const hasDocument = typeof document !== 'undefined' && typeof document.addEventListener === 'function'
    if (browserGuardsRegistered || (!hasWindow && !hasDocument)) return

    const flushEmergencyBuffer = () => {
      const identityId = getActiveIdentity()
      if (!identityId || pendingUpdates.value.size === 0) return
      persistenceClient.writeEmergencyQueue(identityId, pendingUpdates.value)
    }

    if (hasWindow) {
      window.addEventListener('beforeunload', flushEmergencyBuffer)
    }
    if (hasDocument) {
      document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'hidden') {
          flushEmergencyBuffer()
        }
      })
    }
    browserGuardsRegistered = true
  }

  async function saveActiveQueue(identityId = getActiveIdentity()) {
    if (!identityId) return
    await persistenceClient.saveQueue(identityId, pendingUpdates.value)
  }

  function scheduleQueuePersistence(identityId = getActiveIdentity()) {
    if (!identityId) return
    if (persistQueueTimer) {
      clearTimeout(persistQueueTimer)
    }
    persistQueueTimer = setTimeout(() => {
      persistQueueTimer = null
      void saveActiveQueue(identityId)
    }, QUEUE_PERSIST_DELAY_MS)
  }

  async function flushQueuePersistence(identityId = getActiveIdentity()) {
    if (persistQueueTimer) {
      clearTimeout(persistQueueTimer)
      persistQueueTimer = null
    }
    await saveActiveQueue(identityId)
  }

  async function bindSyncIdentity(identityId) {
    const resolvedIdentityId = getActiveIdentity(identityId)
    clearEditTimers()

    if (!resolvedIdentityId) {
      if (currentSyncIdentityId.value) {
        await saveActiveQueue(currentSyncIdentityId.value)
      }
      currentSyncIdentityId.value = null
      pendingUpdates.value = new Map()
      editSyncErrors.value.clear()
      return
    }

    if (currentSyncIdentityId.value === resolvedIdentityId) {
      return
    }

    if (currentSyncIdentityId.value) {
      await saveActiveQueue(currentSyncIdentityId.value)
    }

    pendingUpdates.value = await persistenceClient.loadQueue(resolvedIdentityId)
    currentSyncIdentityId.value = resolvedIdentityId
    editSyncErrors.value.clear()
    createdSegmentIdCache.clear()
    registerBrowserGuards()
  }

  async function ensureProjectContext() {
    if (projectStore.meta.projectId) {
      return String(projectStore.meta.projectId)
    }

    const fallbackJobId = String(projectStore.meta.jobId || '').trim()
    if (!fallbackJobId) {
      throw new Error('缺少 project_id，且无可转换的 job_id')
    }

    const resolved = await legacyApi.resolveTask(fallbackJobId)
    const projectId = String(resolved?.project_id || '').trim()
    if (!projectId) {
      throw new Error(`任务 ${fallbackJobId} 尚未建立 project_id`) 
    }

    projectStore.patchMeta({ projectId })
    return projectId
  }

  function applyPendingEditsToStore(targetStore = projectStore) {
    if (!targetStore || pendingUpdates.value.size === 0) return 0

    let appliedCount = 0
    pendingUpdates.value.forEach((update, queueKey) => {
      const subtitle = findSubtitleByQueueKey(targetStore, queueKey)
      if (!subtitle || !update || typeof update !== 'object') return

      const displayUpdate = { ...update }
      if (update.start !== undefined) {
        displayUpdate.start = targetStore.toDisplayTime(update.start)
      }
      if (update.end !== undefined) {
        displayUpdate.end = targetStore.toDisplayTime(update.end)
      }

      targetStore.updateSubtitle(subtitle.id, displayUpdate, { isUserEdit: true })
      appliedCount += 1
    })
    return appliedCount
  }

  async function runProcessQueuePass() {
    if (pendingUpdates.value.size === 0) return
    const identityId = getActiveIdentity()
    if (!identityId) return

    isEditSyncing.value = true
    try {
      const projectId = await ensureProjectContext()
      const batch = new Map(pendingUpdates.value)
      pendingUpdates.value.clear()
      const snapshotCache = { value: null }

      for (const [queueKey, data] of batch.entries()) {
        try {
          const payload = { ...(data || {}) }
          const clientRevision = payload.clientRevision
          delete payload.clientRevision

          let segmentId = resolveProjectSegmentId(projectStore, queueKey)
          if (!segmentId) {
            segmentId = await resolveProjectSegmentIdFromServer({
              projectApi,
              projectId,
              rawQueueKey: queueKey,
              snapshotCache,
            })
          }
          if (!segmentId) {
            const staleSubtitle = findSubtitleByQueueKey(projectStore, queueKey)
            if (!staleSubtitle) {
              editSyncErrors.value.delete(queueKey)
              continue
            }
            const isWaitingSegmentBinding = !staleSubtitle.segment_id
              && (staleSubtitle.sentenceIndex === undefined || staleSubtitle.sentenceIndex === null)
            if (isWaitingSegmentBinding) {
              mergePendingUpdate(queueKey, data)
              editSyncErrors.value.delete(queueKey)
              continue
            }
            throw new Error(`无法解析 segment_id，同步键=${queueKey}`)
          }

          if (Object.keys(payload).length === 0) {
            editSyncErrors.value.delete(queueKey)
            continue
          }

          await projectApi.updateSubtitle(projectId, segmentId, payload)

          // V3.2.5+dev.20260311.01: 版本号冲突检测 - 跳过过期回填
          const subtitle = findSubtitleByQueueKey(projectStore, queueKey)
          if (subtitle && clientRevision !== undefined) {
            const currentRevision = subtitle.revision || 0
            if (currentRevision > clientRevision) {
              console.warn(`[SyncCoordinator] 跳过过期回填: queueKey=${queueKey}, current=${currentRevision} > synced=${clientRevision}`)
              editSyncErrors.value.delete(queueKey)
              continue
            }
          }

          editSyncErrors.value.delete(queueKey)
        } catch (error) {
          if (error?.status === 404) {
            editSyncErrors.value.delete(queueKey)
            continue
          }
          console.error('[SyncCoordinator] 字幕编辑同步失败:', error)
          editSyncErrors.value.set(queueKey, error?.message || '同步失败')
          mergePendingUpdate(queueKey, data)
        }
      }

      await saveActiveQueue(identityId)
    } finally {
      isEditSyncing.value = false
    }
  }

  async function processQueue() {
    if (inflightEditSyncPromise) {
      await inflightEditSyncPromise
    }
    if (pendingUpdates.value.size === 0) return

    const currentPromise = runProcessQueuePass()
    inflightEditSyncPromise = currentPromise
    editFlushBarrier = currentPromise
    try {
      await currentPromise
    } finally {
      if (inflightEditSyncPromise === currentPromise) {
        inflightEditSyncPromise = null
      }
    }
  }

  async function flushEditQueue() {
    if (processQueueTimer) {
      clearTimeout(processQueueTimer)
      processQueueTimer = null
    }
    await processQueue()
    await editFlushBarrier
  }

  function scheduleEditSync() {
    if (processQueueTimer) {
      clearTimeout(processQueueTimer)
    }
    processQueueTimer = setTimeout(() => {
      processQueueTimer = null
      void processQueue()
    }, EDIT_SYNC_DEBOUNCE_MS)
  }

  function enqueueSubtitleEdit(index, { text, start, end }) {
    const identityId = getActiveIdentity()
    let queueKey = normalizeQueueKey(index)
    if (queueKey === null) return

    const subtitle = findSubtitleByQueueKey(projectStore, queueKey)
    if (!subtitle) return

    const segmentId = resolveProjectSegmentId(projectStore, queueKey)
    if (segmentId) {
      queueKey = segmentId
    }

    const clientRevision = subtitle.revision || 0

    const update = { clientRevision }
    if (text !== undefined) update.text = text
    if (start !== undefined) update.start = editorTimingStore.toBaseTime(start)
    if (end !== undefined) update.end = editorTimingStore.toBaseTime(end)

    mergePendingUpdate(queueKey, update)
    if (identityId) {
      writeEmergencyBuffer(identityId)
      scheduleQueuePersistence(identityId)
    }
    scheduleEditSync()
  }

  async function forceSyncNow() {
    await flushQueuePersistence()
    await flushEditQueue()
  }

  function trackStructuralOperation(type, promise) {
    structuralOpSeq += 1
    const opId = `${type}-${Date.now()}-${structuralOpSeq}`
    inflightStructural.set(opId, promise)

    promise
      .then(() => {
        inflightStructural.delete(opId)
        structuralSyncErrors.value.delete(opId)
      })
      .catch((error) => {
        inflightStructural.delete(opId)
        structuralSyncErrors.value.set(opId, {
          type,
          error: error?.message || '同步失败',
          timestamp: Date.now(),
        })
      })

    return opId
  }

  async function waitAllStructuralOperations() {
    if (inflightStructural.size === 0) return
    await Promise.allSettled([...inflightStructural.values()])
  }

  function clearAllStructuralErrors() {
    structuralSyncErrors.value.clear()
  }

  function clearStructuralError(opId) {
    structuralSyncErrors.value.delete(opId)
  }

  function resetStructuralState() {
    inflightStructural.clear()
    structuralSyncErrors.value.clear()
  }

  function nextCommandId(type) {
    editorCommandSeq += 1
    return `${type}-${Date.now()}-${editorCommandSeq}`
  }

  async function executeEditorCommands(commands = []) {
    if (!Array.isArray(commands) || commands.length === 0) {
      return {
        updated: 0,
        created: 0,
        deleted: 0,
        created_segments: [],
        errors: [],
      }
    }

    console.log('[executeEditorCommands] 发送命令:', commands.length, '条')
    const projectId = await ensureProjectContext()
    const payload = { commands }
    const envelope = await projectApi.applyEditorOps(projectId, payload)
    console.log('[executeEditorCommands] 收到响应:', { success: envelope?.success, result: envelope?.data })
    const result = envelope?.data || {}
    const success = envelope?.success

    if (Array.isArray(result?.created_segments) && result.created_segments.length > 0) {
      patchCreatedSegmentIds(result.created_segments)
    }

    if (success === false || (Array.isArray(result?.errors) && result.errors.length > 0)) {
      const message = (result?.errors || []).join('; ') || '批量命令同步失败'
      console.error('[executeEditorCommands] 同步失败:', message)
      throw new Error(message)
    }

    console.log('[executeEditorCommands] 同步成功')
    return result
  }

  function submitStructuralCommands(type, commands = []) {
    console.log('[submitStructuralCommands] 开始:', { type, commandCount: commands.length })

    const previousBarrier = structuralBarrier

    const promise = (async () => {
      console.log('[submitStructuralCommands] 等待 flushEditQueue...')
      await flushEditQueue()
      console.log('[submitStructuralCommands] 等待 previousBarrier...')
      await previousBarrier
      console.log('[submitStructuralCommands] 开始执行命令...')
      const result = await executeEditorCommands(commands)
      console.log('[submitStructuralCommands] 执行完成')
      return result
    })()

    structuralBarrier = promise.catch(() => {})
    const opId = trackStructuralOperation(type, promise)
    console.log('[submitStructuralCommands] 返回 opId:', opId)
    return { opId, promise }
  }

  async function flushAllSync() {
    await flushEditQueue()
    await flushUndoRedoSync()
    await waitAllStructuralOperations()
    await flushEditQueue()
    await flushUndoRedoSync()
  }

  function captureSnapshot() {
    const snapshot = new Map()
    for (const subtitle of toRaw(projectStore.subtitles)) {
      const key = subtitle.segment_id || subtitle.id
      if (key == null) continue
      snapshot.set(String(key), {
        localId: subtitle.id ?? null,
        segment_id: subtitle.segment_id || null,
        sentenceIndex: subtitle.sentenceIndex,
        text: subtitle.text,
        start: editorTimingStore.toBaseTime(subtitle.start),
        end: editorTimingStore.toBaseTime(subtitle.end),
      })
    }
    return snapshot
  }

  function hydrateSnapshotSegmentIdsFromCache(snapshot) {
    if (!(snapshot instanceof Map)) {
      return
    }
    for (const [, entry] of snapshot) {
      if (!entry || entry.segment_id || entry.localId == null) {
        continue
      }
      const cached = createdSegmentIdCache.get(String(entry.localId))
      if (cached) {
        entry.segment_id = cached
      }
    }
  }

  function patchCreatedSegmentIds(createdSegments) {
    console.log('[patchCreatedSegmentIds] 开始回填，收到:', createdSegments.length, '条记录')
    for (const created of createdSegments) {
      if (!created.segment_id) continue

      console.log('[patchCreatedSegmentIds] 处理:', { local_id: created.local_id, segment_id: created.segment_id })

      // 缓存 local_id -> segment_id，供 undo/redo diff 和 merge 补偿使用
      if (created.local_id != null) {
        createdSegmentIdCache.set(String(created.local_id), created.segment_id)
      }

      let match = null
      if (created.local_id !== undefined && created.local_id !== null) {
        match = projectStore.subtitles.find(
          (subtitle) => String(subtitle.id) === String(created.local_id)
        )
        console.log('[patchCreatedSegmentIds] 通过 local_id 匹配:', match ? `找到 ${match.id}, 已有 segment_id: ${match.segment_id}` : '未找到')
      }

      if (!match) {
        match = projectStore.subtitles.find(
          (subtitle) => !subtitle.segment_id && subtitle.sentenceIndex === created.legacy_index
        )
        console.log('[patchCreatedSegmentIds] 通过 legacy_index 匹配:', match ? `找到 ${match.id}` : '未找到')
      }

      if (!match && created.start !== undefined && created.end !== undefined) {
        match = projectStore.subtitles.find((subtitle) => {
          if (subtitle.segment_id) return false
          const textEqual = (subtitle.text || '') === (created.text || '')
          const startDiff = Math.abs(editorTimingStore.toBaseTime(subtitle.start) - Number(created.start))
          const endDiff = Math.abs(editorTimingStore.toBaseTime(subtitle.end) - Number(created.end))
          return textEqual && startDiff < 0.01 && endDiff < 0.01
        })
        console.log('[patchCreatedSegmentIds] 通过时间+文本匹配:', match ? `找到 ${match.id}` : '未找到')
      }

      if (match) {
        console.log('[patchCreatedSegmentIds] 准备回填 segment_id:', { localId: match.id, segment_id: created.segment_id })
        editorHistoryStore.pauseHistory()
        try {
          const payload = { segment_id: created.segment_id }
          if (
            created.legacy_index !== undefined
            && created.legacy_index !== null
            && (match.sentenceIndex === undefined || match.sentenceIndex === null)
          ) {
            payload.sentenceIndex = created.legacy_index
          }
          projectStore.updateSubtitle(match.id, payload, { isUserEdit: false })
          console.log('[patchCreatedSegmentIds] 回填完成:', match.id)
        } finally {
          editorHistoryStore.resumeHistory()
        }
      } else {
        console.warn('[patchCreatedSegmentIds] 未找到匹配的字幕:', created)
      }
    }
  }

  async function executeUndoRedoSync() {
    if (inflightUndoRedoPromise) {
      return inflightUndoRedoPromise
    }
    if (!initialUndoRedoSnapshot) return

    if (!projectStore.meta.projectId) {
      initialUndoRedoSnapshot = null
      return
    }

    // 等待飞行中的结构性操作（split/merge）完成，确保 segment_id 已回写
    await waitAllStructuralOperations()

    // 补偿快照中缺失的 segment_id（含 initial/final 两端），
    // 避免撤销 split/merge 时误把“恢复旧段”降级为“新建段”。
    hydrateSnapshotSegmentIdsFromCache(initialUndoRedoSnapshot)

    const finalSnapshot = captureSnapshot()
    hydrateSnapshotSegmentIdsFromCache(finalSnapshot)
    const diff = computeSubtitleSnapshotDiff(initialUndoRedoSnapshot, finalSnapshot)
    initialUndoRedoSnapshot = null
    const commands = buildEditorCommandsFromDiff(diff, nextCommandId, 'undo-redo')
    if (commands.length === 0) return

    inflightUndoRedoPromise = (async () => {
      isUndoRedoSyncing.value = true
      undoRedoLastError.value = null
      try {
        await executeEditorCommands(commands)
        clearAllStructuralErrors()
      } catch (error) {
        undoRedoLastError.value = error?.message || '撤销/重做同步失败'
        console.error('[SyncCoordinator] undo/redo 同步异常:', error)
      } finally {
        isUndoRedoSyncing.value = false
      }
    })()

    try {
      await inflightUndoRedoPromise
    } finally {
      inflightUndoRedoPromise = null
      if (initialUndoRedoSnapshot && !undoRedoDebounceTimer) {
        scheduleUndoRedoSync()
      }
    }
  }

  function scheduleUndoRedoSync() {
    if (undoRedoDebounceTimer) {
      clearTimeout(undoRedoDebounceTimer)
    }
    undoRedoDebounceTimer = setTimeout(() => {
      undoRedoDebounceTimer = null
      void executeUndoRedoSync()
    }, UNDO_REDO_SYNC_DEBOUNCE_MS)
  }

  async function flushUndoRedoSync() {
    await undoRedoApplyChain.catch(() => {})
    if (undoRedoDebounceTimer) {
      clearTimeout(undoRedoDebounceTimer)
      undoRedoDebounceTimer = null
    }
    await executeUndoRedoSync()
    if (inflightUndoRedoPromise) {
      await inflightUndoRedoPromise
    }
    if (undoRedoLastError.value) {
      const errorMessage = undoRedoLastError.value
      undoRedoLastError.value = null
      throw new Error(`撤销/重做同步失败: ${errorMessage}`)
    }
  }

  async function waitForUndoRedoStructuralBarrier() {
    // undo/redo 必须在结构性操作稳定后执行，避免 merge/split 回推覆盖本地撤销结果。
    await waitAllStructuralOperations()
    await new Promise((resolve) => setTimeout(resolve, 0))
  }

  async function queueUndoRedoApply(action) {
    undoRedoApplyChain = undoRedoApplyChain
      .catch((error) => {
        console.warn('[SyncCoordinator] 上一次 undo/redo 串行任务异常，继续后续请求:', error)
      })
      .then(async () => {
        await waitForUndoRedoStructuralBarrier()
        const canApply = action === 'undo'
          ? Boolean(editorHistoryStore.canUndo)
          : Boolean(editorHistoryStore.canRedo)
        if (!canApply) {
          return
        }
        if (!initialUndoRedoSnapshot) {
          initialUndoRedoSnapshot = captureSnapshot()
        }
        if (action === 'undo') {
          editorHistoryStore.undo()
        } else {
          editorHistoryStore.redo()
        }
        scheduleUndoRedoSync()
      })
      .catch((error) => {
        undoRedoLastError.value = error?.message || '撤销/重做执行失败'
        console.error('[SyncCoordinator] queueUndoRedoApply 异常:', error)
      })
    return undoRedoApplyChain
  }

  function undoWithSync() {
    void queueUndoRedoApply('undo')
  }

  function redoWithSync() {
    void queueUndoRedoApply('redo')
  }

  /**
   * 通过 local_id 查询缓存中的 segment_id
   * 用于 merge 等场景下补偿尚未回写到 store 的 segment_id
   */
  function resolveSegmentIdFromCache(localId) {
    if (localId == null) return null
    return createdSegmentIdCache.get(String(localId)) || null
  }

  return {
    currentSyncIdentityId,
    pendingUpdates,
    pendingCount,
    editSyncErrors,
    isEditSyncing,
    structuralSyncErrors,
    structuralErrorCount,
    hasStructuralErrors,
    inflightStructuralCount,
    isUndoRedoSyncing,
    undoRedoLastError,
    bindSyncIdentity,
    applyPendingEditsToStore,
    enqueueSubtitleEdit,
    forceSyncNow,
    flushEditQueue,
    trackStructuralOperation,
    waitAllStructuralOperations,
    clearAllStructuralErrors,
    clearStructuralError,
    resetStructuralState,
    nextCommandId,
    executeEditorCommands,
    submitStructuralCommands,
    undoWithSync,
    redoWithSync,
    flushUndoRedoSync,
    flushAllSync,
    writeEmergencyBuffer,
    resolveSegmentIdFromCache,
  }
})
