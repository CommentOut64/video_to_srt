import { nextTick, ref } from 'vue'
import { useEditBufferStore } from '@/core/editor/editBufferStore'

const SAVE_PRIORITY = {
  auto: 1,
  manual: 2,
  force: 3,
}

const FLOAT_COMPARE_EPSILON = 0.0005

function normalizeErrorMessage(error) {
  if (!error) return '未知错误'
  if (typeof error === 'string') return error
  return error.message || '未知错误'
}

function pickFirstSyncErrorDetail(editSyncErrors, structuralSyncErrors) {
  if (editSyncErrors?.size > 0 && typeof editSyncErrors.entries === 'function') {
    const firstError = editSyncErrors.entries().next().value
    if (firstError) {
      return `首条错误: [${firstError[0]}] ${firstError[1]}`
    }
  }
  if (structuralSyncErrors?.size > 0 && typeof structuralSyncErrors.entries === 'function') {
    const firstError = structuralSyncErrors.entries().next().value
    if (firstError) {
      const payload = firstError[1] || {}
      return `首条结构性错误: [${payload.type || 'unknown'}] ${payload.error || '同步失败'}`
    }
  }
  return ''
}

function pickPendingQueueDetail(pendingUpdates, limit = 3) {
  if (!(pendingUpdates instanceof Map) || pendingUpdates.size === 0) {
    return ''
  }
  const keys = []
  for (const key of pendingUpdates.keys()) {
    keys.push(String(key))
    if (keys.length >= limit) break
  }
  const suffix = pendingUpdates.size > limit ? ` ...(+${pendingUpdates.size - limit})` : ''
  return keys.length > 0 ? `待同步键样例: ${keys.join(', ')}${suffix}` : ''
}

function normalizeSaveOptions(options = {}) {
  return {
    needBackendSnapshot: options.needBackendSnapshot === true,
    blockOnError: options.blockOnError === true,
  }
}

export function createSaveCoordinator({
  projectStore,
  syncCoordinator,
  editorSessionStore,
  projectApi,
}) {
  if (!projectStore) {
    throw new Error('缺少 projectStore，无法初始化保存协调器')
  }
  if (!syncCoordinator) {
    throw new Error('缺少 syncCoordinator，无法初始化保存协调器')
  }
  if (!editorSessionStore) {
    throw new Error('缺少 editorSessionStore，无法初始化保存协调器')
  }

  const editBufferStore = useEditBufferStore()
  const isSaving = ref(false)
  const activeSaveMode = ref(null)
  const lastSaveError = ref(null)
  const lastSavedAt = ref(null)

  let runningPromise = null
  let pendingRequest = null

  function resolveSubtitleById(subtitleId) {
    return projectStore.subtitles.find((subtitle) => String(subtitle.id) === String(subtitleId)) || null
  }

  function resolveSyncKey(subtitle) {
    return subtitle?.segment_id ?? subtitle?.sentenceIndex ?? subtitle?.id
  }

  function enqueueSubtitlePatch(subtitle, patch = {}) {
    if (!subtitle || !patch || typeof patch !== 'object') return
    const syncKey = resolveSyncKey(subtitle)
    if (syncKey === undefined || syncKey === null) return
    syncCoordinator.enqueueSubtitleEdit(syncKey, patch)
  }

  function hasPendingDraftBuffers() {
    const pendingTextDrafts = editBufferStore.getPendingTextDraftEntries()
    const pendingTimeDrafts = editBufferStore.getPendingTimeDraftEntries()
    return pendingTextDrafts.length > 0 || pendingTimeDrafts.length > 0
  }

  function shouldSkipAutoSave() {
    const hasDirty = Boolean(projectStore.isDirty)
    const pending = Number(syncCoordinator.pendingCount || 0)
    const structuralInFlight = Number(syncCoordinator.inflightStructuralCount || 0)
    return !hasDirty && pending === 0 && structuralInFlight === 0 && !hasPendingDraftBuffers()
  }

  async function flushEditBufferDrafts() {
    if (typeof document !== 'undefined') {
      const activeElement = document.activeElement
      if (activeElement && typeof activeElement.blur === 'function') {
        activeElement.blur()
      }
    }
    await nextTick()
    await Promise.resolve()

    let textCommitCount = 0
    let timeCommitCount = 0

    const pendingTextDrafts = editBufferStore.getPendingTextDraftEntries()
    pendingTextDrafts.forEach((entry) => {
      const subtitle = resolveSubtitleById(entry.subtitleId)
      if (!subtitle) {
        editBufferStore.clearAllDrafts(entry.subtitleId)
        return
      }

      let committedDraftText = null
      editBufferStore.flushTextCommit(entry.subtitleId, {
        immediate: true,
        getCommittedText: () => subtitle.text,
        onCommit: (draftText) => {
          committedDraftText = draftText
        },
      })

      if (committedDraftText !== null && committedDraftText !== subtitle.text) {
        projectStore.updateSubtitle(subtitle.id, { text: committedDraftText }, { isUserEdit: true })
        editBufferStore.syncCommittedText(subtitle.id, committedDraftText)
        const updatedSubtitle = resolveSubtitleById(subtitle.id) || subtitle
        enqueueSubtitlePatch(updatedSubtitle, { text: committedDraftText })
        textCommitCount += 1
      }

      if (entry.isEditing || entry.isComposing || entry.hasPendingTimer) {
        editBufferStore.endTextEdit(entry.subtitleId)
      }
    })

    const pendingTimeDrafts = editBufferStore.getPendingTimeDraftEntries()
    pendingTimeDrafts.forEach((entry) => {
      const subtitle = resolveSubtitleById(entry.subtitleId)
      if (!subtitle) {
        editBufferStore.clearTimeDraft(entry.subtitleId)
        return
      }

      const currentStart = Number(subtitle.start)
      const currentEnd = Number(subtitle.end)
      const draftStart = Number(entry.start)
      const draftEnd = Number(entry.end)

      const nextStart = Number.isFinite(draftStart) ? draftStart : currentStart
      const nextEnd = Number.isFinite(draftEnd) ? draftEnd : currentEnd

      const hasChangedStart = Math.abs(nextStart - currentStart) > FLOAT_COMPARE_EPSILON
      const hasChangedEnd = Math.abs(nextEnd - currentEnd) > FLOAT_COMPARE_EPSILON

      if (!hasChangedStart && !hasChangedEnd) {
        editBufferStore.clearTimeDraft(entry.subtitleId)
        return
      }

      const patch = {}
      if (hasChangedStart) patch.start = nextStart
      if (hasChangedEnd) patch.end = nextEnd

      projectStore.updateSubtitle(subtitle.id, patch, { isUserEdit: true })
      const updatedSubtitle = resolveSubtitleById(subtitle.id) || subtitle
      editBufferStore.syncCommittedTime(subtitle.id, {
        start: updatedSubtitle.start,
        end: updatedSubtitle.end,
      })
      enqueueSubtitlePatch(updatedSubtitle, patch)
      editBufferStore.clearTimeDraft(entry.subtitleId)
      timeCommitCount += 1
    })

    return {
      textCommitCount,
      timeCommitCount,
    }
  }

  function assertSyncSettled() {
    const pending = Number(syncCoordinator.pendingCount || 0)
    const editSyncErrors = syncCoordinator.editSyncErrors
    const structuralSyncErrors = syncCoordinator.structuralSyncErrors
    const pendingUpdates = syncCoordinator.pendingUpdates
    const structuralInFlight = Number(syncCoordinator.inflightStructuralCount || 0)
    const editErrorCount = Number(editSyncErrors?.size || 0)
    const structuralErrorCount = Number(structuralSyncErrors?.size || 0)

    if (pending === 0 && editErrorCount === 0 && structuralErrorCount === 0 && structuralInFlight === 0) {
      return
    }

    const errorDetail = pickFirstSyncErrorDetail(editSyncErrors, structuralSyncErrors)
    const pendingDetail = pickPendingQueueDetail(pendingUpdates)
    const inflightDetail = structuralInFlight > 0 ? `结构性同步进行中 ${structuralInFlight} 条` : ''
    const detailParts = [pendingDetail, inflightDetail, errorDetail].filter(Boolean)
    const detailSuffix = detailParts.length > 0 ? `，${detailParts.join('；')}` : ''
    throw new Error(
      `保存后仍有未同步修改（待同步 ${pending} 条，结构性进行中 ${structuralInFlight} 条，错误 ${editErrorCount + structuralErrorCount} 条${detailSuffix}）`
    )
  }

  async function fetchBackendSnapshotIfNeeded(needBackendSnapshot) {
    if (!needBackendSnapshot) {
      return null
    }
    if (!projectApi || typeof projectApi.getSubtitles !== 'function') {
      throw new Error('缺少 projectApi.getSubtitles，无法获取后端权威快照')
    }

    const projectId = projectStore.meta.projectId
    if (!projectId) {
      throw new Error('缺少 project_id，无法获取后端权威快照')
    }

    const segments = await projectApi.getSubtitles(projectId)
    if (!Array.isArray(segments)) {
      throw new Error('后端字幕快照格式非法，导出已阻止')
    }
    return segments
  }

  async function executeSaveRequest(request) {
    const options = normalizeSaveOptions(request.options)
    if (request.mode === 'auto' && shouldSkipAutoSave()) {
      return {
        ok: true,
        skipped: true,
        mode: request.mode,
        reason: request.reason,
        backendSegments: null,
      }
    }

    isSaving.value = true
    activeSaveMode.value = request.mode
    lastSaveError.value = null

    try {
      await flushEditBufferDrafts()
      await syncCoordinator.flushAllSync()
      assertSyncSettled()
      await editorSessionStore.saveWorkingCopy()
      const backendSegments = await fetchBackendSnapshotIfNeeded(options.needBackendSnapshot)
      lastSavedAt.value = Date.now()

      return {
        ok: true,
        skipped: false,
        mode: request.mode,
        reason: request.reason,
        backendSegments,
      }
    } catch (error) {
      const message = normalizeErrorMessage(error)
      lastSaveError.value = message

      if (options.blockOnError) {
        throw new Error(message)
      }

      return {
        ok: false,
        skipped: false,
        mode: request.mode,
        reason: request.reason,
        backendSegments: null,
        error: message,
      }
    } finally {
      isSaving.value = false
      activeSaveMode.value = null
    }
  }

  function mergeQueuedRequest(currentRequest, incomingRequest) {
    if (!currentRequest) {
      return incomingRequest
    }

    const currentPriority = SAVE_PRIORITY[currentRequest.mode] || 0
    const incomingPriority = SAVE_PRIORITY[incomingRequest.mode] || 0
    const keepIncoming = incomingPriority >= currentPriority
    const winner = keepIncoming ? incomingRequest : currentRequest
    const loser = keepIncoming ? currentRequest : incomingRequest

    winner.options = {
      ...normalizeSaveOptions(winner.options),
      needBackendSnapshot:
        normalizeSaveOptions(winner.options).needBackendSnapshot
        || normalizeSaveOptions(loser.options).needBackendSnapshot,
      blockOnError:
        normalizeSaveOptions(winner.options).blockOnError
        || normalizeSaveOptions(loser.options).blockOnError,
    }
    winner.waiters = [...(winner.waiters || []), ...(loser.waiters || [])]
    return winner
  }

  async function drainQueue() {
    if (runningPromise) {
      return runningPromise
    }

    runningPromise = (async () => {
      while (pendingRequest) {
        const request = pendingRequest
        pendingRequest = null

        try {
          const result = await executeSaveRequest(request)
          request.waiters.forEach((waiter) => waiter.resolve(result))
        } catch (error) {
          request.waiters.forEach((waiter) => waiter.reject(error))
        }
      }
    })()

    try {
      await runningPromise
    } finally {
      runningPromise = null
    }
  }

  function enqueueSave(mode, reason, options = {}) {
    return new Promise((resolve, reject) => {
      const request = {
        mode,
        reason,
        options: normalizeSaveOptions(options),
        waiters: [{ resolve, reject }],
      }
      pendingRequest = mergeQueuedRequest(pendingRequest, request)
      void drainQueue()
    })
  }

  function runAutoSave(reason = 'interval', options = {}) {
    return enqueueSave('auto', reason, {
      ...options,
      blockOnError: false,
      needBackendSnapshot: false,
    })
  }

  function runManualSave(reason = 'manual', options = {}) {
    return enqueueSave('manual', reason, {
      ...options,
      blockOnError: true,
      needBackendSnapshot: false,
    })
  }

  function runForceSave(reason = 'force', options = {}) {
    return enqueueSave('force', reason, {
      ...options,
      blockOnError: true,
    })
  }

  return {
    isSaving,
    activeSaveMode,
    lastSaveError,
    lastSavedAt,
    runAutoSave,
    runManualSave,
    runForceSave,
  }
}
