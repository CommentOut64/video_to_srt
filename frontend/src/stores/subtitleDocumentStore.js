import { defineStore } from 'pinia'
import { computed, ref, toRaw, watch } from 'vue'
import localforage from 'localforage'
import transcriptionApi from '@/services/api/transcriptionApi'
import projectApi from '@/services/api/projectApi'
import { migrateSubtitleSyncQueue } from '@/state/migrations/migrateSubtitleSyncQueue'
import { useProjectStore } from './projectStore'
import { useTaskRuntimeStore } from './taskRuntimeStore'

const EDIT_QUEUE_PREFIX = 'subtitle-edit-queue-'
const TERMINAL_STATUSES = new Set(['finished', 'canceled', 'force_canceled', 'removed', 'failed'])
let beforeUnloadRegistered = false

function getQueueKey(identityId) {
  return `${EDIT_QUEUE_PREFIX}${identityId}`
}

function debounce(fn, delay) {
  let timeoutId = null
  return (...args) => {
    if (timeoutId) {
      clearTimeout(timeoutId)
    }
    timeoutId = setTimeout(() => {
      fn(...args)
    }, delay)
  }
}

export const useSubtitleDocumentStore = defineStore('subtitleDocument', () => {
  const projectStore = useProjectStore()
  const taskRuntimeStore = useTaskRuntimeStore()

  const activeJobId = ref(null)
  const convergedTerminalStatus = ref(new Map())

  const currentSyncIdentityId = ref(null)
  const pendingUpdates = ref(new Map())
  const isSyncing = ref(false)
  const syncErrors = ref(new Map())

  const subtitles = computed(() => projectStore.subtitles)
  const timeOffsetSec = computed(() => projectStore.subtitleOffset)
  const selectedSubtitleId = computed(() => projectStore.view.selectedSubtitleId)

  const activeTaskStatus = computed(() => {
    if (!activeJobId.value) return 'idle'
    const task = taskRuntimeStore.getTask(activeJobId.value)
    if (task?.status) return task.status
    return taskRuntimeStore.getRawState(activeJobId.value).status || 'idle'
  })

  function getActiveIdentity(candidateIdentityId = null) {
    return (
      candidateIdentityId
      || currentSyncIdentityId.value
      || projectStore.primaryId
      || activeJobId.value
      || null
    )
  }

  function hasProjectContext() {
    return Boolean(projectStore.meta.projectId)
  }

  function resolveProjectSegmentId(segmentOrIndex) {
    if (segmentOrIndex === undefined || segmentOrIndex === null) return null
    const target = String(segmentOrIndex)

    const direct = projectStore.subtitles.find((item) => String(item.segment_id || '') === target)
    if (direct?.segment_id) return direct.segment_id

    const legacyIndex = Number(segmentOrIndex)
    if (Number.isNaN(legacyIndex)) return null
    const fromLegacy = projectStore.subtitles.find((item) => Number(item.sentenceIndex) === legacyIndex)
    return fromLegacy?.segment_id || null
  }

  async function loadQueue(identityId) {
    if (!identityId) return new Map()
    try {
      const saved = await localforage.getItem(getQueueKey(identityId))
      if (!saved || typeof saved !== 'object') return new Map()
      const entries = Object.entries(saved)
        .map(([key, value]) => [Number(key), value])
        .filter(([key]) => Number.isFinite(key))
      return new Map(entries)
    } catch (error) {
      console.warn('[SubtitleDocumentStore] 读取本地同步队列失败:', error)
      return new Map()
    }
  }

  async function saveQueue(identityId, queue) {
    if (!identityId) return
    try {
      const payload = {}
      queue.forEach((value, key) => {
        const rawValue = toRaw(value)
        if (!rawValue || typeof rawValue !== 'object') {
          payload[String(key)] = rawValue
          return
        }

        const sanitized = {}
        if (Object.prototype.hasOwnProperty.call(rawValue, 'text')) {
          sanitized.text = rawValue.text
        }
        if (Object.prototype.hasOwnProperty.call(rawValue, 'start')) {
          sanitized.start = rawValue.start
        }
        if (Object.prototype.hasOwnProperty.call(rawValue, 'end')) {
          sanitized.end = rawValue.end
        }
        payload[String(key)] = sanitized
      })
      await localforage.setItem(getQueueKey(identityId), payload)
    } catch (error) {
      console.warn('[SubtitleDocumentStore] 保存本地同步队列失败:', error)
    }
  }

  async function bindSyncIdentity(identityId) {
    const resolvedIdentityId = getActiveIdentity(identityId)
    if (!resolvedIdentityId) {
      currentSyncIdentityId.value = null
      pendingUpdates.value = new Map()
      syncErrors.value.clear()
      return
    }
    if (currentSyncIdentityId.value === resolvedIdentityId) {
      return
    }

    await migrateSubtitleSyncQueue(resolvedIdentityId)
    pendingUpdates.value = await loadQueue(resolvedIdentityId)
    currentSyncIdentityId.value = resolvedIdentityId
  }

  async function restoreFromServer(segments = [], metadata = {}) {
    projectStore.loadFromProjectData(segments, metadata)
  }

  async function applyLocalEditQueue(identityId = null) {
    const resolvedIdentityId = getActiveIdentity(identityId)
    if (!resolvedIdentityId) return 0

    await bindSyncIdentity(resolvedIdentityId)
    return applyPendingEditsToStore(projectStore)
  }

  function applyPendingEditsToStore(targetStore = projectStore) {
    if (!targetStore || pendingUpdates.value.size === 0) return 0

    let appliedCount = 0
    pendingUpdates.value.forEach((update, index) => {
      const subtitle = targetStore.subtitles.find(
        (item) => item.sentenceIndex === index || String(item.segment_id ?? '') === String(index)
      )
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

  async function processQueue() {
    if (pendingUpdates.value.size === 0) return
    const identityId = getActiveIdentity()
    if (!identityId) return

    isSyncing.value = true
    const batch = new Map(pendingUpdates.value)
    pendingUpdates.value.clear()

    for (const [index, data] of batch.entries()) {
      try {
        if (hasProjectContext()) {
          const segmentId = resolveProjectSegmentId(index)
          if (!segmentId) {
            throw new Error(`segment_id 不存在: ${index}`)
          }
          await projectApi.updateSubtitle(projectStore.meta.projectId, segmentId, data)
        } else if (projectStore.meta.jobId) {
          await transcriptionApi.updateSubtitle(projectStore.meta.jobId, index, data)
        } else {
          throw new Error('缺少可用的字幕同步上下文')
        }

        if (syncErrors.value.has(index)) {
          syncErrors.value.delete(index)
        }
      } catch (error) {
        if (error?.status === 404 || error?.message?.includes('不存在')) {
          syncErrors.value.delete(index)
          continue
        }

        console.error(`[SubtitleDocumentStore] 字幕同步失败: ${index}`, error)
        syncErrors.value.set(index, error?.message || '同步失败')
        if (!pendingUpdates.value.has(index)) {
          pendingUpdates.value.set(index, data)
        }
      }
    }

    await saveQueue(identityId, pendingUpdates.value)
    isSyncing.value = false
  }

  const debouncedProcessQueue = debounce(processQueue, 800)

  function onSubtitleEdit(index, { text, start, end }) {
    const identityId = getActiveIdentity()

    const update = {}
    if (text !== undefined) update.text = text
    if (start !== undefined) update.start = projectStore.toBaseTime(start)
    if (end !== undefined) update.end = projectStore.toBaseTime(end)

    pendingUpdates.value.set(index, update)
    if (identityId) {
      saveQueue(identityId, pendingUpdates.value)
    }
    debouncedProcessQueue()
  }

  async function forceSyncNow() {
    await processQueue()
  }

  function pendingCount() {
    return pendingUpdates.value.size
  }

  async function finalizeDraftSubtitlesOnTerminal(reason = 'terminal_status') {
    const hasDraft = projectStore.subtitles.some((item) => item.isDraft)
    if (!hasDraft) return false
    await projectStore.finalizeDraftSubtitlesOnCancel()
    console.log(`[SubtitleDocumentStore] 终态草稿收敛已执行: ${reason}`)
    return true
  }

  function bindTask(jobId) {
    activeJobId.value = jobId || null
  }

  function clearTerminalConverged(jobId = null) {
    if (!jobId) {
      convergedTerminalStatus.value.clear()
      return
    }
    convergedTerminalStatus.value.delete(jobId)
  }

  watch(
    activeTaskStatus,
    async (status) => {
      const jobId = activeJobId.value
      if (!jobId) return

      if (!TERMINAL_STATUSES.has(status)) {
        convergedTerminalStatus.value.delete(jobId)
        return
      }

      const lastStatus = convergedTerminalStatus.value.get(jobId)
      if (lastStatus === status) return

      await finalizeDraftSubtitlesOnTerminal(`watch:${status}`)
      convergedTerminalStatus.value.set(jobId, status)
    },
    { flush: 'post' }
  )

  function setTimeOffset(offset) {
    projectStore.setSubtitleOffset(offset)
  }

  function setSelectedSubtitleId(subtitleId) {
    projectStore.setSelectedSubtitleId(subtitleId)
  }

  function registerBeforeUnloadIfNeeded() {
    if (beforeUnloadRegistered || typeof window === 'undefined') return
    const handleBeforeUnload = () => {
      if (pendingUpdates.value.size > 0) {
        processQueue()
      }
    }
    window.addEventListener('beforeunload', handleBeforeUnload)
    beforeUnloadRegistered = true
  }

  registerBeforeUnloadIfNeeded()

  return {
    activeJobId,
    subtitles,
    timeOffsetSec,
    selectedSubtitleId,
    isSyncing,
    syncErrors,
    bindTask,
    bindSyncIdentity,
    restoreFromServer,
    applyLocalEditQueue,
    applyPendingEditsToStore,
    onSubtitleEdit,
    forceSyncNow,
    pendingCount,
    finalizeDraftSubtitlesOnTerminal,
    clearTerminalConverged,
    setTimeOffset,
    setSelectedSubtitleId,
  }
})
