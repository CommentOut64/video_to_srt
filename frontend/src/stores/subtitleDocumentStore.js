import { defineStore } from 'pinia'
import { computed, ref, toRaw, watch } from 'vue'
import localforage from 'localforage'
import { legacyApi } from '@/services/api'
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

function normalizeQueueKey(rawKey) {
  const normalized = String(rawKey ?? '').trim()
  if (!normalized) return null
  if (/^-?\d+$/.test(normalized)) {
    const numericKey = Number(normalized)
    if (Number.isFinite(numericKey)) {
      return numericKey
    }
  }
  return normalized
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
  let inflightProcessQueuePromise = null

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

  async function ensureProjectContext() {
    if (projectStore.meta.projectId) {
      return String(projectStore.meta.projectId)
    }
    const fallbackJobId = String(projectStore.meta.jobId || activeJobId.value || '').trim()
    if (!fallbackJobId) {
      throw new Error('缺少 project_id，且无可转换的 job_id')
    }
    const resolved = await legacyApi.resolveTask(fallbackJobId)
    const projectId = String(resolved?.project_id || '').trim()
    if (!projectId) {
      throw new Error(`job_id 转 project_id 失败（job_id=${fallbackJobId}）`)
    }
    projectStore.setIdentity({ projectId })
    return projectId
  }

  function findSubtitleByQueueKey(queueKey) {
    if (queueKey === undefined || queueKey === null) return null
    const target = String(queueKey)
    const bySegmentId = projectStore.subtitles.find(
      (item) => String(item.segment_id || '') === target
    )
    if (bySegmentId) return bySegmentId

    const numericKey = Number(queueKey)
    if (Number.isFinite(numericKey)) {
      const bySentenceIndex = projectStore.subtitles.find(
        (item) => Number(item.sentenceIndex) === numericKey
      )
      if (bySentenceIndex) return bySentenceIndex
    }
    const byLocalId = projectStore.subtitles.find(
      (item) => String(item.id || '') === target
    )
    if (byLocalId) return byLocalId
    return null
  }

  function resolveProjectSegmentId(segmentOrIndex) {
    const subtitle = findSubtitleByQueueKey(segmentOrIndex)
    return subtitle?.segment_id || null
  }

  function resolveLegacySentenceIndex(queueKey) {
    const numericKey = Number(queueKey)
    if (Number.isFinite(numericKey)) return numericKey
    const subtitle = findSubtitleByQueueKey(queueKey)
    const sentenceIndex = Number(subtitle?.sentenceIndex)
    return Number.isFinite(sentenceIndex) ? sentenceIndex : null
  }

  function patchLocalSegmentId(queueKey, segmentId) {
    if (!segmentId) return false
    const subtitle = findSubtitleByQueueKey(queueKey)
    if (!subtitle || subtitle.segment_id === segmentId) return false
    projectStore.updateSubtitle(subtitle.id, { segment_id: String(segmentId) })
    return true
  }

  async function resolveProjectSegmentIdFromServer(queueKey, cache) {
    const sentenceIndex = resolveLegacySentenceIndex(queueKey)
    if (!Number.isFinite(sentenceIndex) || !projectStore.meta.projectId) {
      return null
    }
    if (!cache.snapshot) {
      cache.snapshot = await projectApi.getSubtitles(projectStore.meta.projectId)
    }
    const segments = Array.isArray(cache.snapshot) ? cache.snapshot : []
    const matched = segments.find((segment) => {
      const legacyIndexRaw = segment?.legacy_index ?? segment?.sentence_index
      return Number(legacyIndexRaw) === sentenceIndex
    })
    const segmentId = matched?.segment_id ? String(matched.segment_id) : null
    if (segmentId) {
      patchLocalSegmentId(queueKey, segmentId)
    }
    return segmentId
  }

  async function loadQueue(identityId) {
    if (!identityId) return new Map()
    try {
      const saved = await localforage.getItem(getQueueKey(identityId))
      if (!saved || typeof saved !== 'object') return new Map()
      const entries = Object.entries(saved)
        .map(([key, value]) => [normalizeQueueKey(key), value])
        .filter(([key]) => key !== null)
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
    syncErrors.value.clear()
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
        (item) => item.sentenceIndex === index
          || String(item.segment_id ?? '') === String(index)
          || String(item.id ?? '') === String(index)
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

  async function runProcessQueuePass() {
    if (pendingUpdates.value.size === 0) return
    const identityId = getActiveIdentity()
    if (!identityId) return

    isSyncing.value = true
    try {
      const projectId = await ensureProjectContext()
      const batch = new Map(pendingUpdates.value)
      pendingUpdates.value.clear()
      const projectSegmentLookupCache = { snapshot: null }

      for (const [queueKey, data] of batch.entries()) {
        try {
          let segmentId = resolveProjectSegmentId(queueKey)
          if (!segmentId) {
            segmentId = await resolveProjectSegmentIdFromServer(queueKey, projectSegmentLookupCache)
          }
          if (!segmentId) {
            const staleSubtitle = findSubtitleByQueueKey(queueKey)
            if (!staleSubtitle) {
              // 队列里残留了已不存在的字幕键，直接丢弃避免阻塞后续导出。
              syncErrors.value.delete(queueKey)
              continue
            }
            const isWaitingSegmentBinding = !staleSubtitle.segment_id
              && (staleSubtitle.sentenceIndex === undefined || staleSubtitle.sentenceIndex === null)
            if (isWaitingSegmentBinding) {
              // 本地新增字幕尚未拿到 segment_id：保留队列等待结构性创建落地后再同步。
              if (!pendingUpdates.value.has(queueKey)) {
                pendingUpdates.value.set(queueKey, data)
              }
              syncErrors.value.delete(queueKey)
              continue
            }
            throw new Error(`无法解析 segment_id，同步键=${queueKey}`)
          }
          await projectApi.updateSubtitle(projectId, segmentId, data)

          if (syncErrors.value.has(queueKey)) {
            syncErrors.value.delete(queueKey)
          }
        } catch (error) {
          if (error?.status === 404) {
            syncErrors.value.delete(queueKey)
            continue
          }

          console.error(
            `[SubtitleDocumentStore] 字幕同步失败: key=${queueKey}, identity=${identityId}, project=${projectStore.meta.projectId}, job=${projectStore.meta.jobId}`,
            error
          )
          syncErrors.value.set(queueKey, error?.message || '同步失败')
          if (!pendingUpdates.value.has(queueKey)) {
            pendingUpdates.value.set(queueKey, data)
          }
        }
      }

      await saveQueue(identityId, pendingUpdates.value)
    } finally {
      isSyncing.value = false
    }
  }

  async function processQueue() {
    // 已有同步在飞行时，先等待，避免 forceSyncNow 提前返回。
    if (inflightProcessQueuePromise) {
      await inflightProcessQueuePromise
    }
    if (pendingUpdates.value.size === 0) return

    const currentPassPromise = runProcessQueuePass()
    inflightProcessQueuePromise = currentPassPromise
    try {
      await currentPassPromise
    } finally {
      if (inflightProcessQueuePromise === currentPassPromise) {
        inflightProcessQueuePromise = null
      }
    }
  }

  const debouncedProcessQueue = debounce(processQueue, 800)

  function onSubtitleEdit(index, { text, start, end }) {
    const identityId = getActiveIdentity()
    let queueKey = normalizeQueueKey(index)
    if (queueKey === null) return
    if (hasProjectContext()) {
      const segmentId = resolveProjectSegmentId(queueKey)
      if (segmentId) {
        queueKey = segmentId
      }
    }

    const update = {}
    if (text !== undefined) update.text = text
    if (start !== undefined) update.start = projectStore.toBaseTime(start)
    if (end !== undefined) update.end = projectStore.toBaseTime(end)

    pendingUpdates.value.set(queueKey, update)
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
