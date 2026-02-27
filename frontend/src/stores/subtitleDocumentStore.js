import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import localforage from 'localforage'
import { useProjectStore } from './projectStore'
import { useTaskRuntimeStore } from './taskRuntimeStore'

const EDIT_QUEUE_PREFIX = 'subtitle-edit-queue-'
const TERMINAL_STATUSES = new Set(['finished', 'canceled', 'force_canceled', 'removed', 'failed'])

function getQueueKey(identityId) {
  return `${EDIT_QUEUE_PREFIX}${identityId}`
}

export const useSubtitleDocumentStore = defineStore('subtitleDocument', () => {
  const projectStore = useProjectStore()
  const taskRuntimeStore = useTaskRuntimeStore()

  const activeJobId = ref(null)
  const convergedTerminalStatus = ref(new Map())

  const subtitles = computed(() => projectStore.subtitles)
  const timeOffsetSec = computed(() => projectStore.subtitleOffset)
  const selectedSubtitleId = computed(() => projectStore.view.selectedSubtitleId)

  const activeTaskStatus = computed(() => {
    if (!activeJobId.value) return 'idle'
    const task = taskRuntimeStore.getTask(activeJobId.value)
    if (task?.status) return task.status
    return taskRuntimeStore.getRawState(activeJobId.value).status || 'idle'
  })

  async function restoreFromServer(segments = [], metadata = {}) {
    projectStore.loadFromProjectData(segments, metadata)
  }

  async function applyLocalEditQueue(identityId = null) {
    const resolvedIdentityId = identityId || activeJobId.value || projectStore.primaryId
    if (!resolvedIdentityId) return 0

    try {
      const saved = await localforage.getItem(getQueueKey(resolvedIdentityId))
      if (!saved || typeof saved !== 'object') {
        return 0
      }

      let appliedCount = 0
      const entries = Object.entries(saved)
      entries.forEach(([rawIndex, update]) => {
        const numericIndex = Number(rawIndex)
        const subtitle = projectStore.subtitles.find(
          (item) =>
            item.sentenceIndex === numericIndex
            || String(item.segment_id ?? '') === String(rawIndex)
        )
        if (!subtitle || !update || typeof update !== 'object') return

        const displayUpdate = { ...update }
        if (update.start !== undefined) {
          displayUpdate.start = projectStore.toDisplayTime(update.start)
        }
        if (update.end !== undefined) {
          displayUpdate.end = projectStore.toDisplayTime(update.end)
        }

        projectStore.updateSubtitle(subtitle.id, displayUpdate, { isUserEdit: true })
        appliedCount += 1
      })

      return appliedCount
    } catch (error) {
      console.warn('[SubtitleDocumentStore] 叠加本地编辑队列失败:', error)
      return 0
    }
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
      if (lastStatus === status) {
        return
      }

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

  return {
    activeJobId,
    subtitles,
    timeOffsetSec,
    selectedSubtitleId,
    bindTask,
    restoreFromServer,
    applyLocalEditQueue,
    finalizeDraftSubtitlesOnTerminal,
    clearTerminalConverged,
    setTimeOffset,
    setSelectedSubtitleId,
  }
})
