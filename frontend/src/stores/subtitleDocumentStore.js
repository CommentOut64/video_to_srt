import { defineStore, storeToRefs } from 'pinia'
import { computed, ref, watch } from 'vue'
import { useProjectStore } from './projectStore'
import { useTaskRuntimeStore } from './taskRuntimeStore'
import { useSyncCoordinatorStore } from '@/core/sync/syncCoordinator'

const TERMINAL_STATUSES = new Set(['finished', 'canceled', 'force_canceled', 'removed', 'failed'])

export const useSubtitleDocumentStore = defineStore('subtitleDocument', () => {
  const projectStore = useProjectStore()
  const taskRuntimeStore = useTaskRuntimeStore()
  const syncCoordinator = useSyncCoordinatorStore()
  const { editSyncErrors, isEditSyncing, pendingCount } = storeToRefs(syncCoordinator)

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

  async function bindSyncIdentity(identityId) {
    await syncCoordinator.bindSyncIdentity(identityId)
  }

  async function restoreFromServer(segments = [], metadata = {}) {
    projectStore.loadFromProjectData(segments, metadata)
  }

  async function applyLocalEditQueue(identityId = null) {
    await bindSyncIdentity(identityId)
    return syncCoordinator.applyPendingEditsToStore(projectStore)
  }

  function applyPendingEditsToStore(targetStore = projectStore) {
    return syncCoordinator.applyPendingEditsToStore(targetStore)
  }

  function onSubtitleEdit(index, payload) {
    syncCoordinator.enqueueSubtitleEdit(index, payload)
  }

  async function forceSyncNow() {
    await syncCoordinator.forceSyncNow()
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

  function setTimeOffset(offset) {
    projectStore.setSubtitleOffset(offset)
  }

  function setSelectedSubtitleId(subtitleId) {
    projectStore.setSelectedSubtitleId(subtitleId)
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

  return {
    activeJobId,
    subtitles,
    timeOffsetSec,
    selectedSubtitleId,
    isSyncing: isEditSyncing,
    syncErrors: editSyncErrors,
    bindTask,
    bindSyncIdentity,
    restoreFromServer,
    applyLocalEditQueue,
    applyPendingEditsToStore,
    onSubtitleEdit,
    forceSyncNow,
    pendingCount: () => pendingCount.value,
    finalizeDraftSubtitlesOnTerminal,
    clearTerminalConverged,
    setTimeOffset,
    setSelectedSubtitleId,
  }
})