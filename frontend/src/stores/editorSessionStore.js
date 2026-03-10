import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { useProjectStore } from './projectStore'
import { usePlaybackStore } from './playbackStore'
import { useEditorUiStore } from './editorUiStore'
import { useSyncCoordinatorStore } from '@/core/sync/syncCoordinator'

const workingCopyCache = new Map()
const MAX_WORKING_COPY_CACHE = 10

function normalizeIdentityKey(value) {
  const normalized = String(value || '').trim()
  return normalized || null
}

function trimWorkingCopyCache() {
  while (workingCopyCache.size > MAX_WORKING_COPY_CACHE) {
    const oldestKey = workingCopyCache.keys().next().value
    workingCopyCache.delete(oldestKey)
  }
}

export const useEditorSessionStore = defineStore('editorSession', () => {
  const projectStore = useProjectStore()
  const playbackStore = usePlaybackStore()
  const editorUiStore = useEditorUiStore()
  const syncCoordinator = useSyncCoordinatorStore()

  const currentIdentity = ref({
    projectId: null,
    jobId: null,
  })

  const meta = computed(() => projectStore.meta)
  const capabilitySnapshot = computed(() => projectStore.meta.capabilitySnapshot || null)

  function resolveIdentityKey(preferredIdentityId = null) {
    return normalizeIdentityKey(
      preferredIdentityId
      || currentIdentity.value.projectId
      || currentIdentity.value.jobId
      || projectStore.primaryId
    )
  }

  function setIdentity(payload = {}) {
    currentIdentity.value = {
      projectId: payload.projectId ?? currentIdentity.value.projectId ?? null,
      jobId: payload.jobId ?? currentIdentity.value.jobId ?? null,
    }
    projectStore.setIdentity({
      projectId: currentIdentity.value.projectId,
      jobId: currentIdentity.value.jobId,
      mode: payload.mode,
      flavor: payload.flavor,
      capabilitySnapshot: payload.capabilitySnapshot,
    })
  }

  function restoreMeta(payload = {}) {
    setIdentity(payload)
    projectStore.patchMeta({
      filename: payload.filename,
      title: payload.title,
      duration: payload.duration,
    })
    projectStore.setMediaPaths({
      videoPath: payload.videoPath,
      audioPath: payload.audioPath,
      peaksPath: payload.peaksPath,
    })
    if (payload.currentResolution !== undefined) {
      projectStore.setCurrentResolution(payload.currentResolution)
    }
  }

  function captureWorkingCopy(identityId = null) {
    const cacheKey = resolveIdentityKey(identityId)
    if (!cacheKey) {
      return false
    }

    workingCopyCache.set(cacheKey, projectStore.createWorkingCopySnapshot())
    trimWorkingCopyCache()
    return true
  }

  async function restoreWorkingCopy(identityId = null) {
    const cacheKey = resolveIdentityKey(identityId)
    if (!cacheKey || !workingCopyCache.has(cacheKey)) {
      return false
    }

    projectStore.applyWorkingCopySnapshot(workingCopyCache.get(cacheKey))
    editorUiStore.clearSelectedSubtitleId()
    playbackStore.reset()
    console.log(`[EditorSessionStore] 工作副本已恢复: ${cacheKey}`)
    return true
  }

  async function saveWorkingCopy(identityId = null) {
    const captured = captureWorkingCopy(identityId)
    if (!captured) {
      return false
    }

    projectStore.markWorkingCopySaved()
    console.log('[EditorSessionStore] 工作副本已保存')
    return true
  }

  async function restoreSession({ identity = {}, metaPayload = {}, segments = [] } = {}) {
    setIdentity(identity)
    restoreMeta(metaPayload)
    projectStore.loadFromProjectData(segments, {
      ...metaPayload,
      ...identity,
    })
    editorUiStore.clearSelectedSubtitleId()
    await syncCoordinator.bindSyncIdentity(identity.projectId || identity.jobId || null)
    syncCoordinator.applyPendingEditsToStore(projectStore)
    playbackStore.reset()
  }

  return {
    currentIdentity,
    meta,
    capabilitySnapshot,
    setIdentity,
    restoreMeta,
    captureWorkingCopy,
    restoreWorkingCopy,
    saveWorkingCopy,
    restoreSession,
  }
})
