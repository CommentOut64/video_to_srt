import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { useProjectStore } from './projectStore'
import { useSubtitleDocumentStore } from './subtitleDocumentStore'
import { usePlaybackStore } from './playbackStore'

export const useEditorSessionStore = defineStore('editorSession', () => {
  const projectStore = useProjectStore()
  const subtitleDocumentStore = useSubtitleDocumentStore()
  const playbackStore = usePlaybackStore()

  const currentIdentity = ref({
    projectId: null,
    jobId: null,
  })

  const meta = computed(() => projectStore.meta)
  const capabilitySnapshot = computed(() => projectStore.meta.capabilitySnapshot || null)

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

  async function restoreSession({ identity = {}, metaPayload = {}, segments = [] } = {}) {
    setIdentity(identity)
    restoreMeta(metaPayload)
    subtitleDocumentStore.bindTask(identity.jobId || null)
    await subtitleDocumentStore.restoreFromServer(segments, {
      ...metaPayload,
      ...identity,
    })
    await subtitleDocumentStore.applyLocalEditQueue(identity.projectId || identity.jobId || null)
    playbackStore.reset()
  }

  return {
    currentIdentity,
    meta,
    capabilitySnapshot,
    setIdentity,
    restoreMeta,
    restoreSession,
  }
})
