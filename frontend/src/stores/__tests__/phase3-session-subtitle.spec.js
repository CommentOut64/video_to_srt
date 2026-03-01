import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useProjectStore } from '@/stores/projectStore'
import { useTaskRuntimeStore } from '@/stores/taskRuntimeStore'
import { usePlaybackStore } from '@/stores/playbackStore'
import { useEditorSessionStore } from '@/stores/editorSessionStore'
import { useSubtitleDocumentStore } from '@/stores/subtitleDocumentStore'

vi.mock('localforage', () => ({
  default: {
    getItem: vi.fn(async () => null),
    setItem: vi.fn(async () => true),
  },
}))

describe('Phase 3 会话/字幕拆分', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('restoreSession 按顺序调用恢复链路', async () => {
    const editorSessionStore = useEditorSessionStore()
    const subtitleDocumentStore = useSubtitleDocumentStore()
    const playbackStore = usePlaybackStore()
    const projectStore = useProjectStore()

    const restoreFromServerSpy = vi
      .spyOn(subtitleDocumentStore, 'restoreFromServer')
      .mockResolvedValue(undefined)
    const applyLocalQueueSpy = vi
      .spyOn(subtitleDocumentStore, 'applyLocalEditQueue')
      .mockResolvedValue(0)
    const playbackResetSpy = vi.spyOn(playbackStore, 'reset')

    await editorSessionStore.restoreSession({
      identity: {
        projectId: 'phase3-project-001',
        jobId: 'phase3-job-001',
        mode: 'legacy',
        flavor: 'full',
      },
      metaPayload: {
        title: 'Phase3 Demo',
        filename: 'phase3-demo.mp4',
        videoPath: '/api/media/phase3-project-001/video',
        audioPath: '/api/media/phase3-project-001/audio',
      },
      segments: [{ index: 0, start: 0, end: 1.2, text: 'hello' }],
    })

    expect(projectStore.meta.projectId).toBe('phase3-project-001')
    expect(projectStore.meta.jobId).toBe('phase3-job-001')
    expect(projectStore.meta.title).toBe('Phase3 Demo')
    expect(restoreFromServerSpy).toHaveBeenCalledOnce()
    expect(applyLocalQueueSpy).toHaveBeenCalledOnce()
    expect(playbackResetSpy).toHaveBeenCalledOnce()
    expect(restoreFromServerSpy.mock.invocationCallOrder[0]).toBeLessThan(
      applyLocalQueueSpy.mock.invocationCallOrder[0]
    )
    expect(applyLocalQueueSpy.mock.invocationCallOrder[0]).toBeLessThan(
      playbackResetSpy.mock.invocationCallOrder[0]
    )
  })

  it('任务进入终态后触发草稿收敛', async () => {
    const subtitleDocumentStore = useSubtitleDocumentStore()
    const taskRuntimeStore = useTaskRuntimeStore()
    const projectStore = useProjectStore()
    const jobId = 'phase3-job-terminal-001'
    const now = Date.now()

    taskRuntimeStore.addTask({
      job_id: jobId,
      filename: 'phase3-terminal.mp4',
      status: 'processing',
      progress: 42,
      state_seq: 10,
      updated_at: now,
    })
    projectStore.importSegments(
      [{ index: 0, start: 0.1, end: 0.9, text: 'draft', is_draft: true }],
      { jobId }
    )
    subtitleDocumentStore.bindTask(jobId)

    taskRuntimeStore.updateTaskStatus(jobId, 'canceled', null, {
      state_seq: 11,
      updated_at: now + 10,
      isServer: true,
    })

    await Promise.resolve()
    await Promise.resolve()

    const hasDraft = projectStore.subtitles.some((item) => item.isDraft)
    expect(hasDraft).toBe(false)
  })
})
