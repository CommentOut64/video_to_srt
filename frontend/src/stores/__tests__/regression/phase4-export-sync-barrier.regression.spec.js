import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import localforage from 'localforage'
import { useProjectStore } from '@/stores/projectStore'
import { useSubtitleDocumentStore } from '@/stores/subtitleDocumentStore'

const projectApiMocks = vi.hoisted(() => ({
  updateSubtitle: vi.fn(async () => ({})),
  getSubtitles: vi.fn(async () => []),
}))

vi.mock('@/services/api/projectApi', () => ({
  default: {
    updateSubtitle: projectApiMocks.updateSubtitle,
    getSubtitles: projectApiMocks.getSubtitles,
  },
}))

function deepClone(value) {
  return JSON.parse(JSON.stringify(value))
}

function createDeferred() {
  let resolve
  let reject
  const promise = new Promise((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

vi.mock('localforage', () => {
  const memory = new Map()
  return {
    default: {
      async getItem(key) {
        if (!memory.has(key)) return null
        return deepClone(memory.get(key))
      },
      async setItem(key, value) {
        memory.set(String(key), deepClone(value))
        return value
      },
      async removeItem(key) {
        memory.delete(String(key))
      },
      __reset() {
        memory.clear()
      },
    },
  }
})

describe('Phase 4 回归 - 导出前同步栅栏补写', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localforage.__reset()
    projectApiMocks.updateSubtitle.mockClear()
    projectApiMocks.getSubtitles.mockClear()
  })

  it('新增字幕在未分配 segment_id 时编辑，落地后可二次 forceSync 补写', async () => {
    const projectStore = useProjectStore()
    const subtitleDocumentStore = useSubtitleDocumentStore()
    const projectId = 'phase4-export-project-001'

    projectStore.importSegments(
      [{ index: 0, start: 0, end: 1.5, text: '原始字幕' }],
      { projectId, jobId: 'phase4-export-job-001' }
    )
    await subtitleDocumentStore.bindSyncIdentity(projectId)

    const insertIndex = projectStore.subtitles.length
    projectStore.addSubtitle(insertIndex, {
      start: 1.5,
      end: 3.0,
      text: '',
      isModified: true,
      source: 'manual',
    })
    const localSubtitleId = projectStore.subtitles[insertIndex]?.id
    expect(localSubtitleId).toBeTruthy()

    projectStore.updateSubtitle(localSubtitleId, { text: '新增后立即编辑' }, { isUserEdit: true })
    subtitleDocumentStore.onSubtitleEdit(localSubtitleId, { text: '新增后立即编辑' })

    await subtitleDocumentStore.forceSyncNow()
    expect(projectApiMocks.updateSubtitle).not.toHaveBeenCalled()
    expect(subtitleDocumentStore.pendingCount()).toBe(1)
    expect(subtitleDocumentStore.syncErrors.size).toBe(0)

    projectStore.updateSubtitle(
      localSubtitleId,
      {
        sentenceIndex: 88,
        segment_id: 'seg-phase4-001',
      },
      { isUserEdit: false }
    )

    await subtitleDocumentStore.forceSyncNow()
    expect(projectApiMocks.updateSubtitle).toHaveBeenCalledTimes(1)
    expect(projectApiMocks.updateSubtitle).toHaveBeenCalledWith(
      projectId,
      'seg-phase4-001',
      { text: '新增后立即编辑' }
    )
    expect(subtitleDocumentStore.pendingCount()).toBe(0)
    expect(subtitleDocumentStore.syncErrors.size).toBe(0)
  })

  it('forceSyncNow 会等待已在飞行中的 processQueue，同步完成后再返回', async () => {
    const projectStore = useProjectStore()
    const subtitleDocumentStore = useSubtitleDocumentStore()
    const projectId = 'phase4-export-project-002'
    const deferred = createDeferred()

    projectApiMocks.updateSubtitle.mockImplementationOnce(() => deferred.promise)

    projectStore.loadFromProjectData(
      [
        {
          segment_id: 'seg-phase4-002',
          legacy_index: 0,
          start: 0,
          end: 1,
          text: '原始文本',
          source_type: 'transcribe',
        },
      ],
      { projectId, jobId: 'phase4-export-job-002' }
    )
    await subtitleDocumentStore.bindSyncIdentity(projectId)

    subtitleDocumentStore.onSubtitleEdit('seg-phase4-002', { text: '更新文本' })

    const firstSyncPromise = subtitleDocumentStore.forceSyncNow()
    await Promise.resolve()
    const secondSyncPromise = subtitleDocumentStore.forceSyncNow()

    let secondResolved = false
    secondSyncPromise.then(() => {
      secondResolved = true
    })
    await Promise.resolve()
    expect(secondResolved).toBe(false)

    deferred.resolve({})
    await firstSyncPromise
    await secondSyncPromise

    expect(projectApiMocks.updateSubtitle).toHaveBeenCalledTimes(1)
    expect(projectApiMocks.updateSubtitle).toHaveBeenCalledWith(
      projectId,
      'seg-phase4-002',
      { text: '更新文本' }
    )
    expect(subtitleDocumentStore.pendingCount()).toBe(0)
    expect(subtitleDocumentStore.syncErrors.size).toBe(0)
  })
})
