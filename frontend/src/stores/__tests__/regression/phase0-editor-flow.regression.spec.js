import { beforeEach, describe, expect, it } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { PHASE0_FIXTURES } from './fixtures'

describe('Phase 0 回归 - 编辑链路', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('覆盖 编辑 -> 导出 -> 刷新恢复，并保留 capabilitySnapshot 占位字段', async () => {
    const { useProjectStore } = await import('@/stores/projectStore')
    const store = useProjectStore()
    const { jobId, segments, editedText, capabilitySnapshot } = PHASE0_FIXTURES
    const projectId = 'phase0-project-001'

    store.importSegments(segments, {
      jobId,
      projectId,
      filename: 'phase0.srt',
      title: 'Phase0 编辑回归',
      capabilitySnapshot,
    })

    const firstSubtitle = store.subtitles[0]
    store.updateSubtitle(
      firstSubtitle.id,
      { text: editedText },
      { isUserEdit: true }
    )
    expect(store.subtitles[0].text).toBe(editedText)
    expect(store.subtitles[0].isModified).toBe(true)

    const exportedSrt = store.generateSRT()
    expect(exportedSrt).toContain(editedText)

    await store.saveProject()

    setActivePinia(createPinia())
    const restoredStore = useProjectStore()
    const restored = await restoredStore.restoreProject(projectId)
    expect(restored).toBe(true)
    expect(restoredStore.subtitles[0].text).toBe(editedText)
    expect(restoredStore.meta.capabilitySnapshot).toEqual(capabilitySnapshot)
  })
})
