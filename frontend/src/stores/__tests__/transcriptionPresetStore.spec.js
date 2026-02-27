import { beforeEach, describe, expect, it } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useTranscriptionPresetStore } from '@/stores/transcriptionPresetStore'
import { useTranscriptionConfigStore } from '@/stores/transcriptionConfigStore'

describe('TranscriptionPresetStore Phase 4', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    globalThis.window.localStorage = globalThis.localStorage
  })

  it('新入口可从本地存储恢复并与默认值合并', () => {
    localStorage.setItem(
      'transcription-task-config',
      JSON.stringify({
        preset_id: 'quality',
        transcription: {
          patching_threshold: 0.82
        }
      })
    )

    const presetStore = useTranscriptionPresetStore()

    expect(presetStore.$id).toBe('transcriptionPreset')
    expect(presetStore.taskConfig.preset_id).toBe('quality')
    expect(presetStore.taskConfig.transcription.patching_threshold).toBe(0.82)
    expect(presetStore.taskConfig.preprocessing.demucs_strategy).toBe('auto')
  })

  it('兼容入口与新入口共享同一真源', () => {
    const presetStore = useTranscriptionPresetStore()
    const legacyStore = useTranscriptionConfigStore()

    legacyStore.applyTaskConfig({
      preset_id: 'fast',
      compute: {
        gpu_id: 1
      }
    })

    expect(legacyStore.$id).toBe('transcriptionPreset')
    expect(presetStore.$id).toBe('transcriptionPreset')
    expect(presetStore.taskConfig.preset_id).toBe('fast')
    expect(presetStore.taskConfig.compute.gpu_id).toBe(1)
  })
})
