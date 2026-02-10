/**
 * 转录配置 Store
 *
 * 统一管理编辑器与任务列表的转录配置来源，避免默认值覆盖用户选择。
 * 通过 localStorage 持久化，保证刷新后仍保持最后一次选择。
 */
import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

const STORAGE_KEY = 'transcription-task-config'

function canUseStorage() {
  return typeof window !== 'undefined' && !!window.localStorage
}

function buildDefaultTaskConfig() {
  return {
    preset_id: 'balanced',
    preprocessing: {
      demucs_strategy: 'auto',
      demucs_model: 'htdemucs',
      demucs_shifts: 1,
      separation_mode: 'on_demand',
      spectrum_threshold: 0.35,
      vad_filter: true,
      enable_spectral_triage: true,
      language_detection_mode: 'balanced',
      language_detection_device: 'auto',
      enable_speaker_embedding: false,
      langid_confidence_threshold: 0.7,
      langid_whitelist: ['zh', 'ja', 'en'],
      langid_logit_bias_score: 2.5
    },
    transcription: {
      transcription_profile: 'sv_whisper_patch',
      sensevoice_device: 'auto',
      whisper_model: 'medium',
      patching_threshold: 0.6
    },
    refinement: {
      llm_task: 'proofread',
      llm_scope: 'sparse',
      sparse_threshold: 0.7,
      target_language: 'zh',
      llm_provider: 'openai_compatible',
      llm_model_name: 'gpt-4o-mini'
    },
    compute: {
      concurrency_strategy: 'auto',
      gpu_id: 0,
      output_formats: ['srt'],
      temp_file_policy: 'delete_on_complete'
    }
  }
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value))
}

function mergeValue(baseValue, patchValue) {
  if (patchValue === undefined) return baseValue
  if (patchValue === null) return baseValue
  if (Array.isArray(patchValue)) return [...patchValue]
  if (patchValue && typeof patchValue === 'object') {
    const baseObject =
      baseValue && typeof baseValue === 'object' ? baseValue : {}
    const result = { ...baseObject }
    Object.keys(patchValue).forEach((key) => {
      result[key] = mergeValue(baseObject[key], patchValue[key])
    })
    return result
  }
  return patchValue
}

function mergeTaskConfig(baseConfig, patchConfig) {
  const result = deepClone(baseConfig)
  if (!patchConfig || typeof patchConfig !== 'object') return result
  Object.keys(patchConfig).forEach((key) => {
    result[key] = mergeValue(result[key], patchConfig[key])
  })
  return result
}

export const useTranscriptionConfigStore = defineStore(
  'transcriptionConfig',
  () => {
    const taskConfig = ref(buildDefaultTaskConfig())
    const hasLoaded = ref(false)

    function loadFromStorage() {
      if (!canUseStorage()) {
        hasLoaded.value = true
        return
      }
      try {
        const raw = localStorage.getItem(STORAGE_KEY)
        if (!raw) {
          hasLoaded.value = true
          return
        }
        const parsed = JSON.parse(raw)
        taskConfig.value = mergeTaskConfig(buildDefaultTaskConfig(), parsed)
        hasLoaded.value = true
      } catch (error) {
        console.warn('读取转录配置失败，使用默认值:', error)
        taskConfig.value = buildDefaultTaskConfig()
        hasLoaded.value = true
      }
    }

    function saveToStorage() {
      if (!canUseStorage()) return
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(taskConfig.value))
      } catch (error) {
        console.warn('保存转录配置失败:', error)
      }
    }

    function applyTaskConfig(newConfig) {
      taskConfig.value = mergeTaskConfig(buildDefaultTaskConfig(), newConfig)
      saveToStorage()
    }

    function resetToDefault() {
      taskConfig.value = buildDefaultTaskConfig()
      saveToStorage()
    }

    watch(
      taskConfig,
      () => {
        if (hasLoaded.value) {
          saveToStorage()
        }
      },
      { deep: true }
    )

    loadFromStorage()

    return {
      taskConfig,
      applyTaskConfig,
      resetToDefault
    }
  }
)
