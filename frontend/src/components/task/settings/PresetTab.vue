<template>
  <div class="preset-tab">
    <div class="preset-list">
      <!-- 内置预设 -->
      <div
        v-for="preset in builtinPresets"
        :key="preset.id"
        class="preset-row"
        :class="{
          active: currentPresetId === preset.id,
          disabled: !isPresetAvailable(preset),
        }"
        :title="getPresetTooltip(preset)"
        @click="isPresetAvailable(preset) && selectPreset(preset.id)"
      >
        <div class="row-indicator" />
        <div class="row-main">
          <span class="preset-name">{{ preset.name }}</span>
          <span class="preset-desc">{{ preset.description }}</span>
        </div>
      </div>

      <!-- 自定义预设 -->
      <div
        v-for="preset in customPresets"
        :key="preset.id"
        class="preset-row custom"
        :class="{ active: currentPresetId === preset.id }"
        @click="selectCustomPreset(preset)"
      >
        <div class="row-indicator" />
        <div class="row-main">
          <span class="preset-name">{{ preset.name }}</span>
        </div>
        <div class="row-actions" @click.stop>
          <!-- 覆盖更新：用当前配置覆写此预设 -->
          <button
            class="action-icon"
            title="用当前配置覆盖此预设"
            @click="emit('overwrite-preset', preset.id)"
          >
            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04a1 1 0 000-1.41l-2.34-2.34a1 1 0 00-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/></svg>
          </button>
          <!-- 删除：二次点击确认 -->
          <button
            class="action-icon"
            :class="{ confirming: pendingDeleteId === preset.id }"
            :title="pendingDeleteId === preset.id ? '再次点击确认删除' : '删除预设'"
            @click="handleDelete(preset.id)"
          >
            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, onMounted, onBeforeUnmount } from 'vue'
import { getHardwareInfo } from '@/services/api/systemApi'

const props = defineProps({
  modelValue: {
    type: Object,
    required: true,
  },
  customPresets: {
    type: Array,
    default: () => [],
  },
})

const emit = defineEmits(['update:modelValue', 'delete-preset', 'overwrite-preset'])

/* 硬件状态 */
const hardwareLoaded = ref(false)
const vramMB = ref(0)
const hasGpu = ref(true)

/* 删除二次确认 */
const pendingDeleteId = ref(null)
let deleteTimer = null

function handleDelete(presetId) {
  if (pendingDeleteId.value === presetId) {
    /* 第二次点击：确认删除 */
    pendingDeleteId.value = null
    clearTimeout(deleteTimer)
    emit('delete-preset', presetId)
  } else {
    /* 第一次点击：进入确认态 */
    pendingDeleteId.value = presetId
    clearTimeout(deleteTimer)
    deleteTimer = setTimeout(() => {
      pendingDeleteId.value = null
    }, 3000)
  }
}

onBeforeUnmount(() => {
  clearTimeout(deleteTimer)
})

/* 内置预设定义 */
const builtinPresets = [
  {
    id: 'fast',
    name: '极速预览',
    description: '会议记录、快速浏览',
    minVram: 1500,
    requiresGpu: false,
    config: {
      preprocessing: {
        demucs_strategy: 'off',
        separation_mode: 'on_demand',
        enable_spectral_triage: false,
        language_detection_mode: 'balanced',
      },
      transcription: { transcription_profile: 'sensevoice_only' },
      refinement: { llm_task: 'off', llm_scope: 'sparse' },
    },
  },
  {
    id: 'balanced',
    name: '智能均衡',
    description: '短视频、Vlog',
    minVram: 4000,
    requiresGpu: true,
    config: {
      preprocessing: {
        demucs_strategy: 'auto',
        separation_mode: 'on_demand',
        enable_spectral_triage: true,
        language_detection_mode: 'balanced',
      },
      transcription: { transcription_profile: 'sv_whisper_patch' },
      refinement: { llm_task: 'proofread', llm_scope: 'sparse' },
    },
  },
  {
    id: 'quality',
    name: '影视精修',
    description: '电影压制、高精度',
    minVram: 8000,
    requiresGpu: true,
    config: {
      preprocessing: {
        demucs_strategy: 'force_on',
        separation_mode: 'global',
        enable_spectral_triage: false,
        language_detection_mode: 'balanced',
      },
      transcription: { transcription_profile: 'sv_whisper_dual' },
      refinement: { llm_task: 'proofread', llm_scope: 'global' },
    },
  },
]

/* 当前预设 ID */
const currentPresetId = computed(() => props.modelValue.preset_id)

/* 预设可用性检查 */
function isPresetAvailable(preset) {
  if (preset.requiresGpu && !hasGpu.value) return false
  if (vramMB.value > 0 && vramMB.value < preset.minVram) return false
  return true
}

function getPresetTooltip(preset) {
  if (!isPresetAvailable(preset)) {
    if (preset.requiresGpu && !hasGpu.value) return '需要 GPU'
    return `需要 ${Math.ceil(preset.minVram / 1024)}GB 显存`
  }
  return ''
}

/* 选择内置预设 */
function selectPreset(presetId) {
  const preset = builtinPresets.find((p) => p.id === presetId)
  if (!preset) return

  const updated = JSON.parse(JSON.stringify(props.modelValue))
  updated.preset_id = presetId

  Object.assign(updated.preprocessing, preset.config.preprocessing)
  Object.assign(updated.transcription, preset.config.transcription)
  Object.assign(updated.refinement, preset.config.refinement)

  if (presetId === 'fast') {
    updated.preprocessing.demucs_model = 'htdemucs'
    updated.preprocessing.demucs_shifts = 1
    updated.transcription.whisper_model = 'medium'
  } else if (presetId === 'balanced') {
    updated.preprocessing.demucs_model = 'htdemucs'
    updated.preprocessing.demucs_shifts = 1
    updated.transcription.whisper_model = 'medium'
  } else if (presetId === 'quality') {
    updated.preprocessing.demucs_model = 'mdx_extra'
    updated.preprocessing.demucs_shifts = 2
    updated.transcription.whisper_model = 'large-v3'
  }

  emit('update:modelValue', updated)
}

/* 选择自定义预设 */
function selectCustomPreset(preset) {
  const updated = JSON.parse(JSON.stringify(props.modelValue))
  updated.preset_id = preset.id

  if (preset.config) {
    if (preset.config.preprocessing) Object.assign(updated.preprocessing, preset.config.preprocessing)
    if (preset.config.transcription) Object.assign(updated.transcription, preset.config.transcription)
    if (preset.config.refinement) Object.assign(updated.refinement, preset.config.refinement)
    if (preset.config.compute) Object.assign(updated.compute, preset.config.compute)
  }

  emit('update:modelValue', updated)
}

/* 加载硬件信息 */
onMounted(async () => {
  try {
    const response = await getHardwareInfo()
    if (response.success && response.hardware) {
      const gpuInfo = response.hardware.gpu
      if (gpuInfo) {
        hasGpu.value = gpuInfo.cuda_available === true
        if (gpuInfo.total_memory_mb) {
          vramMB.value = gpuInfo.total_memory_mb
        }
      } else {
        hasGpu.value = false
      }

      hardwareLoaded.value = true
    }
  } catch (error) {
    console.warn('获取硬件信息失败:', error)
    hardwareLoaded.value = false
  }
})
</script>

<style scoped>
.preset-tab {
  display: flex;
  flex-direction: column;
}

.preset-list {
  display: flex;
  flex-direction: column;
}

/* --- 预设行 --- */
.preset-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border-radius: 4px;
  transition: background 0.15s ease;
  cursor: pointer;
}

.preset-row:hover {
  background: rgba(255, 255, 255, 0.04);
}

.preset-row.active {
  background: rgba(var(--af-accent-primary-rgb), 0.08);
}

.preset-row.disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.preset-row.disabled:hover {
  background: transparent;
}

/* 左侧激活指示条 */
.row-indicator {
  flex-shrink: 0;
  width: 3px;
  height: 16px;
  border-radius: 2px;
  background: transparent;
  transition: background 0.15s ease;
}

.preset-row.active .row-indicator {
  background: var(--af-accent-primary);
}

/* 主内容区 */
.row-main {
  flex: 1;
  display: flex;
  align-items: baseline;
  gap: 8px;
  min-width: 0;
}

.preset-name {
  color: var(--af-text-normal);
  font-size: 13px;
  font-weight: 500;
  white-space: nowrap;
}

.preset-row.active .preset-name {
  color: var(--af-accent-primary);
}

.preset-desc {
  color: var(--af-text-muted);
  font-size: 11px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* --- 右侧操作图标 --- */
.row-actions {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 2px;
  opacity: 0;
  transition: opacity 0.15s ease;
}

.preset-row:hover .row-actions {
  opacity: 1;
}

.action-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  padding: 0;
  background: none;
  border: none;
  border-radius: 4px;
  color: var(--af-text-muted);
  cursor: pointer;
  transition:
    color 0.15s ease,
    background 0.15s ease;
}

.action-icon svg {
  width: 13px;
  height: 13px;
}

.action-icon:hover {
  background: rgba(255, 255, 255, 0.08);
  color: var(--af-text-primary);
}

/* 删除确认态：变红警示 */
.action-icon.confirming {
  background: var(--af-accent-danger);
  color: var(--af-text-inverse);
}

.action-icon.confirming:hover {
  background: var(--af-accent-danger);
  color: var(--af-text-inverse);
}
</style>
