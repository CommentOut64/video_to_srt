<template>
  <div class="preset-selector" :class="{ compact: compact }">
    <!-- 顶层: 快捷场景宏 -->
    <div class="macro-presets">
      <!-- <div class="section-header">
        <span class="header-label">快捷场景</span>
        <button
          class="toggle-btn"
          @click="showAdvanced = !showAdvanced"
        >
          {{ showAdvanced ? '收起高级' : '高级设置' }}
        </button>
      </div> -->

      <!-- 硬件信息提示 -->
      <!-- <div v-if="hardwareLoaded" class="hardware-hint">
        <span class="hw-vram">显存: {{ vramGB }}GB</span>
        <span v-if="!hasGpu" class="hw-warning">(无GPU)</span>
        <span v-if="recommendedPresetId" class="hw-recommend">
          推荐: {{ getPresetName(recommendedPresetId) }}
        </span>
      </div> -->

      <!-- 三个快捷预设卡片 -->
      <div class="preset-cards">
        <div
          v-for="preset in macroPresets"
          :key="preset.id"
          class="preset-card"
          :class="{
            active: currentPresetId === preset.id,
            disabled: !isPresetAvailable(preset),
            recommended: preset.id === recommendedPresetId
          }"
          @click="isPresetAvailable(preset) && selectMacroPreset(preset.id)"
          :title="getPresetTooltip(preset)"
        >
          <!-- <div class="preset-icon">
            <component :is="getPresetIcon(preset.icon)" />
          </div> -->
          <div class="preset-info">
            <div class="preset-name">{{ preset.name }}</div>
            <div class="preset-desc">{{ preset.description }}</div>
          </div>
          <div v-if="currentPresetId === preset.id" class="check-mark">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
            </svg>
          </div>
          <div v-if="preset.id === recommendedPresetId && currentPresetId !== preset.id" class="recommend-badge">
            推荐
          </div>
        </div>
      </div>
    </div>

    <!-- 底层: 三个独立模块卡片 -->
    <div class="module-cards">
      <!-- 模块一: 人声分离 -->
      <div class="module-card">
        <div class="module-header">
          <span class="module-icon">1</span>
          <span class="module-title">前处理 (人声分离)</span>
        </div>
        <div class="module-options">
          <label
            v-for="option in demucsOptions"
            :key="option.value"
            class="option-item"
            :class="{
              active: localConfig.preprocessing.demucs_strategy === option.value,
              disabled: option.disabled
            }"
          >
            <input
              type="radio"
              :value="option.value"
              v-model="localConfig.preprocessing.demucs_strategy"
              :disabled="option.disabled"
              @change="onModuleChange"
            />
            <span class="option-label">{{ option.label }}</span>
            <span v-if="option.hint" class="option-hint">{{ option.hint }}</span>
          </label>
        </div>
        <div class="mt-2 flex flex-col gap-2">
          <div class="text-[11px] text-[var(--af-text-muted)]">语言检测</div>
          <label class="flex items-center gap-2 text-[11px]">
            <span class="w-16 text-[var(--af-text-muted)]">模式</span>
            <select
              v-model="localConfig.preprocessing.language_detection_mode"
              class="flex-1 rounded border border-[var(--af-border-default)] bg-[var(--af-bg-tertiary)] px-2 py-1 text-[11px] text-[var(--af-text-normal)]"
              @change="onModuleChange"
            >
              <option value="fast">极速</option>
              <option value="balanced">智能平衡</option>
              <option value="precise">精准</option>
            </select>
          </label>
          <label class="flex items-center gap-2 text-[11px]">
            <input
              type="checkbox"
              v-model="localConfig.preprocessing.enable_speaker_embedding"
              class="h-3 w-3 accent-[var(--af-accent-primary)]"
              @change="onModuleChange"
            />
            <span class="text-[var(--af-text-normal)]">启用声纹提取</span>
          </label>
          <div class="text-[10px] text-[var(--af-text-muted)]">
            声纹数据仅用于后续说话人聚类（默认关闭）
          </div>
        </div>
      </div>

      <!-- 模块二: 转录核心 -->
      <div class="module-card">
        <div class="module-header">
          <span class="module-icon">2</span>
          <span class="module-title">转录核心 (ASR)</span>
        </div>
        <div class="module-options">
          <label
            v-for="option in transcriptionOptions"
            :key="option.value"
            class="option-item"
            :class="{
              active: localConfig.transcription.transcription_profile === option.value,
              disabled: option.disabled
            }"
          >
            <input
              type="radio"
              :value="option.value"
              v-model="localConfig.transcription.transcription_profile"
              :disabled="option.disabled"
              @change="onModuleChange"
            />
            <span class="option-label">{{ option.label }}</span>
            <span v-if="option.hint" class="option-hint">{{ option.hint }}</span>
          </label>
        </div>
      </div>

      <!-- 模块三: 增强 (LLM 暂未集成，整体禁用) -->
      <div class="module-card module-disabled" title="LLM 功能暂未集成，敬请期待">
        <div class="module-header">
          <span class="module-icon">3</span>
          <span class="module-title">增强 (LLM) - 暂未集成</span>
        </div>
        <div class="module-options">
          <label
            v-for="option in llmOptions"
            :key="option.value"
            class="option-item disabled"
          >
            <input
              type="radio"
              :value="option.value"
              :checked="getLLMOptionValue() === option.value"
              disabled
            />
            <span class="option-label">{{ option.label }}</span>
            <span v-if="option.hint" class="option-hint">{{ option.hint }}</span>
          </label>
        </div>
      </div>
    </div>

    <!-- 当前方案说明 -->
    <!-- <div class="current-preset-info">
      <span class="info-label">当前方案:</span>
      <span class="info-value">{{ currentPresetInfo }}</span>
    </div> -->

    <!-- 显存警告 -->
    <!-- <div v-if="vramWarning" class="vram-warning">
      {{ vramWarning }}
    </div> -->
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import { getHardwareInfo } from '@/services/api/systemApi'

const props = defineProps({
  modelValue: {
    type: Object,
    default: () => ({
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
        patching_threshold: 0.60
      },
      refinement: {
        llm_task: 'proofread',
        llm_scope: 'sparse',
        sparse_threshold: 0.70,
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
    })
  },
  /* 紧凑模式 - 用于对话框等空间有限的场景 */
  compact: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['update:modelValue', 'change', 'openAdvanced'])

/* 状态 */
const showAdvanced = ref(false)
const hardwareLoaded = ref(false)
const vramGB = ref(0)
const vramMB = ref(0)
const hasGpu = ref(true)
const recommendedPresetId = ref(null)

/* 本地配置 (深拷贝) */
const localConfig = ref(JSON.parse(JSON.stringify(props.modelValue)))

/* 当前预设 ID */
const currentPresetId = computed(() => localConfig.value.preset_id)

/* 快捷场景宏定义 */
const macroPresets = [
  {
    id: 'fast',
    name: '极速预览',
    description: '会议记录、快速浏览',
    icon: 'bolt',
    minVram: 1500,
    requiresGpu: false,
    config: {
      /* 直通模式: 完全跳过频谱分诊和人声分离 */
      preprocessing: {
        demucs_strategy: 'off',
        separation_mode: 'on_demand',
        enable_spectral_triage: false,
        language_detection_mode: 'balanced',
        enable_speaker_embedding: false
      },
      transcription: { transcription_profile: 'sensevoice_only' },
      refinement: { llm_task: 'off', llm_scope: 'sparse' }
    }
  },
  {
    id: 'balanced',
    name: '智能均衡',
    description: '短视频、Vlog (推荐)',
    icon: 'scale',
    minVram: 4000,
    requiresGpu: true,
    config: {
      /* 智能模式: 启用频谱分诊，按需分离 */
      preprocessing: {
        demucs_strategy: 'auto',
        separation_mode: 'on_demand',
        enable_spectral_triage: true,
        language_detection_mode: 'balanced',
        enable_speaker_embedding: false
      },
      transcription: { transcription_profile: 'sv_whisper_patch' },
      refinement: { llm_task: 'proofread', llm_scope: 'sparse' }
    }
  },
  {
    id: 'quality',
    name: '影视精修',
    description: '电影压制、高精度',
    icon: 'film',
    minVram: 8000,
    requiresGpu: true,
    config: {
      /* 极致模式: 强制全局分离，频谱分诊可跳过（因为会强制分离） */
      preprocessing: {
        demucs_strategy: 'force_on',
        separation_mode: 'global',
        enable_spectral_triage: false,
        language_detection_mode: 'balanced',
        enable_speaker_embedding: false
      },
      transcription: { transcription_profile: 'sv_whisper_dual' },
      refinement: { llm_task: 'proofread', llm_scope: 'global' }
    }
  }
]

/* 模块一: 人声分离选项 */
const demucsOptions = computed(() => [
  {
    value: 'off',
    label: '直通',
    hint: '不分离，最快速度',
    disabled: false
  },
  {
    value: 'auto',
    label: '智能分诊',
    hint: '智能判断',
    disabled: false
  },
  {
    value: 'force_on',
    label: '极致分离',
    hint: '全局分离，最高质量',
    disabled: !hasGpu.value || vramMB.value < 6000
  }
])

/* 模块二: 转录选项 */
const transcriptionOptions = computed(() => [
  {
    value: 'sensevoice_only',
    label: '极速',
    hint: '仅SenseVoice',
    disabled: false
  },
  {
    value: 'sv_whisper_patch',
    label: '智能复核',
    hint: 'SenseVoice+Whisper复核',
    disabled: !hasGpu.value
  },
  {
    value: 'sv_whisper_dual',
    label: '双流精校',
    hint: '最高精度(很慢不推荐)',
    disabled: !hasGpu.value || vramMB.value < 8000
  }
])

/* 模块三: LLM 增强选项 */
const llmOptions = computed(() => [
  {
    value: 'off',
    label: '关闭',
    hint: '原生输出',
    disabled: false
  },
  {
    value: 'proofread_sparse',
    label: '稀疏校对',
    hint: '修正错字',
    disabled: false
  },
  {
    value: 'proofread_global',
    label: '全局校对',
    hint: '全文润色',
    disabled: false
  },
  {
    value: 'translate',
    label: '翻译',
    hint: '含校对',
    disabled: false
  }
])

/* 获取 LLM 选项的组合值 */
function getLLMOptionValue() {
  const { llm_task, llm_scope } = localConfig.value.refinement
  if (llm_task === 'off') return 'off'
  if (llm_task === 'translate') return 'translate'
  if (llm_task === 'proofread') {
    return llm_scope === 'global' ? 'proofread_global' : 'proofread_sparse'
  }
  return 'off'
}

/* 设置 LLM 选项 */
function setLLMOption(value) {
  switch (value) {
    case 'off':
      localConfig.value.refinement.llm_task = 'off'
      localConfig.value.refinement.llm_scope = 'sparse'
      break
    case 'proofread_sparse':
      localConfig.value.refinement.llm_task = 'proofread'
      localConfig.value.refinement.llm_scope = 'sparse'
      break
    case 'proofread_global':
      localConfig.value.refinement.llm_task = 'proofread'
      localConfig.value.refinement.llm_scope = 'global'
      break
    case 'translate':
      localConfig.value.refinement.llm_task = 'translate'
      localConfig.value.refinement.llm_scope = 'global'
      break
  }
  onModuleChange()
}

/* 当前预设信息 */
const currentPresetInfo = computed(() => {
  if (currentPresetId.value === 'custom') {
    return '自定义配置'
  }
  const preset = macroPresets.find(p => p.id === currentPresetId.value)
  return preset ? `${preset.name} - ${preset.description}` : '自定义配置'
})

/* 显存警告 */
const vramWarning = computed(() => {
  if (!hardwareLoaded.value) return null

  const warnings = []

  /* 检查 Demucs mdx_extra */
  if (localConfig.value.preprocessing.demucs_model === 'mdx_extra' && vramMB.value < 6000) {
    warnings.push('mdx_extra 模型建议 6GB+ 显存')
  }

  /* 检查 Whisper large-v3 */
  if (localConfig.value.transcription.whisper_model === 'large-v3' && vramMB.value < 8000) {
    warnings.push('Whisper large-v3 模型建议 8GB+ 显存')
  }

  /* 检查双流并行 */
  if (localConfig.value.transcription.transcription_profile === 'sv_whisper_dual' && vramMB.value < 8000) {
    warnings.push('双流精校模式建议 8GB+ 显存')
  }

  if (warnings.length > 0) {
    return '当前配置可能导致显存不足: ' + warnings.join('; ')
  }

  return null
})

/* 检查预设是否可用 */
function isPresetAvailable(preset) {
  if (preset.requiresGpu && !hasGpu.value) return false
  if (vramMB.value < preset.minVram) return false
  return true
}

/* 获取预设 tooltip */
function getPresetTooltip(preset) {
  if (!isPresetAvailable(preset)) {
    if (preset.requiresGpu && !hasGpu.value) {
      return '需要 GPU'
    }
    return `需要 ${Math.ceil(preset.minVram / 1024)}GB 显存`
  }
  return ''
}

/* 获取预设名称 */
function getPresetName(presetId) {
  const preset = macroPresets.find(p => p.id === presetId)
  return preset?.name || presetId
}

/* 获取预设图标组件 */
function getPresetIcon(iconName) {
  /* 返回简单的文本图标 */
  const icons = {
    bolt: 'span',
    scale: 'span',
    film: 'span'
  }
  return icons[iconName] || 'span'
}

/* 选择快捷预设 */
function selectMacroPreset(presetId) {
  const preset = macroPresets.find(p => p.id === presetId)
  if (!preset) return

  localConfig.value.preset_id = presetId

  /* 应用预设配置 */
  Object.assign(localConfig.value.preprocessing, preset.config.preprocessing)
  Object.assign(localConfig.value.transcription, preset.config.transcription)
  Object.assign(localConfig.value.refinement, preset.config.refinement)

  /* 根据预设设置默认的详细参数 */
  if (presetId === 'fast') {
    localConfig.value.preprocessing.demucs_model = 'htdemucs'
    localConfig.value.preprocessing.demucs_shifts = 1
    localConfig.value.transcription.whisper_model = 'medium'
  } else if (presetId === 'balanced') {
    localConfig.value.preprocessing.demucs_model = 'htdemucs'
    localConfig.value.preprocessing.demucs_shifts = 1
    localConfig.value.transcription.whisper_model = 'medium'
  } else if (presetId === 'quality') {
    localConfig.value.preprocessing.demucs_model = 'mdx_extra'
    localConfig.value.preprocessing.demucs_shifts = 2
    localConfig.value.transcription.whisper_model = 'large-v3'
  }

  emitChange()
}

/* 模块选项变更时检查是否匹配预设 */
function onModuleChange() {
  /* 根据 demucs_strategy 自动推导 enable_spectral_triage 和 separation_mode
   * - off: 直通模式，跳过频谱分诊，按需分离（实际不会执行）
   * - auto: 智能模式，启用频谱分诊，按需分离
   * - force_on: 强制分离，跳过频谱分诊，全局分离
   */
  const strategy = localConfig.value.preprocessing.demucs_strategy
  if (strategy === 'off') {
    localConfig.value.preprocessing.enable_spectral_triage = false
    localConfig.value.preprocessing.separation_mode = 'on_demand'
  } else if (strategy === 'auto') {
    localConfig.value.preprocessing.enable_spectral_triage = true
    localConfig.value.preprocessing.separation_mode = 'on_demand'
  } else if (strategy === 'force_on') {
    localConfig.value.preprocessing.enable_spectral_triage = false
    localConfig.value.preprocessing.separation_mode = 'global'
  }

  /* 检查当前配置是否匹配某个预设 */
  const matchedPreset = macroPresets.find(preset => {
    return (
      localConfig.value.preprocessing.demucs_strategy === preset.config.preprocessing.demucs_strategy &&
      localConfig.value.preprocessing.language_detection_mode === preset.config.preprocessing.language_detection_mode &&
      localConfig.value.preprocessing.enable_speaker_embedding === preset.config.preprocessing.enable_speaker_embedding &&
      localConfig.value.transcription.transcription_profile === preset.config.transcription.transcription_profile &&
      localConfig.value.refinement.llm_task === preset.config.refinement.llm_task &&
      localConfig.value.refinement.llm_scope === preset.config.refinement.llm_scope
    )
  })

  localConfig.value.preset_id = matchedPreset ? matchedPreset.id : 'custom'
  emitChange()
}

/* 发送变更事件 */
function emitChange() {
  emit('update:modelValue', JSON.parse(JSON.stringify(localConfig.value)))
  emit('change', localConfig.value)
}

/* 监听外部值变化 */
watch(() => props.modelValue, (newVal) => {
  localConfig.value = JSON.parse(JSON.stringify(newVal))
}, { deep: true })

/* 监听高级设置开关 */
watch(showAdvanced, (val) => {
  if (val) {
    emit('openAdvanced', localConfig.value)
  }
})

/* 加载硬件信息 */
onMounted(async () => {
  try {
    const response = await getHardwareInfo()
    if (response.success && response.hardware) {
      /* 后端返回的是 gpu 字段，不是 gpu_info */
      const gpuInfo = response.hardware.gpu
      if (gpuInfo) {
        /* cuda_available 表示是否有可用的 GPU */
        hasGpu.value = gpuInfo.cuda_available === true
        /* 显存使用 total_memory_mb 字段 */
        if (gpuInfo.total_memory_mb) {
          vramMB.value = gpuInfo.total_memory_mb
          vramGB.value = Math.floor(gpuInfo.total_memory_mb / 1024)
        }
      } else {
        hasGpu.value = false
      }

      /* 计算推荐预设 - 优先推荐智能均衡 */
      if (!hasGpu.value) {
        recommendedPresetId.value = 'fast'
      } else if (vramMB.value >= 4000) {
        /* 4GB+ 显存优先推荐智能均衡（最佳平衡点） */
        recommendedPresetId.value = 'balanced'
      } else {
        recommendedPresetId.value = 'fast'
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
.preset-selector {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* 顶层: 快捷场景宏 */
.macro-presets .section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.macro-presets .section-header .header-label {
  color: var(--af-text-normal);
  font-size: 13px;
  font-weight: 500;
}

.macro-presets .section-header .toggle-btn {
  padding: 4px 10px;
  background: var(--af-bg-tertiary);
  border-radius: var(--af-radius-sm);
  color: var(--af-text-muted);
  font-size: 11px;
  transition: all var(--af-transition-fast);
}

.macro-presets .section-header .toggle-btn:hover {
  background: var(--af-bg-elevated);
  color: var(--af-text-normal);
}

.macro-presets .hardware-hint {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  margin-bottom: 8px;
  background: var(--af-bg-tertiary);
  border-radius: var(--af-radius-sm);
  font-size: 11px;
}

.macro-presets .hardware-hint .hw-vram {
  color: var(--af-text-secondary);
}

.macro-presets .hardware-hint .hw-warning {
  color: var(--af-accent-warning);
}

.macro-presets .hardware-hint .hw-recommend {
  margin-left: auto;
  color: var(--af-accent-success);
}

.macro-presets .preset-cards {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
}

/* 快捷预设卡片 */
.preset-card {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 12px 8px;
  background: var(--af-bg-secondary);
  border: 1px solid var(--af-border-default);
  border-radius: var(--af-radius-md);
  transition: all var(--af-transition-fast);
  cursor: pointer;
}

.preset-card:hover {
  background: var(--af-bg-tertiary);
  border-color: var(--af-accent-primary);
}

.preset-card.active {
  background: rgb(var(--af-accent-primary-rgb), 0.08);
  border-color: var(--af-accent-primary);
}

.preset-card .preset-icon {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 32px;
  height: 32px;
  margin-bottom: 8px;
  background: var(--af-bg-tertiary);
  border-radius: var(--af-radius-sm);
  color: var(--af-text-secondary);
  font-size: 14px;
  font-weight: 600;
}

.preset-card.active .preset-icon {
  background: var(--af-accent-primary);
  color: var(--af-text-inverse);
}

.preset-card.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.preset-card.disabled:hover {
  background: var(--af-bg-secondary);
  border-color: var(--af-border-default);
}

.preset-card.recommended:not(.active) {
  border-color: var(--af-accent-success);
}

.preset-card.recommended:not(.active) .preset-icon {
  background: var(--af-accent-success);
  color: var(--af-text-inverse);
}

.preset-card .preset-icon::before {
  content: attr(data-icon);
}

.preset-card .preset-info {
  text-align: center;
}

.preset-card .preset-info .preset-name {
  margin-bottom: 2px;
  color: var(--af-text-normal);
  font-size: 12px;
  font-weight: 500;
}

.preset-card .preset-info .preset-desc {
  color: var(--af-text-muted);
  font-size: 10px;
  line-height: 1.3;
}

.preset-card .check-mark {
  position: absolute;
  top: 4px;
  right: 4px;
  width: 16px;
  height: 16px;
  color: var(--af-accent-primary);
}

.preset-card .check-mark svg {
  width: 100%;
  height: 100%;
}

.preset-card .recommend-badge {
  position: absolute;
  top: 2px;
  left: 2px;
  padding: 2px 6px;
  background: var(--af-accent-success);
  border-radius: var(--af-radius-sm);
  color: var(--af-text-inverse);
  font-size: 9px;
}

/* 底层: 三个模块卡片 */
.module-cards {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
}

.module-card {
  padding: 10px;
  background: var(--af-bg-secondary);
  border: 1px solid var(--af-border-default);
  border-radius: var(--af-radius-md);
}

/* 整体禁用状态（LLM 模块暂未集成） */
.module-card.module-disabled {
  opacity: 0.5;
  pointer-events: none;
  cursor: not-allowed;
}

.module-card.module-disabled .module-title {
  color: var(--af-text-muted);
}

.module-card .module-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
}

.module-card .module-header .module-icon {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 18px;
  height: 18px;
  background: var(--af-bg-tertiary);
  border-radius: var(--af-radius-sm);
  color: var(--af-text-muted);
  font-size: 10px;
  font-weight: 600;
}

.module-card .module-header .module-title {
  color: var(--af-text-normal);
  font-size: 12px;
  font-weight: 500;
}

.module-card .module-options {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

/* 模块选项 */
.option-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 8px;
  border-radius: var(--af-radius-sm);
  transition: all var(--af-transition-fast);
  cursor: pointer;
}

.option-item:hover {
  background: var(--af-bg-tertiary);
}

.option-item input[type="radio"] {
  width: 12px;
  height: 12px;
  accent-color: var(--af-accent-primary);
}

.option-item .option-label {
  flex: 1;
  color: var(--af-text-normal);
  font-size: 11px;
}

.option-item.active {
  background: rgb(var(--af-accent-primary-rgb), 0.1);
}

.option-item.active .option-label {
  color: var(--af-accent-primary);
  font-weight: 500;
}

.option-item.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.option-item.disabled:hover {
  background: transparent;
}

.option-item .option-hint {
  color: var(--af-text-muted);
  font-size: 9px;
}

/* 当前方案信息 */
.current-preset-info {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  background: var(--af-bg-secondary);
  border-radius: var(--af-radius-sm);
  font-size: 11px;
}

.current-preset-info .info-label {
  color: var(--af-text-muted);
}

.current-preset-info .info-value {
  color: var(--af-text-normal);
  font-weight: 500;
}

/* 显存警告 */
.vram-warning {
  padding: 8px 12px;
  background: rgb(var(--af-accent-warning-rgb), 0.1);
  border: 1px solid var(--af-accent-warning);
  border-radius: var(--af-radius-sm);
  color: var(--af-accent-warning);
  font-size: 11px;
}

/* 紧凑模式样式 */
.preset-selector.compact {
  gap: 12px;
}

.preset-selector.compact .macro-presets .section-header {
  margin-bottom: 6px;
}

.preset-selector.compact .macro-presets .hardware-hint {
  padding: 4px 8px;
  margin-bottom: 6px;
}

.preset-selector.compact .macro-presets .preset-cards {
  gap: 6px;
}

.preset-selector.compact .preset-card {
  padding: 10px 6px;
}

.preset-selector.compact .preset-card .preset-icon {
  width: 28px;
  height: 28px;
  margin-bottom: 6px;
}

.preset-selector.compact .preset-card .preset-info .preset-name {
  font-size: 11px;
}

.preset-selector.compact .preset-card .preset-info .preset-desc {
  font-size: 9px;
}

.preset-selector.compact .module-cards {
  gap: 6px;
}

.preset-selector.compact .module-card {
  padding: 8px;
}

.preset-selector.compact .module-card .module-header {
  margin-bottom: 6px;
}

.preset-selector.compact .module-card .module-header .module-icon {
  width: 16px;
  height: 16px;
  font-size: 9px;
}

.preset-selector.compact .module-card .module-header .module-title {
  font-size: 11px;
}

.preset-selector.compact .option-item {
  gap: 4px;
  padding: 4px 6px;
}

.preset-selector.compact .option-item input[type="radio"] {
  width: 10px;
  height: 10px;
}

.preset-selector.compact .option-item .option-label {
  font-size: 10px;
}

.preset-selector.compact .option-item .option-hint {
  font-size: 8px;
}

.preset-selector.compact .current-preset-info {
  padding: 6px 8px;
  font-size: 10px;
}

.preset-selector.compact .vram-warning {
  padding: 6px 10px;
  font-size: 10px;
}
</style>
