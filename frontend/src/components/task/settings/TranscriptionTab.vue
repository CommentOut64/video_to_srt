<template>
  <div class="transcription-tab">
    <!-- 转录模式 -->
    <SettingRow label="转录模式">
      <el-select
        :model-value="modelValue.transcription.transcription_profile"
        @update:model-value="handleTranscriptionProfileChange"
      >
        <el-option value="sensevoice_only" label="极速 (仅 SenseVoice)" />
        <el-option value="sv_whisper_patch" label="SV + Whisper 复核" />
        <el-option value="sv_whisper_dual" label="双流精校 (很慢)" />
      </el-select>
    </SettingRow>

    <SettingRow label="选边模式" hint="双流定稿选边策略">
      <el-select
        :model-value="displayEdgeSelectionMode"
        :disabled="isFastProfile"
        @update:model-value="handleEdgeSelectionModeChange"
      >
        <el-option
          v-for="option in edgeSelectionOptions"
          :key="option.value"
          :value="option.value"
          :label="option.label"
        />
      </el-select>
    </SettingRow>

    <!-- 说话人检测 -->
    <SettingRow label="说话人检测" hint="说话人策略">
      <el-checkbox
        :model-value="modelValue.preprocessing.enable_speaker_detection"
        @update:model-value="handleSpeakerToggle($event)"
      >
        启用
      </el-checkbox>
    </SettingRow>

    <!-- 说话人数 -->
    <SettingRow label="说话人数" :disabled="!modelValue.preprocessing.enable_speaker_detection">
      <el-input-number
        :model-value="modelValue.preprocessing.speaker_count"
        :min="0"
        :max="20"
        :step="1"
        :disabled="!modelValue.preprocessing.enable_speaker_detection"
        size="small"
        controls-position="right"
        @update:model-value="updateField('preprocessing', 'speaker_count', $event)"
      />
      <span class="hint-text">(0=自动识别)</span>
    </SettingRow>
  </div>
</template>

<script setup>
import { computed, watch } from 'vue'
import SettingRow from './shared/SettingRow.vue'

const props = defineProps({
  modelValue: {
    type: Object,
    required: true,
  },
})

const emit = defineEmits(['update:modelValue'])

const EDGE_MODE_AUTO = 'auto'
const EDGE_MODE_PREFER_FAST = 'prefer_fast'
const EDGE_MODE_PREFER_SLOW = 'prefer_slow'
const EDGE_MODE_FORCE_FAST = 'force_fast'
const EDGE_MODE_FORCE_SLOW = 'force_slow'

const isFastProfile = computed(
  () => props.modelValue.transcription.transcription_profile === 'sensevoice_only'
)

const edgeSelectionOptions = computed(() => {
  if (isFastProfile.value) {
    return [{ value: EDGE_MODE_FORCE_FAST, label: '强制快流' }]
  }
  return [
    { value: EDGE_MODE_AUTO, label: '自动选边' },
    { value: EDGE_MODE_PREFER_FAST, label: '快流优先' },
    { value: EDGE_MODE_PREFER_SLOW, label: '慢流优先' },
  ]
})

const displayEdgeSelectionMode = computed(() => {
  return normalizeEdgeSelectionModeForProfile(
    props.modelValue.transcription.transcription_profile,
    props.modelValue.transcription.edge_selection_mode
  )
})

function updateField(group, field, value) {
  const updated = JSON.parse(JSON.stringify(props.modelValue))
  updated[group][field] = value
  emit('update:modelValue', updated)
}

function normalizeEdgeSelectionModeForProfile(profile, mode) {
  const normalizedProfile = String(profile || '').trim().toLowerCase()
  const normalizedMode = String(mode || EDGE_MODE_AUTO).trim().toLowerCase()

  if (normalizedProfile === 'sensevoice_only') {
    return EDGE_MODE_FORCE_FAST
  }
  if (
    normalizedMode === EDGE_MODE_AUTO ||
    normalizedMode === EDGE_MODE_PREFER_FAST ||
    normalizedMode === EDGE_MODE_PREFER_SLOW
  ) {
    return normalizedMode
  }
  if (normalizedMode === EDGE_MODE_FORCE_SLOW) {
    return EDGE_MODE_PREFER_SLOW
  }
  if (normalizedMode === EDGE_MODE_FORCE_FAST) {
    return EDGE_MODE_AUTO
  }
  return EDGE_MODE_AUTO
}

function handleEdgeSelectionModeChange(value) {
  const updated = JSON.parse(JSON.stringify(props.modelValue))
  updated.transcription.edge_selection_mode = normalizeEdgeSelectionModeForProfile(
    updated.transcription.transcription_profile,
    value
  )
  emit('update:modelValue', updated)
}

function handleTranscriptionProfileChange(value) {
  const updated = JSON.parse(JSON.stringify(props.modelValue))
  updated.transcription.transcription_profile = value
  updated.transcription.edge_selection_mode = normalizeEdgeSelectionModeForProfile(
    value,
    updated.transcription.edge_selection_mode
  )

  emit('update:modelValue', updated)
}

watch(
  () => [
    props.modelValue.transcription.transcription_profile,
    props.modelValue.transcription.edge_selection_mode,
  ],
  ([profile, mode]) => {
    const normalized = normalizeEdgeSelectionModeForProfile(profile, mode)
    const current = String(mode || EDGE_MODE_AUTO).trim().toLowerCase()
    if (current === normalized) {
      return
    }
    const updated = JSON.parse(JSON.stringify(props.modelValue))
    updated.transcription.edge_selection_mode = normalized
    emit('update:modelValue', updated)
  },
  { immediate: true }
)

/* 说话人检测开关的联动逻辑 */
function handleSpeakerToggle(enabled) {
  const updated = JSON.parse(JSON.stringify(props.modelValue))
  updated.preprocessing.enable_speaker_detection = enabled

  if (enabled) {
    /* 启用时默认开启 speaker_guided_split */
    updated.preprocessing.enable_speaker_guided_split = true
  } else {
    /* 禁用时清零关联字段 */
    updated.preprocessing.enable_speaker_guided_split = false
    updated.preprocessing.speaker_count = 0
  }

  emit('update:modelValue', updated)
}
</script>

<style scoped>
.transcription-tab {
  display: flex;
  flex-direction: column;
}

.hint-text {
  color: var(--af-text-muted);
  font-size: 10px;
  white-space: nowrap;
}

/* --- el-select 样式定制 ---
 * 原因: el-select 的内部 input 背景色需要跟随 Tab 内容区
 * 参考: theme-chalk/src/select.scss
 */
:deep(.el-select) {
  --el-select-input-color: var(--af-text-normal);
  --el-select-input-focus-border-color: var(--af-accent-primary);

  width: 200px;
}

:deep(.el-select) .el-input__wrapper {
  background-color: var(--af-bg-secondary);
  box-shadow: 0 0 0 1px var(--af-border-default) inset;
}

/* --- el-checkbox 样式定制 ---
 * 原因: 调整 checkbox 颜色跟随主题
 * 参考: theme-chalk/src/checkbox.scss
 */
:deep(.el-checkbox) {
  --el-checkbox-checked-bg-color: var(--af-accent-primary);
  --el-checkbox-checked-input-border-color: var(--af-accent-primary);
  --el-checkbox-input-border-color-hover: var(--af-accent-primary);
}

:deep(.el-checkbox) .el-checkbox__label {
  color: var(--af-text-normal);
  font-size: 12px;
}

/* --- el-input-number 样式定制 ---
 * 原因: el-input-number 内部 wrapper 背景需跟随 Tab
 * 参考: theme-chalk/src/input-number.scss
 */
:deep(.el-input-number) {
  width: 120px;
}

:deep(.el-input-number) .el-input__wrapper {
  background-color: var(--af-bg-secondary);
  box-shadow: 0 0 0 1px var(--af-border-default) inset;
}

:deep(.el-select) .el-input__wrapper:hover {
  box-shadow: 0 0 0 1px var(--af-accent-primary) inset;
}

:deep(.el-input-number) .el-input__wrapper:hover {
  box-shadow: 0 0 0 1px var(--af-accent-primary) inset;
}
</style>
