<template>
  <div class="preprocess-tab">
    <!-- 音频预检 -->
    <SettingRow label="人声分离策略" hint="音频预检">
      <el-select
        :model-value="modelValue.preprocessing.demucs_strategy"
        @update:model-value="updateField('preprocessing', 'demucs_strategy', $event)"
      >
        <el-option value="off" label="禁用 (直通)" />
        <el-option value="auto" label="智能分诊" />
        <el-option value="force_on" label="强制分离" />
      </el-select>
    </SettingRow>

    <!-- 语言检测 -->
    <SettingRow label="检测模式" hint="语言检测">
      <el-select
        :model-value="modelValue.preprocessing.language_detection_mode"
        @update:model-value="updateField('preprocessing', 'language_detection_mode', $event)"
      >
        <el-option value="fast" label="极速" />
        <el-option value="balanced" label="均衡" />
        <el-option value="precise" label="精准" />
      </el-select>
    </SettingRow>

    <SettingRow label="指定语言" :disabled="true">
      <el-select
        :model-value="'auto'"
        disabled
      >
        <el-option value="auto" label="自动检测" />
      </el-select>
    </SettingRow>
  </div>
</template>

<script setup>
import SettingRow from './shared/SettingRow.vue'

const props = defineProps({
  modelValue: {
    type: Object,
    required: true,
  },
})

const emit = defineEmits(['update:modelValue'])

function updateField(group, field, value) {
  const updated = JSON.parse(JSON.stringify(props.modelValue))
  updated[group][field] = value

  /* 根据 demucs_strategy 自动推导关联字段 */
  if (group === 'preprocessing' && field === 'demucs_strategy') {
    if (value === 'off') {
      updated.preprocessing.enable_spectral_triage = false
      updated.preprocessing.separation_mode = 'on_demand'
    } else if (value === 'auto') {
      updated.preprocessing.enable_spectral_triage = true
      updated.preprocessing.separation_mode = 'on_demand'
    } else if (value === 'force_on') {
      updated.preprocessing.enable_spectral_triage = false
      updated.preprocessing.separation_mode = 'global'
    }
  }

  emit('update:modelValue', updated)
}
</script>

<style scoped>
.preprocess-tab {
  display: flex;
  flex-direction: column;
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

:deep(.el-select) .el-input__wrapper:hover {
  box-shadow: 0 0 0 1px var(--af-accent-primary) inset;
}
</style>
