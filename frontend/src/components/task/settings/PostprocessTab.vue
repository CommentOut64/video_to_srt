<template>
  <div class="postprocess-tab">
    <!-- 切分配置占位 -->
    <SettingRow label="切分配置" hint="后续适配" :disabled="true">
      <span class="placeholder-text">暂未开放，将在后续版本中支持</span>
    </SettingRow>

    <!-- LLM 校对 -->
    <SettingRow label="校对模式" hint="LLM 校对" :disabled="true">
      <el-select
        :model-value="modelValue.refinement.llm_task"
        disabled
      >
        <el-option value="off" label="关闭" />
        <el-option value="proofread" label="校对" />
        <el-option value="translate" label="翻译" />
      </el-select>
    </SettingRow>
  </div>
</template>

<script setup>
import SettingRow from './shared/SettingRow.vue'

defineProps({
  modelValue: {
    type: Object,
    required: true,
  },
})

defineEmits(['update:modelValue'])
</script>

<style scoped>
.postprocess-tab {
  display: flex;
  flex-direction: column;
}

.placeholder-text {
  color: var(--af-text-disabled);
  font-size: 11px;
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
