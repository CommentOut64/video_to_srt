<template>
  <div class="settings-tabs">
    <el-tabs v-model="activeTab" :before-leave="handleBeforeLeave">
      <el-tab-pane label="预设" name="preset" />
      <el-tab-pane label="前处理" name="preprocess" />
      <el-tab-pane label="转录" name="transcription" />
      <el-tab-pane label="后处理" name="postprocess" />
    </el-tabs>

    <div class="tab-viewport">
      <Transition :name="slideDir">
        <div :key="activeTab" class="tab-content">
          <PresetTab
            v-if="activeTab === 'preset'"
            :model-value="modelValue"
            :custom-presets="customPresets"
            @update:model-value="handleConfigChange"
            @delete-preset="emit('delete-preset', $event)"
            @overwrite-preset="emit('overwrite-preset', $event)"
          />
          <PreprocessTab
            v-else-if="activeTab === 'preprocess'"
            :model-value="modelValue"
            @update:model-value="handleConfigChange"
          />
          <TranscriptionTab
            v-else-if="activeTab === 'transcription'"
            :model-value="modelValue"
            @update:model-value="handleConfigChange"
          />
          <PostprocessTab
            v-else
            :model-value="modelValue"
            @update:model-value="handleConfigChange"
          />
        </div>
      </Transition>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import PresetTab from './PresetTab.vue'
import PreprocessTab from './PreprocessTab.vue'
import TranscriptionTab from './TranscriptionTab.vue'
import PostprocessTab from './PostprocessTab.vue'

/* Tab 顺序表，用于判断滑动方向 */
const tabOrder = ['preset', 'preprocess', 'transcription', 'postprocess']

/* 内置预设匹配字段表 */
const builtinPresets = [
  {
    id: 'fast',
    config: {
      preprocessing: { demucs_strategy: 'off', language_detection_mode: 'balanced' },
      transcription: { transcription_profile: 'sensevoice_only', edge_selection_mode: 'force_fast' },
      refinement: { llm_task: 'off', llm_scope: 'sparse' },
    },
  },
  {
    id: 'balanced',
    config: {
      preprocessing: { demucs_strategy: 'auto', language_detection_mode: 'balanced' },
      transcription: { transcription_profile: 'sv_whisper_patch', edge_selection_mode: 'auto' },
      refinement: { llm_task: 'proofread', llm_scope: 'sparse' },
    },
  },
  {
    id: 'quality',
    config: {
      preprocessing: { demucs_strategy: 'force_on', language_detection_mode: 'balanced' },
      transcription: { transcription_profile: 'sv_whisper_dual', edge_selection_mode: 'auto' },
      refinement: { llm_task: 'proofread', llm_scope: 'global' },
    },
  },
]

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

const emit = defineEmits(['update:modelValue', 'save-preset', 'delete-preset', 'overwrite-preset'])

const activeTab = ref('preset')
const slideDir = ref('slide-left')

/* Tab 切换前：根据新旧索引计算滑动方向 */
function handleBeforeLeave(newTab, oldTab) {
  const newIdx = tabOrder.indexOf(newTab)
  const oldIdx = tabOrder.indexOf(oldTab)
  slideDir.value = newIdx > oldIdx ? 'slide-left' : 'slide-right'
  return true
}

/* 检测当前配置是否匹配某个内置预设 */
function detectPreset(config) {
  for (const preset of builtinPresets) {
    if (
      config.preprocessing.demucs_strategy === preset.config.preprocessing.demucs_strategy &&
      config.preprocessing.language_detection_mode === preset.config.preprocessing.language_detection_mode &&
      config.transcription.transcription_profile === preset.config.transcription.transcription_profile &&
      (config.transcription.edge_selection_mode || 'auto') === preset.config.transcription.edge_selection_mode &&
      config.refinement.llm_task === preset.config.refinement.llm_task &&
      config.refinement.llm_scope === preset.config.refinement.llm_scope
    ) {
      return preset.id
    }
  }
  return 'custom'
}

/* Tab 2-4 修改参数后的统一处理：检测预设匹配 + 上报 */
function handleConfigChange(newConfig) {
  const detectedPresetId = detectPreset(newConfig)
  const updated = { ...newConfig, preset_id: detectedPresetId }
  emit('update:modelValue', updated)
}
</script>

<style scoped>
.settings-tabs {
  margin-top: 8px;
}

/* --- 胶囊分段控件 Tab 头 ---
 * 原因: 替代默认下划线 Tab，提供更精致的分段控件视觉
 * 参考: element-plus/theme-chalk/src/tabs.scss
 */
.settings-tabs :deep(.el-tabs__header) {
  --el-tabs-header-height: 32px;
  margin-bottom: 8px;
  border-bottom: none;
}

/* 移除 nav-wrap 底部伪元素分割线 */
.settings-tabs :deep(.el-tabs__nav-wrap::after) {
  display: none;
}

/* nav 容器作为胶囊轨道 */
.settings-tabs :deep(.el-tabs__nav-scroll) {
  padding: 0 2px;
}

.settings-tabs :deep(.el-tabs__nav) {
  display: inline-flex;
  gap: 2px;
  padding: 3px;
  background: rgba(255, 255, 255, 0.04);
  border-radius: 8px;
  float: none;
}

/* 隐藏默认下划线活动条 */
.settings-tabs :deep(.el-tabs__active-bar) {
  display: none;
}

/* Tab 项: 胶囊按钮 */
.settings-tabs :deep(.el-tabs__item) {
  height: 26px;
  padding: 0 18px !important;
  border-bottom: none;
  border-radius: 6px;
  color: var(--af-text-muted);
  font-size: 12px;
  font-weight: 500;
  line-height: 26px;
  transition:
    color 0.2s ease,
    background 0.2s ease,
    box-shadow 0.2s ease;
}

.settings-tabs :deep(.el-tabs__item:hover) {
  color: var(--af-text-primary);
}

/* 活跃 Tab: 浮起胶囊 */
.settings-tabs :deep(.el-tabs__item.is-active) {
  background: var(--af-bg-elevated, var(--af-bg-secondary));
  border-radius: 6px;
  box-shadow:
    0 1px 3px rgba(0, 0, 0, 0.15),
    0 0 0 1px rgba(255, 255, 255, 0.04) inset;
  color: var(--af-accent-primary);
}

/* 隐藏 el-tabs 默认内容区，由 tab-viewport 接管渲染 */
.settings-tabs :deep(.el-tabs__content) {
  display: none;
}

/* --- 内容视口：固定高度 + 裁切溢出 --- */
.tab-viewport {
  position: relative;
  overflow: hidden;
  height: 160px;
  background: var(--af-bg-primary);
  border: 1px solid var(--af-border-default);
  border-radius: var(--af-radius-md);
}

.tab-content {
  height: 100%;
  padding: 10px 14px;
  overflow-y: auto;
}

/* 细滚动条 */
.tab-content::-webkit-scrollbar {
  width: 4px;
}

.tab-content::-webkit-scrollbar-thumb {
  background-color: var(--af-text-muted);
  border-radius: 2px;
  opacity: 0.4;
}

.tab-content::-webkit-scrollbar-thumb:hover {
  opacity: 0.7;
}

.tab-content::-webkit-scrollbar-track {
  background: transparent;
}

/* --- Tab 切换滑动过渡 ---
 * 原因: 同时存在离开与进入面板，离开面板绝对定位不占流 */
.slide-left-enter-active,
.slide-left-leave-active,
.slide-right-enter-active,
.slide-right-leave-active {
  transition: transform 0.2s ease, opacity 0.2s ease;
}

.slide-left-leave-active,
.slide-right-leave-active {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
}

/* 向左滑：新面板从右侧进入，旧面板向左退出 */
.slide-left-enter-from {
  transform: translateX(40px);
  opacity: 0;
}

.slide-left-leave-to {
  transform: translateX(-40px);
  opacity: 0;
}

/* 向右滑：新面板从左侧进入，旧面板向右退出 */
.slide-right-enter-from {
  transform: translateX(-40px);
  opacity: 0;
}

.slide-right-leave-to {
  transform: translateX(40px);
  opacity: 0;
}
</style>
