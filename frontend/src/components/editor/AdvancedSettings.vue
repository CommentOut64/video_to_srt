<template>
  <div class="advanced-settings">
    <!-- Tab 页切换 -->
    <div class="settings-tabs">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        class="tab-btn"
        :class="{ active: activeTab === tab.id }"
        @click="activeTab = tab.id"
      >
        {{ tab.label }}
      </button>
    </div>

    <!-- 分组零: 常规设置 -->
    <div v-show="activeTab === 'general'" class="settings-panel">
      <div class="panel-header">
        <span class="panel-title">常规设置</span>
      </div>

      <!-- 全局时间偏移 -->
      <div class="setting-row">
        <div class="setting-label">
          <span class="label-text">全局时间偏移</span>
          <span class="label-hint">global_time_offset (秒)</span>
        </div>
        <div class="setting-control slider-control">
          <input
            type="range"
            min="-10"
            max="10"
            step="0.1"
            v-model.number="localConfig.general.global_time_offset"
            @change="emitChange"
          />
          <span class="slider-value">{{ Number(localConfig.general.global_time_offset || 0).toFixed(1) }}s</span>
        </div>
      </div>

      <!-- 精确时间偏移输入 -->
      <div class="setting-row">
        <div class="setting-label">
          <span class="label-text">精确偏移值</span>
          <span class="label-hint">输入精确数值（秒）</span>
        </div>
        <div class="setting-control">
          <input
            type="number"
            step="0.01"
            v-model="localConfig.general.global_time_offset"
            @input="emitChange"
            placeholder="0.00"
          />
        </div>
      </div>

      <!-- 字幕显示时长调整 -->
      <div class="setting-row">
        <div class="setting-label">
          <span class="label-text">字幕显示时长调整</span>
          <span class="label-hint">duration_adjust (秒)</span>
        </div>
        <div class="setting-control slider-control">
          <input
            type="range"
            min="-2"
            max="2"
            step="0.1"
            v-model.number="localConfig.general.duration_adjust"
            @change="emitChange"
          />
          <span class="slider-value">{{ localConfig.general.duration_adjust >= 0 ? '+' : '' }}{{ localConfig.general.duration_adjust.toFixed(1) }}s</span>
        </div>
      </div>

      <!-- 自动保存间隔 -->
      <div class="setting-row">
        <div class="setting-label">
          <span class="label-text">自动保存间隔</span>
          <span class="label-hint">auto_save_interval (秒)</span>
        </div>
        <div class="setting-control">
          <select
            v-model.number="localConfig.general.auto_save_interval"
            @change="emitChange"
          >
            <option :value="0">关闭</option>
            <option :value="30">30 秒</option>
            <option :value="60">1 分钟</option>
            <option :value="120">2 分钟</option>
            <option :value="300">5 分钟</option>
          </select>
        </div>
      </div>

      <!-- 字幕预览字体大小 -->
      <div class="setting-row">
        <div class="setting-label">
          <span class="label-text">预览字体大小</span>
          <span class="label-hint">preview_font_size (px)</span>
        </div>
        <div class="setting-control slider-control">
          <input
            type="range"
            min="12"
            max="48"
            step="1"
            v-model.number="localConfig.general.preview_font_size"
            @change="emitChange"
          />
          <span class="slider-value">{{ localConfig.general.preview_font_size }}px</span>
        </div>
      </div>

      <!-- 快捷键启用 -->
      <div class="setting-row">
        <div class="setting-label">
          <span class="label-text">启用快捷键</span>
          <span class="label-hint">enable_shortcuts</span>
        </div>
        <div class="setting-control">
          <label class="toggle-switch">
            <input
              type="checkbox"
              v-model="localConfig.general.enable_shortcuts"
              @change="emitChange"
            />
            <span class="toggle-slider"></span>
          </label>
        </div>
      </div>

      <!-- 手动滚动后自动恢复跟随 -->
      <div class="setting-row">
        <div class="setting-label">
          <span class="label-text">字幕跟随自动恢复</span>
          <span class="label-hint">subtitle_follow_auto_resume</span>
        </div>
        <div class="setting-control">
          <label class="toggle-switch">
            <input
              type="checkbox"
              v-model="localConfig.general.subtitle_follow_auto_resume"
              @change="emitChange"
            />
            <span class="toggle-slider"></span>
          </label>
        </div>
      </div>

      <!-- V3.2.4+dev.20260302.02: 字幕合并分隔符 -->
      <div class="setting-row">
        <div class="setting-label">
          <span class="label-text">字幕合并分隔符</span>
          <span class="label-hint">merge_separator</span>
        </div>
        <div class="setting-control">
          <select
            v-model="localConfig.general.merge_separator"
            @change="emitChange"
          >
            <option value="space">空格 (默认)</option>
            <option value="comma-full">全角逗号 ，</option>
            <option value="comma-half">半角逗号 ,</option>
            <option value="period-full">全角句号 。</option>
            <option value="period-half">半角句号 .</option>
            <option value="custom">自定义</option>
          </select>
        </div>
      </div>

      <!-- 自定义分隔符输入（仅当选择"自定义"时显示） -->
      <div
        class="setting-row"
        v-if="localConfig.general.merge_separator === 'custom'"
      >
        <div class="setting-label">
          <span class="label-text">自定义分隔符</span>
          <span class="label-hint">merge_separator_custom</span>
        </div>
        <div class="setting-control">
          <input
            type="text"
            v-model="localConfig.general.merge_separator_custom"
            @input="emitChange"
            placeholder="输入自定义分隔符"
            maxlength="10"
          />
        </div>
      </div>
    </div>

    <!-- 分组一: 预处理与音频 -->
    <div v-show="activeTab === 'audio'" class="settings-panel">
      <div class="panel-header">
        <span class="panel-title">预处理与音频 (Demucs)</span>
      </div>

      <!-- 人声分离策略 -->
      <div class="setting-row">
        <div class="setting-label">
          <span class="label-text">人声分离策略</span>
          <span class="label-hint">demucs_strategy</span>
        </div>
        <div class="setting-control">
          <select
            v-model="localConfig.preprocessing.demucs_strategy"
            @change="emitChange"
          >
            <option value="off">Off - 禁止分离</option>
            <option value="auto">Auto - 智能分诊</option>
            <option value="force_on">Force On - 极致分离</option>
          </select>
        </div>
      </div>

      <!-- Demucs 模型 (依赖禁用) -->
      <div
        class="setting-row"
        :class="{ disabled: localConfig.preprocessing.demucs_strategy === 'off' }"
      >
        <div class="setting-label">
          <span class="label-text">分离模型</span>
          <span class="label-hint">demucs_model</span>
        </div>
        <div class="setting-control">
          <select
            v-model="localConfig.preprocessing.demucs_model"
            :disabled="localConfig.preprocessing.demucs_strategy === 'off'"
            @change="emitChange"
          >
            <option value="htdemucs">htdemucs (推荐)</option>
            <option value="htdemucs_ft">htdemucs_ft (Fine-tuned)</option>
            <option value="mdx_q">mdx_q (量化版)</option>
            <option value="mdx_extra">mdx_extra (高质量)</option>
          </select>
        </div>
      </div>

      <!-- Demucs Shifts (依赖禁用) -->
      <div
        class="setting-row"
        :class="{ disabled: localConfig.preprocessing.demucs_strategy === 'off' }"
      >
        <div class="setting-label">
          <span class="label-text">分离预测次数</span>
          <span class="label-hint">demucs_shifts (1-5)</span>
        </div>
        <div class="setting-control slider-control">
          <input
            type="range"
            min="1"
            max="5"
            step="1"
            v-model.number="localConfig.preprocessing.demucs_shifts"
            :disabled="localConfig.preprocessing.demucs_strategy === 'off'"
            @change="emitChange"
          />
          <span class="slider-value">{{ localConfig.preprocessing.demucs_shifts }}</span>
        </div>
      </div>

      <!-- 分诊灵敏度 -->
      <div class="setting-row">
        <div class="setting-label">
          <span class="label-text">分诊灵敏度</span>
          <span class="label-hint">spectrum_threshold (0.0-1.0)</span>
        </div>
        <div class="setting-control slider-control">
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            v-model.number="localConfig.preprocessing.spectrum_threshold"
            @change="emitChange"
          />
          <span class="slider-value">{{ localConfig.preprocessing.spectrum_threshold.toFixed(2) }}</span>
        </div>
      </div>

      <!-- VAD 静音过滤 -->
      <div class="setting-row">
        <div class="setting-label">
          <span class="label-text">VAD 静音过滤</span>
          <span class="label-hint">vad_filter</span>
        </div>
        <div class="setting-control">
          <label class="toggle-switch">
            <input
              type="checkbox"
              v-model="localConfig.preprocessing.vad_filter"
              @change="emitChange"
            />
            <span class="toggle-slider"></span>
          </label>
        </div>
      </div>
    </div>

    <!-- 分组二: 转录核心 -->
    <div v-show="activeTab === 'asr'" class="settings-panel">
      <div class="panel-header">
        <span class="panel-title">转录核心 (ASR)</span>
      </div>

      <!-- 转录流水线模式 -->
      <div class="setting-row">
        <div class="setting-label">
          <span class="label-text">转录流水线</span>
          <span class="label-hint">transcription_profile</span>
        </div>
        <div class="setting-control">
          <select
            v-model="localConfig.transcription.transcription_profile"
            @change="emitChange"
          >
            <option value="sensevoice_only">SenseVoice Only (极速)</option>
            <option value="sv_whisper_patch">SV + Whisper 复核</option>
            <option value="sv_whisper_dual">SV + Whisper 双流并行</option>
          </select>
        </div>
      </div>

      <!-- SenseVoice 运行设备 -->
      <div class="setting-row">
        <div class="setting-label">
          <span class="label-text">SenseVoice 设备</span>
          <span class="label-hint">sensevoice_device</span>
        </div>
        <div class="setting-control">
          <select
            v-model="localConfig.transcription.sensevoice_device"
            @change="emitChange"
          >
            <option value="auto">Auto (优先 GPU)</option>
            <option value="cpu">强制 CPU</option>
          </select>
        </div>
      </div>

      <!-- Whisper 模型 (依赖禁用) -->
      <div
        class="setting-row"
        :class="{ disabled: localConfig.transcription.transcription_profile === 'sensevoice_only' }"
      >
        <div class="setting-label">
          <span class="label-text">Whisper 模型</span>
          <span class="label-hint">whisper_model</span>
        </div>
        <div class="setting-control">
          <select
            v-model="localConfig.transcription.whisper_model"
            :disabled="localConfig.transcription.transcription_profile === 'sensevoice_only'"
            @change="emitChange"
          >
            <option value="tiny">Tiny</option>
            <option value="small">Small</option>
            <option value="medium">Medium (推荐)</option>
            <option value="large-v3">Large-v3 (高精度)</option>
          </select>
        </div>
      </div>

      <!-- 复核触发阈值 (依赖禁用) -->
      <div
        class="setting-row"
        :class="{ disabled: localConfig.transcription.transcription_profile !== 'sv_whisper_patch' }"
      >
        <div class="setting-label">
          <span class="label-text">复核触发阈值</span>
          <span class="label-hint">patching_threshold (0.0-1.0)</span>
        </div>
        <div class="setting-control slider-control">
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            v-model.number="localConfig.transcription.patching_threshold"
            :disabled="localConfig.transcription.transcription_profile !== 'sv_whisper_patch'"
            @change="emitChange"
          />
          <span class="slider-value">{{ localConfig.transcription.patching_threshold.toFixed(2) }}</span>
        </div>
      </div>
    </div>

    <!-- 分组三: 增强与润色 -->
    <div v-show="activeTab === 'llm'" class="settings-panel">
      <div class="panel-header">
        <span class="panel-title">增强与润色 (LLM)</span>
      </div>

      <!-- LLM 任务目标 -->
      <div class="setting-row">
        <div class="setting-label">
          <span class="label-text">LLM 任务</span>
          <span class="label-hint">llm_task</span>
        </div>
        <div class="setting-control">
          <select
            v-model="localConfig.refinement.llm_task"
            @change="emitChange"
          >
            <option value="off">Off - 关闭</option>
            <option value="proofread">Proofread - 校对</option>
            <option value="translate">Translate - 翻译</option>
          </select>
        </div>
      </div>

      <!-- LLM 介入范围 (依赖禁用) -->
      <div
        class="setting-row"
        :class="{ disabled: localConfig.refinement.llm_task === 'off' }"
      >
        <div class="setting-label">
          <span class="label-text">介入范围</span>
          <span class="label-hint">llm_scope</span>
        </div>
        <div class="setting-control">
          <select
            v-model="localConfig.refinement.llm_scope"
            :disabled="localConfig.refinement.llm_task === 'off'"
            @change="emitChange"
          >
            <option value="sparse">Sparse - 稀疏模式</option>
            <option value="global">Global - 全局模式</option>
          </select>
        </div>
      </div>

      <!-- 稀疏校对阈值 (依赖禁用) -->
      <div
        class="setting-row"
        :class="{ disabled: localConfig.refinement.llm_task === 'off' || localConfig.refinement.llm_scope !== 'sparse' }"
      >
        <div class="setting-label">
          <span class="label-text">稀疏校对阈值</span>
          <span class="label-hint">sparse_threshold (0.0-1.0)</span>
        </div>
        <div class="setting-control slider-control">
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            v-model.number="localConfig.refinement.sparse_threshold"
            :disabled="localConfig.refinement.llm_task === 'off' || localConfig.refinement.llm_scope !== 'sparse'"
            @change="emitChange"
          />
          <span class="slider-value">{{ localConfig.refinement.sparse_threshold.toFixed(2) }}</span>
        </div>
      </div>

      <!-- 目标语言 (翻译时使用) -->
      <div
        class="setting-row"
        :class="{ disabled: localConfig.refinement.llm_task !== 'translate' }"
      >
        <div class="setting-label">
          <span class="label-text">目标语言</span>
          <span class="label-hint">target_language</span>
        </div>
        <div class="setting-control">
          <select
            v-model="localConfig.refinement.target_language"
            :disabled="localConfig.refinement.llm_task !== 'translate'"
            @change="emitChange"
          >
            <option value="zh">中文</option>
            <option value="en">English</option>
            <option value="ja">日本语</option>
            <option value="ko">韩语</option>
          </select>
        </div>
      </div>

      <!-- LLM 提供商 (依赖禁用) -->
      <div
        class="setting-row"
        :class="{ disabled: localConfig.refinement.llm_task === 'off' }"
      >
        <div class="setting-label">
          <span class="label-text">LLM 提供商</span>
          <span class="label-hint">llm_provider</span>
        </div>
        <div class="setting-control">
          <select
            v-model="localConfig.refinement.llm_provider"
            :disabled="localConfig.refinement.llm_task === 'off'"
            @change="emitChange"
          >
            <option value="openai_compatible">OpenAI Compatible</option>
            <option value="local_ollama">Local Ollama</option>
          </select>
        </div>
      </div>

      <!-- LLM 模型名称 (依赖禁用) -->
      <div
        class="setting-row"
        :class="{ disabled: localConfig.refinement.llm_task === 'off' }"
      >
        <div class="setting-label">
          <span class="label-text">模型名称</span>
          <span class="label-hint">llm_model_name</span>
        </div>
        <div class="setting-control">
          <input
            type="text"
            v-model="localConfig.refinement.llm_model_name"
            :disabled="localConfig.refinement.llm_task === 'off'"
            @input="emitChange"
            placeholder="gpt-4o-mini"
          />
        </div>
      </div>
    </div>

    <!-- 分组四: 计算与系统 -->
    <div v-show="activeTab === 'system'" class="settings-panel">
      <div class="panel-header">
        <span class="panel-title">计算与系统</span>
      </div>

      <!-- 并发调度策略 -->
      <div class="setting-row">
        <div class="setting-label">
          <span class="label-text">并发策略</span>
          <span class="label-hint">concurrency_strategy</span>
        </div>
        <div class="setting-control">
          <select
            v-model="localConfig.compute.concurrency_strategy"
            @change="emitChange"
          >
            <option value="auto">Auto - 自动</option>
            <option value="parallel">Parallel - 并行</option>
            <option value="serial">Serial - 串行</option>
          </select>
        </div>
      </div>

      <!-- GPU 选择 -->
      <div class="setting-row">
        <div class="setting-label">
          <span class="label-text">GPU 选择</span>
          <span class="label-hint">gpu_id</span>
        </div>
        <div class="setting-control">
          <input
            type="number"
            min="0"
            max="7"
            v-model.number="localConfig.compute.gpu_id"
            @input="emitChange"
          />
        </div>
      </div>

      <!-- 临时文件策略 -->
      <div class="setting-row">
        <div class="setting-label">
          <span class="label-text">临时文件策略</span>
          <span class="label-hint">temp_file_policy</span>
        </div>
        <div class="setting-control">
          <select
            v-model="localConfig.compute.temp_file_policy"
            @change="emitChange"
          >
            <option value="delete_on_complete">完成后删除</option>
            <option value="keep">保留 (Debug)</option>
          </select>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'

const props = defineProps({
  modelValue: {
    type: Object,
    required: true
  }
})

const emit = defineEmits(['update:modelValue', 'change'])

/* 当前激活的 Tab */
const activeTab = ref('general')

/* Tab 定义 */
const tabs = [
  { id: 'general', label: '常规' },
  { id: 'audio', label: 'Audio' },
  { id: 'asr', label: 'ASR' },
  { id: 'llm', label: 'LLM' },
  { id: 'system', label: 'System' }
]

function normalizeConfig(config) {
  const normalized = JSON.parse(JSON.stringify(config))
  if (!normalized.general) {
    normalized.general = {}
  }
  if (normalized.general.subtitle_follow_auto_resume === undefined) {
    normalized.general.subtitle_follow_auto_resume = true
  }
  // V3.2.4+dev.20260302.02: 合并分隔符默认值
  if (!normalized.general.merge_separator) {
    normalized.general.merge_separator = 'space'
  }
  if (normalized.general.merge_separator_custom === undefined) {
    normalized.general.merge_separator_custom = ''
  }
  return normalized
}

/* 本地配置 (深拷贝 + 默认值归一化) */
const localConfig = ref(normalizeConfig(props.modelValue))

/* V3.2.0+dev.20260209.04: 移除自动规范化，允许用户输入空值和负号 */
/* 发送变更事件（仅用于数据同步，不触发实际应用） */
function emitChange() {
  /* 检查是否还匹配某个预设 */
  localConfig.value.preset_id = 'custom'
  emit('update:modelValue', normalizeConfig(localConfig.value))
  // V3.2.0+dev.20260209.03: 移除 change 事件，避免实时触发字幕偏移
  // 只有点击"保存"按钮才真正应用
}

/* 监听外部值变化 */
watch(() => props.modelValue, (newVal) => {
  localConfig.value = normalizeConfig(newVal)
}, { deep: true })
</script>

<style scoped>
.advanced-settings {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

/* Tab 页切换 */
.settings-tabs {
  display: flex;
  gap: 4px;
  padding: 4px;
  background: var(--af-bg-secondary);
  border-radius: var(--af-radius-md);
}

.settings-tabs .tab-btn {
  flex: 1;
  padding: 6px 12px;
  background: transparent;
  border-radius: var(--af-radius-sm);
  color: var(--af-text-muted);
  font-size: 11px;
  font-weight: 500;
  transition: all var(--af-transition-fast);
}

.settings-tabs .tab-btn:hover {
  background: var(--af-bg-tertiary);
  color: var(--af-text-normal);
}

.settings-tabs .tab-btn.active {
  background: var(--af-bg-tertiary);
  color: var(--af-accent-primary);
}

/* 设置面板 */
.settings-panel {
  padding: 12px;
  background: var(--af-bg-secondary);
  border: 1px solid var(--af-border-default);
  border-radius: var(--af-radius-md);
}

.settings-panel .panel-header {
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--af-border-default);
}

.settings-panel .panel-header .panel-title {
  color: var(--af-text-normal);
  font-size: 12px;
  font-weight: 500;
}

/* 设置行 */
.setting-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid var(--af-border-default);
}

.setting-row:last-child {
  border-bottom: none;
}

.setting-row .setting-label {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.setting-row .setting-label .label-text {
  color: var(--af-text-normal);
  font-size: 12px;
}

.setting-row .setting-label .label-hint {
  color: var(--af-text-muted);
  font-size: 10px;
  font-family: var(--af-font-mono);
}

.setting-row .setting-control {
  min-width: 140px;
}

.setting-row.disabled {
  opacity: 0.5;
  pointer-events: none;
}

.setting-row.disabled .setting-control {
  opacity: 0.6;
}

.setting-row .setting-control select,
.setting-row .setting-control input[type="text"],
.setting-row .setting-control input[type="number"] {
  width: 100%;
  padding: 6px 10px;
  background: var(--af-bg-tertiary);
  border: 1px solid var(--af-border-default);
  border-radius: var(--af-radius-sm);
  color: var(--af-text-normal);
  font-size: 11px;
}

.setting-row .setting-control select:focus,
.setting-row .setting-control input[type="text"]:focus,
.setting-row .setting-control input[type="number"]:focus {
  border-color: var(--af-accent-primary);
  outline: none;
}

.setting-row .setting-control select:disabled,
.setting-row .setting-control input[type="text"]:disabled,
.setting-row .setting-control input[type="number"]:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.setting-row .setting-control.slider-control {
  display: flex;
  align-items: center;
  gap: 8px;
}

.setting-row .setting-control.slider-control input[type="range"] {
  flex: 1;
  height: 4px;
  accent-color: var(--af-accent-primary);
}

.setting-row .setting-control.slider-control .slider-value {
  min-width: 36px;
  color: var(--af-text-secondary);
  font-size: 11px;
  font-family: var(--af-font-mono);
  text-align: right;
}

/* Toggle 开关 */
.toggle-switch {
  position: relative;
  display: inline-block;
  width: 36px;
  height: 20px;
}

.toggle-switch input {
  width: 0;
  height: 0;
  opacity: 0;
}

.toggle-switch .toggle-slider {
  position: absolute;
  inset: 0;
  background-color: var(--af-bg-elevated);
  border-radius: 10px;
  transition: all var(--af-transition-fast);
  cursor: pointer;
}

.toggle-switch .toggle-slider::before {
  position: absolute;
  bottom: 2px;
  left: 2px;
  width: 16px;
  height: 16px;
  background-color: var(--af-text-inverse);
  border-radius: 50%;
  transition: all var(--af-transition-fast);
  content: "";
}

.toggle-switch input:checked + .toggle-slider {
  background-color: var(--af-accent-primary);
}

.toggle-switch input:checked + .toggle-slider::before {
  transform: translateX(16px);
}
</style>
