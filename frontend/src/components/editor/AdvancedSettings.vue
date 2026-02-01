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
          <span class="slider-value">{{ localConfig.general.global_time_offset.toFixed(1) }}s</span>
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
            v-model.number="localConfig.general.global_time_offset"
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

// 当前激活的 Tab
const activeTab = ref('general')

// Tab 定义
const tabs = [
  { id: 'general', label: '常规' },
  { id: 'audio', label: 'Audio' },
  { id: 'asr', label: 'ASR' },
  { id: 'llm', label: 'LLM' },
  { id: 'system', label: 'System' }
]

// 本地配置 (深拷贝)
const localConfig = ref(JSON.parse(JSON.stringify(props.modelValue)))

// 发送变更事件
function emitChange() {
  // 检查是否还匹配某个预设
  localConfig.value.preset_id = 'custom'
  emit('update:modelValue', JSON.parse(JSON.stringify(localConfig.value)))
  emit('change', localConfig.value)
}

// 监听外部值变化
watch(() => props.modelValue, (newVal) => {
  localConfig.value = JSON.parse(JSON.stringify(newVal))
}, { deep: true })
</script>

<style lang="scss" scoped>
.advanced-settings {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

// Tab 页切换
.settings-tabs {
  display: flex;
  gap: 4px;
  padding: 4px;
  background: var(--bg-tertiary);
  border-radius: var(--radius-md);

  .tab-btn {
    flex: 1;
    padding: 6px 12px;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-muted);
    background: transparent;
    border-radius: var(--radius-sm);
    transition: all var(--transition-fast);

    &:hover {
      color: var(--text-normal);
      background: var(--bg-quaternary);
    }

    &.active {
      color: var(--primary);
      background: var(--bg-secondary);
    }
  }
}

// 设置面板
.settings-panel {
  background: var(--bg-secondary);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  padding: 12px;

  .panel-header {
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-default);

    .panel-title {
      font-size: 12px;
      font-weight: 500;
      color: var(--text-normal);
    }
  }
}

// 设置行
.setting-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 0;
  border-bottom: 1px solid var(--border-default);

  &:last-child {
    border-bottom: none;
  }

  &.disabled {
    opacity: 0.5;
    pointer-events: none;

    .setting-control {
      opacity: 0.6;
    }
  }

  .setting-label {
    display: flex;
    flex-direction: column;
    gap: 2px;

    .label-text {
      font-size: 12px;
      color: var(--text-normal);
    }

    .label-hint {
      font-size: 10px;
      color: var(--text-muted);
      font-family: 'SF Mono', Monaco, monospace;
    }
  }

  .setting-control {
    min-width: 140px;

    select, input[type="text"], input[type="number"] {
      width: 100%;
      padding: 6px 10px;
      font-size: 11px;
      background: var(--bg-tertiary);
      border: 1px solid var(--border-default);
      border-radius: var(--radius-sm);
      color: var(--text-normal);

      &:focus {
        outline: none;
        border-color: var(--primary);
      }

      &:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }
    }

    &.slider-control {
      display: flex;
      align-items: center;
      gap: 8px;

      input[type="range"] {
        flex: 1;
        height: 4px;
        accent-color: var(--primary);
      }

      .slider-value {
        min-width: 36px;
        font-size: 11px;
        color: var(--text-secondary);
        text-align: right;
        font-family: 'SF Mono', Monaco, monospace;
      }
    }
  }
}

// Toggle 开关
.toggle-switch {
  position: relative;
  display: inline-block;
  width: 36px;
  height: 20px;

  input {
    opacity: 0;
    width: 0;
    height: 0;
  }

  .toggle-slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: var(--bg-quaternary);
    transition: all var(--transition-fast);
    border-radius: 10px;

    &::before {
      position: absolute;
      content: "";
      height: 16px;
      width: 16px;
      left: 2px;
      bottom: 2px;
      background-color: white;
      transition: all var(--transition-fast);
      border-radius: 50%;
    }
  }

  input:checked + .toggle-slider {
    background-color: var(--primary);
  }

  input:checked + .toggle-slider::before {
    transform: translateX(16px);
  }
}
</style>
