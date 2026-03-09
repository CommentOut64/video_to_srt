<template>
  <div class="advanced-settings">
    <!-- Tab 页切换 -->
    <div class="settings-tabs">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        class="tab-btn"
        :class="{ active: activeTab === tab.id }"
        @click="handleTabClick(tab.id)"
      >
        {{ tab.label }}
      </button>
    </div>

    <!-- V3.2.4+dev.20260303.03: 常规设置分组重构 -->
    <div v-show="activeTab === 'general'" class="settings-panel">

      <!-- 字幕组 -->
      <div class="setting-group">
        <div class="group-header">
          <span class="group-title">字幕</span>
        </div>

        <!-- 全局时间偏移 -->
        <div class="setting-row">
          <div class="setting-label">
            <span class="label-text">全局时间偏移</span>
            <span class="label-hint">调整所有字幕的时间偏移（秒）</span>
          </div>
          <div class="setting-control">
            <el-input-number
              v-model="localConfig.general.global_time_offset"
              :step="0.01"
              :precision="2"
              :controls="false"
              @change="emitChange"
              size="small"
            />
          </div>
        </div>

        <!-- 全局时长调整 -->
        <div class="setting-row">
          <div class="setting-label">
            <span class="label-text">全局时长调整</span>
            <span class="label-hint">调整所有字幕的持续时间 (秒)</span>
          </div>
          <div class="setting-control">
            <el-input-number
              v-model="localConfig.general.duration_adjust"
              :step="0.1"
              :precision="1"
              :controls="false"
              @change="emitChange"
              size="small"
            />
          </div>
        </div>
      </div>

      <!-- 预览组 -->
      <div class="setting-group">
        <div class="group-header">
          <span class="group-title">预览</span>
        </div>

        <!-- 字幕预览字体大小 -->
        <div class="setting-row">
          <div class="setting-label">
            <span class="label-text">预览字体大小</span>
          </div>
          <div class="setting-control slider-control">
            <el-slider
              v-model="localConfig.general.preview_font_size"
              :min="12"
              :max="48"
              :step="1"
              :show-tooltip="false"
              @change="emitChange"
            />
            <span class="slider-value">{{ localConfig.general.preview_font_size }}px</span>
          </div>
        </div>
      </div>

      <!-- 界面组 -->
      <div class="setting-group">
        <div class="group-header">
          <span class="group-title">界面</span>
        </div>

        <!-- 隐藏波形刻度 -->
        <div class="setting-row">
          <div class="setting-label">
            <span class="label-text">隐藏波形刻度</span>
            <span class="label-hint">隐藏波形区域顶部的刻度</span>
          </div>
          <div class="setting-control">
            <el-switch
              v-model="localConfig.general.hide_timeline_scale"
              @change="emitChange"
            />
          </div>
        </div>

        <!-- 字幕跟随自动恢复 -->
        <div class="setting-row">
          <div class="setting-label">
            <span class="label-text">字幕跟随自动恢复</span>
            <span class="label-hint">开启后字幕列表会在无滚动操作后5s自动恢复跟随</span>
          </div>
          <div class="setting-control">
            <el-switch
              v-model="localConfig.general.subtitle_follow_auto_resume"
              @change="emitChange"
            />
          </div>
        </div>

        <!-- 自动保存间隔 -->
        <div class="setting-row">
          <div class="setting-label">
            <span class="label-text">自动保存间隔</span>
          </div>
          <div class="setting-control">
            <el-select
              v-model="localConfig.general.auto_save_interval"
              @change="emitChange"
              size="small"
            >
              <el-option :value="0" label="关闭" />
              <el-option :value="30" label="30 秒" />
              <el-option :value="60" label="1 分钟" />
              <el-option :value="120" label="2 分钟" />
              <el-option :value="300" label="5 分钟" />
            </el-select>
          </div>
        </div>
      </div>

      <!-- 合并和拆分组 -->
      <div class="setting-group">
        <div class="group-header">
          <span class="group-title">合并和拆分</span>
        </div>

        <!-- 字幕合并分隔符 -->
        <div class="setting-row">
          <div class="setting-label">
            <span class="label-text">字幕合并分隔符</span>
            <span class="label-hint">会在合并两条字幕时自动添加至合并处</span>
          </div>
          <div class="setting-control">
            <el-select
              v-model="localConfig.general.merge_separator"
              @change="emitChange"
              size="small"
            >
              <el-option value="space" label="空格 (默认)" />
              <el-option value="comma-full" label="全角逗号 ，" />
              <el-option value="comma-half" label="半角逗号 ," />
              <el-option value="period-full" label="全角句号 。" />
              <el-option value="period-half" label="半角句号 ." />
              <el-option value="custom" label="自定义" />
            </el-select>
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
            <el-input
              v-model="localConfig.general.merge_separator_custom"
              @input="emitChange"
              placeholder="输入自定义分隔符"
              :maxlength="10"
              size="small"
            />
          </div>
        </div>
      </div>
    </div>

    <!-- 分组零点五: 快捷键 -->
    <div
      v-if="canShowShortcutTab"
      v-show="activeTab === 'shortcuts'"
      class="settings-panel"
    >
      <div class="panel-header">
        <span class="panel-title">快捷键映射（编辑器）</span>
      </div>
      <div class="setting-row">
        <div class="setting-label">
          <span class="label-text">提示</span>
          <span class="label-hint">点“录制”后按任意组合键；Esc 取消录制；浏览器保留组合会被拦截（如 Ctrl+W / Ctrl+T / F5）</span>
        </div>
        <div class="setting-control">
          <el-button
            size="small"
            @click="resetAllShortcutsToDefault"
          >
            全部恢复默认
          </el-button>
        </div>
      </div>

      <div
        v-for="shortcutField in editorShortcutFields"
        :key="shortcutField.key"
        class="setting-row"
      >
        <div class="setting-label">
          <span class="label-text">{{ shortcutField.label }}</span>
          <span class="label-hint">{{ shortcutField.key }}</span>
        </div>
        <div class="setting-control shortcut-control">
          <el-input
            :model-value="getShortcutDisplay(localConfig.shortcuts[shortcutField.key])"
            readonly
            size="small"
          />
          <el-button
            size="small"
            :type="recordingActionKey === shortcutField.key ? 'danger' : 'primary'"
            @click="startShortcutRecording(shortcutField.key)"
          >
            {{ recordingActionKey === shortcutField.key ? '录制中...' : '录制' }}
          </el-button>
          <el-button
            size="small"
            text
            @click="resetShortcutToDefault(shortcutField.key)"
          >
            恢复默认
          </el-button>
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
          <el-select
            v-model="localConfig.preprocessing.demucs_strategy"
            @change="emitChange"
            size="small"
          >
            <el-option value="off" label="Off - 禁止分离" />
            <el-option value="auto" label="Auto - 智能分诊" />
            <el-option value="force_on" label="Force On - 极致分离" />
          </el-select>
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
          <el-select
            v-model="localConfig.preprocessing.demucs_model"
            :disabled="localConfig.preprocessing.demucs_strategy === 'off'"
            @change="emitChange"
            size="small"
          >
            <el-option value="htdemucs" label="htdemucs (推荐)" />
            <el-option value="htdemucs_ft" label="htdemucs_ft (Fine-tuned)" />
            <el-option value="mdx_q" label="mdx_q (量化版)" />
            <el-option value="mdx_extra" label="mdx_extra (高质量)" />
          </el-select>
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
          <el-slider
            v-model="localConfig.preprocessing.demucs_shifts"
            :min="1"
            :max="5"
            :step="1"
            :show-tooltip="false"
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
          <el-slider
            v-model="localConfig.preprocessing.spectrum_threshold"
            :min="0"
            :max="1"
            :step="0.05"
            :show-tooltip="false"
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
          <el-switch
            v-model="localConfig.preprocessing.vad_filter"
            @change="emitChange"
          />
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
          <el-select
            v-model="localConfig.transcription.transcription_profile"
            @change="emitChange"
            size="small"
          >
            <el-option value="sensevoice_only" label="SenseVoice Only (极速)" />
            <el-option value="sv_whisper_patch" label="SV + Whisper 复核" />
            <el-option value="sv_whisper_dual" label="SV + Whisper 双流并行" />
          </el-select>
        </div>
      </div>

      <!-- SenseVoice 运行设备 -->
      <div class="setting-row">
        <div class="setting-label">
          <span class="label-text">SenseVoice 设备</span>
          <span class="label-hint">sensevoice_device</span>
        </div>
        <div class="setting-control">
          <el-select
            v-model="localConfig.transcription.sensevoice_device"
            @change="emitChange"
            size="small"
          >
            <el-option value="auto" label="Auto (优先 GPU)" />
            <el-option value="cpu" label="强制 CPU" />
          </el-select>
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
          <el-select
            v-model="localConfig.transcription.whisper_model"
            :disabled="localConfig.transcription.transcription_profile === 'sensevoice_only'"
            @change="emitChange"
            size="small"
          >
            <el-option value="tiny" label="Tiny" />
            <el-option value="small" label="Small" />
            <el-option value="medium" label="Medium (推荐)" />
            <el-option value="large-v3" label="Large-v3 (高精度)" />
          </el-select>
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
          <el-slider
            v-model="localConfig.transcription.patching_threshold"
            :min="0"
            :max="1"
            :step="0.05"
            :show-tooltip="false"
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
          <el-select
            v-model="localConfig.refinement.llm_task"
            @change="emitChange"
            size="small"
          >
            <el-option value="off" label="Off - 关闭" />
            <el-option value="proofread" label="Proofread - 校对" />
            <el-option value="translate" label="Translate - 翻译" />
          </el-select>
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
          <el-select
            v-model="localConfig.refinement.llm_scope"
            :disabled="localConfig.refinement.llm_task === 'off'"
            @change="emitChange"
            size="small"
          >
            <el-option value="sparse" label="Sparse - 稀疏模式" />
            <el-option value="global" label="Global - 全局模式" />
          </el-select>
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
          <el-slider
            v-model="localConfig.refinement.sparse_threshold"
            :min="0"
            :max="1"
            :step="0.05"
            :show-tooltip="false"
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
          <el-select
            v-model="localConfig.refinement.target_language"
            :disabled="localConfig.refinement.llm_task !== 'translate'"
            @change="emitChange"
            size="small"
          >
            <el-option value="zh" label="中文" />
            <el-option value="en" label="English" />
            <el-option value="ja" label="日本语" />
            <el-option value="ko" label="韩语" />
          </el-select>
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
          <el-select
            v-model="localConfig.refinement.llm_provider"
            :disabled="localConfig.refinement.llm_task === 'off'"
            @change="emitChange"
            size="small"
          >
            <el-option value="openai_compatible" label="OpenAI Compatible" />
            <el-option value="local_ollama" label="Local Ollama" />
          </el-select>
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
          <el-input
            v-model="localConfig.refinement.llm_model_name"
            :disabled="localConfig.refinement.llm_task === 'off'"
            @input="emitChange"
            placeholder="gpt-4o-mini"
            size="small"
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
          <el-select
            v-model="localConfig.compute.concurrency_strategy"
            @change="emitChange"
            size="small"
          >
            <el-option value="auto" label="Auto - 自动" />
            <el-option value="parallel" label="Parallel - 并行" />
            <el-option value="serial" label="Serial - 串行" />
          </el-select>
        </div>
      </div>

      <!-- GPU 选择 -->
      <div class="setting-row">
        <div class="setting-label">
          <span class="label-text">GPU 选择</span>
          <span class="label-hint">gpu_id</span>
        </div>
        <div class="setting-control">
          <el-input-number
            v-model="localConfig.compute.gpu_id"
            :min="0"
            :max="7"
            :step="1"
            controls-position="right"
            @change="emitChange"
            size="small"
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
          <el-select
            v-model="localConfig.compute.temp_file_policy"
            @change="emitChange"
            size="small"
          >
            <el-option value="delete_on_complete" label="完成后删除" />
            <el-option value="keep" label="保留 (Debug)" />
          </el-select>
        </div>
      </div>

      <div class="setting-group diagnostics-group">
        <div class="group-header diagnostics-header">
          <div>
            <span class="group-title">运行时诊断</span>
            <div class="diagnostics-meta">
              <span>Shell: {{ shellBridgeAvailable ? '已连接' : '不可用' }}</span>
              <span>队列诊断: {{ queueDiagnosticsEnabled ? '已启用' : '未启用' }}</span>
              <span>更新时间: {{ lastUpdatedAt || '未获取' }}</span>
            </div>
          </div>
          <div class="diagnostics-actions">
            <el-button size="small" :loading="diagnosticsLoading" @click="refreshDiagnostics()">刷新</el-button>
            <el-button size="small" :loading="diagnosticsCopying" @click="copySnapshot">复制快照</el-button>
          </div>
        </div>

        <div v-if="diagnosticsErrorMessage" class="diagnostics-error">
          {{ diagnosticsErrorMessage }}
        </div>

        <div class="diagnostics-grid">
          <div class="diagnostics-card">
            <div class="diagnostics-card-title">Electron / GPU</div>
            <div class="diagnostics-card-body">
              <div v-for="item in shellGpuSummaryRows" :key="item.label" class="diagnostics-row">
                <span class="diagnostics-label">{{ item.label }}</span>
                <span class="diagnostics-value">{{ item.value }}</span>
              </div>
              <div class="diagnostics-row diagnostics-row-multiline">
                <span class="diagnostics-label">策略备注</span>
                <span class="diagnostics-value">{{ shellRuntimeInfo?.policy?.notes?.join('；') || '-' }}</span>
              </div>
              <div class="diagnostics-row diagnostics-row-multiline">
                <span class="diagnostics-label">已追加开关</span>
                <span class="diagnostics-value">{{ shellRuntimeInfo?.policy?.appliedSwitches?.join(', ') || '-' }}</span>
              </div>
            </div>
          </div>

          <div class="diagnostics-card">
            <div class="diagnostics-card-title">任务运行时</div>
            <div class="diagnostics-card-body">
              <div v-for="item in queueSummaryRows" :key="item.label" class="diagnostics-row">
                <span class="diagnostics-label">{{ item.label }}</span>
                <span class="diagnostics-value">{{ item.value }}</span>
              </div>
            </div>
          </div>

          <div class="diagnostics-card">
            <div class="diagnostics-card-title">播放链路</div>
            <div class="diagnostics-card-body">
              <div v-for="item in playbackSummaryRows" :key="item.label" class="diagnostics-row">
                <span class="diagnostics-label">{{ item.label }}</span>
                <span class="diagnostics-value">{{ item.value }}</span>
              </div>
            </div>
          </div>
        </div>

        <div class="diagnostics-card diagnostics-json-card">
          <div class="diagnostics-card-title">诊断快照 JSON</div>
          <pre class="diagnostics-json">{{ diagnosticsSnapshotText }}</pre>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onUnmounted, ref, watch } from 'vue'
import { useShellRuntimeDiagnostics } from '@/composables/useShellRuntimeDiagnostics'
import { IS_LITE } from '@/config/flavor'
import {
  buildShortcutComboFromKeyboardEvent,
  EDITOR_SHORTCUT_FIELDS,
  getEditorShortcutComboLabel,
  DEFAULT_EDITOR_SHORTCUT_CONFIG,
  normalizeEditorShortcutConfig,
} from '@/utils/editorShortcuts'

const props = defineProps({
  modelValue: {
    type: Object,
    required: true
  },
  enableShortcutCustomization: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['update:modelValue', 'change', 'open-about'])

/* 当前激活的 Tab */
const activeTab = ref('general')
const isLiteMode = computed(() => IS_LITE)
const canShowShortcutTab = computed(() => isLiteMode.value || props.enableShortcutCustomization)

/* Tab 定义 */
const tabs = computed(() => {
  if (isLiteMode.value) {
    return [
      { id: 'general', label: '常规' },
      { id: 'system', label: '系统' },
      { id: 'shortcuts', label: '快捷键' },
      { id: 'about', label: '关于' }
    ]
  }

  const baseTabs = [
    { id: 'general', label: '常规' },
    { id: 'audio', label: 'Audio' },
    { id: 'asr', label: 'ASR' },
    { id: 'llm', label: 'LLM' },
    { id: 'system', label: 'System' }
  ]
  if (canShowShortcutTab.value) {
    baseTabs.push({ id: 'shortcuts', label: '快捷键' })
  }
  baseTabs.push({ id: 'about', label: '关于' })
  return baseTabs
})

// V3.2.4+dev.20260303.04: 点击"关于"Tab 时 emit 事件，不切换面板
function handleTabClick(tabId) {
  if (tabId === 'about') {
    emit('open-about')
    return
  }
  activeTab.value = tabId
}

const editorShortcutFields = EDITOR_SHORTCUT_FIELDS
const {
  shellRuntimeInfo,
  queueRuntimeInfo,
  playbackSummary,
  shellPolicy,
  gpuFeatureStatus,
  shellBridgeAvailable,
  queueDiagnosticsEnabled,
  isLoading: diagnosticsLoading,
  isCopying: diagnosticsCopying,
  errorMessage: diagnosticsErrorMessage,
  lastUpdatedAt,
  refreshDiagnostics,
  copySnapshot,
  safeJsonStringify,
} = useShellRuntimeDiagnostics()

const shellGpuSummaryRows = computed(() => {
  const runtimeInfo = shellRuntimeInfo.value || {}
  const featureStatus = gpuFeatureStatus.value || {}
  const policy = shellPolicy.value || {}
  return [
    { label: '媒体 Profile', value: policy.mediaProfile || runtimeInfo.policy?.mediaProfile || '-' },
    { label: 'GPU 策略', value: policy.effectiveMode || runtimeInfo.policy?.effectiveMode || '-' },
    { label: '激活 GPU 类型', value: runtimeInfo.activeGpuType || '-' },
    { label: '硬件加速', value: runtimeInfo.hardwareAccelerationEnabled === true ? '开启' : (runtimeInfo.hardwareAccelerationEnabled === false ? '关闭' : '-') },
    { label: 'video_decode', value: featureStatus.video_decode || '-' },
    { label: 'gpu_compositing', value: featureStatus.gpu_compositing || '-' },
    { label: '平台', value: [runtimeInfo.platform, runtimeInfo.arch].filter(Boolean).join(' / ') || '-' },
    { label: 'Electron', value: runtimeInfo.versions?.electron || '-' },
  ]
})

const queueSummaryRows = computed(() => {
  const queueInfo = queueRuntimeInfo.value || {}
  return [
    { label: '运行时诊断', value: queueInfo.runtime_enabled === true ? '已启用' : '未启用' },
    { label: '队列长度', value: queueInfo.queue_length ?? '-' },
    { label: '任务数', value: queueInfo.jobs_count ?? '-' },
    { label: '孤儿执行', value: queueInfo.orphan_execution_count ?? '-' },
    { label: 'Runner 线程', value: queueInfo.runner_thread_count ?? '-' },
    { label: '存活 Runner', value: queueInfo.runner_alive_count ?? '-' },
    { label: 'GPU Busy Override', value: queueInfo.is_gpu_busy_override === true ? '是' : '否' },
    { label: 'Gate Blocking', value: queueInfo.is_runner_gate_blocking === true ? '是' : '否' },
  ]
})

function formatDiagnosticMs(value) {
  return Number.isFinite(Number(value)) ? `${Math.round(Number(value))} ms` : '-'
}

function formatFrameSummary(dropped, total) {
  const droppedValue = Number.isFinite(Number(dropped)) ? Number(dropped) : null
  const totalValue = Number.isFinite(Number(total)) ? Number(total) : null
  if (droppedValue === null && totalValue === null) {
    return '-'
  }
  if (droppedValue === null) {
    return `- / ${totalValue}`
  }
  if (totalValue === null) {
    return `${droppedValue}`
  }
  return `${droppedValue} / ${totalValue}`
}
const playbackSummaryRows = computed(() => {
  const playbackInfo = playbackSummary.value || {}
  return [
    { label: '视频来源', value: playbackInfo.media_source_kind || '-' },
    { label: '媒体 Profile', value: playbackInfo.media_profile || '-' },
    { label: '当前分辨率', value: playbackInfo.current_resolution || '-' },
    { label: '首帧耗时', value: formatDiagnosticMs(playbackInfo.video_first_frame_ms) },
    { label: 'Seek 耗时', value: formatDiagnosticMs(playbackInfo.video_seek_latency_ms) },
    { label: '卡顿次数', value: playbackInfo.video_stall_count ?? 0 },
    { label: '卡顿总时长', value: formatDiagnosticMs(playbackInfo.video_stall_total_ms) },
    { label: '最近卡顿', value: formatDiagnosticMs(playbackInfo.video_last_stall_ms) },
    { label: '掉帧 / 总帧', value: formatFrameSummary(playbackInfo.video_decode_drop_count, playbackInfo.video_total_frame_count) },
    { label: '最近错误', value: playbackInfo.last_error_message || '-' },
    { label: '采样更新时间', value: playbackInfo.last_updated_at || '-' },
  ]
})

const diagnosticsSnapshotText = computed(() => {
  return safeJsonStringify({
    shell_runtime: shellRuntimeInfo.value,
    queue_runtime: queueRuntimeInfo.value,
    playback_runtime: playbackSummary.value,
    updated_at: lastUpdatedAt.value,
  })
})

const recordingActionKey = ref('')

function normalizeConfig(config) {
  const normalized = JSON.parse(JSON.stringify(config))
  if (!normalized.general) {
    normalized.general = {}
  }
  if (normalized.general.subtitle_follow_auto_resume === undefined) {
    normalized.general.subtitle_follow_auto_resume = true
  }
  if (normalized.general.enable_shortcuts === undefined) {
    normalized.general.enable_shortcuts = true
  }
  if (normalized.general.hide_timeline_scale === undefined) {
    normalized.general.hide_timeline_scale = false
  }
  // V3.2.4+dev.20260302.02: 合并分隔符默认值
  if (!normalized.general.merge_separator) {
    normalized.general.merge_separator = 'space'
  }
  if (normalized.general.merge_separator_custom === undefined) {
    normalized.general.merge_separator_custom = ''
  }
  normalized.shortcuts = normalizeEditorShortcutConfig(normalized.shortcuts)
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

function getShortcutDisplay(combo) {
  return getEditorShortcutComboLabel(combo)
}

function startShortcutRecording(actionKey) {
  if (recordingActionKey.value === actionKey) {
    recordingActionKey.value = ''
    return
  }
  recordingActionKey.value = actionKey
}

function resetShortcutToDefault(actionKey) {
  const defaults = normalizeEditorShortcutConfig(DEFAULT_EDITOR_SHORTCUT_CONFIG)
  localConfig.value.shortcuts[actionKey] = defaults[actionKey]
  recordingActionKey.value = ''
  emitChange()
}

function resetAllShortcutsToDefault() {
  localConfig.value.shortcuts = normalizeEditorShortcutConfig(DEFAULT_EDITOR_SHORTCUT_CONFIG)
  recordingActionKey.value = ''
  emitChange()
}

function handleShortcutRecording(event) {
  if (!recordingActionKey.value) return
  event.preventDefault()
  event.stopPropagation()

  if (event.key === 'Escape') {
    recordingActionKey.value = ''
    return
  }

  const combo = buildShortcutComboFromKeyboardEvent(event)
  if (!combo) return

  localConfig.value.shortcuts[recordingActionKey.value] = combo
  recordingActionKey.value = ''
  emitChange()
}

/* 监听外部值变化 */
watch(() => props.modelValue, (newVal) => {
  localConfig.value = normalizeConfig(newVal)
}, { deep: true })

watch(tabs, (nextTabs) => {
  const currentExists = nextTabs.some((tab) => tab.id === activeTab.value)
  if (!currentExists) {
    activeTab.value = 'general'
  }
}, { immediate: true })

watch(recordingActionKey, (nextValue) => {
  if (nextValue) {
    window.addEventListener('keydown', handleShortcutRecording, true)
    return
  }
  window.removeEventListener('keydown', handleShortcutRecording, true)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleShortcutRecording, true)
})
</script>

<style scoped>
/* V3.2.4+dev.20260303.02: 全面替换原生控件为 Element Plus 组件 */

/* 组件本地变量 */
.diagnostics-group {
  margin-top: 12px;
}

.diagnostics-header {
  align-items: flex-start;
  gap: 12px;
}

.diagnostics-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 14px;
  margin-top: 6px;
  font-size: 12px;
  color: var(--af-text-secondary);
}

.diagnostics-actions {
  display: flex;
  gap: 8px;
}

.diagnostics-error {
  margin-top: 8px;
  padding: 8px 10px;
  border-radius: 8px;
  color: var(--af-text-on-dark);
  background: rgba(var(--af-danger-rgb, 220, 38, 38), 0.28);
}

.diagnostics-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 12px;
}

.diagnostics-card {
  padding: 12px;
  border: 1px solid rgba(var(--af-text-muted-rgb), 0.18);
  border-radius: 10px;
  background: rgba(var(--af-text-on-dark-rgb), 0.04);
}

.diagnostics-card-title {
  margin-bottom: 10px;
  font-size: 13px;
  font-weight: 600;
  color: var(--af-text-primary);
}

.diagnostics-card-body {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.diagnostics-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 12px;
}

.diagnostics-row-multiline {
  align-items: flex-start;
}

.diagnostics-label {
  color: var(--af-text-secondary);
  flex: 0 0 112px;
}

.diagnostics-value {
  color: var(--af-text-primary);
  text-align: right;
  word-break: break-word;
  flex: 1;
}

.diagnostics-json-card {
  margin-top: 12px;
}

.diagnostics-json {
  margin: 0;
  max-height: 220px;
  overflow: auto;
  font-size: 12px;
  line-height: 1.5;
  color: var(--af-text-normal);
  white-space: pre-wrap;
  word-break: break-word;
}

@media (max-width: 900px) {
  .diagnostics-grid {
    grid-template-columns: 1fr;
  }
}

.advanced-settings {
  --control-min-width: 180px;

  display: flex;
  flex-direction: column;
  gap: 12px;
}

/* ==========================================
   Tab 页切换 — 胶囊浮起风格
   ========================================== */
.settings-tabs {
  display: flex;
  gap: 4px;
  padding: 3px;
  background: rgba(var(--af-text-on-dark-rgb), 0.04);
  border-radius: var(--af-radius-md);
}

.settings-tabs .tab-btn {
  flex: 1;
  padding: 7px 14px;
  background: transparent;
  border: none;
  border-radius: var(--af-radius-sm);
  color: var(--af-text-muted);
  font-size: 11px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.settings-tabs .tab-btn:hover {
  color: var(--af-text-normal);
  background: rgba(var(--af-text-on-dark-rgb), 0.04);
}

.settings-tabs .tab-btn.active {
  background: var(--af-bg-elevated);
  color: var(--af-accent-primary);
  box-shadow:
    0 1px 3px rgba(0, 0, 0, 0.15),
    0 0 0 1px rgba(var(--af-text-on-dark-rgb), 0.04) inset;
}

/* ==========================================
   设置面板 — 去框，内容为主
   ========================================== */
.settings-panel {
  padding: 0;
  background: transparent;
  border: none;
  border-radius: 0;
}

.settings-panel .panel-header {
  margin-bottom: 10px;
  padding-bottom: 0;
  border-bottom: none;
}

.settings-panel .panel-header .panel-title {
  color: var(--af-text-secondary);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

/* ==========================================
   设置分组 — 常规 Tab 子分组
   ========================================== */
.setting-group + .setting-group {
  margin-top: 8px;
  padding-top: 12px;
  border-top: 1px solid var(--af-border-subtle);
}

.setting-group .group-header {
  margin-bottom: 4px;
}

.setting-group .group-header .group-title {
  color: var(--af-text-secondary);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

/* ==========================================
   设置行 — 精细间距 + 虚线分隔
   ========================================== */
.setting-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px dashed var(--af-border-muted);
  transition: opacity 0.2s ease, filter 0.2s ease;
}

.setting-row:last-child {
  border-bottom: none;
}

.setting-row .setting-label {
  display: flex;
  flex-direction: column;
  gap: 2px;
  flex-shrink: 0;
  min-width: 100px;
}

/* V3.2.4+dev.20260303.02: 标题字号从 12px 增大到 13px */
.setting-row .setting-label .label-text {
  color: var(--af-text-normal);
  font-size: 14px;
  font-weight: 500;
  line-height: 1.4;
}

.setting-row .setting-label .label-hint {
  color: var(--af-text-muted);
  font-size: 10px;
  font-family: var(--af-font-mono);
  line-height: 1.3;
}

.setting-row .setting-control {
  min-width: var(--control-min-width);
}

.setting-row .setting-control.shortcut-control {
  display: flex;
  align-items: center;
  gap: 8px;
}

.setting-row .setting-control.shortcut-control :deep(.el-input) {
  flex: 1;
}

/* 禁用态 */
.setting-row.disabled {
  opacity: 0.4;
  filter: grayscale(0.3);
  pointer-events: none;
}

/* ==========================================
   el-select 下拉框样式适配
   原因：el-select 的内部 DOM 需要 :deep() 穿透设置尺寸和背景
   参考：theme-chalk/src/select.scss
   ========================================== */
.setting-row .setting-control :deep(.el-select) {
  width: 100%;
}

.setting-row .setting-control :deep(.el-select .el-input__wrapper) {
  background-color: var(--af-bg-tertiary);
  box-shadow: 0 0 0 1px var(--af-border-default) inset;
  border-radius: var(--af-radius-sm);
  padding: 2px 8px;
  transition: box-shadow 0.15s ease;
}

.setting-row .setting-control :deep(.el-select .el-input__wrapper:hover) {
  box-shadow: 0 0 0 1px var(--af-border-default) inset,
    0 0 0 1px rgba(var(--af-accent-primary-rgb), 0.1);
}

.setting-row .setting-control :deep(.el-select .el-input.is-focus .el-input__wrapper) {
  box-shadow: 0 0 0 1px var(--af-accent-primary) inset;
}

.setting-row .setting-control :deep(.el-select .el-input__inner) {
  color: var(--af-text-normal);
  font-size: 12px;
}

.setting-row .setting-control :deep(.el-select .el-input__suffix .el-icon) {
  color: var(--af-text-muted);
  font-size: 12px;
}

.setting-row .setting-control :deep(.el-select.is-disabled .el-input__wrapper) {
  background-color: var(--af-bg-tertiary);
  cursor: not-allowed;
}

/* ==========================================
   el-slider 滑块样式适配
   原因：el-slider 内部 DOM 需要 :deep() 穿透自定义轨道和滑块样式
   参考：theme-chalk/src/slider.scss
   ========================================== */
.setting-row .setting-control.slider-control {
  display: flex;
  align-items: center;
  gap: 10px;
}

.setting-row .setting-control.slider-control :deep(.el-slider) {
  flex: 1;
}

.setting-row .setting-control.slider-control :deep(.el-slider__runway) {
  height: 4px;
  background-color: var(--af-bg-tertiary);
  border-radius: 2px;
}

.setting-row .setting-control.slider-control :deep(.el-slider__bar) {
  height: 4px;
  background-color: var(--af-accent-primary);
  border-radius: 2px;
}

.setting-row .setting-control.slider-control :deep(.el-slider__button) {
  width: 14px;
  height: 14px;
  border: 2px solid var(--af-accent-primary);
  background-color: var(--af-bg-secondary);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.setting-row .setting-control.slider-control :deep(.el-slider__button:hover) {
  transform: scale(1.15);
  box-shadow: 0 0 0 3px rgba(var(--af-accent-primary-rgb), 0.2);
}

.setting-row .setting-control.slider-control :deep(.el-slider.is-disabled .el-slider__bar) {
  background-color: var(--af-text-disabled);
}

.setting-row .setting-control.slider-control :deep(.el-slider.is-disabled .el-slider__button) {
  border-color: var(--af-text-disabled);
}

/* 滑块数值标签 */
.setting-row .setting-control.slider-control .slider-value {
  min-width: 40px;
  padding: 3px 6px;
  background: var(--af-bg-tertiary);
  border-radius: var(--af-radius-xs);
  color: var(--af-text-secondary);
  font-size: 11px;
  font-family: var(--af-font-mono);
  text-align: center;
  line-height: 1.3;
}

/* ==========================================
   el-input 输入框样式适配
   原因：el-input 内部 DOM 需要 :deep() 穿透设置背景和边框
   参考：theme-chalk/src/input.scss
   ========================================== */
.setting-row .setting-control :deep(.el-input) {
  width: 100%;
}

.setting-row .setting-control :deep(.el-input .el-input__wrapper) {
  background-color: var(--af-bg-tertiary);
  box-shadow: 0 0 0 1px var(--af-border-default) inset;
  border-radius: var(--af-radius-sm);
  padding: 2px 8px;
  transition: box-shadow 0.15s ease;
}

.setting-row .setting-control :deep(.el-input .el-input__wrapper:hover) {
  box-shadow: 0 0 0 1px var(--af-border-default) inset,
    0 0 0 1px rgba(var(--af-accent-primary-rgb), 0.1);
}

.setting-row .setting-control :deep(.el-input.is-focus .el-input__wrapper),
.setting-row .setting-control :deep(.el-input .el-input__wrapper.is-focus) {
  box-shadow: 0 0 0 1px var(--af-accent-primary) inset;
}

.setting-row .setting-control :deep(.el-input .el-input__inner) {
  color: var(--af-text-normal);
  font-size: 12px;
}

.setting-row .setting-control :deep(.el-input.is-disabled .el-input__wrapper) {
  background-color: var(--af-bg-tertiary);
  cursor: not-allowed;
}

/* ==========================================
   el-input-number 数字输入框样式适配
   原因：el-input-number 包裹 el-input，需要统一宽度和边框
   参考：theme-chalk/src/input-number.scss
   ========================================== */
.setting-row .setting-control :deep(.el-input-number) {
  width: 100%;
}

.setting-row .setting-control :deep(.el-input-number .el-input__wrapper) {
  background-color: var(--af-bg-tertiary);
  box-shadow: 0 0 0 1px var(--af-border-default) inset;
  border-radius: var(--af-radius-sm);
  padding: 2px 8px;
}

.setting-row .setting-control :deep(.el-input-number .el-input-number__increase),
.setting-row .setting-control :deep(.el-input-number .el-input-number__decrease) {
  background-color: transparent;
  color: var(--af-text-muted);
  border-color: var(--af-border-default);
}

.setting-row .setting-control :deep(.el-input-number .el-input-number__increase:hover),
.setting-row .setting-control :deep(.el-input-number .el-input-number__decrease:hover) {
  color: var(--af-accent-primary);
}

/* ==========================================
   el-switch 开关样式适配
   原因：el-switch 需要 :deep() 穿透微调尺寸和边框
   参考：theme-chalk/src/switch.scss
   ========================================== */
.setting-row .setting-control :deep(.el-switch) {
  --el-switch-on-color: var(--af-accent-primary);
  --el-switch-off-color: var(--af-bg-elevated);
  height: 22px;
}

.setting-row .setting-control :deep(.el-switch .el-switch__core) {
  min-width: 38px;
  height: 22px;
  border: 1px solid var(--af-border-default);
  border-radius: 11px;
}

.setting-row .setting-control :deep(.el-switch.is-checked .el-switch__core) {
  border-color: var(--af-accent-primary);
}

.setting-row .setting-control :deep(.el-switch .el-switch__core .el-switch__action) {
  width: 18px;
  height: 18px;
}
</style>
