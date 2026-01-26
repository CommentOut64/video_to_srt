# Phase 4: 前端适配（修订版 v2.0）

> 目标：更新前端界面，支持预设方案选择、警告高亮系统和虚拟滚动
>
> 工期：2-3天
>
> 版本更新：整合 [06_转录层深度优化_时空解耦架构](./06_转录层深度优化_时空解耦架构.md) 设计

---

## ⚠️ 重要修订

### v2.0 新增（时空解耦架构）

- ✅ **新增**：预设选择器组件（6 个预设 + 高级自定义）
- ✅ **新增**：置信度警告高亮系统
- ✅ **新增**：虚拟滚动优化（支持大量字幕）
- ✅ **新增**：高级设置面板
- ✅ **新增**：统一 SSE 事件监听（使用新 Tag）

### v1.0 基础修订

- ❌ **删除**：不创建使用 `hardware_detector.py` 的 API
- ✅ **复用**：使用现有的 `hardware_service.py` 和 `hardware_routes.py`

---

## 一、任务清单

| 任务 | 文件 | 优先级 |
|------|------|--------|
| **预设选择器组件** | `PresetSelector.vue` | **P0** |
| **置信度警告高亮** | `SubtitleItem.vue` | **P0** |
| **虚拟滚动优化** | `SubtitleList.vue` | **P1** |
| **高级设置面板** | `AdvancedSettings.vue` | **P2** |
| 引擎选择器 | `TaskListView.vue` | P0 |
| 硬件状态显示 | `TaskListView.vue` | P1 |
| 实时字幕预览 | `TaskListView.vue` | P1 |
| API 适配 | `api/index.js` | P0 |

---

## 二、预设选择器组件（新增）

**路径**: `frontend/src/components/PresetSelector.vue`

```vue
<template>
  <div class="preset-selector">
    <h3 class="section-title">转录方案</h3>

    <!-- 预设卡片网格 -->
    <div class="preset-grid">
      <div
        v-for="preset in presets"
        :key="preset.id"
        :class="['preset-card', { active: selectedPreset === preset.id }]"
        @click="selectPreset(preset.id)"
      >
        <div class="preset-header">
          <span class="preset-name">{{ preset.name }}</span>
          <span v-if="preset.recommended" class="preset-badge">推荐</span>
        </div>
        <p class="preset-description">{{ preset.description }}</p>
        <div class="preset-meta">
          <span class="meta-item">
            耗时倍率: {{ preset.timeMultiplier }}x
          </span>
        </div>
      </div>

      <!-- 高级自定义卡片 -->
      <div
        :class="['preset-card', 'custom-card', { active: selectedPreset === 'custom' }]"
        @click="openAdvancedSettings"
      >
        <div class="preset-header">
          <span class="preset-name">高级自定义</span>
        </div>
        <p class="preset-description">自由组合各模块，适合专业用户</p>
      </div>
    </div>

    <!-- 当前配置摘要 -->
    <div class="config-summary" v-if="selectedPreset !== 'custom'">
      <div class="summary-item" v-if="currentConfig.enhancement !== 'off'">
        Whisper 复核: {{ getEnhancementLabel(currentConfig.enhancement) }}
      </div>
      <div class="summary-item" v-if="currentConfig.proofread !== 'off'">
        LLM 校对: {{ getProofreadLabel(currentConfig.proofread) }}
      </div>
      <div class="summary-item" v-if="currentConfig.translate !== 'off'">
        LLM 翻译: {{ getTranslateLabel(currentConfig.translate) }}
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'PresetSelector',

  data() {
    return {
      selectedPreset: 'preset1',
      presets: [
        {
          id: 'default',
          name: 'SenseVoice Only',
          description: '极速模式，仅使用 SenseVoice 转录',
          timeMultiplier: 0.1,
          recommended: false
        },
        {
          id: 'preset1',
          name: '智能复核',
          description: 'SV + Whisper 局部复核，平衡速度与质量',
          timeMultiplier: 0.15,
          recommended: true
        },
        {
          id: 'preset2',
          name: '轻度校对',
          description: '智能复核 + LLM 按需校对问题片段',
          timeMultiplier: 0.2,
          recommended: false
        },
        {
          id: 'preset3',
          name: '深度校对',
          description: '智能复核 + LLM 全文精修润色',
          timeMultiplier: 0.3,
          recommended: false
        },
        {
          id: 'preset4',
          name: '校对+翻译',
          description: '深度校对 + 全文翻译',
          timeMultiplier: 0.5,
          recommended: false
        },
        {
          id: 'preset5',
          name: '校对+重点翻译',
          description: '深度校对 + 仅翻译标记的重点段落',
          timeMultiplier: 0.35,
          recommended: false
        }
      ],
      currentConfig: {
        enhancement: 'smart_patch',
        proofread: 'off',
        translate: 'off'
      }
    }
  },

  methods: {
    selectPreset(presetId) {
      this.selectedPreset = presetId
      this.updateConfig(presetId)
      this.$emit('preset-changed', presetId, this.currentConfig)
    },

    updateConfig(presetId) {
      const configs = {
        'default': { enhancement: 'off', proofread: 'off', translate: 'off' },
        'preset1': { enhancement: 'smart_patch', proofread: 'off', translate: 'off' },
        'preset2': { enhancement: 'smart_patch', proofread: 'sparse', translate: 'off' },
        'preset3': { enhancement: 'smart_patch', proofread: 'full', translate: 'off' },
        'preset4': { enhancement: 'smart_patch', proofread: 'full', translate: 'full' },
        'preset5': { enhancement: 'smart_patch', proofread: 'full', translate: 'partial' }
      }
      this.currentConfig = configs[presetId] || configs['default']
    },

    openAdvancedSettings() {
      this.selectedPreset = 'custom'
      this.$emit('open-advanced')
    },

    getEnhancementLabel(mode) {
      const labels = {
        'off': '关闭',
        'smart_patch': '智能复核',
        'deep_listen': '深度重听'
      }
      return labels[mode] || mode
    },

    getProofreadLabel(mode) {
      const labels = {
        'off': '关闭',
        'sparse': '按需校对',
        'full': '全文精修'
      }
      return labels[mode] || mode
    },

    getTranslateLabel(mode) {
      const labels = {
        'off': '关闭',
        'full': '全文翻译',
        'partial': '重点翻译'
      }
      return labels[mode] || mode
    }
  }
}
</script>

<style scoped>
.preset-selector {
  margin-bottom: 2rem;
}

.section-title {
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: #333;
}

.preset-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1rem;
}

.preset-card {
  padding: 1rem;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  background: #fff;
}

.preset-card:hover {
  border-color: #42b983;
  box-shadow: 0 2px 8px rgba(66, 185, 131, 0.2);
}

.preset-card.active {
  border-color: #42b983;
  background: rgba(66, 185, 131, 0.05);
}

.preset-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.preset-name {
  font-weight: 600;
  color: #333;
}

.preset-badge {
  font-size: 0.7rem;
  padding: 0.2rem 0.5rem;
  background: #42b983;
  color: #fff;
  border-radius: 4px;
}

.preset-description {
  font-size: 0.85rem;
  color: #666;
  margin: 0;
  line-height: 1.4;
}

.preset-meta {
  margin-top: 0.75rem;
  font-size: 0.8rem;
  color: #999;
}

.custom-card {
  border-style: dashed;
}

.config-summary {
  margin-top: 1rem;
  padding: 0.75rem;
  background: #f5f5f5;
  border-radius: 4px;
  font-size: 0.85rem;
}

.summary-item {
  display: inline-block;
  margin-right: 1rem;
  color: #42b983;
}
</style>
```

---

## 三、置信度警告高亮系统（新增）

**路径**: `frontend/src/components/SubtitleItem.vue`

```vue
<template>
  <div :class="['subtitle-item', warningClass]">
    <!-- 时间戳 -->
    <div class="time-range">
      {{ formatTime(subtitle.start) }} - {{ formatTime(subtitle.end) }}
    </div>

    <!-- 主文本 -->
    <div class="text-content">
      <span
        v-for="(word, idx) in subtitle.words"
        :key="idx"
        :class="['word', getWordClass(word)]"
        :title="getWordTooltip(word)"
      >
        {{ word.word }}
      </span>
    </div>

    <!-- 来源标签 -->
    <div class="source-badge" v-if="subtitle.is_modified">
      <span :class="['badge', `badge-${subtitle.source}`]">
        {{ getSourceLabel(subtitle.source) }}
      </span>
    </div>

    <!-- 翻译（如果有） -->
    <div class="translation" v-if="subtitle.translation">
      {{ subtitle.translation }}
    </div>

    <!-- 警告指示器 -->
    <div class="warning-indicator" v-if="hasWarning">
      <span class="warning-icon" :title="warningTooltip">!</span>
    </div>
  </div>
</template>

<script>
export default {
  name: 'SubtitleItem',

  props: {
    subtitle: {
      type: Object,
      required: true
    }
  },

  computed: {
    hasWarning() {
      return this.subtitle.warning_type && this.subtitle.warning_type !== 'none'
    },

    warningClass() {
      const classes = {
        'low_transcription': 'warning-low-confidence',
        'high_perplexity': 'warning-high-perplexity',
        'both': 'warning-critical'
      }
      return classes[this.subtitle.warning_type] || ''
    },

    warningTooltip() {
      const tooltips = {
        'low_transcription': `转录置信度较低 (${(this.subtitle.confidence * 100).toFixed(0)}%)`,
        'high_perplexity': `校对困惑度较高 (${this.subtitle.perplexity?.toFixed(1)})`,
        'both': '转录和校对都存在问题，建议人工复核'
      }
      return tooltips[this.subtitle.warning_type] || ''
    }
  },

  methods: {
    formatTime(seconds) {
      const mins = Math.floor(seconds / 60)
      const secs = Math.floor(seconds % 60)
      const ms = Math.floor((seconds % 1) * 1000)
      return `${mins}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`
    },

    getWordClass(word) {
      if (word.confidence < 0.3) return 'word-critical'
      if (word.confidence < 0.5) return 'word-warning'
      if (word.is_pseudo) return 'word-pseudo'
      return ''
    },

    getWordTooltip(word) {
      const parts = []
      parts.push(`置信度: ${(word.confidence * 100).toFixed(0)}%`)
      if (word.is_pseudo) parts.push('(伪对齐)')
      return parts.join(' ')
    },

    getSourceLabel(source) {
      const labels = {
        'sensevoice': 'SV',
        'whisper_patch': 'W',
        'llm_correction': 'LLM',
        'llm_translation': 'Trans'
      }
      return labels[source] || source
    }
  }
}
</script>

<style scoped>
.subtitle-item {
  position: relative;
  padding: 0.75rem 1rem;
  border-left: 3px solid transparent;
  transition: all 0.2s ease;
}

.subtitle-item:hover {
  background: rgba(0, 0, 0, 0.02);
}

/* 警告样式 */
.warning-low-confidence {
  border-left-color: #f39c12;
  background: rgba(243, 156, 18, 0.05);
}

.warning-high-perplexity {
  border-left-color: #9b59b6;
  background: rgba(155, 89, 182, 0.05);
}

.warning-critical {
  border-left-color: #e74c3c;
  background: rgba(231, 76, 60, 0.08);
}

.time-range {
  font-size: 0.8rem;
  font-family: monospace;
  color: #999;
  margin-bottom: 0.25rem;
}

.text-content {
  font-size: 1rem;
  color: #333;
  line-height: 1.6;
}

/* 字级警告高亮 */
.word-warning {
  background: rgba(243, 156, 18, 0.3);
  border-radius: 2px;
  padding: 0 2px;
}

.word-critical {
  background: rgba(231, 76, 60, 0.4);
  border-radius: 2px;
  padding: 0 2px;
}

.word-pseudo {
  border-bottom: 1px dashed #999;
}

.source-badge {
  display: inline-block;
  margin-left: 0.5rem;
}

.badge {
  font-size: 0.7rem;
  padding: 0.1rem 0.4rem;
  border-radius: 3px;
  font-weight: 500;
}

.badge-whisper_patch {
  background: #3498db;
  color: #fff;
}

.badge-llm_correction {
  background: #9b59b6;
  color: #fff;
}

.translation {
  margin-top: 0.5rem;
  font-size: 0.9rem;
  color: #666;
  font-style: italic;
}

.warning-indicator {
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
}

.warning-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  background: #e74c3c;
  color: #fff;
  border-radius: 50%;
  font-size: 0.7rem;
  font-weight: bold;
  cursor: help;
}
</style>
```

---

## 四、虚拟滚动优化（新增）

**路径**: `frontend/src/components/SubtitleList.vue`

```vue
<template>
  <div class="subtitle-list" ref="listContainer">
    <div class="list-header">
      <span class="header-title">字幕预览 ({{ subtitles.length }} 条)</span>
      <div class="filter-controls">
        <label class="filter-checkbox">
          <input type="checkbox" v-model="showWarningsOnly" />
          仅显示警告
        </label>
      </div>
    </div>

    <!-- 虚拟滚动容器 -->
    <div
      class="virtual-scroll-container"
      :style="{ height: containerHeight + 'px' }"
      @scroll="onScroll"
    >
      <div
        class="virtual-scroll-content"
        :style="{ height: totalHeight + 'px', paddingTop: offsetY + 'px' }"
      >
        <SubtitleItem
          v-for="item in visibleItems"
          :key="item.index"
          :subtitle="item.data"
        />
      </div>
    </div>
  </div>
</template>

<script>
import SubtitleItem from './SubtitleItem.vue'

export default {
  name: 'SubtitleList',

  components: { SubtitleItem },

  props: {
    subtitles: {
      type: Array,
      default: () => []
    }
  },

  data() {
    return {
      showWarningsOnly: false,
      containerHeight: 500,
      itemHeight: 80,  // 估算的每项高度
      scrollTop: 0,
      buffer: 5  // 缓冲项数
    }
  },

  computed: {
    filteredSubtitles() {
      if (!this.showWarningsOnly) return this.subtitles
      return this.subtitles.filter(s => s.warning_type && s.warning_type !== 'none')
    },

    totalHeight() {
      return this.filteredSubtitles.length * this.itemHeight
    },

    startIndex() {
      return Math.max(0, Math.floor(this.scrollTop / this.itemHeight) - this.buffer)
    },

    endIndex() {
      const visibleCount = Math.ceil(this.containerHeight / this.itemHeight)
      return Math.min(
        this.filteredSubtitles.length,
        this.startIndex + visibleCount + this.buffer * 2
      )
    },

    visibleItems() {
      return this.filteredSubtitles
        .slice(this.startIndex, this.endIndex)
        .map((data, i) => ({
          index: this.startIndex + i,
          data
        }))
    },

    offsetY() {
      return this.startIndex * this.itemHeight
    }
  },

  methods: {
    onScroll(e) {
      this.scrollTop = e.target.scrollTop
    }
  }
}
</script>

<style scoped>
.subtitle-list {
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  overflow: hidden;
}

.list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 1rem;
  background: #f5f5f5;
  border-bottom: 1px solid #e0e0e0;
}

.header-title {
  font-weight: 600;
  color: #333;
}

.filter-checkbox {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.85rem;
  color: #666;
  cursor: pointer;
}

.virtual-scroll-container {
  overflow-y: auto;
}

.virtual-scroll-content {
  position: relative;
}
</style>
```

---

## 五、统一 SSE 事件监听（更新）

**修改文件**: `frontend/src/views/TaskDetailView.vue`

```javascript
// SSE 事件监听器设置
setupSSE(jobId) {
  const eventSource = new EventSource(`/api/sse/job:${jobId}`)

  // ========== 进度事件 ==========
  eventSource.addEventListener('progress.overall', (e) => {
    const data = JSON.parse(e.data)
    this.job.progress = data.percent
    this.job.phase = data.phase
    this.job.message = data.message
  })

  // ========== 字幕流式事件 ==========
  eventSource.addEventListener('subtitle.sv_sentence', (e) => {
    const data = JSON.parse(e.data)
    this.subtitles.push(data.sentence)
  })

  eventSource.addEventListener('subtitle.whisper_patch', (e) => {
    const data = JSON.parse(e.data)
    this.updateSubtitle(data.index, data.sentence)
  })

  eventSource.addEventListener('subtitle.llm_proof', (e) => {
    const data = JSON.parse(e.data)
    this.updateSubtitle(data.index, data.sentence)
  })

  eventSource.addEventListener('subtitle.llm_trans', (e) => {
    const data = JSON.parse(e.data)
    this.setTranslation(data.index, data.translation)
  })

  // ========== 信号事件 ==========
  eventSource.addEventListener('signal.job_complete', (e) => {
    this.job.status = 'completed'
    eventSource.close()
  })

  eventSource.addEventListener('signal.job_failed', (e) => {
    const data = JSON.parse(e.data)
    this.job.status = 'failed'
    this.job.error = data.message
    eventSource.close()
  })

  this.eventSource = eventSource
}
```

---

## 六、验收标准

### 基础功能

- [ ] 引擎选择器可正常切换
- [ ] 硬件状态正确显示
- [ ] API 参数正确传递到后端

### 时空解耦架构（新增）

- [ ] 预设选择器正确显示 6 个预设
- [ ] 点击预设正确更新配置
- [ ] 置信度警告高亮正确显示
- [ ] 字级低置信度高亮正确显示
- [ ] 虚拟滚动在大量字幕时不卡顿
- [ ] SSE 事件使用新 Tag 正确接收

---

## 七、下一步

完成 Phase 4（修订版 v2.0）后，进入 [Phase 5: 整合测试](./05_Phase5_整合测试_修订版.md)
