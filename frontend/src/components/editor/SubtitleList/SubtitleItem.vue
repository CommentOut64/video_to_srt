<template>
  <div
    class="subtitle-item"
    :class="itemClasses"
    @click="handleClick"
  >
    <!-- 簇颜色条（时间线模式，搜索激活时显示） -->
    <div
      v-if="clusterColor"
      class="cluster-color-bar"
      :style="{ backgroundColor: clusterColor }"
    />

    <!-- 复选框（搜索模式显示） -->
    <el-checkbox
      v-if="isSelectable"
      :model-value="isSelected"
      size="small"
      class="item-checkbox"
      @click.stop
      @change="handleSelectChange"
    />

    <!-- 序号 -->
    <div class="item-index">{{ index + 1 }}</div>

    <!-- 主内容 -->
    <div class="item-content">
      <!-- 时间行 -->
      <div class="time-row">
        <input
          type="text"
          class="time-input"
          :value="formatTime(subtitle.start)"
          :readonly="subtitle.isDraft"
          @change="e => updateTime('start', parseTime(e.target.value))"
          @focus="e => e.target.select()"
        />
        <span class="time-arrow">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M16.01 11H4v2h12.01v3L20 12l-3.99-4z"/>
          </svg>
        </span>
        <input
          type="text"
          class="time-input"
          :value="formatTime(subtitle.end)"
          :readonly="subtitle.isDraft"
          @change="e => updateTime('end', parseTime(e.target.value))"
          @focus="e => e.target.select()"
        />
        <span class="duration-tag">{{ formatDuration(subtitle.end - subtitle.start) }}</span>

        <!-- 草稿状态指示器 -->
        <span v-if="subtitle.isDraft" class="draft-indicator">
          <span class="spinner"></span>
          <span class="draft-text">生成中</span>
        </span>

        <!-- 置信度徽章 -->
        <!-- V3.1.2+dev.20260111.01: 使用 display_confidence（映射后准确率） -->
        <el-tooltip
          v-if="showConfidenceBadge"
          :content="`准确率: ${displayConfidenceText}（来源: ${confidenceSourceText}）`"
          placement="top"
          :show-after="500"
        >
          <span
            class="confidence-badge"
            :class="confidenceBadgeClass"
          >
            {{ displayConfidenceText }}
          </span>
        </el-tooltip>
      </div>

      <!-- 文本行（三态视图） -->
      <div class="text-row">
        <!-- 状态1: 草稿锁定模式 (isDraft=true) -->
        <div
          v-if="subtitle.isDraft"
          class="text-display text-draft"
        >
          {{ subtitle.text || '...' }}
        </div>

        <!-- 状态2: 只读高亮预览模式 (isDraft=false & 非编辑) -->
        <div
          v-else-if="!isEditing"
          class="text-display text-preview"
          :class="{ 'can-edit': editable }"
          @click.stop="startEditing"
          v-html="renderTextWithHighlight()"
        ></div>

        <!-- 状态3: 编辑模式 (isDraft=false & 编辑中) -->
        <textarea
          v-else
          ref="editTextarea"
          class="text-input"
          :value="subtitle.text"
          @input="e => handleTextInput(e.target.value)"
          @blur="stopEditing"
          @keydown.enter.ctrl="stopEditing"
          @keydown.escape="cancelEditing"
          @contextmenu="handleTextareaContextMenu"
          placeholder="输入字幕文本..."
          rows="2"
        ></textarea>

        <span class="char-count">
          {{ subtitle.text.length }}
        </span>
      </div>

      <!-- 警告提示 -->
      <div v-if="showWarning" class="warning-banner">
        <span class="warning-text">{{ warningMessage }}</span>
      </div>

    </div>

    <!-- 操作按钮 -->
    <div v-if="!subtitle.isDraft" class="item-actions" @click.stop>
      <el-tooltip content="在前面插入" placement="left" :show-after="500">
        <button class="action-btn" @click="$emit('insert-before')">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M7 14l5-5 5 5z"/>
          </svg>
        </button>
      </el-tooltip>
      <el-tooltip content="在后面插入" placement="left" :show-after="500">
        <button class="action-btn" @click="$emit('insert-after')">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M7 10l5 5 5-5z"/>
          </svg>
        </button>
      </el-tooltip>
    </div>

    <!-- 删除按钮 - 绝对定位在右下角 -->
    <el-tooltip
      :content="isDeleteConfirming ? '再次点击确认删除' : '删除'"
      placement="left"
      :show-after="500"
    >
      <button
        v-if="!subtitle.isDraft"
        class="delete-btn"
        :class="{ 'delete-btn-confirming': isDeleteConfirming }"
        @click.stop="handleDelete"
      >
        <!-- 正常状态：X图标 -->
        <svg v-if="!isDeleteConfirming" viewBox="0 0 24 24" fill="currentColor">
          <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
        </svg>
        <!-- 确认状态：垃圾桶图标 -->
        <svg v-else viewBox="0 0 24 24" fill="currentColor">
          <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
        </svg>
      </button>
    </el-tooltip>

    <!-- 右键菜单 -->
    <ContextMenu
      ref="contextMenuRef"
      :items="contextMenuItems"
      @select="handleContextMenuSelect"
      @close="handleContextMenuClose"
    />
  </div>
</template>

<script setup>
/**
 * SubtitleItem - 字幕列表单项组件
 *
 * Phase 5 双模态架构: 实现三态视图
 * 1. 草稿锁定模式: isDraft=true, 灰色斜体, 只读
 * 2. 高亮预览模式: isDraft=false & 非编辑, 显示置信度高亮
 * 3. 编辑模式: isDraft=false & 编辑中, 可编辑文本
 */
import { ref, computed, nextTick, watch } from 'vue'
import { useProjectStore } from '@/stores/projectStore'
import transcriptionApi from '@/services/api/transcriptionApi'
import projectApi from '@/services/api/projectApi'
import ContextMenu from '@/components/editor/ContextMenu.vue'

const props = defineProps({
  subtitle: { type: Object, required: true },
  index: { type: Number, required: true },
  isActive: { type: Boolean, default: false },
  isCurrent: { type: Boolean, default: false },
  editable: { type: Boolean, default: true },
  // 同音搜索相关 props
  matchSpans: { type: Array, default: () => [] },  // 匹配区域 [{start, end, readingKey}]
  isSelected: { type: Boolean, default: false },   // 是否选中（批量替换）
  isSelectable: { type: Boolean, default: false }, // 是否显示复选框
  clusterColor: { type: String, default: null },   // 簇颜色条（时间线模式）
})

const emit = defineEmits([
  'click',
  'update-time',
  'update-text',
  'delete',
  'insert-before',
  'insert-after',
  'select-change',  // 同音搜索：选择变化
])

// Store
const projectStore = useProjectStore()

// 编辑状态
const isEditing = ref(false)
const editTextarea = ref(null)
const originalText = ref('')

// 删除确认状态
const isDeleteConfirming = ref(false)

// 右键菜单状态
const contextMenuRef = ref(null)
const cursorPosition = ref(0)
const isContextMenuOpen = ref(false)  // 防止菜单打开时 blur 触发 stopEditing

// 播放状态
const isPlaying = computed(() => projectStore.player.isPlaying)

// 计算属性
const itemClasses = computed(() => ({
  'is-active': props.isActive,
  'is-current': props.isCurrent,
  'is-current-paused': props.isCurrent && !isPlaying.value,
  'is-draft': props.subtitle.isDraft,
  'warning-low-confidence': props.subtitle.warning_type === 'low_confidence',
  'warning-high-perplexity': props.subtitle.warning_type === 'high_perplexity',
  'warning-both': props.subtitle.warning_type === 'both',
  'is-match-selected': props.isSelectable && props.isSelected,
  'has-cluster-color': !!props.clusterColor,
}))

// 置信度徽章
const showConfidenceBadge = computed(() => {
  return props.subtitle.display_confidence !== undefined && props.subtitle.display_confidence !== null
})

const confidenceBadgeClass = computed(() => {
  const conf = props.subtitle.display_confidence
  if (conf >= 0.85) return 'badge-good'
  if (conf >= 0.68) return 'badge-warning'
  return 'badge-danger'
})

// V3.1.2: 显示的置信度文本（百分比）
const displayConfidenceText = computed(() => {
  const conf = props.subtitle.display_confidence
  if (conf === undefined || conf === null) return ''
  return `${Math.round(conf * 100)}%`
})

// 置信度来源文本
const confidenceSourceText = computed(() => {
  const source = props.subtitle.confidence_source
  if (source === 'whisper') return 'Whisper'
  if (source === 'sensevoice') return 'SenseVoice'
  if (source === 'manual') return '手动编辑'
  if (source === 'srt_fallback') return '导入文件'
  return source || '未知'
})

// 警告信息
const showWarning = computed(() => {
  return props.subtitle.warning_type && props.subtitle.warning_type !== 'none'
})

const warningMessage = computed(() => {
  const type = props.subtitle.warning_type
  const messages = {
    'low_confidence': '低置信度，建议人工审核',
    'high_perplexity': 'LLM 困惑度较高，可能有语法问题',
    'both': '低置信度 + 高困惑度，强烈建议审核'
  }
  return messages[type] || ''
})

// 点击处理
function handleClick() {
  // 点击字幕块时重置删除确认状态
  resetDeleteConfirm()
  emit('click', props.subtitle)
}

// 选择变化处理（同音搜索）
function handleSelectChange(checked) {
  emit('select-change', props.index, checked)
}

// 时间更新
function updateTime(field, value) {
  if (isNaN(value) || props.subtitle.isDraft) return
  emit('update-time', props.subtitle.id, field, value)
}

// 自动调整 textarea 高度
function autoResizeTextarea() {
  if (!editTextarea.value) return
  // 重置高度以获取正确的 scrollHeight
  editTextarea.value.style.height = 'auto'
  // 设置高度为内容高度，但不小于 45px
  const newHeight = Math.max(45, editTextarea.value.scrollHeight)
  editTextarea.value.style.height = `${newHeight}px`
}

// 开始编辑
function startEditing() {
  if (!props.editable || props.subtitle.isDraft) return
  originalText.value = props.subtitle.text
  isEditing.value = true
  nextTick(() => {
    if (editTextarea.value) {
      editTextarea.value.focus()
      editTextarea.value.select()
      // 自动调整高度
      autoResizeTextarea()
    }
  })
}

// 停止编辑
function stopEditing() {
  // 右键菜单打开时不触发停止编辑，防止菜单项消失
  if (isContextMenuOpen.value) {
    return
  }
  isEditing.value = false
}

// 取消编辑（恢复原文）
function cancelEditing() {
  if (originalText.value !== props.subtitle.text) {
    emit('update-text', props.subtitle.id, originalText.value)
  }
  isEditing.value = false
}

// 文本输入处理
function handleTextInput(text) {
  emit('update-text', props.subtitle.id, text)
  // 输入时自动调整高度
  nextTick(() => {
    autoResizeTextarea()
  })
}

// 删除处理
function handleDelete() {
  if (!isDeleteConfirming.value) {
    // 第一次点击：进入确认状态
    isDeleteConfirming.value = true
  } else {
    // 第二次点击：执行删除
    emit('delete', props.subtitle.id)
    isDeleteConfirming.value = false
  }
}

// 重置删除确认状态
function resetDeleteConfirm() {
  isDeleteConfirming.value = false
}

// ============ 右键菜单逻辑 ============

// 右键菜单项配置
const contextMenuItems = computed(() => {
  if (!isEditing.value || props.subtitle.isDraft) {
    return []
  }

  return [{
    key: 'split',
    label: '从此处切分',
  }]
})

// 编辑区域右键事件处理
function handleTextareaContextMenu(e) {
  e.preventDefault()
  e.stopPropagation()

  if (!isEditing.value || props.subtitle.isDraft) {
    return
  }

  // 获取光标位置
  const textarea = editTextarea.value
  if (!textarea) {
    return
  }

  cursorPosition.value = textarea.selectionStart

  // 标记菜单打开，防止 blur 触发 stopEditing
  isContextMenuOpen.value = true

  // 显示右键菜单
  contextMenuRef.value?.show(e.clientX, e.clientY)
}

// 右键菜单项选择处理
async function handleContextMenuSelect(key) {
  // 重置菜单打开标志
  isContextMenuOpen.value = false

  if (key === 'split') {
    const result = projectStore.splitSubtitle(props.subtitle.id, {
      cursorPosition: cursorPosition.value
    })

    if (!result.success) {
      console.error('[SubtitleItem] 切分失败:', result.error)
    } else {
      await syncSplitSubtitles(result)
      // 切分成功后退出编辑模式
      isEditing.value = false
    }
  }
}

async function syncSplitSubtitles(result) {
  const jobId = projectStore.meta.jobId
  const projectId = projectStore.meta.projectId
  if (!jobId && !projectId) return

  const { originalSentenceIndex, leftSubtitle, rightSubtitle } = result
  if (!leftSubtitle || !rightSubtitle) return

  try {
    if (projectId) {
      const leftSegmentId = leftSubtitle.segment_id || props.subtitle.segment_id
      if (leftSegmentId) {
        const updatedLeft = await projectApi.updateSubtitle(projectId, leftSegmentId, {
          text: leftSubtitle.text,
          start: projectStore.toBaseTime(leftSubtitle.start),
          end: projectStore.toBaseTime(leftSubtitle.end)
        })
        projectStore.updateSubtitle(leftSubtitle.id, {
          sentenceIndex: updatedLeft?.legacy_index ?? leftSubtitle.sentenceIndex,
          segment_id: updatedLeft?.segment_id ?? leftSegmentId,
          isModified: true,
          source: updatedLeft?.source_type || 'split'
        }, { isUserEdit: true })
      }

      const rightData = await projectApi.createSubtitle(projectId, {
        text: rightSubtitle.text,
        start: projectStore.toBaseTime(rightSubtitle.start),
        end: projectStore.toBaseTime(rightSubtitle.end)
      })
      projectStore.updateSubtitle(rightSubtitle.id, {
        sentenceIndex: rightData?.legacy_index ?? rightSubtitle.sentenceIndex,
        segment_id: rightData?.segment_id ?? rightSubtitle.segment_id,
        isModified: true,
        source: rightData?.source_type || 'manual'
      }, { isUserEdit: true })
    } else if (jobId) {
      if (originalSentenceIndex !== undefined) {
        await transcriptionApi.updateSubtitle(jobId, originalSentenceIndex, {
          text: leftSubtitle.text,
          start: projectStore.toBaseTime(leftSubtitle.start),
          end: projectStore.toBaseTime(leftSubtitle.end)
        })
        projectStore.updateSubtitle(leftSubtitle.id, {
          sentenceIndex: originalSentenceIndex,
          isModified: true,
          source: 'split'
        }, { isUserEdit: true })
      } else {
        const leftResp = await transcriptionApi.createSubtitle(jobId, {
          text: leftSubtitle.text,
          start: projectStore.toBaseTime(leftSubtitle.start),
          end: projectStore.toBaseTime(leftSubtitle.end)
        })
        const leftData = leftResp?.data?.data || leftResp?.data
        if (leftData?.index !== undefined) {
          projectStore.updateSubtitle(leftSubtitle.id, {
            sentenceIndex: leftData.index,
            isModified: true,
            source: leftData.source || 'manual'
          }, { isUserEdit: true })
        }
      }

      const rightResp = await transcriptionApi.createSubtitle(jobId, {
        text: rightSubtitle.text,
        start: projectStore.toBaseTime(rightSubtitle.start),
        end: projectStore.toBaseTime(rightSubtitle.end)
      })
      const rightData = rightResp?.data?.data || rightResp?.data
      if (rightData?.index !== undefined) {
        projectStore.updateSubtitle(rightSubtitle.id, {
          sentenceIndex: rightData.index,
          isModified: true,
          source: rightData.source || 'manual'
        }, { isUserEdit: true })
      }
    }
  } catch (error) {
    console.warn('[SubtitleItem] 切分同步失败:', error)
  }
}

// 右键菜单关闭处理
function handleContextMenuClose() {
  isContextMenuOpen.value = false
}

// 渲染带置信度高亮的文本
function renderTextWithHighlight() {
  const text = props.subtitle.text

  // 如果有同音搜索匹配区域，优先使用匹配高亮
  if (props.matchSpans && props.matchSpans.length > 0) {
    return renderMatchHighlight(text, props.matchSpans)
  }

  const words = props.subtitle.words

  // 如果没有字级数据，直接返回文本
  if (!words || words.length === 0) {
    return escapeHtml(text)
  }

  // SenseVoice 有时会返回整句作为一个词（常见于中文），此时直接去掉高亮，避免整句着色
  if (words.length === 1) {
    const raw = words[0].word || ''
    const chars = [...raw]
    const isCJKChar = (char) => /[\u4e00-\u9fff]/.test(char)
    const isFullWidthPunc = (char) => /[，。！？、《》【】（）…]/.test(char)
    const shouldBypassHighlight = raw.length > 4 && chars.every(ch => isCJKChar(ch) || isFullWidthPunc(ch))
    if (shouldBypassHighlight) {
      return escapeHtml(text)
    }
  }

  const WARN_THRESHOLD = 0.5
  const CRITICAL_THRESHOLD = 0.3

  // 中文段落常被 SenseVoice 合并成整句，这里在前端拆分为逐字，防止整句高亮
  const expandWords = (wordItem) => {
    const raw = wordItem.word || ''
    const chars = [...raw]
    const isCJKChar = (char) => /[\u4e00-\u9fff]/.test(char)
    const isFullWidthPunc = (char) => /[，。！？、《》【】（）…]/.test(char)
    const shouldSplit = raw.length > 1 && chars.every(ch => isCJKChar(ch) || isFullWidthPunc(ch))
    if (!shouldSplit) {
      return [wordItem]
    }
    return chars.map(ch => ({
      ...wordItem,
      word: ch
    }))
  }

  const processedWords = words.flatMap(word => expandWords(word))

  let html = ''
  for (let i = 0; i < processedWords.length; i++) {
    const word = processedWords[i]
    const rawConf = word.confidence_display_raw ?? word.confidence
    const conf = rawConf !== undefined && rawConf !== null ? rawConf : 1.0
    const wordText = escapeHtml(word.word)

    if (conf < CRITICAL_THRESHOLD) {
      html += `<span class="word-critical">${wordText}</span>`
    } else if (conf < WARN_THRESHOLD) {
      html += `<span class="word-warning">${wordText}</span>`
    } else {
      html += wordText
    }

    // 智能添加空格：英文单词之间加空格，中文字符之间不加
    if (i < processedWords.length - 1) {
      const nextWord = processedWords[i + 1].word
      // 如果当前词或下一词是中文字符，不加空格
      // 如果下一词是标点符号，不加空格
      const isChinese = (char) => char && /[\u4e00-\u9fff]/.test(char)
      const isPunctuation = (char) => char && /[,.!?;:'"()[\]{}，。！？；：""''（）【】《》、]/.test(char)

      if (!isChinese(wordText[wordText.length - 1]) &&
          !isChinese(nextWord[0]) &&
          !isPunctuation(nextWord[0])) {
        html += ' '
      }
    }
  }
  return html
}

/**
 * 渲染同音搜索匹配高亮
 * @param {string} text - 原始文本
 * @param {Array<{start: number, end: number, readingKey?: string}>} spans - 匹配区域
 * @returns {string} 带高亮标记的 HTML
 */
function renderMatchHighlight(text, spans) {
  if (!text || !spans || spans.length === 0) {
    return escapeHtml(text)
  }

  // 按 start 排序并去重
  const sortedSpans = [...spans].sort((a, b) => a.start - b.start)

  let html = ''
  let lastEnd = 0

  for (const span of sortedSpans) {
    const { start, end } = span

    // 跳过无效区域
    if (start < lastEnd || start >= text.length) continue

    // 添加匹配前的普通文本
    if (start > lastEnd) {
      html += escapeHtml(text.slice(lastEnd, start))
    }

    // 添加高亮匹配文本
    const matchText = text.slice(start, Math.min(end, text.length))
    html += `<mark class="match-highlight">${escapeHtml(matchText)}</mark>`

    lastEnd = Math.min(end, text.length)
  }

  // 添加剩余文本
  if (lastEnd < text.length) {
    html += escapeHtml(text.slice(lastEnd))
  }

  return html
}

// HTML 转义
function escapeHtml(text) {
  if (!text) return ''
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

// 时间格式化
function formatTime(seconds) {
  if (isNaN(seconds)) return '00:00.000'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  const ms = Math.round((seconds % 1) * 1000)
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`
}

function parseTime(str) {
  const match = str.match(/(\d+):(\d+)\.?(\d*)/)
  if (!match) return NaN
  const m = parseInt(match[1])
  const s = parseInt(match[2])
  const ms = match[3] ? parseInt(match[3].padEnd(3, '0')) : 0
  return m * 60 + s + ms / 1000
}

function formatDuration(seconds) {
  if (isNaN(seconds) || seconds < 0) return '0.0s'
  return seconds.toFixed(1) + 's'
}
</script>

<style scoped>
/* 字幕项 */
.subtitle-item {
  position: relative;
  display: flex;
  gap: 10px;
  padding: 10px;
  margin-bottom: 6px;
  background: var(--af-bg-secondary);
  border: 1px solid transparent;
  border-radius: var(--af-radius-md);
  transition: all var(--af-transition-fast);
  cursor: pointer;
}

.subtitle-item:hover {
  background: var(--af-bg-tertiary);
}

.subtitle-item.is-active {
  border-color: var(--af-accent-primary);
  background: rgb(var(--af-accent-primary-rgb), 0.8);
}

.subtitle-item.is-current {
  border-color: var(--af-accent-success);
  background: rgb(var(--af-accent-success-rgb), 0.8);
}

/* 暂停时显示呼吸灯动画 */
.subtitle-item.is-current-paused {
  animation: breathing-border-green 3s ease-in-out infinite;
}

/* 草稿状态样式 */
.subtitle-item.is-draft {
  background: rgb(var(--af-text-muted-rgb), 0.5);
  border-color: rgb(var(--af-text-muted-rgb), 0.20);
  cursor: wait;
}

/* 置信度警告高亮样式 */
.subtitle-item.warning-low-confidence {
  border-color: var(--af-accent-warning);
  background: rgb(var(--af-accent-warning-rgb), 0.6);
}

.subtitle-item.warning-high-perplexity {
  border-color: var(--af-status-warning);
  background: rgb(var(--af-status-warning-rgb), 0.6);
}

.subtitle-item.warning-both {
  border-color: var(--af-accent-danger);
  background: rgb(var(--af-accent-danger-rgb), 0.8);
  border-width: 2px;
}

/* 序号 */
.item-index {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 28px;
  height: 28px;
  background: var(--af-bg-tertiary);
  border-radius: var(--af-radius-sm);
  color: var(--af-text-secondary);
  font-size: 11px;
  font-weight: 600;
  flex-shrink: 0;
}

.subtitle-item.is-current .item-index {
  background: var(--af-accent-success);
  color: var(--af-text-on-dark);
}

.subtitle-item.is-draft .item-index {
  background: var(--af-text-muted);
  color: var(--af-text-on-dark);
}

.subtitle-item.warning-low-confidence .item-index {
  background: var(--af-accent-warning);
  color: var(--af-text-on-dark);
}

.subtitle-item.warning-high-perplexity .item-index {
  background: var(--af-status-warning);
  color: var(--af-text-on-dark);
}

.subtitle-item.warning-both .item-index {
  background: var(--af-accent-danger);
  color: var(--af-text-on-dark);
}

/* 内容区 */
.item-content {
  flex: 1;
  min-width: 0;
}

/* 时间行 */
.time-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
}

.time-row .time-input {
  width: 75px;
  padding: 3px 6px;
  background: var(--af-bg-tertiary);
  border: 1px solid transparent;
  border-radius: var(--af-radius-sm);
  color: var(--af-text-normal);
  font-size: 11px;
  font-family: var(--af-font-mono);
  text-align: center;
}

.time-row .time-input:focus {
  border-color: var(--af-accent-primary);
  outline: none;
}

.time-row .time-input[readonly] {
  cursor: wait;
  opacity: 0.6;
}

.time-row .time-arrow {
  color: var(--af-text-muted);
}

.time-row .time-arrow svg {
  width: 14px;
  height: 14px;
}

.time-row .duration-tag {
  padding: 2px 6px;
  background: var(--af-bg-tertiary);
  border-radius: var(--af-radius-full);
  color: var(--af-text-muted);
  font-size: 10px;
  font-family: var(--af-font-mono);
}

/* stylelint-disable-next-line no-descending-specificity -- .delete-btn svg 与 .time-row svg 作用于不同元素 */
.delete-btn svg {
  width: 16px;
  height: 16px;
}

/* 草稿状态指示器 */
.draft-indicator {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  background: rgb(var(--af-draft-bg-rgb), 0.15);
  border-radius: var(--af-radius-full);
  color: var(--af-draft-text);
  font-size: 10px;
}

.draft-indicator .spinner {
  width: 10px;
  height: 10px;
  border: 2px solid rgb(var(--af-text-secondary-rgb), 0.30);
  border-top-color: var(--af-draft-text);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.draft-indicator .draft-text {
  font-style: italic;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* 绿色呼吸灯动画 - 边框颜色从透明到绿色再到透明 */
@keyframes breathing-border-green {
  0%, 100% {
    border-color: transparent;
  }

  50% {
    border-color: var(--af-accent-success);
    box-shadow: 0 0 8px rgb(var(--af-accent-success-rgb), 0.50);
  }
}

/* 置信度徽章 */
.confidence-badge {
  padding: 2px 6px;
  border-radius: var(--af-radius-full);
  font-size: 10px;
  font-family: var(--af-font-mono);
  font-weight: 600;
}

.confidence-badge.badge-good {
  background: rgb(var(--af-accent-success-rgb), 0.15);
  color: var(--af-accent-success);
}

.confidence-badge.badge-warning {
  background: rgb(var(--af-accent-warning-rgb), 0.15);
  color: var(--af-accent-warning);
}

.confidence-badge.badge-danger {
  background: rgb(var(--af-accent-danger-rgb), 0.15);
  color: var(--af-accent-danger);
}

/* 文本行 */
.text-row {
  position: relative;
}

.text-row .text-display {
  width: 100%;
  padding: 6px 35px 6px 8px;
  background: var(--af-bg-tertiary);
  border: 1px solid transparent;
  border-radius: var(--af-radius-sm);
  color: var(--af-text-normal);
  font-size: 12px;
  transition: border-color 0.3s ease, background 0.3s ease;
  min-height: 45px;
  line-height: 1.4;
  white-space: pre-wrap;
  overflow-wrap: break-word;
}

/* 草稿文本样式 */
.text-row .text-draft {
  background: rgb(var(--af-draft-bg-rgb), 0.8);
  color: var(--af-draft-text);
  font-style: italic;
  cursor: wait;
}

/* 预览文本样式 */
.text-row .text-preview {
  cursor: default;
}

.text-row .text-preview.can-edit {
  cursor: text;
}

.text-row .text-preview.can-edit:hover {
  border-color: var(--af-accent-primary);
  background: var(--af-bg-secondary);
}

/* 字级警告高亮样式 */
.text-row .text-preview :deep(.word-warning) {
  background-color: rgb(var(--af-confidence-warning-rgb), 0.25);
  border-bottom: 2px solid var(--af-confidence-warning);
  padding: 0 2px;
  border-radius: 2px;
}

.text-row .text-preview :deep(.word-critical) {
  background-color: rgb(var(--af-confidence-critical-rgb), 0.25);
  border-bottom: 2px solid var(--af-confidence-critical);
  padding: 0 2px;
  border-radius: 2px;
  font-weight: 500;
}

.text-row .text-input {
  width: 100%;
  padding: 6px 8px;
  background: var(--af-bg-tertiary);
  border: 1px solid var(--af-accent-primary);
  border-radius: var(--af-radius-sm);
  color: var(--af-text-normal);
  font-size: 12px;
  min-height: 45px;
  padding-right: 35px;
  resize: none;
  line-height: 1.4;
  outline: none;
  overflow: hidden;
  scrollbar-width: none;
}

.text-row .text-input::placeholder {
  color: var(--af-text-muted);
}

/* 隐藏滚动条 */
.text-row .text-input::-webkit-scrollbar {
  display: none;
}

.text-row .char-count {
  position: absolute;
  right: 6px;
  bottom: 6px;
  color: var(--af-text-muted);
  font-size: 10px;
  font-family: var(--af-font-mono);
}

.subtitle-item.is-draft .text-display {
  color: var(--af-text-secondary);
  font-style: italic;
}

/* 警告横幅 */
.warning-banner {
  margin-top: 6px;
  padding: 4px 8px;
  background: rgb(var(--af-accent-warning-rgb), 0.10);
  border-left: 3px solid var(--af-accent-warning);
  border-radius: var(--af-radius-sm);
}

.warning-banner .warning-text {
  color: var(--af-accent-warning);
  font-size: 11px;
}

/* 操作按钮 */
.item-actions {
  display: flex;
  flex-direction: column;
  gap: 2px;
  opacity: 0.5;
  transition: opacity var(--af-transition-fast);
}

.item-actions .action-btn {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 24px;
  height: 24px;
  background: transparent;
  border: none;
  border-radius: var(--af-radius-sm);
  color: var(--af-text-muted);
  transition: all var(--af-transition-fast);
  cursor: pointer;
}

.item-actions .action-btn svg {
  width: 14px;
  height: 14px;
}

.item-actions .action-btn:hover {
  background: var(--af-bg-tertiary);
  color: var(--af-text-normal);
}

.item-actions .action-btn-danger:hover {
  background: rgb(var(--af-accent-danger-rgb), 0.15);
  color: var(--af-accent-danger);
}

.subtitle-item:hover .item-actions {
  opacity: 1;
}

/* 删除按钮 - 绝对定位在右下角，与操作按钮垂直对齐 */
.delete-btn {
  position: absolute;
  right: 12px;
  bottom: 6px;
  display: flex;
  justify-content: center;
  align-items: center;
  width: 20px;
  height: 20px;
  background: transparent;
  border: none;
  border-radius: var(--af-radius-sm);
  color: var(--af-text-muted);
  transition: all 0.2s ease;
  cursor: pointer;
  opacity: 0.5;
}

.delete-btn:hover {
  opacity: 1;
  background: var(--af-bg-tertiary);
}

/* 确认状态 - 红色垃圾桶图标 */
.delete-btn-confirming {
  color: var(--af-accent-danger);
  opacity: 1;
}

.delete-btn-confirming:hover {
  background: rgb(var(--af-accent-danger-rgb), 0.10);
}

/* 同音搜索相关样式 */

/* 簇颜色条 */
.cluster-color-bar {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 4px;
  border-radius: var(--af-radius-md) 0 0 var(--af-radius-md);
}

/* 复选框 */
.item-checkbox {
  flex-shrink: 0;
  margin-right: 4px;
}

/* 匹配高亮样式 */
.text-row .text-preview :deep(.match-highlight) {
  background-color: rgb(var(--af-accent-warning-rgb), 0.35);
  color: var(--af-text-primary);
  padding: 1px 2px;
  border-radius: 2px;
  font-weight: 500;
}

/* 选中状态的匹配项 */
.subtitle-item.is-match-selected {
  border-color: var(--af-accent-primary);
  background: rgb(var(--af-accent-primary-rgb), 0.05);
}

/* 有簇颜色条时增加左侧内边距 */
.subtitle-item.has-cluster-color {
  padding-left: 14px;
}

.subtitle-item.is-match-selected .item-index {
  background: var(--af-accent-primary);
  color: var(--af-text-on-dark);
}
</style>
