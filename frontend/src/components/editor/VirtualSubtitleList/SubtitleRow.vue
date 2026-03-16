<!-- V3.2.5+dev.20260316.01: SubtitleRow - 接入搜索高亮、warning 与词级高亮 -->
<template>
  <div
    class="subtitle-row"
    :class="{
      'is-selected': isSelected,
      'is-active': isActive,
      'is-current': isCurrent,
      'is-draft': entity?.isDraft,
      'is-modified': entity?.isModified,
      'is-match-selected': isMatchSelected,
      'has-cluster-color': Boolean(clusterColor),
    }"
    @click="handleClick"
    @contextmenu.prevent="handleItemContextMenu"
  >
    <div
      v-if="clusterColor"
      class="cluster-color-bar"
      :style="{ backgroundColor: clusterColor }"
    />

    <el-checkbox
      v-if="isSelectable"
      :model-value="isMatchSelected"
      size="small"
      class="item-checkbox"
      @click.stop
      @change="handleSelectChange"
    />

    <div class="item-index">{{ rowIndex + 1 }}</div>

    <div class="item-content">
      <div class="time-row">
        <input
          type="text"
          class="time-input"
          :value="formatTime(entity?.startMs)"
          :readonly="entity?.isDraft"
          @blur="handleTimeUpdate('start', $event.target.value)"
          @keydown.enter="$event.target.blur()"
        />
        <span class="time-arrow">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M16.01 11H4v2h12.01v3L20 12l-3.99-4z"/>
          </svg>
        </span>
        <input
          type="text"
          class="time-input"
          :value="formatTime(entity?.endMs)"
          :readonly="entity?.isDraft"
          @blur="handleTimeUpdate('end', $event.target.value)"
          @keydown.enter="$event.target.blur()"
        />
        <span class="duration-tag">{{ formatDuration(entity?.endMs - entity?.startMs) }}</span>

        <span v-if="entity?.isDraft" class="draft-indicator">
          <span class="spinner"></span>
          <span class="draft-text">生成中</span>
        </span>

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

      <div class="text-row">
        <div
          v-if="!isEditing"
          class="text-display"
          :class="{ 'can-edit': !entity?.isDraft }"
          @click.stop="startEditing($event)"
          v-html="renderTextWithHighlight()"
        ></div>
        <textarea
          v-else
          ref="textareaRef"
          class="text-input"
          :value="editingText"
          @input="handleTextareaInput"
          @click="updateCursorPosition"
          @keyup="updateCursorPosition"
          @select="updateCursorPosition"
          @contextmenu.prevent.stop="handleTextareaContextMenu"
          @blur="stopEditing"
          @keydown.enter.ctrl="stopEditing"
          @keydown.escape="cancelEditing"
          placeholder="输入字幕文本..."
          rows="2"
        />
        <div class="text-meta">
          <span class="char-count">{{ (isEditing ? editingText : displayText).length }}</span>
        </div>
      </div>

      <div v-if="showWarning" class="warning-banner">
        <span class="warning-text">{{ warningMessage }}</span>
      </div>
    </div>

    <div v-if="!entity?.isDraft" class="item-actions" @click.stop>
      <button class="action-btn" @click="$emit('insert-before')">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M7 14l5-5 5 5z"/>
        </svg>
      </button>
      <button class="action-btn" @click="$emit('insert-after')">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M7 10l5 5 5-5z"/>
        </svg>
      </button>
    </div>

    <button
      v-if="!entity?.isDraft"
      class="delete-btn"
      @click.stop="$emit('delete')"
    >
      <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
      </svg>
    </button>

    <ContextMenu
      ref="contextMenuRef"
      :items="contextMenuItems"
      @select="handleContextMenuSelect"
      @close="handleContextMenuClose"
    />
  </div>
</template>

<script setup>
import { computed, nextTick, ref } from 'vue'
import ContextMenu from '@/components/editor/ContextMenu.vue'
import { useEditorCommandBus } from '@/stores/editor/editorCommandBus'
import { useEditorDocumentStore } from '@/stores/editor/editorDocumentStore'
import { useEditorDraftStore } from '@/stores/editor/editorDraftStore'
import { useEditorProjectionBridge } from '@/stores/editor/editorProjectionBridge'
import { createUpdateTextCommand, createUpdateTimingCommand } from '@/stores/editor/editorCommandFactory'
import { useProjectStore } from '@/stores/projectStore'

const props = defineProps({
  localId: { type: String, required: true },
  rowIndex: { type: Number, required: true },
  isSelected: { type: Boolean, default: false },
  isActive: { type: Boolean, default: false },
  isCurrent: { type: Boolean, default: false },
  matchSpans: { type: Array, default: () => [] },
  isMatchSelected: { type: Boolean, default: false },
  isSelectable: { type: Boolean, default: false },
  clusterColor: { type: String, default: null },
  canMergePrev: { type: Boolean, default: false },
  canMergeNext: { type: Boolean, default: false },
})

const emit = defineEmits([
  'click',
  'insert-before',
  'insert-after',
  'delete',
  'select-change',
  'split',
  'merge-prev',
  'merge-next',
])

const commandBus = useEditorCommandBus()
const docStore = useEditorDocumentStore()
const draftStore = useEditorDraftStore()
const editorProjectionBridge = useEditorProjectionBridge()
const projectStore = useProjectStore()

const entity = computed(() => {
  const token = docStore.entityVersionTokens.get(props.localId)
  if (token) {
    token.value
  }
  return docStore.entities.get(props.localId)
})

const subtitle = computed(() => editorProjectionBridge.findSubtitleById(props.localId))

const displayText = computed(() => {
  const draft = draftStore.activeTextDraft
  if (draft?.localId === props.localId) {
    return draft.text
  }
  return entity.value?.text || ''
})

const isEditing = ref(false)
const originalText = ref('')
const editingText = ref('')
const textareaRef = ref(null)
const cursorPosition = ref(null)
const contextMenuRef = ref(null)
const isContextMenuOpen = ref(false)
const pendingBlurWhileContextMenuOpen = ref(false)
let highlightCacheText = null
let highlightCacheWords = null
let highlightCacheMatchSpans = null
let highlightCacheHtml = ''

const canSplitAtCursor = computed(() => {
  return isEditing.value
    && Number.isInteger(cursorPosition.value)
    && cursorPosition.value > 0
    && cursorPosition.value < editingText.value.length
})

const contextMenuItems = computed(() => {
  if (entity.value?.isDraft) {
    return []
  }

  const items = []
  if (isEditing.value) {
    items.push({
      key: 'split',
      label: '从此处切分',
    })
  }
  items.push({
    key: 'merge-prev',
    label: '与前字幕合并',
    disabled: !props.canMergePrev,
  })
  items.push({
    key: 'merge-next',
    label: '与后字幕合并',
    disabled: !props.canMergeNext,
  })
  return items.filter((item) => !item.disabled)
})

const showConfidenceBadge = computed(() => {
  return subtitle.value?.display_confidence !== undefined && subtitle.value?.display_confidence !== null
})

const confidenceBadgeClass = computed(() => {
  const conf = subtitle.value?.display_confidence
  if (conf >= 0.85) return 'badge-good'
  if (conf >= 0.68) return 'badge-warning'
  return 'badge-danger'
})

const displayConfidenceText = computed(() => {
  const conf = subtitle.value?.display_confidence
  if (conf === undefined || conf === null) return ''
  return `${Math.round(conf * 100)}%`
})

const confidenceSourceText = computed(() => {
  const source = subtitle.value?.confidence_source
  if (source === 'whisper') return 'Whisper'
  if (source === 'sensevoice') return 'SenseVoice'
  if (source === 'manual') return '手动编辑'
  if (source === 'srt_fallback') return '导入文件'
  return source || '未知'
})

const showWarning = computed(() => {
  return subtitle.value?.warning_type && subtitle.value.warning_type !== 'none'
})

const warningMessage = computed(() => {
  const messages = {
    low_confidence: '低置信度，建议人工审核',
    high_perplexity: 'LLM 困惑度较高，可能有语法问题',
    both: '低置信度 + 高困惑度，强烈建议审核',
  }
  return messages[subtitle.value?.warning_type] || ''
})

function startEditing(event) {
  if (entity.value?.isDraft) return
  originalText.value = displayText.value
  isEditing.value = true
  editingText.value = displayText.value
  let clickOffset = -1
  if (event) {
    clickOffset = getTextOffsetFromPoint(event.currentTarget, event.clientX, event.clientY)
  }
  nextTick(() => {
    const textarea = textareaRef.value
    textarea?.focus()
    const cursor = clickOffset >= 0 ? clickOffset : editingText.value.length
    if (textarea && typeof textarea.setSelectionRange === 'function') {
      textarea.setSelectionRange(cursor, cursor)
    }
    cursorPosition.value = cursor
    autoResizeTextarea()
  })
}

function stopEditing() {
  if (!isEditing.value) return
  if (isContextMenuOpen.value) {
    pendingBlurWhileContextMenuOpen.value = true
    return
  }
  const newText = editingText.value.trim()
  if (newText && newText !== entity.value?.text) {
    commandBus.dispatch(createUpdateTextCommand({
      localId: props.localId,
      before: { text: entity.value?.text },
      after: { text: newText },
      source: 'user',
    }))
  }
  isEditing.value = false
  cursorPosition.value = null
  pendingBlurWhileContextMenuOpen.value = false
}

function cancelEditing() {
  isEditing.value = false
  editingText.value = originalText.value
  cursorPosition.value = null
  pendingBlurWhileContextMenuOpen.value = false
}

function formatTime(ms) {
  if (!ms && ms !== 0) return '00:00.000'
  const totalSeconds = projectStore.toDisplayTime(ms / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = Math.floor(totalSeconds % 60)
  const milliseconds = Math.round((totalSeconds % 1) * 1000)
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(milliseconds).padStart(3, '0')}`
}

function formatDuration(ms) {
  if (!ms || ms < 0) return '0.0s'
  return `${(ms / 1000).toFixed(1)}s`
}

function parseTime(timeStr) {
  const match = timeStr.match(/(\d+):(\d+)\.(\d+)/)
  if (!match) return null
  const [, minutes, seconds, milliseconds] = match
  return parseInt(minutes, 10) * 60000 + parseInt(seconds, 10) * 1000 + parseInt(milliseconds, 10)
}

function handleTimeUpdate(type, value) {
  const displayMs = parseTime(value)
  if (displayMs === null) return

  const newMs = Math.round(projectStore.toBaseTime(displayMs / 1000) * 1000)
  const before = { startMs: entity.value?.startMs, endMs: entity.value?.endMs }
  const after = { ...before }
  if (type === 'start') after.startMs = newMs
  else after.endMs = newMs

  if (after.startMs === before.startMs && after.endMs === before.endMs) {
    return
  }

  commandBus.dispatch(createUpdateTimingCommand({
    localId: props.localId,
    before,
    after,
    source: 'user',
  }))
}

function updateCursorPosition(event) {
  const textarea = event?.target || textareaRef.value
  if (!textarea || typeof textarea.selectionStart !== 'number') {
    cursorPosition.value = null
    return
  }
  cursorPosition.value = textarea.selectionStart
}

function handleClick(event) {
  emit('click', props.localId, event)
}

function handleSelectChange(checked) {
  emit('select-change', checked)
}

function autoResizeTextarea() {
  const textarea = textareaRef.value
  if (!textarea) return
  textarea.style.height = 'auto'
  textarea.style.height = `${Math.max(45, textarea.scrollHeight)}px`
}

function getTextOffsetFromPoint(container, clientX, clientY) {
  const range = document.caretRangeFromPoint?.(clientX, clientY)
  if (!range) return -1

  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT)
  let offset = 0
  let node
  while ((node = walker.nextNode())) {
    if (node === range.startContainer) {
      return offset + range.startOffset
    }
    offset += node.textContent.length
  }
  return -1
}

function handleTextareaInput(event) {
  editingText.value = event.target.value
  updateCursorPosition(event)
  nextTick(() => {
    autoResizeTextarea()
  })
}

function handleTextareaContextMenu(event) {
  if (!isEditing.value || entity.value?.isDraft || !contextMenuItems.value.length) {
    return
  }
  event.preventDefault()
  event.stopPropagation()
  updateCursorPosition(event)
  isContextMenuOpen.value = true
  pendingBlurWhileContextMenuOpen.value = false
  contextMenuRef.value?.show(event.clientX, event.clientY)
}

function handleItemContextMenu(event) {
  if (entity.value?.isDraft || !contextMenuItems.value.length) {
    return
  }
  event.preventDefault()
  event.stopPropagation()
  isContextMenuOpen.value = true
  pendingBlurWhileContextMenuOpen.value = false
  contextMenuRef.value?.show(event.clientX, event.clientY)
}

function handleContextMenuSelect(key) {
  const hasDraftOverride = isEditing.value
  const draftText = hasDraftOverride ? editingText.value : null

  isContextMenuOpen.value = false
  pendingBlurWhileContextMenuOpen.value = false
  if (key === 'split') {
    if (!canSplitAtCursor.value) {
      return
    }
    isEditing.value = false
    emit('split', {
      localId: props.localId,
      cursorPosition: cursorPosition.value,
      text: editingText.value,
    })
    cursorPosition.value = null
    return
  }
  if (key === 'merge-prev') {
    isEditing.value = false
    cursorPosition.value = null
    emit('merge-prev', {
      localId: props.localId,
      draftText,
      hasDraftOverride,
      originalText: originalText.value,
    })
    return
  }
  if (key === 'merge-next') {
    isEditing.value = false
    cursorPosition.value = null
    emit('merge-next', {
      localId: props.localId,
      draftText,
      hasDraftOverride,
      originalText: originalText.value,
    })
  }
}

function handleContextMenuClose() {
  isContextMenuOpen.value = false
  if (!pendingBlurWhileContextMenuOpen.value) {
    return
  }
  pendingBlurWhileContextMenuOpen.value = false
  nextTick(() => {
    stopEditing()
  })
}

function renderTextWithHighlight() {
  const text = displayText.value || ''
  const matchSpans = props.matchSpans
  const words = subtitle.value?.words

  if (
    highlightCacheText === text
    && highlightCacheWords === words
    && highlightCacheMatchSpans === matchSpans
  ) {
    return highlightCacheHtml
  }

  let html = ''

  if (matchSpans && matchSpans.length > 0) {
    html = renderMatchHighlight(text, matchSpans)
    highlightCacheText = text
    highlightCacheWords = words
    highlightCacheMatchSpans = matchSpans
    highlightCacheHtml = html
    return html
  }

  if (!words || words.length === 0) {
    html = escapeHtml(text)
    highlightCacheText = text
    highlightCacheWords = words
    highlightCacheMatchSpans = matchSpans
    highlightCacheHtml = html
    return html
  }

  if (words.length === 1) {
    const raw = words[0].word || ''
    const chars = [...raw]
    const isCjkChar = (char) => /[\u4e00-\u9fff]/.test(char)
    const isFullWidthPunctuation = (char) => /[，。！？、《》【】（）…]/.test(char)
    const shouldBypassHighlight = raw.length > 4 && chars.every(
      (char) => isCjkChar(char) || isFullWidthPunctuation(char)
    )
    if (shouldBypassHighlight) {
      html = escapeHtml(text)
      highlightCacheText = text
      highlightCacheWords = words
      highlightCacheMatchSpans = matchSpans
      highlightCacheHtml = html
      return html
    }
  }

  const warnThreshold = 0.5
  const criticalThreshold = 0.3
  const expandWords = (wordItem) => {
    const raw = wordItem.word || ''
    const chars = [...raw]
    const isCjkChar = (char) => /[\u4e00-\u9fff]/.test(char)
    const isFullWidthPunctuation = (char) => /[，。！？、《》【】（）…]/.test(char)
    const shouldSplit = raw.length > 1 && chars.every(
      (char) => isCjkChar(char) || isFullWidthPunctuation(char)
    )
    if (!shouldSplit) {
      return [wordItem]
    }
    return chars.map((char) => ({
      ...wordItem,
      word: char,
    }))
  }

  const processedWords = words.flatMap((word) => expandWords(word))
  html = ''
  for (let index = 0; index < processedWords.length; index += 1) {
    const word = processedWords[index]
    const rawConfidence = word.confidence_display_raw ?? word.confidence
    const confidence = rawConfidence !== undefined && rawConfidence !== null ? rawConfidence : 1.0
    const wordText = escapeHtml(word.word)

    if (confidence < criticalThreshold) {
      html += `<span class="word-critical">${wordText}</span>`
    } else if (confidence < warnThreshold) {
      html += `<span class="word-warning">${wordText}</span>`
    } else {
      html += wordText
    }

    if (index < processedWords.length - 1) {
      const nextWord = processedWords[index + 1].word
      const isChinese = (char) => char && /[\u4e00-\u9fff]/.test(char)
      const isPunctuation = (char) => char && /[,.!?;:'"()[\]{}，。！？；：""''（）【】《》、]/.test(char)
      if (!isChinese(word.word?.slice(-1)) && !isChinese(nextWord?.[0]) && !isPunctuation(nextWord?.[0])) {
        html += ' '
      }
    }
  }

  highlightCacheText = text
  highlightCacheWords = words
  highlightCacheMatchSpans = matchSpans
  highlightCacheHtml = html
  return html
}

function renderMatchHighlight(text, spans) {
  if (!text || !spans || spans.length === 0) {
    return escapeHtml(text)
  }

  const sortedSpans = [...spans].sort((left, right) => left.start - right.start)
  let html = ''
  let lastEnd = 0

  for (const span of sortedSpans) {
    const { start, end } = span
    if (start < lastEnd || start >= text.length) continue

    if (start > lastEnd) {
      html += escapeHtml(text.slice(lastEnd, start))
    }

    const matchText = text.slice(start, Math.min(end, text.length))
    html += `<mark class="match-highlight">${escapeHtml(matchText)}</mark>`
    lastEnd = Math.min(end, text.length)
  }

  if (lastEnd < text.length) {
    html += escapeHtml(text.slice(lastEnd))
  }

  return html
}

function escapeHtml(text) {
  if (!text) return ''
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}
</script>

<style scoped>
.subtitle-row {
  position: relative;
  display: flex;
  gap: 10px;
  padding: 10px;
  background: var(--af-bg-secondary);
  border: 1px solid transparent;
  border-radius: var(--af-radius-md);
  transition:
    background-color var(--af-transition-fast),
    border-color var(--af-transition-fast),
    opacity var(--af-transition-fast);
  cursor: pointer;
}

.subtitle-row:hover {
  background: var(--af-bg-tertiary);
}

.subtitle-row.has-cluster-color {
  padding-left: 14px;
}

.cluster-color-bar {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 4px;
  border-radius: var(--af-radius-md) 0 0 var(--af-radius-md);
}

.item-checkbox {
  flex-shrink: 0;
  margin-right: 4px;
}

.subtitle-row.is-selected:not(.is-active):not(.is-current) {
  border-color: rgb(var(--af-accent-primary-rgb), 0.35);
  background: rgb(var(--af-accent-primary-rgb), 0.04);
}

.subtitle-row.is-active:not(.is-current) {
  border-color: var(--af-accent-primary);
  background: rgb(var(--af-accent-primary-rgb), 0.08);
}

.subtitle-row.is-current {
  border-color: var(--af-accent-success);
  background: rgb(var(--af-accent-success-rgb), 0.08);
}

.subtitle-row.is-match-selected:not(.is-active):not(.is-current) {
  border-color: rgb(var(--af-accent-primary-rgb), 0.35);
  background: rgb(var(--af-accent-primary-rgb), 0.05);
}

.subtitle-row.is-draft {
  background: rgb(var(--af-text-muted-rgb), 0.05);
  border-color: rgb(var(--af-text-muted-rgb), 0.20);
  cursor: wait;
  opacity: 0.6;
}

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

.subtitle-row.is-draft .item-index {
  background: var(--af-text-muted);
  color: var(--af-text-inverse);
}

.subtitle-row.is-current .item-index {
  background: var(--af-accent-success);
  color: var(--af-text-inverse);
}

.subtitle-row.is-selected:not(.is-active):not(.is-current) .item-index,
.subtitle-row.is-match-selected:not(.is-active):not(.is-current) .item-index {
  background: rgb(var(--af-accent-primary-rgb), 0.16);
  color: var(--af-accent-primary);
}

.subtitle-row.is-active:not(.is-current) .item-index {
  background: var(--af-accent-primary);
  color: var(--af-text-inverse);
}

.item-content {
  flex: 1;
  min-width: 0;
}

.time-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
}

.time-input {
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

.time-input:focus {
  border-color: var(--af-accent-primary);
  outline: none;
}

.time-input[readonly] {
  cursor: wait;
  opacity: 0.6;
}

.time-arrow {
  color: var(--af-text-muted);
}

.time-arrow svg {
  width: 14px;
  height: 14px;
}

.duration-tag {
  padding: 2px 6px;
  background: var(--af-bg-tertiary);
  border-radius: var(--af-radius-full);
  color: var(--af-text-muted);
  font-size: 10px;
  font-family: var(--af-font-mono);
}

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

.text-row {
  position: relative;
}

.text-meta {
  position: absolute;
  right: 6px;
  bottom: 6px;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
}

.text-display {
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
  cursor: default;
}

.text-display.can-edit {
  cursor: text;
}

.text-display.can-edit:hover {
  border-color: var(--af-accent-primary);
  background: var(--af-bg-secondary);
}

.text-display :deep(.word-warning) {
  background-color: rgb(var(--af-confidence-warning-rgb), 0.25);
  border-bottom: 2px solid var(--af-confidence-warning);
  padding: 0 2px;
  border-radius: 2px;
}

.text-display :deep(.word-critical) {
  background-color: rgb(var(--af-confidence-critical-rgb), 0.25);
  border-bottom: 2px solid var(--af-confidence-critical);
  padding: 0 2px;
  border-radius: 2px;
  font-weight: 500;
}

.text-display :deep(.match-highlight) {
  background-color: rgb(var(--af-accent-warning-rgb), 0.35);
  color: var(--af-text-primary);
  padding: 1px 2px;
  border-radius: 2px;
  font-weight: 500;
}

.text-input {
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

.text-input::placeholder {
  color: var(--af-text-muted);
}

.text-input::-webkit-scrollbar {
  display: none;
}

.char-count {
  color: var(--af-text-muted);
  font-size: 10px;
  font-family: var(--af-font-mono);
}

.subtitle-row.is-draft .text-display {
  color: var(--af-text-secondary);
  font-style: italic;
}

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
  transition:
    background-color var(--af-transition-fast),
    color var(--af-transition-fast);
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

.subtitle-row:hover .item-actions {
  opacity: 1;
}

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
  transition:
    background-color 0.2s ease,
    opacity 0.2s ease;
  cursor: pointer;
  opacity: 0.5;
}

.delete-btn svg {
  width: 16px;
  height: 16px;
}

.delete-btn:hover {
  opacity: 1;
  background: var(--af-bg-tertiary);
}
</style>
