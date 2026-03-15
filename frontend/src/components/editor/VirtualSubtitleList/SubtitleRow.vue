<!-- V3.2.5+dev.20260315.21: SubtitleRow - 虚拟列表行组件（移除行级 memo 缓存） -->
<template>
  <div
    class="subtitle-row"
    :class="{
      'is-selected': isSelected,
      'is-active': isActive,
      'is-current': isCurrent,
      'is-draft': entity?.isDraft,
      'is-modified': entity?.isModified
    }"
    @click="handleClick"
  >
    <!-- 序号 -->
    <div class="item-index">{{ rowIndex + 1 }}</div>

    <!-- 主内容 -->
    <div class="item-content">
      <!-- 时间行 -->
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

        <!-- 草稿状态指示器 -->
        <span v-if="entity?.isDraft" class="draft-indicator">
          <span class="spinner"></span>
          <span class="draft-text">生成中</span>
        </span>
      </div>

      <!-- 文本行 -->
      <div class="text-row">
        <div
          v-if="!isEditing"
          class="text-display"
          :class="{ 'can-edit': !entity?.isDraft }"
          @click.stop="startEditing"
        >
          {{ displayText }}
        </div>
        <textarea
          v-else
          ref="textareaRef"
          class="text-input"
          :value="editingText"
          @input="editingText = $event.target.value"
          @blur="stopEditing"
          @keydown.enter.ctrl="stopEditing"
          @keydown.escape="cancelEditing"
          placeholder="输入字幕文本..."
        />
        <span class="char-count">{{ displayText.length }}</span>
      </div>
    </div>

    <!-- 操作按钮 -->
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

    <!-- 删除按钮 -->
    <button
      v-if="!entity?.isDraft"
      class="delete-btn"
      @click.stop="$emit('delete')"
    >
      <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
      </svg>
    </button>
  </div>
</template>

<script setup>
// V3.2.5+dev.20260315.21: 行级 selector，依赖不可变实体替换触发刷新
import { computed, ref, nextTick } from 'vue'
import { useEditorDocumentStore } from '@/stores/editor/editorDocumentStore'
import { useEditorCommandBus } from '@/stores/editor/editorCommandBus'
import { useEditorDraftStore } from '@/stores/editor/editorDraftStore'
import { useProjectStore } from '@/stores/projectStore'
import { createUpdateTextCommand, createUpdateTimingCommand } from '@/stores/editor/editorCommandFactory'

const props = defineProps({
  localId: { type: String, required: true },
  rowIndex: { type: Number, required: true },
  isSelected: { type: Boolean, default: false },
  isActive: { type: Boolean, default: false },
  isCurrent: { type: Boolean, default: false }
})

const emit = defineEmits(['click', 'insert-before', 'insert-after', 'delete'])

const docStore = useEditorDocumentStore()
const commandBus = useEditorCommandBus()
const draftStore = useEditorDraftStore()
const projectStore = useProjectStore()

// 行级 selector：只订阅当前行的版本令牌
const entity = computed(() => {
  const token = docStore.entityVersionTokens.get(props.localId)
  if (token) {
    token.value // 建立响应式依赖
  }
  return docStore.entities.get(props.localId)
})

// 草稿优先显示
const displayText = computed(() => {
  const draft = draftStore.activeTextDraft
  if (draft?.localId === props.localId) {
    return draft.text
  }
  return entity.value?.text || ''
})

// 编辑状态
const isEditing = ref(false)
const editingText = ref('')
const textareaRef = ref(null)

function startEditing() {
  if (entity.value?.isDraft) return
  isEditing.value = true
  editingText.value = displayText.value
  nextTick(() => {
    textareaRef.value?.focus()
  })
}

function stopEditing() {
  if (!isEditing.value) return
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
}

function cancelEditing() {
  isEditing.value = false
  editingText.value = ''
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
  return (ms / 1000).toFixed(1) + 's'
}

function parseTime(timeStr) {
  const match = timeStr.match(/(\d+):(\d+)\.(\d+)/)
  if (!match) return null
  const [, minutes, seconds, milliseconds] = match
  return parseInt(minutes) * 60000 + parseInt(seconds) * 1000 + parseInt(milliseconds)
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

function handleClick() {
  emit('click', props.localId)
}
</script>

<style scoped>
/* 字幕行 */
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

.subtitle-row.is-active:not(.is-current) {
  border-color: var(--af-accent-primary);
  background: rgb(var(--af-accent-primary-rgb), 0.08);
}

.subtitle-row.is-current {
  border-color: var(--af-accent-success);
  background: rgb(var(--af-accent-success-rgb), 0.08);
}

.subtitle-row.is-draft {
  background: rgb(var(--af-text-muted-rgb), 0.05);
  border-color: rgb(var(--af-text-muted-rgb), 0.20);
  cursor: wait;
  opacity: 0.6;
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

.subtitle-row.is-draft .item-index {
  background: var(--af-text-muted);
  color: var(--af-text-on-dark);
}

.subtitle-row.is-current .item-index {
  background: var(--af-accent-success);
  color: var(--af-text-inverse);
}

.subtitle-row.is-active:not(.is-current) .item-index {
  background: var(--af-accent-primary);
  color: var(--af-text-inverse);
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

/* 文本行 */
.text-row {
  position: relative;
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
  position: absolute;
  right: 6px;
  bottom: 6px;
  color: var(--af-text-muted);
  font-size: 10px;
  font-family: var(--af-font-mono);
}

.subtitle-row.is-draft .text-display {
  color: var(--af-text-secondary);
  font-style: italic;
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

/* 删除按钮 */
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
