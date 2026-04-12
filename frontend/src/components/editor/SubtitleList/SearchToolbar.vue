<template>
  <div
    class="list-toolbar tw-flex tw-items-center tw-px-3 tw-py-2 tw-bg-bg-secondary tw-border-b tw-border-border tw-gap-1.5"
  >
    <!-- 左侧：字幕计数 -->
    <div class="toolbar-left tw-flex tw-items-center tw-flex-shrink-0">
      <span class="tw-text-xs tw-text-text-secondary tw-whitespace-nowrap">
        {{ totalSubtitles }} 条字幕
      </span>
      <span v-if="draftCount > 0" class="tw-text-xs tw-text-accent-warning tw-ml-1">
        ({{ draftCount }} 草稿)
      </span>
      <span
        v-if="isSearchActive && matchCount > 0"
        class="tw-text-xs tw-text-accent-primary tw-ml-2"
      >
        {{ matchCount }} 匹配
      </span>
    </div>

    <!-- 中间：搜索模式按钮 + 悬浮搜索/替换面板，整体居中 -->
    <div
      class="toolbar-center tw-flex tw-items-center tw-justify-center tw-flex-1 tw-min-w-0 tw-mx-1"
    >
      <!-- 居中组：模式按钮 + 搜索面板紧贴在一起 -->
      <div class="search-group tw-flex tw-items-center tw-gap-1 tw-max-w-[268px] tw-w-full">
        <!-- 搜索模式切换按钮 -->
        <el-popover
          ref="modePopoverRef"
          placement="bottom"
          :width="60"
          trigger="click"
          :show-arrow="false"
          popper-class="mode-popover-custom"
        >
          <template #reference>
            <button
              class="toolbar-btn mode-btn tw-flex-shrink-0"
              :class="{ 'is-active': !isLiteralMode }"
            >
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path
                  d="M3 17v2h6v-2H3zM3 5v2h10V5H3zm10 16v-2h8v-2h-8v-2h-2v6h2zM7 9v2H3v2h4v2h2V9H7zm14 4v-2H11v2h10zm-6-4h2V7h4V5h-4V3h-2v6z"
                />
              </svg>
            </button>
          </template>
          <div class="mode-options">
            <div
              v-for="option in searchModeOptions"
              :key="option.value"
              class="mode-option"
              :class="{ 'is-selected': searchMode === option.value }"
              @click="handleModeChange(option.value)"
            >
              <span class="mode-label">{{ option.label }}</span>
            </div>
          </div>
        </el-popover>

        <!-- 悬浮搜索/替换面板 -->
        <div class="search-panel-anchor tw-relative tw-flex-1 tw-min-w-0">
          <div class="search-panel" :class="{ 'is-expanded': isReplaceExpanded }">
            <!-- 搜索行：展开按钮 | 搜索框 | 忽略标点 | 清除 -->
            <button
              class="panel-icon-btn toggle-btn"
              :class="{ 'is-expanded': isReplaceExpanded }"
              @click="isReplaceExpanded = !isReplaceExpanded"
            >
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z" />
              </svg>
            </button>
            <input
              v-model="localSearchText"
              type="text"
              :placeholder="searchPlaceholder"
              class="panel-input"
              @keyup.enter="handleSearch"
            />
            <!-- R1C3: 清除按钮（有内容时显示，否则隐藏占位） -->
            <el-popover content="清除搜索" placement="top" trigger="hover" popper-class="hint-popover-compact" :show-after="500">
              <template #reference>
                <button
                  class="panel-icon-btn"
                  :style="{ visibility: localSearchText ? 'visible' : 'hidden' }"
                  @click="handleClearSearch"
                >
                  <svg viewBox="0 0 24 24" fill="currentColor">
                    <path
                      d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"
                    />
                  </svg>
                </button>
              </template>
            </el-popover>
            <!-- R1C4: 忽略标点（近音模式显示，否则隐藏占位） -->
            <el-popover
              :content="isIgnorePunctuation ? '取消忽略标点' : '忽略标点'"
              placement="top"
              trigger="hover"
              popper-class="hint-popover-compact"
              :show-after="500"
            >
              <template #reference>
                <button
                  class="panel-icon-btn punctuation-btn"
                  :class="{ 'is-active': isIgnorePunctuation }"
                  :style="{ visibility: isHomophoneMode ? 'visible' : 'hidden' }"
                  @click="toggleIgnorePunctuation"
                >
                  <el-icon :size="14">
                    <Filter />
                  </el-icon>
                </button>
              </template>
            </el-popover>
          </div>

          <!-- 替换行：独立元素，从搜索行下方滑出 -->
          <Transition name="replace-row">
            <div v-if="isReplaceExpanded" class="replace-panel">
              <div class="panel-placeholder" />
              <input
                v-model="replaceText"
                type="text"
                placeholder="替换为..."
                class="panel-input"
                @keyup.enter="handleBatchReplace"
              />
              <el-checkbox
                :model-value="isAllSelected"
                :indeterminate="isIndeterminate"
                :style="{ visibility: matchCount >= 1 ? 'visible' : 'hidden' }"
                size="small"
                class="panel-checkbox"
                title="全选"
                @change="handleToggleSelectAll"
              />
              <el-popover content="批量替换" placement="top" trigger="hover" popper-class="hint-popover-compact" :show-after="500">
                <template #reference>
                  <button
                    class="panel-icon-btn replace-btn"
                    :disabled="selectedCount === 0 || !replaceText"
                    @click="handleBatchReplace"
                  >
                    <svg viewBox="0 0 24 24" fill="currentColor">
                      <path
                        d="M11 6c1.38 0 2.63.56 3.54 1.46L12 10h6V4l-2.05 2.05C14.68 4.78 12.93 4 11 4c-3.53 0-6.43 2.61-6.92 6H6.1c.46-2.28 2.48-4 4.9-4zm5.64 9.14c.66-.9 1.12-1.97 1.28-3.14H15.9c-.46 2.28-2.48 4-4.9 4-1.38 0-2.63-.56-3.54-1.46L10 12H4v6l2.05-2.05C7.32 17.22 9.07 18 11 18c1.55 0 2.98-.51 4.14-1.36L20 21.49 21.49 20l-4.85-4.86z"
                      />
                    </svg>
                  </button>
                </template>
              </el-popover>
            </div>
          </Transition>
        </div>
      </div>
    </div>

    <!-- 右侧：批量删除 + 添加字幕 -->
    <div class="toolbar-right tw-flex tw-items-center tw-flex-shrink-0">
      <el-popover
        v-if="multiSelectedCount > 0"
        :content="batchDeleteConfirm ? '再次点击确认删除' : `删除 ${multiSelectedCount} 条字幕`"
        placement="left"
        trigger="hover"
        popper-class="hint-popover-compact"
        :show-after="500"
      >
        <template #reference>
          <button
            class="toolbar-btn"
            :style="batchDeleteConfirm ? 'color: var(--af-accent-danger);' : ''"
            @click="handleBatchDeleteConfirm"
            @blur="handleBatchDeleteBlur"
            tabindex="0"
          >
            <svg v-if="!batchDeleteConfirm" viewBox="0 0 24 24" fill="currentColor">
              <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
            </svg>
            <svg v-else viewBox="0 0 24 24" fill="currentColor" style="color: var(--af-accent-danger);">
              <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
            </svg>
          </button>
        </template>
      </el-popover>
      <el-popover content="添加字幕" placement="top" trigger="hover" popper-class="hint-popover-compact" :show-after="500">
        <template #reference>
          <button class="toolbar-btn" @click="handleAddSubtitle">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z" />
            </svg>
          </button>
        </template>
      </el-popover>
    </div>
  </div>
</template>

<script setup>
/**
 * 搜索工具栏组件
 *
 * 布局：搜索+替换一体化浮动面板，悬浮在字幕列表上方
 * 使用 CSS Grid 确保上下两行严格对齐
 * toolbar 高度固定不变
 */

import { ref, computed, watch } from 'vue'
import { Filter } from '@element-plus/icons-vue'
import {
  SearchMode,
  SortMode,
  IndexStatus,
  normalizeVisibleSearchMode,
} from '@/composables/useHomophoneSearch'

const props = defineProps({
  totalSubtitles: { type: Number, default: 0 },
  draftCount: { type: Number, default: 0 },
  searchMode: { type: String, default: SearchMode.LITERAL },
  sortMode: { type: String, default: SortMode.TIMELINE },
  searchText: { type: String, default: '' },
  searchReading: { type: String, default: '' },
  isIgnorePunctuation: { type: Boolean, default: false },
  replaceText: { type: String, default: '' },
  isSearching: { type: Boolean, default: false },
  canSearch: { type: Boolean, default: false },
  isSearchActive: { type: Boolean, default: false },
  indexStatus: { type: String, default: IndexStatus.UNKNOWN },
  matchCount: { type: Number, default: 0 },
  selectedCount: { type: Number, default: 0 },
  isAllSelected: { type: Boolean, default: false },
  isIndeterminate: { type: Boolean, default: false },
  multiSelectedCount: { type: Number, default: 0 },
})

const emit = defineEmits([
  'update:searchMode',
  'update:sortMode',
  'update:searchText',
  'update:searchReading',
  'update:isIgnorePunctuation',
  'update:replaceText',
  'search',
  'reset',
  'batch-replace',
  'batch-delete',
  'toggle-select-all',
  'add-subtitle',
  'quick-search',
])

// 本地状态
const localSearchText = ref('')
const isReplaceExpanded = ref(false)
const modePopoverRef = ref(null)
const batchDeleteConfirm = ref(false)

// 同步 searchText -> localSearchText
watch(
  () => props.searchText,
  (val) => {
    localSearchText.value = val
  },
  { immediate: true }
)

// 可写计算属性
const searchMode = computed({
  get: () => normalizeVisibleSearchMode(props.searchMode),
  set: (val) => emit('update:searchMode', normalizeVisibleSearchMode(val)),
})

const replaceText = computed({
  get: () => props.replaceText,
  set: (val) => emit('update:replaceText', val),
})

const isIgnorePunctuation = computed({
  get: () => props.isIgnorePunctuation,
  set: (val) => emit('update:isIgnorePunctuation', val),
})

// 派生状态
const isLiteralMode = computed(() => props.searchMode === SearchMode.LITERAL)

const isHomophoneMode = computed(() => {
  return normalizeVisibleSearchMode(props.searchMode) === SearchMode.HOMOPHONE_FUZZY
})

const currentModeLabel = computed(() => {
  const currentMode = normalizeVisibleSearchMode(props.searchMode)
  const option = searchModeOptions.find((o) => o.value === currentMode)
  return option ? `搜索模式: ${option.label}` : '搜索模式'
})

const searchPlaceholder = computed(() => {
  switch (normalizeVisibleSearchMode(props.searchMode)) {
    case SearchMode.REGEX:
      return '正则表达式...'
    case SearchMode.HOMOPHONE_FUZZY:
      return '文字或拼音...'
    default:
      return '搜索字幕...'
  }
})

const searchModeOptions = [
  { label: '精确', value: SearchMode.LITERAL },
  { label: '正则', value: SearchMode.REGEX },
  { label: '近音', value: SearchMode.HOMOPHONE_FUZZY },
]

// 事件处理
function handleModeChange(mode) {
  emit('update:searchMode', normalizeVisibleSearchMode(mode))
  modePopoverRef.value?.hide?.()
}

function toggleIgnorePunctuation() {
  emit('update:isIgnorePunctuation', !props.isIgnorePunctuation)
}

function handleSearch() {
  emit('update:searchText', localSearchText.value)
  // 统一进入“搜索命中态”，保证精确/正则/近音三类模式都可执行批量替换。
  emit('search')
}

function handleClearSearch() {
  localSearchText.value = ''
  emit('update:searchText', '')
  emit('quick-search', '')
  if (props.isSearchActive) {
    emit('reset')
  }
}

function handleBatchReplace() {
  emit('batch-replace')
}

function handleBatchDelete() {
  emit('batch-delete')
}

function handleToggleSelectAll() {
  emit('toggle-select-all')
}

function handleAddSubtitle() {
  emit('add-subtitle')
}

function handleBatchDeleteConfirm() {
  if (!batchDeleteConfirm.value) {
    batchDeleteConfirm.value = true
  } else {
    batchDeleteConfirm.value = false
    handleBatchDelete()
  }
}
function handleBatchDeleteBlur() {
  batchDeleteConfirm.value = false
}
</script>

<style scoped>
/* toolbar 主容器 */
.list-toolbar {
  background: var(--af-bg-secondary);
  border-bottom: 1px solid var(--af-border-default);
  overflow: visible; /* 允许替换框向下溢出 */
  position: relative; /* 建立定位上下文 */
  z-index: 30; /* 确保高于字幕列表 */
}

/* 搜索面板锚点 - 让面板相对于此定位 */
.search-panel-anchor {
  position: relative;
  overflow: visible; /* 确保替换框不会被裁剪 */
}

/* 搜索行浮动面板（位置固定，展开时不移动） */
.search-panel {
  position: absolute;
  top: 50%;
  left: 0;
  right: 0;
  transform: translateY(-50%);
  z-index: 20;

  /* 4列 Grid：展开按钮 | 输入框 | 按钮A | 按钮B */
  display: grid;
  grid-template-columns: 22px 1fr 22px 22px;
  align-items: center;
  gap: 2px 4px;

  background: var(--af-bg-tertiary);
  border-radius: var(--af-radius-md);
  padding: 3px 4px;
  box-shadow: 0 1px 4px rgb(0 0 0 / 0.08);
  transition:
    border-radius 0.2s ease,
    box-shadow 0.2s ease;
}

/* 搜索行展开时：去掉下方圆角，去掉阴影（阴影由替换行统一提供） */
.search-panel.is-expanded {
  border-radius: var(--af-radius-md) var(--af-radius-md) 0 0;
  box-shadow: none;
}

/* 替换行（紧贴搜索行下方） */
.replace-panel {
  position: absolute;
  top: calc(50% + 14px); /* 搜索行中心向下偏移半个搜索行高度 */
  left: 0;
  right: 0;
  z-index: 19;

  display: grid;
  grid-template-columns: 22px 1fr 22px 22px;
  align-items: center;
  gap: 2px 4px;

  background: var(--af-bg-tertiary);
  border-radius: 0 0 var(--af-radius-md) var(--af-radius-md);
  padding: 3px 4px;
  margin-top: -1px; /* 向上1px消除间隙 */
}

/* 替换行 + 搜索行组合的整体阴影（包裹容器提供） */
.search-panel-anchor:has(.replace-panel) {
  filter: drop-shadow(0 2px 8px rgb(0 0 0 / 0.12));
}

/* 替换行渐入渐出动画 */
.replace-row-enter-active,
.replace-row-leave-active {
  transition:
    opacity 0.2s ease,
    transform 0.2s ease;
}

.replace-row-enter-from {
  opacity: 0;
  transform: translateY(-4px);
}

.replace-row-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}

/* 面板内图标按钮 */
.panel-icon-btn {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 22px;
  height: 22px;
  border: none;
  background: transparent;
  color: var(--af-text-muted);
  cursor: pointer;
  flex-shrink: 0;
  border-radius: 3px;
  padding: 0;
  transition: color var(--af-transition-fast);
}

.panel-icon-btn:hover {
  color: var(--af-text-normal);
  /* background: rgb(var(--af-accent-primary-rgb), 0.08); */
}

.panel-icon-btn:disabled {
  opacity: 0.3;
  cursor: default;
}

.panel-icon-btn:disabled:hover {
  background: transparent;
  color: var(--af-text-muted);
}

.panel-icon-btn svg {
  width: 14px;
  height: 14px;
}

/* 展开/收起箭头旋转 */
.toggle-btn svg {
  transition: transform 0.15s ease;
}

.toggle-btn.is-expanded svg {
  transform: rotate(90deg);
}

/* 忽略标点按钮激活 */
.punctuation-btn.is-active {
  background: rgb(var(--af-accent-warning-rgb), 0.15);
  color: var(--af-accent-warning);
}

.punctuation-btn.is-active:hover {
  background: rgb(var(--af-accent-warning-rgb), 0.25);
}

/* 替换按钮 hover */
.replace-btn:hover:not(:disabled) {
  color: var(--af-accent-success);
}

/* 面板内输入框 */
.panel-input {
  width: 100%;
  min-width: 0;
  background: transparent;
  border: none;
  color: var(--af-text-normal);
  font-size: 12px;
  outline: none;
  line-height: 22px;
  padding: 0 2px;
}

.panel-input::placeholder {
  color: var(--af-text-muted);
}

/* 占位块（替换行左侧空白，与展开按钮对齐） */
.panel-placeholder {
  width: 22px;
  height: 22px;
}

/* 面板内全选复选框 */
.panel-checkbox {
  display: flex;
  justify-content: center;
  align-items: center;
  margin: 0;
  height: 22px;
}

/* 工具栏按钮通用 */
.toolbar-btn {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 26px;
  height: 26px;
  border-radius: var(--af-radius-md);
  color: var(--af-text-secondary);
  transition: all var(--af-transition-fast);
  flex-shrink: 0;
  cursor: pointer;
  border: none;
  background: transparent;
}

.toolbar-btn:hover {
  background: var(--af-bg-tertiary);
  color: var(--af-accent-primary);
}

.toolbar-btn svg {
  width: 16px;
  height: 16px;
}

/* 模式按钮 */
.mode-btn.is-active {
  background: rgb(var(--af-accent-primary-rgb), 0.15);
  color: var(--af-accent-primary);
}

.mode-btn.is-active:hover {
  background: rgb(var(--af-accent-primary-rgb), 0.25);
}

/* 模式选项弹窗 */
.mode-options {
  display: flex;
  flex-direction: column;
  /* gap: 1px; */
  width: 60px;
  /* padding: 2px 0; 缩减上下内边距 */
}

.mode-option {
  padding: 4px 8px; /* 从 6px 10px 缩减为 4px 8px */
  border-radius: var(--af-radius-sm);
  cursor: pointer;
  transition: background var(--af-transition-fast);
  text-align: center; /* 文字居中 */
}

.mode-option:hover {
  background: var(--af-bg-tertiary);
}

.mode-option.is-selected {
  background: rgb(var(--af-accent-primary-rgb), 0.1);
  color: var(--af-accent-primary);
}

.mode-option .mode-label {
  font-size: 13px;
}
</style>
