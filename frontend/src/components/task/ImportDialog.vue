<template>
  <el-dialog
    :model-value="showDialog"
    title="导入字幕"
    class="upload-dialog"
    width="700px"
    :close-on-click-modal="false"
    @update:model-value="emit('update:show-dialog', $event)"
    @closed="handleClosed"
  >
    <div class="tabs-container">
      <el-tabs v-model="importMode">
        <el-tab-pane label="直接上传" name="upload">
          <!-- 字幕上传区 -->
          <el-upload
            ref="uploadRef"
            drag
            :auto-upload="false"
            :show-file-list="false"
            :limit="1"
            accept=".srt,.ass,.vtt"
            :on-change="handleSubtitleFileChange"
          >
            <el-icon class="el-icon--upload"><Document /></el-icon>
            <div class="el-upload__text">
              拖拽字幕文件到此处，或 <em>点击选择</em>
            </div>
          </el-upload>
          <div v-if="!subtitleFile" class="upload-tip">
            支持 SRT、ASS、VTT 格式
          </div>
          <div v-else class="selected-files-tags">
            <div class="tags-container">
              <span class="file-tag">
                <el-icon class="tag-close" @click.stop="handleRemoveSubtitle"><Close /></el-icon>
                <span class="tag-name" :title="subtitleFile.name">{{ subtitleFile.name }}</span>
              </span>
            </div>
          </div>

          <!-- 媒体选择区（紧凑） -->
          <hr class="section-divider" />
          <div class="section-label">关联媒体 <span class="optional">（可选）</span></div>
          <div
            class="media-select-zone"
            :class="{ 'has-file': mediaFile }"
            @click="triggerMediaFileInput"
          >
            <template v-if="!mediaFile">
              <el-icon><FolderAdd /></el-icon>
              <span>选择视频或音频文件</span>
              <span class="media-hint">关联媒体以启用波形和视频预览</span>
            </template>
            <template v-else>
              <span class="file-tag">
                <el-icon class="tag-close" @click.stop="removeMediaFile"><Close /></el-icon>
                <span class="tag-name" :title="mediaFile.name">{{ mediaFile.name }}</span>
                <span class="tag-size">{{ formatFileSize(mediaFile.size) }}</span>
              </span>
              <span class="media-hint">点击更换媒体文件</span>
            </template>
          </div>
          <input
            ref="mediaInputRef"
            type="file"
            accept=".mp4,.avi,.mkv,.mov,.wmv,.flv,.webm,.m4v,.mp3,.wav,.flac,.aac,.ogg,.m4a,.wma,video/*,audio/*"
            style="display: none"
            @change="handleMediaFileChange"
          />
        </el-tab-pane>

        <el-tab-pane label="从本地目录选择" name="select">
          <!-- 字幕文件表格 -->
          <div class="file-list-container subtitle-table">
            <div v-if="loadingSubtitles" class="loading-files">
              <el-icon class="is-loading"><Loading /></el-icon>
              <span>加载文件列表中...</span>
            </div>
            <div v-else-if="subtitleFiles.length === 0" class="empty-files">
              <p>input 目录中没有字幕文件</p>
              <p class="hint">请先将 .srt / .ass / .vtt 文件放入 input 目录</p>
            </div>
            <div v-else class="files-table">
              <el-table
                ref="subtitleTableRef"
                :data="subtitleFiles"
                highlight-current-row
                max-height="190"
                class="clickable-rows"
                @current-change="handleSubtitleSelect"
              >
                <el-table-column width="55">
                  <template #default="{ row }">
                    <el-radio :model-value="selectedSubtitle" :value="row.name" />
                  </template>
                </el-table-column>
                <el-table-column prop="name" label="文件名" min-width="200">
                  <template #default="{ row }">
                    <span class="filename">{{ row.name }}</span>
                  </template>
                </el-table-column>
                <el-table-column prop="size" label="大小" width="100">
                  <template #default="{ row }">
                    {{ formatFileSize(row.size) }}
                  </template>
                </el-table-column>
                <el-table-column prop="modified" label="修改时间" width="160" />
              </el-table>
            </div>
          </div>

          <!-- 媒体文件表格 -->
          <hr class="section-divider" />
          <div class="section-label">关联媒体 <span class="optional">（可选）</span></div>
          <div class="file-list-container media-table">
            <div v-if="loadingMedias" class="loading-files">
              <el-icon class="is-loading"><Loading /></el-icon>
              <span>加载中...</span>
            </div>
            <div v-else-if="mediaFiles.length === 0" class="empty-files">
              <p>input 目录中没有媒体文件</p>
            </div>
            <div v-else class="files-table">
              <el-table
                ref="mediaTableRef"
                :data="mediaFiles"
                highlight-current-row
                max-height="190"
                class="clickable-rows"
                @current-change="onMediaRowChange"
              >
                <el-table-column width="55">
                  <template #default="{ row }">
                    <el-radio :model-value="selectedMedia" :value="row.name" />
                  </template>
                </el-table-column>
                <el-table-column prop="name" label="文件名" min-width="200">
                  <template #default="{ row }">
                    <span class="filename">{{ row.name }}</span>
                  </template>
                </el-table-column>
                <el-table-column prop="size" label="大小" width="100">
                  <template #default="{ row }">
                    {{ formatFileSize(row.size) }}
                  </template>
                </el-table-column>
                <el-table-column prop="modified" label="修改时间" width="160" />
              </el-table>
            </div>
          </div>
        </el-tab-pane>
      </el-tabs>

      <!-- "打开input目录"按钮 -->
      <el-button
        v-if="importMode === 'select'"
        text
        size="small"
        class="open-folder-btn"
        @click="handleOpenInputFolder"
      >
        打开input目录
      </el-button>
    </div>

    <template #footer>
      <div class="dialog-footer">
        <div class="footer-left">
          <el-button @click="emit('update:show-dialog', false)">取消</el-button>
        </div>
        <div class="footer-right">
          <el-button
            type="primary"
            class="primary-action-btn"
            :loading="isImporting"
            :disabled="!canImport"
            @click="doImport"
          >
            {{ isImporting ? '导入中...' : '开始导入' }}
          </el-button>
        </div>
      </div>
    </template>
  </el-dialog>
</template>

<script setup>
import { ref, computed } from 'vue'
import { Document, Close, FolderAdd, Loading } from '@element-plus/icons-vue'
import { useTaskImport } from '@/composables/task-list/useTaskImport'

defineProps({
  showDialog: {
    type: Boolean,
    default: false,
  },
})

const emit = defineEmits(['update:show-dialog', 'import-success'])

const uploadRef = ref(null)
const mediaInputRef = ref(null)
const subtitleTableRef = ref(null)
const mediaTableRef = ref(null)

const {
  importMode,
  isImporting,
  subtitleFile,
  mediaFile,
  subtitleFiles,
  mediaFiles,
  selectedSubtitle,
  selectedMedia,
  loadingSubtitles,
  loadingMedias,
  handleSubtitleFileChange,
  removeSubtitleFile,
  handleMediaFileChange,
  removeMediaFile,
  handleOpenInputFolder,
  handleSubtitleSelect,
  handleMediaSelect,
  formatFileSize,
  handleImport,
  resetState,
} = useTaskImport()

/** 是否满足导入条件 */
const canImport = computed(() => {
  if (isImporting.value) return false
  if (importMode.value === 'upload') return !!subtitleFile.value
  return !!selectedSubtitle.value
})

/** 移除字幕文件并清空 el-upload 内部状态 */
function handleRemoveSubtitle() {
  removeSubtitleFile()
  uploadRef.value?.clearFiles()
}

/** 触发隐藏的 media file input */
function triggerMediaFileInput() {
  mediaInputRef.value?.click()
}

/** 媒体表格行选中（支持取消选中） */
function onMediaRowChange(row) {
  handleMediaSelect(row)
  // 如果取消选中，清除 el-table 的 highlight
  if (!row || selectedMedia.value !== row.name) {
    mediaTableRef.value?.setCurrentRow(null)
  }
}

/** 执行导入 */
async function doImport() {
  const projectId = await handleImport()
  if (projectId) {
    emit('import-success', projectId)
    emit('update:show-dialog', false)
  }
}

/** dialog 关闭动画完成后重置状态 */
function handleClosed() {
  resetState()
  uploadRef.value?.clearFiles()
}
</script>

<style scoped>
.tabs-container {
  position: relative;
  min-height: 470px;
}

.tabs-container .open-folder-btn {
  position: absolute;
  top: 8px;
  right: 0;
  padding: 4px 8px;
  color: var(--af-text-secondary);
}

.tabs-container .open-folder-btn:hover {
  background: var(--af-bg-tertiary);
  color: var(--af-accent-primary);
}

.tabs-container .open-folder-btn:active {
  background: var(--af-bg-quaternary);
  color: var(--af-accent-primary);
}

/* === 区域标签 === */
.section-label {
  margin-bottom: 4px;
  color: var(--af-text-primary);
  font-size: 13px;
  font-weight: 500;
}

.section-label .optional {
  color: var(--af-text-muted);
  font-size: 12px;
  font-weight: 400;
}

.section-divider {
  margin: 6px 0;
  border: none;
  border-top: 1px dashed var(--af-border-light);
}

/* === 直接上传 - 字幕拖拽区 === */
:deep(.el-upload) {
  display: block;
}

:deep(.el-upload) .el-upload-dragger {
  height: 173px;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  background: var(--af-bg-tertiary);
  border: 2px dashed var(--af-border-default);
  border-radius: var(--af-radius-md);
  transition: all var(--af-transition-fast);
}

:deep(.el-upload) .el-upload-dragger:hover {
  background: var(--af-bg-elevated);
  border-color: var(--af-accent-primary);
}

:deep(.el-upload) .el-upload-dragger .el-icon--upload {
  margin-bottom: 12px;
  color: var(--af-text-muted);
  font-size: 40px;
}

:deep(.el-upload) .el-upload-dragger .el-upload__text {
  color: var(--af-text-secondary);
}

:deep(.el-upload) .el-upload-dragger .el-upload__text em {
  color: var(--af-accent-primary);
  font-style: normal;
}

.upload-tip {
  height: 30px;
  margin-top: 4px;
  padding: 3px 10px;
  border: 1px solid transparent;
  display: flex;
  align-items: center;
  color: var(--af-text-muted);
  font-size: 12px;
}

/* === 已选文件标签 === */
.selected-files-tags {
  height: 30px;
  margin-top: 4px;
  padding: 3px 10px;
  overflow-y: auto;
  background: var(--af-bg-tertiary);
  border: 1px solid var(--af-border-default);
  border-radius: var(--af-radius-md);
}

.selected-files-tags .tags-container {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  min-height: 100%;
}

.file-tag {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  max-width: 200px;
  padding: 2px 8px 2px 4px;
  background: var(--af-bg-elevated);
  border: 1px solid var(--af-border-default);
  border-radius: 20px;
  color: var(--af-text-primary);
  font-size: 11px;
  line-height: 1;
  transition: all var(--af-transition-fast);
  user-select: none;
  cursor: default;
}

.file-tag .tag-close {
  flex-shrink: 0;
  width: 14px;
  height: 14px;
  padding: 2px;
  border-radius: 50%;
  color: var(--af-text-muted);
  transition: all var(--af-transition-fast);
  cursor: pointer;
}

.file-tag .tag-close:hover {
  background: var(--af-accent-danger);
  color: var(--af-text-on-dark);
}

.file-tag .tag-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-tag .tag-size {
  flex-shrink: 0;
  margin-left: 4px;
  color: var(--af-text-muted);
  font-size: 10px;
}

/* === 媒体选择区（直接上传模式） === */
.media-select-zone {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
  height: 173px;
  padding: 12px;
  background: var(--af-bg-tertiary);
  border: 1px dashed var(--af-border-default);
  border-radius: var(--af-radius-md);
  cursor: pointer;
  transition: all var(--af-transition-fast);
}

.media-select-zone:hover {
  background: var(--af-bg-elevated);
  border-color: var(--af-accent-primary);
}

.media-select-zone.has-file {
  flex-direction: row;
  justify-content: space-between;
  border-style: solid;
}

.media-select-zone .el-icon {
  color: var(--af-text-muted);
  font-size: 18px;
}

.media-select-zone span {
  color: var(--af-text-secondary);
  font-size: 13px;
}

.media-select-zone .media-hint {
  color: var(--af-text-muted);
  font-size: 11px;
}

/* === el-tabs 样式 === */
:deep(.el-tabs) .el-tabs__header {
  margin-bottom: 16px;
}

:deep(.el-tabs) .el-tabs__header .el-tabs__nav-wrap::after {
  background-color: var(--af-border-default);
}

:deep(.el-tabs) .el-tabs__header .el-tabs__item {
  color: var(--af-text-secondary);
}

:deep(.el-tabs) .el-tabs__header .el-tabs__item:hover {
  color: var(--af-text-primary);
}

:deep(.el-tabs) .el-tabs__header .el-tabs__item.is-active {
  color: var(--af-accent-primary);
}

:deep(.el-tabs) .el-tabs__header .el-tabs__active-bar {
  background-color: var(--af-accent-primary);
}

/* === 文件列表容器 === */
.file-list-container.subtitle-table {
  height: 190px;
}

.file-list-container.media-table {
  height: 190px;
}

.file-list-container .loading-files {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  gap: 12px;
  height: 100%;
  color: var(--af-text-secondary);
}

.file-list-container .loading-files .el-icon {
  font-size: 32px;
}

.file-list-container .empty-files {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  height: 100%;
  text-align: center;
}

.file-list-container .empty-files p {
  margin: 4px 0;
  color: var(--af-text-secondary);
}

.file-list-container .empty-files .hint {
  color: var(--af-text-disabled);
  font-size: 13px;
}

.file-list-container .files-table .filename {
  color: var(--af-text-primary);
  font-size: 14px;
  word-break: break-all;
}

/* === el-table 样式（与 TaskCreateDialog 一致） === */
:deep(.el-table) {
  --el-table-bg-color: var(--af-bg-secondary);
  --el-table-tr-bg-color: var(--af-bg-secondary);
  --el-table-header-bg-color: var(--af-bg-secondary);
  --el-table-row-hover-bg-color: rgb(var(--af-accent-primary-rgb, 99, 102, 241), 8%);
  --el-table-border-color: var(--af-border-default);
  --el-table-text-color: var(--af-text-primary);
  --el-table-header-text-color: var(--af-text-secondary);

  overflow: hidden;
  background: var(--af-bg-secondary);
  border-radius: 0;
}

:deep(.el-table) .el-table__header-wrapper th {
  background: var(--af-bg-secondary) !important;
  border-bottom: 1px solid var(--af-border-default);
  color: var(--af-text-secondary);
  font-weight: 500;
}

:deep(.el-table) .el-table__body-wrapper {
  background: var(--af-bg-secondary);
}

:deep(.el-table) .el-table__body-wrapper tr {
  background: var(--af-bg-secondary);
}

:deep(.el-table) .el-table__body-wrapper tr td {
  border-bottom: 1px solid var(--af-border-light);
  color: var(--af-text-primary);
}

:deep(.el-table).clickable-rows .el-table__body-wrapper tr {
  cursor: pointer;
}

:deep(.el-table) .el-table__body-wrapper tr:hover > td {
  background: rgb(var(--af-accent-primary-rgb, 99, 102, 241), 8%) !important;
}

:deep(.el-table) .el-scrollbar__bar.is-vertical {
  right: 2px;
  width: 6px;
}

:deep(.el-table) .el-scrollbar__bar .el-scrollbar__thumb {
  background-color: var(--af-text-muted);
  border-radius: 3px;
  opacity: 0.5;
}

:deep(.el-table) .el-scrollbar__bar .el-scrollbar__thumb:hover {
  opacity: 0.8;
}

:deep(.el-table) .el-table__empty-block {
  background: var(--af-bg-secondary);
}

:deep(.el-table) .el-table__empty-block .el-table__empty-text {
  color: var(--af-text-muted);
}

/* === el-radio 样式 === */
:deep(.el-radio) .el-radio__inner {
  background-color: var(--af-bg-primary);
  border-color: var(--af-text-muted);
  border-width: 2px;
}

:deep(.el-radio) .el-radio__input.is-checked .el-radio__inner {
  background-color: var(--af-accent-primary);
  border-color: var(--af-accent-primary);
}

:deep(.el-radio) .el-radio__input:hover .el-radio__inner {
  border-color: var(--af-accent-primary);
}

/* === el-table 当前行高亮 === */
:deep(.el-table) .el-table__body tr.current-row > td {
  background: rgb(var(--af-accent-primary-rgb, 99, 102, 241), 12%) !important;
}
</style>
