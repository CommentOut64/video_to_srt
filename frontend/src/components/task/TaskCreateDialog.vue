<template>
  <el-dialog
    :model-value="showUploadDialog"
    title="创建任务"
    class="upload-dialog"
    width="700px"
    :close-on-click-modal="false"
    @update:model-value="emit('update:show-upload-dialog', $event)"
  >
    <div class="tabs-container">
      <el-tabs :model-value="uploadMode" @update:model-value="emit('update:upload-mode', $event)">
        <el-tab-pane label="直接上传" name="upload">
          <el-upload
            :ref="setUploadRef"
            drag
            multiple
            :auto-upload="false"
            :show-file-list="false"
            accept="video/*,audio/*"
            :on-change="handleFileChange"
          >
            <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
            <div class="el-upload__text">
              拖拽视频文件到此处，或 <em>点击选择</em>
            </div>
          </el-upload>
          <div v-if="uploadFiles.length === 0" class="upload-tip">
            支持 MP4, AVI, MOV 等常见视频格式，以及 MP3, WAV 等音频格式（最多5个）
          </div>

          <div v-if="uploadFiles.length > 0" class="selected-files-tags">
            <div class="tags-container">
              <span v-for="(file, index) in uploadFiles" :key="index" class="file-tag">
                <el-icon class="tag-close" @click.stop="removeUploadFile(index)"><Close /></el-icon>
                <span class="tag-name" :title="file.name">{{ file.name }}</span>
              </span>
            </div>
          </div>
        </el-tab-pane>

        <el-tab-pane label="从本地目录选择" name="select">
          <div class="file-list-container">
            <div v-if="loadingFiles" class="loading-files">
              <el-icon class="is-loading"><Loading /></el-icon>
              <span>加载文件列表中...</span>
            </div>

            <div v-else-if="inputFiles.length === 0" class="empty-files">
              <p>input 目录中没有可用的媒体文件</p>
              <p class="hint">请先将视频文件放入 input 目录</p>
            </div>

            <div v-else class="files-table">
              <el-table
                :ref="setFileTableRef"
                :data="inputFiles"
                @selection-change="handleFileSelectionChange"
                @row-click="handleRowClick"
                max-height="218"
                class="clickable-rows"
              >
                <el-table-column type="selection" width="55" />
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

      <el-button
        v-if="uploadMode === 'select'"
        text
        size="small"
        @click="emit('open-input-folder')"
        class="open-folder-btn"
      >
        打开input目录
      </el-button>
    </div>

    <!-- V3.2.4: 设置区 Tab 化，始终可见 -->
    <SettingsTabs
      :model-value="taskConfig"
      :custom-presets="customPresets"
      @update:model-value="emit('update:task-config', $event)"
      @save-preset="emit('save-preset')"
      @delete-preset="emit('delete-preset', $event)"
      @overwrite-preset="emit('overwrite-preset', $event)"
    />

    <template #footer>
      <div class="dialog-footer">
        <div class="footer-left">
          <el-button @click="emit('close-upload-dialog')">取消</el-button>
        </div>
        <div class="footer-right">
          <el-button @click="emit('save-preset')">保存预设</el-button>
          <el-button
            v-if="uploadMode === 'upload'"
            type="primary"
            class="primary-action-btn"
            :loading="uploading"
            :disabled="uploadFiles.length === 0"
            @click="emit('upload')"
          >
            {{
              uploading
                ? "上传中..."
                : uploadFiles.length > 1
                ? `上传 ${uploadFiles.length} 个文件`
                : "开始上传"
            }}
          </el-button>
          <el-button
            v-if="uploadMode === 'select'"
            type="primary"
            class="primary-action-btn"
            :loading="creatingBatch"
            :disabled="selectedFiles.length === 0"
            @click="emit('batch-create')"
          >
            {{
              creatingBatch
                ? "创建中..."
                : `创建 ${selectedFiles.length} 个任务`
            }}
          </el-button>
        </div>
      </div>
    </template>
  </el-dialog>
</template>

<script setup>
import { UploadFilled, Close, Loading } from "@element-plus/icons-vue";
import SettingsTabs from "@/components/task/settings/SettingsTabs.vue";

defineProps({
  showUploadDialog: {
    type: Boolean,
    default: false,
  },
  uploadMode: {
    type: String,
    default: "upload",
  },
  uploadFiles: {
    type: Array,
    default: () => [],
  },
  inputFiles: {
    type: Array,
    default: () => [],
  },
  selectedFiles: {
    type: Array,
    default: () => [],
  },
  loadingFiles: {
    type: Boolean,
    default: false,
  },
  creatingBatch: {
    type: Boolean,
    default: false,
  },
  uploading: {
    type: Boolean,
    default: false,
  },
  taskConfig: {
    type: Object,
    required: true,
  },
  customPresets: {
    type: Array,
    default: () => [],
  },
  setUploadRef: {
    type: Function,
    required: true,
  },
  setFileTableRef: {
    type: Function,
    required: true,
  },
  handleFileChange: {
    type: Function,
    required: true,
  },
  removeUploadFile: {
    type: Function,
    required: true,
  },
  handleFileSelectionChange: {
    type: Function,
    required: true,
  },
  handleRowClick: {
    type: Function,
    required: true,
  },
  formatFileSize: {
    type: Function,
    required: true,
  },
});

const emit = defineEmits([
  "update:show-upload-dialog",
  "update:upload-mode",
  "update:task-config",
  "open-input-folder",
  "save-preset",
  "delete-preset",
  "overwrite-preset",
  "close-upload-dialog",
  "upload",
  "batch-create",
]);
</script>

<style scoped>
.tabs-container {
  position: relative;
  margin-bottom: 12px;
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

.file-list-container {
  height: 218px;
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
  margin: 8px 0;
  color: var(--af-text-secondary);
}

.file-list-container .empty-files .hint {
  color: var(--af-text-disabled);
  font-size: 14px;
}

.file-list-container .files-table .filename {
  color: var(--af-text-primary);
  font-size: 14px;
  word-break: break-all;
}

.selected-files-tags {
  height: 30px;
  margin-top: 8px;
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

.selected-files-tags .file-tag {
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

.selected-files-tags .file-tag .tag-close {
  flex-shrink: 0;
  width: 14px;
  height: 14px;
  padding: 2px;
  border-radius: 50%;
  color: var(--af-text-muted);
  transition: all var(--af-transition-fast);
  cursor: pointer;
}

.selected-files-tags .file-tag .tag-close:hover {
  background: var(--af-accent-danger);
  color: var(--af-text-on-dark);
}

.selected-files-tags .file-tag .tag-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

:deep(.el-upload) {
  display: block;
}

:deep(.el-upload) .el-upload-dragger {
  height: 180px;
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
  margin-bottom: 16px;
  color: var(--af-text-muted);
  font-size: 48px;
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
  margin-top: 8px;
  padding: 3px 10px;
  border: 1px solid transparent;
  display: flex;
  align-items: center;
  color: var(--af-text-muted);
  font-size: 12px;
}

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
  /* 原因: el-table 未暴露 header-wrapper 背景色的 CSS 变量 */
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
  /* 原因: el-table hover 行背景需要覆盖默认 hover 色 */
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

:deep(.el-table) .el-checkbox__inner {
  background-color: var(--af-bg-primary);
  border-color: var(--af-text-muted);
  border-width: 2px;
}

:deep(.el-table) .el-checkbox__input.is-checked .el-checkbox__inner {
  background-color: var(--af-accent-primary);
  border-color: var(--af-accent-primary);
}

:deep(.el-table) .el-checkbox__input:hover .el-checkbox__inner {
  border-color: var(--af-accent-primary);
}

:deep(.el-table) .el-table__empty-block {
  background: var(--af-bg-secondary);
}

:deep(.el-table) .el-table__empty-block .el-table__empty-text {
  color: var(--af-text-muted);
}
</style>
