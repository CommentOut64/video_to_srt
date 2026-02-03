<template>
  <div class="task-list-view">
    <!-- 顶部导航栏 -->
    <header class="task-header">
      <div class="header-left">
        <h1 class="app-title" @click="showAboutDialog = true">
          <svg class="app-icon" viewBox="0 0 24 24" fill="currentColor">
            <path
              d="M21 3H3c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H3V5h18v14zM5 10h9v2H5zm0-3h9v2H5zm0 6h6v2H5z"
            />
          </svg>
          AnchorFlux
        </h1>
      </div>
      <div class="header-right">
        <el-button type="primary" @click="showUploadDialog = true">
          <el-icon><Upload /></el-icon>
          上传视频
        </el-button>
        <el-button type="primary" @click="handleExit">
          退出系统
        </el-button>
      </div>
    </header>

    <!-- 主内容区 -->
    <main class="task-main">
      <!-- 空状态 -->
      <div v-if="tasks.length === 0" class="empty-state">
        <svg class="empty-icon" viewBox="0 0 24 24" fill="currentColor">
          <path
            d="M21 3H3c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H3V5h18v14zM5 10h9v2H5zm0-3h9v2H5zm0 6h6v2H5z"
          />
        </svg>
        <h2 class="empty-title">还没有任务</h2>
        <p class="empty-desc">点击上方"上传视频"按钮开始创建字幕任务</p>
        <el-button type="primary" size="large" @click="showUploadDialog = true">
          <el-icon><Upload /></el-icon>
          上传视频
        </el-button>
      </div>

      <!-- 任务列表 -->
      <div v-else class="task-grid">
        <div
          v-for="task in tasks"
          :key="task.job_id"
          class="task-card"
          :class="`status-${task.status}`"
        >
          <!-- 视频缩略图 -->
          <div class="task-thumbnail">
            <img
              v-if="
                thumbnailCache[task.job_id] &&
                thumbnailCache[task.job_id] !== null
              "
              :src="thumbnailCache[task.job_id]"
              class="thumbnail-image"
              alt="Video thumbnail"
            />
            <div
              v-else
              class="thumbnail-placeholder"
              :class="{ clickable: thumbnailCache[task.job_id] === null }"
              @click.stop="
                thumbnailCache[task.job_id] === null &&
                  getThumbnailUrl(task.job_id, true)
              "
              :title="thumbnailCache[task.job_id] === null ? '点击重试' : ''"
            >
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z" />
              </svg>
            </div>
            <!-- 缩略图加载中 -->
            <div
              v-if="thumbnailCache[task.job_id] === undefined"
              class="thumbnail-loading"
            >
              <svg class="loading-spinner" viewBox="0 0 24 24">
                <circle
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  stroke-width="2"
                  fill="none"
                />
              </svg>
            </div>
            <div v-if="task.status !== 'finished'" class="status-overlay">
              <span class="status-text">{{ getStatusText(task.status) }}</span>
            </div>
          </div>

          <!-- 任务信息 -->
          <div class="task-info">
            <!-- 可编辑的任务标题 -->
            <div class="task-title-wrapper">
              <input
                v-if="editingTaskId === task.job_id"
                ref="titleInputRef"
                v-model="editingTitle"
                class="task-title-input"
                @blur="finishEditTitle(task)"
                @keyup.enter="finishEditTitle(task)"
                @keyup.esc="cancelEditTitle"
              />
              <h3
                v-else
                class="task-title task-title-link"
                :title="getTaskDisplayName(task) + ' (点击查看，双击重命名)'"
                @click="handleTitleClick(task)"
                @dblclick.prevent="startEditTitle(task)"
              >
                {{ getTaskDisplayName(task) }}
              </h3>
            </div>
            <div class="task-meta">
              <span class="meta-item">
                <el-icon><Clock /></el-icon>
                {{ formatDate(task.createdAt) }}
              </span>
              <span
                v-if="task.status === 'processing' || task.status === 'queued'"
                class="meta-item"
              >
                <el-icon><Loading /></el-icon>
                {{ (task.progress || 0).toFixed(1) }}%
              </span>
            </div>

            <!-- 进度条 -->
            <el-progress
              v-if="task.status === 'processing' || task.status === 'queued'"
              :percentage="task.progress"
              :show-text="false"
              :stroke-width="4"
            />
          </div>

          <!-- 操作按钮 -->
          <div class="task-actions">
            <el-button
              type="primary"
              size="small"
              @click="openEditor(task.job_id)"
            >
              <el-icon><Edit /></el-icon>
              {{ task.status === "finished" ? "编辑" : "查看" }}
            </el-button>
            <el-button size="small" @click="startEditTitle(task)">
              重命名
            </el-button>
            <el-button size="small" @click="deleteTask(task.job_id)">
              <el-icon><Delete /></el-icon>
              删除
            </el-button>
          </div>
        </div>
      </div>
    </main>

    <!-- 上传对话框（双模式：直接上传 + 从 input 目录选择） -->
    <el-dialog
      v-model="showUploadDialog"
      title="创建任务"
      width="700px"
      :close-on-click-modal="false"
    >
      <!-- 选项卡 -->
      <div class="tabs-container">
        <el-tabs v-model="uploadMode">
        <!-- 模式A：直接上传 -->
        <el-tab-pane label="直接上传" name="upload">
          <el-upload
            ref="uploadRef"
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
            <template #tip>
              <div class="el-upload__tip">
                支持 MP4, AVI, MOV 等常见视频格式，以及 MP3, WAV
                等音频格式（最多5个）
              </div>
            </template>
          </el-upload>

          <!-- 已选择的文件标签列表 -->
          <div v-if="uploadFiles.length > 0" class="selected-files-tags">
            <div class="tags-container">
              <span
                v-for="(file, index) in uploadFiles"
                :key="index"
                class="file-tag"
              >
                <el-icon class="tag-close" @click.stop="removeUploadFile(index)"
                  ><Close
                /></el-icon>
                <span class="tag-name" :title="file.name">{{ file.name }}</span>
              </span>
            </div>
          </div>
        </el-tab-pane>

        <!-- 模式B：从 input 目录选择 -->
        <el-tab-pane label="从本地目录选择" name="select">
          <div class="file-list-container">
            <!-- 文件列表加载中 -->
            <div v-if="loadingFiles" class="loading-files">
              <el-icon class="is-loading"><Loading /></el-icon>
              <span>加载文件列表中...</span>
            </div>

            <!-- 文件列表为空 -->
            <div v-else-if="inputFiles.length === 0" class="empty-files">
              <p>input 目录中没有可用的媒体文件</p>
              <p class="hint">请先将视频文件放入 input 目录</p>
            </div>

            <!-- 文件列表 -->
            <div v-else class="files-table">
              <el-table
                ref="fileTableRef"
                :data="inputFiles"
                @selection-change="handleFileSelectionChange"
                @row-click="handleRowClick"
                max-height="280"
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

      <!-- 打开input目录按钮 - 只在"从本地目录选择"标签页显示 -->
      <el-button
        v-if="uploadMode === 'select'"
        text
        size="small"
        @click="handleOpenInputFolder"
        class="open-folder-btn"
      >
        打开input目录
      </el-button>
    </div>

      <!-- 转录设置区域 - v3.5 预设模式 -->
      <div class="transcription-settings">
        <div class="settings-header" @click="showAdvancedSettings = !showAdvancedSettings">
          <span>转录设置</span>
          <el-icon :class="{ 'is-expanded': showAdvancedSettings }"><ArrowDown /></el-icon>
        </div>

        <div class="settings-content" v-if="showAdvancedSettings">
          <!-- v3.5 预设选择器组件 -->
          <PresetSelector
            v-model="taskConfig"
            :compact="true"
            @change="handlePresetChange"
          />
        </div>
      </div>

      <template #footer>
        <div class="dialog-footer">
          <span v-if="uploadMode === 'select'" class="selection-info">
            已选择 {{ selectedFiles.length }} 个文件
          </span>
          <span v-else-if="uploadFiles.length > 0" class="selection-info">
            已选择 {{ uploadFiles.length }} 个文件
          </span>
          <div class="footer-buttons">
            <el-button @click="closeUploadDialog">取消</el-button>
            <el-button
              v-if="uploadMode === 'upload' && uploadFiles.length > 0"
              type="primary"
              class="primary-action-btn"
              :loading="uploading"
              @click="handleUpload"
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
              @click="handleBatchCreate"
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

    <!-- 关于对话框 -->
    <AboutDialog v-model="showAboutDialog" />
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch, nextTick } from "vue";
import { useRouter } from "vue-router";
import { ElMessage, ElMessageBox, ElLoading } from "element-plus";
import {
  Upload,
  UploadFilled,
  Edit,
  Delete,
  Clock,
  Loading,
  Close,
  ArrowDown,
} from "@element-plus/icons-vue";
import { useUnifiedTaskStore } from "@/stores/unifiedTaskStore";
import { transcriptionApi, systemApi } from "@/services/api";
import fileApi from "@/services/api/fileApi"; // 导入文件 API
// V3.1.0: 移除 sseChannelManager 导入，SSE 订阅由 App.vue 统一管理
import PresetSelector from "@/components/task/PresetSelector.vue"; // v3.5 预设选择器
import AboutDialog from "@/components/AboutDialog.vue"; // 关于对话框

const router = useRouter();
const taskStore = useUnifiedTaskStore();

// 响应式数据 - 上传相关
const showUploadDialog = ref(false);
const showAboutDialog = ref(false);
const uploadMode = ref("upload"); // 上传模式：'upload' 或 'select'
const uploading = ref(false);
const uploadRef = ref(null);
const selectedFile = ref(null);
const uploadFiles = ref([]); // 多文件上传列表
const thumbnailCache = ref({}); // 缩略图缓存，避免重复加载

// 响应式数据 - input 目录文件列表
const inputFiles = ref([]); // input 目录文件列表
const selectedFiles = ref([]); // 用户选中的文件
const loadingFiles = ref(false); // 文件列表加载中
const creatingBatch = ref(false); // 批量创建中
const fileTableRef = ref(null); // 文件表格引用

// 内联重命名相关
const editingTaskId = ref(null); // 当前正在编辑的任务ID
const editingTitle = ref(""); // 编辑中的标题
const originalTitle = ref(""); // 原始标题（用于恢复）
const titleInputRef = ref(null); // 输入框引用

// 转录设置相关 - v3.5 预设模式
const showAdvancedSettings = ref(false); // 是否显示高级设置
// v3.5 任务配置（默认使用 balanced 预设）
const taskConfig = ref({
  preset_id: 'balanced',
  preprocessing: {
    demucs_strategy: 'auto',
    demucs_model: 'htdemucs',
    demucs_shifts: 1,
    separation_mode: 'on_demand',
    spectrum_threshold: 0.35,
    vad_filter: true,
    enable_spectral_triage: true,
    language_detection_mode: 'balanced',
    language_detection_device: 'auto',
    enable_speaker_embedding: false,
    langid_confidence_threshold: 0.7,
    langid_whitelist: ['zh', 'ja', 'en'],
    langid_logit_bias_score: 2.5
  },
  transcription: {
    transcription_profile: 'sv_whisper_patch',
    sensevoice_device: 'auto',
    whisper_model: 'medium',
    patching_threshold: 0.60
  },
  refinement: {
    llm_task: 'proofread',
    llm_scope: 'sparse',
    sparse_threshold: 0.70,
    target_language: 'zh',
    llm_provider: 'openai_compatible',
    llm_model_name: 'gpt-4o-mini'
  },
  compute: {
    concurrency_strategy: 'auto',
    gpu_id: 0,
    output_formats: ['srt'],
    temp_file_policy: 'delete_on_complete'
  }
});

// 处理预设变更事件
function handlePresetChange(newConfig) {
  taskConfig.value = { ...newConfig };
}

// 计算属性 - 使用 computed 包装确保响应式
const tasks = computed(() => taskStore.tasks);

// V3.1.0: 移除 unsubscribeGlobalSSE，SSE 订阅由 App.vue 统一管理

// 监听任务列表变化，自动加载新增任务的缩略图
watch(
  () => tasks.value?.length,
  (newLength) => {
    if (newLength && tasks.value) {
      tasks.value.forEach((task) => {
        // 只加载还没缓存的任务的缩略图
        if (!(task.job_id in thumbnailCache.value)) {
          setTimeout(() => {
            getThumbnailUrl(task.job_id);
          }, 100);
        }
      });
    }
  }
);

// 监听上传模式切换，自动加载文件列表（每次切换到 select 都刷新）
watch(uploadMode, async (newMode) => {
  if (newMode === "select") {
    await loadInputFiles();
  }
});

// 打开上传弹窗时，若当前在 select 标签页则刷新一次
watch(showUploadDialog, async (visible) => {
  if (visible && uploadMode.value === "select") {
    await loadInputFiles();
  }
});

// 加载 input 目录文件列表
async function loadInputFiles() {
  loadingFiles.value = true;
  try {
    const { files } = await fileApi.listFiles();
    inputFiles.value = files || [];
  } catch (error) {
    console.error("加载文件列表失败:", error);
    ElMessage.error(`加载文件列表失败: ${error.message || "未知错误"}`);
    inputFiles.value = [];
  } finally {
    loadingFiles.value = false;
  }
}

// 打开input文件夹
async function handleOpenInputFolder() {
  try {
    await fileApi.openInputFolder();
    ElMessage.success("已打开input文件夹");
  } catch (error) {
    console.error("打开文件夹失败:", error);
    ElMessage.error(`打开文件夹失败: ${error.message || "未知错误"}`);
  }
}

// 处理文件列表多选
function handleFileSelectionChange(selection) {
  selectedFiles.value = selection;
}

// 批量创建任务
async function handleBatchCreate() {
  if (selectedFiles.value.length === 0) {
    ElMessage.warning("请先选择文件");
    return;
  }

  creatingBatch.value = true;
  try {
    const filenames = selectedFiles.value.map((file) => file.name);
    const result = await fileApi.createJobsBatch(filenames);

    // 处理成功的任务 - 为每个任务启动转录
    if (result.succeeded > 0) {
      // v3.5+: 构建转录设置，仅使用 task_config 格式
      const transcriptionSettings = {
        task_config: {
          preset_id: taskConfig.value.preset_id,
          preprocessing: { ...taskConfig.value.preprocessing },
          transcription: { ...taskConfig.value.transcription },
          refinement: { ...taskConfig.value.refinement },
          compute: { ...taskConfig.value.compute }
        }
      };

      // 为每个成功创建的任务启动转录
      for (const job of result.jobs) {
        try {
          await transcriptionApi.startJob(job.job_id, transcriptionSettings);
        } catch (startError) {
          console.warn(`启动任务 ${job.job_id} 失败:`, startError);
        }
      }

      // 同步任务列表
      await taskStore.syncTasksFromBackend();
      ElMessage.success(`成功创建并启动 ${result.succeeded} 个任务`);
    }

    // 处理失败的任务
    if (result.failed_count > 0) {
      const failedList = result.failed
        .map((f) => `${f.filename}: ${f.error}`)
        .join("\n");
      ElMessage.warning({
        message: `${result.failed_count} 个文件创建失败:\n${failedList}`,
        duration: 5000,
      });
    }

    // 关闭对话框
    if (result.succeeded > 0) {
      showUploadDialog.value = false;
      selectedFiles.value = [];
      inputFiles.value = [];
    }
  } catch (error) {
    console.error("批量创建任务失败:", error);
    ElMessage.error(`批量创建任务失败: ${error.message || "未知错误"}`);
  } finally {
    creatingBatch.value = false;
  }
}

// 格式化文件大小
function formatFileSize(bytes) {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
}

// 处理文件选择（支持多文件）
function handleFileChange(file, fileList) {
  // 检查文件数量限制
  if (fileList.length > 5) {
    ElMessage.warning(
      '一次最多上传5个文件，如需批量处理更多文件，请将文件放入 input 目录并使用"从本地目录选择"功能'
    );
    // 移除超出的文件
    uploadRef.value?.handleRemove(file);
    return;
  }

  // 检查是否已存在同名文件
  const exists = uploadFiles.value.some((f) => f.name === file.name);
  if (exists) {
    uploadRef.value?.handleRemove(file);
    return;
  }

  // 更新文件列表
  uploadFiles.value = fileList.map((f) => f.raw || f);
  selectedFile.value =
    uploadFiles.value.length > 0 ? uploadFiles.value[0] : null;
}

// 移除上传文件
function removeUploadFile(index) {
  uploadFiles.value.splice(index, 1);
  selectedFile.value =
    uploadFiles.value.length > 0 ? uploadFiles.value[0] : null;
  // 同步 el-upload 的文件列表
  if (uploadRef.value) {
    uploadRef.value.clearFiles();
    // 重新设置文件列表
    uploadFiles.value.forEach((file) => {
      // el-upload 内部会处理
    });
  }
}

// 处理行点击切换选中状态
function handleRowClick(row) {
  if (fileTableRef.value) {
    fileTableRef.value.toggleRowSelection(row);
  }
}

// 关闭上传对话框并重置状态
function closeUploadDialog() {
  showUploadDialog.value = false;
  selectedFile.value = null;
  uploadFiles.value = [];
  uploadRef.value?.clearFiles();
}

// 处理上传（支持多文件）
async function handleUpload() {
  if (uploadFiles.value.length === 0) {
    ElMessage.warning("请先选择视频文件");
    return;
  }

  uploading.value = true;
  const successCount = ref(0);
  const failCount = ref(0);

  try {
    // V3.1.1+dev.20260106.01: 调试日志
    console.log('[DEBUG] taskConfig.preprocessing:', JSON.stringify(taskConfig.value.preprocessing, null, 2));

    // v3.5+: 构建转录设置，仅使用 task_config 格式
    const transcriptionSettings = {
      task_config: {
        preset_id: taskConfig.value.preset_id,
        preprocessing: { ...taskConfig.value.preprocessing },
        transcription: { ...taskConfig.value.transcription },
        refinement: { ...taskConfig.value.refinement },
        compute: { ...taskConfig.value.compute }
      }
    };

    // V3.1.1+dev.20260106.01: 调试日志
    console.log('[DEBUG] transcriptionSettings.task_config.preprocessing:', JSON.stringify(transcriptionSettings.task_config.preprocessing, null, 2));

    // 逐个上传文件
    for (const file of uploadFiles.value) {
      try {
        // 上传文件到后端
        const { job_id, filename, queue_position } =
          await transcriptionApi.uploadFile(file, (percent) => {
            console.log(`上传进度 [${file.name}]: ${percent}%`);
          });

        // 添加任务到 store
        taskStore.addTask({
          job_id,
          filename,
          file_path: null,
          status: "queued",
          phase: "uploading",
          progress: 0,
          message: `已加入队列 (位置: ${queue_position})`,
          settings: transcriptionSettings,
        });

        // 启动转录任务
        await transcriptionApi.startJob(job_id, transcriptionSettings);

        // 更新任务状态
        taskStore.updateTask(job_id, {
          status: "queued",
          phase: "transcribing",
          message: "等待转录...",
        });

        // 延迟加载缩略图
        setTimeout(() => {
          getThumbnailUrl(job_id, true);
        }, 3000);

        successCount.value++;
      } catch (error) {
        console.error(`上传失败 [${file.name}]:`, error);
        failCount.value++;
      }
    }

    // 同步任务列表
    await taskStore.syncTasksFromBackend();

    // 显示结果
    if (failCount.value === 0) {
      ElMessage.success(`成功上传 ${successCount.value} 个文件`);
    } else {
      ElMessage.warning(
        `上传完成：${successCount.value} 成功，${failCount.value} 失败`
      );
    }

    showUploadDialog.value = false;
    selectedFile.value = null;
    uploadFiles.value = [];
    uploadRef.value?.clearFiles();
  } catch (error) {
    console.error("上传失败:", error);
    ElMessage.error(`上传失败: ${error.message || "未知错误"}`);
  } finally {
    uploading.value = false;
  }
}

// 打开编辑器
function openEditor(jobId) {
  router.push(`/editor/${jobId}`);
}

// 处理标题单击 - 用于区分单击（跳转）和双击（重命名）
let clickTimer = null;
function handleTitleClick(task) {
  if (clickTimer) {
    // 双击时清除单击定时器，让 dblclick 事件处理
    clearTimeout(clickTimer);
    clickTimer = null;
    return;
  }

  // 延迟执行单击操作，给双击留出时间
  clickTimer = setTimeout(() => {
    clickTimer = null;
    openEditor(task.job_id);
  }, 200);
}

// 删除任务
async function deleteTask(jobId) {
  try {
    await ElMessageBox.confirm(
      "确定要删除这个任务吗？此操作无法撤销。",
      "确认删除",
      {
        confirmButtonText: "删除",
        cancelButtonText: "取消",
        type: "warning",
      }
    );

    // 调用后端 API 删除任务数据
    try {
      const res = await transcriptionApi.cancelJob(jobId, true);
      // 后端现在区分“正在执行，等待取消后删除”与“已立即删除”
      if (res.pending_delete) {
        if (res.task) {
          taskStore.applyTaskSnapshot(res.task);
        } else {
          taskStore.updateTaskStatus(
            jobId,
            "canceling",
            res.message || "已请求取消并将在结束后删除",
            { isServer: false }
          );
        }
        ElMessage.info(res.message || "任务正在执行，已请求取消，结束后将删除");
        // 任务会在原子段结束后自动删除，此处不删卡片
      } else {
        // 非运行中，已立即删除
        taskStore.deleteTask(jobId);
        ElMessage.success("任务已删除");
      }
      // 无论哪种情况，刷新 input 列表以反映文件变化
      await loadInputFiles();
      setTimeout(() => {
        taskStore.syncTasksFromBackend();
      }, 500);
    } catch (error) {
      // 处理占用/正在执行的快速失败
      const status = error?.response?.status;
      const detail = error?.response?.data?.detail || error?.message;
      if (status === 409) {
        ElMessage.info(detail || "任务正在取消中，稍后再试删除");
        taskStore.updateTaskStatus(jobId, "canceling", detail, { isServer: false });
        return;
      }
      if (status === 423) {
        ElMessage.warning(detail || "当前有进程占用，请稍后再试");
        return;
      }
      console.error("删除任务失败:", error);
      ElMessage.error(`删除失败: ${detail || "未知错误"}`);
    }
  } catch (error) {
    if (error !== "cancel") {
      console.error("删除任务失败:", error);
      ElMessage.error(`删除失败: ${error.message}`);
    }
  }
}

// 格式化日期 - 显示为 YYYY-MM-DD HH:mm 格式（第二阶段修复：实时更新）
function formatDate(timestamp) {
  if (!timestamp) return "";
  const date = new Date(timestamp);

  // 格式化为 YYYY-MM-DD HH:mm
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  const hours = String(date.getHours()).padStart(2, "0");
  const minutes = String(date.getMinutes()).padStart(2, "0");

  return `${year}-${month}-${day} ${hours}:${minutes}`;
}

// 获取状态文本（与后端状态枚举保持一致）
function getStatusText(status) {
  const statusMap = {
    created: "已创建",
    queued: "排队中",
    processing: "转录中",
    pausing: "正在暂停...",
    paused: "已暂停",
    canceling: "正在取消...",  // V3.1.0
    force_canceled: "已强制取消",  // V3.1.0
    finished: "已完成",
    failed: "失败",
    canceled: "已取消",
  };
  return statusMap[status] || status;
}

// 去除文件扩展名（优先显示 title，否则显示 filename）
function getTaskDisplayName(task) {
  // 优先显示用户自定义的 title
  if (task.title) {
    return task.title;
  }

  // 否则显示文件名（去除扩展名）
  const filename = task.filename || "";
  if (!filename) return "";

  // 去除文件扩展名
  const lastDotIndex = filename.lastIndexOf(".");
  if (lastDotIndex > 0) {
    return filename.substring(0, lastDotIndex);
  }
  return filename;
}

// 开始编辑任务标题
function startEditTitle(task) {
  editingTaskId.value = task.job_id;
  editingTitle.value = task.title || getTaskDisplayName(task);
  originalTitle.value = editingTitle.value;

  // 等待 DOM 更新后聚焦输入框
  nextTick(() => {
    const inputs = document.querySelectorAll(".task-title-input");
    const input = Array.from(inputs).find(
      (el) =>
        el.closest(".task-card")?.querySelector(".task-title-input") === el
    );
    if (input) {
      input.focus();
      input.select();
    }
  });
}

// 完成编辑任务标题
async function finishEditTitle(task) {
  if (editingTaskId.value !== task.job_id) return;

  const newTitle = editingTitle.value.trim();

  // 如果标题为空，提示并恢复原名称
  if (!newTitle) {
    ElMessage.warning("任务名称不能为空");
    editingTitle.value = originalTitle.value;
    editingTaskId.value = null;
    return;
  }

  // 如果没有变化，直接关闭编辑
  if (newTitle === originalTitle.value) {
    editingTaskId.value = null;
    return;
  }

  try {
    // 调用 API 重命名任务
    const result = await transcriptionApi.renameJob(task.job_id, newTitle);

    // 更新本地 store
    if (result?.task) {
      taskStore.applyTaskSnapshot(result.task, {
        updated_at: result.task.updated_at ?? result.updated_at
      });
    } else {
      taskStore.updateTask(
        task.job_id,
        { title: newTitle },
        { updated_at: result?.updated_at }
      );
    }

    ElMessage.success("重命名成功");
  } catch (error) {
    console.error("重命名任务失败:", error);
    ElMessage.error(`重命名失败: ${error.message || "未知错误"}`);
    // 恢复原名称
    editingTitle.value = originalTitle.value;
  } finally {
    editingTaskId.value = null;
  }
}

// 取消编辑
function cancelEditTitle() {
  editingTitle.value = originalTitle.value;
  editingTaskId.value = null;
}

// 组件挂载
onMounted(() => {
  // 任务列表在 store 初始化时已自动加载 (restoreTasks)
  // 无需手动调用

  // 异步加载所有任务的缩略图（不阻塞UI）
  if (tasks.value && tasks.value.length > 0) {
    tasks.value.forEach((task) => {
      // 延迟加载缩略图，避免过多并发请求
      setTimeout(() => {
        getThumbnailUrl(task.job_id);
      }, 100);
    });
  }

  // V3.1.0: 移除重复的 SSE 订阅，由 App.vue 统一处理
  // 避免重复订阅导致的任务重复添加问题
});

// V3.1.0: 监听任务状态变化，自动加载完成任务的缩略图
// 替代原来在 SSE 订阅中的缩略图加载逻辑
watch(
  () => tasks.value,
  (newTasks, oldTasks) => {
    if (!newTasks || !oldTasks) return;

    // 检测状态变为 finished 的任务
    newTasks.forEach((newTask) => {
      const oldTask = oldTasks.find(t => t.job_id === newTask.job_id);

      // 如果任务刚完成，自动加载缩略图
      if (newTask.status === 'finished' && oldTask?.status !== 'finished') {
        console.log(`[TaskListView] 任务完成，加载缩略图: ${newTask.job_id}`);
        setTimeout(() => {
          getThumbnailUrl(newTask.job_id, true);
        }, 1000);
      }
    });
  },
  { deep: true }
);

// 获取任务缩略图（带缓存和重试机制）
async function getThumbnailUrl(jobId, forceReload = false) {
  // 强制重新加载时清除缓存
  if (forceReload && thumbnailCache.value[jobId]) {
    delete thumbnailCache.value[jobId];
  }

  // 检查缓存（避免重复请求）
  if (thumbnailCache.value[jobId] !== undefined && !forceReload) {
    return thumbnailCache.value[jobId];
  }

  // 标记为加载中
  thumbnailCache.value[jobId] = undefined;

  try {
    const result = await transcriptionApi.getThumbnail(jobId);
    const thumbnail = result.thumbnail || null;

    // 如果获取失败但视频可能还在处理中，标记为"待重试"而非永久失败
    if (!thumbnail) {
      const task = taskStore.getTask(jobId);
      // 如果任务正在处理中，保持undefined状态以便后续重试
      if (task && (task.status === "processing" || task.status === "queued")) {
        console.log(
          `[TaskListView] 任务 ${jobId} 正在处理中，稍后重试加载缩略图`
        );
        // 不缓存null，保持为undefined，允许后续重试
        return null;
      }
    }

    thumbnailCache.value[jobId] = thumbnail;
    return thumbnail;
  } catch (error) {
    console.warn(`获取缩略图失败 [${jobId}]:`, error);
    // 失败时也设置为null（而非undefined），这样至少显示占位符
    thumbnailCache.value[jobId] = null;
    return null;
  }
}

// 处理退出系统
async function handleExit() {
  try {
    await ElMessageBox.confirm(
      '确定要退出系统吗？所有更改都会自动保存',
      '确认退出',
      {
        confirmButtonText: '确定退出',
        cancelButtonText: '取消',
        type: 'warning',
      }
    )

    // 显示关闭进度
    const loading = ElLoading.service({
      text: '正在保存断点并关闭系统...',
      background: 'rgba(0, 0, 0, 0.7)'
    })

    let shutdownSuccess = false
    let cleanupReport = null

    try {
      // 调用后端 shutdown API，设置超时
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 10000) // 10秒超时
      
      try {
        const response = await systemApi.shutdownSystem({
          cleanup_temp: false,  // 不清理临时文件，保留断点数据
          force: false
        })
        
        clearTimeout(timeoutId)
        shutdownSuccess = response?.success || false
        cleanupReport = response?.cleanup_report || null
        
        if (cleanupReport) {
          console.log('[Exit] 清理报告:', cleanupReport)
        }
      } catch (e) {
        clearTimeout(timeoutId)
        // 请求可能因后端关闭而失败，这是预期行为
        console.log('[Exit] 后端已关闭或请求超时:', e.message || e)
        shutdownSuccess = true  // 如果后端已关闭，认为成功
      }
    } catch (e) {
      console.log('[Exit] 关闭请求异常:', e)
      shutdownSuccess = true  // 即使异常也认为成功（后端可能已关闭）
    }

    loading.close()

    // 显示关闭完成提示
    const message = shutdownSuccess 
      ? '系统已安全关闭，请手动关闭此浏览器标签页。'
      : '系统关闭可能未完全成功，请手动检查后台进程。'
    
    await ElMessageBox.alert(
      message,
      '关闭完成',
      { type: shutdownSuccess ? 'success' : 'warning' }
    )
    
    // 尝试关闭当前窗口（部分浏览器可能阻止）
    try {
      window.close()
    } catch (e) {
      // 忽略关闭窗口失败
    }
  } catch (error) {
    // 用户取消退出
    if (error !== 'cancel') {
      console.error('[Exit] 退出失败:', error)
    }
  }
}
</script>

<style scoped>
/* 注意：SCSS 变量已迁移到主题系统 */

/* 旧的 SCSS 文件位于 @/styles/legacy/ 供参考 */

/* 现在使用 CSS Variables（由主题系统注入） */

/* 加载动画 */
@keyframes spin {
  from {
    transform: rotate(0deg);
  }

  to {
    transform: rotate(360deg);
  }
}

.task-list-view {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background: var(--af-bg-primary);
}

/* 顶部导航栏 */
.task-header {
  position: sticky;
  top: 0;
  z-index: 200;
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 64px;
  padding: 0 24px;
  background: var(--af-bg-secondary);
  border-bottom: 1px solid var(--af-border-default);
  box-shadow: var(--af-shadow-sm);
}

.task-header .header-left {
  display: flex;
  align-items: center;
  gap: 16px;
}

.task-header .app-title {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0;
  color: var(--af-text-primary);
  font-size: 20px;
  font-weight: 600;
  cursor: pointer;
  transition: color 0.2s;
}

.task-header .app-title:hover {
  color: var(--af-accent-primary);
}

.task-header .app-icon {
  width: 28px;
  height: 28px;
  color: var(--af-accent-primary);
}

/* 主内容区 */
.task-main {
  flex: 1;
  width: 100%;
  padding: 32px 24px;
  margin: 0 auto;
  max-width: 1400px;
}

/* 空状态 */
.empty-state {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 80px 24px;
  text-align: center;
}

.empty-state .empty-icon {
  width: 120px;
  height: 120px;
  margin-bottom: 24px;
  color: var(--af-text-disabled);
  opacity: 0.5;
}

.empty-state .empty-title {
  margin: 0 0 12px;
  color: var(--af-text-primary);
  font-size: 24px;
}

.empty-state .empty-desc {
  margin: 0 0 32px;
  color: var(--af-text-secondary);
  font-size: 14px;
}

/* 任务网格 */
.task-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 24px;
}

/* 任务卡片 */
.task-card {
  overflow: hidden;
  background: var(--af-bg-secondary);
  border: 1px solid var(--af-border-default);
  border-radius: var(--af-radius-lg);
}

.task-card .task-thumbnail {
  position: relative;
  width: 100%;
  padding-top: 56.25%; /* 16:9 */
  background: var(--af-bg-tertiary);
}

.task-card .task-thumbnail .thumbnail-image {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: var(--af-bg-tertiary);
  object-fit: cover;
}

.task-card .task-thumbnail .thumbnail-placeholder {
  position: absolute;
  top: 0;
  left: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
}

.task-card .task-thumbnail .thumbnail-placeholder svg {
  width: 64px;
  height: 64px;
  color: var(--af-text-disabled);
  opacity: 0.3;
}

.task-card .task-thumbnail .thumbnail-placeholder.clickable {
  cursor: pointer;
  transition: background var(--af-transition-fast);
}

.task-card .task-thumbnail .thumbnail-placeholder.clickable:hover {
  background: rgb(0 0 0 / 5%);
}

.task-card .task-thumbnail .thumbnail-placeholder.clickable:hover svg {
  opacity: 0.5;
}

.task-card .task-thumbnail .thumbnail-loading {
  position: absolute;
  top: 0;
  left: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
  background: rgb(0 0 0 / 10%);
  cursor: pointer;
}

.task-card .task-thumbnail .thumbnail-loading .loading-spinner {
  width: 40px;
  height: 40px;
  color: var(--af-accent-primary);
  opacity: 0.6;
  animation: spin 2s linear infinite;
}

.task-card .task-thumbnail .status-overlay {
  position: absolute;
  top: 0;
  left: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
  background: rgb(0 0 0 / 60%);
}

.task-card .task-thumbnail .status-overlay .status-text {
  color: var(--af-text-on-dark);
  font-size: 14px;
  font-weight: 500;
}

.task-card .task-info {
  padding: 16px;
}

.task-card .task-info .task-title-wrapper {
  margin-bottom: 8px;
}

.task-card .task-info .task-title {
  margin: 0;
  overflow: hidden;
  color: var(--af-text-primary);
  font-size: 15px;
  font-weight: 500;
  text-overflow: ellipsis;
  white-space: nowrap;
  cursor: pointer;
}

.task-card .task-info .task-title.task-title-link:hover {
  color: var(--af-accent-primary);
  text-decoration: underline;
}

.task-card .task-info .task-title-input {
  width: 100%;
  padding: 4px 8px;
  background: var(--af-bg-primary);
  border: 1px solid var(--af-accent-primary);
  border-radius: var(--af-radius-sm);
  color: var(--af-text-primary);
  font-size: 15px;
  font-weight: 500;
  outline: none;
  box-sizing: border-box;
}

.task-card .task-info .task-title-input:focus {
  box-shadow: 0 0 0 2px rgb(var(--af-accent-primary-rgb), 0.2);
}

.task-card .task-info .task-meta {
  display: flex;
  gap: 16px;
  margin-bottom: 12px;
}

/* 通用 .el-icon 样式 - 必须在更具体的选择器之前（避免 no-descending-specificity） */
.file-list-container .loading-files .el-icon {
  font-size: 32px;
}

.transcription-settings .settings-header .el-icon {
  color: var(--af-text-muted);
  transition: transform var(--af-transition-fast);
}

.task-card .task-info .task-meta .meta-item {
  display: flex;
  align-items: center;
  gap: 4px;
  color: var(--af-text-secondary);
  font-size: 12px;
}

.task-card .task-info .task-meta .meta-item .el-icon {
  font-size: 14px;
}

.task-card .task-actions {
  display: flex;
  gap: 8px;
  padding: 0 16px 16px;
}

.task-card .task-actions .el-button {
  flex: 1;
}

/* 标签页容器 - 相对定位 */
.tabs-container {
  position: relative;
  margin-bottom: 20px;
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

/* 文件列表容器样式 */
.file-list-container {
  min-height: 300px;
}

.file-list-container .loading-files {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  gap: 12px;
  padding: 80px 24px;
  color: var(--af-text-secondary);
}

.file-list-container .empty-files {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 80px 24px;
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

/* 上传对话框样式修复（暗色模式适配） */
:deep(.el-dialog) {
  background: var(--af-bg-secondary);
  border: 1px solid var(--af-border-default);
}

:deep(.el-dialog) .el-dialog__header {
  padding: 16px 20px;
  border-bottom: 1px solid var(--af-border-default);
}

:deep(.el-dialog) .el-dialog__header .el-dialog__title {
  color: var(--af-text-primary);
  font-weight: 600;
}

:deep(.el-dialog) .el-dialog__header .el-dialog__headerbtn .el-dialog__close {
  color: var(--af-text-secondary);
}

:deep(.el-dialog) .el-dialog__header .el-dialog__headerbtn .el-dialog__close:hover {
  color: var(--af-text-primary);
}

:deep(.el-dialog) .el-dialog__body {
  padding: 20px;
  color: var(--af-text-primary);
}

:deep(.el-dialog) .el-dialog__footer {
  padding: 16px 20px;
  border-top: 1px solid var(--af-border-default);
}

:deep(.el-dialog) .el-dialog__footer .dialog-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

:deep(.el-dialog) .el-dialog__footer .dialog-footer .selection-info {
  color: var(--af-text-secondary);
  font-size: 14px;
}

:deep(.el-dialog) .el-dialog__footer .dialog-footer .footer-buttons {
  display: flex;
  gap: 8px;
}

/* 确认删除对话框按钮样式（必须在 dialog 按钮之前，避免 specificity 问题） */
:deep(.el-message-box) .el-message-box__btns .el-button {
  background: transparent;
  border: 1px solid transparent;
  color: var(--af-text-secondary);
}

:deep(.el-message-box) .el-message-box__btns .el-button:hover {
  background: var(--af-bg-tertiary);
  border-color: var(--af-border-default);
  color: var(--af-text-primary);
}

:deep(.el-message-box) .el-message-box__btns .el-button:active {
  background: var(--af-bg-quaternary);
}

:deep(.el-message-box) .el-message-box__btns .el-button--primary {
  background: transparent;
  border-color: transparent;
  color: var(--af-accent-primary);
}

:deep(.el-message-box) .el-message-box__btns .el-button--primary:hover {
  background: var(--af-bg-tertiary);
  border-color: var(--af-accent-primary);
}

/* 主操作按钮样式 */
:deep(.el-dialog) .el-dialog__footer .dialog-footer .footer-buttons .el-button {
  background: transparent;
  border: 1px solid transparent;
  color: var(--af-text-secondary);
}

:deep(.el-dialog) .el-dialog__footer .dialog-footer .footer-buttons .el-button:hover {
  background: var(--af-bg-tertiary);
  border-color: var(--af-border-default);
  color: var(--af-text-primary);
}

:deep(.el-dialog) .el-dialog__footer .dialog-footer .footer-buttons .primary-action-btn {
  background-color: var(--af-accent-primary);
  border-color: var(--af-accent-primary);
  color: var(--af-text-on-dark);
}

:deep(.el-dialog) .el-dialog__footer .dialog-footer .footer-buttons .primary-action-btn:disabled,
:deep(.el-dialog) .el-dialog__footer .dialog-footer .footer-buttons .primary-action-btn.is-disabled {
  background-color: var(--af-bg-tertiary);
  border-color: var(--af-border-default);
  color: var(--af-text-disabled);
  cursor: not-allowed;
  opacity: 0.7;
}

:deep(.el-dialog) .el-dialog__footer .dialog-footer .footer-buttons .primary-action-btn:hover:not(:disabled) {
  background-color: var(--af-accent-primary-hover);
  border-color: var(--af-accent-primary-hover);
}

/* 上传区域样式修复 */
:deep(.el-upload) .el-upload-dragger {
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

:deep(.el-upload) .el-upload__tip {
  margin-top: 8px;
  color: var(--af-text-muted);
  font-size: 12px;
}

/* 已选择文件标签样式 */
.selected-files-tags {
  max-height: 80px;
  margin-top: 12px;
  padding: 10px;
  overflow-y: auto;
  background: var(--af-bg-tertiary);
  border: 1px solid var(--af-border-default);
  border-radius: var(--af-radius-md);
}

.selected-files-tags .tags-container {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.selected-files-tags .file-tag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  max-width: 200px;
  padding: 4px 10px 4px 6px;
  background: var(--af-bg-elevated);
  border: 1px solid var(--af-border-default);
  border-radius: 20px;
  color: var(--af-text-primary);
  font-size: 12px;
  transition: all var(--af-transition-fast);
  user-select: none;
  cursor: default;
}

.selected-files-tags .file-tag .tag-close {
  flex-shrink: 0;
  width: 16px;
  height: 16px;
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

/* Tabs 样式修复 */
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

/* Table 样式修复 */
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

/* 可点击的行样式 */
:deep(.el-table).clickable-rows .el-table__body-wrapper tr {
  cursor: pointer;
}

:deep(.el-table) .el-table__body-wrapper tr:hover > td {
  background: rgb(var(--af-accent-primary-rgb, 99, 102, 241), 8%) !important;
}

/* 滚动条样式 */
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

/* Checkbox 样式 - 使用更明显的对比色 */
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

/* 空数据提示 */
:deep(.el-table) .el-table__empty-block {
  background: var(--af-bg-secondary);
}

:deep(.el-table) .el-table__empty-block .el-table__empty-text {
  color: var(--af-text-muted);
}

/* 转录设置区域 */
.transcription-settings {
  margin-top: 16px;
  overflow: hidden;
  border: 1px solid var(--af-border-default);
  border-radius: var(--af-radius-md);
}

.transcription-settings .settings-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 14px;
  background: var(--af-bg-secondary);
  transition: background var(--af-transition-fast);
  cursor: pointer;
}

.transcription-settings .settings-header:hover {
  background: var(--af-bg-tertiary);
}

.transcription-settings .settings-header span {
  color: var(--af-text-secondary);
  font-size: 14px;
  font-weight: 500;
}

.transcription-settings .settings-header .el-icon.is-expanded {
  transform: rotate(180deg);
}

.transcription-settings .settings-content {
  padding: 14px;
  background: var(--af-bg-primary);
  border-top: 1px solid var(--af-border-default);
}

.transcription-settings .setting-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.transcription-settings .setting-row:last-child {
  margin-bottom: 0;
}

.transcription-settings .setting-row label {
  min-width: 70px;
  color: var(--af-text-secondary);
  font-size: 12px;
}

/* 确认删除对话框样式修复（非按钮部分） */
:deep(.el-message-box) {
  background: var(--af-bg-secondary);
  border: 1px solid var(--af-border-default);
}

:deep(.el-message-box) .el-message-box__header {
  padding: 16px 20px 12px;
}

:deep(.el-message-box) .el-message-box__title {
  padding-left: 8px;
  color: var(--af-text-primary);
}

:deep(.el-message-box) .el-message-box__content {
  padding: 12px 20px;
  color: var(--af-text-secondary);
}

:deep(.el-message-box) .el-message-box__btns {
  padding: 12px 20px 16px;
}
</style>
