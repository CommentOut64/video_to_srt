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
        <el-button type="danger" @click="handleExit">
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
            d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V5h14v14z"
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
import PresetSelector from "@/components/editor/PresetSelector.vue"; // v3.5 预设选择器
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
    spectrum_threshold: 0.35,
    vad_filter: true,
    enable_spectral_triage: true
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

// 监听上传模式切换，自动加载文件列表
watch(uploadMode, async (newMode) => {
  if (newMode === "select" && inputFiles.value.length === 0) {
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
      // v3.5: 构建转录设置，使用 task_config 格式
      const transcriptionSettings = {
        task_config: {
          preset_id: taskConfig.value.preset_id,
          preprocessing: { ...taskConfig.value.preprocessing },
          transcription: { ...taskConfig.value.transcription },
          refinement: { ...taskConfig.value.refinement },
          compute: { ...taskConfig.value.compute }
        },
        // 保留旧版字段用于兼容
        engine: 'sensevoice',
        model: taskConfig.value.transcription.whisper_model || 'medium',
        compute_type: 'float16',
        device: 'cuda',
        batch_size: 16,
        word_timestamps: false
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
    // v3.5: 构建转录设置，使用 task_config 格式
    const transcriptionSettings = {
      task_config: {
        preset_id: taskConfig.value.preset_id,
        preprocessing: { ...taskConfig.value.preprocessing },
        transcription: { ...taskConfig.value.transcription },
        refinement: { ...taskConfig.value.refinement },
        compute: { ...taskConfig.value.compute }
      },
      // 保留旧版字段用于兼容
      engine: 'sensevoice',
      model: taskConfig.value.transcription.whisper_model || 'medium',
      compute_type: 'float16',
      device: 'cuda',
      batch_size: 16,
      word_timestamps: false
    };

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
      await transcriptionApi.cancelJob(jobId, true);
    } catch (error) {
      console.warn("调用后端删除失败，继续清理本地记录:", error);
      // 即使后端删除失败，也继续清理本地记录
      // 这有助于修复幽灵任务问题
    }

    // [V3.1.0] 优化：立即从本地删除，不再立即调用 syncTasksFromBackend
    // 依赖 SSE job_removed 事件确保一致性
    await taskStore.deleteTask(jobId);

    // [V3.1.0] 延迟同步作为兜底机制
    // 如果 SSE 事件丢失，1秒后的同步可以保证最终一致性
    setTimeout(() => {
      taskStore.syncTasksFromBackend();
    }, 1000);

    ElMessage.success("任务已删除");
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
    await transcriptionApi.renameJob(task.job_id, newTitle);

    // 更新本地 store
    taskStore.updateTask(task.job_id, {
      title: newTitle,
    });

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

<style lang="scss" scoped>
@use "@/styles/variables" as *;

// 加载动画
@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

.task-list-view {
  min-height: 100vh;
  background: var(--bg-primary);
  display: flex;
  flex-direction: column;
}

// 顶部导航栏
.task-header {
  height: 64px;
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-default);
  padding: 0 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: $z-sticky;
  box-shadow: var(--shadow-sm);

  .header-left {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .app-title {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0;
    cursor: pointer;
    transition: color 0.2s;

    &:hover {
      color: var(--primary);
    }
  }

  .app-icon {
    width: 28px;
    height: 28px;
    color: var(--primary);
  }
}

// 主内容区
.task-main {
  flex: 1;
  padding: 32px 24px;
  max-width: 1400px;
  width: 100%;
  margin: 0 auto;
}

// 空状态
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80px 24px;
  text-align: center;

  .empty-icon {
    width: 120px;
    height: 120px;
    color: var(--text-disabled);
    opacity: 0.5;
    margin-bottom: 24px;
  }

  .empty-title {
    font-size: 24px;
    color: var(--text-primary);
    margin: 0 0 12px;
  }

  .empty-desc {
    font-size: 14px;
    color: var(--text-secondary);
    margin: 0 0 32px;
  }
}

// 任务网格
.task-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 24px;
}

// 任务卡片
.task-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  overflow: hidden;

  .task-thumbnail {
    position: relative;
    width: 100%;
    padding-top: 56.25%; // 16:9
    background: var(--bg-tertiary);

    .thumbnail-image {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
      background: var(--bg-tertiary);
    }

    .thumbnail-placeholder {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;

      svg {
        width: 64px;
        height: 64px;
        color: var(--text-disabled);
        opacity: 0.3;
      }

      &.clickable {
        cursor: pointer;
        transition: background var(--transition-fast);

        &:hover {
          background: rgba(0, 0, 0, 0.05);

          svg {
            opacity: 0.5;
          }
        }
      }
    }

    .thumbnail-loading {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(0, 0, 0, 0.1);
      cursor: pointer;

      .loading-spinner {
        width: 40px;
        height: 40px;
        color: var(--primary);
        opacity: 0.6;
        animation: spin 2s linear infinite;
      }
    }

    .status-overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.6);
      display: flex;
      align-items: center;
      justify-content: center;

      .status-text {
        color: white;
        font-size: 14px;
        font-weight: 500;
      }
    }
  }

  .task-info {
    padding: 16px;

    .task-title-wrapper {
      margin-bottom: 8px;
    }

    .task-title {
      font-size: 15px;
      font-weight: 500;
      color: var(--text-primary);
      margin: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      cursor: pointer;

      &.task-title-link {
        &:hover {
          color: var(--primary);
          text-decoration: underline;
        }
      }
    }

    .task-title-input {
      width: 100%;
      font-size: 15px;
      font-weight: 500;
      color: var(--text-primary);
      background: var(--bg-primary);
      border: 1px solid var(--primary);
      border-radius: var(--radius-sm);
      padding: 4px 8px;
      outline: none;
      box-sizing: border-box;

      &:focus {
        box-shadow: 0 0 0 2px rgba(var(--primary-rgb), 0.2);
      }
    }

    .task-meta {
      display: flex;
      gap: 16px;
      margin-bottom: 12px;

      .meta-item {
        font-size: 12px;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 4px;

        .el-icon {
          font-size: 14px;
        }
      }
    }
  }

  .task-actions {
    padding: 0 16px 16px;
    display: flex;
    gap: 8px;

    .el-button {
      flex: 1;
    }
  }
}

// 标签页容器 - 相对定位
.tabs-container {
  position: relative;
  margin-bottom: 20px;

  .open-folder-btn {
    position: absolute;
    top: 8px;
    right: 0;
    color: var(--text-secondary);
    padding: 4px 8px;

    &:hover {
      color: var(--el-color-primary);
      background: var(--bg-tertiary);
    }

    &:active {
      color: var(--el-color-primary);
      background: var(--bg-quaternary);
    }
  }
}

// 文件列表容器样式
.file-list-container {
  min-height: 300px;

  .loading-files {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 80px 24px;
    gap: 12px;
    color: var(--text-secondary);

    .el-icon {
      font-size: 32px;
    }
  }

  .empty-files {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 80px 24px;
    text-align: center;

    p {
      margin: 8px 0;
      color: var(--text-secondary);
    }

    .hint {
      font-size: 14px;
      color: var(--text-disabled);
    }
  }

  .files-table {
    .filename {
      font-size: 14px;
      color: var(--text-primary);
      word-break: break-all;
    }
  }
}

// 上传对话框样式修复（暗色模式适配）
:deep(.el-dialog) {
  background: var(--bg-secondary);
  border: 1px solid var(--border-default);

  .el-dialog__header {
    border-bottom: 1px solid var(--border-default);
    padding: 16px 20px;

    .el-dialog__title {
      color: var(--text-primary);
      font-weight: 600;
    }

    .el-dialog__headerbtn {
      .el-dialog__close {
        color: var(--text-secondary);

        &:hover {
          color: var(--text-primary);
        }
      }
    }
  }

  .el-dialog__body {
    padding: 20px;
    color: var(--text-primary);
  }

  .el-dialog__footer {
    border-top: 1px solid var(--border-default);
    padding: 16px 20px;

    .dialog-footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 100%;

      .selection-info {
        font-size: 14px;
        color: var(--text-secondary);
      }

      .footer-buttons {
        display: flex;
        gap: 8px;

        // 主操作按钮样式
        .primary-action-btn {
          background-color: var(--primary);
          border-color: var(--primary);
          color: white;

          &:hover:not(:disabled) {
            background-color: var(--primary-hover);
            border-color: var(--primary-hover);
          }

          &:disabled,
          &.is-disabled {
            background-color: var(--primary-dim);
            border-color: var(--primary-dim);
            color: rgba(255, 255, 255, 0.7);
            cursor: not-allowed;
          }
        }
      }
    }
  }
}

// 上传区域样式修复
:deep(.el-upload) {
  .el-upload-dragger {
    background: var(--bg-tertiary);
    border: 2px dashed var(--border-default);
    border-radius: var(--radius-md);
    transition: all var(--transition-fast);

    &:hover {
      border-color: var(--primary);
      background: var(--bg-elevated);
    }

    .el-icon--upload {
      color: var(--text-muted);
      font-size: 48px;
      margin-bottom: 16px;
    }

    .el-upload__text {
      color: var(--text-secondary);

      em {
        color: var(--primary);
        font-style: normal;
      }
    }
  }

  .el-upload__tip {
    color: var(--text-muted);
    font-size: 12px;
    margin-top: 8px;
  }
}

// 已选择文件标签样式
.selected-files-tags {
  margin-top: 12px;
  padding: 10px;
  background: var(--bg-tertiary);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  max-height: 80px;
  overflow-y: auto;

  .tags-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .file-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px 4px 6px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-default);
    border-radius: 20px;
    font-size: 12px;
    color: var(--text-primary);
    max-width: 200px;
    transition: all var(--transition-fast);
    user-select: none;
    cursor: default;

    // &:hover {
    //   border-color: var(--primary);
    //   background: var(--bg-secondary);
    // }

    .tag-close {
      flex-shrink: 0;
      width: 16px;
      height: 16px;
      padding: 2px;
      border-radius: 50%;
      cursor: pointer;
      color: var(--text-muted);
      transition: all var(--transition-fast);

      &:hover {
        background: var(--error);
        color: white;
      }
    }

    .tag-name {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
  }
}

// Tabs 样式修复
:deep(.el-tabs) {
  .el-tabs__header {
    margin-bottom: 16px;

    .el-tabs__nav-wrap::after {
      background-color: var(--border-default);
    }

    .el-tabs__item {
      color: var(--text-secondary);

      &:hover {
        color: var(--text-primary);
      }

      &.is-active {
        color: var(--primary);
      }
    }

    .el-tabs__active-bar {
      background-color: var(--primary);
    }
  }
}

// Table 样式修复
:deep(.el-table) {
  --el-table-bg-color: var(--bg-secondary);
  --el-table-tr-bg-color: var(--bg-secondary);
  --el-table-header-bg-color: var(--bg-secondary);
  --el-table-row-hover-bg-color: rgba(var(--primary-rgb, 99, 102, 241), 0.08);
  --el-table-border-color: var(--border-default);
  --el-table-text-color: var(--text-primary);
  --el-table-header-text-color: var(--text-secondary);

  background: var(--bg-secondary);
  border-radius: var(--radius-md);
  overflow: hidden;

  // 可点击的行样式
  &.clickable-rows {
    .el-table__body-wrapper tr {
      cursor: pointer;
    }
  }

  .el-table__header-wrapper {
    th {
      background: var(--bg-secondary) !important;
      color: var(--text-secondary);
      border-bottom: 1px solid var(--border-default);
      font-weight: 500;
    }
  }

  .el-table__body-wrapper {
    background: var(--bg-secondary);

    tr {
      background: var(--bg-secondary);

      &:hover > td {
        background: rgba(var(--primary-rgb, 99, 102, 241), 0.08) !important;
      }

      td {
        border-bottom: 1px solid var(--border-light);
        color: var(--text-primary);
      }
    }
  }

  // 滚动条样式
  .el-scrollbar__bar {
    &.is-vertical {
      width: 6px;
      right: 2px;
    }

    .el-scrollbar__thumb {
      background-color: var(--text-muted);
      border-radius: 3px;
      opacity: 0.5;

      &:hover {
        opacity: 0.8;
      }
    }
  }

  // Checkbox 样式 - 使用更明显的对比色
  .el-checkbox__inner {
    background-color: var(--bg-primary);
    border-color: var(--text-muted);
    border-width: 2px;
  }

  .el-checkbox__input.is-checked .el-checkbox__inner {
    background-color: var(--primary);
    border-color: var(--primary);
  }

  .el-checkbox__input:hover .el-checkbox__inner {
    border-color: var(--primary);
  }

  // 空数据提示
  .el-table__empty-block {
    background: var(--bg-secondary);

    .el-table__empty-text {
      color: var(--text-muted);
    }
  }
}

// 转录设置区域
.transcription-settings {
  margin-top: 16px;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  overflow: hidden;

  .settings-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    background: var(--bg-secondary);
    cursor: pointer;
    transition: background var(--transition-fast);

    &:hover {
      background: var(--bg-tertiary);
    }

    span {
      font-size: 14px;
      font-weight: 500;
      color: var(--text-secondary);
    }

    .el-icon {
      color: var(--text-muted);
      transition: transform var(--transition-fast);

      &.is-expanded {
        transform: rotate(180deg);
      }
    }
  }

  .settings-content {
    padding: 14px;
    background: var(--bg-primary);
    border-top: 1px solid var(--border-default);
  }

  .setting-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;

    &:last-child {
      margin-bottom: 0;
    }

    label {
      font-size: 12px;
      color: var(--text-secondary);
      min-width: 70px;
    }
  }
}

</style>
