import { ref, watch } from "vue";
import { ElMessage } from "element-plus";
import { transcriptionApi } from "@/services/api";
import fileApi from "@/services/api/fileApi";

/**
 * 任务创建与上传流程管理
 * 设计说明：统一“直接上传”和“目录批量创建”流程，减少页面层的分支复杂度。
 */
export function useTaskUpload({ taskStore, taskConfig, getThumbnailUrl }) {
  const showUploadDialog = ref(false);
  const uploadMode = ref("upload");
  const uploading = ref(false);
  const uploadRef = ref(null);
  const selectedFile = ref(null);
  const uploadFiles = ref([]);

  const inputFiles = ref([]);
  const selectedFiles = ref([]);
  const loadingFiles = ref(false);
  const creatingBatch = ref(false);
  const fileTableRef = ref(null);

  watch(uploadMode, async (newMode) => {
    if (newMode === "select") {
      await loadInputFiles();
    }
  });

  watch(showUploadDialog, async (visible) => {
    if (visible && uploadMode.value === "select") {
      await loadInputFiles();
    }
  });

  function buildTranscriptionSettings() {
    return {
      task_config: {
        preset_id: taskConfig.value.preset_id,
        preprocessing: { ...taskConfig.value.preprocessing },
        transcription: { ...taskConfig.value.transcription },
        refinement: { ...taskConfig.value.refinement },
        compute: { ...taskConfig.value.compute },
      },
    };
  }

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

  async function handleOpenInputFolder() {
    try {
      await fileApi.openInputFolder();
      ElMessage.success("已打开input文件夹");
    } catch (error) {
      console.error("打开文件夹失败:", error);
      ElMessage.error(`打开文件夹失败: ${error.message || "未知错误"}`);
    }
  }

  function handleFileSelectionChange(selection) {
    selectedFiles.value = selection;
  }

  async function handleBatchCreate() {
    if (selectedFiles.value.length === 0) {
      ElMessage.warning("请先选择文件");
      return;
    }

    creatingBatch.value = true;
    try {
      const filenames = selectedFiles.value.map((file) => file.name);
      const result = await fileApi.createJobsBatch(filenames);

      if (result.succeeded > 0) {
        const transcriptionSettings = buildTranscriptionSettings();
        for (const job of result.jobs) {
          try {
            await transcriptionApi.startJob(job.job_id, transcriptionSettings);
          } catch (startError) {
            console.warn(`启动任务 ${job.job_id} 失败:`, startError);
          }
        }

        await taskStore.syncTasksFromBackend();
        ElMessage.success(`成功创建并启动 ${result.succeeded} 个任务`);
      }

      if (result.failed_count > 0) {
        const failedList = result.failed.map((item) => `${item.filename}: ${item.error}`).join("\n");
        ElMessage.warning({
          message: `${result.failed_count} 个文件创建失败:\n${failedList}`,
          duration: 5000,
        });
      }

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

  function formatFileSize(bytes) {
    if (bytes === 0) {
      return "0 B";
    }
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${Math.round((bytes / Math.pow(k, i)) * 100) / 100} ${sizes[i]}`;
  }

  function handleFileChange(file, fileList) {
    if (fileList.length > 5) {
      ElMessage.warning(
        '一次最多上传5个文件，如需批量处理更多文件，请将文件放入 input 目录并使用"从本地目录选择"功能'
      );
      uploadRef.value?.handleRemove(file);
      return;
    }

    const exists = uploadFiles.value.some((currentFile) => currentFile.name === file.name);
    if (exists) {
      uploadRef.value?.handleRemove(file);
      return;
    }

    uploadFiles.value = fileList.map((currentFile) => currentFile.raw || currentFile);
    selectedFile.value = uploadFiles.value.length > 0 ? uploadFiles.value[0] : null;
  }

  function removeUploadFile(index) {
    uploadFiles.value.splice(index, 1);
    selectedFile.value = uploadFiles.value.length > 0 ? uploadFiles.value[0] : null;
    if (uploadRef.value) {
      uploadRef.value.clearFiles();
      uploadFiles.value.forEach(() => {
        // 保持与旧逻辑一致：触发上传组件内部同步
      });
    }
  }

  function handleRowClick(row) {
    if (fileTableRef.value) {
      fileTableRef.value.toggleRowSelection(row);
    }
  }

  function closeUploadDialog() {
    showUploadDialog.value = false;
    selectedFile.value = null;
    uploadFiles.value = [];
    uploadRef.value?.clearFiles();
  }

  async function handleUpload() {
    if (uploadFiles.value.length === 0) {
      ElMessage.warning("请先选择视频文件");
      return;
    }

    uploading.value = true;
    let successCount = 0;
    let failCount = 0;

    try {
      const transcriptionSettings = buildTranscriptionSettings();

      for (const file of uploadFiles.value) {
        try {
          const { job_id, filename, queue_position } = await transcriptionApi.uploadFile(file, (percent) => {
            console.log(`上传进度 [${file.name}]: ${percent}%`);
          });

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

          await transcriptionApi.startJob(job_id, transcriptionSettings);

          taskStore.updateTask(job_id, {
            status: "queued",
            phase: "transcribing",
            message: "等待转录...",
          });

          setTimeout(() => {
            getThumbnailUrl(job_id, true);
          }, 3000);

          successCount += 1;
        } catch (error) {
          console.error(`上传失败 [${file.name}]:`, error);
          failCount += 1;
        }
      }

      await taskStore.syncTasksFromBackend();

      if (failCount === 0) {
        ElMessage.success(`成功上传 ${successCount} 个文件`);
      } else {
        ElMessage.warning(`上传完成：${successCount} 成功，${failCount} 失败`);
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

  return {
    showUploadDialog,
    uploadMode,
    uploading,
    uploadRef,
    selectedFile,
    uploadFiles,
    inputFiles,
    selectedFiles,
    loadingFiles,
    creatingBatch,
    fileTableRef,
    loadInputFiles,
    handleOpenInputFolder,
    handleFileSelectionChange,
    handleBatchCreate,
    formatFileSize,
    handleFileChange,
    removeUploadFile,
    handleRowClick,
    closeUploadDialog,
    handleUpload,
  };
}

