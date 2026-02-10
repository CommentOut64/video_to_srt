import { onMounted, ref, watch } from "vue";
import { transcriptionApi } from "@/services/api";

/**
 * 任务列表缩略图管理
 * 设计说明：将缓存、重试和任务状态监听集中在同一处，
 * 避免页面层继续膨胀。
 */
export function useTaskThumbnail({ tasks, taskStore }) {
  const thumbnailCache = ref({});

  // 监听任务列表长度变化（新任务添加）
  watch(
    () => tasks.value?.length,
    (newLength) => {
      if (!newLength || !tasks.value) {
        return;
      }

      tasks.value.forEach((task) => {
        // 只加载还没缓存的任务的缩略图
        if (!(task.job_id in thumbnailCache.value)) {
          setTimeout(() => {
            getThumbnailUrl(task.job_id);
          }, 100);
        }
      });
    },
    { immediate: true } // 立即执行一次
  );

  // 组件挂载时加载所有缩略图
  onMounted(() => {
    if (!tasks.value || tasks.value.length === 0) {
      return;
    }

    console.log('[useTaskThumbnail] 组件挂载，开始加载缩略图，任务数:', tasks.value.length);
    tasks.value.forEach((task) => {
      // 只加载还没缓存的任务的缩略图
      if (!(task.job_id in thumbnailCache.value)) {
        setTimeout(() => {
          getThumbnailUrl(task.job_id);
        }, 100);
      }
    });
  });

  // 监听任务状态变化，自动重新加载完成任务的缩略图
  watch(
    () => tasks.value,
    (newTasks, oldTasks) => {
      if (!newTasks || !oldTasks) {
        return;
      }

      newTasks.forEach((newTask) => {
        const oldTask = oldTasks.find((task) => task.job_id === newTask.job_id);
        if (newTask.status === "finished" && oldTask?.status !== "finished") {
          console.log(`[useTaskThumbnail] 任务完成，重新加载缩略图: ${newTask.job_id}`);
          setTimeout(() => {
            getThumbnailUrl(newTask.job_id, true);
          }, 1000);
        }
      });
    },
    { deep: true }
  );

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
      console.log(`[useTaskThumbnail] 请求缩略图: ${jobId}`);
      const result = await transcriptionApi.getThumbnail(jobId);
      const thumbnail = result.thumbnail || null;

      // 如果获取失败但视频可能还在处理中，标记为"待重试"而非永久失败
      if (!thumbnail) {
        const task = taskStore.getTask(jobId);
        // 如果任务正在处理中，保持undefined状态以便后续重试
        if (task && (task.status === "processing" || task.status === "queued")) {
          console.log(
            `[useTaskThumbnail] 任务 ${jobId} 正在处理中，稍后重试加载缩略图`
          );
          // 不缓存null，保持为undefined，允许后续重试
          return null;
        }
      }

      console.log(`[useTaskThumbnail] 缩略图加载${thumbnail ? '成功' : '失败'}: ${jobId}`);
      thumbnailCache.value[jobId] = thumbnail;
      return thumbnail;
    } catch (error) {
      console.warn(`[useTaskThumbnail] 获取缩略图失败 [${jobId}]:`, error);
      // 失败时也设置为null（而非undefined），这样至少显示占位符
      thumbnailCache.value[jobId] = null;
      return null;
    }
  }

  return {
    thumbnailCache,
    getThumbnailUrl,
  };
}

