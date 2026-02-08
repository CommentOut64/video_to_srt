import { onMounted, ref, watch } from "vue";
import { transcriptionApi } from "@/services/api";

/**
 * 任务列表缩略图管理
 * 设计说明：将缓存、重试和任务状态监听集中在同一处，
 * 避免页面层继续膨胀。
 */
export function useTaskThumbnail({ tasks, taskStore }) {
  const thumbnailCache = ref({});

  watch(
    () => tasks.value?.length,
    (newLength) => {
      if (!newLength || !tasks.value) {
        return;
      }

      tasks.value.forEach((task) => {
        if (!(task.job_id in thumbnailCache.value)) {
          setTimeout(() => {
            getThumbnailUrl(task.job_id);
          }, 100);
        }
      });
    }
  );

  onMounted(() => {
    if (!tasks.value || tasks.value.length === 0) {
      return;
    }

    tasks.value.forEach((task) => {
      setTimeout(() => {
        getThumbnailUrl(task.job_id);
      }, 100);
    });
  });

  watch(
    () => tasks.value,
    (newTasks, oldTasks) => {
      if (!newTasks || !oldTasks) {
        return;
      }

      newTasks.forEach((newTask) => {
        const oldTask = oldTasks.find((task) => task.job_id === newTask.job_id);
        if (newTask.status === "finished" && oldTask?.status !== "finished") {
          console.log(`[TaskListView] 任务完成，加载缩略图: ${newTask.job_id}`);
          setTimeout(() => {
            getThumbnailUrl(newTask.job_id, true);
          }, 1000);
        }
      });
    },
    { deep: true }
  );

  async function getThumbnailUrl(jobId, forceReload = false) {
    if (forceReload && thumbnailCache.value[jobId]) {
      delete thumbnailCache.value[jobId];
    }

    if (thumbnailCache.value[jobId] !== undefined && !forceReload) {
      return thumbnailCache.value[jobId];
    }

    thumbnailCache.value[jobId] = undefined;

    try {
      const result = await transcriptionApi.getThumbnail(jobId);
      const thumbnail = result.thumbnail || null;

      if (!thumbnail) {
        const task = taskStore.getTask(jobId);
        if (task && (task.status === "processing" || task.status === "queued")) {
          console.log(`[TaskListView] 任务 ${jobId} 正在处理中，稍后重试加载缩略图`);
          return null;
        }
      }

      thumbnailCache.value[jobId] = thumbnail;
      return thumbnail;
    } catch (error) {
      console.warn(`获取缩略图失败 [${jobId}]:`, error);
      thumbnailCache.value[jobId] = null;
      return null;
    }
  }

  return {
    thumbnailCache,
    getThumbnailUrl,
  };
}

