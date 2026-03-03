<template>
  <el-dialog
    v-model="visible"
    title="有新的更新可用"
    width="320px"
    :lock-scroll="false"
    :close-on-click-modal="!isUpdating"
    :close-on-press-escape="!isUpdating"
    :show-close="!isUpdating"
    class="update-dialog"
  >
    <div class="update-content">
      <!-- 版本信息 -->
      <div class="version-info">
        <div class="version-row">
          <span class="label">当前版本:</span>
          <span class="value">{{ updateInfo?.currentVersion || "-" }}</span>
        </div>
        <div class="version-row new-version">
          <span class="label">最新版本:</span>
          <span class="value highlight">{{
            updateInfo?.latestVersion || "-"
          }}</span>
        </div>
      </div>

      <!-- 更新日志 -->
      <div class="changelog-section" v-if="updateInfo?.changelog">
        <div class="changelog-label">更新内容:</div>
        <div class="changelog-content">{{ updateInfo.changelog }}</div>
      </div>

      <!-- 延迟更新提示 -->
      <div class="delayed-notice" v-if="hasScheduledUpdate">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path
            d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"
          />
        </svg>
        <span>已安排在重启时更新</span>
      </div>
    </div>

    <!-- V3.1.1+dev.20260105.07: 底部按钮 - 忽略在左，其余在右 -->
    <template #footer>
      <div class="dialog-footer">
        <el-button
          class="btn-ignore"
          @click="handleIgnore"
          :disabled="isUpdating || updateInfo?.forceUpdate"
          size="small"
          plain
        >
          忽略
        </el-button>
        <div class="btn-group-right">
          <el-button
            @click="handleDelayedUpdate"
            :disabled="isUpdating"
            :loading="isScheduling"
            size="small"
            plain
          >
            重启时更新
          </el-button>
          <el-button
            @click="handleImmediateUpdate"
            :loading="isUpdating"
            size="small"
            plain
          >
            立即更新
          </el-button>
        </div>
      </div>
    </template>
  </el-dialog>
</template>

<script setup>
import { ref, computed } from "vue";
import { ElMessage, ElMessageBox, ElLoading } from "element-plus";
import { useAppUpdateStore } from "@/stores/appUpdateStore";

const props = defineProps({
  // 更新信息
  updateInfo: {
    type: Object,
    default: null,
  },
});

const emit = defineEmits(["ignore", "scheduled", "updating"]);

const visible = defineModel({ type: Boolean, default: false });

const appUpdateStore = useAppUpdateStore();
appUpdateStore.initialize();
const {
  ignoreCurrentUpdate,
  triggerImmediateUpdate,
  scheduleDelayedUpdate,
} = appUpdateStore;

const isUpdating = ref(false);
const isScheduling = ref(false);

// 是否已安排延迟更新
const hasScheduledUpdate = computed(() => appUpdateStore.hasDelayedUpdate);

/**
 * 忽略本次更新
 */
async function handleIgnore() {
  ignoreCurrentUpdate();
  visible.value = false;
  emit("ignore");
  ElMessage.info("已忽略此版本，您可以在关于窗口中手动检查更新");
}

/**
 * 重启时更新
 */
async function handleDelayedUpdate() {
  isScheduling.value = true;

  try {
    const result = await scheduleDelayedUpdate();

    if (result.success) {
      ElMessage.success("更新已安排，将在下次重启时自动执行");
      visible.value = false;
      emit("scheduled");
    } else {
      ElMessage.error(result.message || "安排更新失败");
    }
  } catch (e) {
    ElMessage.error("安排更新失败: " + (e.message || "未知错误"));
  } finally {
    isScheduling.value = false;
  }
}

/**
 * 立即更新
 */
async function handleImmediateUpdate() {
  try {
    // 确认对话框
    await ElMessageBox.confirm(
      "立即更新将保存所有任务断点后关闭系统，启动器会自动完成更新并重启。确定要继续吗？",
      "确认更新",
      {
        confirmButtonText: "确定更新",
        cancelButtonText: "取消",
        type: "warning",
        lockScroll: false,
      }
    );
  } catch {
    // 用户取消
    return;
  }

  isUpdating.value = true;
  emit("updating");

  // 显示更新进度
  const loading = ElLoading.service({
    text: "正在保存断点并准备更新...",
    background: "rgba(0, 0, 0, 0.7)",
  });

  try {
    const result = await triggerImmediateUpdate();

    if (result.success) {
      loading.setText("更新即将开始，请等待系统重启...");

      // 等待后端关闭（最多 10 秒）
      await new Promise((resolve) => setTimeout(resolve, 3000));

      // 尝试关闭窗口
      try {
        window.close();
      } catch {
        // 忽略
      }
    } else {
      loading.close();
      ElMessage.error(result.message || "触发更新失败");
      isUpdating.value = false;
    }
  } catch (e) {
    loading.close();
    // 如果请求失败，可能是后端已经关闭了，这是预期行为
    console.log("[UpdateDialog] Update request ended:", e.message || e);
  }
}
</script>

<style scoped>

.update-content {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.version-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 8px 10px;
  background: var(--af-bg-tertiary);
  border-radius: 4px;
}

.version-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.version-row .label {
  color: var(--af-text-secondary);
  font-size: 12px;
}

.version-row .value {
  color: var(--af-text-normal);
  font-size: 12px;
  font-family: monospace;
}

.version-row .value.highlight {
  color: var(--af-accent-primary);
}

.changelog-section {
  display: flex;
  flex: 1;
  flex-direction: column;
}

.changelog-label {
  color: var(--af-text-secondary);
  font-size: 12px;
  margin-bottom: 6px;
}

.changelog-content {
  padding: 10px;
  background: var(--af-bg-tertiary);
  border: 1px solid var(--af-border-default);
  border-radius: 4px;
  color: var(--af-text-normal);
  font-size: 12px;
  line-height: 1.6;
  min-height: 175px;
  max-height: 175px;
  overflow-y: auto;
  white-space: pre-wrap;
  overflow-wrap: break-word;
}

/* 原因：定制滚动条外观适配深色主题 */
.changelog-content::-webkit-scrollbar {
  width: 6px;
}

.changelog-content::-webkit-scrollbar-track {
  background: var(--af-border-default);
  border-radius: 3px;
}

.changelog-content::-webkit-scrollbar-thumb {
  background: var(--af-bg-secondary);
  border-radius: 3px;
}

.changelog-content::-webkit-scrollbar-thumb:hover {
  background: var(--af-text-secondary);
}

.delayed-notice {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  background: rgb(var(--af-accent-primary-rgb), 0.1);
  border: 1px solid rgb(var(--af-accent-primary-rgb), 0.3);
  border-radius: 4px;
  color: var(--af-accent-primary);
  font-size: 12px;
}

.delayed-notice svg {
  width: 16px;
  height: 16px;
  flex-shrink: 0;
}

.dialog-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.dialog-footer .btn-ignore {
  margin: 0;
}

.dialog-footer .btn-group-right {
  display: flex;
  gap: 4px;
}
</style>
