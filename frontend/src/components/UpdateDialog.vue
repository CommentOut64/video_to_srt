<template>
  <el-dialog
    v-model="visible"
    title="有新的更新可用"
    width="320px"
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
import { useUpdateChecker } from "@/composables";
import systemApi from "@/services/api/systemApi";

const props = defineProps({
  // 更新信息
  updateInfo: {
    type: Object,
    default: null,
  },
});

const emit = defineEmits(["ignore", "scheduled", "updating"]);

const visible = defineModel({ type: Boolean, default: false });

const {
  ignoreCurrentUpdate,
  triggerImmediateUpdate,
  scheduleDelayedUpdate,
  hasDelayedUpdate,
} = useUpdateChecker();

const isUpdating = ref(false);
const isScheduling = ref(false);

// 是否已安排延迟更新
const hasScheduledUpdate = computed(() => hasDelayedUpdate.value);

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

<style lang="scss">
/* V3.1.1+dev.20260105.07: 与 AboutDialog 完全一致的尺寸和位置 */

/* 对话框主体样式 - 与 AboutDialog 一致 */
.update-dialog .el-dialog {
  background: var(--bg-secondary) !important;
  border: 1px solid var(--border-color) !important;
}

/* V3.1.1+dev.20260105.07: header - 与 AboutDialog 完全一致 */
.update-dialog .el-dialog__header {
  border-bottom: 1px solid var(--border-color) !important;
  padding: 8px 8px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
}

/* V3.1.1+dev.20260105.07: 标题 - 与 AboutDialog 一致 */
.update-dialog .el-dialog__title {
  color: var(--text-bright) !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  line-height: 1 !important;
}

/* V3.1.1+dev.20260105.07: 关闭按钮 - 与 AboutDialog 一致 */
.update-dialog .el-dialog__headerbtn {
  position: static !important;
  width: 24px !important;
  height: 24px !important;
  top: auto !important;
  right: auto !important;
}

/* V3.1.1+dev.20260105.07: body - 与 AboutDialog 一致 */
.update-dialog .el-dialog__body {
  padding: 24px 4px !important;
}

/* V3.1.1+dev.20260105.07: footer */
.update-dialog .el-dialog__footer {
  border-top: 1px solid var(--border-color) !important;
  padding: 12px 4px !important;
}

/* V3.1.1+dev.20260105.08: 按钮样式 - 仅覆盖 hover/active，与 AboutDialog 一致 */
.update-dialog .el-button {
  &:hover:not(:disabled) {
    background: var(--bg-hover) !important;
    border-color: var(--border-color) !important;
  }

  &:active:not(:disabled) {
    background: var(--bg-tertiary) !important;
  }
}

/* 内容区域样式 */
.update-dialog .update-content {
  display: flex !important;
  flex-direction: column !important;
  gap: 10px !important;
}

/* V3.1.1+dev.20260105.07: 版本卡片 */
.update-dialog .version-info {
  display: flex !important;
  flex-direction: column !important;
  gap: 4px !important;
  padding: 8px 10px !important;
  background: var(--bg-tertiary) !important;
  border-radius: 4px !important;
}

.update-dialog .version-row {
  display: flex !important;
  justify-content: space-between !important;
  align-items: center !important;
}

.update-dialog .version-row .label {
  color: var(--text-secondary) !important;
  font-size: 12px !important;
}

/* V3.1.1+dev.20260105.07: 版本号字体统一 */
.update-dialog .version-row .value {
  color: var(--text-normal) !important;
  font-size: 12px !important;
  font-family: monospace !important;
}

.update-dialog .version-row .value.highlight {
  color: var(--primary) !important;
  font-weight: 600 !important;
  font-size: 12px !important;
}

/* V3.1.1+dev.20260105.09: 更新内容区域 - 适应 320px 宽度 */
.update-dialog .changelog-section {
  flex: 1 !important;
  display: flex !important;
  flex-direction: column !important;
}

.update-dialog .changelog-section .changelog-label {
  color: var(--text-secondary) !important;
  font-size: 12px !important;
  margin-bottom: 6px !important;
}

.update-dialog .changelog-section .changelog-content {
  padding: 10px !important;
  background: var(--bg-tertiary) !important;
  border: 1px solid var(--border-color) !important;
  border-radius: 4px !important;
  color: var(--text-normal) !important;
  font-size: 12px !important;
  line-height: 1.6 !important;
  min-height: 136px !important;
  max-height: 136px !important;
  overflow-y: auto !important;
  white-space: pre-wrap !important;
  word-wrap: break-word !important;
  overflow-wrap: break-word !important;
}

/* 暗色滚动条 */
.update-dialog .changelog-content::-webkit-scrollbar {
  width: 6px;
}

.update-dialog .changelog-content::-webkit-scrollbar-track {
  background: var(--bg-secondary);
  border-radius: 3px;
}

.update-dialog .changelog-content::-webkit-scrollbar-thumb {
  background: var(--border-color);
  border-radius: 3px;
}

.update-dialog .changelog-content::-webkit-scrollbar-thumb:hover {
  background: var(--text-secondary);
}

/* 延迟更新提示 */
.update-dialog .delayed-notice {
  display: flex !important;
  align-items: center !important;
  gap: 8px !important;
  padding: 8px 10px !important;
  background: rgba(var(--primary-rgb), 0.1) !important;
  border: 1px solid rgba(var(--primary-rgb), 0.3) !important;
  border-radius: 4px !important;
  color: var(--primary) !important;
  font-size: 12px !important;
}

.update-dialog .delayed-notice svg {
  width: 16px !important;
  height: 16px !important;
  flex-shrink: 0 !important;
}

/* V3.1.1+dev.20260105.10: 底部按钮布局 */
.update-dialog .dialog-footer {
  display: flex !important;
  justify-content: space-between !important;
  align-items: center !important;
}

.update-dialog .dialog-footer .btn-ignore {
  margin: 0 !important;
}

.update-dialog .dialog-footer .btn-group-right {
  display: flex !important;
  gap: 0 !important;
}

/* V3.1.1+dev.20260105.10: 重置按钮组内按钮的 margin */
.update-dialog .dialog-footer .btn-group-right .el-button {
  margin-left: 4px !important;
}

.update-dialog .dialog-footer .btn-group-right .el-button:first-child {
  margin-left: 0 !important;
}
</style>
