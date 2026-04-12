<template>
  <header class="task-header">
    <div class="header-left">
      <h1 class="app-title" @click="$emit('open-about')">
        <svg class="app-icon" viewBox="0 0 24 24" fill="currentColor">
          <path
            d="M21 3H3c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H3V5h18v14zM5 10h9v2H5zm0-3h9v2H5zm0 6h6v2H5z"
          />
        </svg>
        <span class="brand-name">
          AnchorFlux
          <span v-if="isLite" class="lite-badge">Lite</span>
        </span>
      </h1>
    </div>
    <div class="header-right">
      <el-button v-if="canRenderProjectCreateAction" @click="$emit('open-import')">
        <el-icon><Document /></el-icon>
        字幕编辑
      </el-button>
      <el-button v-if="canRenderTranscribeAction" type="primary" @click="$emit('open-upload')">
        <el-icon><Upload /></el-icon>
        视频转录
      </el-button>
      <el-button type="primary" @click="$emit('exit-system')">
        退出系统
      </el-button>
    </div>
  </header>
</template>

<script setup>
import { Upload } from "@element-plus/icons-vue";
import { Document } from "@element-plus/icons-vue";
import { selectRouteVisibility } from "@/state/capabilities/capabilitySelector";
import { IS_LITE } from "@/config/flavor";

const routeVisibility = selectRouteVisibility();
const canRenderTranscribeAction = routeVisibility.transcribeCreate;
const canRenderProjectCreateAction = routeVisibility.projectCreate;
const isLite = IS_LITE;

defineEmits(["open-about", "open-upload", "open-import", "exit-system"]);
</script>

<style scoped>
.task-header {
  flex-shrink: 0;
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

.task-header .brand-name {
  position: relative;
  display: inline-flex;
  line-height: 1;
}

.task-header .lite-badge {
  position: absolute;
  top: -9px;
  right: -24px;
  padding: 1px 4px;
  border: 1px solid var(--af-accent-primary);
  border-radius: 10px;
  color: var(--af-accent-primary);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.2px;
  line-height: 1.2;
}

.task-header .app-icon {
  width: 28px;
  height: 28px;
  color: var(--af-accent-primary);
}
</style>
