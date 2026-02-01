<template>
  <div class="task-group" :class="`variant-${variant}`">
    <!-- 标题栏 -->
    <div class="group-header" @click="toggleCollapse">
      <div class="header-left">
        <span class="status-dot"></span>
        <span class="group-title">{{ title }}</span>
        <span class="group-count">({{ count }})</span>
      </div>
      <div class="header-right">
        <svg
          class="collapse-icon"
          :class="{ collapsed: isCollapsed }"
          viewBox="0 0 24 24"
          fill="currentColor"
        >
          <path d="M7 10l5 5 5-5z"/>
        </svg>
      </div>
    </div>

    <!-- 内容区（使用 CSS Grid 折叠） -->
    <div
      class="group-content-wrapper"
      :class="{ collapsed: isCollapsed }"
    >
      <div class="group-content">
        <slot></slot>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  title: { type: String, required: true },
  count: { type: Number, default: 0 },
  variant: {
    type: String,
    default: 'default',
    validator: (v) => ['default', 'primary', 'success', 'warning', 'danger'].includes(v)
  },
  defaultCollapsed: { type: Boolean, default: false }
})

const isCollapsed = ref(props.defaultCollapsed)

function toggleCollapse() {
  isCollapsed.value = !isCollapsed.value
}
</script>

<style scoped>
.task-group {
  width: 100%;
  background: var(--af-bg-tertiary);
  border: 1px solid var(--af-border-default);
  border-radius: 8px;
  margin-bottom: 12px;
  overflow: hidden;
  box-sizing: border-box;
}

.group-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  padding: 8px 12px;
  transition: background 0.2s;
  cursor: pointer;
  user-select: none;
  box-sizing: border-box;
}

.group-header:hover {
  background: var(--af-bg-elevated);
}

.group-header .header-left {
  display: flex;
  flex: 1;
  align-items: center;
  gap: 6px;
  min-width: 0;
}

.group-header .status-dot {
  width: 6px;
  height: 6px;
  background: var(--af-text-muted);
  border-radius: 50%;
  flex-shrink: 0;
}

.group-header .group-title {
  color: var(--af-text-primary);
  font-size: 12px;
  font-weight: 600;
}

.group-header .group-count {
  color: var(--af-text-muted);
  font-size: 11px;
}

.group-header .header-right {
  flex-shrink: 0;
}

.group-header .collapse-icon {
  width: 16px;
  height: 16px;
  color: var(--af-text-muted);
  transition: transform 0.3s ease;
}

.group-header .collapse-icon.collapsed {
  transform: rotate(-90deg);
}

/* CSS Grid 折叠动画 */
.group-content-wrapper {
  display: grid;
  width: 100%;
  transition: grid-template-rows 300ms ease-out;
  grid-template-rows: 1fr;
  box-sizing: border-box;
  min-width: 0;
  overflow: hidden;
}

.group-content-wrapper.collapsed {
  grid-template-rows: 0fr;
}

.group-content {
  width: 100%;
  padding: 0 12px 12px;
  transition: padding 300ms ease-out;
  overflow: hidden;
  box-sizing: border-box;
  min-width: 0;
}

.group-content-wrapper.collapsed .group-content {
  padding: 0;
}

/* 变体样式 */
.variant-primary .status-dot {
  background: var(--af-accent-primary);
}

.variant-success .status-dot {
  background: var(--af-accent-success);
}

.variant-warning .status-dot {
  background: var(--af-accent-warning);
}

.variant-danger .status-dot {
  background: var(--af-accent-danger);
}
</style>
