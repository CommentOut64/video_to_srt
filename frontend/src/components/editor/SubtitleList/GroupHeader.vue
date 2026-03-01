<template>
  <div
    class="group-header tw-flex tw-items-center tw-gap-2 tw-px-3 tw-py-2 tw-cursor-pointer tw-select-none"
    :class="{ 'tw-opacity-60': isCollapsed }"
    @click="handleToggleCollapse"
  >
    <!-- 颜色条 -->
    <div
      class="color-bar tw-w-1 tw-h-5 tw-rounded-full tw-flex-shrink-0"
      :style="{ backgroundColor: color }"
    />

    <!-- 展开/折叠图标 -->
    <el-icon
      class="collapse-icon tw-transition-transform tw-duration-200"
      :class="{ 'tw-rotate-90': !isCollapsed }"
    >
      <ArrowRight />
    </el-icon>

    <!-- 读音标签 -->
    <span class="reading-label tw-text-sm tw-font-medium tw-text-text-primary">
      {{ readingLabel }}
    </span>

    <!-- 条目数量 -->
    <span class="item-count tw-text-xs tw-text-text-muted">
      ({{ itemCount }})
    </span>

    <!-- 组内全选复选框 -->
    <el-checkbox
      v-if="showCheckbox"
      :model-value="isGroupAllSelected"
      :indeterminate="isGroupIndeterminate"
      size="small"
      class="tw-ml-auto"
      @click.stop
      @change="handleGroupSelectChange"
    />
  </div>
</template>

<script setup>
/**
 * 分组头部组件
 *
 * 职责：
 * - 显示读音簇标签和颜色
 * - 支持展开/折叠
 * - 支持组内全选
 */

import { ArrowRight } from '@element-plus/icons-vue'

// Props
defineProps({
  // 读音标签
  readingLabel: {
    type: String,
    required: true,
  },
  // 簇颜色
  color: {
    type: String,
    default: '#3b82f6',
  },
  // 组内条目数量
  itemCount: {
    type: Number,
    default: 0,
  },
  // 是否折叠
  isCollapsed: {
    type: Boolean,
    default: false,
  },
  // 是否显示复选框
  showCheckbox: {
    type: Boolean,
    default: true,
  },
  // 组内是否全选
  isGroupAllSelected: {
    type: Boolean,
    default: false,
  },
  // 组内是否部分选中
  isGroupIndeterminate: {
    type: Boolean,
    default: false,
  },
})

// Emits
const emit = defineEmits([
  'toggle-collapse',
  'group-select-change',
])

// 切换折叠状态
function handleToggleCollapse() {
  emit('toggle-collapse')
}

// 组选择变化
function handleGroupSelectChange(checked) {
  emit('group-select-change', checked)
}
</script>

<style scoped>
.group-header {
  background-color: var(--af-bg-tertiary);
  border-bottom: 1px solid var(--af-border-default);
}

.group-header:hover {
  background-color: var(--af-bg-secondary);
}

.collapse-icon {
  color: var(--af-text-muted);
  font-size: 12px;
}
</style>
