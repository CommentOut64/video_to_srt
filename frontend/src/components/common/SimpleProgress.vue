<template>
  <div
    class="simple-progress"
    :class="[`size-${size}`, `status-${status}`]"
    :style="trackStyle"
  >
    <div class="progress-fill" :style="{ width: progress + '%' }"></div>
  </div>
</template>

<script setup>
/**
 * SimpleProgress - 简单进度条
 *
 * 用于队列进度和任务卡片，无节点，单色填充
 */
import { computed } from 'vue'

const props = defineProps({
  // 进度 0-100
  progress: { type: Number, default: 0 },
  // 状态
  status: {
    type: String,
    default: 'normal',
    validator: v => ['normal', 'complete', 'error'].includes(v)
  },
  // 尺寸
  size: {
    type: String,
    default: 'md',
    validator: v => ['sm', 'md'].includes(v)
  },
  // 自定义宽度
  width: { type: [Number, String], default: null }
})

const trackStyle = computed(() => {
  const style = {}
  if (props.width) {
    style.width = typeof props.width === 'number' ? props.width + 'px' : props.width
  }
  return style
})
</script>

<style lang="scss" scoped>
.simple-progress {
  background: var(--border-muted, #21262d);
  border-radius: 2px;
  overflow: hidden;

  &.size-sm {
    height: 4px;
    width: 100px;
  }

  &.size-md {
    height: 4px;
    width: 150px;
  }

  .progress-fill {
    height: 100%;
    transition: width 0.3s ease;
    border-radius: inherit;
  }

  // 正常状态 - 蓝色
  &.status-normal .progress-fill {
    background: var(--primary, #58a6ff);
  }

  // 完成状态 - 绿色
  &.status-complete .progress-fill {
    background: var(--success, #3fb950);
  }

  // 错误状态 - 红色
  &.status-error .progress-fill {
    background: var(--danger, #f85149);
  }
}
</style>
