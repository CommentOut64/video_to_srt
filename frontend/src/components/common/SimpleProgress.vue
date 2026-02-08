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
  /* 进度 0-100 */
  progress: { type: Number, default: 0 },
  /* 状态 */
  status: {
    type: String,
    default: 'normal',
    validator: v => ['normal', 'complete', 'error'].includes(v)
  },
  /* 尺寸 */
  size: {
    type: String,
    default: 'md',
    validator: v => ['sm', 'md'].includes(v)
  },
  /* 自定义宽度 */
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

<style scoped>
.simple-progress {
  overflow: hidden;
  background: var(--af-border-muted);
  border-radius: 2px;
}

.simple-progress.size-sm {
  width: 100px;
  height: 4px;
}

.simple-progress.size-md {
  width: 150px;
  height: 4px;
}

.simple-progress .progress-fill {
  height: 100%;
  border-radius: inherit;
  transition: width 0.3s ease;
}

/* 正常状态 - 蓝色 */
.simple-progress.status-normal .progress-fill {
  background: var(--af-accent-primary);
}

/* 完成状态 - 绿色 */
.simple-progress.status-complete .progress-fill {
  background: var(--af-accent-success);
}

/* 错误状态 - 红色 */
.simple-progress.status-error .progress-fill {
  background: var(--af-accent-danger);
}
</style>
