<template>
  <div
    class="dual-anchor-progress"
    :class="[`size-${size}`, statusClass]"
    :style="trackStyle"
  >
    <!-- Level 1: 异常填充层 -->
    <div v-if="status === 'error'" class="error-layer"></div>

    <!-- Level 2: 快流层 -->
    <div
      v-if="fastProgress > 0"
      class="fast-layer"
      :style="{ width: `calc(${fastProgress}% + var(--af-progress-track-height) / 2)` }"
    ></div>

    <!-- Level 3: 慢流层 -->
    <div
      v-if="slowProgress > 0"
      class="slow-layer"
      :style="{ width: `calc(${slowProgress}% + var(--af-progress-track-height) / 2)` }"
    >
      <div class="glow-tip"></div>
    </div>

    <!-- Level 4: 快流端帽层（几何占位） -->
    <div v-if="showNodes" class="fast-caps-layer">
      <div
        v-for="node in nodes"
        :key="`fast-${node.id}`"
        class="node-cap"
        :style="{ left: node.threshold + '%' }"
      ></div>
    </div>

    <!-- Level 5: 慢流端帽层（几何占位） -->
    <div v-if="showNodes" class="slow-caps-layer">
      <div
        v-for="node in nodes"
        :key="`slow-${node.id}`"
        class="node-cap"
        :style="{ left: node.threshold + '%' }"
      ></div>
    </div>

    <!-- Level 6: 节点圆点层 -->
    <div v-if="showNodes" class="dots-layer">
      <div
        v-for="node in nodes"
        :key="node.id"
        class="node-dot"
        :class="{ active: slowProgress >= node.threshold }"
        :style="{ left: node.threshold + '%' }"
      ></div>
    </div>
  </div>
</template>

<script setup>
/**
 * DualAnchorProgress - 双锚点追赶式进度条
 *
 * 核心视觉概念：
 * - 蓝色（快流）：SenseVoice 生成的草稿
 * - 绿色（慢流）：Whisper 生成的定稿，追赶并覆盖蓝色
 * - 节点：关键阶段完成点（前处理20%、转录75%、精修95%）
 */
import { computed, ref, watch, nextTick } from "vue";
import { DEFAULT_NODES, SIZE_PRESETS } from "./progress/constants";

const props = defineProps({
  /* 快流进度 0-100 */
  fastProgress: { type: Number, default: 0 },
  /* 慢流进度 0-100 */
  slowProgress: { type: Number, default: 0 },
  /* 状态 */
  status: {
    type: String,
    default: "running",
    validator: (v) => ["running", "paused", "error", "completed"].includes(v),
  },
  /* 是否显示节点 */
  showNodes: { type: Boolean, default: true },
  /* 尺寸 */
  size: {
    type: String,
    default: "md",
    validator: (v) => ["sm", "md", "lg"].includes(v),
  },
  /* 自定义节点配置 */
  nodes: { type: Array, default: () => DEFAULT_NODES },
});

/* 禁用过渡动画标志（任务切换时临时禁用） */
const disableTransition = ref(false);

/* 监听进度变化，检测任务切换 */
let lastProgress = { fast: props.fastProgress, slow: props.slowProgress };
watch(
  () => [props.fastProgress, props.slowProgress],
  ([newFast, newSlow]) => {
    const fastDiff = Math.abs(newFast - lastProgress.fast);
    const slowDiff = Math.abs(newSlow - lastProgress.slow);

    /* 如果进度变化超过 10%，认为是任务切换，临时禁用过渡 */
    if (fastDiff > 10 || slowDiff > 10) {
      disableTransition.value = true;
      nextTick(() => {
        setTimeout(() => {
          disableTransition.value = false;
        }, 50);
      });
    }

    lastProgress = { fast: newFast, slow: newSlow };
  }
);

/* 状态样式类 */
const statusClass = computed(() => ({
  "is-paused": props.status === "paused",
  "is-error": props.status === "error",
  "is-completed": props.status === "completed",
  "no-transition": disableTransition.value,
}));

/* 轨道尺寸样式（仅尺寸相关，颜色由 CSS 变量在样式中定义） */
const trackStyle = computed(() => {
  const preset = SIZE_PRESETS[props.size];
  return {
    "--af-progress-track-width": preset.trackWidth + "px",
    "--af-progress-track-height": preset.trackHeight + "px",
    "--af-progress-dot-size": preset.dotSize + "px",
  };
});
</script>

<style scoped>
.dual-anchor-progress {
  position: relative;
  overflow: hidden;
  width: var(--af-progress-track-width);
  height: var(--af-progress-track-height);
  background: var(--af-bg-secondary);
  border: 1px solid var(--af-border-default);
  border-radius: calc(var(--af-progress-track-height) / 2);
}

/* 异常填充层 */
.dual-anchor-progress .error-layer {
  position: absolute;
  inset: 0;
  z-index: 1;
  background: rgb(var(--af-accent-danger-rgb), 0.4);
  border-radius: inherit;
  animation: error-fill 0.5s ease-out;
  transform-origin: right;
}

/* 快流层 */
.dual-anchor-progress .fast-layer {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 2;
  height: 100%;
  background: var(--af-accent-primary);
  border-radius: calc(var(--af-progress-track-height) / 2);

  /* 增加过渡时长，让进度增长更丝滑 */
  transition: width 1.2s cubic-bezier(0.25, 0.8, 0.25, 1);
}

/* 慢流层 */
.dual-anchor-progress .slow-layer {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 3;
  height: 100%;
  background: var(--af-accent-success);
  border-radius: calc(var(--af-progress-track-height) / 2);

  /* 增加过渡时长，让进度增长更丝滑 */
  transition: width 1.2s cubic-bezier(0.25, 0.8, 0.25, 1);
}

/* 慢流层高光尖端 */
.dual-anchor-progress .slow-layer .glow-tip {
  position: absolute;
  top: 0;
  right: 0;
  width: 18px;
  height: 100%;
  background: linear-gradient(to right, transparent, rgb(var(--af-text-on-dark-rgb), 0.15));
  border-top-right-radius: calc(var(--af-progress-track-height) / 2);
  border-bottom-right-radius: calc(var(--af-progress-track-height) / 2);
}

/* 任务切换时禁用过渡动画 */
.dual-anchor-progress.no-transition .fast-layer,
.dual-anchor-progress.no-transition .slow-layer {
  transition: none;
}

/* 快流端帽层（几何占位，完全透明） */
.dual-anchor-progress .fast-caps-layer {
  position: absolute;
  inset: 0;
  z-index: 4;
  pointer-events: none;
}

.dual-anchor-progress .fast-caps-layer .node-cap {
  position: absolute;
  top: 50%;
  width: var(--af-progress-track-height);
  height: var(--af-progress-track-height);
  border-radius: 50%;
  transform: translate(-50%, -50%);
  opacity: 0;
}

/* 慢流端帽层（几何占位，完全透明） */
.dual-anchor-progress .slow-caps-layer {
  position: absolute;
  inset: 0;
  z-index: 5;
  pointer-events: none;
}

.dual-anchor-progress .slow-caps-layer .node-cap {
  position: absolute;
  top: 50%;
  width: var(--af-progress-track-height);
  height: var(--af-progress-track-height);
  border-radius: 50%;
  transform: translate(-50%, -50%);
  opacity: 0;
}

/* 节点圆点层 */
.dual-anchor-progress .dots-layer {
  position: absolute;
  inset: 0;
  z-index: 6;
  pointer-events: none;
}

.dual-anchor-progress .dots-layer .node-dot {
  position: absolute;
  top: 50%;
  width: var(--af-progress-dot-size);
  height: var(--af-progress-dot-size);
  background: var(--af-text-muted);
  border-radius: 50%;
  transition: background 0.3s ease, opacity 0.3s ease;

  /* 圆点居中对齐，不超出轨道 */
  transform: translate(-50%, -50%);
}

.dual-anchor-progress .dots-layer .node-dot.active {
  background: var(--af-text-inverse);

  /* 淡入动画 */
  animation: node-fade-in 0.4s ease-out;
}

/* 暂停状态 - 变灰暗 */
.dual-anchor-progress.is-paused .fast-layer,
.dual-anchor-progress.is-paused .slow-layer,
.dual-anchor-progress.is-paused .fast-caps-layer .node-cap,
.dual-anchor-progress.is-paused .slow-caps-layer .node-cap {
  filter: brightness(0.6);
}

.dual-anchor-progress.is-paused .dots-layer .node-dot {
  opacity: 0.5;
}

/* 完成状态 - 全绿 */
.dual-anchor-progress.is-completed .fast-layer {
  opacity: 0;
}

.dual-anchor-progress.is-completed .slow-layer {
  /* 原因：覆盖 :style 绑定的内联 width，内联样式优先级最高，必须用 !important */
  width: 100% !important;
}

/* 节点淡入动画 */
@keyframes node-fade-in {
  0% {
    opacity: 0;
  }

  100% {
    opacity: 1;
  }
}

/* 异常填充动画 */
@keyframes error-fill {
  from {
    transform: scaleX(0);
  }

  to {
    transform: scaleX(1);
  }
}
</style>
