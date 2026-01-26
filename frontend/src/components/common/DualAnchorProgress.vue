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
      :style="{ width: `calc(${fastProgress}% + var(--track-height) / 2)` }"
    ></div>

    <!-- Level 3: 慢流层 -->
    <div
      v-if="slowProgress > 0"
      class="slow-layer"
      :style="{ width: `calc(${slowProgress}% + var(--track-height) / 2)` }"
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
import { COLORS, DEFAULT_NODES, SIZE_PRESETS } from "./progress/constants";

const props = defineProps({
  // 快流进度 0-100
  fastProgress: { type: Number, default: 0 },
  // 慢流进度 0-100
  slowProgress: { type: Number, default: 0 },
  // 状态
  status: {
    type: String,
    default: "running",
    validator: (v) => ["running", "paused", "error", "completed"].includes(v),
  },
  // 是否显示节点
  showNodes: { type: Boolean, default: true },
  // 尺寸
  size: {
    type: String,
    default: "md",
    validator: (v) => ["sm", "md", "lg"].includes(v),
  },
  // 自定义节点配置
  nodes: { type: Array, default: () => DEFAULT_NODES },
});

// V3.2.0+dev.20260122.02: 禁用过渡动画标志（任务切换时临时禁用）
const disableTransition = ref(false);

// V3.2.0+dev.20260122.02: 监听进度变化，检测任务切换
let lastProgress = { fast: props.fastProgress, slow: props.slowProgress };
watch(
  () => [props.fastProgress, props.slowProgress],
  ([newFast, newSlow]) => {
    const fastDiff = Math.abs(newFast - lastProgress.fast);
    const slowDiff = Math.abs(newSlow - lastProgress.slow);

    // 如果进度变化超过 10%，认为是任务切换，临时禁用过渡
    if (fastDiff > 10 || slowDiff > 10) {
      disableTransition.value = true;
      nextTick(() => {
        // 下一帧重新启用过渡
        setTimeout(() => {
          disableTransition.value = false;
        }, 50);
      });
    }

    lastProgress = { fast: newFast, slow: newSlow };
  }
);

// 状态样式类
const statusClass = computed(() => ({
  "is-paused": props.status === "paused",
  "is-error": props.status === "error",
  "is-completed": props.status === "completed",
  "no-transition": disableTransition.value,
}));

// 轨道尺寸样式
const trackStyle = computed(() => {
  const preset = SIZE_PRESETS[props.size];
  const trackHeight = preset.trackHeight;
  // V3.2.0+dev.20260122.02: 使用预设的圆点尺寸，保持圆点大小不随轨道厚度变化
  const dotSize = preset.dotSize;
  return {
    "--track-width": preset.trackWidth + "px",
    "--track-height": trackHeight + "px",
    "--dot-size": dotSize + "px",
    "--track-bg": COLORS.trackBg,
    "--track-border": COLORS.trackBorder,
    "--fast-color": COLORS.fastStream,
    "--slow-color": COLORS.slowStream,
    "--error-color": COLORS.error,
    "--node-inactive": COLORS.nodeInactive,
    "--node-active": COLORS.nodeActive,
  };
});
</script>

<style lang="scss" scoped>
.dual-anchor-progress {
  width: var(--track-width);
  height: var(--track-height);
  background: var(--track-bg);
  border: 1px solid var(--track-border);
  border-radius: calc(var(--track-height) / 2);
  position: relative;
  // 修复2: 隐藏溢出内容，确保圆角正确显示
  overflow: hidden;

  // 异常层
  .error-layer {
    position: absolute;
    inset: 0;
    background: var(--error-color);
    border-radius: inherit;
    animation: error-fill 0.5s ease-out;
    z-index: 1;
    transform-origin: right;
  }

  // V3.2.0+dev.20260122.02: 快流层（确保右侧圆角）
  .fast-layer {
    position: absolute;
    left: 0;
    top: 0;
    height: 100%;
    background: var(--fast-color);
    border-radius: calc(var(--track-height) / 2);
    border-top-right-radius: calc(var(--track-height) / 2);
    border-bottom-right-radius: calc(var(--track-height) / 2);
    // V3.2.0+dev.20260122.01: 增加过渡时长，让进度增长更丝滑
    transition: width 1.2s cubic-bezier(0.25, 0.8, 0.25, 1);
    z-index: 2;
  }

  // V3.2.0+dev.20260122.02: 慢流层（确保右侧圆角）
  .slow-layer {
    position: absolute;
    left: 0;
    top: 0;
    height: 100%;
    background: var(--slow-color);
    border-radius: calc(var(--track-height) / 2);
    border-top-right-radius: calc(var(--track-height) / 2);
    border-bottom-right-radius: calc(var(--track-height) / 2);
    // V3.2.0+dev.20260122.01: 增加过渡时长，让进度增长更丝滑
    transition: width 1.2s cubic-bezier(0.25, 0.8, 0.25, 1);
    z-index: 3;

    .glow-tip {
      position: absolute;
      right: 0;
      top: 0;
      width: 18px;
      height: 100%;
      background: linear-gradient(
        to right,
        transparent,
        rgba(255, 255, 255, 0.15)
      );
      border-top-right-radius: calc(var(--track-height) / 2);
      border-bottom-right-radius: calc(var(--track-height) / 2);
    }
  }

  // V3.2.0+dev.20260122.02: 任务切换时禁用过渡动画
  &.no-transition {
    .fast-layer,
    .slow-layer {
      transition: none;
    }
  }

  // V3.2.0+dev.20260122.02: 快流端帽层（几何占位，完全透明）
  .fast-caps-layer {
    position: absolute;
    inset: 0;
    z-index: 4;
    pointer-events: none;

    .node-cap {
      position: absolute;
      top: 50%;
      transform: translate(-50%, -50%);
      width: var(--track-height);
      height: var(--track-height);
      border-radius: 50%;
      // V3.2.0+dev.20260122.02: 端帽完全透明，仅作为几何占位
      opacity: 0;
    }
  }

  // V3.2.0+dev.20260122.02: 慢流端帽层（几何占位，完全透明）
  .slow-caps-layer {
    position: absolute;
    inset: 0;
    z-index: 5;
    pointer-events: none;

    .node-cap {
      position: absolute;
      top: 50%;
      transform: translate(-50%, -50%);
      width: var(--track-height);
      height: var(--track-height);
      border-radius: 50%;
      // V3.2.0+dev.20260122.02: 端帽完全透明，仅作为几何占位
      opacity: 0;
    }
  }

  // V3.2.0: 节点圆点层
  .dots-layer {
    position: absolute;
    inset: 0;
    z-index: 6;
    pointer-events: none;

    .node-dot {
      position: absolute;
      top: 50%;
      // 修复1: 圆点居中对齐，不超出轨道
      transform: translate(-50%, -50%);
      width: var(--dot-size);
      height: var(--dot-size);
      border-radius: 50%;
      background: var(--node-inactive);
      // V3.2.0+dev.20260122.01: 移除光晕效果，只保留颜色和透明度过渡
      transition:
        background 0.3s ease,
        opacity 0.3s ease;

      &.active {
        background: var(--node-active);
        // V3.2.0+dev.20260122.01: 移除光晕效果，只保留颜色变化和淡入动画
        animation: node-fade-in 0.4s ease-out;
      }
    }
  }

  // 修复4: 暂停状态 - 变灰暗而非改变饱和度
  &.is-paused {
    .fast-layer,
    .slow-layer,
    .fast-caps-layer .node-cap,
    .slow-caps-layer .node-cap {
      filter: brightness(0.6);
    }
    .dots-layer .node-dot {
      opacity: 0.5;
    }
  }

  // 完成状态 - 全绿
  &.is-completed {
    .fast-layer {
      opacity: 0;
    }
    .slow-layer {
      width: 100% !important;
    }
  }
}

// V3.2.0+dev.20260122.01: 节点淡入动画（移除光晕，避免超出轨道）
@keyframes node-fade-in {
  0% {
    opacity: 0;
  }
  100% {
    opacity: 1;
  }
}

// 异常填充动画
@keyframes error-fill {
  from {
    transform: scaleX(0);
  }
  to {
    transform: scaleX(1);
  }
}
</style>
