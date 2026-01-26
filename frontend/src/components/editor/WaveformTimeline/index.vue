<template>
  <div class="waveform-timeline" ref="containerRef" tabindex="-1">
    <!-- 缩放控制栏 -->
    <div class="timeline-header">
      <div class="zoom-controls">
        <button class="zoom-btn" @click="zoomOut" title="缩小">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 13H5v-2h14v2z" />
          </svg>
        </button>
        <div class="zoom-slider">
          <input
            type="range"
            :value="zoomLevel"
            :min="ZOOM_MIN"
            :max="ZOOM_MAX"
            :step="ZOOM_STEP"
            @input="handleZoomInput"
          />
        </div>
        <button class="zoom-btn" @click="zoomIn" title="放大">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z" />
          </svg>
        </button>
        <span class="zoom-label">{{ zoomLevel }}%</span>
        <button class="fit-btn" @click="fitToScreen" title="适应屏幕">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path
              d="M3 5v4h2V5h4V3H5c-1.1 0-2 .9-2 2zm2 10H3v4c0 1.1.9 2 2 2h4v-2H5v-4zm14 4h-4v2h4c1.1 0 2-.9 2-2v-4h-2v4zm0-16h-4v2h4v4h2V5c0-1.1-.9-2-2-2z"
            />
          </svg>
        </button>
      </div>

      <div class="time-indicator">
        <span class="current-time">{{ formatTime(currentTime) }}</span>
        <span class="separator">/</span>
        <span class="total-time">{{ formatTime(duration) }}</span>
      </div>
    </div>

    <!-- 时间轴刻度（移到波形上方） -->
    <div id="timeline" ref="timelineRef"></div>

    <!-- 波形容器 -->
    <div
      class="waveform-wrapper"
      ref="waveformWrapperRef"
      :style="{ cursor: currentMouseCursor }"
      @contextmenu="handleWaveformContextMenu"
    >
      <!-- 上半部分交互层：只处理光标拖拽，阻止Region操作 -->
      <div
        class="waveform-upper-zone"
        :class="{ 'is-region-dragging': isRegionPointerDragging }"
        @pointerdown="handleUpperZonePointerDown"
        @pointermove="handleUpperZonePointerMove"
        @pointerleave="handleWaveformPointerLeave"
      ></div>

      <!-- 下半部分：WaveSurfer 波形和 Regions -->
      <div id="waveform" ref="waveformRef"></div>

      <!-- 加载状态 -->
      <div v-if="isLoading" class="waveform-loading">
        <div class="loading-spinner"></div>
        <span>加载波形中...</span>
      </div>

      <!-- 错误状态 -->
      <div v-if="hasError" class="waveform-error">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path
            d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"
          />
        </svg>
        <span>{{ errorMessage }}</span>
        <button @click="retryLoad">重试</button>
      </div>
    </div>

    <!-- 自定义滚动条（波形图下方，最底部） -->
    <div class="custom-scrollbar" @wheel="handleScrollbarWheel">
      <div
        class="scrollbar-track"
        ref="scrollbarTrackRef"
        @mousedown="handleScrollbarMouseDown"
      >
        <div class="scrollbar-thumb" :style="scrollbarThumbStyle"></div>
      </div>
    </div>

    <!-- 右键菜单 -->
    <ContextMenu
      ref="contextMenuRef"
      :items="contextMenuItems"
      @select="handleContextMenuSelect"
    />
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted, nextTick, inject } from "vue";
import { useProjectStore } from "@/stores/projectStore";
import { usePlaybackManager } from "@/services/PlaybackManager";
import ContextMenu from "@/components/editor/ContextMenu.vue";
import { detectOverlappingSubtitles, OVERLAP_COLORS } from "@/utils/subtitleUtils";
import { useSubtitleSync } from "@/composables";
import transcriptionApi from "@/services/api/transcriptionApi";

// ============ 缩放配置常量 ============
const ZOOM_MIN = 20; // 最小缩放 20%
const ZOOM_MAX = 800; // 最大缩放 800%（全局限制，让波形明显放大）
const ZOOM_STEP = 5; // 滑块精度 5%
const ZOOM_BUTTON_STEP = 20; // 按钮步进 20%（提高步进速度）
const ZOOM_WHEEL_STEP = 10; // 滚轮步进 10%（提高滚轮缩放速度）
const ZOOM_BASE_PX_PER_SEC = 50; // 100%缩放时的基准：每秒50像素

/**
 * 简单防抖工具，避免高频拖拽导致频繁同步。
 */
function debounce(fn, delay) {
  let timer = null;
  return function (...args) {
    if (timer) {
      clearTimeout(timer);
    }
    timer = setTimeout(() => {
      fn.apply(this, args);
    }, delay);
  };
}

// Props
const props = defineProps({
  audioUrl: String,
  peaksUrl: String,
  jobId: String,
  waveColor: { type: String, default: "#58a6ff" },
  progressColor: { type: String, default: "#238636" },
  cursorColor: { type: String, default: "#f85149" },
  height: { type: Number, default: 128 },
  regionColor: { type: String, default: "rgba(88, 166, 255, 0.25)" },
  dragEnabled: { type: Boolean, default: true },
  resizeEnabled: { type: Boolean, default: true },
});

const emit = defineEmits([
  "ready",
  "region-update",
  "region-click",
  "seek",
  "zoom",
]);

// Store
const projectStore = useProjectStore();
const jobIdRef = computed(() => props.jobId || projectStore.meta.jobId);
const { onSubtitleEdit } = useSubtitleSync(jobIdRef);

// 全局播放管理器（单例）
const playbackManager = usePlaybackManager();

// 注入编辑器上下文（获取视频就绪状态）
const editorContext = inject('editorContext', { isVideoReady: computed(() => true) })
const isVideoReady = computed(() => editorContext.isVideoReady?.value ?? true)

// Refs
const containerRef = ref(null);
const waveformRef = ref(null);
const waveformWrapperRef = ref(null);
const timelineRef = ref(null);
const scrollbarTrackRef = ref(null);

// State
const zoomLevel = ref(100);
const isLoading = ref(true);
const hasError = ref(false);
const errorMessage = ref("");
const isReady = ref(false);
const isUpdatingRegions = ref(false);
const retryCount = ref(0); // 重试计数器
const maxRetries = 3; // 最大重试次数

// 自定义交互状态
const cursorDragMode = ref("hover-only"); // 'hover-only' | 'anywhere'
const isDraggingCursor = ref(false); // 是否正在拖拽光标
const isRegionPointerDragging = ref(false); // Region 是否正在被拖拽
const currentMouseCursor = ref("default"); // 当前鼠标样式
let cursorDragStartTime = 0;
let cursorPointerId = null;
let cursorPointerTarget = null;
let cursorDragGuardsAttached = false;
let regionPointerId = null;
let regionPointerTarget = null;
let regionDragGuardsAttached = false;
let regionPointerGuardEl = null;
let previousBodyUserSelect = "";
let previousBodyWebkitSelect = "";

// 自定义滚动条状态
const scrollbarThumbLeft = ref(0);
const scrollbarThumbWidth = ref(100);
const isDraggingScrollbar = ref(false);
let scrollbarDragStartX = 0;
let scrollbarDragStartScroll = 0;

// 右键菜单状态
const contextMenuRef = ref(null);
const contextMenuTarget = ref(null); // 右键点击的目标字幕ID
const contextMenuTime = ref(0); // 右键点击的时间点

// RAF节流状态
let scrollbarRafId = null;
let pendingScrollEvent = null;

// 时间同步节流状态
let lastSyncTime = 0;

// DOM查询缓存
let cachedScrollWidth = 0;
let cachedClientWidth = 0;
let cachedMaxScrollLeft = 0;

// Wavesurfer实例
let wavesurfer = null;
let regionsPlugin = null;
let regionUpdateTimer = null;

// V3.2.0+dev.20260124.04: 波形拖拽时间同步节流，避免高频写入
const debouncedRegionSync = debounce((sentenceIndex, start, end) => {
  if (sentenceIndex === undefined || sentenceIndex === null) return;
  if (projectStore.isSentenceDeleted?.(sentenceIndex)) return;
  onSubtitleEdit(sentenceIndex, { start, end });
}, 200);

// Computed
const audioSource = computed(() => {
  if (props.audioUrl) return props.audioUrl;
  if (props.jobId) return `/api/media/${props.jobId}/audio`;
  return projectStore.meta.audioPath || "";
});

const peaksSource = computed(() => {
  if (props.peaksUrl) return props.peaksUrl;
  // 移除固定samples=2000，让后端自动计算（动态采样）
  if (props.jobId) return `/api/media/${props.jobId}/peaks?samples=0`;
  return projectStore.meta.peaksPath || "";
});

const currentTime = computed(() => projectStore.player.currentTime);
const duration = computed(() => projectStore.meta.duration || 0);

// 自定义滚动条样式（根据缩放级别动态计算）
const scrollbarThumbStyle = computed(() => {
  return {
    left: `${scrollbarThumbLeft.value}%`,
    width: `${scrollbarThumbWidth.value}%`,
  };
});

// 根据视频时长计算合适的波形配置
function calculateWaveformConfig(videoDuration, containerWidth) {
  // 【关键修改】使用固定基准，而非适应容器宽度
  // 基准：每秒50px（100%缩放时），这样视频一定会超出容器产生滚动
  const basePxPerSec = ZOOM_BASE_PX_PER_SEC; // 固定基准50px/s

  // 计算建议的初始缩放级别（适应屏幕）
  // 例如：60秒视频，800px容器 → 理想缩放 = (800/60/50)*100 ≈ 27%
  const idealFitZoom = Math.round(
    (containerWidth / videoDuration / basePxPerSec) * 100
  );
  const suggestedZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, idealFitZoom));

  // 【修复】根据视频时长选择柱子配置，始终保持柱形外观
  // 之前超过30分钟会使用线条模式，导致波形偶尔变成锯齿状
  let barConfig = {};
  if (videoDuration < 60) {
    // 短视频（<1分钟）：粗柱子
    barConfig = { barWidth: 2, barGap: 1, barRadius: 2 };
  } else if (videoDuration < 300) {
    // 中等视频（1-5分钟）：中等柱子
    barConfig = { barWidth: 1.5, barGap: 0.5, barRadius: 1 };
  } else if (videoDuration < 1800) {
    // 较长视频（5-30分钟）：细柱子
    barConfig = { barWidth: 1, barGap: 0.5, barRadius: 1 };
  } else {
    // 【修复】超过30分钟的视频也使用最细柱子，保持柱形外观
    // 之前不设置 barWidth 会导致波形变成锯齿状线条
    barConfig = { barWidth: 1, barGap: 0.3, barRadius: 0.5 };
  }

  return {
    basePxPerSec, // 固定基准（50px/s）
    suggestedZoom, // 建议的初始缩放级别
    barConfig,
  };
}

// 初始化 Wavesurfer
async function initWavesurfer() {
  if (!waveformRef.value) return;

  try {
    // 动态导入 wavesurfer
    const WaveSurfer = (await import("wavesurfer.js")).default;
    const RegionsPlugin = (
      await import("wavesurfer.js/dist/plugins/regions.js")
    ).default;
    const TimelinePlugin = (
      await import("wavesurfer.js/dist/plugins/timeline.js")
    ).default;

    // 创建插件
    regionsPlugin = RegionsPlugin.create();

    const timelinePlugin = TimelinePlugin.create({
      container: timelineRef.value,
      height: 16, // 更窄的刻度区高度
      primaryLabelInterval: 10,
      secondaryLabelInterval: 5,
      primaryColor: "#6e7681",
      secondaryColor: "#484f58",
      primaryFontColor: "#8b949e",
      secondaryFontColor: "#6e7681",
      style: {
        fontSize: "10px",
        fontFamily: "var(--font-mono)",
      },
    });

    // 获取容器宽度和视频时长，计算最佳配置
    const containerWidth = containerRef.value?.offsetWidth || 800;
    const estimatedDuration = projectStore.meta.duration || 60;
    const { basePxPerSec, suggestedZoom, barConfig } = calculateWaveformConfig(
      estimatedDuration,
      containerWidth
    );

    // 创建实例
    wavesurfer = WaveSurfer.create({
      container: waveformRef.value,
      waveColor: props.waveColor,
      progressColor: props.progressColor,
      cursorColor: props.cursorColor,
      cursorWidth: 2, // 【优化】设置光标宽度为2px，更清晰
      height: props.height,
      normalize: true,
      backend: "MediaElement",
      plugins: [regionsPlugin, timelinePlugin],
      minPxPerSec: basePxPerSec, // 使用固定基准50
      scrollParent: true,
      fillParent: false, // 改为 false，允许滚动
      dragToSeek: false, // 禁用内置拖拽
      interact: false,   // 【关键】禁用波形点击跳转，光标操作仅限上半区域
      autoScroll: false, // 禁用内置自动滚动，自己实现
      autoCenter: false, // 禁用内置居中，自己实现
      hideScrollbar: true, // 【修改】隐藏wavesurfer自带滚动条，使用自定义滚动条
      ...barConfig, // 动态柱子配置
      // 静音波形音频，避免与视频声音重叠
      media: document.createElement("audio"),
    });

    // 确保 WaveSurfer 静音（音频由视频播放器控制）
    wavesurfer.setMuted(true);

    // 初始化缩放级别为建议值（通常会适应屏幕）
    zoomLevel.value = suggestedZoom;

    // 设置事件监听
    setupWavesurferEvents();
    setupRegionEvents();

    // 加载数据
    await loadAudioData();
  } catch (error) {
    console.error("初始化波形失败:", error);
    // 不显示错误，保持加载状态
    hasError.value = false;
    isLoading.value = true;
    // 启动定时检查
    startPeaksPolling();
  }
}

// 设置 Wavesurfer 事件
function setupWavesurferEvents() {
  if (!wavesurfer) return;

  wavesurfer.on("ready", () => {
    isLoading.value = false;
    isReady.value = true;
    retryCount.value = 0; // 成功加载后重置重试计数器

    // 【关键修复】防止 WaveSurfer 的 audio 元素获取焦点和响应空格键
    const audioElement = wavesurfer.getMediaElement();
    if (audioElement) {
      // 禁止通过 Tab 键聚焦
      audioElement.setAttribute('tabindex', '-1');
      // 阻止空格键的默认播放/暂停行为
      audioElement.addEventListener('keydown', (e) => {
        if (e.code === 'Space') {
          e.preventDefault();
          e.stopPropagation();
        }
      });
    }

    // 音频加载完成后，根据实际时长重新调整配置
    const actualDuration = wavesurfer.getDuration();
    const containerWidth = containerRef.value?.offsetWidth || 800;
    if (actualDuration > 0) {
      const { basePxPerSec, suggestedZoom, barConfig } =
        calculateWaveformConfig(actualDuration, containerWidth);

      // 应用建议的缩放级别
      zoomLevel.value = suggestedZoom;
      const initialPxPerSec = basePxPerSec * (suggestedZoom / 100);
      wavesurfer.zoom(initialPxPerSec);

      // 【修复】始终应用柱子配置，确保波形保持柱形外观
      wavesurfer.setOptions(barConfig);
    }

    renderSubtitleRegions();
    emit("ready");

    // 【关键】注册 WaveSurfer 到 PlaybackManager
    playbackManager.registerWaveSurfer(wavesurfer);

    // 【关键】初始化滚动条显示
    nextTick(() => {
      updateScrollbarThumb();

      // 监听波形容器的滚动事件，实时更新滚动条位置
      const wrapper = wavesurfer.getWrapper();
      const scrollContainer = wrapper?.parentElement;
      if (scrollContainer) {
        scrollContainer.addEventListener("scroll", () => {
          updateScrollbarThumb();
        });
      }
    });
  });

  // 注意：不监听 wavesurfer 的 play/pause 事件来修改 Store
  // WaveSurfer 只作为视觉组件，跟随 VideoStage 的状态
  // Store.isPlaying 由 VideoStage 和用户操作统一管理

  // 【关键修改】移除 timeupdate 的反向绑定，避免滚动触发时间跳转
  // 不要监听 timeupdate 来更新 Store，保持单向数据流：Store → WaveSurfer
  // WaveSurfer 的时间由外部 watch 同步（见第588-614行）

  // 【上下分离设计】波形下半部分已禁用所有光标交互（interact: false）
  // 光标操作仅限上半区域（waveform-upper-zone），避免用户误操作
  // Regions 插件有独立的事件系统，字幕块拖拽/调整不受影响

  wavesurfer.on("zoom", (minPxPerSec) => {
    const newZoom = Math.round((minPxPerSec / ZOOM_BASE_PX_PER_SEC) * 100);
    // 限制在全局范围内
    zoomLevel.value = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, newZoom));
    emit("zoom", zoomLevel.value);

    // 【关键】缩放时更新滚动条
    nextTick(() => {
      updateScrollbarThumb();
    });
  });

  wavesurfer.on("error", (error) => {
    console.error("Wavesurfer error:", error);
    // 不显示错误，保持加载状态
    hasError.value = false;
    isLoading.value = true;

    // 自动重试机制
    if (retryCount.value < maxRetries) {
      retryCount.value++;
      console.log(
        `[WaveformTimeline] 自动重试 ${retryCount.value}/${maxRetries}`
      );

      // 延迟1秒后重试
      setTimeout(() => {
        loadAudioData();
      }, 1000);
    } else {
      // 达到最大重试次数，启动定时检查
      console.log("[WaveformTimeline] 达到最大重试次数，启动定时检查");
      startPeaksPolling();
    }
  });
}

// 设置 Region 事件
// V3.1.1+dev.20260106.03: 集成重叠检测，region-updated 时检测并标记重叠区域
function setupRegionEvents() {
  if (!regionsPlugin) return;

  // WaveSurfer.js 7.x 使用 'region-updated' 事件
  regionsPlugin.on("region-updated", (region) => {
    if (isUpdatingRegions.value) return;
    projectStore.updateSubtitle(
      region.id,
      {
        start: region.start,
        end: region.end,
      },
      { isUserEdit: true }
    );
    // V3.2.0+dev.20260124.04: 波形拖拽同步到后端（节流）
    const subtitle = projectStore.subtitles.find((s) => s.id === region.id);
    if (subtitle && subtitle.sentenceIndex !== undefined) {
      debouncedRegionSync(subtitle.sentenceIndex, region.start, region.end);
    }
    emit("region-update", region);

    // V3.1.1+dev.20260106.03: 拖拽结束后检测并标记重叠区域
    checkAndMarkOverlaps();
  });

  regionsPlugin.on("region-clicked", (region, e) => {
    e.stopPropagation();
    projectStore.view.selectedSubtitleId = region.id;
    // 使用 PlaybackManager 进行跳转，确保视频和波形同步
    playbackManager.seekTo(region.start);
    if (wavesurfer) wavesurfer.play();
    emit("region-click", region);
  });

  // V3.1.1+dev.20260106.03: region-in/out 需要考虑重叠状态
  regionsPlugin.on("region-in", (region) => {
    // 如果是重叠区域，保持警示色的 hover 状态
    const overlappingIds = detectOverlappingSubtitles(projectStore.subtitles);
    if (overlappingIds.has(region.id)) {
      region.setOptions({ color: "rgba(248, 81, 73, 0.5)" });  // 重叠区域 hover：更深的红色
    } else {
      region.setOptions({ color: OVERLAP_COLORS.hover });
    }
  });

  regionsPlugin.on("region-out", (region) => {
    // 恢复时需要判断是否为重叠区域
    const overlappingIds = detectOverlappingSubtitles(projectStore.subtitles);
    const isSelected = region.id === projectStore.view.selectedSubtitleId;

    if (overlappingIds.has(region.id)) {
      region.setOptions({ color: OVERLAP_COLORS.error });
    } else if (isSelected) {
      region.setOptions({ color: OVERLAP_COLORS.selected });
    } else {
      region.setOptions({ color: props.regionColor });
    }
  });
}

/**
 * V3.1.1+dev.20260106.03: 检测并标记重叠区域
 * 只修改重叠区域的颜色，不影响其他区域
 */
function checkAndMarkOverlaps() {
  if (!regionsPlugin || !isReady.value) return;

  const overlappingIds = detectOverlappingSubtitles(projectStore.subtitles);
  const regions = regionsPlugin.getRegions();

  if (!regions || regions.length === 0) return;

  regions.forEach(region => {
    const isOverlapping = overlappingIds.has(region.id);
    const isSelected = region.id === projectStore.view.selectedSubtitleId;

    if (isOverlapping) {
      // 重叠区域使用警示色
      region.setOptions({ color: OVERLAP_COLORS.error });
    } else if (isSelected) {
      // 选中区域使用选中色
      region.setOptions({ color: OVERLAP_COLORS.selected });
    } else {
      // 正常区域恢复正常色
      region.setOptions({ color: props.regionColor });
    }
  });

  // 如果存在重叠，记录日志
  if (overlappingIds.size > 0) {
    console.warn(`[WaveformTimeline] 检测到 ${overlappingIds.size} 个重叠区域:`, Array.from(overlappingIds));
  }
}

// 加载音频数据
async function loadAudioData() {
  if (!audioSource.value) {
    isLoading.value = false;
    return;
  }

  try {
    // 尝试加载峰值数据
    if (peaksSource.value) {
      const response = await fetch(peaksSource.value);
      if (response.ok) {
        const data = await response.json();
        wavesurfer.load(audioSource.value, data.peaks, data.duration);
        return;
      }
    }

    // 降级：直接加载音频
    wavesurfer.load(audioSource.value);
  } catch (error) {
    console.error("加载音频失败:", error);
    // 尝试直接加载音频
    wavesurfer.load(audioSource.value);
  }
}

// 定时检查波形数据是否可用
let peaksCheckTimer = null;
function startPeaksPolling() {
  if (peaksCheckTimer) return; // 避免重复启动

  console.log('[WaveformTimeline] 启动波形数据定时检查');
  peaksCheckTimer = setInterval(async () => {
    try {
      // 直接尝试加载波形数据
      if (peaksSource.value) {
        const response = await fetch(peaksSource.value);
        if (response.ok) {
          console.log('[WaveformTimeline] 波形数据已可用，自动重新加载');
          stopPeaksPolling();
          retryCount.value = 0;
          hasError.value = false;
          isLoading.value = true;
          await loadAudioData();
        }
      }
    } catch (e) {
      // 继续等待
      console.debug('[WaveformTimeline] 波形数据尚未可用，继续等待...');
    }
  }, 2000); // 每2秒检查一次
}

function stopPeaksPolling() {
  if (peaksCheckTimer) {
    clearInterval(peaksCheckTimer);
    peaksCheckTimer = null;
  }
}

// 渲染字幕区域
// V3.1.0: 增强日志，便于调试 regions 消失问题
// V3.1.1+dev.20260106.03: 渲染时检测重叠并标记颜色
function renderSubtitleRegions() {
  // 前置检查
  if (!isReady.value) {
    console.warn('[WaveformTimeline] renderSubtitleRegions: 波形未就绪，跳过渲染');
    return;
  }
  if (!regionsPlugin) {
    console.warn('[WaveformTimeline] renderSubtitleRegions: regions 插件未加载，跳过渲染');
    return;
  }

  const subtitleCount = projectStore.subtitles.length;
  console.log(`[WaveformTimeline] renderSubtitleRegions: 开始渲染 ${subtitleCount} 个 regions`);

  if (subtitleCount === 0) {
    console.log('[WaveformTimeline] renderSubtitleRegions: 无字幕数据，清除 regions');
    regionsPlugin.clearRegions();
    return;
  }

  // V3.1.1+dev.20260106.03: 预先检测重叠区域
  const overlappingIds = detectOverlappingSubtitles(projectStore.subtitles);
  if (overlappingIds.size > 0) {
    console.warn(`[WaveformTimeline] 检测到 ${overlappingIds.size} 个重叠区域`);
  }

  isUpdatingRegions.value = true;
  regionsPlugin.clearRegions();

  let addedCount = 0;
  projectStore.subtitles.forEach((subtitle) => {
    if (subtitle.start === undefined || subtitle.end === undefined) {
      console.warn(`[WaveformTimeline] 跳过无效字幕: id=${subtitle.id}, start=${subtitle.start}, end=${subtitle.end}`);
      return;
    }
    const isSelected = subtitle.id === projectStore.view.selectedSubtitleId;
    const isOverlapping = overlappingIds.has(subtitle.id);

    // V3.1.1+dev.20260106.03: 根据重叠状态决定颜色
    let regionColor = props.regionColor;
    if (isOverlapping) {
      regionColor = OVERLAP_COLORS.error;  // 重叠区域优先显示警示色
    } else if (isSelected) {
      regionColor = OVERLAP_COLORS.selected;
    }

    regionsPlugin.addRegion({
      id: subtitle.id,
      start: subtitle.start,
      end: subtitle.end,
      color: regionColor,
      drag: props.dragEnabled,
      resize: props.resizeEnabled,
    });
    addedCount++;
  });

  console.log(`[WaveformTimeline] renderSubtitleRegions: 成功添加 ${addedCount}/${subtitleCount} 个 regions`);

  setTimeout(() => {
    isUpdatingRegions.value = false;
  }, 100);
}

// ============ 智能锚点缩放策略（性能优化版）============

// 缓存 DOM 引用，避免重复查询
let cachedWrapper = null;
let cachedScrollContainer = null;

/**
 * 获取缓存的滚动容器（避免频繁 DOM 查询）
 */
function getScrollContainer() {
  if (cachedScrollContainer && cachedScrollContainer.isConnected) {
    return cachedScrollContainer;
  }
  if (!wavesurfer) return null;
  cachedWrapper = wavesurfer.getWrapper();
  if (!cachedWrapper) return null;
  cachedScrollContainer = cachedWrapper.parentElement;
  return cachedScrollContainer;
}

/**
 * 判断播放头当前是否在可视范围内
 */
function isPlayheadInViewport() {
  const scrollContainer = getScrollContainer();
  if (!scrollContainer) return false;

  const currentPxPerSec = (zoomLevel.value / 100) * ZOOM_BASE_PX_PER_SEC;
  const playheadX = projectStore.player.currentTime * currentPxPerSec;
  const { scrollLeft, clientWidth } = scrollContainer;

  // 留一点边距容错 (10px)
  return playheadX >= scrollLeft - 10 && playheadX <= scrollLeft + clientWidth + 10;
}

/**
 * 获取光标相对于视口的坐标
 */
function getPlayheadRelativeX() {
  const scrollContainer = getScrollContainer();
  if (!scrollContainer) return 0;

  const currentPxPerSec = (zoomLevel.value / 100) * ZOOM_BASE_PX_PER_SEC;
  const playheadTotalX = projectStore.player.currentTime * currentPxPerSec;
  return playheadTotalX - scrollContainer.scrollLeft;
}

// 滚动条更新防抖
let scrollbarUpdateTimer = null;
function debouncedUpdateScrollbar() {
  if (scrollbarUpdateTimer) return;
  scrollbarUpdateTimer = setTimeout(() => {
    scrollbarUpdateTimer = null;
    updateScrollbarThumb();
  }, 16); // ~1帧
}

/**
 * 锚点缩放核心算法（性能优化版）
 * @param {number} targetZoom - 目标缩放比例 (ZOOM_MIN - ZOOM_MAX)
 * @param {number} anchorPx - 锚点相对于视口左侧的像素位置
 */
function setZoomWithAnchor(targetZoom, anchorPx) {
  if (!wavesurfer || !containerRef.value) return;

  const scrollContainer = getScrollContainer();
  if (!scrollContainer) return;
  
  // 1. 记录缩放前的状态（一次性读取，减少 reflow）
  const oldPxPerSec = (zoomLevel.value / 100) * ZOOM_BASE_PX_PER_SEC;
  const oldScroll = scrollContainer.scrollLeft;
  
  // 计算锚点对应的"绝对时间点"
  const anchorTime = (oldScroll + anchorPx) / oldPxPerSec;

  // 2. 应用新的缩放
  const clampedZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, targetZoom));
  
  // 如果缩放值没变，直接返回
  if (clampedZoom === zoomLevel.value) return;
  
  zoomLevel.value = clampedZoom;
  const newPxPerSec = (clampedZoom / 100) * ZOOM_BASE_PX_PER_SEC;
  
  // 调用 wavesurfer 进行缩放
  wavesurfer.zoom(newPxPerSec);
  projectStore.view.zoomLevel = clampedZoom;

  // 3. 计算新的滚动位置
  const newScroll = Math.max(0, (anchorTime * newPxPerSec) - anchorPx);

  // 4. 使用 RAF 确保在下一帧设置滚动位置（避免 nextTick 开销）
  requestAnimationFrame(() => {
    scrollContainer.scrollLeft = newScroll;
    // 延迟更新滚动条，降低优先级
    debouncedUpdateScrollbar();
  });
}

/**
 * 统一的缩放处理函数 (按钮/滑块通用)
 * 策略：
 * - 播放中：始终跟随光标
 * - 暂停且光标可见：锚定光标
 * - 暂停且光标不可见：锚定视口中心
 */
function handleZoomWithSmartAnchor(targetZoom) {
  const scrollContainer = getScrollContainer();
  if (!scrollContainer) return;

  let anchorPx; // 相对于视口左侧的像素位置

  // 策略判断（使用缓存的 scrollContainer）
  if (projectStore.player.isPlaying) {
    // A. 播放中：始终跟随光标
    anchorPx = getPlayheadRelativeX();
  } else if (isPlayheadInViewport()) {
    // B. 暂停且光标可见：锚定光标
    anchorPx = getPlayheadRelativeX();
  } else {
    // C. 暂停且光标不可见：锚定视口中心
    anchorPx = scrollContainer.clientWidth / 2;
  }

  // 执行锚点缩放
  setZoomWithAnchor(targetZoom, anchorPx);
}

// 缩放控制 - 绑定到滑块输入事件（添加节流）
let lastSliderZoomTime = 0;
const SLIDER_THROTTLE_MS = 16; // ~60fps

function handleZoomInput(e) {
  const now = performance.now();
  if (now - lastSliderZoomTime < SLIDER_THROTTLE_MS) return;
  lastSliderZoomTime = now;
  
  const value = parseInt(e.target.value);
  handleZoomWithSmartAnchor(value);
}

// 保留原有的 setZoom 供内部使用（如初始化、fitToScreen等）
function setZoom(value) {
  if (!wavesurfer) return;
  const clampedValue = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, value));
  zoomLevel.value = clampedValue;
  const minPxPerSec = (clampedValue / 100) * ZOOM_BASE_PX_PER_SEC;
  wavesurfer.zoom(minPxPerSec);
  projectStore.view.zoomLevel = clampedValue;
}

// 绑定到放大按钮
function zoomIn() {
  const newValue = Math.min(ZOOM_MAX, zoomLevel.value + ZOOM_BUTTON_STEP);
  handleZoomWithSmartAnchor(newValue);
}

// 绑定到缩小按钮
function zoomOut() {
  const newValue = Math.max(ZOOM_MIN, zoomLevel.value - ZOOM_BUTTON_STEP);
  handleZoomWithSmartAnchor(newValue);
}

function fitToScreen() {
  if (!wavesurfer || !containerRef.value) return;
  const containerWidth = containerRef.value.offsetWidth - 32;
  const audioDuration = wavesurfer.getDuration();
  if (audioDuration > 0) {
    // 计算适合屏幕的缩放级别
    const idealZoom = Math.round(
      (containerWidth / audioDuration / ZOOM_BASE_PX_PER_SEC) * 100
    );
    // 限制在全局范围内
    const fitZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, idealZoom));

    // fitToScreen 使用原始 setZoom（不需要锚点，因为是全局适配）
    setZoom(fitZoom);

    // 【修复】根据时长动态调整柱子配置，始终保持柱形外观
    const { barConfig } = calculateWaveformConfig(
      audioDuration,
      containerWidth
    );
    // barConfig 现在始终非空，直接应用
    wavesurfer.setOptions(barConfig);
  }
}

// 重试加载（手动重试时重置计数器）
function retryLoad() {
  hasError.value = false;
  errorMessage.value = "";
  isLoading.value = true;
  retryCount.value = 0; // 手动重试时重置计数器
  loadAudioData();
}

// ============ 智能跟随滚动逻辑 ============

/**
 * 智能跟随滚动：90%边缘触发，翻页式滚动
 * 仅在播放时调用，暂停时不调用
 */
function smartScrollFollow() {
  if (!wavesurfer || !isReady.value) return;

  const wrapper = wavesurfer.getWrapper();
  if (!wrapper) return;

  const scrollContainer = wrapper.parentElement;
  if (!scrollContainer) return;

  const currentTime = projectStore.player.currentTime;
  const duration = wavesurfer.getDuration();
  if (!duration) return;

  // 计算光标在波形中的绝对位置（像素）
  const pxPerSec = (zoomLevel.value / 100) * ZOOM_BASE_PX_PER_SEC;
  const cursorAbsoluteX = currentTime * pxPerSec;

  // 获取视口信息
  const viewportWidth = scrollContainer.clientWidth;
  const scrollLeft = scrollContainer.scrollLeft;
  const viewportRight = scrollLeft + viewportWidth;

  // 计算光标相对于视口的位置
  const cursorRelativeX = cursorAbsoluteX - scrollLeft;

  // 90%边缘触发阈值
  const rightEdgeThreshold = viewportWidth * 0.9;

  // 【翻页式滚动逻辑】
  if (cursorRelativeX > rightEdgeThreshold) {
    // 光标快要跑出右边缘（超过90%位置）
    // 翻页：将光标移至视口10%位置
    const newScrollLeft = cursorAbsoluteX - viewportWidth * 0.1;
    scrollContainer.scrollLeft = Math.max(0, newScrollLeft);
    // 更新滚动条
    updateScrollbarThumb();
  } else if (cursorAbsoluteX < scrollLeft) {
    // 光标在左侧外面（用户可能手动seek回去）
    // 翻页：将光标移至视口10%位置
    const newScrollLeft = cursorAbsoluteX - viewportWidth * 0.1;
    scrollContainer.scrollLeft = Math.max(0, newScrollLeft);
    // 更新滚动条
    updateScrollbarThumb();
  }
}

// 播放时的RAF循环（用于智能跟随）
let followRafId = null;

function startSmartFollow() {
  if (followRafId) return; // 避免重复启动

  const loop = () => {
    if (projectStore.player.isPlaying && isReady.value) {
      smartScrollFollow();
      followRafId = requestAnimationFrame(loop);
    } else {
      followRafId = null; // 暂停时停止循环
    }
  };

  loop();
}

function stopSmartFollow() {
  if (followRafId) {
    cancelAnimationFrame(followRafId);
    followRafId = null;
  }
}

// ============ 自定义交互逻辑（上下半区域分离）============

// 【指针守护】全局监听指针事件，确保拖拽过程中不会因为冒泡或焦点变化而提前结束
function attachCursorDragGuards() {
  if (cursorDragGuardsAttached) return;
  cursorDragGuardsAttached = true;
  document.addEventListener("pointermove", handleCursorDragMove, true);
  document.addEventListener("pointerup", handleCursorPointerEnd, true);
  document.addEventListener("pointercancel", handleCursorPointerCancel, true);
  window.addEventListener("blur", handleCursorPointerCancel, true);
}

function detachCursorDragGuards() {
  if (!cursorDragGuardsAttached) return;
  cursorDragGuardsAttached = false;
  document.removeEventListener("pointermove", handleCursorDragMove, true);
  document.removeEventListener("pointerup", handleCursorPointerEnd, true);
  document.removeEventListener("pointercancel", handleCursorPointerCancel, true);
  window.removeEventListener("blur", handleCursorPointerCancel, true);
}

function handleCursorPointerEnd(e) {
  if (!isDraggingCursor.value) return;
  if (
    typeof e?.pointerId === "number" &&
    cursorPointerId !== null &&
    e.pointerId !== cursorPointerId
  ) {
    return;
  }
  handleCursorDragEnd();
}

function handleCursorPointerCancel(e) {
  if (!isDraggingCursor.value) return;
  if (
    typeof e?.pointerId === "number" &&
    cursorPointerId !== null &&
    e.pointerId !== cursorPointerId
  ) {
    return;
  }
  handleCursorDragEnd();
}

// 【正文拖拽保护】Region 拖拽时关闭上层遮罩，避免事件被截断
function disableBodySelection() {
  if (typeof document === "undefined" || !document.body) return;
  previousBodyUserSelect = document.body.style.userSelect;
  previousBodyWebkitSelect = document.body.style.webkitUserSelect;
  document.body.style.userSelect = "none";
  document.body.style.webkitUserSelect = "none";
}

function restoreBodySelection() {
  if (typeof document === "undefined" || !document.body) return;
  document.body.style.userSelect = previousBodyUserSelect;
  document.body.style.webkitUserSelect = previousBodyWebkitSelect;
}

function attachRegionDragGuards() {
  if (regionDragGuardsAttached) return;
  regionDragGuardsAttached = true;
  document.addEventListener("pointerup", handleRegionPointerUp, true);
  document.addEventListener("pointercancel", handleRegionPointerCancel, true);
  window.addEventListener("blur", handleRegionPointerCancel, true);
  disableBodySelection();
}

function detachRegionDragGuards() {
  if (!regionDragGuardsAttached) return;
  regionDragGuardsAttached = false;
  document.removeEventListener("pointerup", handleRegionPointerUp, true);
  document.removeEventListener("pointercancel", handleRegionPointerCancel, true);
  window.removeEventListener("blur", handleRegionPointerCancel, true);
  isRegionPointerDragging.value = false;
  regionPointerId = null;
  regionPointerTarget = null;
  restoreBodySelection();
}

function handleRegionPointerUp(e) {
  if (
    typeof e?.pointerId === "number" &&
    regionPointerId !== null &&
    e.pointerId !== regionPointerId
  ) {
    return;
  }
  finalizeRegionPointerDrag();
}

function handleRegionPointerCancel(e) {
  if (
    typeof e?.pointerId === "number" &&
    regionPointerId !== null &&
    e.pointerId !== regionPointerId
  ) {
    return;
  }
  finalizeRegionPointerDrag();
}

function finalizeRegionPointerDrag() {
  if (!isRegionPointerDragging.value) return;

  if (
    regionPointerTarget &&
    typeof regionPointerId === "number" &&
    typeof regionPointerTarget.releasePointerCapture === "function"
  ) {
    try {
      if (regionPointerTarget.hasPointerCapture?.(regionPointerId)) {
        regionPointerTarget.releasePointerCapture(regionPointerId);
      }
    } catch (error) {
      console.debug("[WaveformTimeline] 释放 Region 指针捕获失败:", error);
    }
  }

  regionPointerId = null;
  regionPointerTarget = null;
  isRegionPointerDragging.value = false;
  detachRegionDragGuards();
}

function handleRegionPointerDown(e) {
  if (!wavesurfer || !isReady.value) return;
  if (e.pointerType === "mouse" && e.button !== 0) return;

  const target = e.target;
  if (!(target instanceof HTMLElement)) return;

  const regionEl = target.closest('[part*="region"]');
  if (!(regionEl instanceof HTMLElement)) return;

  regionPointerId = e.pointerId;
  regionPointerTarget = regionEl;
  isRegionPointerDragging.value = true;

  if (typeof regionEl.setPointerCapture === "function") {
    try {
      regionEl.setPointerCapture(e.pointerId);
    } catch (error) {
      console.debug("[WaveformTimeline] Region 指针捕获失败:", error);
    }
  }

  attachRegionDragGuards();
}

function setupRegionPointerGuards() {
  if (regionPointerGuardEl || !waveformRef.value) return;
  regionPointerGuardEl = waveformRef.value;
  regionPointerGuardEl.addEventListener("pointerdown", handleRegionPointerDown, true);
}

function teardownRegionPointerGuards() {
  if (!regionPointerGuardEl) return;
  regionPointerGuardEl.removeEventListener("pointerdown", handleRegionPointerDown, true);
  regionPointerGuardEl = null;
}

/**
 * 获取鼠标点击的时间位置
 * @param {number} clientX - 鼠标的 clientX 坐标
 */
function getTimeFromClientX(clientX) {
  if (!wavesurfer) return 0;

  const wrapper = wavesurfer.getWrapper();
  if (!wrapper) return 0;

  const scrollContainer = wrapper.parentElement;
  if (!scrollContainer) return 0;

  // 获取滚动容器相对于视口的位置
  const containerRect = scrollContainer.getBoundingClientRect();

  // 鼠标相对于滚动容器左边缘的位置
  const mouseRelativeX = clientX - containerRect.left;

  // 加上滚动偏移得到在波形中的绝对位置
  const absoluteX = mouseRelativeX + scrollContainer.scrollLeft;

  // 计算时间
  const pxPerSec = (zoomLevel.value / 100) * ZOOM_BASE_PX_PER_SEC;
  const duration = wavesurfer.getDuration();

  const time = absoluteX / pxPerSec;
  return Math.max(0, Math.min(time, duration));
}

/**
 * 获取光标当前的X位置（像素）
 */
function getCursorX() {
  if (!wavesurfer) return 0;

  const currentTime = projectStore.player.currentTime;
  const pxPerSec = (zoomLevel.value / 100) * ZOOM_BASE_PX_PER_SEC;
  return currentTime * pxPerSec;
}

/**
 * 检查鼠标是否在光标附近（用于hover-only模式）
 * @param {number} clientX - 鼠标的 clientX 坐标
 */
function isMouseNearCursor(clientX, threshold = 10) {
  if (!wavesurfer) return false;

  const wrapper = wavesurfer.getWrapper();
  if (!wrapper) return false;

  const scrollContainer = wrapper.parentElement;
  if (!scrollContainer) return false;

  // 获取滚动容器相对于视口的位置
  const containerRect = scrollContainer.getBoundingClientRect();

  // 鼠标在波形中的绝对位置
  const mouseAbsoluteX =
    clientX - containerRect.left + scrollContainer.scrollLeft;

  // 光标在波形中的绝对位置
  const cursorX = getCursorX();

  return Math.abs(mouseAbsoluteX - cursorX) <= threshold;
}

/**
 * 上半部分区域鼠标按下事件（只处理光标拖拽，完全阻止Region操作）
 */
function handleUpperZonePointerDown(e) {
  if (!wavesurfer || !isReady.value) return;
  if (e.pointerType === "mouse" && e.button !== 0) return; // 只处理鼠标左键

  // 【关键】视频未就绪时拦截所有波形操作
  if (!isVideoReady.value) {
    console.warn('[WaveformTimeline] 视频未就绪，波形操作被拦截')
    return
  }

  // 【关键】始终阻止事件传播，防止触发下层的 Region 操作
  e.preventDefault();
  e.stopPropagation();

  // V3.1.1+dev.20260106.02: 移除强制焦点转移，避免破坏事件链
  // 焦点转移可能导致拖拽过程中事件监听器失效

  // 检查是否允许拖拽
  let canDrag = false;

  if (cursorDragMode.value === "anywhere") {
    canDrag = true;
  } else if (cursorDragMode.value === "hover-only") {
    canDrag = isMouseNearCursor(e.clientX, 10);
  }

  const clickTime = getTimeFromClientX(e.clientX);

  if (canDrag) {
    // 开始拖拽光标
    isDraggingCursor.value = true;
    cursorDragStartTime = projectStore.player.currentTime;
    cursorPointerId = typeof e.pointerId === "number" ? e.pointerId : null;
    cursorPointerTarget =
      e.currentTarget instanceof HTMLElement ? e.currentTarget : null;

    // 使用 PlaybackManager 进行拖拽
    playbackManager.startDragging('waveformCursor');
    playbackManager.updateDragging(clickTime);
    emit("seek", clickTime);

    if (
      cursorPointerTarget &&
      typeof cursorPointerTarget.setPointerCapture === "function" &&
      typeof e.pointerId === "number"
    ) {
      try {
        cursorPointerTarget.setPointerCapture(e.pointerId);
      } catch (error) {
        console.debug("[WaveformTimeline] 光标指针捕获失败:", error);
      }
    }

    attachCursorDragGuards();
  } else {
    // 单击跳转（非拖拽）
    playbackManager.seekTo(clickTime);
    emit("seek", clickTime);
  }
}

/**
 * 上半部分区域鼠标移动事件（动态改变鼠标样式）
 */
function handleUpperZonePointerMove(e) {
  if (!wavesurfer || !isReady.value) return;
  if (isDraggingCursor.value || isRegionPointerDragging.value) return; // 拖拽时不改变样式

  // 检查是否在光标附近
  const nearCursor = isMouseNearCursor(e.clientX, 10);

  if (nearCursor && cursorDragMode.value === "hover-only") {
    // 在光标附近，显示可拖拽样式
    currentMouseCursor.value = "ew-resize";
  } else if (cursorDragMode.value === "anywhere") {
    // anywhere模式，显示可点击/拖拽样式
    currentMouseCursor.value = "col-resize";
  } else {
    // 默认：显示十字光标（表示可以点击跳转）
    currentMouseCursor.value = "crosshair";
  }
}

/**
 * 光标拖拽移动
 * V3.1.1+dev.20260106.02: 添加异常处理，防止拖拽过程中出错导致状态污染
 */
function handleCursorDragMove(e) {
  try {
    if (!isDraggingCursor.value) return;
    if (
      typeof e.pointerId === "number" &&
      cursorPointerId !== null &&
      e.pointerId !== cursorPointerId
    ) {
      return;
    }

    const newTime = getTimeFromClientX(e.clientX);
    playbackManager.updateDragging(newTime);
    emit("seek", newTime);
  } catch (error) {
    console.error('[WaveformTimeline] 拖拽移动出错:', error);
    // 异常时强制终止拖拽，避免状态污染
    handleCursorDragEnd();
  }
}

/**
 * 光标拖拽结束
 * V3.1.1+dev.20260106.02: 使用捕获阶段，添加异常处理
 */
function handleCursorDragEnd() {
  try {
    if (!isDraggingCursor.value) return;
    isDraggingCursor.value = false;

    if (
      cursorPointerTarget &&
      typeof cursorPointerId === "number" &&
      typeof cursorPointerTarget.releasePointerCapture === "function"
    ) {
      try {
        if (cursorPointerTarget.hasPointerCapture?.(cursorPointerId)) {
          cursorPointerTarget.releasePointerCapture(cursorPointerId);
        }
      } catch (error) {
        console.debug("[WaveformTimeline] 光标释放指针捕获失败:", error);
      }
    }

    cursorPointerTarget = null;
    cursorPointerId = null;
    detachCursorDragGuards();

    // 使用 PlaybackManager 结束拖拽
    playbackManager.stopDragging();
  } catch (error) {
    console.error('[WaveformTimeline] 拖拽结束出错:', error);
  }
}

/**
 * 波形区域鼠标离开事件（重置鼠标样式）
 */
function handleWaveformPointerLeave() {
  currentMouseCursor.value = "default";
}

// ============ 右键菜单逻辑 ============

/**
 * 右键菜单项配置
 */
const contextMenuItems = computed(() => {
  const items = [];

  if (contextMenuTarget.value) {
    items.push({
      key: 'split',
      label: '从此处切分',
    });
  }

  return items;
});

/**
 * 波形区域右键事件处理
 */
function handleWaveformContextMenu(e) {
  e.preventDefault();
  e.stopPropagation();

  if (!wavesurfer || !isReady.value) return;

  const clickTime = getTimeFromClientX(e.clientX);

  // 查找点击位置对应的字幕
  const targetSubtitle = projectStore.subtitles.find(
    s => clickTime >= s.start && clickTime < s.end
  );

  // 只有在字幕范围内才显示菜单
  if (targetSubtitle && !targetSubtitle.isDraft) {
    contextMenuTarget.value = targetSubtitle.id;
    contextMenuTime.value = clickTime;
    contextMenuRef.value?.show(e.clientX, e.clientY);
  }
}

/**
 * 右键菜单项选择处理
 */
async function handleContextMenuSelect(key) {
  if (key === 'split' && contextMenuTarget.value) {
    const result = projectStore.splitSubtitle(contextMenuTarget.value, {
      splitTime: contextMenuTime.value
    });

    if (!result.success) {
      console.error('[WaveformTimeline] 切分失败:', result.error);
    } else {
      console.log('[WaveformTimeline] 切分成功:', result);
      await syncSplitSubtitles(result);
    }
  }

  // 清空状态
  contextMenuTarget.value = null;
  contextMenuTime.value = 0;
}

/**
 * V3.2.0+dev.20260124.04: 波形切分结果同步到后端
 */
async function syncSplitSubtitles(result) {
  const jobId = jobIdRef.value;
  if (!jobId) return;
  const { leftSubtitle, rightSubtitle, originalSentenceIndex } = result || {};
  if (!leftSubtitle || !rightSubtitle) return;

  try {
    if (originalSentenceIndex !== undefined) {
      await transcriptionApi.updateSubtitle(jobId, originalSentenceIndex, {
        text: leftSubtitle.text,
        start: leftSubtitle.start,
        end: leftSubtitle.end,
      });
      projectStore.updateSubtitle(
        leftSubtitle.id,
        {
          sentenceIndex: originalSentenceIndex,
          isModified: true,
          source: "split",
        },
        { isUserEdit: true }
      );
    } else {
      const leftResp = await transcriptionApi.createSubtitle(jobId, {
        text: leftSubtitle.text,
        start: leftSubtitle.start,
        end: leftSubtitle.end,
      });
      const leftData = leftResp?.data?.data || leftResp?.data;
      if (leftData?.index !== undefined) {
        projectStore.updateSubtitle(
          leftSubtitle.id,
          {
            sentenceIndex: leftData.index,
            isModified: true,
            source: leftData.source || "manual",
          },
          { isUserEdit: true }
        );
      }
    }

    const rightResp = await transcriptionApi.createSubtitle(jobId, {
      text: rightSubtitle.text,
      start: rightSubtitle.start,
      end: rightSubtitle.end,
    });
    const rightData = rightResp?.data?.data || rightResp?.data;
    if (rightData?.index !== undefined) {
      projectStore.updateSubtitle(
        rightSubtitle.id,
        {
          sentenceIndex: rightData.index,
          isModified: true,
          source: rightData.source || "manual",
        },
        { isUserEdit: true }
      );
    }
  } catch (error) {
    console.warn("[WaveformTimeline] 切分同步失败:", error);
  }
}

// ============ 自定义滚动条逻辑 ============

/**
 * 更新滚动条位置和宽度（优化版：避免不必要的响应式更新）
 */
function updateScrollbarThumb() {
  if (!wavesurfer || !isReady.value) return;

  const wrapper = wavesurfer.getWrapper();
  if (!wrapper) return;

  const scrollContainer = wrapper.parentElement;
  if (!scrollContainer) return;

  const scrollWidth = wrapper.scrollWidth;
  const clientWidth = scrollContainer.clientWidth;
  const scrollLeft = scrollContainer.scrollLeft;

  // 计算新的宽度和位置
  const newThumbWidthPercent = Math.max(5, Math.min(100, (clientWidth / scrollWidth) * 100));

  const maxScrollLeft = scrollWidth - clientWidth;
  let newThumbLeftPercent = 0;
  if (maxScrollLeft > 0) {
    newThumbLeftPercent = (scrollLeft / maxScrollLeft) * (100 - newThumbWidthPercent);
  }

  // 只在值真正变化时才更新（减少响应式触发）
  // 使用 0.1% 的阈值避免浮点数精度问题
  if (Math.abs(scrollbarThumbWidth.value - newThumbWidthPercent) > 0.1) {
    scrollbarThumbWidth.value = newThumbWidthPercent;
  }

  if (Math.abs(scrollbarThumbLeft.value - newThumbLeftPercent) > 0.1) {
    scrollbarThumbLeft.value = newThumbLeftPercent;
  }
}

/**
 * 滚动条鼠标按下事件（优化版：缓存DOM查询）
 */
function handleScrollbarMouseDown(e) {
  if (!wavesurfer || !isReady.value) return;

  const wrapper = wavesurfer.getWrapper();
  if (!wrapper) return;

  const scrollContainer = wrapper.parentElement;
  if (!scrollContainer) return;

  e.preventDefault();

  // 缓存容器尺寸，避免拖拽过程中重复查询
  cachedScrollWidth = wrapper.scrollWidth;
  cachedClientWidth = scrollContainer.clientWidth;
  cachedMaxScrollLeft = cachedScrollWidth - cachedClientWidth;

  const rect = scrollbarTrackRef.value.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const trackWidth = rect.width;

  // 点击位置相对于track的百分比
  const clickPercent = clickX / trackWidth;

  // 计算应该滚动到的位置
  const targetScrollLeft = clickPercent * cachedMaxScrollLeft;
  scrollContainer.scrollLeft = targetScrollLeft;

  // 更新滚动条显示
  updateScrollbarThumb();

  // 如果点击的是thumb本身，开始拖拽
  const thumbRect = rect;
  const thumbLeft = (scrollbarThumbLeft.value / 100) * trackWidth;
  const thumbRight = thumbLeft + (scrollbarThumbWidth.value / 100) * trackWidth;

  if (clickX >= thumbLeft && clickX <= thumbRight) {
    // 点击在thumb上，开始拖拽
    isDraggingScrollbar.value = true;
    scrollbarDragStartX = clickX;
    scrollbarDragStartScroll = scrollContainer.scrollLeft;

    // V3.1.1+dev.20260106.02: 使用捕获阶段处理事件
    document.addEventListener("mousemove", handleScrollbarDragMove, true);
    document.addEventListener("mouseup", handleScrollbarDragEnd, true);
  }
}

/**
 * 滚动条拖拽移动（优化版：使用RAF节流）
 * V3.1.1+dev.20260106.03: 保存 shiftKey 状态用于精细模式
 */
function handleScrollbarDragMove(e) {
  if (!isDraggingScrollbar.value || !wavesurfer) return;

  // 只保存最新的事件信息，不立即计算
  // V3.1.1+dev.20260106.03: 添加 shiftKey 用于精细模式
  pendingScrollEvent = {
    clientX: e.clientX,
    timestamp: Date.now(),
    shiftKey: e.shiftKey  // 精细模式标志
  };

  // 使用RAF在下一帧渲染前统一处理
  if (!scrollbarRafId) {
    scrollbarRafId = requestAnimationFrame(processScrollbarDrag);
  }
}

/**
 * 处理滚动条拖拽（在RAF中执行）
 * V3.1.1+dev.20260106.03: 添加 Shift 精细模式和动态阻尼
 */
function processScrollbarDrag() {
  if (!pendingScrollEvent || !isDraggingScrollbar.value || !wavesurfer) {
    scrollbarRafId = null;
    return;
  }

  const wrapper = wavesurfer.getWrapper();
  if (!wrapper) {
    scrollbarRafId = null;
    return;
  }

  const scrollContainer = wrapper.parentElement;
  if (!scrollContainer) {
    scrollbarRafId = null;
    return;
  }

  // 计算新的滚动位置（使用缓存的尺寸数据）
  const rect = scrollbarTrackRef.value.getBoundingClientRect();
  const trackWidth = rect.width;
  const deltaX = pendingScrollEvent.clientX - rect.left - scrollbarDragStartX;
  const deltaPercent = deltaX / trackWidth;

  // V3.1.1+dev.20260106.03: 精细模式和动态阻尼
  // Shift 精细模式：按住 Shift 时，拖拽灵敏度降低到 1/4
  // 动态阻尼：缩放越大，基础拖拽也会变慢（使用较温和的阻尼曲线）
  let effectiveDeltaPercent = deltaPercent;

  // 动态阻尼：高缩放时适度降低灵敏度（使用 log 曲线，避免太"沉"）
  // 100% 缩放 → 阻尼系数 1.0
  // 400% 缩放 → 阻尼系数约 1.4
  // 800% 缩放 → 阻尼系数约 1.6
  const dampingFactor = 1 + Math.log10(Math.max(1, zoomLevel.value / 100)) * 0.5;
  effectiveDeltaPercent = deltaPercent / dampingFactor;

  // Shift 精细模式：进一步降低灵敏度
  if (pendingScrollEvent.shiftKey) {
    effectiveDeltaPercent *= 0.25;  // 精细模式：1/4 灵敏度
  }

  const newScrollLeft = scrollbarDragStartScroll + effectiveDeltaPercent * cachedMaxScrollLeft;

  // 更新滚动位置
  scrollContainer.scrollLeft = Math.max(0, Math.min(newScrollLeft, cachedMaxScrollLeft));

  // 批量更新响应式变量
  updateScrollbarThumb();

  // 清空状态
  pendingScrollEvent = null;
  scrollbarRafId = null;
}

/**
 * 滚动条拖拽结束
 * V3.1.1+dev.20260106.02: 使用捕获阶段
 */
function handleScrollbarDragEnd() {
  isDraggingScrollbar.value = false;

  // 清理RAF
  if (scrollbarRafId) {
    cancelAnimationFrame(scrollbarRafId);
    scrollbarRafId = null;
  }

  // V3.1.1+dev.20260106.02: 移除事件监听器时参数必须一致
  document.removeEventListener("mousemove", handleScrollbarDragMove, true);
  document.removeEventListener("mouseup", handleScrollbarDragEnd, true);

  // 确保最终状态同步
  nextTick(() => {
    updateScrollbarThumb();
  });
}

// 格式化时间
function formatTime(seconds) {
  if (!seconds || isNaN(seconds)) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

// 监听字幕变化
// V3.1.0: 修复 watch 监听失效问题 - 同时监听数组长度确保 splice 操作也能触发
watch(
  () => projectStore.subtitles,
  (newVal, oldVal) => {
    const lengthChanged = !oldVal || newVal.length !== oldVal.length;
    console.log(`[WaveformTimeline] subtitles watch 触发: length=${newVal.length}, lengthChanged=${lengthChanged}, isReady=${isReady.value}, isUpdating=${isUpdatingRegions.value}`);

    if (isReady.value && !isUpdatingRegions.value) {
      clearTimeout(regionUpdateTimer);
      regionUpdateTimer = setTimeout(() => {
        renderSubtitleRegions();
      }, 100);
    } else {
      // V3.1.0: 如果当前条件不满足，延迟重试
      if (!isReady.value) {
        console.log('[WaveformTimeline] 波形未就绪，延迟 500ms 后重试渲染 regions');
        setTimeout(() => {
          if (isReady.value && projectStore.subtitles.length > 0) {
            renderSubtitleRegions();
          }
        }, 500);
      }
    }
  },
  { deep: true }
);

// 监听播放状态
watch(
  () => projectStore.player.isPlaying,
  (playing) => {
    if (!wavesurfer || !isReady.value) return;

    if (playing) {
      wavesurfer.play();
      // 【关键修改】启动智能跟随
      startSmartFollow();
    } else {
      wavesurfer.pause();
      // 【关键修改】停止智能跟随
      stopSmartFollow();
    }
  }
);

// 监听时间变化（优化：使用 PlaybackManager 的状态判断）
watch(
  () => projectStore.player.currentTime,
  (newTime) => {
    if (!wavesurfer || !isReady.value) return;

    // 【重要】播放时不同步（让 WaveSurfer 自己播放），除非：
    // 1. 正在拖拽（用户主动操作）
    // 2. 正在 seeking
    const isPlaying = projectStore.player.isPlaying;
    const isSeeking = playbackManager.isLocked();
    
    if (isPlaying && !isSeeking) {
      return;
    }

    // 节流：避免过于频繁的同步（最少间隔50ms）
    const now = Date.now();
    if (now - lastSyncTime < 50) {
      return;
    }
    lastSyncTime = now;

    // 检查时间差异
    const currentWsTime = wavesurfer.getCurrentTime();
    const timeDiff = Math.abs(currentWsTime - newTime);

    // 如果时间差异超过0.1秒，进行同步
    if (timeDiff > 0.1) {
      const duration = wavesurfer.getDuration();
      if (duration > 0) {
        wavesurfer.seekTo(newTime / duration);
      }
    }
  }
);

// 监听选中字幕变化
watch(
  () => projectStore.view.selectedSubtitleId,
  () => {
    if (isReady.value) {
      renderSubtitleRegions();
    }
  }
);

// ============ 滚轮事件处理 ============

// 缩放节流状态
let zoomRafId = null;
let pendingZoomDelta = 0;

/**
 * 平滑缩放 - 使用 RAF 批量处理缩放请求
 * 【优化】使用智能锚点策略
 */
function smoothZoom() {
  if (pendingZoomDelta === 0) {
    zoomRafId = null;
    return;
  }

  const newZoom = zoomLevel.value + pendingZoomDelta;
  pendingZoomDelta = 0; // 清空待处理的增量

  // 使用智能锚点缩放
  handleZoomWithSmartAnchor(newZoom);

  zoomRafId = null;
}

/**
 * 波形区域滚轮事件（只处理 Ctrl+滚轮 缩放，普通滚轮不做任何操作）
 */
function handleWheel(e) {
  // 只有 Ctrl+滚轮 才触发缩放
  if (!e.ctrlKey) {
    // 普通滚轮：不做任何操作（不滚动波形）
    // 如果要阻止页面滚动，可以取消注释下面这行
    // e.preventDefault()
    return;
  }

  e.preventDefault();

  // 累积缩放增量
  const delta = e.deltaY < 0 ? ZOOM_WHEEL_STEP : -ZOOM_WHEEL_STEP;
  pendingZoomDelta += delta;

  // 使用 RAF 批量处理，减少重绘次数
  if (!zoomRafId) {
    zoomRafId = requestAnimationFrame(smoothZoom);
  }
}

/**
 * 滚动条区域滚轮事件（允许水平滚动波形）
 * V3.1.1+dev.20260106.03: 添加动态阻尼系数，缩放越大滚动越慢
 */
function handleScrollbarWheel(e) {
  if (!wavesurfer || !isReady.value) return;

  const wrapper = wavesurfer.getWrapper();
  if (!wrapper) return;

  const scrollContainer = wrapper.parentElement;
  if (!scrollContainer) return;

  e.preventDefault();

  // 动态阻尼系数：缩放越大，滚动速度越慢，提供精细控制
  // 基础速度
  const BASE_SPEED = 2;
  // 阻尼因子：缩放越大，因子越大。例如 100%时为1，800%时为8
  const dampingFactor = Math.max(1, zoomLevel.value / 100);
  // 核心公式：速度随着缩放增加而减小（使用平方根让衰减曲线更平滑）
  // 效果：放大时，滚轮变得更"沉"，移动距离变短，提供精确控制
  const dynamicSpeed = BASE_SPEED / Math.sqrt(dampingFactor);

  // 水平滚动波形
  const scrollAmount = e.deltaY * dynamicSpeed;
  scrollContainer.scrollLeft += scrollAmount;
  updateScrollbarThumb();
}

onMounted(async () => {
  await nextTick();
  setupRegionPointerGuards();
  await initWavesurfer();
  containerRef.value?.addEventListener("wheel", handleWheel, {
    passive: false,
  });
});

onUnmounted(() => {
  containerRef.value?.removeEventListener("wheel", handleWheel);

  // 清理缩放RAF
  if (zoomRafId) cancelAnimationFrame(zoomRafId);

  // 清理滚动条拖拽RAF
  if (scrollbarRafId) cancelAnimationFrame(scrollbarRafId);

  // 清理滚动条更新定时器
  if (scrollbarUpdateTimer) clearTimeout(scrollbarUpdateTimer);

  // 清理波形数据检查定时器
  stopPeaksPolling();

  clearTimeout(regionUpdateTimer);
  stopSmartFollow(); // 清理智能跟随RAF循环

  // V3.1.1+dev.20260106.02: 清理所有 document 事件监听器，防止拖拽期间组件卸载导致的事件泄漏
  if (isDraggingCursor.value) {
    handleCursorDragEnd();
  } else {
    detachCursorDragGuards();
  }

  if (isRegionPointerDragging.value) {
    finalizeRegionPointerDrag();
  } else {
    detachRegionDragGuards();
  }
  teardownRegionPointerGuards();
  document.removeEventListener("mousemove", handleScrollbarDragMove, true);
  document.removeEventListener("mouseup", handleScrollbarDragEnd, true);

  // 清理 DOM 缓存
  cachedWrapper = null;
  cachedScrollContainer = null;

  // 【关键】注销 WaveSurfer
  playbackManager.unregisterWaveSurfer();

  if (wavesurfer) {
    wavesurfer.destroy();
    wavesurfer = null;
  }
});
</script>

<style lang="scss" scoped>
.waveform-timeline {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--bg-secondary);
  // border-radius: var(--radius-lg);
  overflow: hidden;
}

// 头部控制栏（压缩高度）
.timeline-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 16px; // 【优化】从10px压缩到6px，减少整体高度
  background: var(--bg-tertiary);
  border-bottom: 1px solid var(--border-default);
}

.zoom-controls {
  display: flex;
  align-items: center;
  gap: 8px;

  .zoom-btn,
  .fit-btn {
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--radius-sm);
    color: var(--text-secondary);
    transition: all var(--transition-fast);

    svg {
      width: 16px;
      height: 16px;
    }

    &:hover {
      background: var(--bg-elevated);
      color: var(--text-primary);
    }
  }

  .zoom-slider {
    width: 100px;

    input[type="range"] {
      appearance: none;
      -webkit-appearance: none;
      width: 100%;
      height: 4px;
      background: var(--bg-elevated);
      border-radius: 2px;
      cursor: pointer;

      &::-webkit-slider-thumb {
        appearance: none;
        -webkit-appearance: none;
        width: 12px;
        height: 12px;
        background: var(--primary);
        border-radius: 50%;
        cursor: pointer;
      }
    }
  }

  .zoom-label {
    min-width: 50px;
    font-size: 12px;
    font-family: var(--font-mono);
    color: var(--text-muted);
    text-align: center;
  }
}

.time-indicator {
  display: flex;
  align-items: center;
  gap: 6px;
  font-family: var(--font-mono);
  font-size: 13px;

  .current-time {
    color: var(--primary);
    font-weight: 600;
  }

  .separator {
    color: var(--text-muted);
  }

  .total-time {
    color: var(--text-secondary);
  }
}

// 自定义滚动条（最底部，分层次显示）
.custom-scrollbar {
  height: 14px; // 减小高度
  padding: 3px 16px;
  background: var(--bg-tertiary);
  flex-shrink: 0;

  // 容器级 hover 状态：使滑块更清晰
  &:hover .scrollbar-thumb {
    background: rgba(139, 148, 158, 0.5) !important;
  }

  .scrollbar-track {
    position: relative;
    width: 100%;
    height: 8px; // 轨道高度
    background: transparent; // 默认状态：完全透明
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.2s ease;

    // 轨道 hover 状态：显现淡淡的灰色背景
    &:hover {
      background: rgba(255, 255, 255, 0.08);

      .scrollbar-thumb {
        background: rgba(139, 148, 158, 0.8) !important; // 高亮激活态
      }
    }
  }

  .scrollbar-thumb {
    position: absolute;
    top: 0;
    height: 100%;
    background: rgba(139, 148, 158, 0.2); // 默认状态：极其微弱的灰色
    border-radius: 4px;
    transition: background 0.15s ease, left 0.05s linear, width 0.05s linear;
    cursor: grab;
    min-width: 20px; // 设置最小宽度，确保thumb始终可见

    &:hover {
      background: rgba(139, 148, 158, 0.75) !important;
    }

    &:active {
      cursor: grabbing;
      background: rgba(139, 148, 158, 0.85) !important;
    }
  }
}

// 波形容器
.waveform-wrapper {
  flex: 1;
  position: relative;
  min-height: 80px; // 减小最小高度
  overflow-x: auto; // 允许水平滚动
  overflow-y: hidden;

  // 隐藏系统滚动条（使用自定义滚动条）
  &::-webkit-scrollbar {
    display: none;
  }
  -ms-overflow-style: none;
  scrollbar-width: none;

  // 上半部分交互遮罩层
  .waveform-upper-zone {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 50%; // 覆盖上半部分
    z-index: 10; // 在波形之上
    // V3.1.1+dev.20260106.02: 防止拖拽时选中文本或触发其他触摸行为
    user-select: none;
    -webkit-user-select: none;
    touch-action: none;
    // background: rgba(255, 0, 0, 0.1);  // 调试用，可删除

    &.is-region-dragging {
      // Region 拖拽时关闭遮罩交互，让拖拽事件不被阻断
      pointer-events: none;
      cursor: inherit;
    }
  }

  #waveform {
    height: 100%;

    // 【优化】为光标添加阴影，提高在不同背景上的可见性
    :deep(.wavesurfer-cursor) {
      // 使用多层阴影：内部黑色描边 + 外部白色光晕
      filter: drop-shadow(0 0 1px rgba(0, 0, 0, 0.8))
        drop-shadow(0 0 2px rgba(255, 255, 255, 0.5))
        drop-shadow(0 0 4px rgba(248, 81, 73, 0.6)); // 红色光晕与光标颜色呼应
    }

    // 隐藏 WaveSurfer 内部滚动容器的滚动条
    :deep(> div) {
      &::-webkit-scrollbar {
        display: none;
      }
      -ms-overflow-style: none;
      scrollbar-width: none;
    }

    :deep(.wavesurfer-region) {
      border-radius: 2px;
      transition: background-color 0.2s;

      &:hover {
        background-color: rgba(88, 166, 255, 0.4) !important;
      }
    }

    :deep(.wavesurfer-handle) {
      background: var(--primary) !important;
      width: 4px !important;
      border-radius: 2px;
      // V3.1.1+dev.20260106.02: 增大 handle 的点击区域，提高拖拽稳定性
      // 使用 padding 或 border 扩大可点击范围，但保持视觉宽度不变
      box-sizing: content-box;
      // 确保 handle 在拖拽时保持鼠标捕获
      touch-action: none;
    }

    // V3.1.1+dev.20260106.02: 确保 Region 容器在拖拽时不会丢失鼠标捕获
    :deep(.wavesurfer-region) {
      // 禁用文本选择，防止拖拽时选中文本导致事件中断
      user-select: none;
      -webkit-user-select: none;
      touch-action: none;
    }
  }
}

// 加载状态
.waveform-loading {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  background: var(--bg-secondary);
  color: var(--text-muted);

  .loading-spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--border-default);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

// 错误状态
.waveform-error {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  background: var(--bg-secondary);
  color: var(--text-muted);

  svg {
    width: 40px;
    height: 40px;
    color: var(--danger);
  }

  button {
    padding: 6px 16px;
    background: var(--primary);
    color: white;
    border-radius: var(--radius-md);
    font-size: 13px;
    &:hover {
      background: var(--primary-hover);
    }
  }
}

// 时间轴刻度（移到波形上方，更窄）
#timeline {
  height: 18px; // 从24px减小到18px，更紧凑
  background: var(--bg-tertiary);
  border-bottom: 1px solid var(--border-default);
  flex-shrink: 0;
}
</style>
