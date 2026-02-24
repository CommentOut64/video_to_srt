/**
 * Proxy 视频状态管理 Composable（重构版）
 *
 * 职责：
 * 1. 管理 Proxy 视频完整生命周期（状态机模式）
 * 2. 自动订阅/取消订阅 SSE 事件
 * 3. 处理刷新后状态恢复
 * 4. 提供统一的 isReady 状态
 * 5. 控件操作拦截支持
 */
import { ref, computed, watch, onMounted, onUnmounted, isRef } from 'vue'
import { sseChannelManager } from '@/services/sseChannelManager'
import { mediaApi } from '@/services/api'
import { IS_LITE } from '@/config/flavor'

/**
 * Proxy 视频状态枚举
 */
export const ProxyState = {
  IDLE: 'idle',                       // 初始状态
  ANALYZING: 'analyzing',             // 分析中
  TRANSCODING_360: 'transcoding_360', // 360p 转码中
  READY_360P: 'ready_360p',           // 360p 就绪，可播放
  TRANSCODING_720: 'transcoding_720', // 720p 转码中
  READY_720P: 'ready_720p',           // 720p 就绪
  REMUXING: 'remuxing',               // 容器重封装中
  DIRECT_PLAY: 'direct_play',         // 原始文件可直接播放
  ERROR: 'error'                      // 错误状态
}

/**
 * 转码决策类型（与后端保持一致）
 */
export const TranscodeDecision = {
  DIRECT_PLAY: 'direct_play',
  REMUX_ONLY: 'remux_only',
  TRANSCODE_AUDIO: 'transcode_audio',
  TRANSCODE_VIDEO: 'transcode_video',
  TRANSCODE_FULL: 'transcode_full'
}

/**
 * Proxy 视频管理 Composable
 * @param {Ref<string>|string} jobIdInput - 任务ID（可以是 ref 或普通值）
 */
export function useProxyVideo(jobIdInput) {
  // 统一转换为 ref
  const jobId = isRef(jobIdInput) ? jobIdInput : ref(jobIdInput)

  // ========== 响应式状态 ==========
  const state = ref(ProxyState.IDLE)
  const progress = ref(0)
  const error = ref(null)
  const decision = ref(null)  // 转码决策
  const autoTrigger720p = ref(true)  // V3.1.2+dev.20260113.02: 是否启用自动触发720p
  const version = ref(0) // V3.1.2+dev.20260114.08: 状态版本号，避免旧快照覆盖新状态

  const urls = ref({
    preview360p: null,
    proxy720p: null,
    source: null
  })

  // 待替换的720p URL（用于延迟替换）
  const pending720pUrl = ref(null)

  // SSE 订阅状态
  const isSubscribed = ref(false)
  let unsubscribeSSE = null

  // ========== 计算属性 ==========

  /**
   * 视频是否就绪（可以播放）
   */
  const isReady = computed(() => {
    if (IS_LITE && !jobId.value) {
      return true
    }
    return [
      ProxyState.READY_360P,
      ProxyState.READY_720P,
      ProxyState.DIRECT_PLAY
    ].includes(state.value)
  })

  /**
   * 是否正在转码/处理中
   */
  const isTranscoding = computed(() => {
    return [
      ProxyState.ANALYZING,
      ProxyState.TRANSCODING_360,
      ProxyState.TRANSCODING_720,
      ProxyState.REMUXING
    ].includes(state.value)
  })

  /**
   * 是否处于错误状态
   */
  const hasError = computed(() => {
    return state.value === ProxyState.ERROR
  })

  /**
   * 当前最佳可用 URL
   * 优先级：720p > 360p > source
   *
   * 注意：如果正在转码中，返回 null，避免加载不兼容的源视频
   */
  const currentUrl = computed(() => {
    if (urls.value.proxy720p) return urls.value.proxy720p
    if (urls.value.preview360p) return urls.value.preview360p
    // 如果正在转码，不返回 source URL（避免加载不兼容的视频）
    if (isTranscoding.value) return null
    // 否则返回 source URL（可直接播放的视频）
    return urls.value.source
  })

  /**
   * 当前分辨率标识
   */
  const currentResolution = computed(() => {
    if (urls.value.proxy720p) return '720p'
    if (urls.value.preview360p) return '360p'
    if (state.value === ProxyState.DIRECT_PLAY) return 'source'
    return null
  })

  /**
   * 是否正在升级画质
   */
  const isUpgrading = computed(() => {
    return (
      state.value === ProxyState.TRANSCODING_720 && !!urls.value.preview360p
    );
  });

  /**
   * 转码状态文本（用于 UI 显示）
   */
  const statusText = computed(() => {
    const texts = {
      [ProxyState.IDLE]: "",
      [ProxyState.ANALYZING]: "分析视频中...",
      [ProxyState.REMUXING]: "容器重封装中...",
      [ProxyState.TRANSCODING_360]: "生成 360p 预览...",
      [ProxyState.READY_360P]: "预览就绪",
      [ProxyState.TRANSCODING_720]: "生成 720p 高清...",
      [ProxyState.READY_720P]: "高清就绪",
      [ProxyState.DIRECT_PLAY]: "可直接播放",
      [ProxyState.ERROR]: "处理失败",
    };
    return texts[state.value] || "处理中...";
  });

  // ========== SSE 事件处理器 ==========

  const sseHandlers = {
    // 分析完成事件
    onAnalyzeComplete: (data) => {
      console.log("[useProxyVideo] 分析完成:", data);
      decision.value = data.decision;

      if (data.decision === TranscodeDecision.DIRECT_PLAY) {
        state.value = ProxyState.DIRECT_PLAY;
        urls.value.source = data.source_url;
        progress.value = 100;
      } else if (data.decision === TranscodeDecision.REMUX_ONLY) {
        state.value = ProxyState.REMUXING;
        progress.value = 0;
      } else {
        state.value = ProxyState.TRANSCODING_360;
        progress.value = 0;
      }
    },

    // 重封装进度
    onRemuxProgress: (data) => {
      console.log("[useProxyVideo] 重封装进度:", data.progress);
      state.value = ProxyState.REMUXING;
      progress.value = data.progress || 0;
    },

    // 重封装完成
    onRemuxComplete: (data) => {
      console.log("[useProxyVideo] 重封装完成:", data);
      state.value = ProxyState.READY_720P;
      urls.value.proxy720p =
        data.video_url || `/api/media/${jobId.value}/video`;
      progress.value = 100;
    },

    // 360p 预览进度
    onPreview360pProgress: (data) => {
      console.log("[useProxyVideo] 360p 进度:", data.progress);
      state.value = ProxyState.TRANSCODING_360;
      progress.value = data.progress || 0;
    },

    // 360p 预览完成
    onPreview360pComplete: (data) => {
      console.log("[useProxyVideo] 360p 完成:", data);
      state.value = ProxyState.READY_360P;
      urls.value.preview360p =
        data.video_url || `/api/media/${jobId.value}/video/preview`;
      progress.value = 0; // 重置进度，准备 720p

      // 自动开始 720p（状态会由后续 SSE 事件更新）
      // 此处不立即切换到 TRANSCODING_720，等待后端推送
    },

    // 720p Proxy 进度（后台静默转码，不更新状态）
    onProxyProgress: (data) => {
      console.log("[useProxyVideo] 720p 后台转码进度:", data.progress);
      // 如果当前已有可播放视频（360p 或 source），保持当前状态，不显示转码提示
      // 只在内部记录进度，用于调试
      if (
        state.value === ProxyState.IDLE ||
        state.value === ProxyState.ANALYZING
      ) {
        // 如果当前没有可播放视频，才显示转码状态
        state.value = ProxyState.TRANSCODING_720;
        progress.value = data.progress || 0;
      }
      // 否则静默转码，不更新 UI 状态
    },

    // 720p Proxy 完成（立即无缝替换）
    onProxyComplete: (data) => {
      console.log("[useProxyVideo] 720p 完成，准备替换:", data);
      const new720pUrl = data.video_url || `/api/media/${jobId.value}/video`;

      console.log("[useProxyVideo] 立即更新720p URL，触发无缝切换");
      console.log("[useProxyVideo] 更新前 - urls.value.proxy720p:", urls.value.proxy720p);

      urls.value.proxy720p = new720pUrl;
      state.value = ProxyState.READY_720P;
      progress.value = 0;

      console.log("[useProxyVideo] 更新后 - urls.value.proxy720p:", urls.value.proxy720p);
      console.log("[useProxyVideo] 720p URL已更新，完整状态:", {
        proxy720p: urls.value.proxy720p,
        preview360p: urls.value.preview360p,
        state: state.value,
        currentUrl: currentUrl.value,
        currentResolution: currentResolution.value
      });
    },

    // Proxy 错误
    onProxyError: (data) => {
      console.error("[useProxyVideo] Proxy 错误:", data);
      const reason = data?.reason || data?.message || "";

      // V3.1.2+dev.20260114.16: 被新任务打断时静默回落到360p，不显示错误覆盖
      const pausedForNewJob = reason.includes("paused_for_new_job");

      if (pausedForNewJob) {
        console.log("[useProxyVideo] 720p 被中断以让路新任务，保持 360p 播放并等待重试");
        urls.value.proxy720p = null;
        pending720pUrl.value = null;
        progress.value = 0;
        // 若有360p则保持 ready_360p，否则回到 idle 等待后续事件
        state.value = urls.value.preview360p ? ProxyState.READY_360P : ProxyState.IDLE;
        error.value = null;
      } else {
        // 其它错误：若有360p可用则静默回落，保留播放；否则进入错误状态
        const has360 = !!urls.value.preview360p;
        urls.value.proxy720p = null;
        pending720pUrl.value = null;
        progress.value = 0;

        if (has360) {
          state.value = ProxyState.READY_360P;
          error.value = null; // 延迟到用户主动升级时再报错
          console.warn("[useProxyVideo] 720p 失败，已回落至 360p 播放:", data.message);
        } else {
          state.value = ProxyState.ERROR;
          error.value = data.message || "转码失败";
        }
      }
    },
  };

  // ========== 方法 ==========

  /**
   * 初始化：从后端恢复状态
   */
  async function initialize() {
    if (!jobId.value) return;

    try {
      // 从后端获取当前状态
      const response = await mediaApi.getProxyStatus(jobId.value);
      const status = response.data || response;

      console.log("[useProxyVideo] 初始化状态:", status);

      // 恢复 URLs（先恢复URL，用于判断是否有可播放视频）
      if (status.urls) {
        urls.value = {
          preview360p: status.urls["360p"] || null,
          proxy720p: status.urls["720p"] || null,
          source: status.urls.source || null,
        };
      }
      // V3.1.2+dev.20260114.21: 若后端提供最佳可播放URL，作为候选直接使用
      if (!urls.value.proxy720p && status.best_playable_url) {
        pending720pUrl.value = status.best_playable_url;
        urls.value.proxy720p = status.best_playable_url;
        console.log("[useProxyVideo] 使用后端最佳可播放URL作为初始候选:", status.best_playable_resolution);
      }

      // V3.1.2+dev.20260113.03: 静默转码逻辑
      // 如果后端返回 transcoding_720 但已有360p可用，保持 ready_360p 状态
      let effectiveState = status.state || ProxyState.IDLE;
      if (effectiveState === ProxyState.TRANSCODING_720 && urls.value.preview360p) {
        console.log("[useProxyVideo] 720p正在后台转码，保持360p就绪状态（静默转码）");
        effectiveState = ProxyState.READY_360P;
      }

      // 恢复状态
      // V3.1.2+dev.20260114.20: 清理 paused_for_new_job 错误，静默保持 360p
      const incomingError = status.error || null;
      const isPausedError = typeof incomingError === 'string' && incomingError.includes('paused_for_new_job');

      state.value = effectiveState;
      progress.value = status.progress || 0;
      decision.value = status.decision || null;
      error.value = isPausedError ? null : incomingError;
      if (isPausedError && urls.value.preview360p) {
        state.value = ProxyState.READY_360P;
      }
      autoTrigger720p.value = status.auto_trigger_720p !== undefined ? status.auto_trigger_720p : true;
      version.value = status.version !== undefined ? status.version : version.value;

      // 注意：不在这里订阅 SSE，由外部统一订阅并转发事件到 handlers
    } catch (e) {
      console.error("[useProxyVideo] 恢复状态失败:", e);
      // 失败时设为初始状态，等待 SSE 事件
      state.value = ProxyState.ANALYZING;
      // 注意：不在这里订阅 SSE，由外部统一订阅并转发事件到 handlers
    }
  }

  /**
   * 订阅 SSE 事件
   */
  function subscribeSSE() {
    if (!jobId.value || isSubscribed.value) return;

    console.log("[useProxyVideo] 订阅 SSE:", jobId.value);

    unsubscribeSSE = sseChannelManager.subscribeJob(jobId.value, {
      ...sseHandlers,
      // 连接成功后刷新状态
      onConnected: () => {
        console.log("[useProxyVideo] SSE 连接成功");
      },
    });

    isSubscribed.value = true;
  }

  /**
   * 取消订阅 SSE
   */
  function unsubscribe() {
    if (unsubscribeSSE) {
      console.log("[useProxyVideo] 取消 SSE 订阅");
      unsubscribeSSE();
      unsubscribeSSE = null;
      isSubscribed.value = false;
    }
  }

  /**
   * 重试转码
   */
  async function retry() {
    if (state.value !== ProxyState.ERROR) return;

    console.log("[useProxyVideo] 重试转码");
    error.value = null;
    state.value = ProxyState.ANALYZING;
    progress.value = 0;

    try {
      // 触发后端重新分析
      await mediaApi.generatePreview(jobId.value);
      subscribeSSE();
    } catch (e) {
      console.error("[useProxyVideo] 重试失败:", e);
      state.value = ProxyState.ERROR;
      error.value = "重试失败";
    }
  }

  /**
   * 手动刷新状态
   */
  async function refresh() {
    await initialize();
  }

  /**
   * 执行720p替换（由 VideoStage 在合适的时机调用）
   */
  function apply720pUpgrade() {
    if (!pending720pUrl.value) {
      console.log("[useProxyVideo] 没有待替换的720p URL");
      return false;
    }

    console.log("[useProxyVideo] 执行720p替换:", {
      pending: pending720pUrl.value,
      currentState: state.value,
      current720p: urls.value.proxy720p,
      current360p: urls.value.preview360p,
    });

    urls.value.proxy720p = pending720pUrl.value;
    state.value = ProxyState.READY_720P;
    pending720pUrl.value = null; // 清除待替换状态
    progress.value = 0;

    console.log("[useProxyVideo] 720p替换完成，新的URLs:", {
      proxy720p: urls.value.proxy720p,
      preview360p: urls.value.preview360p,
    });

    return true;
  }

  /**
   * 降级到360p（视频加载失败时调用）
   */
  function fallbackTo360p(reason = "") {
    console.warn("[useProxyVideo] 视频加载失败，降级至360p:", reason);
    pending720pUrl.value = null;
    urls.value.proxy720p = null;
    if (urls.value.preview360p) {
      state.value = ProxyState.READY_360P;
      error.value = null;
    } else {
      state.value = ProxyState.ERROR;
      error.value = reason || "视频加载失败";
    }
  }

  // ========== 生命周期 ==========

  onMounted(() => {
    if (jobId.value) {
      initialize();
    }
  });

  onUnmounted(() => {
    unsubscribe();
  });

  // 监听 jobId 变化
  watch(jobId, (newId, oldId) => {
    if (oldId) {
      unsubscribe();
      // 重置状态
      state.value = ProxyState.IDLE;
      progress.value = 0;
      error.value = null;
      urls.value = { preview360p: null, proxy720p: null, source: null };
    }
    if (newId) {
      initialize();
    }
  });

  // 调试：监听 currentUrl 变化
  watch(currentUrl, (newUrl, oldUrl) => {
    console.log("[useProxyVideo] currentUrl 变化:", {
      oldUrl,
      newUrl,
      state: state.value,
      urls: { ...urls.value },
    });
  });

  // ========== 返回值 ==========
  return {
    // 状态
    state,
    progress,
    error,
    decision,
    urls,
    autoTrigger720p,  // V3.1.2+dev.20260113.02: 自动触发720p配置

    // 计算属性
    isReady,
    isTranscoding,
    hasError,
    currentUrl,
    currentResolution,
    isUpgrading,
    statusText,

    // 方法
    retry,
    refresh,
    subscribeSSE,
    unsubscribe,
    apply720pUpgrade, // 执行720p替换
    fallbackTo360p,   // 视频加载失败降级
    // 从后端快照直接同步当前状态（SSE initial_state/HTTP 兜底）
    applySnapshot(proxyState) {
      if (!proxyState) return;
      // V3.1.2+dev.20260114.08: 防止旧快照覆盖新状态（版本递增）
      if (proxyState.version !== undefined && version.value !== 0 && proxyState.version < version.value) {
        console.log("[useProxyVideo] 忽略过期的状态快照", { incoming: proxyState.version, current: version.value });
        return;
      }
      if (proxyState.version !== undefined) {
        version.value = proxyState.version;
      }
      let nextState = proxyState.state || state.value;
      let nextError = proxyState.error || null;
      const pausedErr = typeof nextError === 'string' && nextError.includes('paused_for_new_job');
      if (pausedErr && urls.value.preview360p) {
        nextError = null;
        nextState = ProxyState.READY_360P;
      }

      state.value = nextState;
      progress.value = proxyState.progress ?? progress.value;
      decision.value = proxyState.decision || decision.value;
      error.value = nextError;
      if (proxyState.urls) {
        urls.value = {
          preview360p: proxyState.urls["360p"] || null,
          proxy720p: proxyState.urls["720p"] || null,
          source: proxyState.urls.source || null,
        };
      }
    },

    // 待替换状态
    pending720pUrl,

    // SSE 处理器（供外部直接调用）
    handlers: sseHandlers,

    // 常量
    ProxyState,
    TranscodeDecision,
  };
}
