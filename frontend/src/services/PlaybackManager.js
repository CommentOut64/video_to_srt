/**
 * PlaybackManager - 全局播放控制管理器（单例）
 *
 * 核心设计原则：
 * 1. 全局单例：整个应用只有一个实例，所有组件共享状态
 * 2. 单一数据源：所有播放态通过 playbackStore 管理
 * 3. 明确的数据流：User Action → Manager → Store → Video/WaveSurfer
 * 4. 集中式锁管理：避免多实例导致的锁状态不同步
 *
 * 使用方式：
 * - VideoStage: playbackManager.registerVideo(videoRef)
 * - PlaybackControls/WaveformTimeline: playbackManager.seekTo(time)
 */

import { useProjectStore } from "@/stores/projectStore";
import { usePlaybackStore } from "@/stores/playbackStore";

// ============ 配置常量 ============
const CONFIG = {
  // 时间同步阈值（秒）：小于此差值不触发同步
  SYNC_THRESHOLD: 0.05,

  // Seek 锁超时（毫秒）：防止锁永久卡住
  SEEK_LOCK_TIMEOUT: 3000,

  // Seek 完成后的保护期（毫秒）：期间忽略 Video 的 timeupdate
  SEEK_PROTECTION_PERIOD: 150,

  // 防抖延迟（毫秒）
  DEBOUNCE_DELAY: 30,
};

/**
 * 创建播放管理器（全局单例）
 */
function createPlaybackManager() {
  // ============ 私有状态 ============

  // Video 元素引用（由 VideoStage 注册）
  let videoElement = null;

  // WaveSurfer 实例引用（由 WaveformTimeline 注册）
  let wavesurferInstance = null;
  let waveSurferHandlers = null;

  // 锁状态
  let isSeekingInternal = false;
  let isDraggingInternal = false;
  let dragSource = "";

  // 时间保护
  let lastSeekTime = 0; // 上次 seek 的时间戳

  // 定时器
  let seekLockTimer = null;
  let timeUpdateTimer = null;
  let lastWaveSurferSyncAt = 0;

  // Store 引用（延迟获取，避免循环依赖）
  let _store = null;
  let _playbackStore = null;
  function getStore() {
    if (!_store) {
      _store = useProjectStore();
    }
    return _store;
  }
  function getPlaybackStore() {
    if (!_playbackStore) {
      _playbackStore = usePlaybackStore();
    }
    return _playbackStore;
  }

  /**
   * 兼容读取 Pinia setup store 字段（值或 Ref）
   * 说明：setup store 在大多数场景会自动解包，历史代码里的 `.value` 会读到 undefined。
   */
  function readStoreValue(maybeRefValue) {
    if (
      maybeRefValue &&
      typeof maybeRefValue === "object" &&
      "value" in maybeRefValue
    ) {
      return maybeRefValue.value;
    }
    return maybeRefValue;
  }

  // ============ Video 注册 ============

  /**
   * 注册 Video 元素（由 VideoStage 调用）
   * @param {HTMLVideoElement} video - Video 元素
   */
  function registerVideo(video) {
    if (videoElement === video) return;

    if (videoElement) {
      videoElement.removeEventListener("timeupdate", handleVideoTimeUpdate);
      videoElement.removeEventListener("seeking", handleVideoSeeking);
      videoElement.removeEventListener("seeked", handleVideoSeeked);
      videoElement.removeEventListener("play", handleVideoPlay);
      videoElement.removeEventListener("pause", handleVideoPause);
    }

    videoElement = video;

    // 设置 Video 事件监听
    if (video) {
      video.addEventListener("timeupdate", handleVideoTimeUpdate);
      video.addEventListener("seeking", handleVideoSeeking);
      video.addEventListener("seeked", handleVideoSeeked);
      video.addEventListener("play", handleVideoPlay);
      video.addEventListener("pause", handleVideoPause);
    }
  }

  /**
   * 注销 Video 元素
   */
  function unregisterVideo() {
    if (videoElement) {
      try {
        if (!videoElement.paused) {
          videoElement.pause();
        }
      } catch {
        // 某些浏览器在元素销毁阶段可能抛异常，忽略并继续解绑
      }
      videoElement.removeEventListener("timeupdate", handleVideoTimeUpdate);
      videoElement.removeEventListener("seeking", handleVideoSeeking);
      videoElement.removeEventListener("seeked", handleVideoSeeked);
      videoElement.removeEventListener("play", handleVideoPlay);
      videoElement.removeEventListener("pause", handleVideoPause);
      videoElement = null;
    }
    getPlaybackStore().setPlaying(false);
  }

  /**
   * 注册 WaveSurfer 实例（由 WaveformTimeline 调用）
   * @param {WaveSurfer} ws - WaveSurfer 实例
   */
  function registerWaveSurfer(ws) {
    if (wavesurferInstance === ws) return;
    unregisterWaveSurfer();
    wavesurferInstance = ws;
    if (!wavesurferInstance?.on) return;

    waveSurferHandlers = {
      timeupdate: (currentTime) => handleWaveSurferTimeUpdate(currentTime),
      play: () => handleWaveSurferPlay(),
      pause: () => handleWaveSurferPause(),
      finish: () => handleWaveSurferFinish(),
    };

    wavesurferInstance.on("timeupdate", waveSurferHandlers.timeupdate);
    wavesurferInstance.on("play", waveSurferHandlers.play);
    wavesurferInstance.on("pause", waveSurferHandlers.pause);
    wavesurferInstance.on("finish", waveSurferHandlers.finish);
  }

  /**
   * 注销 WaveSurfer 实例
   */
  function unregisterWaveSurfer() {
    if (wavesurferInstance && waveSurferHandlers) {
      wavesurferInstance.un?.("timeupdate", waveSurferHandlers.timeupdate);
      wavesurferInstance.un?.("play", waveSurferHandlers.play);
      wavesurferInstance.un?.("pause", waveSurferHandlers.pause);
      wavesurferInstance.un?.("finish", waveSurferHandlers.finish);
    }
    waveSurferHandlers = null;
    wavesurferInstance = null;
  }

  // ============ 锁管理 ============

  /**
   * 获取 Seek 锁
   */
  function acquireLock(reason = "") {
    isSeekingInternal = true;
    getPlaybackStore().setSeeking(true);

    // 清除之前的超时
    if (seekLockTimer) {
      clearTimeout(seekLockTimer);
    }

    // 设置超时保护
    seekLockTimer = setTimeout(() => {
      if (isSeekingInternal) {
        forceReleaseLock();
      }
    }, CONFIG.SEEK_LOCK_TIMEOUT);
  }

  /**
   * 释放 Seek 锁
   * @param {number} delay - 延迟释放时间（毫秒）
   */
  function releaseLock(delay = 0) {
    const doRelease = () => {
      isSeekingInternal = false;
      isDraggingInternal = false;
      dragSource = "";
      getPlaybackStore().setSeeking(false);

      if (seekLockTimer) {
        clearTimeout(seekLockTimer);
        seekLockTimer = null;
      }
    };

    if (delay > 0) {
      setTimeout(doRelease, delay);
    } else {
      doRelease();
    }
  }

  /**
   * 强制释放锁
   */
  function forceReleaseLock() {
    releaseLock(0);
  }

  /**
   * 检查是否被锁定
   */
  function isLocked() {
    return isSeekingInternal || Boolean(readStoreValue(getPlaybackStore().isSeeking));
  }

  /**
   * 检查是否在保护期内
   */
  function isInProtectionPeriod() {
    return Date.now() - lastSeekTime < CONFIG.SEEK_PROTECTION_PERIOD;
  }

  // ============ 核心操作 ============

  /**
   * 解析可用时长（优先使用 store，缺失时回退 video/wavesurfer）
   * 纯音频模式下，store.duration 可能在视频元数据缺失时为 0，需要兜底。
   */
  function resolveDuration(store) {
    let duration = Number(store.meta.duration) || 0;

    if (videoElement && Number.isFinite(videoElement.duration) && videoElement.duration > 0) {
      duration = Math.max(duration, videoElement.duration);
    }

    if (wavesurferInstance) {
      try {
        const wsDuration = Number(wavesurferInstance.getDuration()) || 0;
        if (wsDuration > 0) {
          duration = Math.max(duration, wsDuration);
        }
      } catch {
        // WaveSurfer 可能尚未就绪，忽略
      }
    }

    if (duration > 0 && Math.abs((Number(store.meta.duration) || 0) - duration) > 0.01) {
      store.setProjectDuration(duration);
    }

    return duration;
  }

  /**
   * 当前是否存在可用的视频时钟源
   */
  function hasVideoClockAvailable() {
    const domSrc = videoElement?.getAttribute?.("src");
    return Boolean(
      videoElement &&
      domSrc &&
      !videoElement.error
    );
  }

  /**
   * 跳转到指定时间（核心方法）
   * 这是唯一应该直接修改播放时间的入口
   *
   * @param {number} time - 目标时间（秒）
   * @param {Object} options - 选项
   * @param {boolean} options.fromDrag - 是否来自拖拽操作
   */
  function seekTo(time, options = {}) {
    const { fromDrag = false } = options;
    const store = getStore();
    const duration = resolveDuration(store);

    // 确保时间在有效范围内
    const clampedTime = duration > 0
      ? Math.max(0, Math.min(duration, time))
      : Math.max(0, time);

    // 如果不是拖拽操作，获取锁
    if (!fromDrag && !isLocked()) {
      acquireLock("seekTo");
    }

    // 更新保护时间
    lastSeekTime = Date.now();

    // 1. 更新 Store（单一数据源）
    const playbackStore = getPlaybackStore();
    playbackStore.updateCurrentTimeRaw(clampedTime);
    playbackStore.commitCurrentTime(clampedTime);

    // 2. 直接更新 Video 元素
    if (
      videoElement &&
      Math.abs(videoElement.currentTime - clampedTime) > CONFIG.SYNC_THRESHOLD
    ) {
      videoElement.currentTime = clampedTime;
    }

    // 3. 更新 WaveSurfer（如果已注册）
    syncWaveSurfer(clampedTime);

    // 如果不是拖拽操作，延迟释放锁
    if (!fromDrag) {
      releaseLock(CONFIG.SEEK_PROTECTION_PERIOD);
    }

    return clampedTime;
  }

  /**
   * 同步 WaveSurfer 到指定时间
   */
  function syncWaveSurfer(time) {
    if (!wavesurferInstance) return;

    try {
      const duration = wavesurferInstance.getDuration();
      if (duration > 0) {
        wavesurferInstance.seekTo(time / duration);
      }
    } catch (e) {
      // WaveSurfer 可能未准备好
    }
  }

  /**
   * 开始拖拽
   * @param {string} source - 拖拽来源（'progressBar', 'waveformCursor' 等）
   */
  function startDragging(source = "unknown") {
    isDraggingInternal = true;
    dragSource = source;
    acquireLock(`drag:${source}`);
  }

  /**
   * 拖拽中更新时间
   * @param {number} time - 当前拖拽位置对应的时间
   */
  function updateDragging(time) {
    if (!isDraggingInternal) {
      return;
    }

    seekTo(time, { fromDrag: true });
  }

  /**
   * 结束拖拽
   */
  function stopDragging() {
    if (!isDraggingInternal) return;

    // 更新保护时间
    lastSeekTime = Date.now();

    // 延迟释放锁，给 Video 一些时间完成 seek
    releaseLock(CONFIG.SEEK_PROTECTION_PERIOD);
  }

  /**
   * 检查是否正在拖拽
   */
  function isDragging() {
    return isDraggingInternal;
  }

  // ============ 播放控制 ============

  /**
   * 切换播放/暂停
   */
  function togglePlay() {
    const playbackStore = getPlaybackStore();
    playbackStore.setPlaying(!Boolean(readStoreValue(playbackStore.isPlaying)));
  }

  /**
   * 播放
   */
  function play() {
    getPlaybackStore().setPlaying(true);
  }

  /**
   * 暂停
   */
  function pause() {
    getPlaybackStore().setPlaying(false);
  }

  // ============ Video 事件处理 ============

  /**
   * 处理 Video 的 timeupdate 事件
   */
  function handleVideoTimeUpdate() {
    if (!videoElement) return;

    // 锁定时不处理
    if (isLocked()) {
      return;
    }

    // 保护期内不处理
    if (isInProtectionPeriod()) {
      return;
    }

    // 使用防抖
    if (timeUpdateTimer) {
      clearTimeout(timeUpdateTimer);
    }

    timeUpdateTimer = setTimeout(() => {
      // 再次检查状态（防抖期间可能发生变化）
      if (isLocked() || isInProtectionPeriod()) {
        return;
      }

      const playbackStore = getPlaybackStore();
      const latestVideoTime = videoElement?.currentTime;
      const currentStoreTime = Number(readStoreValue(playbackStore.currentTime));
      if (!Number.isFinite(latestVideoTime) || !Number.isFinite(currentStoreTime)) {
        return;
      }
      const diff = Math.abs(latestVideoTime - currentStoreTime);

      // 只有差异足够大时才更新
      if (diff > CONFIG.SYNC_THRESHOLD) {
        playbackStore.updateCurrentTimeRaw(latestVideoTime);
        playbackStore.commitCurrentTime(latestVideoTime);
      }
    }, CONFIG.DEBOUNCE_DELAY);
  }

  /**
   * 处理 WaveSurfer 的 timeupdate（仅在无视频时钟时生效）
   */
  function handleWaveSurferTimeUpdate(currentTime) {
    if (hasVideoClockAvailable()) {
      return;
    }

    if (!Number.isFinite(currentTime)) {
      return;
    }

    if (isLocked() || isInProtectionPeriod()) {
      return;
    }

    const now = Date.now();
    // WaveSurfer 在播放中高频触发 timeupdate，使用节流避免持续重置定时器导致“仅暂停时更新”。
    if (now - lastWaveSurferSyncAt < CONFIG.DEBOUNCE_DELAY) {
      return;
    }
    lastWaveSurferSyncAt = now;

    const playbackStore = getPlaybackStore();
    const currentStoreTime = Number(readStoreValue(playbackStore.currentTime));
    if (!Number.isFinite(currentStoreTime)) {
      return;
    }
    const diff = Math.abs(currentTime - currentStoreTime);
    if (diff > CONFIG.SYNC_THRESHOLD) {
      playbackStore.updateCurrentTimeRaw(currentTime);
      playbackStore.commitCurrentTime(currentTime);
    }
    resolveDuration(getStore());
  }

  /**
   * 处理 Video 的 seeking 事件
   */
  function handleVideoSeeking() {
    // 可用于调试
  }

  /**
   * 处理 Video 的 seeked 事件
   */
  function handleVideoSeeked() {
    // 可用于调试
  }

  /**
   * 处理 Video 的 play 事件
   */
  function handleVideoPlay() {
    const playbackStore = getPlaybackStore();
    if (!Boolean(readStoreValue(playbackStore.isPlaying))) {
      playbackStore.setPlaying(true);
    }
  }

  /**
   * 处理 WaveSurfer 的 play（仅在无视频时钟时生效）
   */
  function handleWaveSurferPlay() {
    if (hasVideoClockAvailable()) {
      return;
    }
    const playbackStore = getPlaybackStore();
    if (!Boolean(readStoreValue(playbackStore.isPlaying))) {
      playbackStore.setPlaying(true);
    }
  }

  /**
   * 处理 Video 的 pause 事件
   */
  function handleVideoPause() {
    const playbackStore = getPlaybackStore();
    if (Boolean(readStoreValue(playbackStore.isPlaying))) {
      playbackStore.setPlaying(false);
    }
  }

  /**
   * 处理 WaveSurfer 的 pause（仅在无视频时钟时生效）
   */
  function handleWaveSurferPause() {
    if (hasVideoClockAvailable()) {
      return;
    }
    const playbackStore = getPlaybackStore();
    if (Boolean(readStoreValue(playbackStore.isPlaying))) {
      playbackStore.setPlaying(false);
    }
  }

  /**
   * 处理 WaveSurfer 的 finish（仅在无视频时钟时生效）
   */
  function handleWaveSurferFinish() {
    if (hasVideoClockAvailable()) {
      return;
    }
    getPlaybackStore().setPlaying(false);
  }

  // ============ 清理 ============

  function cleanup() {
    pause();
    unregisterVideo();
    unregisterWaveSurfer();
    forceReleaseLock();

    if (timeUpdateTimer) {
      clearTimeout(timeUpdateTimer);
      timeUpdateTimer = null;
    }
    lastWaveSurferSyncAt = 0;
  }

  // ============ 公共 API ============

  return {
    // Video/WaveSurfer 注册
    registerVideo,
    unregisterVideo,
    registerWaveSurfer,
    unregisterWaveSurfer,

    // 状态查询
    isLocked,
    isDragging,
    isInProtectionPeriod,

    // 核心操作
    seekTo,
    startDragging,
    updateDragging,
    stopDragging,

    // 播放控制
    togglePlay,
    play,
    pause,

    // 锁管理（高级用法）
    acquireLock,
    releaseLock,
    forceReleaseLock,

    // 清理
    cleanup,
  };
}

// ============ 导出全局单例 ============

// 使用懒加载单例模式
let _instance = null;

export function usePlaybackManager() {
  if (!_instance) {
    _instance = createPlaybackManager();
  }
  return _instance;
}

// 提供重置方法（用于测试或热重载）
export function resetPlaybackManager() {
  if (_instance) {
    _instance.cleanup();
    _instance = null;
  }
}

export default usePlaybackManager;
