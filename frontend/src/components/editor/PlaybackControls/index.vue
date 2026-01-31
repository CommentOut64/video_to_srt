<template>
  <!-- 底座模式或紧凑模式 -->
  <div class="playback-controls" :class="{ compact, pedestal, disabled: !isVideoReady }">
    <!-- 主控制区 -->
    <div class="controls-main">
      <!-- 快退 -->
      <el-tooltip content="快退5秒" placement="top" :show-after="500">
        <button class="ctrl-btn" :disabled="!isVideoReady" @click="seek(-seekStep)">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M11 18V6l-8.5 6 8.5 6zm.5-6l8.5 6V6l-8.5 6z" />
          </svg>
        </button>
      </el-tooltip>

      <!-- 播放/暂停 -->
      <el-tooltip :content="isPlaying ? '暂停' : '播放'" placement="top" :show-after="500">
        <button class="ctrl-btn play" :disabled="!isVideoReady" @click="togglePlay">
          <svg v-if="!isPlaying" viewBox="0 0 24 24" fill="currentColor">
            <path d="M8 5v14l11-7z" />
          </svg>
          <svg v-else viewBox="0 0 24 24" fill="currentColor">
            <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
          </svg>
        </button>
      </el-tooltip>

      <!-- 快进 -->
      <el-tooltip content="快进5秒" placement="top" :show-after="500">
        <button class="ctrl-btn" :disabled="!isVideoReady" @click="seek(seekStep)">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M4 18l8.5-6L4 6v12zm9-12v12l8.5-6L13 6z" />
          </svg>
        </button>
      </el-tooltip>
    </div>

    <!-- 进度条区域 -->
    <div class="controls-progress" :class="{ dragging: isDraggingProgress }">
      <span class="time-display time-current">{{ formatTime(currentTime) }}</span>

      <div class="progress-bar" ref="progressRef" @mousedown="handleProgressMouseDown">
        <div class="progress-track">
          <div class="progress-buffered" :style="{ width: bufferedPercent + '%' }"></div>
          <div class="progress-fill" :style="{ width: progressPercent + '%' }">
            <div class="progress-thumb"></div>
          </div>
        </div>
      </div>

      <span class="time-display time-total">{{ formatTime(duration) }}</span>
    </div>

    <!-- 辅助控制区 -->
    <div class="controls-extra">
      <!-- 音量控制 -->
      <div class="volume-control" v-if="showVolume">
        <el-tooltip :content="isMuted ? '取消静音' : '静音'" placement="top" :show-after="500">
          <button class="ctrl-btn sm" @click="toggleMute">
            <svg v-if="isMuted || volume === 0" viewBox="0 0 24 24" fill="currentColor">
              <path
                d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z"
              />
            </svg>
            <svg v-else-if="volume < 0.5" viewBox="0 0 24 24" fill="currentColor">
              <path
                d="M18.5 12c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM5 9v6h4l5 5V4L9 9H5z"
              />
            </svg>
            <svg v-else viewBox="0 0 24 24" fill="currentColor">
              <path
                d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"
              />
            </svg>
          </button>
        </el-tooltip>
        <div class="volume-slider" @click.stop>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            :value="volume"
            @input="handleVolumeChange"
          />
        </div>
      </div>

      <!-- 倍速控制 -->
      <div class="speed-control" v-if="showSpeed">
        <button class="ctrl-btn text" @click="toggleSpeedMenu" ref="speedBtnRef">
          {{ playbackRate }}x
        </button>
        <div class="speed-menu" v-show="showSpeedMenu" ref="speedMenuRef">
          <button
            v-for="speed in speedOptions"
            :key="speed"
            class="speed-option"
            :class="{ active: playbackRate === speed }"
            @click="setSpeed(speed)"
          >
            {{ speed }}x
          </button>
        </div>
      </div>

      <!-- 循环播放 -->
      <el-tooltip v-if="showLoop" content="循环播放" placement="top" :show-after="500">
        <button class="ctrl-btn sm" :class="{ active: isLooping }" @click="toggleLoop">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path
              d="M12 4V1L8 5l4 4V6c3.31 0 6 2.69 6 6 0 1.01-.25 1.97-.7 2.8l1.46 1.46C19.54 15.03 20 13.57 20 12c0-4.42-3.58-8-8-8zm0 14c-3.31 0-6-2.69-6-6 0-1.01.25-1.97.7-2.8L5.24 7.74C4.46 8.97 4 10.43 4 12c0 4.42 3.58 8 8 8v3l4-4-4-4v3z"
            />
          </svg>
        </button>
      </el-tooltip>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted, inject } from 'vue'
import { useProjectStore } from '@/stores/projectStore'
import { usePlaybackManager } from '@/services/PlaybackManager'

// Props
const props = defineProps({
  compact: { type: Boolean, default: false },
  pedestal: { type: Boolean, default: false }, // 底座模式：融入背景，无圆角
  showSpeed: { type: Boolean, default: true },
  showVolume: { type: Boolean, default: true },
  showLoop: { type: Boolean, default: true },
  seekStep: { type: Number, default: 5 },
  microSeekStep: { type: Number, default: 0.1 },
})

const emit = defineEmits(['play', 'pause', 'seek', 'speed-change', 'volume-change'])

// Store
const projectStore = useProjectStore()

// 全局播放管理器（单例）
const playbackManager = usePlaybackManager()

// 注入编辑器上下文（获取视频就绪状态）
const editorContext = inject('editorContext', { isVideoReady: computed(() => true) })
const isVideoReady = computed(() => editorContext.isVideoReady?.value ?? true)

// Refs
const progressRef = ref(null)
const speedBtnRef = ref(null)
const speedMenuRef = ref(null)

// State
const isLooping = ref(false)
const isMuted = ref(false)
const previousVolume = ref(1)
const showSpeedMenu = ref(false)
const bufferedPercent = ref(0)
const isDraggingProgress = ref(false) // 是否正在拖动进度条
const dragProgressPercent = ref(0) // 拖动时的进度百分比

// 倍速选项
const speedOptions = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

// 从 Store 获取状态
const currentTime = computed(() => projectStore.player.currentTime)
const duration = computed(() => projectStore.meta.duration || 0)
const isPlaying = computed(() => projectStore.player.isPlaying)
const playbackRate = computed(() => projectStore.player.playbackRate)
const volume = computed(() => (isMuted.value ? 0 : projectStore.player.volume))

// 进度百分比
const progressPercent = computed(() => {
  // 拖动时使用本地状态，避免响应式更新延迟导致的视觉分离
  if (isDraggingProgress.value) {
    return dragProgressPercent.value
  }
  if (!duration.value) return 0
  return (currentTime.value / duration.value) * 100
})

// 播放/暂停
function togglePlay() {
  if (!isVideoReady.value) {
    console.warn('[PlaybackControls] 视频未就绪，播放操作被拦截')
    return
  }
  playbackManager.togglePlay()
  emit(projectStore.player.isPlaying ? 'play' : 'pause')
}

// 跳转
function seek(seconds) {
  if (!isVideoReady.value) {
    console.warn('[PlaybackControls] 视频未就绪，跳转操作被拦截')
    return
  }
  const newTime = Math.max(0, Math.min(duration.value, currentTime.value + seconds))
  playbackManager.seekTo(newTime)
  emit('seek', newTime)
}

// 进度条交互 - 统一处理点击和拖拽
let rafId = null // requestAnimationFrame ID，用于节流
let pendingTime = null // 待更新的时间

function handleProgressMouseDown(e) {
  if (!isVideoReady.value) {
    console.warn('[PlaybackControls] 视频未就绪，进度条操作被拦截')
    return
  }
  if (!progressRef.value || !duration.value) return

  e.preventDefault() // 阻止默认行为，防止文本选择

  // 计算点击位置
  const rect = progressRef.value.getBoundingClientRect()
  const percent = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
  const newTime = percent * duration.value

  // 更新本地进度（立即响应 UI）
  isDraggingProgress.value = true
  dragProgressPercent.value = percent * 100
  pendingTime = newTime

  // 通知播放管理器开始拖拽（暂停播放同步）
  playbackManager.startDragging('progressBar')

  // 监听后续拖拽
  document.addEventListener('mousemove', onDrag)
  document.addEventListener('mouseup', stopDrag)
}

function onDrag(e) {
  if (!progressRef.value || !duration.value) return
  if (!isDraggingProgress.value) return

  const rect = progressRef.value.getBoundingClientRect()
  const percent = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))

  // 立即更新本地进度百分比（UI 响应）
  dragProgressPercent.value = percent * 100
  pendingTime = percent * duration.value

  // 使用 RAF 节流更新 store（减少频繁更新导致的卡顿）
  if (!rafId) {
    rafId = requestAnimationFrame(() => {
      if (pendingTime !== null) {
        playbackManager.updateDragging(pendingTime)
      }
      rafId = null
    })
  }
}

function stopDrag() {
  document.removeEventListener('mousemove', onDrag)
  document.removeEventListener('mouseup', stopDrag)

  // 取消未完成的 RAF
  if (rafId) {
    cancelAnimationFrame(rafId)
    rafId = null
  }

  if (isDraggingProgress.value) {
    // 最终 seek 到目标位置
    if (pendingTime !== null) {
      playbackManager.updateDragging(pendingTime)
    }
    playbackManager.stopDragging()
    isDraggingProgress.value = false
    pendingTime = null
  }
}

// 音量控制
function toggleMute() {
  if (isMuted.value) {
    projectStore.player.volume = previousVolume.value
    isMuted.value = false
  } else {
    previousVolume.value = projectStore.player.volume
    projectStore.player.volume = 0
    isMuted.value = true
  }
}

function handleVolumeChange(e) {
  const val = parseFloat(e.target.value)
  projectStore.player.volume = val
  if (val > 0) isMuted.value = false
  emit('volume-change', val)
}

// 倍速控制
function toggleSpeedMenu() {
  showSpeedMenu.value = !showSpeedMenu.value
}

function setSpeed(speed) {
  projectStore.player.playbackRate = speed
  showSpeedMenu.value = false
  emit('speed-change', speed)
}

// 循环播放
function toggleLoop() {
  isLooping.value = !isLooping.value
}

// 格式化时间
function formatTime(seconds) {
  if (!seconds || isNaN(seconds)) return '0:00'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) {
    return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
  }
  return `${m}:${s.toString().padStart(2, '0')}`
}

// 点击外部关闭倍速菜单
function handleClickOutside(e) {
  if (
    speedBtnRef.value &&
    !speedBtnRef.value.contains(e.target) &&
    speedMenuRef.value &&
    !speedMenuRef.value.contains(e.target)
  ) {
    showSpeedMenu.value = false
  }
}

// 循环播放逻辑
watch(currentTime, (time) => {
  if (isLooping.value && duration.value && time >= duration.value - 0.1) {
    playbackManager.seekTo(0)
    if (!isPlaying.value) togglePlay()
  }
})

// 键盘快捷键已由 EditorView 的 useShortcuts 统一管理，此处不再重复监听

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>

<style scoped>
.playback-controls {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 12px 16px;
  background: var(--af-bg-secondary);
  border-radius: var(--af-radius-lg);
  user-select: none;
}

/* 紧凑模式 */
.playback-controls.compact {
  gap: 12px;
  padding: 8px 12px;
}

/* 底座模式：融入背景，无圆角 */
.playback-controls.pedestal {
  height: 48px;
  padding: 0 20px;
  background: var(--af-bg-primary);
  border: none;
  border-radius: 0;
}

/* 禁用状态：视频未就绪时整体变灰 */
.playback-controls.disabled {
  cursor: not-allowed;
  pointer-events: none;
  opacity: 0.5;
}

/* 主控制按钮 */
.controls-main {
  display: flex;
  align-items: center;
  gap: 4px;
}

/* 控制按钮 */
.ctrl-btn {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 36px;
  height: 36px;
  border-radius: var(--af-radius-md);
  color: var(--af-text-secondary);
  transition: all var(--af-transition-fast);
}

.ctrl-btn svg {
  width: 20px;
  height: 20px;
}

.ctrl-btn:hover {
  background: var(--af-bg-tertiary);
  color: var(--af-text-primary);
}

.ctrl-btn.play {
  width: 44px;
  height: 44px;
  background: var(--af-accent-primary);
  color: var(--af-text-inverse);
}

.ctrl-btn.play svg {
  width: 24px;
  height: 24px;
}

.ctrl-btn.play:hover {
  background: var(--af-accent-primary-hover);
  color: var(--af-text-inverse);
}

.ctrl-btn.sm {
  width: 32px;
  height: 32px;
}

.ctrl-btn.sm svg {
  width: 18px;
  height: 18px;
}

.ctrl-btn.text {
  width: auto;
  padding: 0 10px;
  font-size: 13px;
  font-weight: 500;
  font-family: var(--af-font-mono);
}

.ctrl-btn.active {
  background: var(--af-accent-primary-bg);
  color: var(--af-accent-primary);
}

.ctrl-btn:disabled {
  cursor: not-allowed;
  pointer-events: none;
  opacity: 0.4;
}

/* 进度条区域 */
.controls-progress {
  display: flex;
  flex: 1;
  align-items: center;
  gap: 12px;
  min-width: 200px;
}

.controls-progress .time-display {
  min-width: 45px;
  color: var(--af-text-muted);
  font-size: 12px;
  font-family: var(--af-font-mono);
}

.controls-progress .time-display.time-total {
  text-align: right;
}

.playback-controls.compact .controls-progress .time-display {
  font-size: 11px;
}

.playback-controls.pedestal .controls-progress .time-display {
  font-size: 11px;
}

.controls-progress .progress-track {
  position: relative;
  width: 100%;
  height: 4px;
  background: var(--af-bg-tertiary);
  border-radius: 2px;
  transition: height 0.15s ease-out;
}

.controls-progress .progress-thumb {
  position: absolute;
  top: 50%;
  right: 0;
  width: 14px;
  height: 14px;
  background: var(--af-accent-primary);
  border-radius: 50%;
  transform: translateX(50%) translateY(-50%) scale(0.8);
  opacity: 0;
  transition:
    opacity 0.15s ease-out,
    transform 0.15s ease-out;
  box-shadow: var(--af-shadow-sm);
  pointer-events: none;
}

.controls-progress .progress-bar {
  position: relative;
  display: flex;
  flex: 1;
  align-items: center;
  height: 24px;
  cursor: pointer;
}

.controls-progress .progress-bar:hover .progress-track {
  height: 6px;
}

.controls-progress .progress-bar:hover .progress-thumb {
  transform: translateX(50%) translateY(-50%) scale(1);
  opacity: 1;
}

.controls-progress .progress-buffered {
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  background: var(--af-bg-buffered);
  border-radius: 2px;
}

/* 已播放进度 - 包含 thumb，使用 CSS transition 实现丝滑进度更新 */
.controls-progress .progress-fill {
  position: absolute;
  top: 0;
  left: 0;
  min-width: 0;
  height: 100%;
  background: var(--af-accent-primary);
  border-radius: 2px;

  /* 平滑过渡：在两次 timeupdate 之间插值 */
  transition: width 0.25s linear;
  will-change: width;
}


/* 拖拽时禁用进度条过渡，实现即时响应 */
.controls-progress.dragging .progress-fill {
  transition: none;
}

/* 辅助控制 */
.controls-extra {
  display: flex;
  align-items: center;
  gap: 8px;
}

/* 音量控制 */
.volume-control {
  display: flex;
  align-items: center;
  gap: 8px;
}

.volume-control .volume-slider {
  display: flex;
  align-items: center;
  width: 80px;
}

.volume-control .volume-slider input[type='range'] {
  width: 100%;
  height: 4px;
  padding: 0;
  margin: 0;
  background: var(--af-bg-tertiary);
  border-radius: 2px;
  cursor: pointer;
  appearance: none;
}

.volume-control .volume-slider input[type='range']::-webkit-slider-thumb {
  width: 12px;
  height: 12px;
  background: var(--af-text-primary);
  border-radius: 50%;
  cursor: pointer;
  transition: transform var(--af-transition-fast);
  appearance: none;
}

.volume-control .volume-slider input[type='range']::-webkit-slider-thumb:hover {
  transform: scale(1.2);
}

.volume-control .volume-slider input[type='range']::-moz-range-thumb {
  width: 12px;
  height: 12px;
  background: var(--af-text-primary);
  border: none;
  border-radius: 50%;
  cursor: pointer;
}

/* 倍速控制 */
.speed-control {
  position: relative;
}

.speed-control .speed-menu {
  position: absolute;
  bottom: 100%;
  left: 50%;
  z-index: 100;
  padding: 4px;
  background: var(--af-bg-elevated);
  border: 1px solid var(--af-border-default);
  border-radius: var(--af-radius-md);
  transform: translateX(-50%);
  margin-bottom: 8px;
  box-shadow: var(--af-shadow-lg);
}

.speed-control .speed-option {
  display: block;
  width: 100%;
  padding: 6px 16px;
  border-radius: var(--af-radius-sm);
  color: var(--af-text-normal);
  font-size: 13px;
  font-family: var(--af-font-mono);
  text-align: center;
  transition: all var(--af-transition-fast);
}

.speed-control .speed-option:hover {
  background: var(--af-bg-tertiary);
}

.speed-control .speed-option.active {
  background: var(--af-accent-primary-bg);
  color: var(--af-accent-primary);
}
</style>
