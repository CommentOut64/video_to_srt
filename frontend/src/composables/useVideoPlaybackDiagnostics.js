import { computed, readonly, ref } from 'vue'

const GLOBAL_STATE_KEY = '__AF_VIDEO_PLAYBACK_DIAGNOSTICS__'
const GLOBAL_API_KEY = '__AF_VIDEO_PLAYBACK_DIAGNOSTICS_API__'

const state = ref({
  mediaId: null,
  mediaProfile: null,
  mediaSourceKind: 'audio_only',
  currentResolution: null,
  currentSourceUrl: null,
  sourceChangedAt: 0,
  lastFirstFrameMs: null,
  pendingFirstFrame: false,
  pendingSeekStartedAt: 0,
  lastSeekLatencyMs: null,
  lastSeekTargetTime: null,
  lastWaitingAt: 0,
  stallCount: 0,
  stallTotalMs: 0,
  lastStallMs: null,
  isStalling: false,
  lastErrorMessage: '',
  lastErrorCode: null,
  droppedVideoFrames: null,
  totalVideoFrames: null,
  lastQualitySampleAt: 0,
  lastVideoTime: 0,
  lastUpdatedAt: '',
})

let activeVideoElement = null

function setLastUpdatedAt() {
  state.value.lastUpdatedAt = new Date().toLocaleString()
}

function ensureGlobalStateExposure() {
  if (typeof window === 'undefined') {
    return
  }
  window[GLOBAL_STATE_KEY] = state.value
  window[GLOBAL_API_KEY] = {
    getState: () => state.value,
    reset: resetPlaybackDiagnostics,
  }
}

function cloneState() {
  return {
    mediaId: null,
    mediaProfile: null,
    mediaSourceKind: 'audio_only',
    currentResolution: null,
    currentSourceUrl: null,
    sourceChangedAt: 0,
    lastFirstFrameMs: null,
    pendingFirstFrame: false,
    pendingSeekStartedAt: 0,
    lastSeekLatencyMs: null,
    lastSeekTargetTime: null,
    lastWaitingAt: 0,
    stallCount: 0,
    stallTotalMs: 0,
    lastStallMs: null,
    isStalling: false,
    lastErrorMessage: '',
    lastErrorCode: null,
    droppedVideoFrames: null,
    totalVideoFrames: null,
    lastQualitySampleAt: 0,
    lastVideoTime: 0,
    lastUpdatedAt: '',
  }
}

function classifyMediaSourceKind({ sourceUrl, resolution }) {
  if (!sourceUrl) {
    return 'audio_only'
  }
  if (resolution === 'source' || String(sourceUrl).includes('/video/source')) {
    return 'source_video'
  }
  return 'proxy_video'
}

function sampleVideoQuality(video = activeVideoElement) {
  if (!video) {
    return
  }

  let droppedVideoFrames = null
  let totalVideoFrames = null

  if (typeof video.getVideoPlaybackQuality === 'function') {
    const quality = video.getVideoPlaybackQuality()
    droppedVideoFrames = Number.isFinite(quality?.droppedVideoFrames)
      ? quality.droppedVideoFrames
      : null
    totalVideoFrames = Number.isFinite(quality?.totalVideoFrames)
      ? quality.totalVideoFrames
      : null
  } else {
    const dropped = Number(video.webkitDroppedFrameCount)
    const decoded = Number(video.webkitDecodedFrameCount)
    droppedVideoFrames = Number.isFinite(dropped) ? dropped : null
    totalVideoFrames = Number.isFinite(decoded) ? decoded : null
  }

  state.value.droppedVideoFrames = droppedVideoFrames
  state.value.totalVideoFrames = totalVideoFrames
  state.value.lastQualitySampleAt = Date.now()
  setLastUpdatedAt()
  ensureGlobalStateExposure()
}

function updatePlaybackContext(payload = {}) {
  const sourceUrl = payload.sourceUrl || null
  const resolution = payload.resolution || null
  const mediaProfile = payload.mediaProfile || state.value.mediaProfile || null
  const mediaId = payload.mediaId || state.value.mediaId || null
  const mediaSourceKind = classifyMediaSourceKind({ sourceUrl, resolution })

  const sourceChanged = sourceUrl !== state.value.currentSourceUrl
  state.value.mediaId = mediaId
  state.value.mediaProfile = mediaProfile
  state.value.currentResolution = resolution
  state.value.mediaSourceKind = mediaSourceKind

  if (sourceChanged) {
    state.value.currentSourceUrl = sourceUrl
    state.value.sourceChangedAt = sourceUrl ? Date.now() : 0
    state.value.pendingFirstFrame = Boolean(sourceUrl)
    state.value.lastFirstFrameMs = sourceUrl ? null : state.value.lastFirstFrameMs
    state.value.lastErrorMessage = ''
    state.value.lastErrorCode = null
  }

  if (!sourceUrl) {
    state.value.pendingSeekStartedAt = 0
    state.value.isStalling = false
    state.value.lastWaitingAt = 0
  }

  setLastUpdatedAt()
  ensureGlobalStateExposure()
}

function registerVideoElement(video, mediaId = '') {
  activeVideoElement = video || null
  if (mediaId) {
    state.value.mediaId = mediaId
  }
  sampleVideoQuality(activeVideoElement)
  ensureGlobalStateExposure()
}

function recordLoadedMetadata(payload = {}) {
  if (payload.duration && Number.isFinite(payload.duration)) {
    state.value.duration = payload.duration
  }
  sampleVideoQuality(payload.video)
  setLastUpdatedAt()
  ensureGlobalStateExposure()
}

function recordLoadedData(payload = {}) {
  if (state.value.pendingFirstFrame && state.value.sourceChangedAt > 0) {
    state.value.lastFirstFrameMs = Date.now() - state.value.sourceChangedAt
    state.value.pendingFirstFrame = false
  }
  sampleVideoQuality(payload.video)
  setLastUpdatedAt()
  ensureGlobalStateExposure()
}

function recordTimeUpdate(payload = {}) {
  if (Number.isFinite(payload.currentTime)) {
    state.value.lastVideoTime = payload.currentTime
  }
  sampleVideoQuality(payload.video)
  setLastUpdatedAt()
  ensureGlobalStateExposure()
}

function recordSeeking(payload = {}) {
  state.value.pendingSeekStartedAt = Date.now()
  state.value.lastSeekTargetTime = Number.isFinite(payload.currentTime) ? payload.currentTime : state.value.lastVideoTime
  setLastUpdatedAt()
  ensureGlobalStateExposure()
}

function recordSeeked(payload = {}) {
  if (state.value.pendingSeekStartedAt > 0) {
    state.value.lastSeekLatencyMs = Date.now() - state.value.pendingSeekStartedAt
  }
  state.value.pendingSeekStartedAt = 0
  if (Number.isFinite(payload.currentTime)) {
    state.value.lastVideoTime = payload.currentTime
  }
  sampleVideoQuality(payload.video)
  setLastUpdatedAt()
  ensureGlobalStateExposure()
}

function beginStall() {
  if (state.value.isStalling) {
    return
  }
  state.value.isStalling = true
  state.value.lastWaitingAt = Date.now()
  state.value.stallCount += 1
  setLastUpdatedAt()
  ensureGlobalStateExposure()
}

function endStall(payload = {}) {
  if (!state.value.isStalling) {
    sampleVideoQuality(payload.video)
    return
  }
  const elapsedMs = Math.max(0, Date.now() - state.value.lastWaitingAt)
  state.value.lastStallMs = elapsedMs
  state.value.stallTotalMs += elapsedMs
  state.value.isStalling = false
  state.value.lastWaitingAt = 0
  sampleVideoQuality(payload.video)
  setLastUpdatedAt()
  ensureGlobalStateExposure()
}

function recordWaiting(payload = {}) {
  beginStall()
  sampleVideoQuality(payload.video)
}

function recordStalled(payload = {}) {
  beginStall()
  sampleVideoQuality(payload.video)
}

function recordPlaying(payload = {}) {
  endStall(payload)
}

function recordCanPlay(payload = {}) {
  endStall(payload)
}

function recordError(payload = {}) {
  state.value.lastErrorCode = payload.code ?? null
  state.value.lastErrorMessage = payload.message || '未知视频错误'
  state.value.pendingFirstFrame = false
  state.value.pendingSeekStartedAt = 0
  state.value.isStalling = false
  state.value.lastWaitingAt = 0
  setLastUpdatedAt()
  ensureGlobalStateExposure()
}

function resetPlaybackDiagnostics() {
  activeVideoElement = null
  state.value = cloneState()
  ensureGlobalStateExposure()
}

const playbackSummary = computed(() => ({
  media_source_kind: state.value.mediaSourceKind,
  media_profile: state.value.mediaProfile,
  current_resolution: state.value.currentResolution,
  video_first_frame_ms: state.value.lastFirstFrameMs,
  video_seek_latency_ms: state.value.lastSeekLatencyMs,
  video_stall_count: state.value.stallCount,
  video_stall_total_ms: state.value.stallTotalMs,
  video_last_stall_ms: state.value.lastStallMs,
  video_decode_drop_count: state.value.droppedVideoFrames,
  video_total_frame_count: state.value.totalVideoFrames,
  last_error_message: state.value.lastErrorMessage,
  last_updated_at: state.value.lastUpdatedAt,
}))

ensureGlobalStateExposure()

export function useVideoPlaybackDiagnostics() {
  return {
    state: readonly(state),
    playbackSummary,
    registerVideoElement,
    updatePlaybackContext,
    recordLoadedMetadata,
    recordLoadedData,
    recordTimeUpdate,
    recordSeeking,
    recordSeeked,
    recordWaiting,
    recordStalled,
    recordPlaying,
    recordCanPlay,
    recordError,
    sampleVideoQuality,
    resetPlaybackDiagnostics,
  }
}
