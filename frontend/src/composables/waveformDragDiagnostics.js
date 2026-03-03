/**
 * 波形拖动诊断开关（运行时可热更新）
 *
 * 用法（浏览器控制台）：
 * window.__AF_WAVEFORM_DRAG_DIAG__ = {
 *   enabled: true,
 *   disablePointerCapture: true,
 *   skipRegionRenderWhileDragging: true,
 *   bypassPlaybackManagerDuringCursorDrag: false,
 *   logRegionUpdateFlow: true,
 * }
 */

const GLOBAL_DIAG_KEY = '__AF_WAVEFORM_DRAG_DIAG__'

const DEFAULT_DIAG_CONFIG = {
  enabled: false,
  disablePointerCapture: false,
  skipRegionRenderWhileDragging: false,
  bypassPlaybackManagerDuringCursorDrag: false,
  logRegionUpdateFlow: false,
}

function sanitizeConfig(rawConfig) {
  if (!rawConfig || typeof rawConfig !== 'object') {
    return {}
  }

  return {
    enabled: rawConfig.enabled === true,
    disablePointerCapture: rawConfig.disablePointerCapture === true,
    skipRegionRenderWhileDragging: rawConfig.skipRegionRenderWhileDragging === true,
    bypassPlaybackManagerDuringCursorDrag: rawConfig.bypassPlaybackManagerDuringCursorDrag === true,
    logRegionUpdateFlow: rawConfig.logRegionUpdateFlow === true,
  }
}

export function getWaveformDragDiagnosticsConfig() {
  if (typeof window === 'undefined') {
    return { ...DEFAULT_DIAG_CONFIG }
  }

  return {
    ...DEFAULT_DIAG_CONFIG,
    ...sanitizeConfig(window[GLOBAL_DIAG_KEY]),
  }
}

export function setWaveformDragDiagnosticsConfig(nextConfig) {
  if (typeof window === 'undefined') {
    return { ...DEFAULT_DIAG_CONFIG }
  }

  const mergedConfig = {
    ...getWaveformDragDiagnosticsConfig(),
    ...sanitizeConfig(nextConfig),
  }
  window[GLOBAL_DIAG_KEY] = mergedConfig
  return mergedConfig
}

export function logWaveformDragDiagnostics(eventName, payload = {}) {
  const config = getWaveformDragDiagnosticsConfig()
  if (!config.enabled) {
    return
  }

  console.log(`[WaveformDragDiag] ${eventName}`, payload)
}

