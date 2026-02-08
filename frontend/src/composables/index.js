/**
 * Composables 统一导出入口
 */

export { useSseManager } from './useSseManager'
export { useProxyVideo, ProxyState, TranscodeDecision } from './useProxyVideo'
export { useUpdateChecker } from './useUpdateChecker'
export { useSubtitleSync } from './useSubtitleSync'

// WaveformTimeline 相关 composables
export {
  useWaveformZoom,
  getAdaptiveBarConfig,
  calculateWaveformConfig,
  ZOOM_MIN,
  ZOOM_MAX,
  ZOOM_STEP,
  ZOOM_BUTTON_STEP,
  ZOOM_WHEEL_STEP,
  ZOOM_BASE_PX_PER_SEC,
} from './useWaveformZoom'
export { useWaveformScroll } from './useWaveformScroll'
export { useWaveformCursorDrag } from './useWaveformCursorDrag'
export { useWaveformRegions } from './useWaveformRegions'
export { useWaveformContextMenu } from './useWaveformContextMenu'
