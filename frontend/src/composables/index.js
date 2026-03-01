/**
 * Composables 统一导出入口
 */

export { useProxyVideo, ProxyState, TranscodeDecision } from './useProxyVideo'

// 同音检索与批量替换
export {
  useHomophoneSearch,
  SearchMode,
  SortMode,
  IndexStatus,
} from './useHomophoneSearch'

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
