/**
 * 转录配置 Store 兼容入口（Phase 4）
 *
 * 正式入口迁移至 `transcriptionPresetStore`。
 * 本文件仅保留向后兼容导出，Phase 5 删除。
 */
import { useTranscriptionPresetStore } from './transcriptionPresetStore'

export { useTranscriptionPresetStore }

export const useTranscriptionConfigStore = (...args) => {
  return useTranscriptionPresetStore(...args)
}
