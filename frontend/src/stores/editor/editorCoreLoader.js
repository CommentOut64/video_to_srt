// V3.2.5+dev.20260315.16: 编辑器后端字幕加载器（按需动态导入）
import { useEditorSyncEngine } from './editorSyncEngine'

function assertUnifiedSubtitleDtos(segments = []) {
  const invalidSegment = segments.find((segment) => !String(segment?.segment_id ?? '').trim())
  if (invalidSegment) {
    throw new Error('[editorCoreLoader] project subtitles 缺少 segment_id，无法作为统一 DTO 加载')
  }
}

export async function loadSubtitlesFromBackend(projectId) {
  console.log('[editorCoreLoader] 开始加载字幕，projectId:', projectId)
  const syncEngine = useEditorSyncEngine()
  const { projectApi } = await import('@/services/api')

  const segments = await projectApi.getSubtitles(projectId)
  assertUnifiedSubtitleDtos(segments)
  console.log('[editorCoreLoader] 获取到字幕数量:', segments.length)
  syncEngine.applyAuthoritativeSegments(segments, {
    preservePendingCommands: true,
  })

  console.log('[editorCoreLoader] 字幕加载完成，保留本地未同步命令并完成权威回拉')
}
