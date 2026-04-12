// V3.2.5+dev.20260316.11: 实时事件后的权威回拉策略辅助

/**
 * V2 下部分 SSE 事件会先做本地增量投影，再补一次权威回拉。
 * 这样既保留即时 UI，又避免事件载荷不完整时让本地真源长期漂移。
 */
export function scheduleAuthoritativeReloadForRealtimeEvent({
  useEditorV2,
  projectId,
  reason,
  activeJobId = null,
  scheduleReload,
}) {
  if (!useEditorV2) {
    return false
  }

  if (!projectId) {
    throw new Error(`[EditorRealtimeSyncPolicy] ${reason} 在 V2 下缺少 project_id，无法执行权威回拉`)
  }

  if (typeof scheduleReload !== 'function') {
    throw new Error('[EditorRealtimeSyncPolicy] 缺少 scheduleReload 回调')
  }

  // 任务模式收敛到 SSE 单通道：实时字幕事件不再触发即时整表回拉。
  // 权威回拉保留在冷启动、终态收口、断线恢复等显式屏障路径。
  if (activeJobId && String(activeJobId).trim()) {
    return false
  }

  scheduleReload(projectId, reason)
  return true
}

/**
 * 纯项目模式下没有转录任务态的终态回拉屏障，
 * 因此 project 频道的 added/edited/deleted 在 V2 下需要补一次权威加载。
 * 转录任务模式仍以专门的实时事件链路为主，避免每次项目镜像事件都触发额外整表回拉。
 */
export function shouldScheduleAuthoritativeReloadForProjectModeEvent({
  useEditorV2,
  activeJobId,
}) {
  if (!useEditorV2) {
    return false
  }

  return !(activeJobId && String(activeJobId).trim())
}
