const ENABLE_TASK_EVENT_TRACE =
  String(import.meta.env.VITE_TRACE_STATE_EVENTS || '').toLowerCase() === 'true'

/**
 * Phase 0 事件投影追踪（默认关闭）
 * 用于排查 state_seq 乱序与事件投影是否被接纳。
 */
export function traceTaskProjection(eventName, payload = {}) {
  if (!ENABLE_TASK_EVENT_TRACE) {
    return
  }
  console.info('[TaskProjectionTrace]', {
    event: eventName,
    source: payload.source || 'unknown',
    jobId: payload.jobId || null,
    state_seq: payload.state_seq ?? null,
    accepted: payload.accepted ?? null,
    detail: payload.detail || null,
    at: Date.now(),
  })
}

export function isTaskEventTraceEnabled() {
  return ENABLE_TASK_EVENT_TRACE
}
