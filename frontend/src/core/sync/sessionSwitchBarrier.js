import { runEditorLeaveBarrier } from './leaveBarrier'

/**
 * 路由参数切换到另一个编辑会话前的统一栅栏。
 *
 * 设计原因：
 * - `/editor/:jobId` -> `/editor/project/:projectId` 或两个项目之间切换时，组件可能不会卸载，
 *   因而不会触发 `onBeforeRouteLeave`。
 * - 这种“同组件会话切换”如果不先跑总栅栏，就会在 `resetProject()` 前丢掉当前会话最后一批
 *   待上送文本编辑、undo/redo 差异或结构性请求。
 */
export async function runEditorSessionSwitchBarrier({
  previousIdentityId = '',
  syncCoordinator,
  projectStore,
  isDirty = false,
}) {
  const normalizedPreviousIdentityId = String(previousIdentityId || '').trim()
  if (!normalizedPreviousIdentityId) {
    return
  }

  await runEditorLeaveBarrier({
    syncCoordinator,
    projectStore,
    isDirty,
  })
}
