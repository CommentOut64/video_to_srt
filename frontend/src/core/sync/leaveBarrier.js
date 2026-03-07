/**
 * 离开编辑页前统一执行同步栅栏。
 *
 * 设计原因：
 * - 导出前已经统一走 `flushAllSync()`，离开页面也必须使用同一套栅栏，
 *   否则会出现“导出不丢数据，但返回列表可能丢最后一次编辑”的行为分裂。
 * - `flushAllSync()` 必须无条件执行，不能仅依赖 `isDirty`，因为待上送队列/结构性请求
 *   与本地脏标记并不总是严格等价。
 */
export async function runEditorLeaveBarrier({ syncCoordinator, projectStore, isDirty = false }) {
  if (!syncCoordinator || typeof syncCoordinator.flushAllSync !== 'function') {
    throw new Error('缺少 syncCoordinator.flushAllSync，无法执行离开栅栏')
  }
  if (!projectStore || typeof projectStore.saveProject !== 'function') {
    throw new Error('缺少 projectStore.saveProject，无法执行离开栅栏')
  }

  await syncCoordinator.flushAllSync()

  if (isDirty) {
    await projectStore.saveProject()
  }
}
