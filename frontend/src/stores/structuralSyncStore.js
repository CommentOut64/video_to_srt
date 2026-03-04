/**
 * V3.2.4+dev.20260304.01: 结构性操作飞行追踪 Store
 *
 * 职责：追踪 delete/insert/split/merge 四类即时 API 调用的
 *       inflight 状态和失败记录，供导出栅栏检查。
 *
 * 背景：这四类操作不走 subtitleDocumentStore 的防抖队列，
 *       而是直接 await API。其飞行中状态和失败对导出栅栏不可见，
 *       导致导出时可能拿到不一致的后端数据。
 *
 * 核心接口：
 * - trackOperation(type, promise) → opId  注册飞行中操作
 * - waitAll()                              等待所有飞行中操作完成
 * - clearAllErrors()                       undo/redo 对账成功后清除过期错误
 * - clearError(opId)                       插入回滚后清除单条错误
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

let _opSeq = 0

export const useStructuralSyncStore = defineStore('structuralSync', () => {
  // 飞行中的操作: opId → Promise
  const _inflight = new Map()

  // 失败记录: opId → { type, error, timestamp }
  const syncErrors = ref(new Map())

  // 计算属性
  const inflightCount = computed(() => _inflight.size)
  const errorCount = computed(() => syncErrors.value.size)
  const hasErrors = computed(() => syncErrors.value.size > 0)

  /**
   * 注册一个飞行中的结构性操作。
   *
   * @param {string} type - 操作类型: 'delete' | 'insert' | 'split' | 'merge'
   * @param {Promise} promise - API 调用 Promise
   * @returns {string} opId，可用于后续 clearError
   */
  function trackOperation(type, promise) {
    _opSeq += 1
    const opId = `${type}-${Date.now()}-${_opSeq}`

    _inflight.set(opId, promise)

    // promise 完成后自动更新状态
    promise
      .then(() => {
        _inflight.delete(opId)
        syncErrors.value.delete(opId)
      })
      .catch((error) => {
        _inflight.delete(opId)
        syncErrors.value.set(opId, {
          type,
          error: error?.message || '同步失败',
          timestamp: Date.now(),
        })
      })

    return opId
  }

  /**
   * 等待所有飞行中操作完成（不关心成功或失败）。
   * 导出栅栏调用，确保所有即时 API 调用已落地。
   */
  async function waitAll() {
    if (_inflight.size === 0) return
    await Promise.allSettled([..._inflight.values()])
  }

  /**
   * 清除所有错误记录。
   * 用于 undo/redo batch-sync 成功后：后端已被全量快照重新对账，
   * 先前的结构性操作失败不再有意义。
   */
  function clearAllErrors() {
    syncErrors.value.clear()
  }

  /**
   * 清除指定操作的错误（如插入回滚后本地已恢复一致）。
   */
  function clearError(opId) {
    syncErrors.value.delete(opId)
  }

  /**
   * 重置全部状态（切换项目时调用）。
   */
  function $reset() {
    _inflight.clear()
    syncErrors.value.clear()
  }

  return {
    syncErrors,
    inflightCount,
    errorCount,
    hasErrors,
    trackOperation,
    waitAll,
    clearAllErrors,
    clearError,
    $reset,
  }
})
