/**
 * V3.2.0+dev.20260124.05: 字幕编辑同步 Composable
 *
 * 核心功能：
 * - 防抖同步用户编辑到后端
 * - 乐观更新 UI
 * - 失败重试机制
 * - 防止 AI 覆盖用户编辑
 *
 * 架构设计：
 * - 单一数据源：后端是唯一真理来源
 * - 原子化锁定：用户编辑后立即标记 is_modified=true
 * - 零竞态：AI 推送时检查标记，跳过已修改的句子
 */

import { ref, onUnmounted, unref, watch, toRaw } from 'vue'
import localforage from 'localforage'
import transcriptionApi from '@/services/api/transcriptionApi'
import projectApi from '@/services/api/projectApi'
import { useProjectStore } from '@/stores/projectStore'

/**
 * 防抖函数
 * @param {Function} fn - 要防抖的函数
 * @param {number} delay - 延迟时间（毫秒）
 * @returns {Function} 防抖后的函数
 */
function debounce(fn, delay) {
  let timeoutId = null
  return function (...args) {
    if (timeoutId) {
      clearTimeout(timeoutId)
    }
    timeoutId = setTimeout(() => {
      fn.apply(this, args)
    }, delay)
  }
}

/**
 * 字幕同步 Composable
 * @param {string} identityId - 同步身份 ID（projectId 或 jobId）
 * @returns {Object} 同步方法和状态
 */
const EDIT_QUEUE_PREFIX = 'subtitle-edit-queue-'

function getQueueKey(identityId) {
  return `${EDIT_QUEUE_PREFIX}${identityId}`
}

async function loadQueue(identityId) {
  if (!identityId) return new Map()
  try {
    const saved = await localforage.getItem(getQueueKey(identityId))
    if (!saved || typeof saved !== 'object') {
      return new Map()
    }
    const entries = Object.entries(saved).map(([key, value]) => [Number(key), value])
    return new Map(entries)
  } catch (error) {
    console.warn('[useSubtitleSync] 读取本地同步队列失败:', error)
    return new Map()
  }
}

async function saveQueue(identityId, queue) {
  if (!identityId) return
  try {
    const payload = {}
    queue.forEach((value, key) => {
      const rawValue = toRaw(value)
      if (!rawValue || typeof rawValue !== 'object') {
        payload[String(key)] = rawValue
        return
      }
      // 仅持久化可序列化字段，避免 Proxy 导致 DataCloneError
      const sanitized = {}
      if (Object.prototype.hasOwnProperty.call(rawValue, 'text')) {
        sanitized.text = rawValue.text
      }
      if (Object.prototype.hasOwnProperty.call(rawValue, 'start')) {
        sanitized.start = rawValue.start
      }
      if (Object.prototype.hasOwnProperty.call(rawValue, 'end')) {
        sanitized.end = rawValue.end
      }
      payload[String(key)] = sanitized
    })
    await localforage.setItem(getQueueKey(identityId), payload)
  } catch (error) {
    console.warn('[useSubtitleSync] 保存本地同步队列失败:', error)
  }
}

export function useSubtitleSync(identityRef) {
  const projectStore = useProjectStore()
  // 待同步的编辑队列（Map: index -> update data）
  const pendingUpdates = ref(new Map())

  // 同步状态
  const isSyncing = ref(false)
  const syncErrors = ref(new Map())
  const currentIdentityId = ref(null)

  function getActiveIdentity() {
    return unref(identityRef) || projectStore.primaryId
  }

  function hasProjectContext() {
    return Boolean(projectStore.meta.projectId)
  }

  function resolveProjectSegmentId(segmentOrIndex) {
    if (segmentOrIndex === undefined || segmentOrIndex === null) {
      return null
    }
    const target = String(segmentOrIndex)
    const direct = projectStore.subtitles.find((item) => item.segment_id === target)
    if (direct?.segment_id) {
      return direct.segment_id
    }
    const index = Number(segmentOrIndex)
    if (Number.isNaN(index)) {
      return null
    }
    const byLegacyIndex = projectStore.subtitles.find((item) => Number(item.sentenceIndex) === index)
    return byLegacyIndex?.segment_id || null
  }

  /**
   * 真正的同步逻辑
   */
  const processQueue = async () => {
    if (pendingUpdates.value.size === 0) {
      return
    }

    const identityId = getActiveIdentity()
    if (!identityId) {
      return
    }

    isSyncing.value = true

    // 复制并清空队列，防止发送过程中又有新输入
    const batch = new Map(pendingUpdates.value)
    pendingUpdates.value.clear()

    for (const [index, data] of batch) {
      try {
        if (hasProjectContext()) {
          const segmentId = resolveProjectSegmentId(index)
          if (!segmentId) {
            throw new Error(`segment_id 不存在: ${index}`)
          }
          await projectApi.updateSubtitle(projectStore.meta.projectId, segmentId, data)
        } else if (projectStore.meta.jobId) {
          await transcriptionApi.updateSubtitle(projectStore.meta.jobId, index, data)
        } else {
          throw new Error('缺少可用的字幕同步上下文')
        }

        // 成功后清除错误记录
        if (syncErrors.value.has(index)) {
          syncErrors.value.delete(index)
        }
      } catch (error) {
        // V3.2.0+dev.20260124.01: 404 错误说明任务已完成或句子不存在，无需同步
        if (error.status === 404 || error.message?.includes('不存在')) {
          // 清除错误记录，不重试
          if (syncErrors.value.has(index)) {
            syncErrors.value.delete(index)
          }
          continue
        }

        console.error(`字幕同步失败: index=${index}`, error)

        // 记录错误
        syncErrors.value.set(index, error.message || '同步失败')

        // 失败重试：放回队列
        if (!pendingUpdates.value.has(index)) {
          pendingUpdates.value.set(index, data)
        }
      }
    }

    await saveQueue(identityId, pendingUpdates.value)
    isSyncing.value = false
  }

  // 防抖的同步触发器（800ms）
  const debouncedSync = debounce(processQueue, 800)

  /**
   * 用户编辑字幕时调用
   * @param {number} index - 句子索引
   * @param {string} [text] - 新文本
   * @param {number} [start] - 新开始时间
   * @param {number} [end] - 新结束时间
   */
  const onSubtitleEdit = (index, { text, start, end }) => {
    const identityId = getActiveIdentity()
    // 构建更新数据（只包含有值的字段）
    const update = {}
    if (text !== undefined) update.text = text
    if (start !== undefined) update.start = projectStore.toBaseTime(start)
    if (end !== undefined) update.end = projectStore.toBaseTime(end)

    // 加入同步队列（Map 会自动去重，只保留最后一次修改）
    pendingUpdates.value.set(index, update)
    if (identityId) {
      saveQueue(identityId, pendingUpdates.value)
    }

    // 触发防抖同步
    debouncedSync()
  }

  /**
   * 强制立即同步（用于页面关闭前）
   */
  const forceSyncNow = async () => {
    await processQueue()
  }

  /**
   * 叠加本地未同步的编辑到 UI（意外退出恢复）
   */
  const applyPendingEditsToStore = (projectStore) => {
    if (!projectStore || pendingUpdates.value.size === 0) return 0
    let appliedCount = 0
    pendingUpdates.value.forEach((update, index) => {
      const subtitle = projectStore.subtitles.find(
        (s) => s.sentenceIndex === index || s.segment_id === index
      )
      if (!subtitle) return
      const displayUpdate = { ...update }
      if (update.start !== undefined) {
        displayUpdate.start = projectStore.toDisplayTime(update.start)
      }
      if (update.end !== undefined) {
        displayUpdate.end = projectStore.toDisplayTime(update.end)
      }
      projectStore.updateSubtitle(subtitle.id, displayUpdate, { isUserEdit: true })
      appliedCount += 1
    })
    return appliedCount
  }

  // 页面关闭前尝试强制同步
  const handleBeforeUnload = () => {
    if (pendingUpdates.value.size > 0) {
      // 注意：beforeunload 中的异步操作可能不会完成
      // 这里只是尽力而为
      processQueue()
    }
  }

  // 载入本地队列（基于当前 identity）
  watch(
    () => getActiveIdentity(),
    async (identityId) => {
      if (!identityId) return
      if (currentIdentityId.value === identityId) return
      pendingUpdates.value = await loadQueue(identityId)
      currentIdentityId.value = identityId
    },
    { immediate: true }
  )

  // 注册 beforeunload 事件
  if (typeof window !== 'undefined') {
    window.addEventListener('beforeunload', handleBeforeUnload)
  }

  // 组件卸载时清理
  onUnmounted(() => {
    if (typeof window !== 'undefined') {
      window.removeEventListener('beforeunload', handleBeforeUnload)
    }
  })

  return {
    onSubtitleEdit,
    forceSyncNow,
    applyPendingEditsToStore,
    isSyncing,
    syncErrors,
    pendingCount: () => pendingUpdates.value.size
  }
}
