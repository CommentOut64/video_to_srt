import localforage from 'localforage'

const EDIT_QUEUE_PREFIX = 'subtitle-edit-queue-'

function getQueueKey(identityId) {
  return `${EDIT_QUEUE_PREFIX}${identityId}`
}

function normalizeQueuePayload(rawPayload) {
  if (!rawPayload || typeof rawPayload !== 'object') return {}

  const normalized = {}
  Object.entries(rawPayload).forEach(([rawIndex, value]) => {
    const numericIndex = Number(rawIndex)
    if (!Number.isFinite(numericIndex)) return
    if (!value || typeof value !== 'object') return

    const sanitized = {}
    if (Object.prototype.hasOwnProperty.call(value, 'text')) {
      sanitized.text = value.text
    }
    if (Object.prototype.hasOwnProperty.call(value, 'start')) {
      sanitized.start = Number(value.start)
    }
    if (Object.prototype.hasOwnProperty.call(value, 'end')) {
      sanitized.end = Number(value.end)
    }
    normalized[String(numericIndex)] = sanitized
  })

  return normalized
}

/**
 * localforage 字幕同步队列迁移
 *
 * Phase 4：字幕同步队列迁移到 subtitleDocumentStore。
 * 当前 key 保持不变，仅做一次结构归一化，避免旧数据格式导致读取失败。
 */
export async function migrateSubtitleSyncQueue(identityId) {
  if (!identityId) return { migrated: false, count: 0 }

  try {
    const queueKey = getQueueKey(identityId)
    const rawPayload = await localforage.getItem(queueKey)
    if (!rawPayload) {
      return { migrated: false, count: 0 }
    }

    const normalized = normalizeQueuePayload(rawPayload)
    const entryCount = Object.keys(normalized).length
    await localforage.setItem(queueKey, normalized)
    return { migrated: true, count: entryCount }
  } catch (error) {
    console.warn('[migrateSubtitleSyncQueue] 队列迁移失败:', error)
    return { migrated: false, count: 0 }
  }
}
