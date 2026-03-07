import localforage from 'localforage'
import { migrateSubtitleSyncQueue } from '@/state/migrations/migrateSubtitleSyncQueue'

const EDIT_QUEUE_PREFIX = 'subtitle-edit-queue-'
const EMERGENCY_QUEUE_PREFIX = 'subtitle-edit-emergency-'
const WORKER_TIMEOUT_MS = 3000

function getQueueKey(identityId) {
  return `${EDIT_QUEUE_PREFIX}${identityId}`
}

function getEmergencyQueueKey(identityId) {
  return `${EMERGENCY_QUEUE_PREFIX}${identityId}`
}

function normalizeQueueKey(rawKey) {
  const normalized = String(rawKey ?? '').trim()
  if (!normalized) return null
  if (/^-?\d+$/.test(normalized)) {
    const numericKey = Number(normalized)
    if (Number.isFinite(numericKey)) {
      return numericKey
    }
  }
  return normalized
}

function normalizeNumericField(value) {
  if (value === undefined) return undefined
  const normalized = Number(value)
  return Number.isFinite(normalized) ? normalized : undefined
}

function sanitizeQueueUpdate(value) {
  if (!value || typeof value !== 'object') return null
  const sanitized = {}
  if (Object.prototype.hasOwnProperty.call(value, 'text')) {
    sanitized.text = value.text
  }
  const start = normalizeNumericField(value.start)
  if (start !== undefined) {
    sanitized.start = start
  }
  const end = normalizeNumericField(value.end)
  if (end !== undefined) {
    sanitized.end = end
  }
  return Object.keys(sanitized).length > 0 ? sanitized : null
}

function serializeQueue(queue) {
  const payload = {}
  if (!(queue instanceof Map)) {
    return payload
  }
  queue.forEach((value, rawKey) => {
    const normalizedKey = normalizeQueueKey(rawKey)
    const sanitized = sanitizeQueueUpdate(value)
    if (normalizedKey === null || !sanitized) return
    payload[String(normalizedKey)] = sanitized
  })
  return payload
}

function deserializeQueue(payload) {
  if (!payload || typeof payload !== 'object') {
    return new Map()
  }

  return new Map(
    Object.entries(payload)
      .map(([rawKey, value]) => {
        const normalizedKey = normalizeQueueKey(rawKey)
        const sanitized = sanitizeQueueUpdate(value)
        return [normalizedKey, sanitized]
      })
      .filter(([key, value]) => key !== null && value)
  )
}

function mergeQueues(primaryQueue = new Map(), secondaryQueue = new Map()) {
  const mergedQueue = new Map(primaryQueue instanceof Map ? primaryQueue : [])
  if (!(secondaryQueue instanceof Map)) {
    return mergedQueue
  }

  secondaryQueue.forEach((value, key) => {
    const previous = mergedQueue.get(key) || {}
    const nextValue = sanitizeQueueUpdate({ ...previous, ...value })
    if (nextValue) {
      mergedQueue.set(key, nextValue)
    }
  })
  return mergedQueue
}

class PersistenceClient {
  constructor() {
    this.worker = null
    this.useWorker = typeof Worker !== 'undefined'
    this.requestSeq = 0
    this.pendingWorkerRequests = new Map()
  }

  _ensureWorker() {
    if (!this.useWorker || this.worker || typeof Worker === 'undefined') {
      return this.worker
    }

    try {
      this.worker = new Worker(
        new URL('../../workers/editorPersistence.worker.js', import.meta.url),
        { type: 'module' }
      )
      this.worker.onmessage = (event) => {
        const message = event?.data || {}
        const { type, requestId, payload, error } = message
        const pending = this.pendingWorkerRequests.get(requestId)
        if (!pending) return
        this.pendingWorkerRequests.delete(requestId)
        clearTimeout(pending.timeoutId)
        if (type === 'serialize:result') {
          pending.resolve(payload)
          return
        }
        pending.reject(new Error(error?.message || 'Worker 持久化失败'))
      }
      this.worker.onerror = (error) => {
        this.pendingWorkerRequests.forEach(({ reject, timeoutId }) => {
          clearTimeout(timeoutId)
          reject(error instanceof Error ? error : new Error('Worker 持久化失败'))
        })
        this.pendingWorkerRequests.clear()
        this.worker?.terminate()
        this.worker = null
        this.useWorker = false
      }
    } catch (error) {
      console.warn('[PersistenceClient] Worker 初始化失败，降级主线程:', error)
      this.worker = null
      this.useWorker = false
    }

    return this.worker
  }

  async _serialize(payload) {
    const worker = this._ensureWorker()
    if (!worker) {
      return JSON.parse(JSON.stringify(payload))
    }

    this.requestSeq += 1
    const requestId = `persist-${Date.now()}-${this.requestSeq}`

    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        this.pendingWorkerRequests.delete(requestId)
        reject(new Error('Worker 持久化超时'))
      }, WORKER_TIMEOUT_MS)

      this.pendingWorkerRequests.set(requestId, { resolve, reject, timeoutId })
      worker.postMessage({ type: 'serialize', requestId, payload })
    }).catch((error) => {
      console.warn('[PersistenceClient] Worker 序列化失败，降级主线程:', error)
      this.pendingWorkerRequests.delete(requestId)
      return JSON.parse(JSON.stringify(payload))
    })
  }

  readEmergencyQueue(identityId) {
    if (!identityId || typeof window === 'undefined' || !window.localStorage) {
      return new Map()
    }

    try {
      const raw = window.localStorage.getItem(getEmergencyQueueKey(identityId))
      if (!raw) return new Map()
      return deserializeQueue(JSON.parse(raw))
    } catch (error) {
      console.warn('[PersistenceClient] 读取紧急缓冲失败:', error)
      return new Map()
    }
  }

  writeEmergencyQueue(identityId, queue) {
    if (!identityId || typeof window === 'undefined' || !window.localStorage) return

    try {
      const storageKey = getEmergencyQueueKey(identityId)
      const payload = serializeQueue(queue)
      if (Object.keys(payload).length === 0) {
        window.localStorage.removeItem(storageKey)
        return
      }
      window.localStorage.setItem(storageKey, JSON.stringify(payload))
    } catch (error) {
      console.warn('[PersistenceClient] 写入紧急缓冲失败:', error)
    }
  }

  clearEmergencyQueue(identityId) {
    if (!identityId || typeof window === 'undefined' || !window.localStorage) return

    try {
      window.localStorage.removeItem(getEmergencyQueueKey(identityId))
    } catch (error) {
      console.warn('[PersistenceClient] 清理紧急缓冲失败:', error)
    }
  }

  async loadQueue(identityId) {
    if (!identityId) return new Map()

    await migrateSubtitleSyncQueue(identityId)

    let persistedQueue = new Map()
    try {
      const payload = await localforage.getItem(getQueueKey(identityId))
      persistedQueue = deserializeQueue(payload)
    } catch (error) {
      console.warn('[PersistenceClient] 读取持久化队列失败:', error)
    }

    return mergeQueues(persistedQueue, this.readEmergencyQueue(identityId))
  }

  async saveQueue(identityId, queue) {
    if (!identityId) return

    const payload = serializeQueue(queue)
    try {
      if (Object.keys(payload).length === 0) {
        await localforage.removeItem(getQueueKey(identityId))
        this.clearEmergencyQueue(identityId)
        return
      }

      const serialized = await this._serialize(payload)
      await localforage.setItem(getQueueKey(identityId), serialized)
      this.clearEmergencyQueue(identityId)
    } catch (error) {
      console.warn('[PersistenceClient] 保存持久化队列失败:', error)
      this.writeEmergencyQueue(identityId, queue)
    }
  }

  destroy() {
    if (this.worker) {
      this.worker.terminate()
      this.worker = null
    }
    this.pendingWorkerRequests.forEach(({ reject, timeoutId }) => {
      clearTimeout(timeoutId)
      reject(new Error('PersistenceClient 已销毁'))
    })
    this.pendingWorkerRequests.clear()
  }
}

const persistenceClient = new PersistenceClient()

export {
  persistenceClient,
  getQueueKey,
  getEmergencyQueueKey,
  normalizeQueueKey,
  sanitizeQueueUpdate,
  serializeQueue,
  deserializeQueue,
  mergeQueues,
}

export default persistenceClient