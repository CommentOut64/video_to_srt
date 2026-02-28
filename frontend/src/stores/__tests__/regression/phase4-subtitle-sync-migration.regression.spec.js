import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import localforage from 'localforage'
import { migrateSubtitleSyncQueue } from '@/state/migrations/migrateSubtitleSyncQueue'
import { useSubtitleDocumentStore } from '@/stores/subtitleDocumentStore'

function deepClone(value) {
  return JSON.parse(JSON.stringify(value))
}

vi.mock('localforage', () => {
  const memory = new Map()
  return {
    default: {
      async getItem(key) {
        if (!memory.has(key)) {
          return null
        }
        return deepClone(memory.get(key))
      },
      async setItem(key, value) {
        memory.set(String(key), deepClone(value))
        return value
      },
      async removeItem(key) {
        memory.delete(String(key))
      },
      __reset() {
        memory.clear()
      },
      __dump(key) {
        if (!memory.has(String(key))) {
          return null
        }
        return deepClone(memory.get(String(key)))
      },
    }
  }
})

describe('Phase 4 回归 - 字幕同步队列迁移', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localforage.__reset()
  })

  it('迁移脚本归一化旧队列并保留有效条目', async () => {
    const identityId = 'phase4-job-queue-001'
    const queueKey = `subtitle-edit-queue-${identityId}`

    await localforage.setItem(queueKey, {
      '1': { text: '第一条', start: '1.25', end: 2.5, extra: 'ignore' },
      '2': { text: '第二条' },
      abc: { text: 'segment-id 索引' },
      '3': null,
      '4': 'invalid',
    })

    const result = await migrateSubtitleSyncQueue(identityId)
    const migratedPayload = localforage.__dump(queueKey)

    expect(result).toEqual({ migrated: true, count: 3 })
    expect(migratedPayload).toEqual({
      '1': { text: '第一条', start: 1.25, end: 2.5 },
      '2': { text: '第二条' },
      abc: { text: 'segment-id 索引' },
    })
  })

  it('bindSyncIdentity 可读取迁移后的队列并应用到字幕', async () => {
    const identityId = 'phase4-job-queue-002'
    const queueKey = `subtitle-edit-queue-${identityId}`

    await localforage.setItem(queueKey, {
      '0': { text: '旧文本A', start: '0.5', end: '1.5' },
      x: { text: 'segment-id 样式键' },
      '1': { text: '旧文本B' },
    })

    const subtitleDocumentStore = useSubtitleDocumentStore()
    await subtitleDocumentStore.bindSyncIdentity(identityId)

    const updates = []
    const targetStore = {
      subtitles: [
        { id: 's0', sentenceIndex: 0, segment_id: 'seg-0' },
        { id: 's1', sentenceIndex: 1, segment_id: 'seg-1' },
      ],
      toDisplayTime(value) {
        return value
      },
      updateSubtitle(id, payload) {
        updates.push({ id, payload })
      }
    }

    expect(subtitleDocumentStore.pendingCount()).toBe(3)
    expect(subtitleDocumentStore.applyPendingEditsToStore(targetStore)).toBe(2)
    expect(updates).toEqual([
      { id: 's0', payload: { text: '旧文本A', start: 0.5, end: 1.5 } },
      { id: 's1', payload: { text: '旧文本B' } },
    ])
  })
})
