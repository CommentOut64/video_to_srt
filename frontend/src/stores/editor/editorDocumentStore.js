// V3.2.5+dev.20260314.01: 字幕文档唯一真源
import { defineStore } from 'pinia'
import { shallowReactive, shallowRef, ref, markRaw } from 'vue'

export const useEditorDocumentStore = defineStore('editorDocument', () => {
  // ─── 核心状态 ───
  const entities = shallowReactive(new Map())
  const coldEntities = new Map()
  const order = shallowRef([])
  const indexById = new Map()
  const bindingBySegmentId = new Map()
  const bindingBySentenceIndex = new Map()
  const revision = ref(0)
  const dirtySet = new Set()
  const tombstones = shallowRef([])
  const entityVersionTokens = new Map()

  // ─── 只读查询 ───
  function getEntity(localId) {
    return entities.get(localId)
  }

  function getCold(localId) {
    return coldEntities.get(localId)
  }

  function getBySegmentId(segmentId) {
    const localId = bindingBySegmentId.get(segmentId)
    return localId ? entities.get(localId) : undefined
  }

  function getBySentenceIndex(idx) {
    const localId = bindingBySentenceIndex.get(idx)
    return localId ? entities.get(localId) : undefined
  }

  function getOrderIndex(localId) {
    return indexById.get(localId) ?? -1
  }

  function getNeighbors(localId) {
    const idx = indexById.get(localId)
    if (idx === undefined) return { prev: null, next: null }
    const arr = order.value
    return {
      prev: idx > 0 ? arr[idx - 1] : null,
      next: idx < arr.length - 1 ? arr[idx + 1] : null,
    }
  }

  function getCount() {
    return entities.size
  }

  function isEmpty() {
    return entities.size === 0
  }

  // ─── 时间范围查询 ───
  function findInsertPosition(targetMs) {
    const arr = order.value
    let lo = 0, hi = arr.length
    while (lo < hi) {
      const mid = (lo + hi) >>> 1
      const entity = entities.get(arr[mid])
      if (entity && entity.startMs < targetMs) {
        lo = mid + 1
      } else {
        hi = mid
      }
    }
    return lo
  }

  function findByTimeRange(rangeStartMs, rangeEndMs) {
    const arr = order.value
    let lo = 0, hi = arr.length
    while (lo < hi) {
      const mid = (lo + hi) >>> 1
      const entity = entities.get(arr[mid])
      if (entity && entity.endMs < rangeStartMs) {
        lo = mid + 1
      } else {
        hi = mid
      }
    }
    const result = []
    for (let i = lo; i < arr.length; i++) {
      const entity = entities.get(arr[i])
      if (!entity || entity.startMs > rangeEndMs) break
      if (!entity.isDeleted) result.push(arr[i])
    }
    return result
  }

  // ─── 索引重建 ───
  function rebuildIndexById() {
    indexById.clear()
    order.value.forEach((localId, idx) => {
      indexById.set(localId, idx)
    })
  }

  // ─── 原子写操作（仅由 reducer 调用） ───
  function _applyInsert(localId, hot, cold, afterLocalId) {
    entities.set(localId, hot)
    if (cold) {
      coldEntities.set(localId, markRaw(cold))
      if (cold.segmentId) {
        bindingBySegmentId.set(cold.segmentId, localId)
      }
      if (cold.sentenceIndex !== null && cold.sentenceIndex !== undefined) {
        bindingBySentenceIndex.set(cold.sentenceIndex, localId)
      }
    }

    const pos = afterLocalId !== null && afterLocalId !== undefined
      ? (indexById.get(afterLocalId) ?? -1) + 1
      : findInsertPosition(hot.startMs)

    const newOrder = [...order.value]
    newOrder.splice(pos, 0, localId)
    order.value = newOrder

    rebuildIndexById()
    dirtySet.add(localId)
    revision.value++
    return true
  }

  function _applyDelete(localId) {
    const entity = entities.get(localId)
    if (!entity) return false

    const cold = coldEntities.get(localId)
    if (cold?.segmentId) {
      tombstones.value = [...tombstones.value, {
        localId,
        segmentId: cold.segmentId,
        deletedAt: Date.now(),
      }]
    }

    entities.delete(localId)
    coldEntities.delete(localId)
    entityVersionTokens.delete(localId)

    const newOrder = order.value.filter(id => id !== localId)
    order.value = newOrder

    rebuildIndexById()
    if (cold?.segmentId) bindingBySegmentId.delete(cold.segmentId)
    if (cold && cold.sentenceIndex !== null && cold.sentenceIndex !== undefined) {
      bindingBySentenceIndex.delete(cold.sentenceIndex)
    }
    dirtySet.delete(localId)
    revision.value++
    return true
  }

  function _applyUpdate(localId, patch) {
    const entity = entities.get(localId)
    if (!entity) return false

    Object.assign(entity, patch)
    entity.revision++

    let token = entityVersionTokens.get(localId)
    if (!token) {
      token = shallowRef(entity.revision)
      entityVersionTokens.set(localId, token)
    } else {
      token.value = entity.revision
    }

    dirtySet.add(localId)
    revision.value++
    return true
  }

  function _applyReorder(localId) {
    const entity = entities.get(localId)
    if (!entity) return false

    const newOrder = [...order.value].sort((a, b) => {
      const ea = entities.get(a)
      const eb = entities.get(b)
      return (ea?.startMs ?? 0) - (eb?.startMs ?? 0)
    })
    order.value = newOrder
    rebuildIndexById()
    revision.value++
    return true
  }

  function _applyBinding(localId, segmentId) {
    const cold = coldEntities.get(localId)
    if (!cold) return false
    cold.segmentId = segmentId
    bindingBySegmentId.set(segmentId, localId)
    return true
  }

  function updateColdBinding(localId, segmentId) {
    return _applyBinding(localId, segmentId)
  }

  function _applyColdUpdate(localId, coldPatch) {
    const cold = coldEntities.get(localId)
    if (!cold) return false
    Object.assign(cold, coldPatch)
    return true
  }

  function _applyBatchReplace(replacements) {
    for (const r of replacements) {
      _applyUpdate(r.localId, { text: r.after.text, isModified: true })
    }
    return true
  }

  function _applyServerReplace(oldLocalIds, newEntities) {
    oldLocalIds.forEach(id => _applyDelete(id))
    newEntities.forEach(e => {
      const hot = { localId: e.localId, text: e.text, startMs: e.startMs, endMs: e.endMs, isDraft: false, isModified: false, isDeleted: false, revision: 0 }
      const cold = e.cold ? { localId: e.localId, ...e.cold } : null
      _applyInsert(e.localId, hot, cold, null)
    })
    return true
  }

  function takeSnapshot() {
    const entitiesArray = Array.from(entities.entries()).map(([id, e]) => e)
    const coldEntitiesArray = Array.from(coldEntities.entries()).map(([id, c]) => ({ localId: id, ...c }))
    const bindings = Array.from(bindingBySegmentId.entries()).map(([segmentId, localId]) => ({ localId, segmentId }))
    const sentenceBindings = Array.from(bindingBySentenceIndex.entries()).map(([sentenceIndex, localId]) => ({ localId, sentenceIndex }))
    return {
      revision: revision.value,
      entities: entitiesArray,
      coldEntities: coldEntitiesArray,
      order: order.value,
      bindings,
      sentenceBindings,
      tombstones: tombstones.value,
      createdAt: Date.now(),
    }
  }

  function restoreFromSnapshot(snapshot) {
    entities.clear()
    coldEntities.clear()
    indexById.clear()
    bindingBySegmentId.clear()
    bindingBySentenceIndex.clear()
    entityVersionTokens.clear()
    dirtySet.clear()

    snapshot.entities.forEach(e => entities.set(e.localId, e))
    if (snapshot.coldEntities) {
      snapshot.coldEntities.forEach(c => {
        const { localId, ...cold } = c
        coldEntities.set(localId, markRaw(cold))
      })
    }
    order.value = snapshot.order
    rebuildIndexById()
    snapshot.bindings.forEach(b => bindingBySegmentId.set(b.segmentId, b.localId))
    snapshot.sentenceBindings.forEach(b => bindingBySentenceIndex.set(b.sentenceIndex, b.localId))
    tombstones.value = snapshot.tombstones || []
    revision.value = snapshot.revision
  }

  return {
    entities,
    coldEntities,
    order,
    revision,
    tombstones,
    entityVersionTokens,
    bindingBySegmentId,
    bindingBySentenceIndex,
    getEntity,
    getCold,
    getBySegmentId,
    getBySentenceIndex,
    getOrderIndex,
    getNeighbors,
    getCount,
    isEmpty,
    findInsertPosition,
    findByTimeRange,
    _applyInsert,
    _applyDelete,
    _applyUpdate,
    _applyReorder,
    _applyBinding,
    _applyColdUpdate,
    _applyBatchReplace,
    _applyServerReplace,
    updateColdBinding,
    takeSnapshot,
    restoreFromSnapshot,
  }
})
