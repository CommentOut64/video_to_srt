// V3.2.5+dev.20260315.21: 字幕文档唯一真源（热实体更新改为不可变替换）
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

  function ensureEntityVersionToken(localId, entity) {
    if (!entity) return null

    let token = entityVersionTokens.get(localId)
    if (!token) {
      token = shallowRef(entity.revision ?? 0)
      entityVersionTokens.set(localId, token)
      return token
    }

    token.value = entity.revision ?? 0
    return token
  }

  // ─── 原子写操作（仅由 reducer 调用） ───
  function _applyInsert(localId, hot, cold, afterLocalId) {
    const nextHot = {
      ...hot,
      localId,
    }

    entities.set(localId, nextHot)
    ensureEntityVersionToken(localId, nextHot)
    if (cold) {
      const nextCold = markRaw({
        ...cold,
        localId,
      })
      coldEntities.set(localId, nextCold)
      if (nextCold.segmentId) {
        bindingBySegmentId.set(nextCold.segmentId, localId)
      }
      if (nextCold.sentenceIndex !== null && nextCold.sentenceIndex !== undefined) {
        bindingBySentenceIndex.set(nextCold.sentenceIndex, localId)
      }
    }

    const pos = afterLocalId !== null && afterLocalId !== undefined
      ? (indexById.get(afterLocalId) ?? -1) + 1
      : findInsertPosition(nextHot.startMs)

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

    // 不能原地变异：下游组件会通过 computed 缓存实体引用，
    // 若引用不变，Vue 会把它视为“等值”并吞掉文本/时间戳刷新。
    const nextEntity = {
      ...entity,
      ...patch,
      localId,
      revision: (entity.revision ?? 0) + 1,
    }

    entities.set(localId, nextEntity)
    ensureEntityVersionToken(localId, nextEntity)

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
    if (cold.segmentId && cold.segmentId !== segmentId) {
      bindingBySegmentId.delete(cold.segmentId)
    }
    const nextCold = markRaw({
      ...cold,
      localId,
      segmentId,
    })
    coldEntities.set(localId, nextCold)
    if (segmentId) {
      bindingBySegmentId.set(segmentId, localId)
    }
    revision.value++
    return true
  }

  function updateColdBinding(localId, segmentId) {
    return _applyBinding(localId, segmentId)
  }

  function _applyColdUpdate(localId, coldPatch) {
    const cold = coldEntities.get(localId)
    if (!cold) return false
    const nextCold = markRaw({
      ...cold,
      ...coldPatch,
      localId,
    })
    coldEntities.set(localId, nextCold)
    revision.value++
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
    const entitiesArray = Array.from(entities.entries()).map(([id, e]) => ({ ...e, localId: id }))
    const coldEntitiesArray = Array.from(coldEntities.entries()).map(([id, c]) => ({ localId: id, ...c }))
    const bindings = Array.from(bindingBySegmentId.entries()).map(([segmentId, localId]) => ({ localId, segmentId }))
    const sentenceBindings = Array.from(bindingBySentenceIndex.entries()).map(([sentenceIndex, localId]) => ({ localId, sentenceIndex }))
    return {
      revision: revision.value,
      entities: entitiesArray,
      coldEntities: coldEntitiesArray,
      order: [...order.value],
      bindings,
      sentenceBindings,
      tombstones: [...tombstones.value],
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

    snapshot.entities.forEach((e) => {
      const nextEntity = {
        ...e,
        localId: e.localId,
      }
      entities.set(e.localId, nextEntity)
      ensureEntityVersionToken(e.localId, nextEntity)
    })
    if (snapshot.coldEntities) {
      snapshot.coldEntities.forEach(c => {
        const { localId, ...cold } = c
        coldEntities.set(localId, markRaw({
          ...cold,
          localId,
        }))
      })
    }
    order.value = Array.isArray(snapshot.order) ? [...snapshot.order] : []
    rebuildIndexById()
    ;(snapshot.bindings || []).forEach(b => bindingBySegmentId.set(b.segmentId, b.localId))
    ;(snapshot.sentenceBindings || []).forEach(b => bindingBySentenceIndex.set(b.sentenceIndex, b.localId))
    tombstones.value = Array.isArray(snapshot.tombstones) ? [...snapshot.tombstones] : []
    revision.value = Number.isFinite(snapshot.revision) ? snapshot.revision : 0
  }

  // V3.2.5+dev.20260315.01: 清空文档（切换项目时使用）
  function clearDocument() {
    entities.clear()
    coldEntities.clear()
    indexById.clear()
    bindingBySegmentId.clear()
    bindingBySentenceIndex.clear()
    entityVersionTokens.clear()
    dirtySet.clear()
    order.value = []
    tombstones.value = []
    revision.value = 0
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
    clearDocument,
  }
})
