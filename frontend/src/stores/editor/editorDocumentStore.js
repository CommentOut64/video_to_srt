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
  const dirtyLocalIds = new Set()
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

  function clearMatchingTombstones({ localId = null, segmentId = null } = {}) {
    if (!localId && !segmentId) {
      return
    }

    tombstones.value = tombstones.value.filter((item) => {
      if (localId && item.localId === localId) {
        return false
      }
      if (segmentId && item.segmentId === segmentId) {
        return false
      }
      return true
    })
  }

  function collectLocalIdsBySegmentId(segmentId) {
    if (!segmentId) {
      return []
    }

    const matchedLocalIds = []
    for (const [candidateLocalId, cold] of coldEntities.entries()) {
      if (cold?.segmentId === segmentId) {
        matchedLocalIds.push(candidateLocalId)
      }
    }
    return matchedLocalIds
  }

  function enforceUniqueSegmentBinding(preferredLocalId, segmentId) {
    if (!preferredLocalId || !segmentId) {
      return
    }

    const duplicateLocalIds = collectLocalIdsBySegmentId(segmentId)
      .filter((candidateLocalId) => candidateLocalId !== preferredLocalId)

    for (const duplicateLocalId of duplicateLocalIds) {
      _applyDelete(duplicateLocalId, { trackTombstone: false })
    }

    clearMatchingTombstones({
      localId: preferredLocalId,
      segmentId,
    })
    bindingBySegmentId.set(segmentId, preferredLocalId)
  }

  // ─── 原子写操作（仅由 reducer 调用） ───
  function _applyInsert(localId, hot, cold, afterLocalId) {
    const existingCold = coldEntities.get(localId)
    const existingSentenceIndex = existingCold?.sentenceIndex
    const existingSegmentId = existingCold?.segmentId

    if (existingSegmentId && (!cold || cold.segmentId !== existingSegmentId)) {
      bindingBySegmentId.delete(existingSegmentId)
    }
    if (
      existingSentenceIndex !== null
      && existingSentenceIndex !== undefined
      && (!cold || cold.sentenceIndex !== existingSentenceIndex)
    ) {
      bindingBySentenceIndex.delete(existingSentenceIndex)
    }

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
      clearMatchingTombstones({
        localId,
        segmentId: nextCold.segmentId ?? null,
      })
      coldEntities.set(localId, nextCold)
      if (nextCold.segmentId) {
        bindingBySegmentId.set(nextCold.segmentId, localId)
        enforceUniqueSegmentBinding(localId, nextCold.segmentId)
      }
      if (nextCold.sentenceIndex !== null && nextCold.sentenceIndex !== undefined) {
        bindingBySentenceIndex.set(nextCold.sentenceIndex, localId)
      }
    }

    const hasExistingOrderEntry = indexById.has(localId)
    let newOrder = null
    let pos = 0
    if (!hasExistingOrderEntry) {
      pos = afterLocalId !== null && afterLocalId !== undefined
        ? (indexById.get(afterLocalId) ?? -1) + 1
        : findInsertPosition(nextHot.startMs)
      newOrder = [...order.value]
    } else {
      // Trade-off: 结构命令回放可能因并发对账导致同 localId 被重复插入；
      // 仅在命中重复插入时做去重重排，普通插入保持原性能路径。
      newOrder = order.value.filter((id) => id !== localId)
      if (afterLocalId !== null && afterLocalId !== undefined) {
        const anchorIndex = newOrder.indexOf(afterLocalId)
        pos = anchorIndex >= 0 ? anchorIndex + 1 : newOrder.length
      } else {
        while (pos < newOrder.length) {
          const candidate = entities.get(newOrder[pos])
          if ((candidate?.startMs ?? 0) >= nextHot.startMs) {
            break
          }
          pos += 1
        }
      }
    }
    newOrder.splice(pos, 0, localId)
    order.value = newOrder

    rebuildIndexById()
    dirtyLocalIds.add(localId)
    revision.value++
    return true
  }

  function _applyDelete(localId, options = {}) {
    const { trackTombstone = true } = options
    const entity = entities.get(localId)
    if (!entity) return false

    const cold = coldEntities.get(localId)
    if (trackTombstone && cold?.segmentId) {
      clearMatchingTombstones({
        localId,
        segmentId: cold.segmentId,
      })
      tombstones.value = [...tombstones.value, {
        localId,
        segmentId: cold.segmentId,
        deletedAt: Date.now(),
        before: {
          text: entity.text ?? '',
          startMs: entity.startMs ?? 0,
          endMs: entity.endMs ?? 0,
          isDraft: Boolean(entity.isDraft),
        },
      }]
    } else {
      clearMatchingTombstones({
        localId,
        segmentId: cold?.segmentId ?? null,
      })
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
    dirtyLocalIds.delete(localId)
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

    dirtyLocalIds.add(localId)
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
      clearMatchingTombstones({
        localId,
        segmentId,
      })
      bindingBySegmentId.set(segmentId, localId)
      enforceUniqueSegmentBinding(localId, segmentId)
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
    oldLocalIds.forEach(id => _applyDelete(id, { trackTombstone: false }))
    newEntities.forEach(e => {
      const hot = { localId: e.localId, text: e.text, startMs: e.startMs, endMs: e.endMs, isDraft: false, isModified: false, isDeleted: false, revision: 0 }
      const cold = e.cold ? { localId: e.localId, ...e.cold } : null
      _applyInsert(e.localId, hot, cold, null)
    })
    return true
  }

  function markDirty(localId) {
    if (!localId) return
    dirtyLocalIds.add(localId)
  }

  function getDirtyLocalIds() {
    return [...dirtyLocalIds]
  }

  function clearDirtyFlags(localIds = []) {
    if (!Array.isArray(localIds) || localIds.length === 0) {
      return
    }
    localIds.forEach((localId) => {
      dirtyLocalIds.delete(localId)
    })
  }

  function clearAllDirty() {
    dirtyLocalIds.clear()
  }

  function getTombstones() {
    return [...tombstones.value]
  }

  function clearTombstones({ localIds = [], segmentIds = [] } = {}) {
    const localIdSet = new Set(localIds.filter(Boolean))
    const segmentIdSet = new Set(segmentIds.filter(Boolean))
    if (localIdSet.size === 0 && segmentIdSet.size === 0) {
      return
    }

    tombstones.value = tombstones.value.filter((item) => {
      if (localIdSet.has(item.localId)) {
        return false
      }
      if (segmentIdSet.has(item.segmentId)) {
        return false
      }
      return true
    })
  }

  function replaceTombstones(nextTombstones = []) {
    tombstones.value = Array.isArray(nextTombstones) ? [...nextTombstones] : []
  }

  function upsertTombstone(nextTombstone) {
    if (!nextTombstone || typeof nextTombstone !== 'object') {
      return false
    }

    const localId = nextTombstone.localId ?? null
    const segmentId = nextTombstone.segmentId ?? null
    if (!localId && !segmentId) {
      return false
    }

    const tombstone = {
      localId,
      segmentId,
      deletedAt: Number(nextTombstone.deletedAt ?? Date.now()),
      before: {
        text: String(nextTombstone?.before?.text ?? ''),
        startMs: Number(nextTombstone?.before?.startMs ?? 0),
        endMs: Number(nextTombstone?.before?.endMs ?? 0),
        isDraft: Boolean(nextTombstone?.before?.isDraft),
      },
    }

    const nextTombstones = tombstones.value.filter((item) => {
      if (localId && item.localId === localId) {
        return false
      }
      if (segmentId && item.segmentId === segmentId) {
        return false
      }
      return true
    })

    nextTombstones.push(tombstone)
    tombstones.value = nextTombstones
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
    dirtyLocalIds.clear()

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
    dirtyLocalIds.clear()
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
    markDirty,
    getDirtyLocalIds,
    clearDirtyFlags,
    clearAllDirty,
    getTombstones,
    clearTombstones,
    replaceTombstones,
    upsertTombstone,
    takeSnapshot,
    restoreFromSnapshot,
    clearDocument,
  }
})
