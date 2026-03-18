// V3.2.5+dev.20260316.20: 基于 Dirty/Tombstone + AckShadow 的 primitive diff 构建器

function toNormalizedSegmentId(value) {
  const normalized = String(value ?? '').trim()
  return normalized || null
}

function toSafeMs(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) {
    return 0
  }
  return Math.round(numeric)
}

function resolveAckSnapshot(localId, segmentId, ackShadowByLocalId, ackLocalIdBySegmentId) {
  const directAck = localId ? ackShadowByLocalId.get(localId) : null
  if (directAck) {
    return directAck
  }

  const normalizedSegmentId = toNormalizedSegmentId(segmentId)
  if (!normalizedSegmentId) {
    return null
  }

  const mappedLocalId = ackLocalIdBySegmentId.get(normalizedSegmentId)
  if (!mappedLocalId) {
    return null
  }

  return ackShadowByLocalId.get(mappedLocalId) ?? null
}

function sortDirtyLocalIds(docStore, dirtyLocalIds) {
  const orderIndexByLocalId = new Map()
  docStore.order.forEach((localId, index) => {
    orderIndexByLocalId.set(localId, index)
  })

  return [...dirtyLocalIds].sort((left, right) => {
    const leftIndex = orderIndexByLocalId.get(left)
    const rightIndex = orderIndexByLocalId.get(right)
    if (Number.isFinite(leftIndex) && Number.isFinite(rightIndex)) {
      return leftIndex - rightIndex
    }
    if (Number.isFinite(leftIndex)) return -1
    if (Number.isFinite(rightIndex)) return 1
    return String(left).localeCompare(String(right))
  })
}

export function buildPrimitiveDiffEntries({
  docStore,
  sessionStore,
  ackShadowByLocalId,
  ackLocalIdBySegmentId,
  maxOps = Infinity,
}) {
  const entries = []
  const touchedDirtyLocalIds = new Set()
  const touchedTombstones = []

  const dirtyLocalIds = sortDirtyLocalIds(docStore, docStore.getDirtyLocalIds())
  for (const localId of dirtyLocalIds) {
    if (entries.length >= maxOps) {
      break
    }

    const entity = docStore.getEntity(localId)
    if (!entity) {
      continue
    }

    const cold = docStore.getCold(localId)
    const segmentId = toNormalizedSegmentId(cold?.segmentId)
    const ackSnapshot = resolveAckSnapshot(
      localId,
      segmentId,
      ackShadowByLocalId,
      ackLocalIdBySegmentId
    )

    // 无绑定且无 shadow，按新增字幕处理
    if (!segmentId && !ackSnapshot) {
      const neighbors = docStore.getNeighbors(localId)
      entries.push({
        kind: 'dirty',
        localId,
        segmentId: null,
        op: {
          op_id: sessionStore.nextCommandId(),
          type: 'insert_subtitle',
          client_ref_id: localId,
          anchor: {
            before_client_ref_id: neighbors.prev ?? null,
            after_client_ref_id: neighbors.next ?? null,
          },
          after: {
            text: String(entity.text ?? ''),
            start_ms: toSafeMs(entity.startMs),
            end_ms: toSafeMs(entity.endMs),
          },
        },
      })
      touchedDirtyLocalIds.add(localId)
      continue
    }

    const effectiveSegmentId = segmentId ?? toNormalizedSegmentId(ackSnapshot?.segmentId)
    const isTextChanged = !ackSnapshot || String(entity.text ?? '') !== String(ackSnapshot.text ?? '')
    const isTimingChanged = !ackSnapshot
      || toSafeMs(entity.startMs) !== toSafeMs(ackSnapshot.startMs)
      || toSafeMs(entity.endMs) !== toSafeMs(ackSnapshot.endMs)

    if (isTextChanged && entries.length < maxOps) {
      entries.push({
        kind: 'dirty',
        localId,
        segmentId: effectiveSegmentId,
        op: {
          op_id: sessionStore.nextCommandId(),
          type: 'update_text',
          client_ref_id: localId,
          segment_id: effectiveSegmentId,
          before: {
            text: String(ackSnapshot?.text ?? ''),
          },
          after: {
            text: String(entity.text ?? ''),
          },
        },
      })
    }

    if (isTimingChanged && entries.length < maxOps) {
      entries.push({
        kind: 'dirty',
        localId,
        segmentId: effectiveSegmentId,
        op: {
          op_id: sessionStore.nextCommandId(),
          type: 'update_timing',
          client_ref_id: localId,
          segment_id: effectiveSegmentId,
          before: {
            start_ms: toSafeMs(ackSnapshot?.startMs),
            end_ms: toSafeMs(ackSnapshot?.endMs),
          },
          after: {
            start_ms: toSafeMs(entity.startMs),
            end_ms: toSafeMs(entity.endMs),
          },
        },
      })
    }

    touchedDirtyLocalIds.add(localId)
  }

  if (entries.length < maxOps) {
    const tombstones = [...docStore.getTombstones()].sort((left, right) => {
      return Number(left?.deletedAt ?? 0) - Number(right?.deletedAt ?? 0)
    })

    for (const tombstone of tombstones) {
      if (entries.length >= maxOps) {
        break
      }

      const segmentId = toNormalizedSegmentId(tombstone?.segmentId)
      if (!segmentId) {
        // 本地无 binding 的删除无法映射到后端实体，直接消费墓碑。
        touchedTombstones.push(tombstone)
        continue
      }

      const ackLocalId = ackLocalIdBySegmentId.get(segmentId) ?? null
      const ackSnapshot = ackLocalId ? ackShadowByLocalId.get(ackLocalId) : null
      const before = tombstone?.before ?? {
        text: String(ackSnapshot?.text ?? ''),
        startMs: toSafeMs(ackSnapshot?.startMs),
        endMs: toSafeMs(ackSnapshot?.endMs),
      }
      entries.push({
        kind: 'tombstone',
        localId: tombstone.localId ?? ackLocalId,
        segmentId,
        tombstone,
        op: {
          op_id: sessionStore.nextCommandId(),
          type: 'delete_subtitle',
          client_ref_id: tombstone.localId ?? ackLocalId,
          segment_id: segmentId,
          before: {
            text: String(before.text ?? ''),
            start_ms: toSafeMs(before.startMs),
            end_ms: toSafeMs(before.endMs),
          },
        },
      })
      touchedTombstones.push(tombstone)
    }
  }

  return {
    entries,
    touchedDirtyLocalIds: [...touchedDirtyLocalIds],
    touchedTombstones,
  }
}
