function buildDefaultRegionModel(entry) {
  return entry.payload
}

export function createRegionModelCache(options = {}) {
  const buildModel = options.buildModel ?? buildDefaultRegionModel
  const modelById = new Map()
  const signatureById = new Map()

  function rebuildVisibleModels(entries = []) {
    return entries.map((entry) => {
      const { localId, signature } = entry
      const shouldReuse = (
        localId
        && signatureById.get(localId) === signature
        && modelById.has(localId)
      )

      if (shouldReuse) {
        return modelById.get(localId)
      }

      const nextModel = buildModel(entry)
      modelById.set(localId, nextModel)
      signatureById.set(localId, signature)
      return nextModel
    })
  }

  function clear() {
    modelById.clear()
    signatureById.clear()
  }

  return {
    rebuildVisibleModels,
    clear,
  }
}

export function createRegionModelEntry(entity, options = {}) {
  const selected = options.selected === true
  const overlapping = options.overlapping === true
  const dragging = options.dragging === true
  const color = options.color || null
  const localId = String(entity?.localId ?? '')
  const startMs = Number(entity?.startMs ?? 0)
  const endMs = Number(entity?.endMs ?? startMs)
  const text = String(entity?.text ?? '')

  return {
    localId,
    signature: [
      localId,
      startMs,
      endMs,
      selected ? 'selected' : 'idle',
      overlapping ? 'overlap' : 'normal',
      dragging ? 'dragging' : 'static',
      color ?? '',
      text,
    ].join('|'),
    payload: {
      localId,
      startMs,
      endMs,
      text,
      color,
      selected,
      overlapping,
      dragging,
    },
  }
}
