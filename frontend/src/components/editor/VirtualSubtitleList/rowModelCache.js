export function createRowModelCache(options = {}) {
  const buildRow = options.buildRow ?? ((entry) => entry.payload)
  const rowById = new Map()
  const signatureById = new Map()

  function rebuildVisibleRows(entries = []) {
    return entries.map((entry) => {
      const { localId, signature } = entry
      const shouldReuse = (
        localId
        && signatureById.get(localId) === signature
        && rowById.has(localId)
      )

      if (shouldReuse) {
        return rowById.get(localId)
      }

      const nextRow = buildRow(entry)
      rowById.set(localId, nextRow)
      signatureById.set(localId, signature)
      return nextRow
    })
  }

  return {
    rebuildVisibleRows,
  }
}
