export function computeSubtitleSnapshotDiff(beforeSnapshot, afterSnapshot) {
  const updates = []
  const creates = []
  const deletes = []

  for (const [key, beforeSubtitle] of beforeSnapshot) {
    if (!afterSnapshot.has(key) && beforeSubtitle.segment_id) {
      deletes.push({ segment_id: beforeSubtitle.segment_id })
    }
  }

  for (const [key, afterSubtitle] of afterSnapshot) {
    const beforeSubtitle = beforeSnapshot.get(key)
    if (!beforeSubtitle) {
      const created = {
        local_id: afterSubtitle.localId ?? null,
        text: afterSubtitle.text,
        start: afterSubtitle.start,
        end: afterSubtitle.end,
      }
      if (afterSubtitle.segment_id) {
        created.restore_segment_id = afterSubtitle.segment_id
      }
      creates.push(created)
      continue
    }

    if (!afterSubtitle.segment_id) {
      continue
    }

    const diffPayload = {}
    if (afterSubtitle.text !== beforeSubtitle.text) diffPayload.text = afterSubtitle.text
    if (Math.abs(afterSubtitle.start - beforeSubtitle.start) > 0.001) diffPayload.start = afterSubtitle.start
    if (Math.abs(afterSubtitle.end - beforeSubtitle.end) > 0.001) diffPayload.end = afterSubtitle.end
    if (Object.keys(diffPayload).length > 0) {
      updates.push({ segment_id: afterSubtitle.segment_id, ...diffPayload })
    }
  }

  return { updates, creates, deletes }
}

export function buildEditorCommandsFromDiff(diff, nextCommandId, commandPrefix = 'sync') {
  const commands = []

  diff.updates.forEach((item, index) => {
    commands.push({
      type: 'update_subtitle',
      command_id: nextCommandId(`${commandPrefix}-update-${index}`),
      segment_id: item.segment_id,
      text: item.text,
      start: item.start,
      end: item.end,
    })
  })

  diff.deletes.forEach((item, index) => {
    commands.push({
      type: 'remove_subtitle',
      command_id: nextCommandId(`${commandPrefix}-delete-${index}`),
      segment_id: item.segment_id,
    })
  })

  diff.creates.forEach((item, index) => {
    commands.push({
      type: 'add_subtitle',
      command_id: nextCommandId(`${commandPrefix}-create-${index}`),
      local_id: item.local_id ?? null,
      text: item.text,
      start: item.start,
      end: item.end,
      restore_segment_id: item.restore_segment_id,
    })
  })

  return commands
}