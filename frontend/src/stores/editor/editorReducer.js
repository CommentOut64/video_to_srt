// V3.2.5+dev.20260314.01: 命令 Reducer - 统一执行所有编辑操作
import { createSubtitleHotEntity, createSubtitleColdEntity } from './models'

/**
 * 纯函数风格的 reducer
 * 输入：documentStore 引用 + command
 * 输出：执行结果 + 反向命令
 */
export function editorReducer(docStore, command) {
  switch (command.type) {
    case 'update_text': {
      const entity = docStore.getEntity(command.localId)
      if (!entity) return { success: false, reason: 'entity_not_found' }

      const undoCommand = {
        ...command,
        commandId: command.commandId + '_undo',
        source: 'undo_redo',
        before: command.after,
        after: command.before,
      }

      docStore._applyUpdate(command.localId, {
        text: command.after.text,
        isModified: true,
      })

      return { success: true, undoCommand }
    }

    case 'update_timing': {
      const entity = docStore.getEntity(command.localId)
      if (!entity) return { success: false, reason: 'entity_not_found' }

      const undoCommand = {
        ...command,
        commandId: command.commandId + '_undo',
        source: 'undo_redo',
        before: command.after,
        after: command.before,
      }

      docStore._applyUpdate(command.localId, {
        startMs: command.after.startMs,
        endMs: command.after.endMs,
        isModified: true,
      })
      docStore._applyReorder(command.localId)

      return { success: true, undoCommand }
    }

    case 'insert_subtitle': {
      const hot = createSubtitleHotEntity(
        command.localId,
        command.entity.text,
        command.entity.startMs,
        command.entity.endMs
      )
      hot.isModified = true
      if (command.entity.isDraft !== undefined) {
        hot.isDraft = command.entity.isDraft
      }

      const cold = command.coldInit
        ? { ...createSubtitleColdEntity(command.localId), ...command.coldInit }
        : createSubtitleColdEntity(command.localId)

      docStore._applyInsert(command.localId, hot, cold, command.afterLocalId)

      const undoCommand = {
        type: 'delete_subtitle',
        commandId: command.commandId + '_undo',
        source: 'undo_redo',
        createdAt: Date.now(),
        localId: command.localId,
        snapshot: {
          text: hot.text,
          startMs: hot.startMs,
          endMs: hot.endMs,
          orderIndex: docStore.getOrderIndex(command.localId),
        },
        coldSnapshot: cold,
      }

      return { success: true, undoCommand }
    }

    case 'delete_subtitle': {
      const entity = docStore.getEntity(command.localId)
      if (!entity) return { success: false, reason: 'entity_not_found' }

      const cold = docStore.getCold(command.localId)
      const orderIndex = docStore.getOrderIndex(command.localId)

      const undoCommand = {
        type: 'insert_subtitle',
        commandId: command.commandId + '_undo',
        source: 'undo_redo',
        createdAt: Date.now(),
        localId: command.localId,
        entity: { text: entity.text, startMs: entity.startMs, endMs: entity.endMs },
        afterLocalId: orderIndex > 0 ? docStore.order[orderIndex - 1] : null,
        coldInit: cold || undefined,
      }

      docStore._applyDelete(command.localId)
      return { success: true, undoCommand }
    }

    case 'split_subtitle': {
      const entity = docStore.getEntity(command.sourceLocalId)
      if (!entity) return { success: false, reason: 'entity_not_found' }

      docStore._applyUpdate(command.sourceLocalId, {
        text: command.afterKept.text,
        startMs: command.afterKept.startMs,
        endMs: command.afterKept.endMs,
        isModified: true,
      })
      docStore._applyReorder(command.sourceLocalId)

      const newHot = createSubtitleHotEntity(
        command.createdLocalId,
        command.afterCreated.text,
        command.afterCreated.startMs,
        command.afterCreated.endMs
      )
      newHot.isModified = true

      docStore._applyInsert(
        command.createdLocalId,
        newHot,
        createSubtitleColdEntity(command.createdLocalId),
        command.sourceLocalId
      )

      const undoCommand = {
        type: 'merge_subtitles',
        commandId: command.commandId + '_undo',
        source: 'undo_redo',
        createdAt: Date.now(),
        keptLocalId: command.sourceLocalId,
        removedLocalId: command.createdLocalId,
        separator: '',
        beforeKept: command.afterKept,
        beforeRemoved: command.afterCreated,
        after: command.before,
        removedColdSnapshot: null,
      }

      return { success: true, undoCommand }
    }

    case 'merge_subtitles': {
      const kept = docStore.getEntity(command.keptLocalId)
      const removed = docStore.getEntity(command.removedLocalId)
      if (!kept || !removed) return { success: false, reason: 'entity_not_found' }

      docStore._applyUpdate(command.keptLocalId, {
        text: command.after.text,
        startMs: command.after.startMs,
        endMs: command.after.endMs,
        isModified: true,
      })
      docStore._applyReorder(command.keptLocalId)
      docStore._applyDelete(command.removedLocalId)

      const undoCommand = {
        type: 'split_subtitle',
        commandId: command.commandId + '_undo',
        source: 'undo_redo',
        createdAt: Date.now(),
        sourceLocalId: command.keptLocalId,
        createdLocalId: command.removedLocalId,
        splitAtMs: command.beforeRemoved.startMs,
        splitAtTextOffset: command.beforeKept.text.length,
        before: command.after,
        afterKept: command.beforeKept,
        afterCreated: command.beforeRemoved,
      }

      return { success: true, undoCommand }
    }

    case 'move_boundary': {
      const upper = docStore.getEntity(command.upperLocalId)
      const lower = docStore.getEntity(command.lowerLocalId)
      if (!upper || !lower) return { success: false, reason: 'entity_not_found' }

      const undoCommand = {
        ...command,
        commandId: command.commandId + '_undo',
        source: 'undo_redo',
        before: command.after,
        after: command.before,
      }

      docStore._applyUpdate(command.upperLocalId, { endMs: command.after.upperEndMs, isModified: true })
      docStore._applyUpdate(command.lowerLocalId, { startMs: command.after.lowerStartMs, isModified: true })

      return { success: true, undoCommand }
    }

    case 'batch_replace': {
      const undoReplacements = []
      for (const r of command.replacements) {
        const entity = docStore.getEntity(r.localId)
        if (!entity) continue
        undoReplacements.push({
          localId: r.localId,
          before: r.after,
          after: r.before,
        })
        docStore._applyUpdate(r.localId, { text: r.after.text, isModified: true })
      }

      const undoCommand = {
        type: 'batch_replace',
        commandId: command.commandId + '_undo',
        source: 'undo_redo',
        createdAt: Date.now(),
        replacements: undoReplacements,
      }

      return { success: true, undoCommand }
    }

    case 'bind_segment_id': {
      for (const b of command.bindings) {
        docStore._applyBinding(b.localId, b.segmentId)
      }
      return { success: true, undoCommand: null }
    }

    case 'apply_server_replace': {
      docStore._applyServerReplace(command.oldLocalIds, command.newEntities)
      return { success: true, undoCommand: null }
    }

    case 'finalize_draft_chunk': {
      for (const u of command.updates) {
        docStore._applyUpdate(u.localId, {
          text: u.text,
          startMs: u.startMs,
          endMs: u.endMs,
          isDraft: false,
        })
        docStore._applyReorder(u.localId)
        if (u.cold) {
          docStore._applyColdUpdate(u.localId, u.cold)
        }
      }
      return { success: true, undoCommand: null }
    }

    case 'finalize_draft_chunk': {
      for (const localId of command.localIds) {
        docStore._applyUpdate(localId, { isDraft: false })
      }
      return { success: true, undoCommand: null }
    }

    case 'apply_server_replace': {
      docStore._applyServerReplace(command.oldLocalIds, command.newEntities)
      return { success: true, undoCommand: null }
    }

    default:
      return { success: false, reason: 'unknown_command_type' }
  }
}
