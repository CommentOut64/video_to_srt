// V3.2.5+dev.20260314.01: 命令总线 - 唯一写入口
import { defineStore } from 'pinia'
import { useEditorDocumentStore } from './editorDocumentStore'
import { useEditorSessionStore } from './editorSessionStore'
import { useEditorHistoryStore } from './editorHistoryStore'
import { normalizeEditorCommand } from './editorCommandFactory'
import { editorReducer } from './editorReducer'

export const useEditorCommandBus = defineStore('editorCommandBus', () => {
  const listeners = []

  function normalizeSegmentId(value) {
    const normalized = String(value ?? '').trim()
    return normalized || null
  }

  function collectAffectedLocalIds(command) {
    if (!command) return []

    switch (command.type) {
      case 'update_text':
      case 'update_timing':
      case 'insert_subtitle':
      case 'delete_subtitle':
        return command.localId ? [command.localId] : []
      case 'split_subtitle':
        return [command.sourceLocalId, command.createdLocalId].filter(Boolean)
      case 'merge_subtitles':
        return [command.keptLocalId, command.removedLocalId].filter(Boolean)
      case 'move_boundary':
        return [command.upperLocalId, command.lowerLocalId].filter(Boolean)
      case 'batch_replace':
        return Array.isArray(command.replacements)
          ? command.replacements.map((item) => item.localId).filter(Boolean)
          : []
      case 'bind_segment_id':
        return Array.isArray(command.bindings)
          ? command.bindings.map((item) => item.localId).filter(Boolean)
          : []
      case 'apply_server_replace': {
        const oldLocalIds = Array.isArray(command.oldLocalIds) ? command.oldLocalIds : []
        const newLocalIds = Array.isArray(command.newEntities)
          ? command.newEntities.map((item) => item.localId).filter(Boolean)
          : []
        return [...oldLocalIds, ...newLocalIds].filter(Boolean)
      }
      case 'finalize_draft_chunk': {
        const updateLocalIds = Array.isArray(command.updates)
          ? command.updates.map((item) => item.localId).filter(Boolean)
          : []
        const directLocalIds = Array.isArray(command.localIds) ? command.localIds : []
        return [...updateLocalIds, ...directLocalIds].filter(Boolean)
      }
      default:
        return []
    }
  }

  function isSyncEligibleCommand(command) {
    if (!command || command.skipSync === true) {
      return false
    }

    return !['system', 'rehydrate', 'reconcile_replay'].includes(command.source)
  }

  function notifyListeners(command, result) {
    listeners.forEach((fn) => fn(command, result))
  }

  function canReplayCommand(command, docStore) {
    if (!command || !command.type) {
      return false
    }

    switch (command.type) {
      case 'update_text':
      case 'update_timing':
      case 'delete_subtitle':
        return Boolean(command.localId && docStore.getEntity(command.localId))
      case 'split_subtitle':
        return Boolean(command.sourceLocalId && docStore.getEntity(command.sourceLocalId))
      case 'merge_subtitles':
        return Boolean(
          command.keptLocalId
          && command.removedLocalId
          && docStore.getEntity(command.keptLocalId)
          && docStore.getEntity(command.removedLocalId)
        )
      case 'move_boundary':
        return Boolean(
          command.upperLocalId
          && command.lowerLocalId
          && docStore.getEntity(command.upperLocalId)
          && docStore.getEntity(command.lowerLocalId)
        )
      default:
        return true
    }
  }

  function enrichHistoryReplayCommand(command, docStore) {
    if (!command) {
      return command
    }

    const resolveCurrentSegmentId = (localId, fallbackSegmentId = null) => {
      const currentSegmentId = localId ? docStore.getCold(localId)?.segmentId ?? null : null
      return currentSegmentId ?? fallbackSegmentId ?? null
    }

    const shouldReuseSplitCreatedBinding = (splitCommand, candidateSegmentId) => {
      const normalizedSegmentId = normalizeSegmentId(candidateSegmentId)
      if (!normalizedSegmentId) {
        return false
      }

      const boundLocalId = docStore.bindingBySegmentId.get(normalizedSegmentId)
      if (boundLocalId && boundLocalId !== splitCommand.createdLocalId) {
        return false
      }

      const hasPendingDelete = docStore.getTombstones().some((tombstone) => (
        tombstone?.localId === splitCommand.createdLocalId
        && normalizeSegmentId(tombstone?.segmentId) === normalizedSegmentId
      ))
      if (hasPendingDelete) {
        return true
      }

      return boundLocalId === splitCommand.createdLocalId
    }

    const resolveSplitCreatedFallbackSegmentId = (splitCommand, createdColdInit) => {
      const directSegmentId = createdColdInit?.segmentId ?? null
      if (normalizeSegmentId(directSegmentId)) {
        return directSegmentId
      }

      const tombstoneSegmentId = docStore.getTombstones()
        .find((tombstone) => tombstone?.localId === splitCommand.createdLocalId)
        ?.segmentId ?? null
      return normalizeSegmentId(tombstoneSegmentId) || null
    }

    if (command.type === 'split_subtitle') {
      const createdColdInit = command.createdColdInit
      if (!createdColdInit || typeof createdColdInit !== 'object') {
        return {
          ...command,
          sourceSegmentId: resolveCurrentSegmentId(command.sourceLocalId, command.sourceSegmentId),
        }
      }

      const fallbackCreatedSegmentId = resolveSplitCreatedFallbackSegmentId(command, createdColdInit)
      const liveCreatedSegmentId = resolveCurrentSegmentId(command.createdLocalId, fallbackCreatedSegmentId)
      const shouldKeepCreatedBinding = shouldReuseSplitCreatedBinding(command, liveCreatedSegmentId)
      const nextCreatedColdInit = shouldKeepCreatedBinding
        ? {
            ...createdColdInit,
            segmentId: liveCreatedSegmentId,
          }
        : {
            ...createdColdInit,
            segmentId: null,
            sentenceIndex: null,
          }

      return {
        ...command,
        sourceSegmentId: resolveCurrentSegmentId(command.sourceLocalId, command.sourceSegmentId),
        createdColdInit: nextCreatedColdInit,
      }
    }

    if (command.type === 'merge_subtitles') {
      return {
        ...command,
        keptSegmentId: resolveCurrentSegmentId(command.keptLocalId, command.keptSegmentId),
        removedSegmentId: resolveCurrentSegmentId(command.removedLocalId, command.removedSegmentId),
      }
    }

    if (command.type === 'delete_subtitle') {
      return {
        ...command,
        segmentId: resolveCurrentSegmentId(command.localId, command.segmentId),
      }
    }

    if (command.type === 'move_boundary') {
      return {
        ...command,
        upperSegmentId: resolveCurrentSegmentId(command.upperLocalId, command.upperSegmentId),
        lowerSegmentId: resolveCurrentSegmentId(command.lowerLocalId, command.lowerSegmentId),
      }
    }

    return command
  }

  function materializeHistoryReplayCommands(commands = []) {
    const sessionStore = useEditorSessionStore()
    const docStore = useEditorDocumentStore()
    const replayCreatedAt = Date.now()

    return commands
      .filter(Boolean)
      .map((command, index) => {
        const enrichedCommand = enrichHistoryReplayCommand(command, docStore)
        const { skipSync, ...rest } = enrichedCommand
        return {
          ...rest,
          commandId: sessionStore.nextCommandId(),
          source: 'undo_redo',
          createdAt: replayCreatedAt + index,
        }
      })
  }

  function dispatch(rawCommand) {
    const docStore = useEditorDocumentStore()
    const sessionStore = useEditorSessionStore()
    const historyStore = useEditorHistoryStore()
    const command = normalizeEditorCommand(rawCommand)

    if (!command?.type) {
      console.warn('[CommandBus] 收到无效命令，已忽略:', rawCommand)
      return { success: false, reason: 'invalid_command' }
    }

    // 执行 reducer
    const result = editorReducer(docStore, command)
    if (!result.success) {
      console.warn(`[CommandBus] Command ${command.type} failed:`, result.reason)
      return result
    }

    // 历史记录（仅 user 来源）
    if (command.source === 'user') {
      historyStore.push(command, result.undoCommand)
    }

    // 仅用户侧编辑和撤销重做标记脏会话
    const shouldMarkDirty = ['user', 'undo_redo'].includes(command.source)
    if (shouldMarkDirty) {
      sessionStore.markDirty()
    }

    if (!isSyncEligibleCommand(command)) {
      const affectedLocalIds = collectAffectedLocalIds(command)
      docStore.clearDirtyFlags(affectedLocalIds)
      docStore.clearTombstones({ localIds: affectedLocalIds })
    }

    notifyListeners(command, result)

    return result
  }

  function dispatchTransaction(rawCommands, options = {}) {
    const commands = Array.isArray(rawCommands)
      ? rawCommands.map(normalizeEditorCommand).filter((command) => command?.type)
      : []
    if (commands.length === 0) {
      return { success: false, reason: 'empty_transaction', results: [] }
    }

    const docStore = useEditorDocumentStore()
    const sessionStore = useEditorSessionStore()
    const historyStore = useEditorHistoryStore()
    const revisionBefore = docStore.revision
    const executedCommands = []
    const undoCommands = []
    const results = []
    let shouldMarkDirty = false

    for (const command of commands) {
      const result = editorReducer(docStore, command)
      if (!result.success) {
        console.warn(`[CommandBus] Transaction command ${command.type} failed:`, result.reason)
        return {
          success: false,
          reason: result.reason || 'transaction_command_failed',
          failedCommand: command,
          results,
        }
      }

      executedCommands.push(command)
      if (result.undoCommand) {
        undoCommands.unshift(result.undoCommand)
      }
      results.push(result)

      if (['user', 'undo_redo'].includes(command.source)) {
        shouldMarkDirty = true
      }

      if (!isSyncEligibleCommand(command)) {
        const affectedLocalIds = collectAffectedLocalIds(command)
        docStore.clearDirtyFlags(affectedLocalIds)
        docStore.clearTombstones({ localIds: affectedLocalIds })
      }

      notifyListeners(command, result)
    }

    if (executedCommands.some((command) => command.source === 'user')) {
      historyStore.pushTransaction({
        type: options.historyType || executedCommands[0]?.type || 'transaction',
        doCommands: executedCommands,
        undoCommands,
        revisionBefore,
        revisionAfter: docStore.revision,
      })
    }

    if (shouldMarkDirty) {
      sessionStore.markDirty()
    }

    return { success: true, results }
  }

  function undo() {
    const historyStore = useEditorHistoryStore()
    const docStore = useEditorDocumentStore()
    const candidateEntry = historyStore.peekUndo()
    if (!candidateEntry) return false
    const replayCommands = materializeHistoryReplayCommands(candidateEntry.undoCommands)
    if (!replayCommands.every((command) => canReplayCommand(command, docStore))) {
      console.warn('[CommandBus] undo 预检失败，已拒绝回放，历史栈保持不变')
      return false
    }

    if (!historyStore.commitUndo()) return false
    historyStore.closeTransaction()
    let hasFailure = false
    replayCommands.forEach((cmd) => {
      const result = dispatch(cmd)
      if (!result?.success) {
        hasFailure = true
      }
    })
    if (hasFailure) {
      // Trade-off: 回放失败后无法回滚已触发的外部副作用（监听器/持久化），
      // 这里至少回滚历史游标，避免 past/future 与文档状态持续漂移。
      historyStore.commitRedo()
      console.warn('[CommandBus] undo 回放部分失败，已回滚历史游标')
      return false
    }
    return true
  }

  function redo() {
    const historyStore = useEditorHistoryStore()
    const docStore = useEditorDocumentStore()
    const candidateEntry = historyStore.peekRedo()
    if (!candidateEntry) return false
    const replayCommands = materializeHistoryReplayCommands(candidateEntry.doCommands)
    if (!replayCommands.every((command) => canReplayCommand(command, docStore))) {
      console.warn('[CommandBus] redo 预检失败，已拒绝回放，历史栈保持不变')
      return false
    }

    if (!historyStore.commitRedo()) return false
    historyStore.closeTransaction()
    let hasFailure = false
    replayCommands.forEach((cmd) => {
      const result = dispatch(cmd)
      if (!result?.success) {
        hasFailure = true
      }
    })
    if (hasFailure) {
      historyStore.commitUndo()
      console.warn('[CommandBus] redo 回放部分失败，已回滚历史游标')
      return false
    }
    return true
  }

  function dispatchBatch(commands) {
    const results = commands.map(cmd => dispatch(cmd))
    return results
  }

  function onCommandApplied(callback) {
    listeners.push(callback)
    return () => {
      const idx = listeners.indexOf(callback)
      if (idx > -1) listeners.splice(idx, 1)
    }
  }

  return { dispatch, dispatchBatch, dispatchTransaction, undo, redo, onCommandApplied }
})
