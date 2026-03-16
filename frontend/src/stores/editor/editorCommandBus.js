// V3.2.5+dev.20260314.01: 命令总线 - 唯一写入口
import { defineStore } from 'pinia'
import { useEditorDocumentStore } from './editorDocumentStore'
import { useEditorSessionStore } from './editorSessionStore'
import { useEditorHistoryStore } from './editorHistoryStore'
import { useEditorSyncEngine } from './editorSyncEngine'
import { normalizeEditorCommand } from './editorCommandFactory'
import { editorReducer } from './editorReducer'

export const useEditorCommandBus = defineStore('editorCommandBus', () => {
  const listeners = []

  function notifyListeners(command, result) {
    listeners.forEach((fn) => fn(command, result))
  }

  function enrichHistoryReplayCommand(command, docStore) {
    if (!command) {
      return command
    }

    const resolveCurrentSegmentId = (localId, fallbackSegmentId = null) => {
      const currentSegmentId = localId ? docStore.getCold(localId)?.segmentId ?? null : null
      return currentSegmentId ?? fallbackSegmentId ?? null
    }

    if (command.type === 'split_subtitle') {
      return {
        ...command,
        sourceSegmentId: resolveCurrentSegmentId(command.sourceLocalId, command.sourceSegmentId),
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

    // 根据 source 决定是否标记 dirty
    const shouldMarkDirty = !['rehydrate', 'reconcile_replay'].includes(command.source)
    if (shouldMarkDirty) {
      sessionStore.markDirty()
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

      if (!['rehydrate', 'reconcile_replay'].includes(command.source)) {
        shouldMarkDirty = true
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
    const syncEngine = useEditorSyncEngine()
    const entry = historyStore.undo()
    if (!entry) return false
    historyStore.closeTransaction()
    const replayCommands = materializeHistoryReplayCommands(entry.undoCommands)
    replayCommands.forEach(cmd => dispatch(cmd))
    syncEngine.rewritePendingForHistory({
      entry,
      replayCommands,
      direction: 'undo',
    })
    return true
  }

  function redo() {
    const historyStore = useEditorHistoryStore()
    const syncEngine = useEditorSyncEngine()
    const entry = historyStore.redo()
    if (!entry) return false
    historyStore.closeTransaction()
    const replayCommands = materializeHistoryReplayCommands(entry.doCommands)
    replayCommands.forEach(cmd => dispatch(cmd))
    syncEngine.rewritePendingForHistory({
      entry,
      replayCommands,
      direction: 'redo',
    })
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
