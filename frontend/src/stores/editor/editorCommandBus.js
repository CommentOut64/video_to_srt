// V3.2.5+dev.20260314.01: 命令总线 - 唯一写入口
import { defineStore } from 'pinia'
import { useEditorDocumentStore } from './editorDocumentStore'
import { useEditorSessionStore } from './editorSessionStore'
import { useEditorHistoryStore } from './editorHistoryStore'
import { normalizeEditorCommand } from './editorCommandFactory'
import { editorReducer } from './editorReducer'

export const useEditorCommandBus = defineStore('editorCommandBus', () => {
  const listeners = []

  function materializeHistoryReplayCommands(commands = []) {
    const sessionStore = useEditorSessionStore()
    const replayCreatedAt = Date.now()

    return commands
      .filter(Boolean)
      .map((command, index) => {
        const { skipSync, ...rest } = command
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

    // 通知订阅者
    listeners.forEach(fn => fn(command, result))

    return result
  }

  function undo() {
    const historyStore = useEditorHistoryStore()
    const commands = historyStore.undo()
    if (!commands) return false
    historyStore.closeTransaction()
    materializeHistoryReplayCommands(commands).forEach(cmd => dispatch(cmd))
    return true
  }

  function redo() {
    const historyStore = useEditorHistoryStore()
    const commands = historyStore.redo()
    if (!commands) return false
    historyStore.closeTransaction()
    materializeHistoryReplayCommands(commands).forEach(cmd => dispatch(cmd))
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

  return { dispatch, dispatchBatch, undo, redo, onCommandApplied }
})
