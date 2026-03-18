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
  const HISTORY_NOOP_FLAG = '__historyNoop'
  const SNAPSHOT_STRICT_TIME_TOLERANCE_MS = 1
  const SNAPSHOT_FUZZY_TIME_TOLERANCE_MS = 80

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
    if (command[HISTORY_NOOP_FLAG] === true) {
      return true
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

    const markHistoryNoop = (nextCommand, reason) => ({
      ...nextCommand,
      [HISTORY_NOOP_FLAG]: true,
      __historyNoopReason: reason,
    })

    const isSnapshotEntityMatch = (entity, snapshot, options = {}) => {
      if (!entity || !snapshot || typeof snapshot !== 'object') {
        return false
      }
      const {
        timeToleranceMs = SNAPSHOT_STRICT_TIME_TOLERANCE_MS,
        ignoreTime = false,
      } = options
      const hasText = Object.prototype.hasOwnProperty.call(snapshot, 'text')
      const hasStartMs = Object.prototype.hasOwnProperty.call(snapshot, 'startMs')
      const hasEndMs = Object.prototype.hasOwnProperty.call(snapshot, 'endMs')
      if (!hasText && !hasStartMs && !hasEndMs) {
        return false
      }
      if (hasText && String(entity.text ?? '') !== String(snapshot.text ?? '')) {
        return false
      }
      if (!ignoreTime && hasStartMs) {
        const delta = Math.abs(Number(entity.startMs ?? 0) - Number(snapshot.startMs ?? 0))
        if (delta > timeToleranceMs) {
          return false
        }
      }
      if (!ignoreTime && hasEndMs) {
        const delta = Math.abs(Number(entity.endMs ?? 0) - Number(snapshot.endMs ?? 0))
        if (delta > timeToleranceMs) {
          return false
        }
      }
      return true
    }

    const findSnapshotCandidates = (snapshot, options = {}) => {
      if (!snapshot || typeof snapshot !== 'object') {
        return []
      }
      const excludeLocalIdSet = new Set(
        Array.isArray(options.excludeLocalIds) ? options.excludeLocalIds.filter(Boolean) : []
      )
      return docStore.order.filter((candidateLocalId) => {
        if (excludeLocalIdSet.has(candidateLocalId)) {
          return false
        }
        return isSnapshotEntityMatch(docStore.getEntity(candidateLocalId), snapshot, options)
      })
    }

    const resolveLocalIdBySnapshot = (snapshot) => {
      const strictCandidates = findSnapshotCandidates(snapshot, {
        timeToleranceMs: SNAPSHOT_STRICT_TIME_TOLERANCE_MS,
      })
      if (strictCandidates.length === 1) {
        return strictCandidates[0]
      }
      if (strictCandidates.length > 1) {
        return null
      }

      const fuzzyCandidates = findSnapshotCandidates(snapshot, {
        timeToleranceMs: SNAPSHOT_FUZZY_TIME_TOLERANCE_MS,
      })
      if (fuzzyCandidates.length === 1) {
        return fuzzyCandidates[0]
      }
      if (fuzzyCandidates.length > 1) {
        return null
      }

      const hasText = snapshot && typeof snapshot === 'object'
        && Object.prototype.hasOwnProperty.call(snapshot, 'text')
      if (!hasText) {
        return null
      }

      const textOnlyCandidates = findSnapshotCandidates(snapshot, {
        ignoreTime: true,
      })
      if (textOnlyCandidates.length === 1) {
        return textOnlyCandidates[0]
      }

      return null
    }

    const hasSnapshotCandidate = (snapshot, options = {}) => {
      const strictCandidates = findSnapshotCandidates(snapshot, {
        ...options,
        timeToleranceMs: SNAPSHOT_STRICT_TIME_TOLERANCE_MS,
      })
      if (strictCandidates.length > 0) {
        return true
      }

      const fuzzyCandidates = findSnapshotCandidates(snapshot, {
        ...options,
        timeToleranceMs: SNAPSHOT_FUZZY_TIME_TOLERANCE_MS,
      })
      if (fuzzyCandidates.length > 0) {
        return true
      }

      const hasText = snapshot && typeof snapshot === 'object'
        && Object.prototype.hasOwnProperty.call(snapshot, 'text')
      if (!hasText) {
        return false
      }

      return findSnapshotCandidates(snapshot, {
        ...options,
        ignoreTime: true,
      }).length > 0
    }

    const isLocalIdMatchingSnapshot = (localId, snapshot, options = {}) => {
      if (!localId) {
        return false
      }
      const entity = docStore.getEntity(localId)
      if (!entity) {
        return false
      }
      const strictMatch = isSnapshotEntityMatch(entity, snapshot, {
        ...options,
        timeToleranceMs: SNAPSHOT_STRICT_TIME_TOLERANCE_MS,
      })
      if (strictMatch) {
        return true
      }
      const fuzzyMatch = isSnapshotEntityMatch(entity, snapshot, {
        ...options,
        timeToleranceMs: SNAPSHOT_FUZZY_TIME_TOLERANCE_MS,
      })
      if (fuzzyMatch) {
        return true
      }
      return isSnapshotEntityMatch(entity, snapshot, {
        ...options,
        ignoreTime: true,
      })
    }

    const resolveReplayLocalId = (localId, segmentId = null, snapshots = []) => {
      if (localId && docStore.getEntity(localId)) {
        return localId
      }

      const normalizedSegmentId = normalizeSegmentId(segmentId)
      const normalizedLocalId = normalizeSegmentId(localId)
      if (!normalizedSegmentId) {
        if (normalizedLocalId) {
          const boundLocalId = docStore.bindingBySegmentId.get(normalizedLocalId)
          if (boundLocalId && docStore.getEntity(boundLocalId)) {
            return boundLocalId
          }
        }

        for (const snapshot of snapshots) {
          const localIdFromSnapshot = resolveLocalIdBySnapshot(snapshot)
          if (localIdFromSnapshot) {
            return localIdFromSnapshot
          }
        }

        return localId ?? null
      }

      const boundLocalId = docStore.bindingBySegmentId.get(normalizedSegmentId)
      if (boundLocalId && docStore.getEntity(boundLocalId)) {
        return boundLocalId
      }

      if (docStore.getEntity(normalizedSegmentId)) {
        return normalizedSegmentId
      }

      return localId ?? null
    }

    const resolveCurrentSegmentId = (localId, fallbackSegmentId = null) => {
      const currentSegmentId = localId ? docStore.getCold(localId)?.segmentId ?? null : null
      return currentSegmentId ?? fallbackSegmentId ?? null
    }

    const shouldReuseSplitCreatedBinding = (createdLocalId, candidateSegmentId) => {
      const normalizedSegmentId = normalizeSegmentId(candidateSegmentId)
      if (!normalizedSegmentId) {
        return false
      }

      const boundLocalId = docStore.bindingBySegmentId.get(normalizedSegmentId)
      if (boundLocalId && boundLocalId !== createdLocalId) {
        return false
      }

      const hasPendingDelete = docStore.getTombstones().some((tombstone) => (
        tombstone?.localId === createdLocalId
        && normalizeSegmentId(tombstone?.segmentId) === normalizedSegmentId
      ))

      const syncEngine = useEditorSyncEngine()
      const hasAckBinding = syncEngine.hasAckBinding(
        createdLocalId,
        normalizedSegmentId
      )
      if (hasPendingDelete && hasAckBinding) {
        return true
      }

      return boundLocalId === createdLocalId
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
      const sourceLocalId = resolveReplayLocalId(
        command.sourceLocalId,
        command.sourceSegmentId,
        [command.before, command.afterKept]
      )
      const createdColdInit = command.createdColdInit
      if (!createdColdInit || typeof createdColdInit !== 'object') {
        const nextCommand = {
          ...command,
          sourceLocalId,
          sourceSegmentId: resolveCurrentSegmentId(sourceLocalId, command.sourceSegmentId),
        }
        const sourceExists = Boolean(sourceLocalId && docStore.getEntity(sourceLocalId))
        if (sourceExists && isLocalIdMatchingSnapshot(sourceLocalId, command.afterKept)) {
          return markHistoryNoop(nextCommand, 'split_already_applied')
        }
        return nextCommand
      }

      const fallbackCreatedSegmentId = resolveSplitCreatedFallbackSegmentId(command, createdColdInit)
      const liveCreatedSegmentId = resolveCurrentSegmentId(command.createdLocalId, fallbackCreatedSegmentId)
      const createdLocalId = resolveReplayLocalId(
        command.createdLocalId,
        liveCreatedSegmentId,
        [command.afterCreated]
      )
      const resolvedCreatedSegmentId = resolveCurrentSegmentId(createdLocalId, liveCreatedSegmentId)
      const shouldKeepCreatedBinding = shouldReuseSplitCreatedBinding(
        createdLocalId,
        resolvedCreatedSegmentId
      )
      const nextCreatedColdInit = shouldKeepCreatedBinding
        ? {
            ...createdColdInit,
            segmentId: resolvedCreatedSegmentId,
          }
        : {
            ...createdColdInit,
            segmentId: null,
            sentenceIndex: null,
          }

      const nextCommand = {
        ...command,
        sourceLocalId,
        createdLocalId,
        sourceSegmentId: resolveCurrentSegmentId(sourceLocalId, command.sourceSegmentId),
        createdColdInit: nextCreatedColdInit,
      }
      const sourceExists = Boolean(sourceLocalId && docStore.getEntity(sourceLocalId))
      const createdExists = Boolean(createdLocalId && docStore.getEntity(createdLocalId))
      const alreadySplit = sourceExists
        && createdExists
        && !isLocalIdMatchingSnapshot(sourceLocalId, command.before)
        && isLocalIdMatchingSnapshot(sourceLocalId, command.afterKept)
        && isLocalIdMatchingSnapshot(createdLocalId, command.afterCreated)
      if (alreadySplit) {
        return markHistoryNoop(nextCommand, 'split_already_applied')
      }
      return nextCommand
    }

    if (command.type === 'merge_subtitles') {
      const keptLocalId = resolveReplayLocalId(
        command.keptLocalId,
        command.keptSegmentId,
        [command.beforeKept, command.after]
      )
      let removedLocalId = resolveReplayLocalId(
        command.removedLocalId,
        command.removedSegmentId,
        [command.beforeRemoved]
      )
      if ((!removedLocalId || !docStore.getEntity(removedLocalId)) && keptLocalId && docStore.getEntity(keptLocalId)) {
        const keptOrderIndex = docStore.getOrderIndex(keptLocalId)
        const adjacentLocalId = keptOrderIndex >= 0
          ? docStore.order[keptOrderIndex + 1] ?? null
          : null
        if (adjacentLocalId && isLocalIdMatchingSnapshot(adjacentLocalId, command.beforeRemoved)) {
          removedLocalId = adjacentLocalId
        }
      }
      const nextCommand = {
        ...command,
        keptLocalId,
        removedLocalId,
        keptSegmentId: resolveCurrentSegmentId(keptLocalId, command.keptSegmentId),
        removedSegmentId: resolveCurrentSegmentId(removedLocalId, command.removedSegmentId),
      }
      const keptExists = Boolean(keptLocalId && docStore.getEntity(keptLocalId))
      const removedExists = Boolean(removedLocalId && docStore.getEntity(removedLocalId))
      const normalizedRemovedSegmentId = normalizeSegmentId(nextCommand.removedSegmentId)
      const removedSegmentBinding = normalizedRemovedSegmentId
        ? docStore.bindingBySegmentId.get(normalizedRemovedSegmentId)
        : null
      const hasRemovedSegmentBinding = Boolean(
        removedSegmentBinding
        && removedSegmentBinding !== keptLocalId
      )
      const hasRemovedSnapshotCandidate = hasSnapshotCandidate(command.beforeRemoved, {
        excludeLocalIds: [keptLocalId],
      })
      const alreadyMerged = keptExists
        && !removedExists
        && isLocalIdMatchingSnapshot(keptLocalId, command.after)
        && !hasRemovedSegmentBinding
        && !hasRemovedSnapshotCandidate
      if (alreadyMerged) {
        return markHistoryNoop(nextCommand, 'merge_already_applied')
      }
      return nextCommand
    }

    if (command.type === 'delete_subtitle') {
      const localId = resolveReplayLocalId(command.localId, command.segmentId, [command.snapshot])
      const nextCommand = {
        ...command,
        localId,
        segmentId: resolveCurrentSegmentId(localId, command.segmentId),
      }
      const hasTarget = Boolean(localId && docStore.getEntity(localId))
      const normalizedSegmentId = normalizeSegmentId(nextCommand.segmentId)
      const boundLocalId = normalizedSegmentId ? docStore.bindingBySegmentId.get(normalizedSegmentId) : null
      const hasSnapshotMatch = hasSnapshotCandidate(command.snapshot)
      if (!hasTarget && !boundLocalId && !hasSnapshotMatch) {
        return markHistoryNoop(nextCommand, 'already_deleted')
      }
      return nextCommand
    }

    if (command.type === 'move_boundary') {
      const upperLocalId = resolveReplayLocalId(command.upperLocalId, command.upperSegmentId)
      const lowerLocalId = resolveReplayLocalId(command.lowerLocalId, command.lowerSegmentId)
      return {
        ...command,
        upperLocalId,
        lowerLocalId,
        upperSegmentId: resolveCurrentSegmentId(upperLocalId, command.upperSegmentId),
        lowerSegmentId: resolveCurrentSegmentId(lowerLocalId, command.lowerSegmentId),
      }
    }

    if (command.type === 'update_text' || command.type === 'update_timing') {
      const localId = resolveReplayLocalId(command.localId, command.segmentId, [command.before, command.after])
      return {
        ...command,
        localId,
        segmentId: resolveCurrentSegmentId(localId, command.segmentId),
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

  function buildHistoryReplayCommand(command, index, replayCreatedAt) {
    const sessionStore = useEditorSessionStore()
    const docStore = useEditorDocumentStore()
    const enrichedCommand = enrichHistoryReplayCommand(command, docStore)
    const { skipSync, ...rest } = enrichedCommand
    return {
      ...rest,
      commandId: sessionStore.nextCommandId(),
      source: 'undo_redo',
      createdAt: replayCreatedAt + index,
    }
  }

  function replayHistoryCommandsSequentially(commands = []) {
    const replayCreatedAt = Date.now()
    const docStore = useEditorDocumentStore()

    // Trade-off: 多命令历史项若在回放前整批预物化，后续命令会拿到“前一条命令尚未生效”
    // 的旧文档上下文，结构性编辑尤其容易把 localId/segmentId 解析错位。
    // 这里改为逐条基于最新文档状态现算回放，牺牲掉伪原子式整批预检，换取更可靠的目标解析。
    for (const [index, rawCommand] of commands.filter(Boolean).entries()) {
      const replayCommand = buildHistoryReplayCommand(rawCommand, index, replayCreatedAt)
      if (!canReplayCommand(replayCommand, docStore)) {
        return false
      }
      if (replayCommand[HISTORY_NOOP_FLAG] === true) {
        continue
      }
      const result = dispatch(replayCommand)
      if (!result?.success) {
        return false
      }
    }

    return true
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
    const historyCommands = Array.isArray(candidateEntry.undoCommands)
      ? candidateEntry.undoCommands.filter(Boolean)
      : []
    const useSequentialReplay = historyCommands.length > 1
    if (!useSequentialReplay) {
      const replayCommands = materializeHistoryReplayCommands(candidateEntry.undoCommands)
      if (!replayCommands.every((command) => canReplayCommand(command, docStore))) {
        console.warn('[CommandBus] undo 预检失败，已拒绝回放，历史栈保持不变')
        return false
      }
    }

    if (!historyStore.commitUndo()) return false
    historyStore.closeTransaction()
    const hasFailure = useSequentialReplay
      ? !replayHistoryCommandsSequentially(historyCommands)
      : (() => {
          const replayCommands = materializeHistoryReplayCommands(candidateEntry.undoCommands)
          let failed = false
          replayCommands.forEach((cmd) => {
            if (cmd[HISTORY_NOOP_FLAG] === true) {
              return
            }
            const result = dispatch(cmd)
            if (!result?.success) {
              failed = true
            }
          })
          return failed
        })()
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
    const historyCommands = Array.isArray(candidateEntry.doCommands)
      ? candidateEntry.doCommands.filter(Boolean)
      : []
    const useSequentialReplay = historyCommands.length > 1
    if (!useSequentialReplay) {
      const replayCommands = materializeHistoryReplayCommands(candidateEntry.doCommands)
      if (!replayCommands.every((command) => canReplayCommand(command, docStore))) {
        console.warn('[CommandBus] redo 预检失败，已拒绝回放，历史栈保持不变')
        return false
      }
    }

    if (!historyStore.commitRedo()) return false
    historyStore.closeTransaction()
    const hasFailure = useSequentialReplay
      ? !replayHistoryCommandsSequentially(historyCommands)
      : (() => {
          const replayCommands = materializeHistoryReplayCommands(candidateEntry.doCommands)
          let failed = false
          replayCommands.forEach((cmd) => {
            if (cmd[HISTORY_NOOP_FLAG] === true) {
              return
            }
            const result = dispatch(cmd)
            if (!result?.success) {
              failed = true
            }
          })
          return failed
        })()
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
