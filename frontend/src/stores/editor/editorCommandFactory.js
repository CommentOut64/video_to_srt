// V3.2.5+dev.20260315.05: 命令工厂 - 统一命令默认值与最小补全
import { useEditorSessionStore } from './editorSessionStore'
import { useEditorDocumentStore } from './editorDocumentStore'

const DEFAULT_INSERT_DURATION_MS = 3000
const MIN_INSERT_DURATION_MS = 500

function createBaseCommand(type, overrides = {}) {
  const sessionStore = useEditorSessionStore()
  const {
    commandId,
    source,
    createdAt,
    ...rest
  } = overrides

  return {
    ...rest,
    commandId: commandId || sessionStore.nextCommandId(),
    type,
    source: source || 'user',
    createdAt: createdAt ?? Date.now(),
  }
}

function resolveInsertWindow(afterLocalId, docStore) {
  const anchorLocalId = afterLocalId ?? docStore.order[docStore.order.length - 1] ?? null
  const anchor = anchorLocalId ? docStore.getEntity(anchorLocalId) : null
  const neighbors = anchorLocalId ? docStore.getNeighbors(anchorLocalId) : { next: null }
  const nextEntity = neighbors.next ? docStore.getEntity(neighbors.next) : null

  const startMs = anchor?.endMs ?? 0
  const naturalEndMs = startMs + DEFAULT_INSERT_DURATION_MS
  const boundedEndMs = nextEntity?.startMs && nextEntity.startMs > startMs
    ? Math.min(naturalEndMs, nextEntity.startMs)
    : naturalEndMs
  const endMs = boundedEndMs > startMs
    ? boundedEndMs
    : startMs + MIN_INSERT_DURATION_MS

  return { startMs, endMs }
}

export function createUpdateTextCommand(overrides = {}) {
  const docStore = useEditorDocumentStore()
  const entity = overrides.localId ? docStore.getEntity(overrides.localId) : null
  return createBaseCommand('update_text', {
    before: overrides.before ?? { text: entity?.text ?? '' },
    after: overrides.after ?? { text: entity?.text ?? '' },
    mergeKey: overrides.mergeKey ?? null,
    ...overrides,
  })
}

export function createUpdateTimingCommand(overrides = {}) {
  const docStore = useEditorDocumentStore()
  const entity = overrides.localId ? docStore.getEntity(overrides.localId) : null
  return createBaseCommand('update_timing', {
    before: overrides.before ?? {
      startMs: entity?.startMs ?? 0,
      endMs: entity?.endMs ?? 0,
    },
    after: overrides.after ?? {
      startMs: entity?.startMs ?? 0,
      endMs: entity?.endMs ?? 0,
    },
    ...overrides,
  })
}

export function createInsertSubtitleCommand(overrides = {}) {
  const sessionStore = useEditorSessionStore()
  const docStore = useEditorDocumentStore()
  const localId = overrides.localId || sessionStore.nextLocalId()
  const afterLocalId = overrides.afterLocalId ?? null
  const fallbackWindow = resolveInsertWindow(afterLocalId, docStore)

  return createBaseCommand('insert_subtitle', {
    localId,
    afterLocalId,
    entity: overrides.entity ?? {
      text: '',
      startMs: fallbackWindow.startMs,
      endMs: fallbackWindow.endMs,
      isDraft: false,
    },
    coldInit: overrides.coldInit,
    ...overrides,
  })
}

export function createDeleteSubtitleCommand(overrides = {}) {
  return createBaseCommand('delete_subtitle', overrides)
}

export function createSplitSubtitleCommand(overrides = {}) {
  return createBaseCommand('split_subtitle', overrides)
}

export function createMergeSubtitlesCommand(overrides = {}) {
  return createBaseCommand('merge_subtitles', overrides)
}

export function createMoveBoundaryCommand(overrides = {}) {
  return createBaseCommand('move_boundary', overrides)
}

export function createBatchReplaceCommand(overrides = {}) {
  return createBaseCommand('batch_replace', {
    replacements: Array.isArray(overrides.replacements) ? overrides.replacements : [],
    ...overrides,
  })
}

export function normalizeEditorCommand(command) {
  if (!command || !command.type) {
    return command
  }

  switch (command.type) {
    case 'update_text':
      return createUpdateTextCommand(command)
    case 'update_timing':
      return createUpdateTimingCommand(command)
    case 'insert_subtitle':
      return createInsertSubtitleCommand(command)
    case 'delete_subtitle':
      return createDeleteSubtitleCommand(command)
    case 'split_subtitle':
      return createSplitSubtitleCommand(command)
    case 'merge_subtitles':
      return createMergeSubtitlesCommand(command)
    case 'move_boundary':
      return createMoveBoundaryCommand(command)
    case 'batch_replace':
      return createBatchReplaceCommand(command)
    default:
      return createBaseCommand(command.type, command)
  }
}
