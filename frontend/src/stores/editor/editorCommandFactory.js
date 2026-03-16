// V3.2.5+dev.20260315.05: 命令工厂 - 统一命令默认值与最小补全
import { useEditorSessionStore } from './editorSessionStore'
import { useEditorDocumentStore } from './editorDocumentStore'

const DEFAULT_INSERT_DURATION_MS = 3000
const MIN_INSERT_DURATION_MS = 500
const STRUCTURAL_COMMAND_TYPES = new Set([
  'insert_subtitle',
  'delete_subtitle',
  'split_subtitle',
  'merge_subtitles',
  'move_boundary',
])

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

function resolveInsertAnchor(afterLocalId, docStore) {
  if (afterLocalId !== null && afterLocalId !== undefined) {
    return {
      beforeClientRefId: afterLocalId,
      afterClientRefId: docStore.getNeighbors(afterLocalId).next ?? null,
    }
  }

  return {
    beforeClientRefId: null,
    afterClientRefId: docStore.order[0] ?? null,
  }
}

function assertHasOwnField(command, fieldName) {
  if (!Object.prototype.hasOwnProperty.call(command, fieldName)) {
    throw new Error(`[CommandFactory] ${command.type} 缺少必填字段：${fieldName}`)
  }
}

function assertObjectField(command, fieldName) {
  assertHasOwnField(command, fieldName)
  const value = command[fieldName]
  if (!value || typeof value !== 'object') {
    throw new Error(`[CommandFactory] ${command.type} 字段必须是对象：${fieldName}`)
  }
}

function validateStructuralCommand(command) {
  if (!command || !STRUCTURAL_COMMAND_TYPES.has(command.type)) {
    return command
  }

  switch (command.type) {
    case 'insert_subtitle':
      assertHasOwnField(command, 'localId')
      assertHasOwnField(command, 'beforeClientRefId')
      assertHasOwnField(command, 'afterClientRefId')
      assertObjectField(command, 'entity')
      break
    case 'delete_subtitle':
      assertHasOwnField(command, 'localId')
      assertHasOwnField(command, 'segmentId')
      assertObjectField(command, 'snapshot')
      break
    case 'split_subtitle':
      assertHasOwnField(command, 'sourceLocalId')
      assertHasOwnField(command, 'sourceSegmentId')
      assertHasOwnField(command, 'createdLocalId')
      assertHasOwnField(command, 'splitAtMs')
      assertHasOwnField(command, 'splitAtTextOffset')
      assertObjectField(command, 'before')
      assertObjectField(command, 'afterKept')
      assertObjectField(command, 'afterCreated')
      break
    case 'merge_subtitles':
      assertHasOwnField(command, 'keptLocalId')
      assertHasOwnField(command, 'removedLocalId')
      assertHasOwnField(command, 'keptSegmentId')
      assertHasOwnField(command, 'removedSegmentId')
      assertObjectField(command, 'beforeKept')
      assertObjectField(command, 'beforeRemoved')
      assertObjectField(command, 'after')
      break
    case 'move_boundary':
      assertHasOwnField(command, 'upperLocalId')
      assertHasOwnField(command, 'lowerLocalId')
      assertHasOwnField(command, 'upperSegmentId')
      assertHasOwnField(command, 'lowerSegmentId')
      assertObjectField(command, 'before')
      assertObjectField(command, 'after')
      break
    default:
      break
  }

  return command
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
  const frozenAnchor = resolveInsertAnchor(afterLocalId, docStore)

  return validateStructuralCommand(createBaseCommand('insert_subtitle', {
    localId,
    afterLocalId,
    beforeClientRefId: overrides.beforeClientRefId ?? frozenAnchor.beforeClientRefId,
    afterClientRefId: overrides.afterClientRefId ?? frozenAnchor.afterClientRefId,
    entity: overrides.entity ?? {
      text: '',
      startMs: fallbackWindow.startMs,
      endMs: fallbackWindow.endMs,
      isDraft: false,
    },
    coldInit: overrides.coldInit,
    ...overrides,
  }))
}

export function createDeleteSubtitleCommand(overrides = {}) {
  const docStore = useEditorDocumentStore()
  const entity = overrides.localId ? docStore.getEntity(overrides.localId) : null
  const cold = overrides.localId ? docStore.getCold(overrides.localId) : null
  const neighbors = overrides.localId ? docStore.getNeighbors(overrides.localId) : { prev: null, next: null }

  return validateStructuralCommand(createBaseCommand('delete_subtitle', {
    segmentId: overrides.segmentId ?? cold?.segmentId ?? null,
    snapshot: overrides.snapshot ?? {
      text: entity?.text ?? '',
      startMs: entity?.startMs ?? 0,
      endMs: entity?.endMs ?? 0,
      isDraft: entity?.isDraft ?? false,
      orderIndex: overrides.localId ? docStore.getOrderIndex(overrides.localId) : -1,
    },
    coldSnapshot: overrides.coldSnapshot ?? (cold ? { ...cold } : undefined),
    beforeClientRefId: overrides.beforeClientRefId ?? neighbors.prev ?? null,
    afterClientRefId: overrides.afterClientRefId ?? neighbors.next ?? null,
    ...overrides,
  }))
}

export function createSplitSubtitleCommand(overrides = {}) {
  const docStore = useEditorDocumentStore()
  const sourceCold = overrides.sourceLocalId ? docStore.getCold(overrides.sourceLocalId) : null

  return validateStructuralCommand(createBaseCommand('split_subtitle', {
    sourceSegmentId: overrides.sourceSegmentId ?? sourceCold?.segmentId ?? null,
    ...overrides,
  }))
}

export function createMergeSubtitlesCommand(overrides = {}) {
  const docStore = useEditorDocumentStore()
  const keptCold = overrides.keptLocalId ? docStore.getCold(overrides.keptLocalId) : null
  const removedCold = overrides.removedLocalId ? docStore.getCold(overrides.removedLocalId) : null

  return validateStructuralCommand(createBaseCommand('merge_subtitles', {
    keptSegmentId: overrides.keptSegmentId ?? keptCold?.segmentId ?? null,
    removedSegmentId: overrides.removedSegmentId ?? removedCold?.segmentId ?? null,
    ...overrides,
  }))
}

export function createMoveBoundaryCommand(overrides = {}) {
  const docStore = useEditorDocumentStore()
  const upperCold = overrides.upperLocalId ? docStore.getCold(overrides.upperLocalId) : null
  const lowerCold = overrides.lowerLocalId ? docStore.getCold(overrides.lowerLocalId) : null

  return validateStructuralCommand(createBaseCommand('move_boundary', {
    upperSegmentId: overrides.upperSegmentId ?? upperCold?.segmentId ?? null,
    lowerSegmentId: overrides.lowerSegmentId ?? lowerCold?.segmentId ?? null,
    ...overrides,
  }))
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
      return validateStructuralCommand(createBaseCommand(command.type, command))
  }
}
