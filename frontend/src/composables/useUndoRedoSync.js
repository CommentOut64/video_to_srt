/**
 * V3.2.4+dev.20260303.01: undo/redo 后端增量同步 composable
 *
 * 职责：undo/redo 前后拍快照 -> diff -> 防抖批量同步到后端
 *
 * 设计要点：
 * - "初始快照 + 最终快照"防抖策略：连续操作只记初始态，debounce 结束后一次 diff
 * - flushSync() 供导出栅栏调用：清除 debounce timer 立即执行，失败时 re-throw
 * - batchSyncSubtitles 保留完整 envelope（不走 unwrapEnvelope），按 errors 判断业务失败
 */

import { computed, ref, toRaw } from 'vue'
import { isFeatureEnabled } from '@/config/featureFlags'
import projectApi from '@/services/api/projectApi'
import { useProjectStore } from '@/stores/projectStore'
import { useStructuralSyncStore } from '@/stores/structuralSyncStore'
import { useEditorCommandBus } from '@/stores/editor/editorCommandBus'
import { useEditorSyncEngine } from '@/stores/editor/editorSyncEngine'

const DEBOUNCE_MS = 300

export function useUndoRedoSync() {
  const useEditorV2 = isFeatureEnabled('USE_EDITOR_V2')
  const projectStore = useProjectStore()
  const structuralSyncStore = useStructuralSyncStore()
  const editorCommandBus = useEditorV2 ? useEditorCommandBus() : null
  const editorSyncEngine = useEditorV2 ? useEditorSyncEngine() : null
  let initialSnapshot = null // 连续操作中第一次的 before 快照
  let debounceTimer = null
  let inflightSyncPromise = null
  const isSyncing = useEditorV2
    ? computed(() => Boolean(editorSyncEngine?.isSyncing))
    : ref(false)
  const lastSyncError = ref(null)

  if (useEditorV2 && editorCommandBus && editorSyncEngine) {
    async function flushSyncV2() {
      try {
        await editorSyncEngine.flush()
        lastSyncError.value = null
      } catch (error) {
        lastSyncError.value = error?.message || '新编辑器同步失败'
        throw error
      }
    }

    return {
      undoWithSync() {
        return editorCommandBus.undo()
      },
      redoWithSync() {
        return editorCommandBus.redo()
      },
      flushSync: flushSyncV2,
      isSyncing,
      lastSyncError,
    }
  }

  /**
   * 拍摄当前字幕列表快照（以 segment_id 为主键，base time）。
   */
  function captureSnapshot() {
    const map = new Map()
    for (const sub of toRaw(projectStore.subtitles)) {
      const key = sub.segment_id || sub.id
      if (key == null) continue
      map.set(String(key), {
        segment_id: sub.segment_id || null,
        sentenceIndex: sub.sentenceIndex,
        text: sub.text,
        start: projectStore.toBaseTime(sub.start),
        end: projectStore.toBaseTime(sub.end),
      })
    }
    return map
  }

  /**
   * 计算两个快照间的增量差异。
   */
  function computeDiff(before, after) {
    const updates = []
    const creates = []
    const deletes = []

    // before 有、after 无 → 删除
    for (const [key, bSub] of before) {
      if (!after.has(key) && bSub.segment_id) {
        deletes.push({ segment_id: bSub.segment_id })
      }
    }

    // after 有、before 无 → 新增/恢复；after 有且 before 有 → 比较差异
    for (const [key, aSub] of after) {
      const bSub = before.get(key)
      if (!bSub) {
        const item = { text: aSub.text, start: aSub.start, end: aSub.end }
        if (aSub.segment_id) item.restore_segment_id = aSub.segment_id
        creates.push(item)
      } else if (aSub.segment_id) {
        const diffs = {}
        if (aSub.text !== bSub.text) diffs.text = aSub.text
        if (Math.abs(aSub.start - bSub.start) > 0.001) diffs.start = aSub.start
        if (Math.abs(aSub.end - bSub.end) > 0.001) diffs.end = aSub.end
        if (Object.keys(diffs).length > 0) {
          updates.push({ segment_id: aSub.segment_id, ...diffs })
        }
      }
    }

    return { updates, creates, deletes }
  }

  /**
   * 执行同步：计算 diff 并调用 batch-sync API。
   */
  async function executeSync() {
    // 已有同步在进行时，直接复用，避免并发写后端
    if (inflightSyncPromise) {
      return inflightSyncPromise
    }
    if (!initialSnapshot) return
    const projectId = projectStore.meta.projectId
    if (!projectId) {
      initialSnapshot = null
      return
    }

    const finalSnapshot = captureSnapshot()
    const diff = computeDiff(initialSnapshot, finalSnapshot)
    initialSnapshot = null

    const total = diff.updates.length + diff.creates.length + diff.deletes.length
    if (total === 0) return

    inflightSyncPromise = (async () => {
      isSyncing.value = true
      lastSyncError.value = null
      try {
        // batchSyncSubtitles 返回完整 envelope {success, data}
        const envelope = await projectApi.batchSyncSubtitles(projectId, diff)
        const result = envelope?.data
        const success = envelope?.success

        // 回传 created_segments，patch 本地无 segment_id 的字幕
        if (result?.created_segments?.length) {
          patchCreatedSegmentIds(result.created_segments)
        }

        // 按 success 字段 + errors 数组判断业务失败
        if (success === false || (result?.errors && result.errors.length > 0)) {
          const errMsg = (result?.errors || []).join('; ') || '部分操作失败'
          lastSyncError.value = errMsg
          console.warn('[UndoRedoSync] 后端部分操作失败:', errMsg)
        } else {
          // V3.2.4+dev.20260304.01: batch-sync 成功意味着后端已被全量快照重新对账，
          // 先前的结构性操作失败不再相关
          structuralSyncStore.clearAllErrors()
        }
      } catch (error) {
        lastSyncError.value = error?.message || '撤销/重做同步失败'
        console.error('[UndoRedoSync] 后端同步失败:', error)
      } finally {
        isSyncing.value = false
      }
    })()

    try {
      await inflightSyncPromise
    } finally {
      inflightSyncPromise = null
      // 同步期间如果又累积了新快照，补一次防抖提交，避免遗漏
      if (initialSnapshot && !debounceTimer) {
        scheduleDebouncedSync()
      }
    }
  }

  /**
   * 用后端返回的 segment_id patch 本地无 segment_id 的字幕。
   * 按 legacy_index（对应前端 sentenceIndex）精确匹配。
   */
  function patchCreatedSegmentIds(createdSegments) {
    for (const created of createdSegments) {
      if (!created.segment_id) continue
      // 1) 优先按 legacy_index（精确）匹配
      let match = projectStore.subtitles.find(
        (sub) => !sub.segment_id && sub.sentenceIndex === created.legacy_index
      )
      // 2) 兜底按 text/start/end（近似）匹配，覆盖短暂无 sentenceIndex 的边界窗口
      if (!match && created.start !== undefined && created.end !== undefined) {
        match = projectStore.subtitles.find((sub) => {
          if (sub.segment_id) return false
          const textEqual = (sub.text || '') === (created.text || '')
          const startDiff = Math.abs(projectStore.toBaseTime(sub.start) - Number(created.start))
          const endDiff = Math.abs(projectStore.toBaseTime(sub.end) - Number(created.end))
          return textEqual && startDiff < 0.01 && endDiff < 0.01
        })
      }
      if (match) {
        projectStore.pauseHistory()
        try {
          const payload = { segment_id: created.segment_id }
          if (
            created.legacy_index !== undefined
            && created.legacy_index !== null
            && match.sentenceIndex === undefined
          ) {
            payload.sentenceIndex = created.legacy_index
          }
          projectStore.updateSubtitle(match.id, payload, { isUserEdit: false })
        } finally {
          projectStore.resumeHistory()
        }
      }
    }
  }

  function scheduleDebouncedSync() {
    if (debounceTimer) clearTimeout(debounceTimer)
    debounceTimer = setTimeout(() => {
      debounceTimer = null
      executeSync()
    }, DEBOUNCE_MS)
  }

  /**
   * 导出栅栏：清除 debounce timer 并立即执行同步。
   * 如果 executeSync 记录了 lastSyncError，re-throw 让调用方捕获。
   */
  async function flushSync() {
    if (debounceTimer) {
      clearTimeout(debounceTimer)
      debounceTimer = null
    }
    await executeSync()
    // 关键：若 flush 期间命中了“已有进行中的同步”，这里确保等待其完成
    if (inflightSyncPromise) {
      await inflightSyncPromise
    }
    if (lastSyncError.value) {
      const err = lastSyncError.value
      lastSyncError.value = null
      throw new Error(`撤销/重做同步失败: ${err}`)
    }
  }

  function undoWithSync() {
    if (!projectStore.canUndo) return
    if (!initialSnapshot) initialSnapshot = captureSnapshot()
    projectStore.undo()
    scheduleDebouncedSync()
  }

  function redoWithSync() {
    if (!projectStore.canRedo) return
    if (!initialSnapshot) initialSnapshot = captureSnapshot()
    projectStore.redo()
    scheduleDebouncedSync()
  }

  return { undoWithSync, redoWithSync, flushSync, isSyncing, lastSyncError }
}
