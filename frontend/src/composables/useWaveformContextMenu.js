/**
 * useWaveformContextMenu - 右键菜单逻辑 Composable
 *
 * 职责：右键切分、后端同步
 * 提取自 WaveformTimeline/index.vue L1391-1520
 */
import { ref, computed } from 'vue'
import { isFeatureEnabled } from '@/config/featureFlags'
import projectApi from '@/services/api/projectApi'
import { useStructuralSyncStore } from '@/stores/structuralSyncStore'
import { useEditorCommandBus } from '@/stores/editor/editorCommandBus'
import { useEditorDocumentStore } from '@/stores/editor/editorDocumentStore'
import { useEditorSessionStore } from '@/stores/editor/editorSessionStore'
import { createSplitSubtitleCommand } from '@/stores/editor/editorCommandFactory'

/**
 * 右键菜单 Composable
 * @param {object} projectStore - Pinia store
 * @param {object} subtitleDocumentStore - 字幕文档 store
 */
export function useWaveformContextMenu(projectStore, subtitleDocumentStore) {
  // ============ 状态 ============
  const useEditorV2 = isFeatureEnabled('USE_EDITOR_V2')
  const structuralSyncStore = useStructuralSyncStore()
  const docStore = useEditorV2 ? useEditorDocumentStore() : null
  const commandBus = useEditorV2 ? useEditorCommandBus() : null
  const sessionStore = useEditorV2 ? useEditorSessionStore() : null
  const contextMenuRef = ref(null)
  const contextMenuTarget = ref(null) // 右键点击的目标字幕ID
  const contextMenuTime = ref(0) // 右键点击的时间点

  function resolveMaybeRefValue(valueOrRef) {
    if (valueOrRef && typeof valueOrRef === 'object' && 'value' in valueOrRef) {
      return valueOrRef.value
    }
    return valueOrRef
  }

  function getSubtitleList() {
    const subtitles = resolveMaybeRefValue(subtitleDocumentStore?.subtitles)
    if (Array.isArray(subtitles)) {
      return subtitles
    }
    return Array.isArray(projectStore?.subtitles) ? projectStore.subtitles : []
  }

  function findMirroredProjectSubtitle(targetSubtitle) {
    if (!targetSubtitle) return null

    return (
      projectStore.subtitles.find((item) => item.id === targetSubtitle.id)
      || projectStore.subtitles.find(
        (item) => targetSubtitle.segment_id && item.segment_id === targetSubtitle.segment_id
      )
      || projectStore.subtitles.find(
        (item) => targetSubtitle.sentenceIndex !== undefined && item.sentenceIndex === targetSubtitle.sentenceIndex
      )
      || null
    )
  }

  function toBaseMs(displaySeconds) {
    return Math.round(projectStore.toBaseTime(displaySeconds) * 1000)
  }

  function normalizeWordsToBase(words = []) {
    if (!Array.isArray(words) || words.length === 0) {
      return []
    }

    return words
      .map((word) => {
        const start = Number(word?.start)
        const end = Number(word?.end)
        if (!Number.isFinite(start) || !Number.isFinite(end)) {
          return null
        }
        return {
          startMs: toBaseMs(start),
          endMs: toBaseMs(end),
          text: word?.word ?? word?.text ?? '',
        }
      })
      .filter(Boolean)
  }

  function splitSubtitleByTime(subtitle, splitTime) {
    const { start, end, text, words } = subtitle

    if (splitTime <= start || splitTime >= end) {
      return { success: false, error: '切分点必须在字幕时间范围内' }
    }

    if (Array.isArray(words) && words.length > 0) {
      const splitIndex = words.findIndex((word) => Number(word?.end) >= splitTime)
      if (splitIndex > 0) {
        const leftWords = words.slice(0, splitIndex)
        const rightWords = words.slice(splitIndex)
        const leftText = leftWords.map((word) => word.word ?? word.text ?? '').join('')
        const rightText = rightWords.map((word) => word.word ?? word.text ?? '').join('')
        const boundaryTime = Number(rightWords[0]?.start ?? leftWords[leftWords.length - 1]?.end ?? splitTime)

        return {
          success: true,
          splitAtTime: boundaryTime,
          splitAtTextOffset: leftText.length,
          left: {
            start,
            end: Number(leftWords[leftWords.length - 1]?.end ?? boundaryTime),
            text: leftText,
            words: leftWords,
          },
          right: {
            start: Number(rightWords[0]?.start ?? boundaryTime),
            end,
            text: rightText,
            words: rightWords,
          },
        }
      }
    }

    if (String(text ?? '').length < 2) {
      return { success: false, error: '字幕文本过短，无法切分' }
    }

    const ratio = (splitTime - start) / (end - start)
    const splitAtTextOffset = Math.max(1, Math.min(text.length - 1, Math.round(text.length * ratio)))

    return {
      success: true,
      splitAtTime: splitTime,
      splitAtTextOffset,
      left: {
        start,
        end: splitTime,
        text: text.slice(0, splitAtTextOffset),
        words: [],
      },
      right: {
        start: splitTime,
        end,
        text: text.slice(splitAtTextOffset),
        words: [],
      },
    }
  }

  function buildV2SplitCommand(localId, splitTime) {
    if (!docStore || !commandBus || !sessionStore) {
      return { success: false, error: '编辑器内核未就绪' }
    }

    const subtitle = getSubtitleList().find((item) => item.id === localId)
    const entity = docStore.getEntity(localId)
    const cold = docStore.getCold(localId)
    if (!subtitle || !entity) {
      return { success: false, error: '字幕不存在' }
    }

    const splitResult = splitSubtitleByTime(subtitle, splitTime)
    if (!splitResult.success) {
      return splitResult
    }

    const createdLocalId = sessionStore.nextLocalId()
    return {
      success: true,
      command: createSplitSubtitleCommand({
        sourceLocalId: localId,
        createdLocalId,
        splitAtMs: toBaseMs(splitResult.splitAtTime),
        splitAtTextOffset: splitResult.splitAtTextOffset,
        before: {
          text: entity.text,
          startMs: entity.startMs,
          endMs: entity.endMs,
        },
        afterKept: {
          text: splitResult.left.text,
          startMs: toBaseMs(splitResult.left.start),
          endMs: toBaseMs(splitResult.left.end),
        },
        afterCreated: {
          text: splitResult.right.text,
          startMs: toBaseMs(splitResult.right.start),
          endMs: toBaseMs(splitResult.right.end),
        },
        createdColdInit: cold
          ? {
              ...cold,
              localId: createdLocalId,
              segmentId: null,
              sentenceIndex: null,
              originalText: null,
              words: normalizeWordsToBase(splitResult.right.words),
            }
          : undefined,
        source: 'user',
      }),
    }
  }

  // ============ 计算属性 ============
  const contextMenuItems = computed(() => {
    const items = []

    if (contextMenuTarget.value) {
      items.push({
        key: 'split',
        label: '从此处切分',
      })
    }

    return items
  })

  // ============ 方法 ============

  /**
   * 波形区域右键事件处理
   * @param {MouseEvent} e - 鼠标事件
   * @param {Function} getTimeFromClientX - 获取时间的函数
   */
  function handleWaveformContextMenu(e, getTimeFromClientX) {
    e.preventDefault()
    e.stopPropagation()

    if (typeof getTimeFromClientX !== 'function') {
      console.warn('[WaveformContextMenu] 缺少时间映射函数，忽略本次右键事件')
      return
    }

    const clickTime = getTimeFromClientX(e.clientX)

    // 查找点击位置对应的字幕
    const targetSubtitle = getSubtitleList().find(
      (s) => clickTime >= s.start && clickTime < s.end
    )
    const editableTarget = findMirroredProjectSubtitle(targetSubtitle)

    // 只有在字幕范围内才显示菜单
    if (targetSubtitle && !targetSubtitle.isDraft && (useEditorV2 || editableTarget)) {
      contextMenuTarget.value = useEditorV2 ? targetSubtitle.id : editableTarget.id
      contextMenuTime.value = clickTime
      contextMenuRef.value?.show(e.clientX, e.clientY)
    }
  }

  /**
   * 右键菜单项选择处理
   */
  async function handleContextMenuSelect(key) {
    if (key === 'split' && contextMenuTarget.value) {
      if (useEditorV2 && commandBus) {
        const result = buildV2SplitCommand(contextMenuTarget.value, contextMenuTime.value)
        if (!result.success) {
          console.error('[WaveformContextMenu] 切分失败:', result.error)
        } else {
          const dispatchResult = commandBus.dispatch(result.command)
          if (!dispatchResult?.success) {
            console.error('[WaveformContextMenu] 切分命令执行失败:', dispatchResult?.reason)
          }
        }
      } else {
        const result = projectStore.splitSubtitle(contextMenuTarget.value, {
          splitTime: contextMenuTime.value,
        })

        if (!result.success) {
          console.error('[WaveformContextMenu] 切分失败:', result.error)
        } else {
          await syncSplitSubtitles(result)
        }
      }
    }

    // 清空状态
    contextMenuTarget.value = null
    contextMenuTime.value = 0
  }

  /**
   * V3.2.4+dev.20260304.01: 波形切分结果同步到后端 — structuralSyncStore 飞行追踪
   */
  async function syncSplitSubtitles(result) {
    const projectId = projectStore.meta.projectId
    if (!projectId) {
      throw new Error('缺少 project_id，禁止走 job 字幕切分分支')
    }

    const { leftSubtitle, rightSubtitle } = result || {}
    if (!leftSubtitle || !rightSubtitle) return

    const syncPromise = (async () => {
      const leftSegmentId = leftSubtitle.segment_id
      if (!leftSegmentId) {
        throw new Error('切分左半字幕缺少 segment_id，无法同步到 project 字幕真源')
      }

      const updatedLeft = await projectApi.updateSubtitle(projectId, leftSegmentId, {
        text: leftSubtitle.text,
        start: projectStore.toBaseTime(leftSubtitle.start),
        end: projectStore.toBaseTime(leftSubtitle.end),
      })
      projectStore.updateSubtitle(
        leftSubtitle.id,
        {
          sentenceIndex: updatedLeft?.legacy_index ?? leftSubtitle.sentenceIndex,
          segment_id: updatedLeft?.segment_id ?? leftSegmentId,
          isModified: true,
          source: updatedLeft?.source_type || 'split',
        },
        { isUserEdit: true }
      )

      const rightData = await projectApi.createSubtitle(projectId, {
        text: rightSubtitle.text,
        start: projectStore.toBaseTime(rightSubtitle.start),
        end: projectStore.toBaseTime(rightSubtitle.end),
      })
      projectStore.updateSubtitle(
        rightSubtitle.id,
        {
          sentenceIndex: rightData?.legacy_index ?? rightSubtitle.sentenceIndex,
          segment_id: rightData?.segment_id ?? rightSubtitle.segment_id,
          isModified: true,
          source: rightData?.source_type || 'manual',
        },
        { isUserEdit: true }
      )
    })()
    structuralSyncStore.trackOperation('split', syncPromise)
    try {
      await syncPromise
    } catch (error) {
      console.warn('[WaveformContextMenu] 切分同步失败:', error)
    }
  }

  return {
    // Refs
    contextMenuRef,
    // 状态
    contextMenuTarget,
    contextMenuTime,
    contextMenuItems,
    // 方法
    handleWaveformContextMenu,
    handleContextMenuSelect,
  }
}
