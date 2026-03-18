/**
 * useWaveformContextMenu - 右键菜单逻辑 Composable
 *
 * 职责：右键切分、后端同步
 * 提取自 WaveformTimeline/index.vue L1391-1520
 */
import { ref, computed } from 'vue'
import { isFeatureEnabled } from '@/config/featureFlags'
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
    if (useEditorV2 && docStore) {
      return docStore.order
        .map((localId) => {
          const entity = docStore.getEntity(localId)
          if (!entity || entity.isDeleted) {
            return null
          }

          const cold = docStore.getCold(localId)
          const toDisplayTime = typeof projectStore?.toDisplayTime === 'function'
            ? projectStore.toDisplayTime.bind(projectStore)
            : (seconds) => seconds
          const start = Math.max(0, toDisplayTime(entity.startMs / 1000))
          const end = Math.max(start, toDisplayTime(entity.endMs / 1000))
          return {
            id: localId,
            localId,
            segment_id: cold?.segmentId ?? null,
            sentenceIndex: cold?.sentenceIndex ?? null,
            start,
            end,
            text: entity.text ?? '',
            words: Array.isArray(cold?.words)
              ? cold.words.map((word) => ({
                ...word,
                start: Number.isFinite(Number(word?.startMs))
                  ? toDisplayTime(Number(word.startMs) / 1000)
                  : word?.start,
                end: Number.isFinite(Number(word?.endMs))
                  ? toDisplayTime(Number(word.endMs) / 1000)
                  : word?.end,
                word: word?.word ?? word?.text ?? '',
              }))
              : [],
            isDraft: Boolean(entity.isDraft),
          }
        })
        .filter(Boolean)
    }

    const subtitles = resolveMaybeRefValue(subtitleDocumentStore?.subtitles)
    if (Array.isArray(subtitles)) {
      return subtitles
    }
    return []
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
        sourceSegmentId: cold?.segmentId ?? null,
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

    // 只有在字幕范围内才显示菜单
    if (targetSubtitle && !targetSubtitle.isDraft) {
      contextMenuTarget.value = targetSubtitle.id
      contextMenuTime.value = clickTime
      contextMenuRef.value?.show(e.clientX, e.clientY)
    }
  }

  /**
   * 右键菜单项选择处理
   */
  async function handleContextMenuSelect(key) {
    if (key === 'split' && contextMenuTarget.value) {
      if (!useEditorV2 || !commandBus) {
        throw new Error('[WaveformContextMenu] Phase F-0 已禁用旧波形切分链路')
      }

      const result = buildV2SplitCommand(contextMenuTarget.value, contextMenuTime.value)
      if (!result.success) {
        throw new Error(`[WaveformContextMenu] 切分失败: ${result.error}`)
      }

      const dispatchResult = commandBus.dispatch(result.command)
      if (!dispatchResult?.success) {
        throw new Error(
          `[WaveformContextMenu] 切分命令执行失败: ${dispatchResult?.reason || '未知原因'}`
        )
      }
    }

    // 清空状态
    contextMenuTarget.value = null
    contextMenuTime.value = 0
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
