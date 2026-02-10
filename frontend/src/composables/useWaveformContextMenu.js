/**
 * useWaveformContextMenu - 右键菜单逻辑 Composable
 *
 * 职责：右键切分、后端同步
 * 提取自 WaveformTimeline/index.vue L1391-1520
 */
import { ref, computed } from 'vue'
import transcriptionApi from '@/services/api/transcriptionApi'

/**
 * 右键菜单 Composable
 * @param {object} projectStore - Pinia store
 * @param {Ref<string>} jobIdRef - 当前任务 ID
 */
export function useWaveformContextMenu(projectStore, jobIdRef) {
  // ============ 状态 ============
  const contextMenuRef = ref(null)
  const contextMenuTarget = ref(null) // 右键点击的目标字幕ID
  const contextMenuTime = ref(0) // 右键点击的时间点

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

    const clickTime = getTimeFromClientX(e.clientX)

    // 查找点击位置对应的字幕
    const targetSubtitle = projectStore.subtitles.find(
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
      const result = projectStore.splitSubtitle(contextMenuTarget.value, {
        splitTime: contextMenuTime.value,
      })

      if (!result.success) {
        console.error('[WaveformContextMenu] 切分失败:', result.error)
      } else {
        console.log('[WaveformContextMenu] 切分成功:', result)
        await syncSplitSubtitles(result)
      }
    }

    // 清空状态
    contextMenuTarget.value = null
    contextMenuTime.value = 0
  }

  /**
   * 波形切分结果同步到后端
   */
  async function syncSplitSubtitles(result) {
    const jobId = jobIdRef.value
    if (!jobId) return

    const { leftSubtitle, rightSubtitle, originalSentenceIndex } = result || {}
    if (!leftSubtitle || !rightSubtitle) return

    try {
      if (originalSentenceIndex !== undefined) {
        await transcriptionApi.updateSubtitle(jobId, originalSentenceIndex, {
          text: leftSubtitle.text,
          start: leftSubtitle.start,
          end: leftSubtitle.end,
        })
        projectStore.updateSubtitle(
          leftSubtitle.id,
          {
            sentenceIndex: originalSentenceIndex,
            isModified: true,
            source: 'split',
          },
          { isUserEdit: true }
        )
      } else {
        const leftResp = await transcriptionApi.createSubtitle(jobId, {
          text: leftSubtitle.text,
          start: leftSubtitle.start,
          end: leftSubtitle.end,
        })
        const leftData = leftResp?.data?.data || leftResp?.data
        if (leftData?.index !== undefined) {
          projectStore.updateSubtitle(
            leftSubtitle.id,
            {
              sentenceIndex: leftData.index,
              isModified: true,
              source: leftData.source || 'manual',
            },
            { isUserEdit: true }
          )
        }
      }

      const rightResp = await transcriptionApi.createSubtitle(jobId, {
        text: rightSubtitle.text,
        start: rightSubtitle.start,
        end: rightSubtitle.end,
      })
      const rightData = rightResp?.data?.data || rightResp?.data
      if (rightData?.index !== undefined) {
        projectStore.updateSubtitle(
          rightSubtitle.id,
          {
            sentenceIndex: rightData.index,
            isModified: true,
            source: rightData.source || 'manual',
          },
          { isUserEdit: true }
        )
      }
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
