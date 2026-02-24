/**
 * useWaveformContextMenu - 右键菜单逻辑 Composable
 *
 * 职责：右键切分、后端同步
 * 提取自 WaveformTimeline/index.vue L1391-1520
 */
import { ref, computed } from 'vue'
import transcriptionApi from '@/services/api/transcriptionApi'
import projectApi from '@/services/api/projectApi'

/**
 * 右键菜单 Composable
 * @param {object} projectStore - Pinia store
 * @param {Ref<string>} identityRef - 当前媒体身份 ID（projectId 或 jobId）
 */
export function useWaveformContextMenu(projectStore, identityRef) {
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

    if (typeof getTimeFromClientX !== 'function') {
      console.warn('[WaveformContextMenu] 缺少时间映射函数，忽略本次右键事件')
      return
    }

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
    const projectId = projectStore.meta.projectId
    const jobId = projectStore.meta.jobId || identityRef.value
    if (!jobId && !projectId) return

    const { leftSubtitle, rightSubtitle, originalSentenceIndex } = result || {}
    if (!leftSubtitle || !rightSubtitle) return

    try {
      if (projectId) {
        const leftSegmentId = leftSubtitle.segment_id
        if (leftSegmentId) {
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
        }

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
      } else if (jobId) {
        if (originalSentenceIndex !== undefined) {
          await transcriptionApi.updateSubtitle(jobId, originalSentenceIndex, {
            text: leftSubtitle.text,
            start: projectStore.toBaseTime(leftSubtitle.start),
            end: projectStore.toBaseTime(leftSubtitle.end),
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
            start: projectStore.toBaseTime(leftSubtitle.start),
            end: projectStore.toBaseTime(leftSubtitle.end),
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
          start: projectStore.toBaseTime(rightSubtitle.start),
          end: projectStore.toBaseTime(rightSubtitle.end),
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
