/**
 * useWaveformContextMenu - 右键菜单逻辑 Composable
 *
 * 职责：右键切分、后端同步
 * 提取自 WaveformTimeline/index.vue L1391-1520
 */
import { ref, computed } from 'vue'
import projectApi from '@/services/api/projectApi'
import { useStructuralSyncStore } from '@/stores/structuralSyncStore'

/**
 * 右键菜单 Composable
 * @param {object} projectStore - Pinia store
 * @param {object} subtitleDocumentStore - 字幕文档 store
 */
export function useWaveformContextMenu(projectStore, subtitleDocumentStore) {
  // ============ 状态 ============
  const structuralSyncStore = useStructuralSyncStore()
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
    if (targetSubtitle && editableTarget && !targetSubtitle.isDraft) {
      contextMenuTarget.value = editableTarget.id
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
