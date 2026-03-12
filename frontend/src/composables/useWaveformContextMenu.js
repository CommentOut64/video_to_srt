/**
 * useWaveformContextMenu - 右键菜单逻辑 Composable
 *
 * 职责：右键切分、统一命令同步
 * 提取自 WaveformTimeline/index.vue L1391-1520
 */
import { ref, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { useSyncCoordinatorStore } from '@/core/sync/syncCoordinator'
import { useEditorTimingStore } from '@/stores/editorTimingStore'

/**
 * 右键菜单 Composable
 * @param {object} projectStore - Pinia store
 */
export function useWaveformContextMenu(projectStore) {
  // ============ 状态 ============
  const syncCoordinator = useSyncCoordinatorStore()
  const editorTimingStore = useEditorTimingStore()
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

  function buildUpdateCommand(type, subtitle) {
    return {
      type: 'update_subtitle',
      command_id: syncCoordinator.nextCommandId(type),
      segment_id: subtitle.segment_id,
      text: subtitle.text,
      start: editorTimingStore.toBaseTime(subtitle.start),
      end: editorTimingStore.toBaseTime(subtitle.end),
    }
  }

  function buildAddCommand(type, subtitle) {
    return {
      type: 'add_subtitle',
      command_id: syncCoordinator.nextCommandId(type),
      local_id: subtitle?.id == null ? null : String(subtitle.id),
      text: subtitle.text,
      start: editorTimingStore.toBaseTime(subtitle.start),
      end: editorTimingStore.toBaseTime(subtitle.end),
    }
  }

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
   * 波形切分结果统一走 editor-ops:apply，避免组件层继续分散调用旧 API。
   */
  async function syncSplitSubtitles(result) {
    const projectId = projectStore.meta.projectId
    if (!projectId) {
      throw new Error('缺少 project_id，禁止走 job 字幕切分分支')
    }

    const { leftSubtitle, rightSubtitle } = result || {}
    if (!leftSubtitle || !rightSubtitle) return

    const commands = []
    if (leftSubtitle.segment_id) {
      commands.push(buildUpdateCommand('waveform-split-left', leftSubtitle))
    } else {
      commands.push(buildAddCommand('waveform-split-left', leftSubtitle))
    }
    commands.push(buildAddCommand('waveform-split-right', rightSubtitle))

    const { promise } = syncCoordinator.submitStructuralCommands('waveform-split', commands)
    try {
      await promise
    } catch (error) {
      console.warn('[WaveformContextMenu] 切分同步失败:', error)
      ElMessage.warning(`切分同步失败：${error?.message || '请重试'}`)
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
