import { onMounted, onUnmounted } from 'vue'

/**
 * 快捷键管理 Hook
 * @param {Object} actions - 包含所有操作的回调函数对象
 *
 * 支持的快捷键操作：
 * 1. 播放与导航：togglePlay, stepBackward, stepForward, seekBackward, seekForward, seekToStart, seekToEnd
 * 2. 视图缩放：zoomInWave, zoomOutWave, fitWave, zoomInVideo, zoomOutVideo, fitVideo
 * 3. 字幕编辑：fontSizeUp, fontSizeDown, splitSubtitle, mergeSubtitle
 * 4. 全局操作：save, undo, redo, export, openTaskMonitor
 */
export function useShortcuts(actions) {
  const handleKeydown = (e) => {
    if (e.defaultPrevented) return

    // 1. 基础键位检测
    const key = e.key.toLowerCase()
    const code = e.code
    const isCtrl = e.ctrlKey || e.metaKey // 兼容 Mac Command
    const isShift = e.shiftKey
    const isAlt = e.altKey

    // 2. 焦点检测：如果在输入框内，部分快捷键失效
    const target = e.target
    const tagName = target?.tagName
    // contentEditable 兼容富文本编辑器
    const isInputActive = tagName === 'INPUT' ||
                          tagName === 'TEXTAREA' ||
                          target?.isContentEditable

    // ==========================================
    // 第一类：绝对拦截 (即使在输入框中也要生效)
    // ==========================================

    // 保存 (Ctrl + S)
    if (isCtrl && key === 's') {
      e.preventDefault() // 阻止浏览器"另存为"
      actions.save?.()
      return
    }

    // 撤销 (Ctrl + Z)
    if (isCtrl && key === 'z' && !isShift) {
      e.preventDefault()
      actions.undo?.()
      return
    }

    // 重做 (Ctrl + Shift + Z)
    if (isCtrl && isShift && key === 'z') {
      e.preventDefault()
      actions.redo?.()
      return
    }

    // 重做 (Ctrl + Y) - 替代快捷键
    if (isCtrl && key === 'y') {
      e.preventDefault()
      actions.redo?.()
      return
    }

    // ==========================================
    // 第二类：输入框保护 (输入文字时不触发)
    // ==========================================
    if (isInputActive) return

    // 播放/暂停 (Space)
    if (code === 'Space') {
      // 长按空格会触发重复 keydown，忽略重复触发避免连续闪切
      if (e.repeat) return
      e.preventDefault() // 阻止网页向下滚动
      actions.togglePlay?.()
      return
    }

    // 左右逐帧/快进快退
    if (code === 'ArrowLeft') {
      e.preventDefault() // 阻止横向滚动
      if (isShift) {
        // Shift + Left: 快退5秒
        actions.seekBackward?.()
      } else {
        // Left: 上一帧
        actions.stepBackward?.()
      }
      return
    }

    if (code === 'ArrowRight') {
      e.preventDefault() // 阻止横向滚动
      if (isShift) {
        // Shift + Right: 快进5秒
        actions.seekForward?.()
      } else {
        // Right: 下一帧
        actions.stepForward?.()
      }
      return
    }

    // Home: 跳转到开头
    if (code === 'Home') {
      e.preventDefault() // 阻止滚到页首
      actions.seekToStart?.()
      return
    }

    // End: 跳转到结尾
    if (code === 'End') {
      e.preventDefault() // 阻止滚到页尾
      actions.seekToEnd?.()
      return
    }

    // ==========================================
    // 第三类：视图控制 (波形/画面)
    // ==========================================

    // 波形放大 (=)
    if (key === '=' && !isCtrl) {
      e.preventDefault()
      actions.zoomInWave?.()
      return
    }

    // 波形缩小 (-)
    if (key === '-' && !isCtrl) {
      e.preventDefault()
      actions.zoomOutWave?.()
      return
    }

    // 波形适应屏幕 (\)
    if (key === '\\') {
      e.preventDefault()
      actions.fitWave?.()
      return
    }

    // 画面放大 (.)
    if (key === '.') {
      e.preventDefault()
      actions.zoomInVideo?.()
      return
    }

    // 画面缩小 (,)
    if (key === ',') {
      e.preventDefault()
      actions.zoomOutVideo?.()
      return
    }

    // 画面适应窗口 (Shift + Z)
    if (isShift && key === 'z' && !isCtrl) {
      e.preventDefault()
      actions.fitVideo?.()
      return
    }

    // ==========================================
    // 第四类：字幕样式 (Alt组合)
    // ==========================================

    // 字体变大 (Alt + ])
    if (isAlt && key === ']') {
      e.preventDefault()
      actions.fontSizeUp?.()
      return
    }

    // 字体变小 (Alt + [)
    if (isAlt && key === '[') {
      e.preventDefault()
      actions.fontSizeDown?.()
      return
    }

    // ==========================================
    // 第五类：编辑操作
    // ==========================================

    // 导出 (Ctrl + E)
    if (isCtrl && key === 'e') {
      e.preventDefault() // 阻止浏览器地址栏聚焦
      actions.export?.()
      return
    }

    // 分割字幕 (Ctrl + K)
    if (isCtrl && key === 'k') {
      e.preventDefault() // 阻止浏览器搜索
      actions.splitSubtitle?.()
      return
    }

    // 合并字幕 (Ctrl + J)
    if (isCtrl && key === 'j') {
      e.preventDefault() // 阻止浏览器下载列表
      actions.mergeSubtitle?.()
      return
    }

    // 打开任务监控 (Ctrl + M)
    if (isCtrl && key === 'm') {
      e.preventDefault() // 阻止浏览器最小化
      actions.openTaskMonitor?.()
      return
    }
  }

  // 使用捕获阶段监听，避免子组件在冒泡阶段 stopPropagation 后丢失快捷键信号
  const useCapture = true

  onMounted(() => {
    window.addEventListener('keydown', handleKeydown, useCapture)
  })

  onUnmounted(() => {
    window.removeEventListener('keydown', handleKeydown, useCapture)
  })
}
