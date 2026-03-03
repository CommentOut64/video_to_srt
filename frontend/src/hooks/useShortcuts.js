import { onMounted, onUnmounted } from 'vue'
import {
  DEFAULT_EDITOR_SHORTCUT_CONFIG,
  normalizeEditorShortcutConfig,
  resolveEditorShortcutAction,
} from '@/utils/editorShortcuts'

const ACTIONS_ALLOWED_IN_INPUT = new Set(['save', 'undo', 'redo'])

function resolveOptionValue(rawOption, fallbackValue) {
  if (rawOption && typeof rawOption === 'object' && 'value' in rawOption) {
    return rawOption.value
  }
  return rawOption ?? fallbackValue
}

/**
 * 快捷键管理 Hook（核心可配置动作）
 * @param {Object} actions - 快捷键回调表
 * @param {Object} options - 运行时选项
 * @param {boolean|Ref<boolean>} options.isShortcutEnabled - 是否启用快捷键
 * @param {Object|Ref<Object>} options.shortcutConfig - 快捷键映射配置
 */
export function useShortcuts(actions, options = {}) {
  const handleKeydown = (event) => {
    if (event.defaultPrevented) return

    const isShortcutEnabled = resolveOptionValue(options.isShortcutEnabled, true)
    if (isShortcutEnabled === false) return

    const shortcutConfig = normalizeEditorShortcutConfig(
      resolveOptionValue(options.shortcutConfig, DEFAULT_EDITOR_SHORTCUT_CONFIG)
    )
    const actionKey = resolveEditorShortcutAction(event, shortcutConfig)
    if (!actionKey) return

    const target = event.target
    const tagName = target?.tagName
    const isInputActive = (
      tagName === 'INPUT'
      || tagName === 'TEXTAREA'
      || target?.isContentEditable
    )
    if (isInputActive && !ACTIONS_ALLOWED_IN_INPUT.has(actionKey)) {
      return
    }

    if (actionKey === 'togglePlay' && event.repeat) {
      return
    }

    event.preventDefault()
    actions[actionKey]?.()
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
