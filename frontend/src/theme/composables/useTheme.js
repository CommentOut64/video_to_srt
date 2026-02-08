/**
 * 主题管理 Composable
 * 提供主题初始化、切换和持久化功能
 */
import { ref, computed } from 'vue'
import { themes, defaultThemeId, getTheme, getAvailableThemes } from '../tokens/index.js'
import { injectTheme, injectBaseTokens } from '../utils/inject.js'

const STORAGE_KEY = 'anchorflux-theme'

// 全局状态
const currentThemeId = ref(defaultThemeId)
const isInitialized = ref(false)

/**
 * 主题管理 Composable
 */
export function useTheme() {
  const currentTheme = computed(() => getTheme(currentThemeId.value))
  const isDark = computed(() => currentThemeId.value === 'dark')
  const availableThemes = computed(() => getAvailableThemes())

  /**
   * 初始化主题系统
   * 从 localStorage 恢复用户偏好，并注入 CSS 变量
   */
  function initialize() {
    if (isInitialized.value) return

    // 先注入基础令牌
    injectBaseTokens()

    // 从 localStorage 恢复主题偏好
    const savedTheme = localStorage.getItem(STORAGE_KEY)
    if (savedTheme && themes[savedTheme]) {
      currentThemeId.value = savedTheme
    }

    // 注入主题
    injectTheme(currentTheme.value)
    isInitialized.value = true
  }

  /**
   * 设置主题
   * @param {string} id - 主题 ID
   */
  function setTheme(id) {
    if (!themes[id]) {
      console.warn(`主题 "${id}" 不存在，回退到默认主题`)
      id = defaultThemeId
    }

    currentThemeId.value = id
    localStorage.setItem(STORAGE_KEY, id)
    injectTheme(getTheme(id))
  }

  /**
   * 切换深色/浅色模式
   */
  function toggleDarkMode() {
    // 当前只有深色模式，预留浅色模式接口
    setTheme(isDark.value ? 'light' : 'dark')
  }

  return {
    currentThemeId: computed(() => currentThemeId.value),
    currentTheme,
    isDark,
    availableThemes,
    isInitialized: computed(() => isInitialized.value),
    initialize,
    setTheme,
    toggleDarkMode,
  }
}
