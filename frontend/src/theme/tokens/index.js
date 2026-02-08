/**
 * 主题注册表
 * 管理所有可用主题
 */
import { darkTheme } from './dark.js'

// 主题注册表
export const themes = {
  dark: darkTheme,
  // 预留浅色主题
  // light: lightTheme,
}

// 默认主题
export const defaultThemeId = 'dark'

/**
 * 获取主题配置
 * @param {string} id - 主题 ID
 * @returns {object} 主题配置对象
 */
export function getTheme(id) {
  return themes[id] || themes[defaultThemeId]
}

/**
 * 获取所有可用主题列表
 * @returns {Array<{id: string, name: string, displayName: string}>}
 */
export function getAvailableThemes() {
  return Object.entries(themes).map(([id, theme]) => ({
    id,
    name: theme.name,
    displayName: theme.displayName,
  }))
}
