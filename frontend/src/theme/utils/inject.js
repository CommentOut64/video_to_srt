/**
 * CSS 变量注入工具
 * 负责将 Design Tokens 转换为 CSS Variables 并注入到 DOM
 */
import { baseTokens } from '../tokens/base.js'

/**
 * 将驼峰命名转换为 kebab-case
 * @param {string} str - 驼峰命名字符串
 * @returns {string} kebab-case 字符串
 */
function toKebabCase(str) {
  return str.replace(/([a-z0-9])([A-Z])/g, '$1-$2').toLowerCase()
}

/**
 * 将嵌套对象展平为 CSS 变量格式
 * @param {object} obj - 嵌套对象
 * @param {string} prefix - 前缀
 * @returns {object} 展平后的 CSS 变量对象
 */
function flattenObject(obj, prefix = '') {
  const result = {}

  for (const [key, value] of Object.entries(obj)) {
    // 将驼峰命名转换为 kebab-case
    const kebabKey = toKebabCase(key)
    const newKey = prefix ? `${prefix}-${kebabKey}` : kebabKey

    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      Object.assign(result, flattenObject(value, newKey))
    } else {
      result[`--af-${newKey}`] = value
    }
  }

  return result
}

/**
 * 注入基础令牌（与主题无关的设计令牌）
 */
export function injectBaseTokens() {
  const vars = flattenObject(baseTokens)
  const root = document.documentElement

  for (const [key, value] of Object.entries(vars)) {
    root.style.setProperty(key, value)
  }
}

/**
 * 注入主题（包含兼容层）
 * @param {object} theme - 主题配置对象
 */
export function injectTheme(theme) {
  const root = document.documentElement

  // 注入新的 CSS 变量
  const vars = {}

  // 处理 colors
  if (theme.colors) {
    const colorVars = flattenObject(theme.colors)
    Object.assign(vars, colorVars)

    // 调试日志：打印 accent-success 相关变量
    console.log('[Theme Debug] Accent success variables:',
      Object.keys(colorVars).filter(k => k.includes('success')))
  }

  // 处理 functional
  if (theme.functional) {
    const functionalVars = flattenObject(theme.functional, 'functional')
    Object.assign(vars, functionalVars)

    // 调试日志：打印 functional 相关变量
    console.log('[Theme Debug] Functional variables generated:',
      Object.keys(functionalVars).filter(k => k.includes('processing')))
  }

  // === 兼容层：生成旧变量名映射 ===
  // 这样旧代码中的 var(--bg-primary) 仍然可以工作
  const legacyVars = generateLegacyVars(theme)
  Object.assign(vars, legacyVars)

  // 应用所有变量
  for (const [key, value] of Object.entries(vars)) {
    root.style.setProperty(key, value)
  }

  // 调试日志：验证变量是否成功注入到 DOM
  console.log('[Theme Debug] --af-functional-status-processing =',
    getComputedStyle(root).getPropertyValue('--af-functional-status-processing'))
  console.log('[Theme Debug] --af-accent-success-rgb =',
    getComputedStyle(root).getPropertyValue('--af-accent-success-rgb'))
  console.log('[Theme Debug] Total variables injected:', Object.keys(vars).length)
}

/**
 * 生成旧变量名的兼容映射
 * @param {object} theme - 主题配置
 * @returns {object} 旧变量名映射
 */
function generateLegacyVars(theme) {
  const legacy = {}

  if (!theme.colors) return legacy

  const { bg, text, accent, border } = theme.colors

  // 背景色兼容
  if (bg) {
    legacy['--bg-base'] = bg.base
    legacy['--bg-primary'] = bg.primary
    legacy['--bg-secondary'] = bg.secondary
    legacy['--bg-tertiary'] = bg.tertiary
    legacy['--bg-elevated'] = bg.elevated
    legacy['--bg-overlay'] = bg.overlay
  }

  // 文本色兼容
  if (text) {
    legacy['--text-primary'] = text.primary
    legacy['--text-normal'] = text.normal
    legacy['--text-secondary'] = text.secondary
    legacy['--text-muted'] = text.muted
    legacy['--text-disabled'] = text.disabled
  }

  // 强调色兼容
  if (accent) {
    legacy['--primary'] = accent.primary
    legacy['--primary-hover'] = accent.primaryHover
    legacy['--primary-active'] = accent.primaryActive
    legacy['--success'] = accent.success
    legacy['--success-dim'] = accent.successDim
    legacy['--warning'] = accent.warning
    legacy['--warning-dim'] = accent.warningDim
    legacy['--danger'] = accent.danger
    legacy['--danger-dim'] = accent.dangerDim
    legacy['--purple'] = accent.purple
    legacy['--pink'] = accent.pink
  }

  // 边框兼容
  if (border) {
    legacy['--border-color'] = border.default
    legacy['--border-default'] = border.default
    legacy['--border-muted'] = border.muted
    legacy['--border-subtle'] = border.subtle
  }

  return legacy
}
