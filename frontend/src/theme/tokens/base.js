/**
 * 基础设计令牌 - 与主题无关
 * 定义圆角、阴影、过渡动画等通用设计元素
 */
export const baseTokens = {
  // 圆角
  radius: {
    xs: '3px',
    sm: '6px',
    md: '8px',
    lg: '12px',
    xl: '16px',
    full: '9999px',
  },

  // 阴影
  shadow: {
    sm: '0 1px 2px rgba(0, 0, 0, 0.4)',
    md: '0 3px 6px rgba(0, 0, 0, 0.3), 0 1px 3px rgba(0, 0, 0, 0.4)',
    lg: '0 8px 24px rgba(0, 0, 0, 0.4), 0 3px 8px rgba(0, 0, 0, 0.3)',
    xl: '0 16px 48px rgba(0, 0, 0, 0.5), 0 6px 16px rgba(0, 0, 0, 0.4)',
    glow: '0 0 16px rgba(88, 166, 255, 0.15)',
  },

  // 过渡动画
  transition: {
    fast: '0.1s ease',
    normal: '0.2s ease',
    slow: '0.3s ease',
  },

  // 字体
  font: {
    mono: "'JetBrains Mono', 'SF Mono', Monaco, Consolas, monospace",
  },

  // z-index 层级
  zIndex: {
    dropdown: 100,
    sticky: 200,
    fixed: 300,
    modalBackdrop: 400,
    modal: 500,
    popover: 600,
    tooltip: 700,
    notification: 800,
  },
}
