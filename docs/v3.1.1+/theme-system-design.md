# AnchorFlux 主题系统设计方案

**版本**: V3.2.0+dev.20260130.01  
**创建日期**: 2026-01-30  
**状态**: 设计完成

---

## 1. 设计目标

| 目标 | 说明 |
|------|------|
| **一键切换主题** | 支持深色/浅色/自定义主题切换 |
| **统一管理** | 所有颜色、间距、圆角等从单一来源获取 |
| **零 SCSS 依赖** | 纯 CSS Variables + JS，不需要预处理器 |
| **类型安全** | TypeScript 支持，IDE 自动补全 |
| **运行时切换** | 无需重新构建，实时切换主题 |
| **持久化** | 记住用户偏好设置 |

---

## 2. 推荐方案：Design Tokens + CSS Variables + Vue Composable

### 2.1 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    Design Tokens (JS/TS)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  dark.ts    │  │  light.ts   │  │  custom.ts  │  ...    │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   useTheme() Composable                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • 加载主题配置                                        │   │
│  │ • 注入 CSS Variables 到 :root                        │   │
│  │ • 提供响应式 currentTheme                            │   │
│  │ • 持久化到 localStorage                              │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      CSS Variables                           │
│  :root { --af-bg-primary: #161b22; ... }                    │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              Tailwind / Components / Element Plus            │
│  class="tw-bg-bg-primary" / var(--af-bg-primary)            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 为什么这个方案最好？

| 方案 | 优点 | 缺点 |
|------|------|------|
| **SCSS 变量** | 编译时优化 | 无法运行时切换，需要重新构建 |
| **CSS-in-JS** | 动态灵活 | 运行时开销大，与 Vue 生态不兼容 |
| **纯 CSS Variables** | 运行时切换 | 缺少类型检查，容易拼写错误 |
| **✅ Design Tokens + CSS Variables** | 类型安全 + 运行时切换 | 最佳平衡 |

---

## 3. 具体实现（纯 JavaScript 版本）

### 3.1 目录结构

```
frontend/src/
├── theme/                          # 主题系统（纯 JS）
│   ├── index.js                    # 导出入口
│   ├── tokens/                     # 设计令牌
│   │   ├── base.js                 # 基础令牌（与主题无关）
│   │   ├── dark.js                 # 深色主题
│   │   ├── light.js                # 浅色主题
│   │   └── index.js                # 主题注册表
│   ├── composables/
│   │   └── useTheme.js             # 主题切换逻辑
│   └── utils/
│       └── inject.js               # CSS 变量注入工具
├── styles/
│   ├── base.css                    # 基础样式（重置、字体等）
│   ├── tailwind.css                # Tailwind 入口
│   └── element-override.css        # Element Plus 覆盖
└── main.js
```

### 3.2 基础令牌（与主题无关）

```javascript
// src/theme/tokens/base.js

/**
 * 基础设计令牌 - 与主题无关
 */
export const baseTokens = {
  radius: {
    xs: '3px',
    sm: '6px',
    md: '8px',
    lg: '12px',
    xl: '16px',
    full: '9999px',
  },
  
  shadow: {
    sm: '0 1px 2px rgba(0, 0, 0, 0.4)',
    md: '0 3px 6px rgba(0, 0, 0, 0.3), 0 1px 3px rgba(0, 0, 0, 0.4)',
    lg: '0 8px 24px rgba(0, 0, 0, 0.4), 0 3px 8px rgba(0, 0, 0, 0.3)',
    xl: '0 16px 48px rgba(0, 0, 0, 0.5), 0 6px 16px rgba(0, 0, 0, 0.4)',
    glow: '0 0 16px rgba(88, 166, 255, 0.15)',
  },
  
  transition: {
    fast: '0.1s ease',
    normal: '0.2s ease',
    slow: '0.3s ease',
  },
  
  font: {
    mono: "'JetBrains Mono', 'SF Mono', Monaco, Consolas, monospace",
  },
  
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
```

### 3.3 深色主题

```javascript
// src/theme/tokens/dark.js

/**
 * 深色主题 - GitHub Dark Dimmed 风格
 */
export const darkTheme = {
  name: 'dark',
  displayName: '深色模式',
  
  colors: {
    bg: {
      base: '#0d1117',
      primary: '#161b22',
      secondary: '#21262d',
      tertiary: '#30363d',
      elevated: '#2d333b',
      overlay: 'rgba(0, 0, 0, 0.5)',
    },
    
    text: {
      primary: '#e6edf3',
      normal: '#c9d1d9',
      secondary: '#8b949e',
      muted: '#6e7681',
      disabled: '#484f58',
    },
    
    accent: {
      primary: '#58a6ff',
      primaryHover: '#79c0ff',
      primaryActive: '#388bfd',
      primaryRgb: '88, 166, 255',
      success: '#3fb950',
      successDim: '#238636',
      successRgb: '63, 185, 80',
      warning: '#d29922',
      warningDim: '#9e6a03',
      warningRgb: '210, 153, 34',
      danger: '#f85149',
      dangerDim: '#da3633',
      dangerRgb: '248, 81, 73',
      purple: '#a371f7',
      pink: '#db61a2',
    },
    
    border: {
      default: '#30363d',
      muted: '#21262d',
      subtle: 'rgba(240, 246, 252, 0.1)',
    },
  },
  
  functional: {
    stage: {
      sensevoice: '#58a6ff',
      whisper: '#3fb950',
      llm: '#a371f7',
    },
    
    status: {
      processing: '#028AC5',
      warning: '#e67700',
      error: '#f85149',
    },
    
    waveform: {
      color: '#58a6ff',
      progress: '#238636',
      cursor: '#f85149',
      regionDefault: 'rgba(88, 166, 255, 0.25)',
      regionActive: 'rgba(88, 166, 255, 0.4)',
      regionSelected: 'rgba(163, 113, 247, 0.35)',
    },
    
    video: {
      bg: '#000000',
      text: '#ffffff',
      overlay: 'rgba(0, 0, 0, 0.6)',
    },
  },
}
```

### 3.4 浅色主题

```javascript
// src/theme/tokens/light.js

/**
 * 浅色主题 - GitHub Light 风格
 */
export const lightTheme = {
  name: 'light',
  displayName: '浅色模式',
  
  colors: {
    bg: {
      base: '#ffffff',
      primary: '#f6f8fa',
      secondary: '#ffffff',
      tertiary: '#f3f4f6',
      elevated: '#ffffff',
      overlay: 'rgba(0, 0, 0, 0.3)',
    },
    
    text: {
      primary: '#1f2328',
      normal: '#24292f',
      secondary: '#57606a',
      muted: '#6e7781',
      disabled: '#8c959f',
    },
    
    accent: {
      primary: '#0969da',
      primaryHover: '#0860ca',
      primaryActive: '#0550ae',
      primaryRgb: '9, 105, 218',
      success: '#1a7f37',
      successDim: '#2da44e',
      successRgb: '26, 127, 55',
      warning: '#9a6700',
      warningDim: '#bf8700',
      warningRgb: '154, 103, 0',
      danger: '#cf222e',
      dangerDim: '#a40e26',
      dangerRgb: '207, 34, 46',
      purple: '#8250df',
      pink: '#bf3989',
    },
    
    border: {
      default: '#d0d7de',
      muted: '#d8dee4',
      subtle: 'rgba(31, 35, 40, 0.1)',
    },
  },
  
  functional: {
    stage: {
      sensevoice: '#0969da',
      whisper: '#1a7f37',
      llm: '#8250df',
    },
    
    status: {
      processing: '#0969da',
      warning: '#9a6700',
      error: '#cf222e',
    },
    
    waveform: {
      color: '#0969da',
      progress: '#1a7f37',
      cursor: '#cf222e',
      regionDefault: 'rgba(9, 105, 218, 0.15)',
      regionActive: 'rgba(9, 105, 218, 0.25)',
      regionSelected: 'rgba(130, 80, 223, 0.2)',
    },
    
    video: {
      bg: '#000000',
      text: '#ffffff',
      overlay: 'rgba(0, 0, 0, 0.6)',
    },
  },
}
```

### 3.5 主题注册表

```javascript
// src/theme/tokens/index.js

import { darkTheme } from './dark'
import { lightTheme } from './light'

/**
 * 主题注册表
 */
export const themes = {
  dark: darkTheme,
  light: lightTheme,
}

/**
 * 默认主题
 */
export const defaultThemeId = 'dark'

/**
 * 获取主题配置
 */
export function getTheme(id) {
  return themes[id] ?? themes[defaultThemeId]
}

/**
 * 获取所有可用主题
 */
export function getAvailableThemes() {
  return Object.entries(themes).map(([id, config]) => ({
    id,
    displayName: config.displayName,
  }))
}
```

### 3.6 CSS 变量注入工具

```javascript
// src/theme/utils/inject.js

import { baseTokens } from '../tokens/base'

/**
 * 将主题配置转换为 CSS 变量并注入到 :root
 */
export function injectTheme(theme) {
  const root = document.documentElement
  const variables = generateCssVariables(theme)
  
  // 注入颜色变量
  for (const [key, value] of Object.entries(variables)) {
    root.style.setProperty(key, value)
  }
  
  // 设置主题标识（用于条件样式）
  root.dataset.theme = theme.name
}

/**
 * 注入基础令牌（只需要调用一次）
 */
export function injectBaseTokens() {
  const root = document.documentElement
  const variables = generateBaseVariables(baseTokens)
  
  for (const [key, value] of Object.entries(variables)) {
    root.style.setProperty(key, value)
  }
}

/**
 * 生成颜色相关的 CSS 变量
 */
function generateCssVariables(theme) {
  const prefix = '--af'
  const vars = {}
  
  // 背景色
  const { bg } = theme.colors
  vars[`${prefix}-bg-base`] = bg.base
  vars[`${prefix}-bg-primary`] = bg.primary
  vars[`${prefix}-bg-secondary`] = bg.secondary
  vars[`${prefix}-bg-tertiary`] = bg.tertiary
  vars[`${prefix}-bg-elevated`] = bg.elevated
  vars[`${prefix}-bg-overlay`] = bg.overlay
  
  // 文本色
  const { text } = theme.colors
  vars[`${prefix}-text-primary`] = text.primary
  vars[`${prefix}-text-normal`] = text.normal
  vars[`${prefix}-text-secondary`] = text.secondary
  vars[`${prefix}-text-muted`] = text.muted
  vars[`${prefix}-text-disabled`] = text.disabled
  
  // 强调色
  const { accent } = theme.colors
  vars[`${prefix}-accent-primary`] = accent.primary
  vars[`${prefix}-accent-primary-hover`] = accent.primaryHover
  vars[`${prefix}-accent-primary-active`] = accent.primaryActive
  vars[`${prefix}-accent-primary-rgb`] = accent.primaryRgb
  vars[`${prefix}-accent-success`] = accent.success
  vars[`${prefix}-accent-success-dim`] = accent.successDim
  vars[`${prefix}-accent-success-rgb`] = accent.successRgb
  vars[`${prefix}-accent-warning`] = accent.warning
  vars[`${prefix}-accent-warning-dim`] = accent.warningDim
  vars[`${prefix}-accent-warning-rgb`] = accent.warningRgb
  vars[`${prefix}-accent-danger`] = accent.danger
  vars[`${prefix}-accent-danger-dim`] = accent.dangerDim
  vars[`${prefix}-accent-danger-rgb`] = accent.dangerRgb
  vars[`${prefix}-accent-purple`] = accent.purple
  vars[`${prefix}-accent-pink`] = accent.pink
  
  // 边框
  const { border } = theme.colors
  vars[`${prefix}-border-default`] = border.default
  vars[`${prefix}-border-muted`] = border.muted
  vars[`${prefix}-border-subtle`] = border.subtle
  
  // 功能色 - 阶段
  const { stage } = theme.functional
  vars[`${prefix}-stage-sensevoice`] = stage.sensevoice
  vars[`${prefix}-stage-whisper`] = stage.whisper
  vars[`${prefix}-stage-llm`] = stage.llm
  
  // 功能色 - 状态
  const { status } = theme.functional
  vars[`${prefix}-status-processing`] = status.processing
  vars[`${prefix}-status-warning`] = status.warning
  vars[`${prefix}-status-error`] = status.error
  
  // 波形
  const { waveform } = theme.functional
  vars[`${prefix}-waveform-color`] = waveform.color
  vars[`${prefix}-waveform-progress`] = waveform.progress
  vars[`${prefix}-waveform-cursor`] = waveform.cursor
  vars[`${prefix}-region-default`] = waveform.regionDefault
  vars[`${prefix}-region-active`] = waveform.regionActive
  vars[`${prefix}-region-selected`] = waveform.regionSelected
  
  // 视频
  const { video } = theme.functional
  vars[`${prefix}-video-bg`] = video.bg
  vars[`${prefix}-video-text`] = video.text
  vars[`${prefix}-video-overlay`] = video.overlay
  
  // ============================================
  // 兼容层 - 旧变量名映射（迁移期间使用）
  // ============================================
  vars['--bg-base'] = bg.base
  vars['--bg-primary'] = bg.primary
  vars['--bg-secondary'] = bg.secondary
  vars['--bg-tertiary'] = bg.tertiary
  vars['--bg-elevated'] = bg.elevated
  vars['--bg-overlay'] = bg.overlay
  
  vars['--text-primary'] = text.primary
  vars['--text-normal'] = text.normal
  vars['--text-secondary'] = text.secondary
  vars['--text-muted'] = text.muted
  vars['--text-disabled'] = text.disabled
  vars['--text-bright'] = text.primary
  
  vars['--primary'] = accent.primary
  vars['--primary-hover'] = accent.primaryHover
  vars['--primary-dim'] = accent.primaryActive
  vars['--primary-rgb'] = accent.primaryRgb
  vars['--accent-blue'] = accent.primary
  vars['--accent-green'] = accent.success
  vars['--accent-orange'] = accent.warning
  vars['--accent-red'] = accent.danger
  
  vars['--success'] = accent.success
  vars['--success-dim'] = accent.successDim
  vars['--warning'] = accent.warning
  vars['--warning-dim'] = accent.warningDim
  vars['--danger'] = accent.danger
  vars['--danger-dim'] = accent.dangerDim
  vars['--purple'] = accent.purple
  vars['--pink'] = accent.pink
  
  vars['--border-default'] = border.default
  vars['--border-muted'] = border.muted
  vars['--border-subtle'] = border.subtle
  vars['--border-color'] = border.default
  
  vars['--waveform-color'] = waveform.color
  vars['--waveform-progress'] = waveform.progress
  vars['--waveform-cursor'] = waveform.cursor
  vars['--region-default'] = waveform.regionDefault
  vars['--region-active'] = waveform.regionActive
  vars['--region-selected'] = waveform.regionSelected
  
  return vars
}

/**
 * 生成基础令牌的 CSS 变量
 */
function generateBaseVariables(tokens) {
  const prefix = '--af'
  const vars = {}
  
  // 圆角
  vars[`${prefix}-radius-xs`] = tokens.radius.xs
  vars[`${prefix}-radius-sm`] = tokens.radius.sm
  vars[`${prefix}-radius-md`] = tokens.radius.md
  vars[`${prefix}-radius-lg`] = tokens.radius.lg
  vars[`${prefix}-radius-xl`] = tokens.radius.xl
  vars[`${prefix}-radius-full`] = tokens.radius.full
  
  // 阴影
  vars[`${prefix}-shadow-sm`] = tokens.shadow.sm
  vars[`${prefix}-shadow-md`] = tokens.shadow.md
  vars[`${prefix}-shadow-lg`] = tokens.shadow.lg
  vars[`${prefix}-shadow-xl`] = tokens.shadow.xl
  vars[`${prefix}-shadow-glow`] = tokens.shadow.glow
  
  // 过渡
  vars[`${prefix}-transition-fast`] = tokens.transition.fast
  vars[`${prefix}-transition-normal`] = tokens.transition.normal
  vars[`${prefix}-transition-slow`] = tokens.transition.slow
  
  // 字体
  vars[`${prefix}-font-mono`] = tokens.font.mono
  
  // Z-Index
  vars[`${prefix}-z-dropdown`] = String(tokens.zIndex.dropdown)
  vars[`${prefix}-z-sticky`] = String(tokens.zIndex.sticky)
  vars[`${prefix}-z-fixed`] = String(tokens.zIndex.fixed)
  vars[`${prefix}-z-modal-backdrop`] = String(tokens.zIndex.modalBackdrop)
  vars[`${prefix}-z-modal`] = String(tokens.zIndex.modal)
  vars[`${prefix}-z-popover`] = String(tokens.zIndex.popover)
  vars[`${prefix}-z-tooltip`] = String(tokens.zIndex.tooltip)
  vars[`${prefix}-z-notification`] = String(tokens.zIndex.notification)
  
  // 兼容层
  vars['--radius-xs'] = tokens.radius.xs
  vars['--radius-sm'] = tokens.radius.sm
  vars['--radius-md'] = tokens.radius.md
  vars['--radius-lg'] = tokens.radius.lg
  vars['--radius-xl'] = tokens.radius.xl
  vars['--radius-full'] = tokens.radius.full
  
  vars['--shadow-sm'] = tokens.shadow.sm
  vars['--shadow-md'] = tokens.shadow.md
  vars['--shadow-lg'] = tokens.shadow.lg
  vars['--shadow-xl'] = tokens.shadow.xl
  vars['--shadow-glow'] = tokens.shadow.glow
  
  vars['--transition-fast'] = tokens.transition.fast
  vars['--transition-normal'] = tokens.transition.normal
  vars['--transition-slow'] = tokens.transition.slow
  
  vars['--font-mono'] = tokens.font.mono
  
  return vars
}
```

### 3.7 useTheme Composable

```javascript
// src/theme/composables/useTheme.js

import { ref, computed } from 'vue'
import { themes, defaultThemeId, getTheme, getAvailableThemes } from '../tokens'
import { injectTheme, injectBaseTokens } from '../utils/inject'

const STORAGE_KEY = 'anchorflux-theme'

// 全局状态
const currentThemeId = ref(defaultThemeId)
const isInitialized = ref(false)

/**
 * 主题管理 Composable
 */
export function useTheme() {
  // 当前主题配置
  const currentTheme = computed(() => getTheme(currentThemeId.value))
  
  // 是否是深色主题
  const isDark = computed(() => currentThemeId.value === 'dark')
  
  // 可用主题列表
  const availableThemes = computed(() => getAvailableThemes())
  
  /**
   * 初始化主题系统
   */
  function initialize() {
    if (isInitialized.value) return
    
    // 注入基础令牌
    injectBaseTokens()
    
    // 从 localStorage 恢复主题偏好
    const savedTheme = localStorage.getItem(STORAGE_KEY)
    if (savedTheme && themes[savedTheme]) {
      currentThemeId.value = savedTheme
    }
    
    // 注入当前主题
    injectTheme(currentTheme.value)
    
    isInitialized.value = true
  }
  
  /**
   * 切换主题
   */
  function setTheme(id) {
    if (!themes[id]) {
      console.warn(`Theme "${id}" not found, falling back to default`)
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
    setTheme(isDark.value ? 'light' : 'dark')
  }
  
  /**
   * 注册自定义主题
   */
  function registerTheme(id, config) {
    themes[id] = config
  }
  
  return {
    // 状态
    currentThemeId: computed(() => currentThemeId.value),
    currentTheme,
    isDark,
    availableThemes,
    isInitialized: computed(() => isInitialized.value),
    
    // 方法
    initialize,
    setTheme,
    toggleDarkMode,
    registerTheme,
  }
}
```

### 3.8 主入口

```javascript
// src/theme/index.js

export * from './tokens'
export { useTheme } from './composables/useTheme'
export { injectTheme, injectBaseTokens } from './utils/inject'
```

### 3.9 添加新主题

```javascript
// src/theme/tokens/solarized.js

/**
 * Solarized 主题示例
 */
export const solarizedTheme = {
  name: 'solarized',
  displayName: 'Solarized Dark',
  
  colors: {
    bg: {
      base: '#002b36',
      primary: '#073642',
      secondary: '#073642',
      tertiary: '#586e75',
      elevated: '#073642',
      overlay: 'rgba(0, 0, 0, 0.5)',
    },
    
    text: {
      primary: '#fdf6e3',
      normal: '#eee8d5',
      secondary: '#93a1a1',
      muted: '#839496',
      disabled: '#586e75',
    },
    
    accent: {
      primary: '#268bd2',
      primaryHover: '#2aa198',
      primaryActive: '#2075b8',
      primaryRgb: '38, 139, 210',
      success: '#859900',
      successDim: '#6b7a00',
      successRgb: '133, 153, 0',
      warning: '#b58900',
      warningDim: '#8b6900',
      warningRgb: '181, 137, 0',
      danger: '#dc322f',
      dangerDim: '#b32826',
      dangerRgb: '220, 50, 47',
      purple: '#6c71c4',
      pink: '#d33682',
    },
    
    border: {
      default: '#586e75',
      muted: '#073642',
      subtle: 'rgba(253, 246, 227, 0.1)',
    },
  },
  
  functional: {
    stage: {
      sensevoice: '#268bd2',
      whisper: '#859900',
      llm: '#6c71c4',
    },
    
    status: {
      processing: '#268bd2',
      warning: '#b58900',
      error: '#dc322f',
    },
    
    waveform: {
      color: '#268bd2',
      progress: '#859900',
      cursor: '#dc322f',
      regionDefault: 'rgba(38, 139, 210, 0.25)',
      regionActive: 'rgba(38, 139, 210, 0.4)',
      regionSelected: 'rgba(108, 113, 196, 0.35)',
    },
    
    video: {
      bg: '#000000',
      text: '#ffffff',
      overlay: 'rgba(0, 0, 0, 0.6)',
    },
  },
}

// 在 tokens/index.js 中注册
// import { solarizedTheme } from './solarized'
// themes.solarized = solarizedTheme
```

---

```javascript
// src/main.js

import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import 'element-plus/dist/index.css'

// 导入基础样式（不再需要 SCSS）
import './styles/base.css'
import './styles/element-override.css'

// 导入主题系统
import { useTheme } from './theme'

import App from './App.vue'
import router from './router'

// 创建应用实例
const app = createApp(App)

// 注册 Pinia
const pinia = createPinia()
app.use(pinia)

// 注册路由
app.use(router)

// 注册 Element Plus
app.use(ElementPlus)

// 注册 Element Plus 图标
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

// ✅ 初始化主题系统
const { initialize } = useTheme()
initialize()

// 挂载应用
app.mount('#app')
```

---

## 4. 使用示例

### 4.1 在组件中使用

```vue
<script setup>
import { useTheme } from '@/theme'

const { currentTheme, isDark, availableThemes, setTheme, toggleDarkMode } = useTheme()
</script>

<template>
  <div class="tw-flex tw-items-center tw-gap-2">
    <!-- 主题切换按钮 -->
    <button @click="toggleDarkMode" class="theme-toggle-btn">
      {{ isDark ? '🌙' : '☀️' }}
    </button>
    
    <!-- 主题选择下拉 -->
    <select 
      :value="currentTheme.name" 
      @change="setTheme($event.target.value)"
      class="tw-bg-bg-secondary tw-text-text-normal tw-rounded-md tw-px-3 tw-py-1"
    >
      <option 
        v-for="theme in availableThemes" 
        :key="theme.id" 
        :value="theme.id"
      >
        {{ theme.displayName }}
      </option>
    </select>
  </div>
</template>
```

### 4.2 CSS 中使用变量

```css
.my-component {
  /* 使用新命名 */
  background: var(--af-bg-primary);
  color: var(--af-text-normal);
  border: 1px solid var(--af-border-default);
  border-radius: var(--af-radius-md);
  
  /* 带透明度的颜色 */
  box-shadow: 0 0 20px rgba(var(--af-accent-primary-rgb), 0.2);
}

.my-component:hover {
  background: var(--af-bg-secondary);
}
```

### 4.3 Tailwind 中使用

```vue
<template>
  <!-- Tailwind 类会自动使用 CSS 变量 -->
  <div class="tw-bg-bg-primary tw-text-text-normal tw-border-border tw-rounded-md">
    <span class="tw-text-accent-primary">高亮文本</span>
  </div>
</template>
```

### 4.4 添加新主题

```javascript
// src/theme/tokens/solarized.js
export const solarizedTheme = {
  name: 'solarized',
  displayName: 'Solarized Dark',
  colors: {
    bg: { base: '#002b36', primary: '#073642', ... },
    // ...
  },
  // ...
}

// 在 tokens/index.js 中注册
import { solarizedTheme } from './solarized'

export const themes = {
  dark: darkTheme,
  light: lightTheme,
  solarized: solarizedTheme,  // 新增
}
```

---

## 5. Element Plus 主题适配

```css
/* src/styles/element-override.css */

/* Element Plus 通过 CSS 变量自动适配主题 */
:root {
  /* 按钮 */
  --el-button-bg-color: var(--af-bg-secondary);
  --el-button-border-color: var(--af-border-default);
  --el-button-text-color: var(--af-text-normal);
  --el-button-hover-bg-color: var(--af-bg-tertiary);
  --el-button-hover-border-color: var(--af-border-default);
  --el-button-hover-text-color: var(--af-text-primary);
  
  /* 输入框 */
  --el-input-bg-color: var(--af-bg-secondary);
  --el-input-border-color: var(--af-border-default);
  --el-input-text-color: var(--af-text-normal);
  --el-input-placeholder-color: var(--af-text-muted);
  
  /* 对话框 */
  --el-dialog-bg-color: var(--af-bg-secondary);
  --el-dialog-border-radius: var(--af-radius-lg);
  
  /* 卡片 */
  --el-card-bg-color: var(--af-bg-secondary);
  --el-card-border-color: var(--af-border-default);
  
  /* 主色 */
  --el-color-primary: var(--af-accent-primary);
  --el-color-success: var(--af-accent-success);
  --el-color-warning: var(--af-accent-warning);
  --el-color-danger: var(--af-accent-danger);
  
  /* 文本色 */
  --el-text-color-primary: var(--af-text-normal);
  --el-text-color-regular: var(--af-text-secondary);
  --el-text-color-secondary: var(--af-text-muted);
  --el-text-color-placeholder: var(--af-text-disabled);
  
  /* 边框色 */
  --el-border-color: var(--af-border-default);
  --el-border-color-light: var(--af-border-muted);
  --el-border-color-lighter: var(--af-border-subtle);
  
  /* 背景色 */
  --el-bg-color: var(--af-bg-primary);
  --el-bg-color-overlay: var(--af-bg-overlay);
}
```

---

## 6. 迁移路径

### 6.1 阶段一：并行运行（1-2h）

1. 创建 `src/theme/` 目录结构
2. 实现主题系统代码
3. 在 `main.js` 中初始化
4. 保留原有 SCSS 文件（兼容层自动映射）

### 6.2 阶段二：渐进迁移（按需）

1. 新组件使用 `--af-*` 变量
2. 修改组件时顺便更新变量名
3. 无需一次性迁移所有文件

### 6.3 阶段三：移除 SCSS（可选）

1. 所有组件使用新变量后
2. 删除 `_variables.scss` 和 `_mixins.scss`
3. 移除 `package.json` 中的 `sass` 依赖

---

## 7. 总结

| 特性 | 旧方案 (SCSS) | 新方案 (Design Tokens) |
|------|---------------|------------------------|
| **运行时切换** | ❌ 需要重新构建 | ✅ 即时切换 |
| **类型安全** | ❌ 无 | ⚪ 可选（用 JSDoc 注释） |
| **IDE 补全** | ⚠️ 部分 | ✅ 配合 JSDoc 完整 |
| **新增主题** | ❌ 需要新建 SCSS 文件 | ✅ 添加 JS 对象即可 |
| **持久化** | ❌ 无 | ✅ localStorage |
| **构建依赖** | 需要 sass | ✅ 纯 JS |
| **兼容性** | 无限制 | 无限制（CSS Variables） |

**这个方案完美满足您的需求：**
- ✅ 一键切换主题
- ✅ 统一管理所有界面颜色
- ✅ 方便添加新主题
- ✅ 所有界面/窗口统一使用
- ✅ 无需 SCSS

---

**文档结束**
