# AnchorFlux 前端样式系统重构开发文档

**版本**: V3.2.0+dev.20260130.01  
**创建日期**: 2026-01-30  
**状态**: 执行中

---

## 目录

1. [核心原则](#1-核心原则)
2. [技术架构](#2-技术架构)
3. [开发规范](#3-开发规范)
4. [主题系统](#4-主题系统)
5. [分阶段开发计划](#5-分阶段开发计划)
6. [组件重构指南](#6-组件重构指南)
7. [组件最终审核规则](#7-组件最终审核规则-)
8. [验收标准](#8-验收标准)
9. [风险与缓解](#9-风险与缓解)
10. [附录](#10-附录)

---

## 1. 核心原则

### 1.1 不可违反的红线 🚫

| 原则 | 说明 | 检查点 |
|------|------|--------|
| **纯 JavaScript** | 完全不使用 TypeScript，所有代码用 .js 文件 | 无 .ts/.tsx 文件 |
| **只动前端** | 绝对不修改后端任何代码 | 无 backend/ 目录变更 |
| **逐步放弃 SCSS** | 不再新增 SCSS 代码，逐渐迁移现有代码 | 新代码只用 CSS/Tailwind |
| **保持功能完整** | 重构不能破坏任何现有功能 | 全功能回归测试 |
| **谨慎简化** | 简化代码时必须完全理解原逻辑 | 每次改动有明确理由 |

### 1.2 重构安全准则 ⚠️

```
✅ 正确做法：
  1. 先充分理解组件的现有逻辑
  2. 单个文件小步迭代，每次改动可测试
  3. 保留原有的类名结构，逐步替换样式实现
  4. 添加兼容层，确保旧变量名仍然工作
  5. 每个阶段完成后进行完整功能测试

❌ 错误做法：
  1. 大规模重写组件逻辑
  2. 删除"看起来没用"的代码
  3. 同时修改多个相互依赖的组件
  4. 跳过测试直接进入下一阶段
  5. 修改后端 API 或数据结构
```

### 1.3 最小改动原则

对于每个组件，优先采用"样式层替换"而非"结构重写"：

```vue
<!-- ✅ 推荐：保留结构，替换样式实现 -->
<template>
  <!-- 保持原有的 class 名称，只改实现方式 -->
  <div class="editor-header tw-flex tw-items-center tw-gap-4">
    ...
  </div>
</template>

<!-- ❌ 避免：大幅修改模板结构 -->
<template>
  <!-- 不要为了"更好的结构"而重写 -->
  <header class="new-header">
    ...
  </header>
</template>
```

---

## 2. 技术架构

### 2.1 技术栈

| 层级 | 当前 | 目标 | 迁移策略 |
|------|------|------|----------|
| **布局** | SCSS 手写 | Tailwind CSS | 逐步替换 |
| **主题** | SCSS 变量 + CSS 变量 | 纯 JS Design Tokens | 新建系统 + 兼容层 |
| **组件库** | Element Plus | Element Plus | 不变 |
| **颜色** | 混合使用 | CSS Variables 统一 | 全面迁移 |
| **预处理器** | SCSS | 不使用 | 逐步移除 |

### 2.2 目录结构

重构后的前端样式目录：

```
frontend/src/
├── theme/                          # 主题系统（纯 JS）
│   ├── index.js                    # 导出入口
│   ├── tokens/                     # 设计令牌
│   │   ├── base.js                 # 基础令牌（圆角、阴影等）
│   │   ├── dark.js                 # 深色主题
│   │   ├── light.js                # 浅色主题（预留）
│   │   └── index.js                # 主题注册表
│   ├── composables/
│   │   └── useTheme.js             # 主题切换逻辑
│   └── utils/
│       └── inject.js               # CSS 变量注入
├── styles/
│   ├── base.css                    # 基础样式（重置、字体等）
│   ├── tailwind.css                # Tailwind 入口
│   ├── element-override.css        # Element Plus 样式覆盖
│   ├── utilities.css               # 工具类（过渡期保留）
│   └── legacy/                     # [过渡期] 旧 SCSS 文件
│       ├── _variables.scss         # 旧变量（保留供参考）
│       └── _mixins.scss            # 旧 mixin（保留供参考）
└── main.js
```

### 2.3 职责边界

```
┌─────────────────────────────────────────────────────────────┐
│                    Design Tokens (JS)                        │
│  定义所有颜色、间距、圆角等设计令牌                           │
│  运行时注入 CSS Variables 到 :root                          │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     CSS Variables                            │
│  所有颜色值、动态主题切换                                     │
│  命名规范: --af-{category}-{name}[-{variant}]               │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     Tailwind CSS                             │
│  布局（flex/grid）、间距、尺寸、基础交互                      │
│  前缀: tw-                                                   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              Scoped CSS (仅用于特殊场景)                     │
│  复杂动画、波形图交互、Element Plus 覆盖                     │
│  必须添加注释说明为什么不能用 Tailwind                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 开发规范

### 3.1 样式优先级规则

```
1️⃣ Tailwind 工具类 (最高优先级)
   └─ 布局、间距、尺寸、基础状态
   └─ 示例: tw-flex tw-gap-4 tw-p-4 tw-rounded-md

2️⃣ CSS Variables (通过 Tailwind 或直接使用)
   └─ 所有颜色、主题相关值
   └─ 示例: tw-bg-bg-primary 或 var(--af-bg-primary)

3️⃣ Scoped CSS (最低优先级，仅限特殊场景)
   └─ 复杂动画、:deep() 穿透、复杂选择器
   └─ 必须添加注释说明原因
```

### 3.2 代码示例

#### ✅ 正确写法

```vue
<template>
  <div class="subtitle-item tw-flex tw-items-center tw-gap-4 tw-p-3 
              tw-bg-bg-secondary tw-rounded-md tw-border tw-border-border
              hover:tw-bg-bg-tertiary tw-transition-colors tw-duration-normal">
    <span class="tw-text-text-normal">{{ subtitle.text }}</span>
    <button class="action-btn tw-text-accent-primary hover:tw-text-accent-primary/80">
      编辑
    </button>
  </div>
</template>

<style scoped>
/* 
 * 仅保留无法用 Tailwind 实现的样式
 * 原因：需要复杂的条件动画
 */
.subtitle-item.is-editing {
  animation: editing-pulse 2s infinite;
}

@keyframes editing-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(var(--af-accent-primary-rgb), 0.4); }
  50% { box-shadow: 0 0 0 4px rgba(var(--af-accent-primary-rgb), 0); }
}
</style>
```

#### ❌ 错误写法

```vue
<template>
  <div class="subtitle-item">...</div>
</template>

<style lang="scss" scoped>
.subtitle-item {
  /* ❌ 这些应该用 Tailwind */
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 12px;
  border-radius: 8px;
  
  /* ❌ 硬编码颜色 */
  background: #21262d;
  color: #c9d1d9;
  
  &:hover {
    background: #30363d;
  }
}
</style>
```

### 3.3 命名规范

#### CSS Variables

```
格式: --af-{category}-{name}[-{variant}]

category:
  bg       - 背景色
  text     - 文本色
  accent   - 强调色
  border   - 边框
  shadow   - 阴影
  radius   - 圆角
  stage    - 阶段标记
  status   - 状态
  waveform - 波形专用
  video    - 视频专用

示例:
  --af-bg-primary
  --af-text-muted
  --af-accent-primary-hover
  --af-stage-whisper
```

#### CSS 类名（保留原有 BEM 风格）

```
保留原有类名，添加 Tailwind 类：

原来: <div class="subtitle-item">
现在: <div class="subtitle-item tw-flex tw-gap-4">

注意：不要删除原有类名，它们可能被 JS 引用或用于测试
```

### 3.4 禁止清单 🚫

| 类别 | 禁止用法 | 替代方案 |
|------|----------|----------|
| **硬编码颜色** | `#f85149`, `rgb(...)` | `var(--af-accent-danger)` |
| **SCSS 变量** | `$primary-color` | `var(--af-accent-primary)` |
| **新增 SCSS** | `lang="scss"` | `scoped` 或 Tailwind |
| **手写布局** | `display: flex; gap: 8px;` | `tw-flex tw-gap-2` |
| **!important** | `color: red !important;` | 提高选择器优先级 |
| **滥用 :deep()** | 大量穿透 Element Plus | 使用官方 CSS 变量 |

### 3.5 Element Plus 覆盖规范

```css
/* src/styles/element-override.css */

/* ✅ 正确：使用 Element Plus 官方 CSS 变量 */
:root {
  --el-color-primary: var(--af-accent-primary);
  --el-color-success: var(--af-accent-success);
  --el-color-warning: var(--af-accent-warning);
  --el-color-danger: var(--af-accent-danger);
  
  --el-bg-color: var(--af-bg-primary);
  --el-text-color-primary: var(--af-text-normal);
  --el-border-color: var(--af-border-default);
  
  --el-button-bg-color: var(--af-bg-secondary);
  --el-input-bg-color: var(--af-bg-secondary);
  --el-dialog-bg-color: var(--af-bg-secondary);
}

/* ⚠️ 仅在必要时使用选择器覆盖，并添加注释 */
/* 原因：Element Plus 不提供对应的 CSS 变量 */
.el-dialog {
  --el-dialog-border-radius: var(--af-radius-lg);
}
```

---

## 4. 主题系统

### 4.1 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    Design Tokens (JS)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  dark.js    │  │  light.js   │  │  custom.js  │  ...    │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   useTheme() Composable                      │
│  • 初始化主题  • 注入 CSS Variables  • 持久化偏好           │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                :root { --af-*: ... }                         │
│                    运行时 CSS Variables                      │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 核心文件

#### 4.2.1 基础令牌

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

#### 4.2.2 深色主题

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

#### 4.2.3 useTheme Composable

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
  const currentTheme = computed(() => getTheme(currentThemeId.value))
  const isDark = computed(() => currentThemeId.value === 'dark')
  const availableThemes = computed(() => getAvailableThemes())
  
  function initialize() {
    if (isInitialized.value) return
    
    injectBaseTokens()
    
    const savedTheme = localStorage.getItem(STORAGE_KEY)
    if (savedTheme && themes[savedTheme]) {
      currentThemeId.value = savedTheme
    }
    
    injectTheme(currentTheme.value)
    isInitialized.value = true
  }
  
  function setTheme(id) {
    if (!themes[id]) {
      console.warn(`Theme "${id}" not found, falling back to default`)
      id = defaultThemeId
    }
    
    currentThemeId.value = id
    localStorage.setItem(STORAGE_KEY, id)
    injectTheme(getTheme(id))
  }
  
  function toggleDarkMode() {
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
```

### 4.3 兼容层

为确保渐进式迁移，主题系统会自动生成旧变量名的映射：

```javascript
// src/theme/utils/inject.js 中的兼容层

// 新变量 → 旧变量映射（迁移期间自动生成）
vars['--bg-base'] = bg.base           // 兼容旧代码
vars['--bg-primary'] = bg.primary
vars['--text-primary'] = text.primary
vars['--text-normal'] = text.normal
vars['--primary'] = accent.primary
vars['--success'] = accent.success
// ... 更多映射

// 这样旧代码 var(--bg-primary) 仍然工作
// 新代码使用 var(--af-bg-primary)
```

---

## 5. 分阶段开发计划

### 5.1 总览

| 阶段 | 内容 | 工时 | 风险 |
|------|------|------|------|
| **Phase 0** | 环境准备与工具链配置 | 3-4h | 🟢 低 |
| **Phase 1** | 主题系统搭建 | 4-5h | 🟢 低 |
| **Phase 2** | 基础样式迁移 | 2-3h | 🟢 低 |
| **Phase 3** | 低风险组件迁移 (3个) | 6-8h | 🟡 中 |
| **Phase 4** | 违规修复 (2个) | 4-6h | 🟡 中 |
| **Phase 5** | 中等复杂度组件 (3个) | 8-10h | 🟡 中 |
| **Phase 6** | 验证与文档 | 2-3h | 🟢 低 |
| **合计** | | **29-39h** | |

### 5.2 Phase 0: 环境准备（3-4h）

#### 目标
- 安装所有必需的开发依赖
- 配置 Tailwind CSS
- 配置代码质量工具（Stylelint、Prettier）
- 验证构建正常

#### 任务清单

```
□ 0.1 安装依赖 (30min)
  □ npm install -D tailwindcss postcss autoprefixer
  □ npm install -D stylelint stylelint-config-standard
  □ npm install clsx
  □ npx tailwindcss init -p

□ 0.2 创建 Tailwind 配置 (45min)
  □ 创建 frontend/tailwind.config.js
  □ 配置 prefix: 'tw-'
  □ 配置 darkMode: 'class'
  □ 配置 corePlugins.preflight: false
  □ 配置 content 路径

□ 0.3 创建 PostCSS 配置 (15min)
  □ 创建 frontend/postcss.config.js

□ 0.4 配置 Stylelint (30min)
  □ 创建 frontend/.stylelintrc.js
  □ 配置规则禁止硬编码颜色
  □ 配置规则禁止 !important

□ 0.5 配置 Prettier (15min)
  □ 创建 frontend/.prettierrc.js

□ 0.6 更新 VSCode 配置 (15min)
  □ 更新 .vscode/settings.json
  □ 添加 Tailwind CSS IntelliSense 配置

□ 0.7 创建入口文件 (30min)
  □ 创建 src/styles/tailwind.css
  □ 创建 src/styles/base.css
  □ 更新 vite.config.js（如需要）

□ 0.8 验证 (30min)
  □ npm run dev 正常启动
  □ Tailwind 类生效
  □ 现有样式不受影响
```

#### 关键文件

**tailwind.config.js**
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  prefix: 'tw-',
  content: ['./index.html', './src/**/*.{vue,js}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        'bg': {
          base: 'var(--af-bg-base)',
          primary: 'var(--af-bg-primary)',
          secondary: 'var(--af-bg-secondary)',
          tertiary: 'var(--af-bg-tertiary)',
          elevated: 'var(--af-bg-elevated)',
        },
        'text': {
          primary: 'var(--af-text-primary)',
          normal: 'var(--af-text-normal)',
          secondary: 'var(--af-text-secondary)',
          muted: 'var(--af-text-muted)',
          disabled: 'var(--af-text-disabled)',
        },
        'accent': {
          primary: 'var(--af-accent-primary)',
          success: 'var(--af-accent-success)',
          warning: 'var(--af-accent-warning)',
          danger: 'var(--af-accent-danger)',
        },
        'border': {
          DEFAULT: 'var(--af-border-default)',
          muted: 'var(--af-border-muted)',
          subtle: 'var(--af-border-subtle)',
        },
      },
      borderRadius: {
        'xs': 'var(--af-radius-xs)',
        'sm': 'var(--af-radius-sm)',
        'md': 'var(--af-radius-md)',
        'lg': 'var(--af-radius-lg)',
        'xl': 'var(--af-radius-xl)',
      },
      boxShadow: {
        'sm': 'var(--af-shadow-sm)',
        'md': 'var(--af-shadow-md)',
        'lg': 'var(--af-shadow-lg)',
        'glow': 'var(--af-shadow-glow)',
      },
      transitionDuration: {
        'fast': '100ms',
        'normal': '200ms',
        'slow': '300ms',
      },
      zIndex: {
        'dropdown': '100',
        'sticky': '200',
        'fixed': '300',
        'modal-backdrop': '400',
        'modal': '500',
        'popover': '600',
        'tooltip': '700',
        'notification': '800',
      },
    },
  },
  corePlugins: {
    preflight: false,
  },
  plugins: [],
}
```

**postcss.config.js**
```javascript
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
```

**src/styles/tailwind.css**
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

#### 验收标准
- [ ] `npm run dev` 正常启动
- [ ] 在任意组件添加 `tw-flex` 类后生效
- [ ] 现有页面显示正常，无样式丢失

---

### 5.3 Phase 1: 主题系统搭建（4-5h）

#### 目标
- 创建完整的主题系统目录结构
- 实现 Design Tokens
- 实现 CSS Variables 注入
- 实现 useTheme Composable
- 添加兼容层确保旧代码工作

#### 任务清单

```
□ 1.1 创建目录结构 (15min)
  □ 创建 src/theme/ 目录
  □ 创建 tokens/, composables/, utils/ 子目录

□ 1.2 实现基础令牌 (30min)
  □ 创建 tokens/base.js
  □ 定义 radius, shadow, transition, font, zIndex

□ 1.3 实现深色主题 (45min)
  □ 创建 tokens/dark.js
  □ 迁移所有现有颜色值

□ 1.4 实现主题注册表 (30min)
  □ 创建 tokens/index.js
  □ 实现 getTheme, getAvailableThemes

□ 1.5 实现 CSS 变量注入 (60min)
  □ 创建 utils/inject.js
  □ 实现 injectTheme 函数
  □ 实现 injectBaseTokens 函数
  □ 添加兼容层（旧变量名映射）

□ 1.6 实现 useTheme Composable (45min)
  □ 创建 composables/useTheme.js
  □ 实现 initialize, setTheme, toggleDarkMode

□ 1.7 创建入口文件 (15min)
  □ 创建 theme/index.js

□ 1.8 集成到应用 (30min)
  □ 修改 main.js 初始化主题
  □ 验证所有颜色正确显示

□ 1.9 创建 Element Plus 覆盖 (45min)
  □ 创建 styles/element-override.css
  □ 迁移现有 Element Plus 样式覆盖
```

#### 验收标准
- [ ] 应用启动时自动注入 CSS Variables
- [ ] 所有现有颜色显示正确
- [ ] 控制台无错误
- [ ] 旧代码 `var(--bg-primary)` 仍然工作

---

### 5.4 Phase 2: 基础样式迁移（2-3h）

#### 目标
- 创建新的基础样式文件
- 迁移全局样式到新系统
- 移除对 SCSS 变量的依赖

#### 任务清单

```
□ 2.1 创建基础 CSS (60min)
  □ 创建 styles/base.css
  □ 迁移 CSS Reset 样式
  □ 迁移全局字体设置
  □ 迁移滚动条样式
  □ 迁移链接、按钮基础样式

□ 2.2 更新样式导入顺序 (30min)
  □ 修改 main.js 导入顺序：
    1. element-plus/dist/index.css
    2. styles/tailwind.css
    3. styles/base.css
    4. styles/element-override.css

□ 2.3 移动旧文件 (30min)
  □ 将 _variables.scss 移动到 styles/legacy/
  □ 将 _mixins.scss 移动到 styles/legacy/
  □ 保留文件供参考但不再使用

□ 2.4 验证 (30min)
  □ 全局样式正常
  □ 滚动条样式正确
  □ Element Plus 组件显示正常
```

#### 验收标准
- [ ] 全局样式正常工作
- [ ] 不再依赖 SCSS 变量
- [ ] 构建正常，无 SCSS 编译警告

---

### 5.5 Phase 3: 低风险组件迁移（6-8h）

#### 目标
迁移样式占比高但逻辑简单的组件

#### 组件列表

| 组件 | 当前样式占比 | 目标占比 | 预计工时 |
|------|-------------|----------|----------|
| SubtitleList/index.vue | 65.3% | 30% | 2-3h |
| EditorHeader.vue | 45.8% | 25% | 2-2.5h |
| PlaybackControls/index.vue | 45.4% | 25% | 2-2.5h |

#### 迁移步骤（每个组件）

```
1. 备份：记录当前组件状态
2. 分析：识别所有样式规则
3. 分类：
   - 布局类 → Tailwind
   - 颜色类 → CSS Variables
   - 特殊类 → 保留 scoped CSS
4. 迁移：逐步替换样式
5. 验证：功能测试
6. 清理：移除未使用的样式
```

#### SubtitleList/index.vue 详细迁移

```vue
<!-- Before -->
<style lang="scss" scoped>
.subtitle-list {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--bg-secondary);
}

.list-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  border-bottom: 1px solid var(--border-color);
}
</style>

<!-- After -->
<template>
  <div class="subtitle-list tw-flex tw-flex-col tw-h-full tw-bg-bg-secondary">
    <div class="list-toolbar tw-flex tw-items-center tw-justify-between 
                tw-px-3 tw-py-2 tw-border-b tw-border-border">
      ...
    </div>
  </div>
</template>

<style scoped>
/* 仅保留特殊样式 */
</style>
```

#### 验收标准
- [ ] 每个组件功能完全正常
- [ ] 视觉无变化
- [ ] 移除 `lang="scss"`（或仅保留空壳）
- [ ] 无硬编码颜色

---

### 5.6 Phase 4: 违规修复（4-6h）

#### 目标
修复高违规文件中的 `:deep()` 和 `!important`

#### 组件列表

| 组件 | :deep() | !important | 预计工时 |
|------|---------|------------|----------|
| TaskListView.vue | 16处 | 9处 | 2-3h |
| UpdateDialog.vue | 0 | 12处 | 2-3h |

#### 修复策略

**:deep() 替代方案**
```css
/* Before */
:deep(.el-dialog) {
  background: var(--bg-secondary);
}

/* After: 使用 Element Plus CSS 变量 */
/* 在 element-override.css 中 */
.el-dialog {
  --el-dialog-bg-color: var(--af-bg-secondary);
}
```

**!important 替代方案**
```css
/* Before */
.my-button {
  background: red !important;
}

/* After: 提高选择器优先级 */
.my-component .my-button {
  background: var(--af-accent-danger);
}

/* 或使用 CSS 层级 */
@layer components {
  .my-button {
    background: var(--af-accent-danger);
  }
}
```

#### 验收标准
- [ ] :deep() 使用降至 5 处以下
- [ ] !important 使用降至 3 处以下
- [ ] Element Plus 组件显示正常
- [ ] 对话框样式正确

---

### 5.7 Phase 5: 中等复杂度组件（8-10h）

#### 目标
迁移涉及组件拆分的文件

#### 组件列表

| 组件 | 当前行数 | 目标行数 | 预计工时 |
|------|---------|----------|----------|
| PresetSelector.vue | 962 | ~450 | 3-4h |
| SubtitleItem.vue | 862 | ~400 | 3-3.5h |
| TaskCard.vue | 462 | ~300 | 2-2.5h |

#### 重要提醒 ⚠️

```
此阶段涉及组件拆分，风险较高：
1. 不要同时修改多个文件
2. 每拆分一个子组件就测试一次
3. 保持 Props/Events 接口不变
4. 不要"顺便"优化不相关的逻辑
```

#### 验收标准
- [ ] 所有功能正常工作
- [ ] 父子组件通信正常
- [ ] 无控制台错误

---

### 5.8 Phase 6: 验证与文档（2-3h）

#### 任务清单

```
□ 6.1 全面功能测试 (60min)
  □ 任务列表页面
  □ 编辑器页面
  □ 波形图交互
  □ 字幕编辑
  □ 视频播放
  □ 导出功能

□ 6.2 代码质量检查 (30min)
  □ 运行 Stylelint
  □ 检查硬编码颜色
  □ 检查 !important 使用

□ 6.3 更新文档 (60min)
  □ 更新 README
  □ 创建样式开发指南
  □ 创建 Tailwind 速查表

□ 6.4 清理 (30min)
  □ 删除未使用的 SCSS 文件
  □ 移除 package.json 中的 sass（如果完全不用）
  □ 更新 vite.config.js
```

---

## 6. 组件重构指南

### 6.1 重构前检查清单

每个组件重构前，必须完成以下检查：

```
□ 阅读并理解组件的完整代码
□ 确认组件的所有功能点
□ 确认组件与其他组件的依赖关系
□ 确认组件暴露的 Props、Events、Slots
□ 在浏览器中操作所有功能，记录预期行为
□ 如有疑问，先询问再动手
```

### 6.2 重构中注意事项

```
□ 保留原有的类名（可能被 JS 或测试引用）
□ 不删除任何"看起来没用"的代码
□ 每次修改后立即验证功能
□ 遇到不确定的逻辑，保持原样
□ 添加注释说明为什么保留某些样式
```

### 6.3 重构后验证清单

```
□ 所有原有功能正常工作
□ 视觉效果与重构前一致
□ 控制台无错误或警告
□ 交互响应正常（悬停、点击、焦点等）
□ 在不同窗口大小下正常显示
```

---

## 7. 组件最终审核规则 🔍

### 7.1 核心审核原则

**本章节定义了每个 Vue 组件完成重构后必须通过的强制性审核规则。**

这些规则是**不可妥协的红线**，任何违反都必须修复后才能合并代码。

#### 审核目标

| 目标 | 说明 |
|------|------|
| **零 SCSS 依赖** | 已迁移组件完全不使用 SCSS 特性 |
| **样式隔离** | 组件样式不污染全局命名空间 |
| **可维护性** | 样式代码清晰、一致、易于理解 |
| **工具可检测** | 所有规则可通过自动化工具验证 |

---

### 7.2 强制性规则清单 🚫

#### 规则 1：禁止使用任何 SCSS 特性

**适用范围**：所有已完成迁移的 Vue 组件

| SCSS 特性 | 状态 | 替代方案 |
|-----------|------|----------|
| SCSS 变量 (`$variable`) | ❌ 完全禁止 | CSS Variables (`var(--af-*)`) |
| SCSS 嵌套 (`&` 选择器) | ❌ 完全禁止 | 展平选择器或使用 PostCSS |
| SCSS 函数 (`darken()`, `lighten()`) | ❌ 完全禁止 | CSS `color-mix()` 或预定义变量 |
| SCSS Mixins (`@mixin`, `@include`) | ❌ 完全禁止 | Tailwind 工具类或 CSS 自定义属性 |
| SCSS 运算 (`$size * 2`) | ❌ 完全禁止 | CSS `calc()` |
| SCSS 导入 (`@import`) | ❌ 完全禁止 | CSS `@import` 或 ES6 import |

**检查方法**：
```bash
# 检查是否使用 SCSS 变量
grep -r '\$[a-zA-Z]' frontend/src/components/**/*.vue

# 检查是否使用 SCSS 嵌套
grep -r '&:' frontend/src/components/**/*.vue

# 检查是否使用 SCSS 函数
grep -rE '(darken|lighten|saturate|desaturate|mix)\(' frontend/src/components/**/*.vue
```

**违规示例**：
```vue
<!-- ❌ 错误：使用 SCSS 变量 -->
<style lang="scss" scoped>
$header-height: 56px;
.header { height: $header-height; }
</style>

<!-- ✅ 正确：使用 CSS Variables -->
<style scoped>
.header {
  --header-height: 56px;
  height: var(--header-height);
}
</style>
```

---

#### 规则 2：禁止混用 scoped 和非 scoped 样式块

**适用范围**：所有 Vue 组件

**核心原则**：一个组件文件只能有一个 `<style scoped>` 块，不得出现无 `scoped` 的 `<style>` 块。

| 模式 | 状态 | 说明 |
|------|------|------|
| 单个 `<style scoped>` | ✅ 允许 | 标准模式 |
| 多个 `<style scoped>` | ⚠️ 不推荐 | 应合并为一个 |
| `<style>` (无 scoped) | ❌ 禁止 | 污染全局命名空间 |
| 混用 scoped 和非 scoped | ❌ 严格禁止 | 违反样式隔离原则 |

**全局样式的正确位置**：

| 样式类型 | 正确位置 | 示例 |
|---------|---------|------|
| Element Plus 覆盖 | `src/styles/element-override.css` | `:root { --el-color-primary: ... }` |
| 全局工具类 | `src/styles/utilities.css` | `.text-ellipsis { ... }` |
| 基础样式 | `src/styles/base.css` | `body { ... }` |
| 第三方库覆盖 | `src/styles/vendor-override.css` | `.some-lib { ... }` |

**检查方法**：
```bash
# 检查是否存在无 scoped 的 style 块
grep -rn '<style lang="scss">' frontend/src/components/**/*.vue | grep -v scoped

# 检查是否混用
awk '/<style/ {count++} END {if(count>1) print FILENAME " has multiple style blocks"}' frontend/src/components/**/*.vue
```

**违规示例**：
```vue
<!-- ❌ 错误：混用 scoped 和非 scoped -->
<template>...</template>

<style scoped>
.my-component { ... }
</style>

<style>
/* 全局样式，污染命名空间！ */
.el-button { background: red; }
</style>
```

**正确做法**：
```vue
<!-- ✅ 方案 A：使用 :deep() -->
<style scoped>
.my-component { ... }

:deep(.el-button) {
  background: var(--af-accent-primary);
}
</style>

<!-- ✅ 方案 B：移到全局样式文件 -->
<!-- src/styles/element-override.css -->
.el-button {
  background: var(--af-accent-primary);
}
```

---

### 7.3 条件性规则清单 ⚠️

#### 规则 3：非必须禁止使用 :deep()

**原则**：`:deep()` 是样式穿透的最后手段，应尽量避免使用。

**允许使用的场景**：

| 场景 | 是否允许 | 条件 |
|------|---------|------|
| Element Plus 组件样式覆盖 | ✅ 允许 | 官方 CSS 变量无法满足需求 |
| 第三方库组件覆盖 | ✅ 允许 | 无其他方式修改样式 |
| 动态插入的 DOM | ✅ 允许 | 如富文本编辑器内容 |
| 子组件内部样式 | ❌ 禁止 | 应通过 Props 或 CSS Variables 传递 |
| 覆盖自己的子组件 | ❌ 禁止 | 直接在子组件中修改 |

**使用 :deep() 的正确姿势**：

```vue
<style scoped>
/* ✅ 正确：针对特定组件，添加注释说明原因 */
.my-dialog :deep(.el-dialog__header) {
  /* 原因：Element Plus 不提供 header 背景色的 CSS 变量 */
  background: var(--af-bg-secondary);
}

/* ❌ 错误：过度使用，应该用 CSS 变量 */
.my-button :deep(.el-button) {
  background: red; /* 应该用 --el-button-bg-color */
}

/* ❌ 错误：穿透自己的子组件 */
.parent :deep(.my-child-component) {
  /* 应该在 MyChildComponent.vue 中直接修改 */
  color: blue;
}
</style>
```

**检查方法**：
```bash
# 统计 :deep() 使用次数
grep -rn ':deep(' frontend/src/components/**/*.vue | wc -l

# 查看所有 :deep() 使用位置
grep -rn ':deep(' frontend/src/components/**/*.vue
```

**目标值**：每个组件 `:deep()` 使用次数 ≤ 3

---

### 7.4 代码质量规则

#### 规则 4：禁止硬编码颜色

**原则**：所有颜色必须使用 CSS Variables，不得使用 Hex、RGB、HSL 等硬编码值。

| 颜色格式 | 状态 | 替代方案 |
|---------|------|----------|
| Hex (`#ffffff`) | ❌ 禁止 | `var(--af-bg-primary)` |
| RGB (`rgb(255, 255, 255)`) | ❌ 禁止 | `var(--af-text-normal)` |
| RGBA (`rgba(0, 0, 0, 0.5)`) | ⚠️ 特殊情况 | `rgba(var(--af-accent-primary-rgb), 0.5)` |
| HSL (`hsl(0, 0%, 100%)`) | ❌ 禁止 | `var(--af-bg-primary)` |
| 颜色关键字 (`white`, `black`) | ❌ 禁止 | `var(--af-text-primary)` |

**特殊情况：透明度**

当需要动态透明度时，使用 RGB 变量：

```css
/* ✅ 正确：使用 RGB 变量 + 动态透明度 */
.overlay {
  background: rgba(var(--af-accent-primary-rgb), 0.3);
}

/* ❌ 错误：硬编码 RGBA */
.overlay {
  background: rgba(88, 166, 255, 0.3);
}
```

---

#### 规则 5：严格限制 !important 使用

**原则**：`!important` 是样式优先级问题的症状，不是解决方案。

**目标值**：每个组件 `!important` 使用次数 ≤ 1

**允许使用的场景**：

| 场景 | 是否允许 | 替代方案 |
|------|---------|----------|
| 覆盖第三方库内联样式 | ✅ 允许 | 无其他方式 |
| 覆盖 Element Plus | ❌ 禁止 | 使用 CSS 变量或提高选择器优先级 |
| 覆盖自己的样式 | ❌ 禁止 | 重构选择器结构 |
| 工具类 | ⚠️ 谨慎使用 | 仅在 `utilities.css` 中 |

**正确的优先级提升方法**：

```css
/* ❌ 错误：使用 !important */
.button {
  background: red !important;
}

/* ✅ 方案 A：提高选择器优先级 */
.my-component .button {
  background: var(--af-accent-danger);
}

/* ✅ 方案 B：使用 CSS 层级 */
@layer components {
  .button {
    background: var(--af-accent-danger);
  }
}

/* ✅ 方案 C：使用 Element Plus CSS 变量 */
.my-component {
  --el-button-bg-color: var(--af-accent-danger);
}
```

---

### 7.5 Stylelint 自动化检测配置 🤖

**目标**：通过 Stylelint 自动检测所有违规行为，确保规则可强制执行。

#### 7.5.1 完整 Stylelint 配置

创建 `frontend/.stylelintrc.js`：

```javascript
module.exports = {
  extends: [
    'stylelint-config-standard',
    'stylelint-config-recommended-vue'
  ],

  plugins: [
    'stylelint-scss',
    'stylelint-order'
  ],

  customSyntax: 'postcss-html',

  rules: {
    // ========== 规则 1：禁止 SCSS 特性 ==========

    // 禁止 SCSS 变量（$variable）
    'scss/dollar-variable-pattern': null,
    'scss/at-rule-no-unknown': null,

    // 禁止 SCSS 嵌套（&）
    'selector-nested-pattern': null,

    // 禁止 SCSS 函数
    'function-no-unknown': [
      true,
      {
        ignoreFunctions: [
          'var',           // CSS Variables
          'calc',          // CSS calc
          'rgba',          // CSS rgba
          'rgb',           // CSS rgb
          'url',           // CSS url
          'linear-gradient',
          'radial-gradient'
        ]
      }
    ],

    // ========== 规则 4：禁止硬编码颜色 ==========

    // 禁止 Hex 颜色
    'color-no-hex': true,

    // 禁止颜色关键字（除了 transparent 和 currentColor）
    'color-named': [
      'never',
      {
        ignore: ['inside-function']
      }
    ],

    // 允许 rgba() 但必须使用 CSS Variables
    'function-allowed-list': null,

    // ========== 规则 5：限制 !important ==========

    // 警告使用 !important（不完全禁止，但需要审查）
    'declaration-no-important': [
      true,
      {
        severity: 'warning'
      }
    ],

    // ========== 代码质量规则 ==========

    // 禁止未知的伪类选择器
    'selector-pseudo-class-no-unknown': [
      true,
      {
        ignorePseudoClasses: ['deep', 'global', 'slotted']
      }
    ],

    // 禁止未知的伪元素选择器
    'selector-pseudo-element-no-unknown': [
      true,
      {
        ignorePseudoElements: ['v-deep', 'v-global', 'v-slotted']
      }
    ],

    // 属性顺序（提高可读性）
    'order/properties-order': [
      'position',
      'top',
      'right',
      'bottom',
      'left',
      'z-index',
      'display',
      'flex',
      'flex-direction',
      'flex-wrap',
      'justify-content',
      'align-items',
      'gap',
      'width',
      'height',
      'padding',
      'margin',
      'background',
      'border',
      'border-radius',
      'color',
      'font-size',
      'font-weight',
      'transition',
      'animation'
    ],

    // 禁止空规则
    'block-no-empty': true,

    // 禁止重复的选择器
    'no-duplicate-selectors': true,

    // 禁止重复的属性
    'declaration-block-no-duplicate-properties': true,

    // 最大嵌套深度（防止过度嵌套）
    'max-nesting-depth': [
      3,
      {
        ignore: ['pseudo-classes']
      }
    ],

    // 选择器类名模式（推荐 kebab-case）
    'selector-class-pattern': [
      '^[a-z][a-z0-9]*(-[a-z0-9]+)*$',
      {
        message: 'Expected class selector to be kebab-case'
      }
    ]
  },

  overrides: [
    {
      // 对 Vue 文件的特殊规则
      files: ['**/*.vue'],
      rules: {
        // Vue 文件中允许使用 :deep()
        'selector-pseudo-class-no-unknown': [
          true,
          {
            ignorePseudoClasses: ['deep', 'global', 'slotted']
          }
        ]
      }
    },
    {
      // 全局样式文件可以使用 Hex 颜色（仅限定义 CSS Variables）
      files: ['**/styles/**/*.css'],
      rules: {
        'color-no-hex': null
      }
    }
  ]
}
```

#### 7.5.2 安装依赖

```bash
npm install -D \
  stylelint \
  stylelint-config-standard \
  stylelint-config-recommended-vue \
  stylelint-scss \
  stylelint-order \
  postcss-html
```

#### 7.5.3 package.json 脚本

在 `frontend/package.json` 中添加：

```json
{
  "scripts": {
    "lint:style": "stylelint 'src/**/*.{vue,css,scss}'",
    "lint:style:fix": "stylelint 'src/**/*.{vue,css,scss}' --fix",
    "lint:style:report": "stylelint 'src/**/*.{vue,css,scss}' --formatter json --output-file stylelint-report.json"
  }
}
```

#### 7.5.4 VSCode 集成

在 `.vscode/settings.json` 中添加：

```json
{
  "stylelint.enable": true,
  "stylelint.validate": ["css", "scss", "vue"],
  "editor.codeActionsOnSave": {
    "source.fixAll.stylelint": true
  },
  "css.validate": false,
  "scss.validate": false
}
```

---

### 7.6 审核流程 📋

#### 7.6.1 自动化检查（必须通过）

**在提交代码前，必须运行以下命令并确保全部通过：**

```bash
# 1. Stylelint 检查
npm run lint:style

# 2. 检查 SCSS 特性使用
grep -rn '\$[a-zA-Z]' src/components/**/*.vue
grep -rn '&:' src/components/**/*.vue

# 3. 检查混用 scoped/非 scoped
grep -rn '<style lang="scss">' src/components/**/*.vue | grep -v scoped

# 4. 统计 :deep() 使用次数
grep -rn ':deep(' src/components/**/*.vue | wc -l

# 5. 统计 !important 使用次数
grep -rn '!important' src/components/**/*.vue | wc -l
```

**通过标准**：
- Stylelint 0 errors
- SCSS 特性使用：0 处
- 混用 scoped/非 scoped：0 处
- :deep() 使用：≤ 3 处/组件
- !important 使用：≤ 1 处/组件

#### 7.6.2 人工审查（代码审查）

**审查者必须检查以下项目：**

| 检查项 | 检查方法 | 不通过标准 |
|--------|---------|-----------|
| **SCSS 特性** | 查看 `<style>` 块 | 存在 `$`、`&`、`@mixin` 等 |
| **样式隔离** | 查看 `<style>` 标签 | 存在无 `scoped` 的块 |
| **:deep() 使用** | 查看每个 `:deep()` | 无注释说明原因 |
| **颜色使用** | 查看颜色值 | 存在 Hex、RGB 硬编码 |
| **!important** | 查看每个 `!important` | 无注释说明原因 |
| **Tailwind 使用** | 查看布局样式 | 手写 flex/grid 而非 Tailwind |

#### 7.6.3 审核不通过的处理

**如果审核不通过，必须：**

1. **立即修复**：违反强制性规则（规则 1、2、4、5）
2. **添加注释**：违反条件性规则（规则 3）但有合理原因
3. **重新审核**：修复后重新提交审核

**审核不通过示例**：

```vue
<!-- ❌ 审核不通过：使用 SCSS 变量 -->
<style lang="scss" scoped>
$spacing: 16px;
.container { padding: $spacing; }
</style>

<!-- ✅ 修复后：使用 CSS Variables -->
<style scoped>
.container {
  --spacing: 16px;
  padding: var(--spacing);
}
</style>
```

---

### 7.7 完整检查清单 ✅

**每个组件完成重构后，必须逐项检查以下清单：**

#### 7.7.1 强制性检查（必须全部通过）

```
□ 移除所有 SCSS 变量（$variable）
□ 移除所有 SCSS 嵌套（& 选择器）
□ 移除所有 SCSS 函数（darken、lighten 等）
□ 移除所有 SCSS Mixins（@mixin、@include）
□ 只有一个 <style scoped> 块
□ 无 <style>（非 scoped）块
□ 所有颜色使用 CSS Variables（var(--af-*)）
□ 无 Hex 颜色（#ffffff）
□ 无 RGB/HSL 硬编码
□ 布局样式已迁移到 Tailwind（tw-flex、tw-grid 等）
□ Stylelint 检查通过（0 errors）
```

#### 7.7.2 条件性检查（需要审查）

```
□ :deep() 使用次数 ≤ 3
□ 每个 :deep() 都有注释说明原因
□ !important 使用次数 ≤ 1
□ 每个 !important 都有注释说明原因
□ 保留的 scoped CSS 都有注释说明为什么不能用 Tailwind
```

#### 7.7.3 代码质量检查

```
□ 保留原有类名（可能被 JS 引用）
□ 选择器嵌套深度 ≤ 3 层
□ 无重复的选择器
□ 无重复的属性
□ 无空的样式规则
□ 类名使用 kebab-case
□ 属性按推荐顺序排列
```

#### 7.7.4 功能验证检查

```
□ 所有原有功能正常工作
□ 视觉效果与重构前一致
□ 控制台无错误或警告
□ 交互响应正常（悬停、点击、焦点等）
□ 在不同窗口大小下正常显示
□ 主题切换正常（如果支持）
```

#### 7.7.5 文档检查

```
□ 复杂样式有注释说明
□ :deep() 使用有注释说明原因
□ !important 使用有注释说明原因
□ 特殊处理有注释说明
```

---

### 7.8 常见违规案例与修复 🔧

#### 案例 1：使用 SCSS 变量

```vue
<!-- ❌ 违规 -->
<style lang="scss" scoped>
$header-height: 56px;
.header { height: $header-height; }
</style>

<!-- ✅ 修复方案 A：CSS Variables -->
<style scoped>
.header {
  --header-height: 56px;
  height: var(--header-height);
}
</style>

<!-- ✅ 修复方案 B：直接使用值（如果只用一次）-->
<style scoped>
.header { height: 56px; }
</style>
```

#### 案例 2：混用 scoped 和非 scoped

```vue
<!-- ❌ 违规 -->
<style scoped>
.my-component { ... }
</style>

<style>
.el-button { background: red; }
</style>

<!-- ✅ 修复：移到全局样式文件 -->
<!-- src/styles/element-override.css -->
.el-button {
  --el-button-bg-color: var(--af-accent-danger);
}
```

#### 案例 3：硬编码颜色

```vue
<!-- ❌ 违规 -->
<style scoped>
.button {
  background: #58a6ff;
  color: #ffffff;
  border: 1px solid rgba(88, 166, 255, 0.3);
}
</style>

<!-- ✅ 修复 -->
<style scoped>
.button {
  background: var(--af-accent-primary);
  color: var(--af-text-primary);
  border: 1px solid rgba(var(--af-accent-primary-rgb), 0.3);
}
</style>
```

#### 案例 4：过度使用 :deep()

```vue
<!-- ❌ 违规：穿透自己的子组件 -->
<style scoped>
.parent :deep(.my-child) {
  color: red;
}
</style>

<!-- ✅ 修复：在子组件中直接修改 -->
<!-- MyChild.vue -->
<style scoped>
.my-child {
  color: var(--af-accent-danger);
}
</style>
```

#### 案例 5：滥用 !important

```vue
<!-- ❌ 违规 -->
<style scoped>
.button {
  background: red !important;
  color: white !important;
}
</style>

<!-- ✅ 修复：提高选择器优先级 -->
<style scoped>
.my-component .button {
  background: var(--af-accent-danger);
  color: var(--af-text-primary);
}
</style>
```

---

## 8. 验收标准

### 8.1 代码指标

| 指标 | 当前值 | 目标值 | 检查方法 |
|------|--------|--------|----------|
| 硬编码颜色 | 21 | 0 | Stylelint |
| !important 用法 | 31 | < 3 | grep 搜索 |
| :deep() 用法 | 18 | < 5 | grep 搜索 |
| SCSS 文件 | 多个 | 0 (或仅 legacy) | 目录检查 |
| 新增 SCSS | - | 0 | CR 检查 |

### 8.2 功能验收清单

- [ ] **任务列表页**
  - [ ] 任务卡片显示正常
  - [ ] 上传视频功能正常
  - [ ] 任务状态更新正常
  - [ ] 缩略图加载正常
  
- [ ] **编辑器页**
  - [ ] 视频播放正常
  - [ ] 波形图显示和交互正常
  - [ ] 字幕列表显示正常
  - [ ] 字幕编辑正常
  - [ ] 时间调整正常
  - [ ] 导出功能正常
  
- [ ] **主题系统**
  - [ ] 深色主题正确显示
  - [ ] 主题持久化正常
  - [ ] Element Plus 组件主题正确

### 8.3 质量验收清单

- [ ] Stylelint 检查通过
- [ ] 构建无错误和警告
- [ ] 所有 CSS Variables 使用正确命名
- [ ] Tailwind 类使用 `tw-` 前缀

---

## 9. 风险与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| 样式回归 | 高 | 中 | 每步都测试，小步迭代 |
| Element Plus 冲突 | 中 | 中 | 使用官方 CSS 变量，添加前缀 |
| 功能破坏 | 中 | 高 | 不修改逻辑，仅改样式 |
| 构建问题 | 低 | 中 | 保留旧文件作为备份 |
| 波形图颜色 | 中 | 低 | 创建主题配置对象传入 |
| 工时超出 | 中 | 低 | 分阶段执行，可中断 |

---

## 10. 附录

### 10.1 Tailwind 类名速查表

```
布局:
  tw-flex tw-inline-flex tw-grid tw-block tw-hidden
  tw-flex-row tw-flex-col tw-flex-wrap
  tw-items-start tw-items-center tw-items-end
  tw-justify-start tw-justify-center tw-justify-end tw-justify-between

间距:
  tw-p-{0-12} tw-px-{0-12} tw-py-{0-12}
  tw-m-{0-12} tw-mx-{0-12} tw-my-{0-12} tw-m-auto
  tw-gap-{0-12}

尺寸:
  tw-w-full tw-w-auto tw-w-screen
  tw-h-full tw-h-auto tw-h-screen
  tw-min-w-0 tw-max-w-full

颜色（使用自定义变量）:
  tw-bg-bg-primary tw-bg-bg-secondary tw-bg-bg-tertiary
  tw-text-text-primary tw-text-text-normal tw-text-text-muted
  tw-border-border

圆角:
  tw-rounded-none tw-rounded-xs tw-rounded-sm tw-rounded-md tw-rounded-lg tw-rounded-full

交互:
  hover:tw-bg-bg-tertiary
  focus:tw-ring-2 focus:tw-ring-accent-primary
  active:tw-scale-[0.98]
  disabled:tw-opacity-50 disabled:tw-cursor-not-allowed

过渡:
  tw-transition-colors tw-transition-opacity tw-transition-all
  tw-duration-fast tw-duration-normal tw-duration-slow
```

### 10.2 CSS Variables 完整列表

```css
/* 背景 */
--af-bg-base, --af-bg-primary, --af-bg-secondary, --af-bg-tertiary, --af-bg-elevated, --af-bg-overlay

/* 文本 */
--af-text-primary, --af-text-normal, --af-text-secondary, --af-text-muted, --af-text-disabled

/* 强调色 */
--af-accent-primary, --af-accent-primary-hover, --af-accent-primary-active, --af-accent-primary-rgb
--af-accent-success, --af-accent-success-dim, --af-accent-success-rgb
--af-accent-warning, --af-accent-warning-dim, --af-accent-warning-rgb
--af-accent-danger, --af-accent-danger-dim, --af-accent-danger-rgb
--af-accent-purple, --af-accent-pink

/* 边框 */
--af-border-default, --af-border-muted, --af-border-subtle

/* 阶段 */
--af-stage-sensevoice, --af-stage-whisper, --af-stage-llm

/* 状态 */
--af-status-processing, --af-status-warning, --af-status-error

/* 波形 */
--af-waveform-color, --af-waveform-progress, --af-waveform-cursor
--af-region-default, --af-region-active, --af-region-selected

/* 视频 */
--af-video-bg, --af-video-text, --af-video-overlay

/* 基础 */
--af-radius-xs, --af-radius-sm, --af-radius-md, --af-radius-lg, --af-radius-xl, --af-radius-full
--af-shadow-sm, --af-shadow-md, --af-shadow-lg, --af-shadow-xl, --af-shadow-glow
--af-transition-fast, --af-transition-normal, --af-transition-slow
--af-font-mono
--af-z-dropdown, --af-z-sticky, --af-z-fixed, --af-z-modal-backdrop, --af-z-modal, --af-z-popover, --af-z-tooltip, --af-z-notification
```

### 10.3 迁移检查清单

每个组件迁移后，检查以下项目：

```
□ 移除所有 #hex 颜色
□ 移除所有 !important
□ 移除或注释说明所有 :deep()
□ 布局样式已迁移到 Tailwind
□ 颜色使用 CSS Variables
□ 保留原有类名
□ 功能测试通过
□ 视觉无变化
```

---

## 更新历史

| 日期 | 版本 | 变更内容 |
|------|------|----------|
| 2026-01-30 | V3.2.0+dev.20260130.01 | 整合三份文档，创建完整开发文档 |

---

**文档结束**
