# AnchorFlux 前端样式重构方案评审与优化建议

**评审版本**: V3.1.2+dev.20260130.01  
**评审日期**: 2026-01-30  
**状态**: 评审完成

---

## 1. 总体评估

### 1.1 方案优势 ✅

原重构方案具有以下亮点：

| 维度 | 评价 | 说明 |
|------|------|------|
| **问题分析深入** | ⭐⭐⭐⭐⭐ | 详细统计了 287 处违规，分类清晰 |
| **组件复杂度分析** | ⭐⭐⭐⭐⭐ | 按行数、样式占比、方法数排序，数据驱动 |
| **三类策略合理** | ⭐⭐⭐⭐ | 纯样式/同步解耦/后期解耦分层处理 |
| **工时估算现实** | ⭐⭐⭐⭐ | 25-30h 的预估合理 |
| **技术选型正确** | ⭐⭐⭐⭐ | Tailwind + CSS Variables 符合现代最佳实践 |

### 1.2 需要改进的方面 ⚠️

| 维度 | 问题 | 风险等级 |
|------|------|----------|
| **缺少开发规范文档** | 没有配套的样式编写规范、命名约定、代码审查标准 | 🔴 高 |
| **Tailwind 配置不完整** | 缺少 `prefix`、`darkMode`、`safelist` 等关键配置 | 🟡 中 |
| **缺少代码质量工具链** | 无 Stylelint、Prettier 配置，难以保持一致性 | 🔴 高 |
| **CSS Variables 命名混乱** | 存在 `--accent-*` 和 `--primary` 等重复语义 | 🟡 中 |
| **缺少组件样式隔离策略** | 对 Element Plus 覆盖和自定义组件界限模糊 | 🟡 中 |
| **无响应式设计规范** | 现有 mixins 未纳入 Tailwind 体系 | 🟢 低 |

---

## 2. 技术选型深度评审

### 2.1 Tailwind CSS + CSS Variables + SCSS 组合评估

**结论：技术选型正确，但需要更精细的边界定义**

#### 当前定义的职责边界
```
Tailwind CSS    → 布局、间距、基础样式
CSS Variables   → 主题色、语义变量
SCSS (极少)     → 复杂动画、波形图等特殊场景
```

#### 建议优化的职责边界
```
Tailwind CSS    → 布局（flex/grid）、间距、尺寸、基础交互状态
                  ✅ 推荐用法: flex, gap, p-*, m-*, w-*, h-*, rounded-*
                  ❌ 避免用法: 颜色类（使用 CSS 变量代替）
                  
CSS Variables   → 颜色系统、主题切换、动态值
                  ✅ 统一前缀: --af-* (AnchorFlux)
                  ✅ 层次化: --af-color-*, --af-spacing-*, --af-radius-*
                  
SCSS            → Element Plus 覆盖、复杂选择器、Mixin 复用
                  ✅ 限制: 仅用于无法用 Tailwind 表达的场景
                  ✅ 注释: 必须说明为什么不能用 Tailwind
```

### 2.2 是否需要引入额外工具？

| 工具 | 建议 | 理由 |
|------|------|------|
| **Stylelint** | ✅ 必须 | 强制执行代码规范，自动检测违规 |
| **Prettier** | ✅ 必须 | 格式化 CSS/SCSS，保持一致性 |
| **postcss-preset-env** | ✅ 建议 | 使用现代 CSS 特性，自动降级 |
| **@tailwindcss/typography** | ⚪ 可选 | 如果有富文本内容展示 |
| **tailwind-merge** | ✅ 建议 | 解决动态类名冲突问题 |
| **clsx** | ✅ 建议 | 条件类名拼接，比模板字符串更清晰 |

---

## 3. 优化后的技术规范

### 3.1 增强的 Tailwind 配置

```javascript
// frontend/tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  // 添加前缀避免与 Element Plus 冲突
  prefix: 'tw-',
  
  // 确保只处理自己的文件
  content: [
    './index.html',
    './src/**/*.{vue,js,ts,jsx,tsx}',
    // 排除 node_modules
  ],
  
  // 暗色模式配置（为未来浅色主题预留）
  darkMode: 'class',
  
  theme: {
    extend: {
      // === 颜色系统（使用 CSS 变量） ===
      colors: {
        // 背景层次
        'bg': {
          base: 'var(--af-bg-base)',
          primary: 'var(--af-bg-primary)',
          secondary: 'var(--af-bg-secondary)',
          tertiary: 'var(--af-bg-tertiary)',
          elevated: 'var(--af-bg-elevated)',
        },
        // 文本层次
        'text': {
          primary: 'var(--af-text-primary)',
          normal: 'var(--af-text-normal)',
          secondary: 'var(--af-text-secondary)',
          muted: 'var(--af-text-muted)',
          disabled: 'var(--af-text-disabled)',
        },
        // 语义色（支持透明度）
        'accent': {
          primary: 'rgb(var(--af-accent-primary-rgb) / <alpha-value>)',
          success: 'rgb(var(--af-accent-success-rgb) / <alpha-value>)',
          warning: 'rgb(var(--af-accent-warning-rgb) / <alpha-value>)',
          danger: 'rgb(var(--af-accent-danger-rgb) / <alpha-value>)',
        },
        // 阶段标记
        'stage': {
          sensevoice: 'var(--af-stage-sensevoice)',
          whisper: 'var(--af-stage-whisper)',
          llm: 'var(--af-stage-llm)',
        },
        // 边框
        'border': {
          DEFAULT: 'var(--af-border-default)',
          muted: 'var(--af-border-muted)',
          subtle: 'var(--af-border-subtle)',
        },
      },
      
      // === 间距系统 ===
      spacing: {
        'header': '48px',
        'sidebar': '280px',
        'toolbar': '40px',
        'panel-gap': '12px',
      },
      
      // === 圆角 ===
      borderRadius: {
        'xs': 'var(--af-radius-xs)',
        'sm': 'var(--af-radius-sm)',
        'md': 'var(--af-radius-md)',
        'lg': 'var(--af-radius-lg)',
        'xl': 'var(--af-radius-xl)',
      },
      
      // === 阴影 ===
      boxShadow: {
        'sm': 'var(--af-shadow-sm)',
        'md': 'var(--af-shadow-md)',
        'lg': 'var(--af-shadow-lg)',
        'glow': 'var(--af-shadow-glow)',
      },
      
      // === 过渡 ===
      transitionDuration: {
        'fast': '100ms',
        'normal': '200ms',
        'slow': '300ms',
      },
      
      // === 字体 ===
      fontFamily: {
        'mono': ['JetBrains Mono', 'SF Mono', 'Monaco', 'Consolas', 'monospace'],
      },
      
      // === Z-Index 层级 ===
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
  
  plugins: [],
  
  // 防止与 Element Plus 的类名冲突
  corePlugins: {
    preflight: false, // 禁用 Tailwind 的 reset，使用我们自己的
  },
}
```

### 3.2 优化的 CSS Variables 系统

```scss
// src/styles/theme/_variables.scss
// ==============================================
// AnchorFlux 设计令牌系统
// 命名规范: --af-{category}-{name}[-{variant}]
// ==============================================

:root {
  // ============================================
  // 原始色板（仅供内部引用）
  // ============================================
  --_palette-gray-50: #0d1117;
  --_palette-gray-100: #161b22;
  --_palette-gray-200: #21262d;
  --_palette-gray-300: #30363d;
  --_palette-gray-400: #484f58;
  --_palette-gray-500: #6e7681;
  --_palette-gray-600: #8b949e;
  --_palette-gray-700: #c9d1d9;
  --_palette-gray-800: #e6edf3;
  
  --_palette-blue: #58a6ff;
  --_palette-blue-light: #79c0ff;
  --_palette-blue-dark: #388bfd;
  
  --_palette-green: #3fb950;
  --_palette-green-dark: #238636;
  
  --_palette-orange: #d29922;
  --_palette-orange-dark: #9e6a03;
  
  --_palette-red: #f85149;
  --_palette-red-dark: #da3633;
  
  --_palette-purple: #a371f7;
  --_palette-pink: #db61a2;

  // ============================================
  // 语义层 - 背景
  // ============================================
  --af-bg-base: var(--_palette-gray-50);
  --af-bg-primary: var(--_palette-gray-100);
  --af-bg-secondary: var(--_palette-gray-200);
  --af-bg-tertiary: var(--_palette-gray-300);
  --af-bg-elevated: #2d333b;
  --af-bg-overlay: rgba(0, 0, 0, 0.5);

  // ============================================
  // 语义层 - 文本
  // ============================================
  --af-text-primary: var(--_palette-gray-800);
  --af-text-normal: var(--_palette-gray-700);
  --af-text-secondary: var(--_palette-gray-600);
  --af-text-muted: var(--_palette-gray-500);
  --af-text-disabled: var(--_palette-gray-400);

  // ============================================
  // 语义层 - 强调色
  // ============================================
  --af-accent-primary: var(--_palette-blue);
  --af-accent-primary-hover: var(--_palette-blue-light);
  --af-accent-primary-active: var(--_palette-blue-dark);
  --af-accent-primary-rgb: 88, 166, 255;

  --af-accent-success: var(--_palette-green);
  --af-accent-success-dim: var(--_palette-green-dark);
  --af-accent-success-rgb: 63, 185, 80;

  --af-accent-warning: var(--_palette-orange);
  --af-accent-warning-dim: var(--_palette-orange-dark);
  --af-accent-warning-rgb: 210, 153, 34;

  --af-accent-danger: var(--_palette-red);
  --af-accent-danger-dim: var(--_palette-red-dark);
  --af-accent-danger-rgb: 248, 81, 73;

  --af-accent-purple: var(--_palette-purple);
  --af-accent-pink: var(--_palette-pink);

  // ============================================
  // 语义层 - 边框
  // ============================================
  --af-border-default: var(--_palette-gray-300);
  --af-border-muted: var(--_palette-gray-200);
  --af-border-subtle: rgba(240, 246, 252, 0.1);

  // ============================================
  // 功能层 - 阶段标记
  // ============================================
  --af-stage-sensevoice: #58a6ff;
  --af-stage-whisper: #3fb950;
  --af-stage-llm: #a371f7;

  // ============================================
  // 功能层 - 状态色
  // ============================================
  --af-status-processing: #028AC5;
  --af-status-warning: #e67700;
  --af-status-error: #f85149;

  // ============================================
  // 功能层 - 波形编辑器
  // ============================================
  --af-waveform-color: var(--_palette-blue);
  --af-waveform-progress: var(--_palette-green-dark);
  --af-waveform-cursor: var(--_palette-red);
  --af-region-default: rgba(88, 166, 255, 0.25);
  --af-region-active: rgba(88, 166, 255, 0.4);
  --af-region-selected: rgba(163, 113, 247, 0.35);

  // ============================================
  // 功能层 - 视频播放器
  // ============================================
  --af-video-bg: #000;
  --af-video-text: #fff;
  --af-video-overlay: rgba(0, 0, 0, 0.6);

  // ============================================
  // 基础令牌 - 圆角
  // ============================================
  --af-radius-xs: 3px;
  --af-radius-sm: 6px;
  --af-radius-md: 8px;
  --af-radius-lg: 12px;
  --af-radius-xl: 16px;
  --af-radius-full: 9999px;

  // ============================================
  // 基础令牌 - 阴影
  // ============================================
  --af-shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.4);
  --af-shadow-md: 0 3px 6px rgba(0, 0, 0, 0.3), 0 1px 3px rgba(0, 0, 0, 0.4);
  --af-shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.4), 0 3px 8px rgba(0, 0, 0, 0.3);
  --af-shadow-xl: 0 16px 48px rgba(0, 0, 0, 0.5), 0 6px 16px rgba(0, 0, 0, 0.4);
  --af-shadow-glow: 0 0 16px rgba(88, 166, 255, 0.15);

  // ============================================
  // 基础令牌 - 过渡
  // ============================================
  --af-transition-fast: 0.1s ease;
  --af-transition-normal: 0.2s ease;
  --af-transition-slow: 0.3s ease;

  // ============================================
  // 基础令牌 - 字体
  // ============================================
  --af-font-mono: 'JetBrains Mono', 'SF Mono', Monaco, Consolas, monospace;

  // ============================================
  // 兼容层 - 旧变量映射（迁移期间使用）
  // TODO: 迁移完成后删除
  // ============================================
  --bg-base: var(--af-bg-base);
  --bg-primary: var(--af-bg-primary);
  --bg-secondary: var(--af-bg-secondary);
  --bg-tertiary: var(--af-bg-tertiary);
  --bg-elevated: var(--af-bg-elevated);
  --bg-overlay: var(--af-bg-overlay);
  
  --text-primary: var(--af-text-primary);
  --text-normal: var(--af-text-normal);
  --text-secondary: var(--af-text-secondary);
  --text-muted: var(--af-text-muted);
  --text-disabled: var(--af-text-disabled);
  --text-bright: var(--af-text-primary);
  
  --primary: var(--af-accent-primary);
  --primary-hover: var(--af-accent-primary-hover);
  --primary-dim: var(--af-accent-primary-active);
  --primary-rgb: var(--af-accent-primary-rgb);
  --accent-blue: var(--af-accent-primary);
  
  --success: var(--af-accent-success);
  --success-dim: var(--af-accent-success-dim);
  --accent-green: var(--af-accent-success);
  
  --warning: var(--af-accent-warning);
  --warning-dim: var(--af-accent-warning-dim);
  --accent-orange: var(--af-accent-warning);
  
  --danger: var(--af-accent-danger);
  --danger-dim: var(--af-accent-danger-dim);
  --accent-red: var(--af-accent-danger);
  
  --purple: var(--af-accent-purple);
  --pink: var(--af-accent-pink);
  
  --border-default: var(--af-border-default);
  --border-muted: var(--af-border-muted);
  --border-subtle: var(--af-border-subtle);
  --border-color: var(--af-border-default);
  
  --waveform-color: var(--af-waveform-color);
  --waveform-progress: var(--af-waveform-progress);
  --waveform-cursor: var(--af-waveform-cursor);
  --region-default: var(--af-region-default);
  --region-active: var(--af-region-active);
  --region-selected: var(--af-region-selected);
  
  --radius-xs: var(--af-radius-xs);
  --radius-sm: var(--af-radius-sm);
  --radius-md: var(--af-radius-md);
  --radius-lg: var(--af-radius-lg);
  --radius-xl: var(--af-radius-xl);
  --radius-full: var(--af-radius-full);
  
  --shadow-sm: var(--af-shadow-sm);
  --shadow-md: var(--af-shadow-md);
  --shadow-lg: var(--af-shadow-lg);
  --shadow-xl: var(--af-shadow-xl);
  --shadow-glow: var(--af-shadow-glow);
  
  --transition-fast: var(--af-transition-fast);
  --transition-normal: var(--af-transition-normal);
  --transition-slow: var(--af-transition-slow);
  
  --font-mono: var(--af-font-mono);
}
```

---

## 4. 前端开发规范（新增）

### 4.1 样式编写规范

#### 4.1.1 优先级规则

```
1️⃣ Tailwind 工具类 (最高优先级)
   └─ 布局、间距、尺寸、基础状态

2️⃣ CSS 变量
   └─ 颜色、主题相关的动态值

3️⃣ Scoped SCSS (最低优先级)
   └─ 复杂选择器、动画、Element Plus 覆盖
```

#### 4.1.2 代码示例

```vue
<!-- ✅ 正确示例 -->
<template>
  <div class="tw-flex tw-items-center tw-gap-4 tw-p-4 tw-rounded-md 
              tw-bg-bg-secondary hover:tw-bg-bg-tertiary
              tw-transition-colors tw-duration-normal">
    <span class="tw-text-text-secondary">标签</span>
    <button class="btn-primary">操作</button>
  </div>
</template>

<style lang="scss" scoped>
// ✅ 仅用于无法用 Tailwind 表达的场景
.btn-primary {
  // 复杂渐变、多状态按钮
  background: linear-gradient(135deg, var(--af-accent-primary), var(--af-accent-primary-active));
  
  &:active {
    transform: scale(0.98);
  }
}
</style>

<!-- ❌ 错误示例 -->
<template>
  <div class="container">...</div>
</template>

<style lang="scss" scoped>
.container {
  // ❌ 这些应该用 Tailwind
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px;
  border-radius: 8px;
  // ❌ 硬编码颜色
  background: #21262d;
}
</style>
```

#### 4.1.3 Element Plus 覆盖规范

```scss
// ✅ 正确方式：使用 Element Plus CSS 变量
.el-button {
  --el-button-bg-color: var(--af-bg-secondary);
  --el-button-border-color: var(--af-border-default);
  --el-button-text-color: var(--af-text-normal);
  --el-button-hover-bg-color: var(--af-bg-tertiary);
}

// ⚠️ 仅在必要时使用 :deep()，并添加注释说明原因
:deep(.el-dialog) {
  // 原因：Element Plus 不提供对应的 CSS 变量
  --el-dialog-bg-color: var(--af-bg-secondary);
}

// ❌ 禁止：直接覆盖 + !important
.el-dialog {
  background: #21262d !important; // ❌ 绝对禁止
}
```

### 4.2 命名规范

#### 4.2.1 CSS 类名

```
BEM 命名法（仅用于无法用 Tailwind 的组件）:
  .block
  .block__element
  .block--modifier
  .block__element--modifier

示例:
  .waveform-timeline
  .waveform-timeline__track
  .waveform-timeline__track--active
  .waveform-timeline__cursor
```

#### 4.2.2 CSS 变量

```
格式: --af-{category}-{name}[-{variant}]

category 可选值:
  bg       - 背景色
  text     - 文本色
  accent   - 强调色
  border   - 边框
  shadow   - 阴影
  radius   - 圆角
  spacing  - 间距
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

### 4.3 组件样式组织

```vue
<style lang="scss" scoped>
// ==============================================
// 1. 布局骨架（使用 Tailwind 的情况下可省略）
// ==============================================

// ==============================================
// 2. 主题变量覆盖（如需局部覆盖）
// ==============================================

// ==============================================
// 3. 元素基础样式
// ==============================================

// ==============================================
// 4. 状态样式 (:hover, :active, :focus, etc.)
// ==============================================

// ==============================================
// 5. 动画和过渡
// ==============================================

// ==============================================
// 6. Element Plus 覆盖（添加注释说明原因）
// ==============================================

// ==============================================
// 7. 响应式样式
// ==============================================
</style>
```

### 4.4 禁止清单 🚫

| 类别 | 禁止用法 | 替代方案 |
|------|----------|----------|
| **颜色** | `#f85149`, `rgb(248, 81, 73)` | `var(--af-accent-danger)` |
| **布局** | `display: flex; gap: 8px;` | `class="tw-flex tw-gap-2"` |
| **间距** | `padding: 16px; margin: 8px;` | `class="tw-p-4 tw-m-2"` |
| **圆角** | `border-radius: 8px;` | `class="tw-rounded-md"` 或 `var(--af-radius-md)` |
| **覆盖** | `!important` | 使用更高优先级选择器或 CSS 变量 |
| **穿透** | 滥用 `:deep()` | 使用 Element Plus CSS 变量 |

---

## 5. 工具链配置

### 5.1 Stylelint 配置

```javascript
// .stylelintrc.js
module.exports = {
  extends: [
    'stylelint-config-standard-scss',
    'stylelint-config-recommended-vue/scss',
  ],
  plugins: [
    'stylelint-order',
  ],
  rules: {
    // 禁止硬编码颜色
    'color-no-hex': true,
    'color-named': 'never',
    
    // 强制使用 CSS 变量
    'declaration-property-value-disallowed-list': {
      '/color$/': ['/^#/', '/^rgb/', '/^hsl/'],
      '/background$/': ['/^#/', '/^rgb/', '/^hsl/'],
      'border-color': ['/^#/', '/^rgb/', '/^hsl/'],
    },
    
    // 禁止 !important（可豁免特定情况）
    'declaration-no-important': true,
    
    // 选择器深度限制
    'selector-max-compound-selectors': 4,
    'selector-max-specificity': '0,4,0',
    
    // 属性排序
    'order/properties-order': [
      // 布局
      'display', 'flex', 'flex-direction', 'flex-wrap', 'justify-content', 'align-items', 'gap',
      'grid', 'grid-template-columns', 'grid-template-rows',
      'position', 'top', 'right', 'bottom', 'left', 'z-index',
      // 盒模型
      'width', 'height', 'min-width', 'min-height', 'max-width', 'max-height',
      'margin', 'padding',
      // 边框
      'border', 'border-radius',
      // 背景
      'background', 'background-color',
      // 文本
      'color', 'font-size', 'font-weight', 'line-height', 'text-align',
      // 其他
      'opacity', 'visibility', 'overflow',
      'transition', 'transform', 'animation',
    ],
    
    // Vue scoped 样式
    'selector-pseudo-class-no-unknown': [true, {
      ignorePseudoClasses: ['deep', 'global', 'slotted'],
    }],
    
    // SCSS 特定
    'scss/dollar-variable-pattern': '^_?[a-z][a-z0-9-]*$',
    'scss/at-mixin-pattern': '^[a-z][a-z0-9-]*$',
  },
  
  // 忽略 Tailwind 指令
  ignoreAtRules: ['tailwind', 'apply', 'layer', 'config'],
}
```

### 5.2 Prettier 配置

```javascript
// .prettierrc.js
module.exports = {
  // 通用
  printWidth: 100,
  tabWidth: 2,
  useTabs: false,
  semi: false,
  singleQuote: true,
  trailingComma: 'es5',
  
  // CSS/SCSS
  singleAttributePerLine: false,
  
  // 覆盖特定文件
  overrides: [
    {
      files: '*.scss',
      options: {
        singleQuote: false,
      },
    },
  ],
}
```

### 5.3 VSCode 配置

```json
// .vscode/settings.json
{
  "css.validate": false,
  "scss.validate": false,
  "stylelint.validate": ["css", "scss", "vue"],
  
  "editor.codeActionsOnSave": {
    "source.fixAll.stylelint": "explicit"
  },
  
  "tailwindCSS.experimental.classRegex": [
    ["class=\"([^\"]*)", "([^\"]*?)"],
    [":class=\"([^\"]*)", "([^\"]*?)"],
    ["class: \"([^\"]*)", "([^\"]*?)"]
  ],
  
  "tailwindCSS.includeLanguages": {
    "vue": "html"
  }
}
```

### 5.4 package.json 脚本

```json
{
  "scripts": {
    "lint:style": "stylelint \"src/**/*.{css,scss,vue}\" --fix",
    "lint:all": "npm run lint:style && npm run lint",
    "format": "prettier --write \"src/**/*.{vue,js,ts,css,scss}\""
  },
  "devDependencies": {
    "stylelint": "^16.0.0",
    "stylelint-config-standard-scss": "^13.0.0",
    "stylelint-config-recommended-vue": "^1.5.0",
    "stylelint-order": "^6.0.0",
    "prettier": "^3.2.0",
    "tailwindcss": "^3.4.0",
    "postcss": "^8.4.0",
    "autoprefixer": "^10.4.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.2.0"
  }
}
```

---

## 6. 修订后的实施计划

### 6.1 Phase 0: 基础设施（建议 3-4h）

```
原计划: 2-3h
修订后: 3-4h（增加工具链配置）

任务清单:
├── 1. 安装依赖
│   ├── npm install -D tailwindcss postcss autoprefixer
│   ├── npm install -D stylelint stylelint-config-* stylelint-order
│   ├── npm install clsx tailwind-merge
│   └── npx tailwindcss init -p
│
├── 2. 配置 Tailwind
│   ├── 创建增强版 tailwind.config.js
│   ├── 配置 prefix: 'tw-'
│   └── 配置 darkMode 和 corePlugins
│
├── 3. 配置工具链
│   ├── 创建 .stylelintrc.js
│   ├── 创建 .prettierrc.js
│   ├── 更新 .vscode/settings.json
│   └── 更新 package.json scripts
│
└── 4. 更新 vite.config.js
    └── 确保 SCSS 和 PostCSS 正确集成
```

### 6.2 Phase 1: 主题系统迁移（建议 4-5h）

```
原计划: 3-4h
修订后: 4-5h（增加变量迁移和兼容层）

任务清单:
├── 1. 创建新的样式目录结构
│   ├── src/styles/
│   │   ├── tailwind.css      # Tailwind 入口
│   │   ├── theme/
│   │   │   ├── _tokens.scss  # 设计令牌
│   │   │   ├── _compat.scss  # 兼容层（旧变量映射）
│   │   │   └── _element.scss # Element Plus 覆盖
│   │   ├── _mixins.scss
│   │   └── main.scss
│   └── 迁移现有 _variables.scss 到新结构
│
├── 2. 更新 main.js
│   └── 调整样式导入顺序
│
└── 3. 验证
    ├── 确保所有旧变量仍然工作
    └── 检查 Element Plus 主题正确
```

### 6.3 新增 Phase: 规范文档编写（建议 2h）

```
新增阶段，在正式迁移前完成

任务清单:
├── 1. 创建 docs/frontend-style-guide.md
│   ├── 完整的样式编写规范
│   ├── Tailwind 类名速查表
│   └── 常见问题和解决方案
│
├── 2. 创建组件模板
│   └── 提供符合规范的组件模板代码
│
└── 3. 团队培训材料（如需要）
```

---

## 7. 总结

### 7.1 原方案评分

| 维度 | 评分 | 说明 |
|------|------|------|
| 技术选型 | 8/10 | 正确但需要更精细的边界定义 |
| 分析深度 | 9/10 | 数据驱动，分类清晰 |
| 实施规划 | 7/10 | 缺少工具链和规范文档 |
| 风险控制 | 8/10 | 识别了主要风险 |
| **总体** | **8/10** | 良好的起点，需要补充 |

### 7.2 核心改进点

1. **添加 `tw-` 前缀** - 避免与 Element Plus 类名冲突
2. **统一变量命名** - 使用 `--af-*` 前缀，层次化组织
3. **引入工具链** - Stylelint + Prettier 强制执行规范
4. **补充开发规范** - 明确的优先级规则、禁止清单
5. **增加兼容层** - 平滑迁移，不破坏现有代码
6. **引入 clsx/tailwind-merge** - 解决动态类名问题

### 7.3 建议的执行顺序

```
1. Phase 0: 基础设施搭建 (3-4h)
2. Phase 新增: 规范文档编写 (2h)
3. Phase 1: 主题系统迁移 (4-5h)
4. Phase 2A: 纯样式重构 (4-6h) - 原计划不变
5. Phase 2B: 样式重构+解耦 (8-10h) - 原计划不变
6. Phase 3: 违规修复 (4-6h) - 原计划不变
7. Phase 4: 验证和完善 (2-3h) - 原计划不变

总工时: 27-36h (原计划 25-30h)
```

---

## 附录 A: Tailwind 类名速查表

```
布局:
  tw-flex tw-inline-flex tw-grid tw-block tw-hidden
  tw-flex-row tw-flex-col tw-flex-wrap
  tw-items-start tw-items-center tw-items-end tw-items-stretch
  tw-justify-start tw-justify-center tw-justify-end tw-justify-between tw-justify-around

间距:
  tw-p-{0-12} tw-px-{0-12} tw-py-{0-12}
  tw-m-{0-12} tw-mx-{0-12} tw-my-{0-12} tw-m-auto
  tw-gap-{0-12}

尺寸:
  tw-w-full tw-w-auto tw-w-{px|1/2|1/3|screen}
  tw-h-full tw-h-auto tw-h-screen
  tw-min-w-0 tw-max-w-full

圆角:
  tw-rounded-none tw-rounded-xs tw-rounded-sm tw-rounded-md tw-rounded-lg tw-rounded-full

颜色（使用 CSS 变量）:
  tw-bg-bg-primary tw-bg-bg-secondary tw-bg-bg-tertiary
  tw-text-text-primary tw-text-text-normal tw-text-text-muted
  tw-border-border

过渡:
  tw-transition-colors tw-transition-opacity tw-transition-transform tw-transition-all
  tw-duration-fast tw-duration-normal tw-duration-slow

交互:
  hover:tw-bg-bg-tertiary active:tw-scale-[0.98]
  focus:tw-ring-2 focus:tw-ring-accent-primary
  disabled:tw-opacity-50 disabled:tw-cursor-not-allowed
```

---

## 附录 B: 迁移检查清单

- [ ] 所有 `#hex` 颜色已替换为 CSS 变量
- [ ] 所有 `!important` 已移除（或添加了豁免注释）
- [ ] 所有 `:deep()` 已审查并添加原因注释
- [ ] 布局样式已迁移到 Tailwind 类
- [ ] Element Plus 覆盖使用官方 CSS 变量
- [ ] Stylelint 检查通过
- [ ] 视觉回归测试通过
- [ ] 波形图交互正常
- [ ] 字幕编辑正常
- [ ] 响应式布局正常

---

**文档结束**
