# AnchorFlux 前端开发样式规范 (v1.0)

**版本**: V3.1.2+dev.20260109.01
**创建日期**: 2026-01-09
**状态**: 待实施

---

## 1. 核心架构哲学 (Core Philosophy)

我们采用 **"Hybrid Architecture" (混合架构)**：

* **布局与原子样式**：完全托管给 **Tailwind CSS**。
* **主题与组件皮肤**：完全托管给 **CSS Variables (CSS变量)**。
* **复杂交互逻辑**：仅在极少数情况下使用 **SCSS**。

**原则：**

> **"Write variables for themes, write classes for layout, never write hex codes."**
> (为主题写变量，为布局写类名，永远不要写十六进制颜色码。)

---

## 2. 目录结构标准 (Directory Structure)

所有样式资源必须严格遵循以下分层结构：

```text
src/styles/
├── theme/                  # [核心] 主题系统的唯一真理来源
│   ├── _palettes.scss      # 原始色板 (定义 hex 值，如 $blue-500: #58a6ff)
│   ├── _semantics.scss     # 语义变量 (定义 var(--bg-primary): $gray-900)
│   └── _element-map.scss   # 映射层 (将语义变量映射给 Element Plus)
├── tailwind.css            # Tailwind 入口文件
└── main.scss               # 应用入口 (仅引入上述文件，不写具体样式)
```

---

## 3. 强制开发规约 (The Iron Rules)

以下规则在 Code Review 中具有**一票否决权**。

### 3.1 布局规约 (Layout Rules)

* **禁止**：在 `<style>` 标签中手写标准布局属性（`display`, `flex`, `margin`, `padding`, `width`, `height`, `gap`）。
* **强制**：必须使用 **Tailwind Utility Classes**。

**对比示例：**

```html
<!-- 错误 -->
<div class="header-left">...</div>
<style scoped>
.header-left { display: flex; align-items: center; gap: 8px; margin-right: 16px; }
</style>

<!-- 正确 -->
<div class="flex items-center gap-2 mr-4">...</div>
```

### 3.2 颜色规约 (Color Rules)

* **禁止**：在代码的任何地方（HTML, CSS, JS）出现硬编码的 Hex 值（如 `#ffffff`, `#333`）。
* **强制**：
  * **Tailwind**: 使用语义化类名 (`bg-bg-primary`, `text-text-muted`)。
  * **CSS**: 使用 CSS 变量 (`var(--border-default)`)。

**理由**：任何硬编码颜色都会导致未来"明亮模式"或"高对比度模式"的开发工作量爆炸。

### 3.3 组件覆盖规约 (Override Rules)

* **禁止**：使用 `.el-xxx` 类名强行覆盖样式（如 `.el-button { background: red }`）。
* **禁止**：使用 `!important` 覆盖组件库样式。
* **禁止**：滥用 `:deep()` 修改组件内部结构（除非是无法通过变量控制的第三方黑盒组件）。
* **强制**：通过 **CSS 变量** 进行覆盖。

**正确做法示例：**

```vue
<div class="custom-context" style="--el-dialog-bg-color: var(--bg-tertiary)">
  <el-dialog ... />
</div>
```

---

## 4. 像素级精度处理标准 (Pixel Perfection Strategy)

针对设计稿中对 `px` 的不同需求，采用分级处理策略：

| 场景 | 处理方式 | 示例代码 |
| --- | --- | --- |
| **通用规范** | **配置化** (写入 `tailwind.config.js`) | `h-header` (48px), `min-w-sidebar` (280px) |
| **一次性微调** | **方括号语法** (Arbitrary Values) | `top-[3px]`, `w-[13px]` |
| **动态交互** | **Vue Style 绑定** (仅限动态属性) | `:style="{ width: sidebarWidth + 'px' }"` |

**严禁**：在 SCSS 中写死魔法数值（如 `width: 283px`），除非该数值是通过复杂的 `calc()` 计算且无法通过 Tailwind 表达的。

---

## 5. 复杂组件处理例外 (Complex Component Exception)

对于 **波形图 (WaveformTimeline)**、**视频舞台 (VideoStage)** 等高复杂度组件：

1. **容器层 (Container)**：必须使用 Tailwind 控制宽高、边距和定位。
2. **内容层 (Content)**：允许使用 `<style lang="scss">`，但必须遵循 **BEM 命名法**，且内部颜色必须引用 `var(--xxx)`。

**示例 (WaveformTimeline.vue)：**

```vue
<template>
  <div class="relative w-full h-[200px] bg-bg-secondary select-none">
    <canvas class="waveform__canvas" ... />
    <div class="waveform__cursor waveform__cursor--active" ...></div>
  </div>
</template>

<style lang="scss" scoped>
.waveform {
  &__cursor {
    position: absolute;
    top: 0;
    // 强制使用变量
    background-color: var(--waveform-cursor);

    &--active {
      box-shadow: 0 0 4px var(--primary);
    }
  }
}
</style>
```

---

## 6. 代码审查清单 (Code Review Checklist)

在提交代码前，请自查以下问题。如果存在任何一项"是"，则代码不达标：

1. [ ] 我是否写了 `.el-` 开头的 CSS 类来修改样式？
2. [ ] 我是否在文件中留下了 `#` 开头的颜色代码？
3. [ ] 我是否在 `<style>` 里写了 `flex` 或 `margin` 等布局代码？
4. [ ] 我是否使用了 `!important`？
5. [ ] 我是否为只用一次的 `div` 起了一个类似 `wrapper` 或 `container` 的模糊类名？

---

## 7. 迁移策略 (Migration Strategy)

不要试图一次性重构整个项目。

1. **新功能**：100% 遵循本规范。
2. **修改旧功能**：遵循 **"童子军军规" (Boy Scout Rule)** —— 离开营地时要比到达时更干净。每次修改旧组件时，顺手将其布局迁移到 Tailwind，将其颜色迁移到 CSS 变量。

---

## 8. CSS 变量命名规范

### 8.1 背景色

```scss
--bg-base       // 最深背景
--bg-primary    // 主背景
--bg-secondary  // 次级背景/卡片
--bg-tertiary   // 三级背景/悬停
--bg-elevated   // 浮层背景
--bg-overlay    // 遮罩层
```

### 8.2 文本色

```scss
--text-primary    // 主要文本
--text-normal     // 普通文本
--text-secondary  // 次要文本
--text-muted      // 更弱的文本
--text-disabled   // 禁用文本
```

### 8.3 语义色

```scss
--primary         // 主色（蓝）
--primary-hover   // 主色悬停
--primary-dim     // 主色深色

--success         // 成功（绿）
--warning         // 警告（橙）
--danger          // 危险（红）
```

### 8.4 阶段标记色（新增）

```scss
--stage-sensevoice   // SenseVoice 阶段（蓝）
--stage-whisper      // Whisper 阶段（绿）
--stage-llm          // LLM 阶段（紫）
```

### 8.5 状态色（新增）

```scss
--status-processing  // 处理中
--status-warning     // 警告
--status-error       // 错误
```

### 8.6 边框色

```scss
--border-default   // 默认边框
--border-muted     // 淡边框
--border-subtle    // 极淡边框
```

---

## 9. Tailwind 自定义配置

### 9.1 颜色映射

```javascript
// tailwind.config.js
colors: {
  // 背景
  'bg-base': 'var(--bg-base)',
  'bg-primary': 'var(--bg-primary)',
  'bg-secondary': 'var(--bg-secondary)',
  'bg-tertiary': 'var(--bg-tertiary)',

  // 文本
  'text-primary': 'var(--text-primary)',
  'text-normal': 'var(--text-normal)',
  'text-secondary': 'var(--text-secondary)',
  'text-muted': 'var(--text-muted)',

  // 语义色
  'primary': 'var(--primary)',
  'success': 'var(--success)',
  'warning': 'var(--warning)',
  'danger': 'var(--danger)',

  // 阶段标记
  'stage-sensevoice': 'var(--stage-sensevoice)',
  'stage-whisper': 'var(--stage-whisper)',
}
```

### 9.2 间距扩展

```javascript
spacing: {
  'header': '48px',
  'sidebar': '280px',
  'card-padding': '16px',
}
```

### 9.3 圆角映射

```javascript
borderRadius: {
  'xs': 'var(--radius-xs)',
  'sm': 'var(--radius-sm)',
  'md': 'var(--radius-md)',
  'lg': 'var(--radius-lg)',
  'xl': 'var(--radius-xl)',
}
```

---

## 10. Element Plus 定制规范

### 10.1 正确方式：CSS 变量覆盖

```scss
// src/styles/theme/_element-map.scss
:root {
  // Button
  --el-button-bg-color: var(--bg-secondary);
  --el-button-border-color: var(--border-default);
  --el-button-text-color: var(--text-normal);
  --el-button-hover-bg-color: var(--bg-tertiary);

  // Input
  --el-input-bg-color: var(--bg-secondary);
  --el-input-border-color: var(--border-default);
  --el-input-text-color: var(--text-normal);

  // Dialog
  --el-dialog-bg-color: var(--bg-secondary);
  --el-dialog-border-radius: var(--radius-lg);

  // Table
  --el-table-bg-color: var(--bg-secondary);
  --el-table-header-bg-color: var(--bg-secondary);
  --el-table-border-color: var(--border-default);
}
```

### 10.2 错误方式（禁止）

```scss
// 禁止：直接覆盖类名
.el-button {
  background: #1a1a1a !important;
}

// 禁止：使用 :deep() 覆盖内部结构
:deep(.el-table__header-wrapper) {
  th { background: #2d2d2d; }
}
```

---

## 11. 常用 Tailwind 类速查

### 11.1 Flex 布局

| 用途 | Tailwind 类 |
|------|-------------|
| 水平居中 | `flex items-center justify-center` |
| 垂直堆叠 | `flex flex-col` |
| 两端对齐 | `flex items-center justify-between` |
| 间距 8px | `gap-2` |
| 间距 16px | `gap-4` |

### 11.2 间距

| 用途 | Tailwind 类 |
|------|-------------|
| padding 8px | `p-2` |
| padding 16px | `p-4` |
| margin-top 8px | `mt-2` |
| margin-bottom 16px | `mb-4` |
| 水平 padding | `px-4` |
| 垂直 padding | `py-2` |

### 11.3 尺寸

| 用途 | Tailwind 类 |
|------|-------------|
| 全宽 | `w-full` |
| 全高 | `h-full` |
| 视口高度 | `h-screen` |
| 最小宽度 | `min-w-[280px]` |
| 固定高度 | `h-[48px]` |

### 11.4 定位

| 用途 | Tailwind 类 |
|------|-------------|
| 相对定位 | `relative` |
| 绝对定位 | `absolute` |
| 固定定位 | `fixed` |
| 左上角 | `top-0 left-0` |
| 居中 | `inset-0 m-auto` |

---

## 12. 工具链配置

### 12.1 Stylelint 配置

```json
// .stylelintrc.json
{
  "extends": ["stylelint-config-standard-scss"],
  "rules": {
    "color-no-hex": true,
    "declaration-no-important": true,
    "selector-max-specificity": ["0,3,0"],
    "scss/at-rule-no-unknown": ["error", {
      "ignoreAtRules": ["tailwind", "apply", "layer"]
    }]
  },
  "overrides": [{
    "files": ["**/*.vue"],
    "customSyntax": "postcss-html"
  }]
}
```

### 12.2 VS Code 扩展推荐

- **Tailwind CSS IntelliSense** - 类名自动补全
- **Stylelint** - 样式规范检查
- **SCSS IntelliSense** - SCSS 变量提示

---

## 更新历史

| 日期 | 版本 | 变更内容 |
|------|------|----------|
| 2026-01-09 | v1.0 | 初始版本 |
