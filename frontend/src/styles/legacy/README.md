# Legacy SCSS 归档目录

本目录包含已废弃的 SCSS 文件，保留用于历史参考。

## 归档时间
- 初始创建：2026-01-30（主题重构启动）
- 最终归档：2026-02-09（SCSS 完全清除）

## 归档原因
- 所有 Vue 组件已完成 SCSS → CSS 迁移
- 主题系统已迁移至 `src/theme/` (JavaScript Tokens)
- 全局样式已迁移至 `base.css` 和 `element-override.css`

## 文件清单

### 全局样式文件
- **`main.scss`** - 已废弃的全局样式入口（空文件）
  - 状态：已在 main.js 中注释掉导入
  - 功能：所有样式已迁移至 base.css 和 element-override.css

### SCSS 工具文件
- **`_mixins.scss`** (来自 styles/) - 临时 SCSS mixins（未被使用）
  - 包含：响应式断点、Flexbox 工具、文本工具、滚动条样式、交互反馈
  - 替代方案：Tailwind 工具类 + 原生 CSS

- **`_mixins.scss`** (原 legacy/) - 旧 SCSS mixins
  - 与上述文件内容相同（重复备份）

- **`_variables.scss`** - 旧 SCSS 变量和 CSS 变量定义
  - 包含：旧主题色彩系统、间距系统、字体系统、Z-Index 层级
  - 替代方案：`src/theme/tokens/dark.js` (JavaScript Design Tokens)

## 替代方案

### 旧系统 → 新系统对照表

| 旧系统 (SCSS) | 新系统 (Pure CSS + JS Tokens) |
|--------------|-------------------------------|
| `$bg-primary: #161b22` | `{ bg: { primary: '#161b22' } }` (JS) |
| `--bg-primary` | `--af-bg-primary` (动态注入) |
| `@mixin flex-center` | `flex justify-center items-center` (Tailwind) |
| `@include text-ellipsis` | `truncate` (Tailwind) |
| `@mixin mobile { ... }` | `@media (max-width: 767px) { ... }` (原生 CSS) |

### 新系统架构

```
src/
├── theme/
│   ├── tokens/
│   │   └── dark.js          ← Design Tokens (JavaScript)
│   └── utils/
│       └── inject.js         ← CSS 变量动态注入
├── styles/
│   ├── base.css              ← 基础样式
│   ├── element-override.css  ← Element Plus 覆盖
│   ├── tailwind.css          ← Tailwind 入口
│   └── legacy/               ← SCSS 归档（本目录）
└── components/
    └── **/*.vue              ← 纯 CSS（无 lang="scss"）
```

## 技术收益

✅ **零 SCSS 依赖**：无需 sass、stylelint-scss 等开发依赖
✅ **统一变量系统**：所有 CSS 变量使用 `--af-*` 命名空间
✅ **动态主题切换**：JavaScript Tokens 支持运行时主题切换
✅ **更快的构建**：无需编译 SCSS，直接使用原生 CSS
✅ **更简单的工具链**：仅需 PostCSS + Tailwind

## 注意事项

⚠️ **此目录仅供历史参考，请勿恢复使用**

如需查看旧样式定义，请参考：
- 变量定义 → `_variables.scss`
- Mixin 工具 → `_mixins.scss`

如需修改主题，请编辑：
- `src/theme/tokens/dark.js` (Design Tokens)
- `src/styles/base.css` (基础样式)

---

**归档负责人**：Claude Code
**归档日期**：2026-02-09
**重构分支**：refactor/v3.2-frontend
