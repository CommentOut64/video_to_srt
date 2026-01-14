# AnchorFlux 前端样式重构计划

**版本**: V3.1.2+dev.20260109.01
**创建日期**: 2026-01-09
**状态**: 待执行

---

## 1. 执行摘要

本文档描述 AnchorFlux 前端的样式系统重构计划，目标是实施 **AnchorFlux 前端开发样式规范 v1.0**，采用 **Tailwind CSS + CSS Variables + SCSS (极少)** 的混合架构。

### 1.1 当前状态

| 维度 | 现状 | 目标状态 |
|------|------|----------|
| **CSS 架构** | 纯 SCSS + Element Plus | Tailwind + CSS Variables + SCSS (极少) |
| **Tailwind** | 未安装 | 核心布局引擎 |
| **CSS Variables** | 已有良好基础 | 扩展为完整主题系统 |
| **组件数量** | 17 个 .vue 文件 | 不变 |
| **违规总数** | 287 处 | 0 处 |

### 1.2 违规分布

| 违规类型 | 数量 | 占比 |
|----------|------|------|
| 硬编码 Hex 颜色 | 21 | 7.3% |
| 手写布局属性 | 186 | 64.8% |
| 直接覆盖 Element Plus | 52 | 18.1% |
| 使用 !important | 31 | 10.8% |
| 滥用 :deep() | 2 | 0.7% |

---

## 2. 组件复杂度分析

### 2.1 按行数排序

| 排名 | 组件 | 总行数 | 样式占比 | 方法数 | 复杂度 |
|---:|------|-----:|-------:|-----:|---------|
| 1 | WaveformTimeline/index.vue | 2136 | 14.3% | 75 | **严重过大** |
| 2 | TaskListView.vue | 1758 | 39.4% | 22 | **过大** |
| 3 | EditorView.vue | 1750 | 18.5% | 64 | **过大** |
| 4 | VideoStage/index.vue | 1277 | 32.3% | 27 | **过大** |
| 5 | PresetSelector.vue | 962 | 41.8% | 9 | **过大** |
| 6 | SubtitleItem.vue | 862 | 44.3% | 19 | **过大** |
| 7 | EditorHeader.vue | 848 | 45.8% | 4 | **过大** |
| 8 | SubtitleList/index.vue | 611 | **65.3%** | 9 | **过大** |
| 9 | PlaybackControls/index.vue | 564 | 45.4% | 14 | **过大** |
| 10 | TaskCard.vue | 462 | 51.7% | 8 | 正常 |
| 11 | TaskMonitor/index.vue | 370 | 44.6% | 2 | 正常 |

### 2.2 核心问题

1. **样式占比过高**（平均 36-65%）
   - SubtitleList: 65.3% - 最严重
   - EditorHeader: 45.8%
   - SubtitleItem: 44.3%

2. **脚本逻辑过度集中**
   - WaveformTimeline: 1727 行脚本，75 个方法
   - EditorView: 1245 行脚本，64 个方法，19 个 computed

3. **关键违规文件**
   - TaskListView: `:deep()` 16 处，`!important` 9 处
   - UpdateDialog: `!important` 12 处

---

## 3. 重构策略

### 3.1 三类处理策略

#### 策略 A：样式重构时同步解耦

适用于**样式和逻辑边界清晰**的组件：

| 组件 | 拆分方案 | 预期行数 |
|------|----------|----------|
| **PresetSelector.vue** | MacroPresetCards + ModuleCard + PresetSelectorCore | 962 → ~450 |
| **SubtitleItem.vue** | SubtitleTextEditor + SubtitleTimeInput + SubtitleMetadata | 862 → ~400 |
| **VideoStage/index.vue** | VideoPlayer + VideoStateOverlay + SubtitleOverlay + ProxyIndicator | 1277 → ~600 |

#### 策略 B：纯样式重构（逻辑已足够简洁）

| 组件 | 样式占比 | 预期效果 |
|------|----------|----------|
| **SubtitleList/index.vue** | 65.3% → 30% | 减少 ~35% 代码 |
| **EditorHeader.vue** | 45.8% → 25% | 减少 ~20% 代码 |
| **PlaybackControls/index.vue** | 45.4% → 25% | 减少 ~20% 代码 |

#### 策略 C：先样式重构，后期独立解耦

| 组件 | 复杂度原因 | 建议时机 |
|------|------------|----------|
| **WaveformTimeline** (2136行) | 75 个方法，复杂波形交互 | 独立 Sprint |
| **EditorView.vue** (1750行) | SSE/加载/导出，19 个 computed | 需架构设计 |
| **TaskListView.vue** (1758行) | 上传对话框、批量操作 | 提取对话框 |

---

## 4. 实施计划

### 4.1 总工时预估

| 阶段 | 内容 | 工时 |
|------|------|------|
| Phase 0 | Tailwind 基础设施搭建 | 2-3h |
| Phase 1 | 主题系统重构 | 3-4h |
| Phase 2A | 纯样式重构（3 个组件） | 4-6h |
| Phase 2B | 样式重构 + 同步解耦（3 个组件） | 8-10h |
| Phase 3 | 关键违规修复（TaskListView, UpdateDialog） | 4-6h |
| Phase 4 | 验证、文档、Lint 配置 | 2-3h |
| **合计** | **样式重构 + 轻量解耦** | **25-30h** |

### 4.2 详细时间线

```
Week 1: 基础设施 + 主题系统 + 策略B组件
├── Day 1-2: Phase 0 + Phase 1
│   ├── 安装 Tailwind CSS
│   ├── 创建 tailwind.config.js
│   ├── 重构 src/styles/ 目录结构
│   └── 迁移 _variables.scss 到新主题系统
│
├── Day 3-4: Phase 2A（纯样式重构）
│   ├── SubtitleList/index.vue (65.3% → 30%)
│   ├── EditorHeader.vue (45.8% → 25%)
│   └── PlaybackControls/index.vue (45.4% → 25%)
│
└── Day 5: Phase 3（关键违规修复）
    ├── TaskListView.vue（移除 :deep() 和 !important）
    └── UpdateDialog.vue（Element Plus 变量覆盖）

Week 2: 策略A组件（样式重构 + 同步解耦）
├── Day 1-2: PresetSelector.vue
│   ├── 提取 MacroPresetCards 组件
│   ├── 提取 ModuleCard 组件
│   └── 样式迁移到 Tailwind
│
├── Day 3-4: SubtitleItem.vue + VideoStage/index.vue
│   ├── SubtitleItem → 4 个子组件
│   └── VideoStage → 4 个子组件
│
└── Day 5: Phase 4（验证和完善）
    ├── 配置 Stylelint
    ├── 全面测试
    └── 更新文档
```

---

## 5. 技术规范

### 5.1 目录结构

重构后的样式目录结构：

```
src/styles/
├── theme/                  # [核心] 主题系统
│   ├── _palettes.scss      # 原始色板
│   ├── _semantics.scss     # 语义变量
│   └── _element-map.scss   # Element Plus 映射
├── tailwind.css            # Tailwind 入口
└── main.scss               # 应用入口
```

### 5.2 Tailwind 配置

```javascript
// frontend/tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts}'],
  theme: {
    extend: {
      colors: {
        // 背景层次
        'bg-base': 'var(--bg-base)',
        'bg-primary': 'var(--bg-primary)',
        'bg-secondary': 'var(--bg-secondary)',
        'bg-tertiary': 'var(--bg-tertiary)',
        // 文本层次
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
      },
      spacing: {
        'header': '48px',
        'sidebar': '280px',
      },
      borderRadius: {
        'xs': 'var(--radius-xs)',
        'sm': 'var(--radius-sm)',
        'md': 'var(--radius-md)',
        'lg': 'var(--radius-lg)',
      },
    },
  },
  plugins: [],
}
```

### 5.3 新增语义变量

```scss
// src/styles/theme/_semantics.scss
:root {
  // 阶段标记（新增）
  --stage-sensevoice: #58a6ff;
  --stage-whisper: #3fb950;
  --stage-llm: #a371f7;

  // 状态色（新增）
  --status-processing: #028AC5;
  --status-warning: #e67700;
  --status-error: #f85149;

  // 视频舞台专用（新增）
  --video-bg: #000;
  --video-text: #fff;
}
```

### 5.4 代码示例

#### Before（EditorView.vue）
```vue
<template>
  <div class="editor-view">
    <div class="header-row">...</div>
  </div>
</template>

<style lang="scss" scoped>
.editor-view {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--bg-base);
  overflow: hidden;
}
.header-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
}
</style>
```

#### After
```vue
<template>
  <div class="flex flex-col h-screen bg-bg-base overflow-hidden">
    <div class="flex items-center gap-2 px-4 py-2">...</div>
  </div>
</template>

<style lang="scss" scoped>
/* 无基础布局样式，仅保留复杂交互 */
</style>
```

---

## 6. 组件拆分详情

### 6.1 PresetSelector.vue 拆分方案

**当前**: 962 行（样式 41.8%）

**拆分后**:
```
PresetSelector/
├── index.vue              # 主容器 (~300 行)
├── MacroPresetCards.vue   # 快捷预设卡片 (~180 行)
├── ModuleCard.vue         # 可复用模块卡片 (~150 行)
└── styles/
    └── preset.scss        # 共享样式 (~150 行)
```

### 6.2 SubtitleItem.vue 拆分方案

**当前**: 862 行（样式 44.3%）

**拆分后**:
```
SubtitleItem/
├── index.vue              # 主容器 (~250 行)
├── TextEditor.vue         # 三态文本编辑器 (~180 行)
├── TimeInput.vue          # 时间戳输入 (~120 行)
├── MetadataBadge.vue      # 置信度和状态 (~100 行)
└── WarningBanner.vue      # 警告提示条 (~80 行)
```

### 6.3 VideoStage/index.vue 拆分方案

**当前**: 1277 行（样式 32.3%）

**拆分后**:
```
VideoStage/
├── index.vue              # 主容器 (~400 行)
├── VideoPlayer.vue        # 核心播放器 (~250 行)
├── StateOverlay.vue       # 状态覆盖层 (~180 行)
├── SubtitleOverlay.vue    # 字幕浮层 (~150 行)
└── ProxyIndicator.vue     # 分辨率指示器 (~100 行)
```

---

## 7. 后续计划：复杂组件解耦

以下组件建议在样式重构完成后，独立进行逻辑解耦：

### 7.1 WaveformTimeline（P0，独立 Sprint）

**当前**: 2136 行，75 个方法

**拆分方案**:
1. **WaveformRenderer** - WaveSurfer 初始化/销毁 (~200 行)
2. **WaveformScrollbar** - 自定义滚动条 (~150 行)
3. **WaveformZoomControls** - 缩放控制 (~100 行)
4. **RegionInteractionHandler** - 区域拖拽逻辑 (~300 行)
5. **WaveformContextMenu** - 右键菜单 (~150 行)

**预估工时**: 8-10h

### 7.2 EditorView.vue（P1，需架构设计）

**当前**: 1750 行，64 个方法，19 个 computed

**拆分方案**:
1. **SSEManager** - SSE 连接和事件处理 (~250 行)
2. **ProjectDataLoader** - 项目数据加载 (~250 行)
3. **SubtitleStreamHandler** - 字幕流式更新 (~300 行)
4. **ExportManager** - 导出功能 (~200 行)

**预估工时**: 10-12h

### 7.3 TaskListView.vue（P1，提取对话框）

**当前**: 1758 行

**拆分方案**:
1. **UploadDialog** - 上传对话框 (~250 行)
2. **TaskBatchActions** - 批量操作 (~150 行)

**预估工时**: 4-6h

---

## 8. 验收标准

### 8.1 代码指标

| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| 硬编码颜色 | 21 | 0 |
| !important 用法 | 31 | < 3 |
| :deep() 用法 | 52 | < 5 |
| 单文件最大行数 | 2136 | < 800 |
| 样式平均占比 | 42% | < 30% |

### 8.2 功能验收

- [ ] 所有现有功能正常工作
- [ ] 视觉样式无回归
- [ ] Element Plus 组件主题正确
- [ ] 响应式布局正常
- [ ] 波形图交互正常
- [ ] 字幕编辑正常

### 8.3 质量验收

- [ ] Stylelint 检查通过
- [ ] 无硬编码 Hex 颜色
- [ ] 无 `!important`（除必要情况）
- [ ] CSS 变量命名规范
- [ ] Tailwind 类名使用正确

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Element Plus 兼容 | 升级时可能出问题 | 使用官方 CSS 变量覆盖 |
| Tailwind 学习曲线 | 初期效率下降 | 提供速查表和示例 |
| WaveformTimeline 特殊性 | 波形图颜色从 JS 传入 | 创建主题配置对象 |
| 构建体积 | Tailwind 可能增加体积 | PurgeCSS 自动清理 |
| 回归风险 | 样式变更可能影响 UI | 逐组件迁移，充分测试 |

---

## 10. 相关文档

- [前端样式违规分析报告](../../llmdoc/agent/frontend-style-analysis-report.md)
- [Vue 组件复杂度分析报告](../../llmdoc/agent/vue-component-complexity-analysis.md)
- [AnchorFlux 前端开发样式规范 v1.0](./anchorflux-style-guide-v1.md)（待创建）

---

## 更新历史

| 日期 | 版本 | 变更内容 |
|------|------|----------|
| 2026-01-09 | V3.1.2+dev.20260109.01 | 初始版本，完成分析和计划制定 |
