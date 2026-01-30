# 大型复杂组件样式重构策略

> 本文档专门针对 500+ 行的大型 Vue 组件，提供安全、渐进式的样式重构策略。

---

## 一、当前项目大型组件清单

| 组件                         | 行数 | 样式行数 | 复杂度     | 优先级    |
| ---------------------------- | ---- | -------- | ---------- | --------- |
| `WaveformTimeline/index.vue` | 2239 | ~300     | ⭐⭐⭐⭐⭐ | P3 (最后) |
| `EditorView.vue`             | 1755 | ~200     | ⭐⭐⭐⭐   | P2        |
| `TaskListView.vue`           | 1584 | ~250     | ⭐⭐⭐⭐   | P2        |
| `VideoStage/index.vue`       | 1273 | ~150     | ⭐⭐⭐     | P2        |
| `PresetSelector.vue`         | 903  | ~180     | ⭐⭐⭐     | P1        |
| `SubtitleItem.vue`           | 841  | ~120     | ⭐⭐⭐     | P1        |
| `EditorHeader.vue`           | 814  | ~200     | ⭐⭐⭐     | P1        |
| `SubtitleList/index.vue`     | 627  | ~100     | ⭐⭐       | P1        |
| `AdvancedSettings.vue`       | 573  | ~80      | ⭐⭐       | P1        |

---

## 二、核心原则：分层渐进重构

### 2.1 黄金法则

```
┌─────────────────────────────────────────────────────────────┐
│  ❌ 错误做法：一次性重写整个组件的样式                         │
│  ✅ 正确做法：分区域、分批次，每次只动一小块                    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 安全边界

**绝对不能改的：**

- 组件的 `<script>` 逻辑部分
- `<template>` 的结构层级
- 已经稳定运行的交互行为
- 后端 API 调用

**可以安全改的：**

- `<style>` 块内的 CSS 规则
- 模板中的 `class` 属性值
- CSS 变量的使用方式

---

## 三、大型组件重构四步法

### Step 1：样式分区（Style Mapping）

首先，对组件的样式进行逻辑分区，绘制"样式地图"：

以 `WaveformTimeline` 为例：

```
┌─────────────────────────────────────────────────────────┐
│  .waveform-timeline (容器)                               │
│  ├── .timeline-header (头部控制栏)                        │
│  │   ├── .zoom-controls                                 │
│  │   │   ├── .zoom-btn                                  │
│  │   │   ├── .zoom-slider                               │
│  │   │   └── .zoom-label                                │
│  │   └── .time-indicator                                │
│  ├── #timeline (时间轴刻度)                              │
│  ├── .waveform-wrapper (波形容器) ⚠️ 复杂                 │
│  │   ├── .waveform-upper-zone                           │
│  │   ├── #waveform                                      │
│  │   ├── .waveform-loading                              │
│  │   └── .waveform-error                                │
│  └── .custom-scrollbar (自定义滚动条)                     │
│      ├── .scrollbar-track                               │
│      └── .scrollbar-thumb                               │
└─────────────────────────────────────────────────────────┘
```

**分区标记：**

- 🟢 **简单区域**：独立、无复杂交互（如 `.timeline-header`）
- 🟡 **中等区域**：有一定嵌套（如 `.custom-scrollbar`）
- 🔴 **复杂区域**：涉及第三方库、动态样式（如 `#waveform`、`:deep()`）

### Step 2：创建过渡层（Transition Layer）

在不删除原有样式的情况下，逐步添加新的样式：

```vue
<style lang="scss" scoped>
/* ========================================
 * 样式重构进度追踪
 * ========================================
 * [✅] .timeline-header - 2026-01-30 已迁移到 CSS 变量
 * [🔄] .zoom-controls   - 进行中
 * [⏳] .waveform-wrapper - 待处理
 * [⏳] .custom-scrollbar - 待处理
 * ======================================== */

/* === 已迁移区域 === */
.timeline-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--af-spacing-sm) var(--af-spacing-lg);
  background: var(--af-bg-tertiary);
  border-bottom: 1px solid var(--af-border-default);
}

/* === 待迁移区域（保留原样式）=== */
.zoom-controls {
  /* 原有样式暂时保留 */
  display: flex;
  align-items: center;
  gap: 8px; /* TODO: 替换为 var(--af-spacing-sm) */

  .zoom-btn,
  .fit-btn {
    width: 28px; /* TODO: 替换为 var(--af-size-btn-sm) */
    height: 28px;
    /* ... */
  }
}
</style>
```

### Step 3：逐区域迁移

**迁移顺序策略（由简到难）：**

```
Phase 1: 静态展示区域
         ↓
Phase 2: 简单交互区域
         ↓
Phase 3: 复杂交互区域
         ↓
Phase 4: 第三方库集成区域 (:deep)
```

**单区域迁移检查清单：**

- [ ] 硬编码颜色 → CSS 变量
- [ ] 硬编码尺寸 → CSS 变量或 Tailwind
- [ ] SCSS 变量 → CSS 变量
- [ ] SCSS 嵌套 → 平铺或保留（如果简洁）
- [ ] `!important` → 尽量移除
- [ ] `:deep()` → 评估是否可移除

### Step 4：验证与回归测试

每迁移一个区域后，必须验证：

```bash
# 1. 视觉检查
- 在浅色/深色主题下查看
- 检查 hover/active/focus 状态
- 检查响应式表现

# 2. 功能检查
- 所有交互行为正常
- 动画/过渡效果正常
- 无 JS 控制台错误

# 3. 回归检查
- 未修改区域样式无变化
```

---

## 四、具体案例：WaveformTimeline 重构计划

### 4.1 分批次执行计划

| 批次    | 区域                                    | 风险 | 预计时间 |
| ------- | --------------------------------------- | ---- | -------- |
| Batch 1 | `.timeline-header` + `.time-indicator`  | 低   | 30min    |
| Batch 2 | `.zoom-controls` (含 range input)       | 中   | 45min    |
| Batch 3 | `.custom-scrollbar`                     | 中   | 30min    |
| Batch 4 | `.waveform-loading` + `.waveform-error` | 低   | 20min    |
| Batch 5 | `.waveform-wrapper` 基础样式            | 中   | 30min    |
| Batch 6 | `#waveform :deep()` 规则                | 高   | 60min    |

**总计：约 3.5-4 小时（不含测试时间）**

### 4.2 Batch 1 详细示例

**原始代码：**

```scss
.timeline-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 16px;
  background: var(--bg-tertiary);
  border-bottom: 1px solid var(--border-default);
}

.time-indicator {
  display: flex;
  align-items: center;
  gap: 6px;
  font-family: var(--font-mono);
  font-size: 13px;

  .current-time {
    color: var(--primary);
    font-weight: 600;
  }

  .separator {
    color: var(--text-muted);
  }

  .total-time {
    color: var(--text-secondary);
  }
}
```

**迁移后代码：**

```vue
<template>
  <!-- 添加 Tailwind 类到模板 -->
  <div
    class="timeline-header tw-flex tw-items-center tw-justify-between tw-px-4 tw-py-1.5"
  >
    <!-- ... -->
    <div
      class="time-indicator tw-flex tw-items-center tw-gap-1.5 tw-font-mono tw-text-sm"
    >
      <span class="tw-text-primary tw-font-semibold">{{
        formatTime(currentTime)
      }}</span>
      <span class="tw-text-muted">/</span>
      <span class="tw-text-secondary">{{ formatTime(duration) }}</span>
    </div>
  </div>
</template>

<style scoped>
/* 保留必要的自定义样式 */
.timeline-header {
  background: var(--af-bg-tertiary);
  border-bottom: 1px solid var(--af-border-default);
}

/* 注释掉已迁移的样式，保留一段时间以便回滚 */
/*
.time-indicator {
  旧样式已迁移到 Tailwind
}
*/
</style>
```

### 4.3 处理 `:deep()` 规则的策略

对于 `WaveformTimeline` 中的第三方库样式覆盖：

```scss
// 原始代码
:deep(.wavesurfer-cursor) {
  filter: drop-shadow(0 0 1px rgba(0, 0, 0, 0.8))
    drop-shadow(0 0 2px rgba(255, 255, 255, 0.5))
    drop-shadow(0 0 4px rgba(248, 81, 73, 0.6));
}

:deep(.wavesurfer-region) {
  border-radius: 2px;
  transition: background-color 0.2s;

  &:hover {
    background-color: rgba(88, 166, 255, 0.4) !important;
  }
}

:deep(.wavesurfer-handle) {
  background: var(--primary) !important;
  width: 4px !important;
  border-radius: 2px;
}
```

**处理策略：**

**方案 A：保留 `:deep()` 但使用 CSS 变量**

```scss
:deep(.wavesurfer-region) {
  border-radius: var(--af-radius-sm);
  transition: background-color var(--af-transition-fast);

  &:hover {
    background-color: var(--af-waveform-region-hover) !important;
  }
}

:deep(.wavesurfer-handle) {
  background: var(--af-color-primary) !important;
  width: 4px !important;
  border-radius: var(--af-radius-sm);
}
```

**方案 B：提取到全局样式文件（推荐用于复用）**

```css
/* src/styles/vendors/wavesurfer.css */
.wavesurfer-cursor {
  filter: drop-shadow(0 0 1px rgba(0, 0, 0, 0.8))
    drop-shadow(0 0 2px rgba(255, 255, 255, 0.5))
    drop-shadow(0 0 4px var(--af-waveform-cursor-glow));
}

.wavesurfer-region {
  border-radius: var(--af-radius-sm);
  transition: background-color var(--af-transition-fast);
}

.wavesurfer-region:hover {
  background-color: var(--af-waveform-region-hover) !important;
}

.wavesurfer-handle {
  background: var(--af-color-primary) !important;
  width: 4px !important;
  border-radius: var(--af-radius-sm);
}
```

然后在 `main.js` 中导入：

```js
import "./styles/vendors/wavesurfer.css";
```

---

## 五、大型组件专用 Design Tokens

对于复杂组件，建议创建组件级 Design Tokens：

```js
// src/theme/tokens/components/waveform.js
export const waveformTokens = {
  // 波形容器
  "--af-waveform-bg": "var(--af-bg-secondary)",
  "--af-waveform-header-bg": "var(--af-bg-tertiary)",

  // 时间轴
  "--af-waveform-timeline-height": "18px",
  "--af-waveform-timeline-bg": "var(--af-bg-tertiary)",

  // Region 样式
  "--af-waveform-region-bg": "rgba(88, 166, 255, 0.2)",
  "--af-waveform-region-hover": "rgba(88, 166, 255, 0.4)",
  "--af-waveform-region-selected": "rgba(88, 166, 255, 0.5)",

  // 光标
  "--af-waveform-cursor-color": "#f85149",
  "--af-waveform-cursor-glow": "rgba(248, 81, 73, 0.6)",

  // 滚动条
  "--af-waveform-scrollbar-height": "14px",
  "--af-waveform-scrollbar-track": "transparent",
  "--af-waveform-scrollbar-thumb": "rgba(139, 148, 158, 0.2)",
  "--af-waveform-scrollbar-thumb-hover": "rgba(139, 148, 158, 0.5)",
  "--af-waveform-scrollbar-thumb-active": "rgba(139, 148, 158, 0.8)",
};
```

---

## 六、渐进式样式块管理

### 6.1 样式块拆分策略

对于 300+ 行的 `<style>` 块，可以拆分为多个逻辑文件：

```
src/components/editor/WaveformTimeline/
├── index.vue                 # 主组件
├── styles/
│   ├── index.css            # 入口文件，导入所有样式
│   ├── header.css           # .timeline-header 相关
│   ├── waveform.css         # .waveform-wrapper 相关
│   ├── scrollbar.css        # .custom-scrollbar 相关
│   └── states.css           # 加载/错误状态
```

**index.vue 中引用：**

```vue
<style scoped>
@import "./styles/index.css";
</style>
```

**styles/index.css：**

```css
@import "./header.css";
@import "./waveform.css";
@import "./scrollbar.css";
@import "./states.css";
```

### 6.2 样式块拆分的好处

1. **更好的可维护性**：每个文件职责单一
2. **方便团队协作**：不同人可以同时编辑不同样式文件
3. **渐进式迁移**：可以逐个文件迁移，不影响整体
4. **更好的 Git 历史**：更清晰地追踪每个区域的变化

---

## 七、保守策略：样式冻结区

对于风险极高的区域，可以标记为"冻结区"，暂不迁移：

```scss
/* ╔══════════════════════════════════════════════════════════╗
   ║  🔒 FROZEN ZONE - 请勿修改                                ║
   ║  原因：涉及 WaveSurfer.js 内部 DOM 结构                    ║
   ║  责任人：需要与库版本升级一起处理                           ║
   ╚══════════════════════════════════════════════════════════╝ */

:deep(.wavesurfer-cursor) {
  /* 保持原样 */
}

:deep(.wavesurfer-region) {
  /* 保持原样 */
}

/* ═══════════════════ END FROZEN ZONE ═══════════════════════ */
```

---

## 八、重构进度追踪模板

为每个大型组件创建追踪文档：

```markdown
# WaveformTimeline 样式重构追踪

## 总览

- 开始日期：2026-01-30
- 预计完成：2026-02-05
- 负责人：XXX

## 进度

| 区域              | 状态      | 完成日期 | 备注                     |
| ----------------- | --------- | -------- | ------------------------ |
| .timeline-header  | ✅ 完成   | 01-30    |                          |
| .zoom-controls    | ✅ 完成   | 01-31    | range input 需要特殊处理 |
| .time-indicator   | ✅ 完成   | 01-30    |                          |
| .custom-scrollbar | 🔄 进行中 | -        |                          |
| .waveform-wrapper | ⏳ 待开始 | -        |                          |
| .waveform-loading | ⏳ 待开始 | -        |                          |
| .waveform-error   | ⏳ 待开始 | -        |                          |
| :deep() 规则      | 🔒 冻结   | -        | 等待 WaveSurfer 升级     |

## 问题记录

1. ...
2. ...

## 回滚点

- commit: abc123 - 迁移前完整备份
- commit: def456 - Phase 1 完成
```

---

## 九、总结：大型组件重构决策树

```
开始重构大型组件
        │
        ▼
   ┌────────────────┐
   │ 1. 绘制样式地图 │
   └────────────────┘
        │
        ▼
   ┌────────────────┐
   │ 2. 标记风险区域 │
   │  🟢 低 🟡 中 🔴 高 │
   └────────────────┘
        │
        ▼
   ┌────────────────────┐
   │ 3. 从 🟢 低风险开始 │
   └────────────────────┘
        │
        ▼
   ┌────────────────────────────┐
   │ 4. 逐区域迁移               │
   │   - 添加 Tailwind 类        │
   │   - 替换为 CSS 变量         │
   │   - 保留原样式（注释）       │
   └────────────────────────────┘
        │
        ▼
   ┌────────────────┐
   │ 5. 验证 + 测试  │
   └────────────────┘
        │
    通过？ ──否──▶ 回滚到上一个稳定版本
        │
       是
        │
        ▼
   ┌────────────────┐
   │ 6. 提交 + 记录 │
   └────────────────┘
        │
        ▼
   ┌────────────────────┐
   │ 7. 下一个区域       │
   │   还有更多？──是──▶ 返回步骤 4
   └────────────────────┘
        │
       否
        │
        ▼
   ┌────────────────────────┐
   │ 8. 清理注释掉的旧样式   │
   │    移除 lang="scss"    │
   └────────────────────────┘
        │
        ▼
      ✅ 完成
```

---

## 十、建议的重构顺序

基于风险评估，推荐以下顺序：

### 第一批（练手 + 建立信心）

1. `AdvancedSettings.vue` - 573行，样式相对简单
2. `SubtitleList/index.vue` - 627行，逻辑清晰

### 第二批（中等复杂度）

3. `SubtitleItem.vue` - 841行
4. `EditorHeader.vue` - 814行
5. `PresetSelector.vue` - 903行

### 第三批（高复杂度）

6. `VideoStage/index.vue` - 1273行
7. `TaskListView.vue` - 1584行

### 第四批（最高复杂度，最后处理）

8. `EditorView.vue` - 1755行
9. `WaveformTimeline/index.vue` - 2239行

---

> **记住：重构的目标是改进，不是重写。保持每一步都可以独立验证和回滚。**
