# Task7: 任务创建设置区 Tab 化重构方案

> 版本: V3.2.4+dev.20260224.01
> 状态: 草案
> 涉及范围: 前端 TaskCreateDialog / PresetSelector 重构 + 后端自定义预设 API

---

## 1. 背景与目标

### 1.1 现状问题

当前 `TaskCreateDialog` 的"转录设置"区域采用**折叠面板 + 3 列模块卡片**的布局：

- 展开后内容过长，推动 dialog 整体向下滚动
- 随着配置参数增多（语言检测、说话人策略、切分配置等），3 列布局空间不足
- 无法模板化支持不同任务模式（转录 vs 字幕导入编辑）

### 1.2 目标

将"转录设置"区域重构为**固定高度、始终可见的 Tab 容器**，实现：

1. **始终展示**: 设置区不再折叠，Tab 头和内容区始终可见
2. **Tab 化分组**: 按用户认知分为 4 个 Tab（预设 / 前处理 / 转录 / 后处理）
3. **紧凑高度**: 精确计算像素预算，dialog 总高度 <= 640px，不触发主界面滚动
4. **自由导航**: 所有 Tab 可随意跳转，无线性步骤约束
5. **创建按钮始终可见**: 无论当前在哪个 Tab，dialog footer 始终可用
6. **样式统一**: 遵循项目 Design Token 体系和 `frontend-dev-rules.md`
7. **自定义预设持久化**: localStorage 为缓存层，后端为真相源，懒同步

### 1.3 设计依据

调研 HandBrake / DaVinci Resolve / Adobe Media Encoder / OBS Studio 等工具后确认：
视频/音频处理工具**一致采用自由导航 Tab + 始终可见的提交按钮**。
线性 Wizard 仅适用于高成本、强依赖的云平台首次配置场景，不适用于本项目。

---

## 2. 整体布局与高度预算

### 2.1 布局结构

```
+---------------------------------------------------+
|  创建任务                                      [X] |  <- dialog header
+---------------------------------------------------+
|  [直接上传] [从本地目录选择]                        |  <- 文件选择 tabs
|  ┌───────────────────────────────────────────────┐ |
|  │  拖拽视频文件到此处，或 点击选择                │ |  <- 上传区 (固定)
|  └───────────────────────────────────────────────┘ |
|  支持 MP4, AVI, MOV 等格式                         |  <- tip / tags
|  ─────────────────────────────────────────────── ─ |
|  [预设]  [前处理]  [转录]  [后处理]                 |  <- 设置 tabs (始终可见)
|  ┌───────────────────────────────────────────────┐ |
|  │  (当前 Tab 内容，固定高度，内部可滚动)          │ |
|  └───────────────────────────────────────────────┘ |
+---------------------------------------------------+
|  已选择 N 个文件     [保存预设]  [创建 N 个任务]    |  <- footer (始终可见)
+---------------------------------------------------+
```

**核心原则**: 设置区始终展开，不再有折叠/展开状态。原 `.transcription-settings > .settings-header`（折叠头 + ArrowDown 图标）完全移除，由 el-tabs 的 Tab 头直接取代。

### 2.2 高度预算（像素级精确计算）

**目标**: dialog 总高度 <= 640px，确保在 768p 笔记本屏幕（可用视口约 668px）下不溢出。

```
区域                          高度(px)    来源
────────────────────────────────────────────────
el-dialog header              54          title + padding
el-dialog body padding-top    30          Element Plus 默认
─ 文件选择 tabs header        40          el-tabs__header
─ 文件选择 tabs margin-bottom 16          现有 CSS
─ 上传区内容                  218         dragger(180) + tip/tags(30+8)
─ tabs-container margin-bottom 12         缩减: 20 → 12
─ 设置区分隔                  8           margin-top
─ 设置 tabs header            36          el-tabs__header (紧凑)
─ 设置 tabs 内容区            160         固定高度，内部滚动
el-dialog body padding-bottom 20          缩减: 30 → 20
el-dialog footer              52          按钮行 + padding
────────────────────────────────────────────────
总计                          636px
```

### 2.3 关键约束

| 项目 | 规格 | 说明 |
|------|------|------|
| dialog 宽度 | 700px | 保持不变 |
| 上传区域 | 218px | 保持不变（dragger 180 + tip/tags 38） |
| 设置 Tab header | 36px | 紧凑模式: `--el-tabs-header-height: 36px` |
| 设置 Tab 内容区 | **160px** | 固定高度，`overflow-y: auto` |
| dialog body padding | 30px top / 20px bottom | 底部缩减以节省空间 |
| dialog 总高度 | <= 640px | 适配 768p 笔记本 |
| footer | 始终可见 | 文件计数 + 保存预设 + 创建任务 |

### 2.4 设置 Tab 内容区 160px 的空间验证

每个 SettingRow 高度 = padding(8+8) + content(~20) = **36px**，含 1px 分隔线。
160px 内容区可容纳 **4 行设置项**无需滚动，超出则自动出现滚动条。

- 预设 Tab: 3 张内置卡片(~70px) + 自定义预设区 → 刚好或微滚动
- 前处理 Tab: 3 行设置 → 不需滚动
- 转录 Tab: 3 行设置 → 不需滚动
- 后处理 Tab: 2 行设置 → 不需滚动

---

## 3. 各 Tab 内容设计

### 3.1 Tab 1: 预设

**职责**: 快速选择预设方案，或从自定义预设列表中选择。

```
+-----------------------------------------------+
|  内置预设                                       |
|  +----------+ +----------+ +----------+        |
|  | 极速预览  | | 智能均衡  | | 影视精修  |        |
|  | 会议记录  | | 短视频..  | | 电影压制  |        |
|  +----------+ +----------+ +----------+        |
|                                                |
|  自定义预设                                     |
|  +----------+ +----------+                     |
|  | 我的日语  | | 会议专用  |  (用户自建的预设)    |
|  +----------+ +----------+                     |
+-----------------------------------------------+
```

**组件**: `PresetTab.vue`

**交互逻辑**:
- 点击内置预设卡片 → 填充所有 Tab 的参数
- 点击自定义预设卡片 → 填充所有 Tab 的参数
- 自定义预设卡片右上角有删除按钮（hover 显示）
- 选中预设后，用户可直接点击 footer 的"创建任务"按钮
- 预设卡片复用现有 `.preset-card` 样式体系

**数据来源**:
- 内置预设: 硬编码在前端（fast / balanced / quality）
- 自定义预设: localStorage 优先 → 后端 fallback

### 3.2 Tab 2: 前处理

**职责**: 音频预检和语言检测配置。

```
+-----------------------------------------------+
|  音频预检                                       |
|  ┌───────────────┐  ┌────────────────────────┐ |
|  │ 人声分离策略    │  │ [智能分诊 v]            │ |
|  └───────────────┘  └────────────────────────┘ |
|                                                |
|  语言检测                    (待实现，灰色状态)   |
|  ┌───────────────┐  ┌────────────────────────┐ |
|  │ 检测模式       │  │ [均衡 v]     (禁用)     │ |
|  └───────────────┘  └────────────────────────┘ |
|  ┌───────────────┐  ┌────────────────────────┐ |
|  │ 指定语言       │  │ [自动检测 v]  (禁用)    │ |
|  └───────────────┘  └────────────────────────┘ |
+-----------------------------------------------+
```

**组件**: `PreprocessTab.vue`

**表单项**:

| 字段 | 控件类型 | 映射到 taskConfig | 当前状态 |
|------|---------|------------------|---------|
| 人声分离策略 | el-select | `preprocessing.demucs_strategy` | 可用 |
| 语言检测模式 | el-select | `preprocessing.language_detection_mode` | 灰色占位 |
| 指定语言 | el-select（多选） | `preprocessing.langid_whitelist` | 灰色占位 |

**说明**:
- "语言检测"相关控件渲染但 disabled，等待后端 API 实现后激活
- 人声分离策略的选项：禁用(off) / 智能分诊(auto) / 强制分离(force_on)
- 不暴露 `demucs_model`、`demucs_shifts`、`spectrum_threshold` 等高级参数（保留在代码中，后续可扩展为"高级"折叠区）

### 3.3 Tab 3: 转录

**职责**: 转录引擎和说话人策略配置。

```
+-----------------------------------------------+
|  转录模式                                       |
|  ┌───────────────┐  ┌────────────────────────┐ |
|  │ 转录模式       │  │ [SV + Whisper 复核 v]   │ |
|  └───────────────┘  └────────────────────────┘ |
|                                                |
|  说话人策略                                     |
|  ┌───────────────┐  ┌────────────────────────┐ |
|  │ 说话人检测     │  │ [x] 启用               │ |
|  └───────────────┘  └────────────────────────┘ |
|  ┌───────────────┐  ┌────────────────────────┐ |
|  │ 说话人数       │  │ [0]  (0=自动识别)       │ |
|  └───────────────┘  └────────────────────────┘ |
+-----------------------------------------------+
```

**组件**: `TranscriptionTab.vue`

**表单项**:

| 字段 | 控件类型 | 映射到 taskConfig | 说明 |
|------|---------|------------------|------|
| 转录模式 | el-select | `transcription.transcription_profile` | sensevoice_only / sv_whisper_patch / sv_whisper_dual |
| 说话人检测 | checkbox | `preprocessing.enable_speaker_detection` | 勾选后启用下方说话人数 |
| 说话人数 | number input | `preprocessing.speaker_count` | 勾选说话人检测后可编辑；未勾选时灰色 |

**交互逻辑**:
- 勾选"说话人检测" → `enable_speaker_detection = true`，同时默认 `enable_speaker_guided_split = true`（不显示此配置）
- 勾选后说话人数输入框从灰色变为可编辑
- 说话人数后方始终显示灰色提示文本 "(0=自动识别)"
- 注意：说话人相关字段存储在 `preprocessing` 分组中（后端模型约束），但前端 Tab 归属于"转录"（用户认知）

### 3.4 Tab 4: 后处理

**职责**: 切分配置和 LLM 校对配置。

```
+-----------------------------------------------+
|  切分配置                          (后续适配)    |
|  ┌───────────────────────────────────────────┐ |
|  │ 暂未开放，将在后续版本中支持切分参数配置      │ |
|  └───────────────────────────────────────────┘ |
|                                                |
|  LLM 校对                          (灰色状态)   |
|  ┌───────────────┐  ┌────────────────────────┐ |
|  │ 校对模式       │  │ [关闭 v]    (禁用)      │ |
|  └───────────────┘  └────────────────────────┘ |
+-----------------------------------------------+
```

**组件**: `PostprocessTab.vue`

**表单项**:

| 字段 | 控件类型 | 映射到 taskConfig | 当前状态 |
|------|---------|------------------|---------|
| 切分配置 | 占位文本 | — | 后续适配 |
| 校对模式 | el-select | `refinement.llm_task` | 灰色禁用 |

**说明**:
- 两个区域当前均为灰色/禁用状态
- 切分配置显示"暂未开放"的占位提示
- LLM 校对显示下拉但 disabled（保留 off / proofread / translate 选项结构）

---

## 4. Dialog Footer 设计

```
+---------------------------------------------------+
| 已选择 3 个文件          [保存预设]  [创建 3 个任务]  |
+---------------------------------------------------+
```

### 4.1 元素说明

| 元素 | 位置 | 条件 | 说明 |
|------|------|------|------|
| 文件计数 | 左侧 | 选择文件数 > 0 | "已选择 N 个文件" |
| 保存预设按钮 | 右侧 | 始终显示 | 点击弹出命名对话框 |
| 创建任务按钮 | 右侧 | 始终显示 | 根据模式显示"上传"或"创建 N 个任务" |

### 4.2 保存预设流程

1. 用户点击"保存预设"按钮
2. 弹出 `el-dialog`（小型命名对话框），输入预设名称
3. 确认后：
   - 将当前 taskConfig 的 preprocessing / transcription / refinement / compute 四个分组打包
   - 生成 `id: "custom_" + timestamp`
   - 写入 localStorage `user-custom-presets` 数组
   - POST 到后端 `/api/presets/custom`
4. 预设 Tab 中立即出现新的自定义预设卡片

---

## 5. 自定义预设持久化方案

### 5.1 存储模型

```javascript
// localStorage key: 'user-custom-presets'
[
  {
    id: "custom_1708761600000",
    name: "我的日语视频设置",
    created_at: "2026-02-24T12:00:00Z",
    config: {
      preprocessing: { ... },
      transcription: { ... },
      refinement: { ... },
      compute: { ... }
    }
  }
]
```

### 5.2 同步策略: Write-Through + Read-Fallback

```
读取:
  localStorage 有自定义预设 → 直接使用（零网络开销）
  localStorage 为空 → GET /api/presets/custom → 写入 localStorage 缓存

写入:
  保存新预设  → 写 localStorage + POST /api/presets/custom
  删除预设    → 删 localStorage + DELETE /api/presets/custom/:id

普通参数调整（未保存为预设）:
  仅写 localStorage（现有 transcriptionConfigStore 机制不变）
```

### 5.3 后端 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/presets/custom` | 获取所有自定义预设 |
| POST | `/api/presets/custom` | 创建/更新自定义预设（同名覆盖） |
| DELETE | `/api/presets/custom/:id` | 删除自定义预设 |

**后端存储**: `data/user_presets.json`（JSON 文件，单用户场景无需数据库）

### 5.4 冲突处理

- 后端为真相源。前端 localStorage 仅作为缓存
- 如果用户清除浏览器数据后重新访问，自动从后端恢复
- 如果后端文件不存在或为空，返回空数组，前端显示"暂无自定义预设"

---

## 6. 预设匹配逻辑

### 6.1 从参数反推预设

当用户在 Tab 2-4 中修改参数时，系统自动检查当前参数组合是否匹配某个内置预设：

```javascript
function detectPreset(config) {
  for (const preset of builtinPresets) {
    if (matchesPreset(config, preset)) {
      return preset.id  // 'fast' | 'balanced' | 'quality'
    }
  }
  return 'custom'
}
```

匹配字段：
- `preprocessing.demucs_strategy`
- `transcription.transcription_profile`
- `refinement.llm_task`
- `refinement.llm_scope`

### 6.2 预设 Tab 状态联动

- 当 Tab 2-4 的修改导致参数不再匹配任何内置预设 → 预设 Tab 中所有内置卡片取消选中
- 当参数恰好匹配某个内置预设 → 自动选中对应卡片
- 自定义预设的匹配更严格：四个分组的所有字段完全一致才算匹配

---

## 7. 组件架构

### 7.1 文件结构

```
frontend/src/components/task/
  TaskCreateDialog.vue        # 修改：设置区替换为 Tab 容器
  PresetSelector.vue          # 废弃（拆分为下方 4 个组件）
  settings/
    SettingsTabs.vue           # 新增：Tab 容器壳（el-tabs + 4 个 tab-pane）
    PresetTab.vue              # 新增：预设选择
    PreprocessTab.vue          # 新增：前处理设置
    TranscriptionTab.vue       # 新增：转录设置
    PostprocessTab.vue         # 新增：后处理设置
    SavePresetDialog.vue       # 新增：保存预设命名对话框
    shared/
      SettingRow.vue           # 新增：通用设置行（label + control）
```

### 7.2 数据流

```
TaskCreateDialog.vue
  props: taskConfig (Object, required)
  emits: update:task-config

  └─ SettingsTabs.vue
       props: modelValue (taskConfig)
       emits: update:modelValue, save-preset

       ├─ PresetTab.vue
       │    props: modelValue, customPresets
       │    emits: update:modelValue, delete-preset
       │
       ├─ PreprocessTab.vue
       │    props: modelValue
       │    emits: update:modelValue
       │
       ├─ TranscriptionTab.vue
       │    props: modelValue
       │    emits: update:modelValue
       │
       └─ PostprocessTab.vue
            props: modelValue
            emits: update:modelValue
```

### 7.3 SettingRow 通用组件

每个设置项的标准布局组件：

```vue
<template>
  <div class="setting-row" :class="{ disabled }">
    <div class="setting-label">
      <span class="label-text">{{ label }}</span>
      <span v-if="hint" class="label-hint">{{ hint }}</span>
    </div>
    <div class="setting-control">
      <slot />
    </div>
  </div>
</template>
```

**样式规范**（遵循现有 token，紧凑布局适配 160px 内容区）:

```css
.setting-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  padding: 7px 0;         /* 紧凑: 总行高 ≈ 36px */
}

.setting-row + .setting-row {
  border-top: 1px solid var(--af-border-muted);
}

.setting-row.disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.setting-label {
  display: flex;
  flex-direction: column;
  gap: 1px;
  flex-shrink: 0;
  min-width: 100px;
}

.label-text {
  color: var(--af-text-normal);
  font-size: 13px;
  font-weight: 500;
}

.label-hint {
  color: var(--af-text-muted);
  font-size: 11px;
}

.setting-control {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
}
```

---

## 8. 样式规范

### 8.1 遵循规则

所有新组件必须遵循 `llmdoc/guides/frontend-dev-rules.md`：

- **R1**: 禁止 SCSS，只用 `<style scoped>` + 纯 CSS
- **R2**: 禁止非 scoped 样式块
- **R3**: 禁止硬编码颜色，全部使用 `--af-*` Token 或组件本地变量
- **R4**: `!important` 必须附带注释
- **R5**: Element Plus 定制先查源码，优先用 `--el-*` 变量
- **R6**: 单 `.vue` 文件超 400 行考虑拆分

### 8.2 Tab 容器样式

设置 Tab 容器替代原 `.transcription-settings`，**始终可见**，无折叠状态。

```css
.settings-tabs {
  margin-top: 8px;
}

/* --- 设置 Tab 紧凑头 ---
 * 原因: 节省纵向空间，适配 636px 总高度预算
 * 参考: element-plus/theme-chalk/src/tabs.scss
 */
.settings-tabs :deep(.el-tabs__header) {
  --el-tabs-header-height: 36px;
  margin-bottom: 0;
}

.settings-tabs :deep(.el-tabs__item) {
  height: 36px;
  line-height: 36px;
  font-size: 13px;
  padding: 0 16px;
}

.settings-tabs .tab-content {
  height: 160px;      /* 精确匹配高度预算 */
  padding: 10px 14px;
  overflow-y: auto;
  background: var(--af-bg-primary);
  border: 1px solid var(--af-border-default);
  border-top: none;   /* 与 Tab 头的下划线衔接 */
  border-radius: 0 0 var(--af-radius-md) var(--af-radius-md);
}
```

**el-tabs 配色** — 复用 TaskCreateDialog 已有的方案：

```css
.settings-tabs :deep(.el-tabs__item) { color: var(--af-text-secondary); }
.settings-tabs :deep(.el-tabs__item:hover) { color: var(--af-text-primary); }
.settings-tabs :deep(.el-tabs__item.is-active) { color: var(--af-accent-primary); }
.settings-tabs :deep(.el-tabs__active-bar) { background-color: var(--af-accent-primary); }
.settings-tabs :deep(.el-tabs__nav-wrap::after) { background-color: var(--af-border-default); }
```

### 8.3 预设卡片样式

复用现有 PresetSelector.vue 的 `.preset-card` 样式体系，保持视觉一致：

```css
.preset-card {
  padding: 10px;
  background: var(--af-bg-secondary);
  border: 1px solid var(--af-border-default);
  border-radius: var(--af-radius-md);
  cursor: pointer;
  transition: all var(--af-transition-fast);
}

.preset-card:hover {
  background: var(--af-bg-tertiary);
  border-color: var(--af-accent-primary);
}

.preset-card.active {
  background: rgba(var(--af-accent-primary-rgb), 0.08);
  border-color: var(--af-accent-primary);
}

.preset-card.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
```

### 8.4 el-select 样式

使用 Element Plus 的 `el-select` 而非原生 `<select>`，保持与全局覆盖一致：

```css
/* --- el-select 样式定制 ---
 * 原因：el-select 的内部 input 背景色需要跟随 Tab 内容区
 * 参考：theme-chalk/src/select.scss
 */
:deep(.el-select) {
  --el-select-input-color: var(--af-text-normal);
  --el-select-input-focus-border-color: var(--af-accent-primary);
  width: 200px;
}

:deep(.el-select) .el-input__wrapper {
  background-color: var(--af-bg-secondary);
  box-shadow: 0 0 0 1px var(--af-border-default) inset;
}

:deep(.el-select) .el-input__wrapper:hover {
  box-shadow: 0 0 0 1px var(--af-accent-primary) inset;
}
```

### 8.5 禁用状态样式

统一的禁用态处理（用于语言检测、LLM 校对等待实现的区域）：

```css
.setting-row.disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.setting-row.disabled * {
  pointer-events: none;
}
```

### 8.6 滚动条样式

Tab 内容区 160px 固定高度，超出时显示细滚动条，与 el-table 风格一致：

```css
.tab-content::-webkit-scrollbar {
  width: 4px;
}

.tab-content::-webkit-scrollbar-thumb {
  background-color: var(--af-text-muted);
  border-radius: 2px;
  opacity: 0.4;
}

.tab-content::-webkit-scrollbar-thumb:hover {
  opacity: 0.7;
}

.tab-content::-webkit-scrollbar-track {
  background: transparent;
}
```

---

## 9. 实施计划

### Phase 1: 基础框架（前端骨架）

**目标**: 搭建 Tab 容器结构，4 个空 Tab 壳可切换，设置区始终可见。

1. 创建 `settings/SettingsTabs.vue`
   - el-tabs 容器 + 4 个 tab-pane
   - Tab header 紧凑模式（36px）
   - 内容区固定 160px + overflow-y auto
   - props/emits 对接 taskConfig
2. 创建 `settings/shared/SettingRow.vue` 通用组件
3. 修改 `TaskCreateDialog.vue`
   - 用 SettingsTabs 替换原 `.transcription-settings` 整个区域
   - **移除**：折叠展开逻辑（`showAdvancedSettings`、`toggle-advanced-settings` 事件、ArrowDown 图标）
   - **移除**：`.transcription-settings`、`.settings-header`、`.settings-content` 全部样式
   - 缩减 `.tabs-container` margin-bottom（20 → 12）
   - dialog body 底部 padding 缩减（30 → 20）
4. 修改 `TaskListView.vue`
   - **移除**：`showAdvancedSettings` 状态和 `toggle-advanced-settings` 事件处理
5. footer 添加"保存预设"按钮

### Phase 2: Tab 内容实现

**目标**: 填充 4 个 Tab 的表单控件。

1. 创建 `PresetTab.vue`
   - 内置预设卡片（复用现有样式和逻辑）
   - 自定义预设卡片列表（读 localStorage）
   - 硬件检测 + 预设可用性判断
2. 创建 `PreprocessTab.vue`
   - 人声分离策略 el-select
   - 语言检测占位（disabled）
3. 创建 `TranscriptionTab.vue`
   - 转录模式 el-select
   - 说话人检测 checkbox + 说话人数 number input
4. 创建 `PostprocessTab.vue`
   - 切分配置占位
   - LLM 校对 el-select（disabled）

### Phase 3: 预设系统

**目标**: 自定义预设的完整 CRUD 和同步。

1. 创建 `SavePresetDialog.vue`
2. 前端 composable `useCustomPresets.js`
   - localStorage 读写
   - 后端 API 调用（保存/删除/拉取）
   - Read-Fallback 逻辑
3. 后端 API
   - `GET /api/presets/custom`
   - `POST /api/presets/custom`
   - `DELETE /api/presets/custom/:id`
   - 存储: `data/user_presets.json`
4. 预设匹配逻辑（从参数反推当前预设）

### Phase 4: 清理与测试

**目标**: 移除旧代码，确保功能完整。

1. 废弃 `PresetSelector.vue`（确认无其他引用后删除）
2. 清理 `TaskListView.vue` 中的相关事件处理
3. 编译测试 `npm run build`
4. 验证所有配置字段在创建任务时正确传递到后端
5. 验证自定义预设的保存/加载/删除/同步

---

## 10. 风险与注意事项

### 10.1 字段归属 vs 用户认知

**说话人策略**的字段存储在后端 `PreprocessingSettings` 中，但前端 Tab 归属于"转录"Tab。
提交时按后端字段名映射即可，前端 Tab 组织是面向用户认知的视图层概念。

### 10.2 语言检测 API 未实现

语言检测相关的 `language_detection_mode`、`langid_whitelist` 等字段，后端 API 尚未实现。
前端做 UI 占位（渲染控件但 disabled），提交时发送默认值。待后端实现后去掉 disabled。

### 10.3 向后兼容

- `taskConfig` 的数据结构不变，不影响现有的 `transcriptionConfigStore` 持久化
- `buildTranscriptionSettings()` 构建请求体的逻辑不变
- 后端 API 契约不变（`POST /api/upload`、`POST /api/create-jobs-batch`）

### 10.4 高度预算刚性约束

设置 Tab 内容区 **160px 不可放大**。如果后续新增配置项导致某个 Tab 超出 4 行，通过内部滚动消化，不得增加 Tab 容器高度。整个 dialog 的 636px 高度预算是面向 768p 笔记本的底线约束。

### 10.5 PresetSelector.vue 废弃策略

PresetSelector.vue 当前承载了预设选择 + 模块配置两项职责。
拆分为 4 个 Tab 组件后，PresetSelector 的逻辑分散到各 Tab 中。
在 Phase 4 确认无其他组件引用后安全删除。

---

## 11. 验收标准

1. **高度不溢出**: dialog 总高度 <= 640px，768p 笔记本下不触发主界面滚动
2. 设置 Tab 始终可见，无折叠/展开状态，无 ArrowDown 图标
3. dialog 无论在哪个 Tab，创建任务按钮始终可见且可用
4. Tab 之间自由切换，无线性约束
5. 选择内置预设后参数正确填充，切到其他 Tab 可验证
6. 在 Tab 2-4 修改参数后，预设 Tab 自动取消选中（或匹配到对应预设）
7. 自定义预设的保存/加载/删除正常工作
8. 清除浏览器 localStorage 后，自定义预设从后端恢复
9. 禁用状态的控件（语言检测、LLM 校对）不影响任务创建
10. `npm run build` 编译通过，无样式警告
11. 所有颜色使用 `--af-*` Token，无硬编码色值
