# v3.2.4 开发路线图

> 版本：V3.2.0+dev.20260221.01
> 状态：Draft（待评审）
> 范围：12 项任务的深度分析、依赖拆解、风险评估与 6 Sprint 排布

---

## 1. 一句话结论

以"止血 + Lite 贯穿 + 独立并行 + 未知后置"为核心策略，将 12 项任务编排为 6 个 Sprint；Lite 采用共用代码库 + env 开关方案，在 Sprint 3 结束时以 Electron 形态交付 v1。

---

## 2. 设计原则（来自用户约束）

1. **Lite 最短交付**：Lite 版本的关键路径必须最短，且必须在 Task 7（前端样式）完成之后才能启动。
2. **合理并行 + 冲突可控**：多流并行开发的前提是后续合并分支时冲突面最小化。
3. **已知优先、未知后置**：工作量未知的任务尽可能放在后面，优先将系统调整到最佳状态。
4. **Lite = Electron**：Lite 必须以 Electron 形态打包分发，需移除大部分后端重依赖（torch/onnx/whisper 等）和 proxy 视频生成逻辑。
5. **共用代码库**：Lite 不单开分支，与主项目共用前后端代码，通过 env 开关控制运行路径，打包时按 profile 裁剪依赖。

---

## 3. 12 项任务全景分析

### 3.1 任务清单

| # | 任务 | 工作量 | 主要涉及层 | Lite 关键路径 |
|---|------|--------|-----------|:------------:|
| 1 | 中日文后处理针对性优化 | 未知/大 | 后端(textflow/segmentation/language_policy) | - |
| 2 | 双流模式性能优化（GPU 占用 + pyannote/Whisper 时序） | 未知 | 后端(dual_pipeline/workers) | - |
| 3 | DNSMOS 替换 Brouhaha | 可估/中 | 后端(spectral_triage 独立子系统) | - |
| 4 | 前端说话人着色功能适配 | 可估/小 | 前端(SubtitleList/SubtitleItem) | - |
| 5 | 断点与任务生命周期彻底修复（5 Phase） | 大 | 前后端(状态机/队列/SSE/Store) | - |
| 6 | 转录与编辑解耦 | 大(完整)/中(Lite 最小集) | 前后端(领域模型/路由/Store) | 是 |
| 7 | 前端样式调整 + 页面重设计 + bug 修复 | 可估/中 | 前端(views/components) | 是(前置) |
| 8 | 完善任务进度系统 | 未知 | 前后端(SSE/progressStore) | - |
| 9 | 前端状态管理重构 | 可估/中大 | 前端(stores 全部) | - |
| 10 | 前端完整配置界面 + 后端配置统一 | 可估/中 | 前后端(config/routes/AdvancedSettings) | - |
| 11 | Electron 打包分发系统 | 未知/中 | 基建(新增 electron 目录) | 是 |
| 12 | Lite 版本构建配置 | 可估/小 | 基建(env 开关/构建 profile/依赖裁剪) | 是 |

### 3.2 Task 6 拆分策略

Task 6 原始方案定义了 6 个里程碑（M0-M5）的完整 Project/Run/SubtitleDoc 领域重构。但 Lite 只需要一个核心能力：**编辑器不依赖转录任务即可工作**。因此拆为两层：

| 层 | 内容 | 工作量 | Lite 需要 | 完整版需要 |
|---|------|--------|:---------:|:---------:|
| **Task 6-Lite** | env 开关基础设施 + 字幕导入 API + 编辑器 Lite 入口 + 媒体直接挂载 + 后端条件加载 | 中 | 是 | 是 |
| **Task 6-Full** | Project/Run/SubtitleDoc 领域模型 + 旧任务投影层 + 能力目录 + 双通道 SSE + 收敛清理 | 大 | 否 | 是 |

Task 6-Lite 的详细设计见本文档第 7 节。

---

## 4. 依赖关系与关键路径

### 4.1 Lite 交付关键路径

```
Task 7 (前端样式, Sprint 1)
  -> Task 6-Lite (env 开关 + 导入模式, Sprint 2)
     -> Task 11 (Electron 壳 + 构建流水线, Sprint 2 并行)
        -> Task 12 (Lite 构建 profile + 打包, Sprint 3)
           ★ Lite v1 交付
```

### 4.2 系统稳定性关键路径

```
Task 5 Phase 0+1 (止血, Sprint 1)
  -> Task 5 Phase 2 (断点恢复, Sprint 3)
     -> Task 5 Phase 3+4+5 (执行模型 + 前端收口 + 验收, Sprint 4)
```

### 4.3 前端改造依赖链

```
Task 7 (样式, Sprint 1)
  -> Task 9 (状态管理重构, Sprint 2)
     -> Task 6-Lite 前端部分 (Sprint 2, 基于重构后的 store)
     -> Task 4 (说话人着色, Sprint 3)
     -> Task 5 Phase 4 (前端状态收口, Sprint 4)
     -> Task 8 (进度系统, Sprint 5)
```

### 4.4 关键约束

1. **Task 9 必须在 Task 6-Lite 前端部分之前完成**：Task 6-Lite 需要修改 `EditorView.vue` 和 `projectStore.js`，Task 9 也要重构全部 store。如果顺序反了会导致 Task 6-Lite 的改动被 Task 9 二次重写。
2. **Task 5 Phase 0+1 必须在 Task 6-Lite 后端部分之前完成**：两者都要修改 `transcription_routes.py`（102KB）。Task 5 改 cancel/delete 契约，Task 6-Lite 加条件路由。先稳定状态机再加新路径。
3. **Task 11 (Electron) 和 Task 6-Lite 可以并行**：Electron 搭壳不依赖业务逻辑，Task 6-Lite 不依赖 Electron。Sprint 3 组装时两者都就绪即可。

---

## 5. 冲突风险矩阵

### 5.1 高风险组合（必须串行）

| 组合 | 冲突文件 | 应对策略 |
|------|---------|---------|
| Task 5 vs Task 6 | `transcription_routes.py`, `job_lifecycle_service.py` | Task 5 Phase 0+1 先合入，Task 6 基于新基线 |
| Task 9 vs Task 6-Lite 前端 | `projectStore.js`, `EditorView.vue`, `unifiedTaskStore.js` | Task 9 先合入，Task 6-Lite 前端基于新 store 结构 |
| Task 9 vs Task 5 Phase 4 | 全部 store 文件 | Task 9 做职责拆分，Task 5 Phase 4 做状态序号收口 |

### 5.2 低风险组合（可并行）

| 组合 | 理由 |
|------|------|
| Task 3 (DNSMOS) vs 所有 | spectral_triage 是独立子系统，仅改分诊链路 |
| Task 11 (Electron) vs 所有 | 纯新增基建目录，不改现有业务代码 |
| Task 4 (着色) vs Task 5/6/7 | 只改 `SubtitleItem.vue` 渲染层 |
| Task 10 (配置) vs Task 5/6 | 改 `config_routes`/`model_runtime_config_service`，与状态机/编辑解耦无交集 |

### 5.3 冻结点

为保障并行安全，设三道契约冻结点：

1. **SSE 事件契约冻结**（Sprint 1 结束时）：Task 5 Phase 0+1 定义完状态迁移事件格式后冻结，后续任务只消费不修改事件结构。
2. **Store 职责边界冻结**（Sprint 2 中期）：Task 9 完成后冻结 store 的文件划分和公开 API，后续任务在冻结边界内工作。
3. **配置 Schema 冻结**（Sprint 4 结束时）：Task 10 统一配置后冻结 `model_runtime_config` 的字段规范，后续只新增不修改。

---

## 6. Sprint 排布

### Sprint 1 — 止血 + Lite 前置

**目标**：消除系统最痛问题；完成 Lite 的 UI 前置条件。

| 流 | 任务 | 主要改动 | 分支 |
|----|------|---------|------|
| 后端 | **Task 5 Phase 0+1** | 状态机字典冻结 + 迁移白名单 + 状态事件加序号 + 取消/删除终态收敛 + 前端 SSE 信号补齐 | `fix/lifecycle-phase01` |
| 前端 | **Task 7** | 页面样式重设计 + 关键 bug 修复 + Lite 相关页面优先 | `feature/frontend-style` |

**并行安全**：Task 5 改后端状态机 + 前端 SSE 信号层；Task 7 改组件样式和视觉 bug。文件交集极小。

**Sprint 1 合入策略**：
1. `fix/lifecycle-phase01` 合入 main（后端状态机稳定）
2. `feature/frontend-style` 合入 main（样式就绪）
3. 冻结点 1 生效：SSE 事件契约冻结

**Sprint 1 验收标准**：
- 任意任务取消后不再长期卡 `canceling`
- 运行中删除后任务可稳定进入 `removed` 或明确的 `delete_blocked` 状态
- 前端页面样式符合新设计稿
- Lite 相关页面（编辑器、导入入口）视觉就绪

---

### Sprint 2 — 解耦基础 + 独立优化（四流并行）

**目标**：建立 Lite 的后端基础；搭建 Electron 壳；重构前端状态管理；独立推进 DNSMOS。

| 流 | 任务 | 主要改动 | 分支 |
|----|------|---------|------|
| 后端主线 | **Task 6-Lite** | env 开关基础设施 + 字幕导入 API + 后端条件路由注册 + 媒体直接挂载 | `feature/edit-decouple` |
| 后端支线 | **Task 3** (DNSMOS) | `dnsmos_service.py` + 分诊主链改造 + SmartProbe v2 + 缓存键升级 | `feature/dnsmos-triage` |
| 前端 | **Task 9** | store 职责拆分 + 命名规范 + 公开 API 收口 | `refactor/frontend-stores` |
| 基建 | **Task 11** | Electron 壳搭建 + H265 播放验证 + 构建流水线 + 启动器适配 | `feature/electron` |

**并行安全**：
- Task 6-Lite 后端新增文件为主，仅小幅修改 `transcription_routes`（已在 Sprint 1 稳定）和 `main.py`
- Task 3 只改 `spectral_triage` 独立链路
- Task 9 只改前端 store 文件
- Task 11 纯新增 `electron/` 目录

**Sprint 2 合入策略**：
1. `refactor/frontend-stores` 先合入 main（冻结点 2 生效：store 职责冻结）
2. `feature/dnsmos-triage` 合入 main（独立子系统，随时可合）
3. `feature/edit-decouple` 合入 main（基于已冻结的 store 结构）
4. `feature/electron` 暂不合入（Sprint 3 与 Lite 构建一起合入）

**Sprint 2 验收标准**：
- 后端 `ANCHORFLUX_LITE=true` 启动时，不加载转录相关服务，字幕导入 API 可用
- DNSMOS 主链可运行，分诊决策源标记正确
- 前端 store 职责边界清晰，公开 API 文档化
- Electron 壳可启动，H265 视频可原生播放

---

### Sprint 3 — Lite 交付 + 着色 + 断点深化

**目标**：交付 Lite v1；完成说话人着色；推进断点恢复。

| 流 | 任务 | 主要改动 | 分支 |
|----|------|---------|------|
| 基建(关键) | **Task 12** | Lite 构建 profile（env 开关 + 依赖裁剪 + Electron 打包脚本） | `feature/electron`(续) |
| 前端 | **Task 4** | 说话人着色渲染 + 颜色分配策略 + SubtitleItem 适配 | `feature/speaker-coloring` |
| 后端 | **Task 5 Phase 2** | `ResumeStateMerger` + 恢复优先级规则 + `RuntimeCheckpointSnapshot` 扩展 | `fix/lifecycle-phase2` |

**并行安全**：Task 12 改构建配置和 Electron 目录；Task 4 改 `SubtitleItem.vue` 渲染层；Task 5 Phase 2 改 checkpoint 服务。三者文件无交集。

**Sprint 3 合入策略**：
1. `feature/electron` 合入 main（含 Electron 壳 + Lite 构建 profile）
2. `feature/speaker-coloring` 合入 main
3. `fix/lifecycle-phase2` 合入 main

**Sprint 3 里程碑**：
> ★ **Lite v1 交付**：Electron + LITE_MODE=true + 字幕导入/编辑/导出 + H265 原生播放

**Sprint 3 验收标准**：
- Lite v1 可独立安装运行，不包含 torch/onnx 等重依赖
- Lite 模式下可导入 SRT/ASS，完整编辑并导出
- Lite 模式下视频 H265 原生播放，无 proxy 生成
- 完整版说话人着色正常渲染
- 暂停后恢复不丢 `finalized_indices/sentences_snapshot`

---

### Sprint 4 — 配置统一 + 执行模型重构

**目标**：治理配置分层混乱；完成断点修复剩余部分。

| 流 | 任务 | 主要改动 | 分支 |
|----|------|---------|------|
| 前后端 | **Task 10** | 后端配置来源统一 + 前端完整配置面板 + 数值校验/同步机制 | `feature/config-unified` |
| 前后端 | **Task 5 Phase 3+4+5** | 队列调度→独立 runner + 前端状态投影收口 + 测试矩阵 | `fix/lifecycle-phase345` |

**并行安全**：Task 10 改 `config_routes`/`model_runtime_config_service`/`AdvancedSettings.vue`；Task 5 改 `job_queue_service`/`orchestrator`/stores。路由和服务文件不同。

**注意**：Task 5 Phase 4（前端收口）涉及 store 改动，必须在 Sprint 2 Task 9 冻结的边界内工作。Phase 4 聚焦"状态序号应用 + 终态处理统一"，不重新拆分 store 结构。

**Sprint 4 合入策略**：
1. `feature/config-unified` 合入 main（冻结点 3 生效：配置 schema 冻结）
2. `fix/lifecycle-phase345` 分阶段合入（Phase 3 风险最高，需独立灰度验证后再合）

**Sprint 4 验收标准**：
- 前端配置面板覆盖所有运行参数，数值校验生效
- 后端配置来源追踪清晰，不再出现"不知道用的哪层值"
- 取消超时后下一个任务可实际启动（非仅状态变化）
- 前端不再出现"列表已取消、编辑器仍正在取消"

---

### Sprint 5 — 完整解耦 + 进度系统

**目标**：完成 Task 6 完整重构；统一进度系统。

| 流 | 任务 | 主要改动 | 分支 |
|----|------|---------|------|
| 前后端 | **Task 6-Full** | Project/Run/SubtitleDoc 领域模型 + 旧任务投影层 + 能力目录 + 双通道 SSE + M4 能力解耦 + M5 收敛清理 | `feature/edit-decouple-full` |
| 前后端 | **Task 8** | 进度系统前后端统一改造 | `feature/progress-system` |

**顺序约束**：Task 8 与 Task 5 Phase 4（Sprint 4）都涉及 `progressStore`/`unifiedTaskStore`。Task 8 应在 Phase 4 收口之后进行，基于已统一的状态投影体系工作。

**Sprint 5 验收标准**：
- 旧任务可通过投影层无缝迁移到新 Project 模型
- 独立能力（分诊/分离/纯人声导出）可通过 Run API 调用
- 进度系统前后端一致，覆盖所有阶段

---

### Sprint 6 — 算法与性能深水区

**目标**：推进工作量未知的优化任务，需要实际数据驱动。

| 优先级 | 任务 | 理由 |
|--------|------|------|
| 6a | **Task 1** (中日文后处理优化) | 当前已有 language_policy 框架（`policy.yaml` 支持中/英/日差异化规则）和四层后处理架构（Collection/Scoring/Decision/Output），可在此基础上迭代。需要实际样本驱动调参 |
| 6b | **Task 2** (双流性能优化) | GPU 占用波动和 pyannote/Whisper 时序调整需要实际 profiling 数据。`dual_pipeline/implementation.py`（250KB）是全项目最大文件，改动风险最高，放在最后降低系统性风险 |

**Sprint 6 验收标准**：
- 中文/日文转录后处理质量指标（CER/断句准确率）有可量化提升
- 双流模式 GPU 占用波动幅度降低，Whisper 时序利用率提升

---

## 7. Task 6-Lite 详细设计

### 7.1 env 开关体系

**后端环境变量**：
- `ANCHORFLUX_LITE`：`true`/`false`（默认 `false`）
- 读取点：`backend/app/config/settings.py` 或 `backend/app/core/config.py`，启动时一次读取，全局不可变

**前端环境变量**：
- `VITE_LITE_MODE`：`true`/`false`（默认 `false`）
- 编译时注入，tree-shaking 可移除死代码路径

### 7.2 后端改动

#### 7.2.1 条件路由注册（`backend/app/main.py`）

Lite 模式下不注册的路由：
- `transcription_routes`（转录 API）
- `demucs_routes`（人声分离控制）
- `speaker_routes`（说话人识别）
- `model_routes`（模型下载管理）

始终注册的路由：
- `media_routes`（媒体服务）
- `config_routes`（配置管理）
- `system_routes`（系统状态/心跳，简化）
- `file_routes`（文件操作）
- `hardware_routes`（硬件查询）
- 新增：`project_routes`（项目管理 + 字幕导入）

#### 7.2.2 条件服务加载

Lite 模式下不初始化的服务：
- `PipelineOrchestrator`
- `ModelManagerV2`
- `SenseVoiceOnnxService`
- `WhisperService`
- `BrouhahaService` / `DnsmosService`
- `PreprocessCacheService`
- `JobQueueService`（转录队列）
- `SmartProbeService`
- `AudioSpectrumClassifier`
- proxy 视频生成相关逻辑

始终加载的服务：
- `MediaPrepService`（媒体文件管理，去除 proxy 部分）
- `SubtitleEditStore`（字幕编辑存储）
- 新增：`SubtitleImportService`（字幕导入解析）
- 导出相关服务

#### 7.2.3 新增字幕导入 API

```
POST /api/projects/import
Body: { video_path?: string, subtitle_file: UploadFile, format: "srt"|"ass"|"vtt" }
Response: { project_id: string, subtitle_count: int }
```

行为：
1. 解析字幕文件为内部 `SentenceSegment` 格式
2. 创建 `job` 记录（`mode=edit_only`，跳过所有转录状态）
3. 若提供 `video_path`，挂载媒体资产（提取音频 + peaks）
4. 返回可直接用于编辑器的 `project_id`

#### 7.2.4 编辑模式 Job

在 `JobSettings` 或 `job_models` 中新增：
- `mode: "transcribe" | "edit_only"`（默认 `"transcribe"`）
- `edit_only` 模式下：`status` 直接设为 `finished`，不进入队列，不触发流水线

### 7.3 前端改动

#### 7.3.1 路由守卫

```javascript
// router/index.js
const isLite = import.meta.env.VITE_LITE_MODE === 'true'

// Lite 模式下：
// - TaskListView 变为 ProjectListView（显示导入项目列表）
// - 隐藏"新建转录任务"入口
// - 保留 /editor/:jobId 路由
// - 新增 /import 路由
```

#### 7.3.2 条件组件渲染

Lite 模式下隐藏：
- `TaskCreateDialog`（任务创建对话框）
- `PresetSelector`（预设选择器）
- `TaskMonitor`（任务监视面板）
- `DualAnchorProgress`（双流进度条）
- 转录相关的高级设置 tab

Lite 模式下保留：
- `EditorHeader`
- `SubtitleList` + `SubtitleItem`（完整编辑能力）
- `WaveformTimeline`（波形交互）
- `PlaybackControls`（播放控制）
- `VideoStage`（视频播放）
- `ContextMenu`（右键菜单）
- 导出相关 UI

#### 7.3.3 EditorView 适配

`EditorView.vue` 中移除 Lite 模式下的前置拦截：
- 跳过"等待转录完成"的状态检查
- 跳过"视频就绪"的强制等待（`isVideoReady` 拦截）
- 直接进入可编辑状态

#### 7.3.4 Store 退化

| Store | Lite 模式行为 |
|-------|-------------|
| `projectStore` | 保留全部编辑功能 |
| `unifiedTaskStore` | 退化为只管理导入项目列表（无队列/进度概念） |
| `progressStore` | 不加载 |
| `transcriptionConfigStore` | 不加载 |

### 7.4 Electron 构建配置（Task 11 + Task 12）

#### 7.4.1 目录结构

```
electron/
  ├── main.js              # Electron 主进程
  ├── preload.js           # 预加载脚本
  ├── package.json         # Electron 依赖
  ├── forge.config.js      # Electron Forge 打包配置
  ├── profiles/
  │   ├── full.env         # ANCHORFLUX_LITE=false
  │   └── lite.env         # ANCHORFLUX_LITE=true, VITE_LITE_MODE=true
  └── scripts/
      ├── build-full.js    # 完整版构建脚本
      └── build-lite.js    # Lite 版构建脚本
```

#### 7.4.2 Lite 构建流程

1. **前端构建**：`VITE_LITE_MODE=true vite build` -> tree-shaking 移除转录相关死代码
2. **后端打包**：排除重依赖的 Python 包（torch/onnxruntime/faster-whisper/pyannote/demucs/brouhaha）
3. **Electron 打包**：Electron Forge 按 `lite.env` profile 打包
4. **产物**：`AnchorFlux-Lite-v{version}-win64.exe`（预期体积 < 200MB，对比完整版 > 2GB）

#### 7.4.3 Lite 与完整版的构建差异

| 维度 | 完整版 | Lite 版 |
|------|--------|---------|
| 后端依赖 | 全量（含 torch/onnx/whisper） | 仅 FastAPI + 媒体处理 + 字幕解析 |
| 前端代码 | 全量 | tree-shaking 移除转录路径 |
| proxy 视频 | 生成 360p/720p | 不生成（Electron 原生 H265） |
| 心跳保活 | HTTP 轮询 | Electron 进程内通信 |
| GPU 监控 | 完整 | 不加载 |
| 模型文件 | SenseVoice/Whisper/VAD/YAMNet/Demucs | 无 |
| 预期体积 | > 2GB | < 200MB |

### 7.5 env 开关影响范围汇总

| 层 | Lite 模式下的变化 |
|----|-----------------|
| 后端路由 | 不注册 `transcription/demucs/speaker/model` 路由 |
| 后端服务 | 不初始化 Pipeline/ASR/分诊/分离/缓存等重服务 |
| 后端保留 | 媒体管理、字幕编辑、导出、系统管理（简化心跳）、配置 |
| 前端路由 | 隐藏任务创建；新增 `/import` 入口 |
| 前端组件 | 隐藏 `TaskCreateDialog/PresetSelector/TaskMonitor/DualAnchorProgress` |
| 前端 Store | `progressStore/transcriptionConfigStore` 不加载；`unifiedTaskStore` 退化 |
| Electron | 去除 proxy 生成、简化心跳、启动器适配 |
| 打包依赖 | 排除 torch/onnxruntime/faster-whisper/pyannote/demucs/brouhaha |

---

## 8. 分支管理策略

### 8.1 分支清单

| 分支 | 生命周期 | 涉及文件域 |
|------|---------|-----------|
| `fix/lifecycle-phase01` | Sprint 1 | 后端状态机 + SSE + stores 信号层 |
| `feature/frontend-style` | Sprint 1 | 前端组件样式/布局 |
| `feature/edit-decouple` | Sprint 2 | 后端新增服务/路由 + `main.py` + `transcription_routes` 条件分支 |
| `feature/dnsmos-triage` | Sprint 2 | `spectral_triage_stage` + `audio_spectrum_classifier` + 新增 `dnsmos_service` |
| `refactor/frontend-stores` | Sprint 2 | 前端全部 store 文件 |
| `feature/electron` | Sprint 2-3 | 新增 `electron/` 目录 + 构建脚本 |
| `feature/speaker-coloring` | Sprint 3 | `SubtitleItem.vue` + `projectStore` 着色相关 |
| `fix/lifecycle-phase2` | Sprint 3 | checkpoint 服务文件 |
| `feature/config-unified` | Sprint 4 | `config_routes` + `model_runtime_config_service` + `AdvancedSettings.vue` |
| `fix/lifecycle-phase345` | Sprint 4 | `job_queue_service` + `orchestrator` + stores |
| `feature/edit-decouple-full` | Sprint 5 | 新增领域模型 + 服务 + 路由 + SSE 双通道 |
| `feature/progress-system` | Sprint 5 | `progressStore` + SSE 进度事件 |

### 8.2 合入顺序约束（必须遵守）

```
Sprint 1:
  fix/lifecycle-phase01       -> main (先合)
  feature/frontend-style      -> main (后合)

Sprint 2:
  refactor/frontend-stores    -> main (最先合, 冻结 store)
  feature/dnsmos-triage       -> main (随时可合)
  feature/edit-decouple       -> main (在 stores 之后合)
  feature/electron            -> 不合入 (Sprint 3 合)

Sprint 3:
  feature/electron            -> main (含 Lite 构建)
  feature/speaker-coloring    -> main
  fix/lifecycle-phase2        -> main
```

### 8.3 Feature Flag

对高风险改动使用 feature flag 保护：

| Flag | 控制范围 | 默认值 | 灰度策略 |
|------|---------|--------|---------|
| `LIFECYCLE_V2_ENABLED` | Task 5 新状态机逻辑 | `false` | Phase 0+1 完成后开启 |
| `RESUME_MERGE_ENABLED` | Task 5 Phase 2 双源合并 | `false` | Phase 2 完成后开启 |
| `RUNNER_ISOLATION_ENABLED` | Task 5 Phase 3 独立 runner | `false` | Phase 3 验证后开启 |
| `DNSMOS_TRIAGE_ENABLED` | Task 3 DNSMOS 主链 | `false` | Optuna 校准后开启 |

---

## 9. 时间线总览

```
Sprint 1 ──┬── [后端] Task 5 Phase 0+1 (断点止血)
            └── [前端] Task 7 (样式 + bug)
                │
                ▼ 合入: 状态机稳定 + 样式就绪
                │
Sprint 2 ──┬── [后端主] Task 6-Lite (env 开关 + 导入模式)
            ├── [后端支] Task 3 (DNSMOS 替换)
            ├── [前端]   Task 9 (状态管理重构)
            └── [基建]   Task 11 (Electron 壳)
                │
                ▼ 合入: stores 冻结 -> DNSMOS -> 解耦后端
                │
Sprint 3 ──┬── [基建] Task 12 (Lite 构建 + 打包)  ★ Lite v1 交付
            ├── [前端] Task 4 (说话人着色)
            └── [后端] Task 5 Phase 2 (断点恢复)
                │
                ▼ Lite v1 里程碑
                │
Sprint 4 ──┬── [前后端] Task 10 (配置统一)
            └── [前后端] Task 5 Phase 3+4+5 (执行模型 + 前端收口 + 验收)
                │
Sprint 5 ──┬── [前后端] Task 6-Full (完整 Project 模型重构)
            └── [前后端] Task 8 (进度系统)
                │
Sprint 6 ──┬── Task 1 (中日文后处理优化)
            └── Task 2 (双流性能优化)
```

---

## 10. 各任务参考文档索引

| 任务 | 参考文档 |
|------|---------|
| Task 1 | `llmdoc/architecture/preparation-layer-language-adapter-four-layer.md`, `backend/app/config/language_policy/policy.yaml` |
| Task 2 | `llmdoc/architecture/async-dual-pipeline.md`, `backend/app/pipelines/dual_pipeline/implementation.py` |
| Task 3 | `docs/v3.2.3/dnsmos-替代brouhaha-深度评估与最终方案.md` |
| Task 4 | `backend/app/api/routes/speaker_routes.py`, `frontend/src/components/editor/SubtitleList/SubtitleItem.vue` |
| Task 5 | `docs/v3.2.3/P0-L0-L8-断点与任务生命周期彻底修复方案.md` |
| Task 6 | `docs/v3.2.3/转录与编辑解耦及旧任务兼容重构方案.md` |
| Task 7 | 待设计稿 |
| Task 8 | `llmdoc/guides/v37-progress-pause-resume-fix.md` |
| Task 9 | `frontend/src/stores/` 全部文件 |
| Task 10 | `backend/app/services/model_runtime_config_service.py`, `backend/app/api/routes/model_runtime_routes.py`, `frontend/src/components/editor/AdvancedSettings.vue` |
| Task 11 | `docs/v3.2.3/electron-shell-plan.md`, `launcher/` 目录 |
| Task 12 | 本文档第 7 节 |

---

## 11. 风险登记簿

| 风险 | 影响 | 概率 | 应对 |
|------|------|------|------|
| Task 11 Electron 的 H265 原生播放在部分系统上不支持 | Lite 核心功能受损 | 中 | Sprint 2 提前验证主流 Windows 10/11 版本；保留 fallback 到软解码 |
| Task 5 Phase 3 (runner 隔离) 改动风险最高 | 队列调度回归 | 高 | 使用 `RUNNER_ISOLATION_ENABLED` flag 灰度；先在测试环境长跑 |
| Task 6-Lite env 开关遗漏导致 Lite 模式下加载了重依赖 | Lite 打包体积爆炸 | 中 | CI 中加入 Lite 模式启动检查（验证不 import torch 等） |
| Sprint 2 四流并行合入冲突 | 延期 | 中 | 严格按合入顺序执行（stores 先 -> DNSMOS -> 解耦）；每流设中间合入点 |
| Task 6-Full 与 Task 6-Lite 的 API 断裂 | 旧任务投影失效 | 低 | Task 6-Lite 设计时预留 `project_id` 字段，Task 6-Full 直接复用 |
| `dual_pipeline/implementation.py` (250KB) 改动导致全量回归 | 转录质量倒退 | 高 | Task 2 必须有 A/B 对比基线；仅在 Sprint 6 专项处理 |

---

## 12. 完成定义（DoD）

### 12.1 Lite v1 交付标准（Sprint 3 结束）

1. Lite Electron 安装包可在 Windows 10/11 上独立运行
2. 可导入 SRT/ASS/VTT 格式字幕并进入编辑器
3. 可关联视频文件，H265 原生播放
4. 字幕编辑全功能可用（编辑/切分/合并/删除/撤销/重做）
5. 可导出 SRT/ASS/VTT
6. 安装包体积 < 200MB
7. 不包含 torch/onnxruntime/faster-whisper/pyannote/demucs 等依赖

### 12.2 系统稳定性标准（Sprint 4 结束）

1. 取消后队列推进成功率 >= 99.9%
2. 前端状态与后端状态一致率 100%（按 state_seq 校验）
3. 暂停/重启恢复后字幕与进度一致性通过回归矩阵
4. 配置来源可追踪，运行参数 API 覆盖所有可调参数

### 12.3 完整版标准（Sprint 5 结束）

1. 旧任务通过投影层可无缝迁移到 Project 模型
2. 独立能力（分诊/分离/人声导出）可通过 Run API 调用
3. 进度系统前后端一致

### 12.4 算法质量标准（Sprint 6 结束）

1. 中文/日文 CER 和断句准确率有可量化提升（需基线对比）
2. 双流模式 GPU 利用率波动降低（需 profiling 基线）
