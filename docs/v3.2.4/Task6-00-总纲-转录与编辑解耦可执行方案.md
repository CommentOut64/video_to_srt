# Task 6 总纲：转录与编辑解耦可执行方案

> Type: Architecture | Status: Active
> Version: V3.2.4+dev.20260222.01
> Scope: 转录链路与编辑链路解耦、能力可组合调用、旧任务全量兼容、Full/Lite 共库建模
> Non-Goal: 本文不包含 ASR 模型算法升级、UI 视觉重设计、长期 Lite 独立分支

## 1. Summary

* **Goal**: 将 `job_id` 单体任务模型重构为"Project + SubtitleDoc + (Run)"分层架构，使编辑器可独立于转录任务工作，并保证旧任务无需重跑转录即可完整编辑、预览、导出。
* **核心策略**: 渐进式领域模型 + 底层存储复用 + 适配器桥接
* **参考文档**: `docs/v3.2.3/转录与编辑解耦及旧任务兼容重构方案.md`（原始架构设计）

## 2. 方案选型与决策记录

### 2.1 候选方案

经过代码库深度分析（后端 10,331 行 / 前端 10,434 行核心代码，job_id 四层耦合点），设计并评估了三种方案：

| 维度 | A: 身份演进 | **B: 渐进式领域模型** | C: 完整领域重写 |
|------|:-----------:|:--------------------:|:--------------:|
| 新增代码量 | ~600 行 | ~2,300 行 | ~5,500 行 |
| 修改代码量 | ~200 行 | ~400 行 | ~1,200 行 |
| 代码复用率 | 9/10 | 8/10 | 3/10 |
| 风险评估 | 8/10 | 7/10 | 3/10 |
| 渐进性 | 9/10 | 8/10 | 4/10 |
| 架构质量 | 4/10 | 8/10 | 10/10 |
| Lite 交付路径 | 10/10 | 8/10 | 4/10 |
| Task 6-Full 兼容 | 4/10 | 9/10 | 10/10 |
| **综合评分** | 7.3 | **8.0** | 5.6 |

### 2.2 选择结果：方案 B（渐进式领域模型）

**选择理由**（Codex 评估结论）：

1. **A 方案过于保守**：将 JobState 继续做胖导致职责不单一，后续 Task 6-Full 二次重构成本高，没有引入领域概念属于"补丁式"设计。
2. **C 方案风险过高**：6,700 行改动量与 Task 5（生命周期修复）和 Task 9（前端状态重构）冲突面最大，新 SQLite 层 + 新 SSE 通道 + 新 API 三线并进风险不可控。
3. **B 方案最优平衡**：在"复用现有存储"前提下引入 Project/SubtitleDoc 语义层，能先交 Lite，再平滑走向 Full。与路线图约束一致：Task 9 先于 Task 6-Lite 前端、Task 5 Phase 0+1 先于 Task 6-Lite 后端。

### 2.3 核心设计约束

基于 Codex 评审补充的四个关键约束：

1. **权威 ID**: 前端与新 API 一律以 `project_id` 为主键，`job_id` 仅在兼容层使用
2. **映射一致性**: `legacy_projection_service` 必须保证幂等映射与可重建
3. **统一字幕主键**: 新域使用 `segment_id`（UUID），旧 `sentence_index` 只在桥接层转换
4. **SSE 迁移策略**: 先别名双发（`job:*` + `project:*`），稳定后再收敛到单通道

## 3. 总体架构

### 3.1 数据流

```
新建项目（Lite/Full）:
  POST /api/projects/import  →  ProjectService  →  {JOBS_DIR}/{project_id}/
                                     ↓
                              SubtitleDocService  →  subtitle_edits.json（复用）
                                     ↓
                              前端 /editor/project/:projectId

旧任务兼容:
  GET /editor/:jobId
    →  LegacyProjectionService.resolve(jobId)
    →  Project(mode=legacy)
    →  重定向 /editor/project/:projectId

转录任务（Full 模式）:
  POST /api/transcription/start  →  现有转录链路不变
    →  完成后自动创建 Project(mode=normal)
    →  前端可切换到 /editor/project/:projectId
```

### 3.2 分层架构

```
┌─────────────────────────────────────────────────┐
│                   API 路由层                      │
│  project_routes  legacy_compat_routes  旧路由保留  │
├─────────────────────────────────────────────────┤
│                   服务层                          │
│  ProjectService  SubtitleDocService  LegacyProj  │
├─────────────────────────────────────────────────┤
│                 适配器层                          │
│  subtitle_edit_store（完全复用）                    │
│  streaming_subtitle（频道别名扩展）                 │
│  media_prep_service（project_id 兼容）            │
├─────────────────────────────────────────────────┤
│                 存储层                           │
│  {JOBS_DIR}/{id}/                               │
│  subtitle_edits.json  job_meta.json  media files │
│  _legacy_map.json（映射索引）                      │
└─────────────────────────────────────────────────┘
```

### 3.3 Task 6-Lite vs Task 6-Full 拆分

本方案覆盖 **Task 6-Lite**（Sprint 2 交付）的全部内容，并为 **Task 6-Full**（Sprint 5）铺路：

| 层 | Task 6-Lite（本方案） | Task 6-Full（后续） |
|---|---|---|
| 领域模型 | Project + SubtitleDocMeta（轻量数据类） | Run + Artifact + CapabilitySnapshot |
| 服务 | ProjectService + SubtitleDocService + LegacyProjection | RunService + 能力目录 + 能力守卫 |
| API | /api/projects/* + /api/legacy/* | /api/runs/* |
| SSE | 频道别名 project:{id} | 独立 run:{id} 通道 |
| 前端 | 双入口路由 + flavor 门控 | runStore + 独立能力 UI |
| 存储 | 复用 JOBS_DIR 目录 | projects.db SQLite |

## 4. 文件变更总览

### 4.1 新增文件（后端）

| 文件 | 估算行数 | 职责 |
|------|---------|------|
| `backend/app/models/project_models.py` | ~250 | Project/SubtitleDocMeta/MediaAssetRef 数据类 |
| `backend/app/services/project_service.py` | ~400 | 项目 CRUD + 聚合查询 |
| `backend/app/services/subtitle_doc_service.py` | ~350 | 字幕文档服务（封装 subtitle_edit_store） |
| `backend/app/services/legacy_projection_service.py` | ~300 | 旧任务投影与懒迁移 |
| `backend/app/api/routes/project_routes.py` | ~500 | Project API 端点 |
| `backend/app/api/routes/legacy_compat_routes.py` | ~200 | 旧 API 桥接端点 |
| `backend/tests/test_project_service.py` | ~200 | 项目服务单测 |
| `backend/tests/test_subtitle_doc_service.py` | ~200 | 字幕文档服务单测 |
| `backend/tests/test_legacy_projection.py` | ~200 | 旧任务投影单测 |
| `backend/tests/test_flavor_guard.py` | ~150 | flavor 守卫单测 |

### 4.2 修改文件（后端）

| 文件 | 当前行数 | 修改量 | 改动内容 |
|------|---------|--------|---------|
| `backend/app/core/config.py` | 489 | ~30 | ANCHORFLUX_FLAVOR 解析 |
| `backend/app/main.py` | 746 | ~60 | 条件路由注册 + 新路由挂载 |
| `backend/app/models/job_models.py` | 653 | ~20 | mode 字段 + project_id 别名 |
| `backend/app/api/routes/transcription_routes.py` | 2445 | ~80 | 字幕编辑端点转发 |
| `backend/app/services/streaming_subtitle.py` | 1026 | ~40 | project:{id} 频道别名 |
| `backend/app/api/routes/media_routes.py` | 2148 | ~30 | project_id 路径兼容 |

### 4.3 新增文件（前端）

| 文件 | 估算行数 | 职责 |
|------|---------|------|
| `frontend/src/config/flavor.js` | ~30 | flavor 单一真值 + 能力门控 |
| `frontend/src/services/api/projectApi.js` | ~150 | Project API 客户端 |
| `frontend/src/services/api/legacyApi.js` | ~50 | Legacy resolve API |
| `frontend/src/views/ImportView.vue` | ~200 | Lite 字幕导入页面 |

### 4.4 修改文件（前端）

| 文件 | 当前行数 | 修改量 | 改动内容 |
|------|---------|--------|---------|
| `frontend/src/router/index.js` | 43 | ~30 | 双路由 + /import |
| `frontend/src/views/EditorView.vue` | 2496 | ~120 | 双入口 + Lite 跳过转录检查 |
| `frontend/src/stores/projectStore.js` | 1598 | ~80 | projectId + flavor 门控 |
| `frontend/src/services/sseChannelManager.js` | 780 | ~30 | subscribeProject 方法 |
| `frontend/vite.config.js` | 43 | ~10 | VITE_LITE_MODE + VITE_APP_FLAVOR define |

### 4.5 总量

- **新增**: ~3,230 行（后端 ~2,750 + 前端 ~480）
- **修改**: ~530 行（后端 ~260 + 前端 ~270）
- **删除**: 0（过渡期保持向后兼容）

## 5. 与其他任务的依赖关系

```
Task 5 Phase 0+1 (Sprint 1, 已完成)
  → Task 6-Lite 后端（本方案后端部分）

Task 9 前端状态重构 (Sprint 2)
  → Task 6-Lite 前端（本方案前端部分）

Task 11 Electron (Sprint 2, 并行)
  + Task 6-Lite
  → Task 12 Lite 构建 (Sprint 3)
```

**关键约束**:
- Task 6-Lite 后端可在 Task 9 之前开始（无冲突）
- Task 6-Lite 前端必须在 Task 9 完成后开始（避免 store 二次重写）
- Task 6-Lite 后端修改 `transcription_routes.py` 需基于 Task 5 Phase 0+1 的稳定基线

## 6. 文档索引

本方案由以下 5 份文档组成：

| 编号 | 文件名 | 内容 |
|------|--------|------|
| 00 | 本文件 | 总纲：方案选型、架构总览、文件变更清单 |
| 01 | `Task6-01-领域模型与存储设计.md` | 领域模型定义、存储策略、ID 体系 |
| 02 | `Task6-02-后端实施方案.md` | 后端逐文件实施细节 |
| 03 | `Task6-03-前端实施方案.md` | 前端逐文件实施细节 |
| 04 | `Task6-04-兼容测试与实施计划.md` | 旧任务兼容、测试策略、分阶段计划 |
