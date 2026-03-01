# Task6-05 纯 Project 单语义与目录命名统一实施方案

> 文档类型：Implementation Plan（仅方案，不代表已落地）
> 版本：V3.2.4+dev.20260227.11
> 状态：Proposed
> 适用范围：后端 `backend/app/*`、前端 `frontend/src/*`、迁移脚本与测试

---

## 1. 目标与约束

### 1.1 总目标

1. 统一语义为 `project_id`，不再保留业务层 `job` 分支。
2. 转录模式与导入模式都生成统一 Project 目录与统一 API 语义。
3. 旧 `job`/旧 `project` 自动转换为新 Project 语义，转换失败直接报错。
4. 目录命名从旧方案改为可读命名，随机后缀从 `rand6` 改为 `rand4`。

### 1.2 强约束

1. 兼容入口仅允许“自动转换后继续”，不允许长期双轨逻辑。
2. 字幕真源只允许 `project subtitle_doc`，不允许回落 `job subtitles` 写链路。
3. 不在本阶段实现 Full/Lite 能力矩阵功能，仅保留接缝。

### 1.3 非目标

1. 不改 ASR 算法与双流推理策略本身。
2. 不改用户可见交互流程（任务列表/编辑器入口保持一致）。

---

## 2. 现状基线（代码事实）

1. 转录任务目录仍由 `job_id` 创建：`backend/app/services/job_lifecycle_service.py`。
2. 转录入口仍生成随机 `job_id`：`backend/app/api/routes/transcription_routes.py`。
3. Project 元信息创建后仍复用 `jobs/{job_id}` 目录：`backend/app/services/project_service.py#create_normal_project`。
4. 兼容层存在 `job_id -> project_id` 懒迁移：`backend/app/services/legacy_projection_service.py`。
5. 前端编辑器已以 `/editor/project/:projectId` 为主，但任务控制仍大量走 `transcriptionApi(job_id)`：`frontend/src/views/EditorView.vue`、`frontend/src/services/api/transcriptionApi.js`。
6. 媒体接口参数名仍为 `{job_id}`，但内部已支持 project 反查：`backend/app/api/routes/media_routes.py`。

结论：当前是“项目语义+任务语义混合态”，尚未达到纯 Project 单语义。

---

## 3. 目标态设计（Pure Project Semantic）

### 3.1 统一身份模型

1. **唯一业务主键**：`project_id`。
2. `job_id` 仅作为历史输入兼容字段，进入系统后必须立即转换为 `project_id`。
3. 新写入数据（API 响应、SSE、持久化元数据）默认字段为 `project_id`。

### 3.2 统一任务生命周期入口

1. 创建转录任务即创建 Project（目录、meta、runtime 绑定一次完成）。
2. 暂停/恢复/取消/状态查询统一以 `project_id` 作为路径参数。
3. 旧 `/api/*{job_id}` 路径作为薄兼容层，仅做 `resolve_or_fail(job_id)` 后转发。

### 3.3 统一字幕与导出语义

1. 编辑、同步、导出仅走 `/api/projects/{project_id}/subtitles*`。
2. `/api/jobs/{job_id}/subtitles*` 保留短期桥接，内部强制转换后调用 project 服务。
3. 转换失败直接返回错误，不再走旧 job 字幕逻辑兜底。

### 3.4 统一 SSE 语义

1. 主频道统一为 `project:{project_id}`。
2. `job:{job_id}` 在过渡期只做镜像广播，最终删除。

---

## 4. 目录命名规范（含 rand4）

### 4.1 新命名格式

```
p-{YYYYMMDD}-{HHmmss}-{mode}-{slug}-{rand4}
```

字段说明：

1. `mode`：`tr`（transcribe）/ `im`（import）/ `lg`（legacy-migrated）
2. `slug`：标题或文件名归一化后的人类可读标识（小写、`-` 分隔）
3. `rand4`：4 位随机串（`[0-9a-z]`），替代旧 `rand6`

示例：

1. `p-20260227-231530-tr-interview-ep01-a9f2`
2. `p-20260227-231905-im-my-subtitle-file-3c7d`
3. `p-20260227-232011-lg-old-task-5e1b`

### 4.2 slug 规则

1. 仅允许 `[a-z0-9-]`。
2. 连续分隔符压缩为单个 `-`。
3. 首尾 `-` 去除。
4. 截断至 32 字符；为空时使用 `untitled`。

### 4.3 冲突处理

1. 同秒内同 slug 冲突时重试新的 `rand4`。
2. 最大重试 16 次，仍冲突则报错并中止创建。
3. Windows 下按大小写不敏感进行冲突判断。

---

## 5. 自动迁移与自动改名策略

### 5.1 迁移范围

1. 目录内存在 `project_meta.json` 的旧 Project 目录。
2. 仅有 `job_meta.json/checkpoint/srt` 的旧 Job 目录（先转 Project，再改名）。

### 5.2 执行时机

1. 启动阶段执行轻量扫描（可配置开关）。
2. 提供显式迁移命令用于离线全量迁移。

### 5.3 迁移算法（严格）

1. 扫描 `jobs/*`。
2. 对每个候选目录解析或创建 `project_id`（旧 job 先 `resolve_or_fail`）。
3. 依据 meta 生成目标目录名（新规范 + `rand4`）。
4. 若目录名已符合新规范则跳过。
5. 校验任务运行态：
   - `running/queued/pausing/canceling`：跳过并记录，防止运行中目录被改。
6. 执行原子改名（同卷 `os.replace`）。
7. 回写以下引用：
   - `project_meta.json` 的 `dir`
   - `job_meta.json` 的 `dir`（若存在）
   - 任务状态仓库中该任务的 `dir` 字段（若存在）
8. 写入迁移审计日志（成功/失败/跳过原因）。
9. 任一步失败：该目录回滚并标记失败，不继续链式兜底。

### 5.4 回滚策略

1. 迁移日志保留 `before_dir -> after_dir`。
2. 提供逆向脚本按日志回滚。
3. 回滚同样要求目标目录不存在且任务不在运行态。

---

## 6. 文件级改造矩阵（逐文件）

## 6.1 后端

### 6.1.1 新增文件

1. `backend/app/services/project_naming_service.py`：目录名生成、slug 归一化、`rand4` 冲突重试。
2. `backend/app/services/project_directory_migrator.py`：目录扫描、自动改名、审计日志、回滚入口。
3. `backend/app/services/project_id_resolver.py`：统一 `resolve_or_fail(identifier)`（`project_id` 或旧 `job_id`）。
4. `backend/app/api/routes/project_task_routes.py`：project 语义任务控制路由（创建/暂停/恢复/取消/状态）。
5. `backend/app/scripts/migrate_project_dirs.py`：离线全量迁移命令。

### 6.1.2 修改文件

1. `backend/app/models/project_models.py`：补充目录命名元字段（`mode/slug/source_filename/created_at` 用于命名）。
2. `backend/app/models/job_models.py`：明确 `job_id` 为兼容字段，新增 `project_id` 强制非空落库规则。
3. `backend/app/services/project_service.py`：
   - `create_import_project/create_normal_project` 改为统一目录命名工厂。
   - 新增 `rename_project_dir()` 与 `rebind_project_dir_cache()`。
4. `backend/app/services/legacy_projection_service.py`：
   - 仅保留转换职责，失败直接抛错。
   - 取消“旧链路继续工作”的宽松兜底。
5. `backend/app/services/transcription_service.py`：
   - 任务创建前置 project 初始化。
   - 运行中/完成后统一写 project 元信息，不再出现“仅 job 无 project”。
6. `backend/app/services/job_lifecycle_service.py`：
   - 目录创建从 `job_id` 切换为 `project` 目录。
   - 持久化读取优先 `project_id`。
7. `backend/app/services/job_queue_service.py`：
   - 运行态主索引切换到 `project_id`。
   - 旧 `job_id` 输入通过 resolver 转换。
8. `backend/app/services/streaming_subtitle.py`：主推送频道统一 `project:{project_id}`。
9. `backend/app/services/sse_publisher.py`：事件 payload 主键改为 `project_id`。
10. `backend/app/api/routes/transcription_routes.py`：
    - 所有旧接口改为兼容壳，内部强制 resolve 后转 project 路由。
    - 响应体新增/主用 `project_id`。
11. `backend/app/api/routes/project_routes.py`：
    - 补 task 生命周期相关 endpoint（或转由 `project_task_routes.py` 承担）。
12. `backend/app/api/routes/legacy_compat_routes.py`：收敛为单一转换入口，不承载业务逻辑。
13. `backend/app/api/routes/media_routes.py`：
    - 路径参数从语义上改为 `identifier`。
    - 返回体主字段统一 `project_id`。
14. `backend/app/api/routes/stream_routes.py`：
    - project 流为唯一主入口。
    - 旧 job stream 标注弃用并转发。
15. `backend/app/main.py`：挂载新路由、迁移器启动钩子与配置开关。

### 6.1.3 删除文件（阶段末）

1. 仅在所有客户端切换完成后删除旧 job 任务控制路由分支（位于 `transcription_routes.py` 内部分支）。
2. 删除 job 专用 SSE 订阅入口（`/api/stream/{job_id}`）。

## 6.2 前端

### 6.2.1 新增文件

1. `frontend/src/services/api/projectTaskApi.js`：project 语义任务控制 API。
2. `frontend/src/utils/projectIdentity.js`：`resolveProjectIdentityOrThrow`，统一处理 `job_id` 输入。
3. `frontend/src/state/adapters/projectTaskApiAdapter.js`：状态层统一映射器。

### 6.2.2 修改文件

1. `frontend/src/services/api/transcriptionApi.js`：
   - 标记为兼容层（内部全部调用 `projectTaskApi`）。
   - 输出结构对齐 `project_id`。
2. `frontend/src/services/api/projectApi.js`：补齐 project 任务生命周期方法。
3. `frontend/src/services/api/legacyApi.js`：仅保留 ID 解析与错误透传。
4. `frontend/src/services/api/index.js`：导出统一 API 面，默认 project 语义。
5. `frontend/src/services/sseChannelManager.js`：
   - 默认订阅 `project:{project_id}`。
   - `subscribeJob` 进入弃用桥接路径。
6. `frontend/src/router/index.js`：`/editor/:jobId` 仅做跳板并强制转换失败即报错退回。
7. `frontend/src/utils/editorNavigation.js`：去除 `job` 成功后继续保留 job 路径的逻辑。
8. `frontend/src/views/TaskListView.vue`：
   - 打开、取消、重命名等操作改走 project 语义。
9. `frontend/src/views/EditorView.vue`：
   - 任务控制、进度、字幕同步、导出全部以 `projectId` 驱动。
   - 删除任何 job 字幕写回分支。
10. `frontend/src/components/editor/EditorHeader.vue`：控制按钮与重命名改走 project task API。
11. `frontend/src/components/editor/TaskMonitor/index.vue`：任务实体主键改为 `project_id`。
12. `frontend/src/stores/taskRuntimeStore.js`：状态主键切到 `project_id`，保留 job alias 字段只读。
13. `frontend/src/stores/editorSessionStore.js`：会话 identity 以 `project_id` 唯一化。
14. `frontend/src/stores/subtitleDocumentStore.js`：同步键强制 `segment_id` + `project_id`。
15. `frontend/src/composables/task-list/useTaskThumbnail.js`：缩略图接口入参语义调整为 identifier/project。

### 6.2.3 删除文件/删除分支（阶段末）

1. 删除 `EditorView.vue` 中 job-only 的 SSE/API fallback 分支。
2. 删除 `transcriptionApi` 中 `/api/jobs/{job_id}/subtitles*` 直接调用分支。

## 6.3 测试与脚本

### 6.3.1 新增测试

1. `backend/tests/services/test_project_naming_service.py`
2. `backend/tests/services/test_project_directory_migrator.py`
3. `backend/tests/routes/test_project_task_routes.py`
4. `backend/tests/routes/test_legacy_job_to_project_compat.py`
5. `frontend/src/services/api/__tests__/projectTaskApi.spec.js`
6. `frontend/src/stores/__tests__/taskRuntimeStore.project-id.spec.js`
7. `frontend/src/stores/__tests__/regression/phaseX-project-only-flow.spec.js`

### 6.3.2 新增运维脚本

1. `scripts/project_dir_migration_dry_run.ps1`
2. `scripts/project_dir_migration_apply.ps1`
3. `scripts/project_dir_migration_rollback.ps1`

---

## 7. 分阶段落地方案（可执行）

## Phase 0：契约冻结与观测前置

目标：

1. 冻结“project 唯一语义”契约文档与响应字段。
2. 为迁移器与兼容壳加入审计日志。

交付：

1. 新增 `project_id_resolver` 与审计日志结构。
2. 所有兼容入口统一使用 `resolve_or_fail`。

Gate：

1. 任一 `job_id` 入口都能明确看到“转换成功或直接失败”。
2. 无静默 fallback。

---

## Phase 0.5：目录命名引擎（rand4）与 Dry Run

目标：

1. 命名规则上线但不改线上目录。
2. 先跑全量 dry-run 报告。

交付：

1. `project_naming_service` 与 `project_directory_migrator --dry-run`。
2. 输出冲突率、跳过率、预计改名数量。

Gate：

1. 命名冲突率可接受（<0.5% 且可重试解决）。
2. dry-run 无 P0 错误。

---

## Phase 1：后端创建链路纯 Project 化

目标：

1. 新任务创建即 project 目录，目录名符合新规范。
2. 转录/导入两种模式统一目录生成逻辑。

交付：

1. `transcription_routes + job_lifecycle_service + project_service` 主链路改造。
2. 新响应字段以 `project_id` 为主。

Gate：

1. 新建任务目录不再是纯随机 `job_id` 目录。
2. 刷新任务后字幕/媒体/状态一致。

---

## Phase 2：API 与前端主路径切换

目标：

1. 前端默认调用 project task API。
2. SSE 默认使用 project 频道。

交付：

1. `projectTaskApi` 接入 `TaskListView/EditorView/TaskMonitor`。
2. `subscribeProject` 成为默认订阅。

Gate：

1. 上传 -> 转录 -> 编辑 -> 导出 -> 刷新恢复 全链路无 `job` 主分支依赖。

---

## Phase 3：自动改名迁移正式执行

目标：

1. 旧 project/job 目录自动迁移到新目录命名。
2. 引用同步与回滚机制可用。

交付：

1. 启动迁移器（按运行态跳过策略）。
2. 离线脚本用于一次性全量迁移。

Gate：

1. 迁移后 `project_meta.dir`、运行态 `dir`、媒体读取一致。
2. 无“目录存在但 project 不可解析”。

---

## Phase 4：兼容层收口与旧分支清理

目标：

1. 删除旧 job 业务分支，仅保留最薄转换入口。
2. 删除 job 主 SSE 路径与 job 字幕写链路。

交付：

1. 后端旧路由降级为 410/明确错误（视发布策略）。
2. 前端去除 job fallback 代码。

Gate：

1. 全量回归通过，且代码扫描不再出现 `job subtitles` 写接口调用。

---

## 8. 兼容策略（严格版）

1. 任意旧 `job_id` 输入：
   - 可解析则立即转换并继续；
   - 不可解析直接报错。
2. 禁止“失败后继续走旧 job 字幕逻辑”。
3. 旧接口保留时间窗由发布策略控制，但逻辑必须是“转换壳”，不能是第二业务实现。

---

## 9. Full/Lite 留位说明（本期不实现）

1. 保持 `capabilitySnapshot` 字段与 selector 调用面不变。
2. 本方案仅变更身份语义与目录命名，不改能力矩阵逻辑。
3. 后续 Full/Lite 改造可在纯 project 语义上继续推进，不再引入 job 分支。

---

## 10. 风险清单与规避

1. **运行中任务目录改名风险**：通过“运行态跳过 + 下次重试”规避。
2. **Windows 路径大小写/占用风险**：迁移前加存在性与占用检查，失败可回滚。
3. **旧客户端仍写 job 接口**：兼容壳统一转换并输出 deprecation 日志。
4. **SSE 双频道期间事件重复**：以 `project` 为主，`job` 频道仅镜像且有下线日期。
5. **导出前同步不一致**：继续保留“未同步禁止导出”硬校验。

---

## 11. 验收清单（最终）

1. 新任务目录命名全部符合 `p-{date}-{time}-{mode}-{slug}-{rand4}`。
2. 前后端主链路只使用 `project_id`。
3. 旧 `job_id` 输入全部走自动转换，失败直接报错。
4. 无任何 `job subtitles` 主写入分支残留。
5. 目录迁移具备 dry-run、apply、rollback 三套脚本且可重复执行。

