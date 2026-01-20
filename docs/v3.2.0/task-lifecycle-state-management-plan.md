# V3.2.0 任务生命周期与状态仓库改造方案（方案A）

## 1. 背景与目标

当前任务状态来自 `job_meta.json` / `checkpoint.json` / `queue_state.json` / 内存队列，存在多源不一致与“扫描兜底”逻辑，导致前后端状态漂移。目标是建立**单一权威状态源**，完整追踪任务生命周期，支持暂停、断点恢复、取消、删除，彻底解决状态不明确与管理混乱。

**最终目标**：
- 所有任务状态完全可控（无推断/扫描兜底）。
- 暂停/恢复/取消/删除的状态转移可追踪、可回放。
- 重启后任务状态准确恢复，前后端一致。

## 2. 设计理念

- **单一权威状态源**：任务状态只从一个存储读写，所有接口统一依赖。
- **显式状态机**：明确允许的状态迁移，禁止隐式推断。
- **事务一致性**：一次操作内完成“队列 + 任务状态 + 事件记录”的一致更新。
- **可追踪**：记录任务事件流，复盘每次暂停/恢复/失败原因。
- **可恢复**：重启时根据心跳/租约自动纠偏，避免“假运行态”。

## 3. 核心方案（SQLite 状态仓库）

**选择 SQLite 的理由**：
- 无外部依赖，适合本地单机部署。
- 支持事务与索引，适合一致性与查询性能需求。
- 便于后续扩展统计/审计。

**核心数据域**：
- 任务主表（Task）：当前状态、阶段、进度、路径、设置摘要。
- 任务事件表（TaskEvent）：状态迁移、暂停/恢复/取消/失败原因。
- 队列状态表（QueueState）：队列顺序、运行任务、被中断任务。
- 检查点表（Checkpoint）：最新检查点摘要（指针/哈希/更新时间）。
- 运行心跳（TaskHeartbeat）：运行中任务的最后心跳时间。

## 4. 状态机定义

**允许的主状态流**：
- created → uploaded → queued → processing → finished
- processing → paused → queued → processing
- processing → failed / canceled
- queued → paused / canceled

**终态定义**：
- finished / failed / canceled

**重启统一暂停规则**：
- 启动时将所有**非终态**任务统一标记为 `paused`（包含 created/uploaded/queued/processing/paused）。
- 为 `processing/queued` 记录事件：`reason=system_restart`，防止前端误判为“转录中”。
- 运行中租约清空，避免恢复后误认为仍在占用执行器。

**心跳/租约纠偏规则**：
- 运行态任务必须持有租约（`lease_owner`、`lease_expires_at`），并定期心跳刷新。
- `processing` 且租约过期 → `paused`（原因：lease_expired）。
- `processing` 且心跳过期 → `paused`（原因：heartbeat_timeout）。
- `queued` 但不在队列 → `paused`（原因：queue_desync）。

## 5. 中断恢复流程

1. 启动时加载仓库中的 Task/QueueState/Checkpoint。
2. 执行“重启统一暂停规则”，将所有非终态任务写回 `paused` 并写事件。
3. 清空内存队列并重建为空队列，确保不会自动继续执行。
4. `/sync-tasks` 从仓库读取状态，任务列表与编辑器统一显示暂停。
5. 用户恢复任务时，从仓库读取检查点；若缺失则从头转录。

## 6. 实现方案（分阶段）

### Phase A：引入状态仓库与读写入口
- 新增 `TaskStateRepository`（SQLite CRUD + 事务）。
- `JobLifecycleService` 改为只读/写仓库，不再写 `job_meta.json` 作为权威。

### Phase B：队列服务接入仓库
- `JobQueueService` 只通过仓库维护队列顺序与任务状态。
- `resume/pause/cancel` 均事务化更新任务表 + 队列表 + 事件表。
- 重启恢复：读取 `QueueState` + `TaskHeartbeat` 自动纠偏。

### Phase C：API 统一读取
- `/sync-tasks` 直接读仓库，不再扫描目录推断状态。
- `/status/{job_id}`、`/resume/{job_id}` 统一读仓库状态。
- SSE 状态推送基于仓库事件。

### Phase D：断点恢复统一
- `Checkpoint` 表只保存最新摘要（路径/哈希/更新时间）。
- 断点恢复 API 从仓库读取并验证一致性。

## 7. 修改细节（文件级）

**新增模块**：
- `backend/app/services/task_state_repository.py`：SQLite 访问层
- `backend/app/services/task_event_bus.py`：事件落库与广播
- `backend/app/services/task_heartbeat.py`：运行心跳记录

**改造模块**：
- `backend/app/services/job_lifecycle_service.py`：改为仓库读写
- `backend/app/services/job_queue_service.py`：状态迁移事务化
- `backend/app/api/routes/transcription_routes.py`：API 统一读仓库
- `backend/app/services/progress_emitter.py`：进度状态统一来源

**移除/替代**：
- `/sync-tasks` 的目录扫描兜底逻辑
- 直接读取 `job_meta.json` 的前端状态源

## 8. 数据迁移策略

- 首次启动：读取旧 `job_meta.json` / `queue_state.json` / `checkpoint.json`，
  写入 SQLite 并记录迁移事件。
- 迁移完成后不再依赖旧文件，只保留备份用于诊断。

## 9. 测试与验收

- 单元测试：状态机迁移、事务一致性、心跳过期纠偏。
- 集成测试：暂停→重启→恢复→继续转录，状态一致。
- 回归测试：任务列表与编辑器状态一致、恢复不再 400。

## 10. 风险与回滚

- 风险：迁移错误导致状态丢失。
- 回滚：保留旧 `job_meta.json` 镜像，可恢复旧模式。

## 11. 交付标准

- 任务生命周期全链路可追踪（事件表完整）。
- 前后端仅使用单一状态源，无推断/扫描兜底。
- 暂停/恢复/取消/删除全流程可控且一致。
