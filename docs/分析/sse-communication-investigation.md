# SSE 管理和前后端通讯机制深度调查报告

**版本**: v3.1.0.3+
**调查日期**: 2025-12-26
**调查范围**: SSE 核心架构、事件系统、进度推送、字幕流式传输、暂停/恢复机制

---

## Code Sections (The Evidence)

### 后端 SSE 核心

- `backend/app/services/sse_service.py` (SSEManager): 统一 SSE 连接管理器，支持多频道、自动重连、线程安全
  - `SSEManager.__init__` (31-53): 初始化连接池和配置（心跳间隔、队列大小）
  - `SSEManager.subscribe` (55-140): 订阅频道并推送 SSE 流（支持初始状态回调、心跳检测）
  - `SSEManager.broadcast` (141-184): 向频道异步广播消息（非阻塞队列、容量检查）
  - `SSEManager.broadcast_sync` (195-249): 从后台线程安全地推送 SSE（线程安全调度）
  - `SSEManager._format_sse` (251-263): 标准 SSE 格式化（`event: xxx\ndata: {...}\n\n`）
  - 辅助函数 (350-369): `push_progress_event`, `push_subtitle_event`, `push_signal_event`

### 事件类型系统（命名空间化）

- `backend/app/services/sse_service.py` (事件 Tag 定义)
  - **进度事件** (312-324): `progress.overall`, `progress.extract`, `progress.bgm_detect`, `progress.spectrum_analysis`, `progress.demucs`, `progress.vad`, `progress.sensevoice`, `progress.whisper`, `progress.llm_proof`, `progress.llm_trans`, `progress.srt`
  - **字幕流式事件** (326-334): `subtitle.sv_segment`, `subtitle.sv_sentence`, `subtitle.whisper_patch`, `subtitle.llm_proof`, `subtitle.llm_trans`, `subtitle.batch_update`, `subtitle.draft`, `subtitle.finalized`, `subtitle.replace_chunk`, `subtitle.restored`
  - **信号事件** (336-345): `signal.job_start`, `signal.job_complete`, `signal.job_failed`, `signal.job_paused`, `signal.job_canceled`, `signal.job_resumed`, `signal.phase_start`, `signal.phase_complete`, `signal.circuit_breaker`, `signal.model_upgrade`

### 后端进度发射器（v3.1.0.1+）

- `backend/app/services/progress_emitter.py` (ProgressEventEmitter): 统一进度发射器
  - `ProgressEventEmitter.__init__` (96-120): 初始化模式和权重（SENSEVOICE_ONLY/WHISPER_PATCH/DUAL_STREAM）
  - 权重配置 (74-94): 预处理、快流、慢流、对齐的阶段权重（动态模式调整）
  - `update_preprocess` (129-149): 更新预处理进度（音频提取、VAD、频谱分诊、人声分离）
  - `update_fast` (151-182): 更新快流（SenseVoice）进度
  - `update_slow` (184-214): 更新慢流（Whisper）进度
  - `update_align` (216-245): 更新对齐进度
  - `_recalculate_and_push` (284-337): 计算加权总进度并推送 SSE（防倒退保护）
  - `_push_overall` (350-391): 推送总体进度事件（包含 `detail` 字段：fast/slow/align/preprocess）
  - `to_checkpoint_data` (439-456): 导出进度数据到 checkpoint
  - `restore_from_checkpoint` (458-556): 从 checkpoint 恢复进度状态

### 后端字幕管理器

- `backend/app/services/streaming_subtitle.py` (StreamingSubtitleManager): 流式字幕管理器
  - `add_sentence` (54-91): 添加新句子（SenseVoice 阶段）并推送 SSE
  - `update_sentence` (93-147): 更新已有句子（Whisper 复核或 LLM 校对）
  - `mark_for_deletion` (177-202): 标记句子为待删除（Whisper 仲裁后）
  - `add_draft_sentences` (279-347): 添加草稿句子（快流/双模态架构）
  - `replace_chunk` (349-439): 替换 Chunk 的所有句子（慢流/双模态架构）
  - `add_finalized_sentences` (441-514): 添加定稿句子（极速模式专用）
  - `to_checkpoint_data` (518-550): 导出字幕快照到 checkpoint（v3.1.0.3）
  - `restore_from_checkpoint` (552-655): 从 checkpoint 恢复字幕状态（v3.1.0.3）
  - `push_restored_subtitles_to_frontend` (657-698): 恢复后推送所有字幕到前端

### 前端 SSE 频道管理器

- `frontend/src/services/sseChannelManager.js` (SSEChannelManager): 前端 SSE 频道管理器
  - `subscribeGlobal` (43-87): 订阅全局事件流（所有任务状态变化）
  - `subscribeJob` (105-327): 订阅单个任务事件流（支持双模态架构）
  - 事件映射 (164-326): 完整的事件处理器映射（进度、信号、字幕、视频转码）
  - `handleOverallProgress` (111-114): 总体进度处理函数（更新主进度条）
  - `handlePhaseProgress` (118-123): 阶段进度处理函数（仅日志，不更新主进度条）
  - `_createConnection` (409-489): 创建 EventSource 连接（自动重连、404 检测）
  - `_scheduleReconnect` (495-543): 计划重连（指数退避、最大重连次数）

### 前端辅助 SSE 服务

- `frontend/src/services/sseService.js` (SSEService): 基础 SSE 服务（带降级轮询）
  - `createEventSource` (81-122): 创建 EventSource 连接
  - `handleMessage` (156-192): 消息处理和去重
  - `startPolling` (322-354): 降级到轮询模式（SSE 不可用时）
  - `startHeartbeat` (293-300): 心跳检测（超时触发重连）

- `frontend/src/composables/useSseManager.js` (useSseManager): Vue 3 组合式 API 封装
  - `subscribe` (23-36): 订阅 SSE 事件
  - `ensureConnection` (60-139): 确保 SSE 连接已建立
  - `subscribeVideoProgress` (146-180): 订阅视频生成进度事件

### 暂停/恢复 SSE 支持（v3.1.0+）

- `backend/app/services/job_queue_service.py` (暂停机制)
  - `pause_job` (175-234): 暂停任务并推送 `pause_pending` 信号
  - v3.1.0.2 新增状态 `pausing`: 区分"正在暂停"和"已暂停"
  - 预处理→转录过渡检查点 (708-718): 防止暂停信号被忽略

- `backend/app/utils/cancellation_token.py` (CancellationToken)
  - `enter_atomic_region` (255-263): 进入原子区域（禁止中断）
  - `exit_atomic_region` (267-293): 退出原子区域并处理待处理暂停/取消
  - `check_and_save` (约300行): 检查暂停/取消并保存 checkpoint

### 双流进度推送机制（v3.1.0.1+）

- `backend/app/services/progress_emitter.py`
  - `_push_overall` (350-391): 推送 `progress.overall` 事件，包含 `detail` 字段
    - `detail.fast`: FastWorker (SenseVoice) 进度
    - `detail.slow`: SlowWorker (Whisper) 进度
    - `detail.align`: AlignmentWorker 进度
    - `detail.preprocess`: 预处理阶段进度
  - 防倒退保护 (300-308): 只允许进度增加，防止阶段不同步导致抖动

- `frontend/src/services/sseChannelManager.js`
  - 区分 `handleOverallProgress` 和 `handlePhaseProgress` (109-123)
  - 仅 `progress.overall` 更新主进度条，其他阶段进度单独处理

### 字幕实时流式传输（双模态架构）

- `backend/app/services/streaming_subtitle.py`
  - `add_draft_sentences` (279-347): 推送草稿字幕（快流/SenseVoice）
    - SSE 事件: `subtitle.draft`
    - 深拷贝句子对象，避免共享引用（V3.8）
  - `replace_chunk` (349-439): 推送 Chunk 替换（慢流/Whisper）
    - SSE 事件: `subtitle.replace_chunk`
    - 批量推送所有句子数据
  - `add_finalized_sentences` (441-514): 推送定稿字幕（极速模式）
    - SSE 事件: `subtitle.finalized`

- `frontend/src/services/sseChannelManager.js`
  - 双模态架构事件处理 (221-247)
    - `subtitle.draft`: 草稿字幕（快流）
    - `subtitle.replace_chunk`: 替换 Chunk（慢流）
    - `subtitle.finalized`: 定稿字幕（极速模式）
    - `subtitle.restored`: 恢复字幕（断点续传）

### 字幕实时持久化（v3.1.0.3）

- `backend/app/services/streaming_subtitle.py`
  - `to_checkpoint_data` (518-550): 导出字幕快照
    - `sentences_snapshot`: 所有句子的序列化数据
    - `sentence_count`: 全局句子计数器（恢复时从此值继续）
    - `chunk_sentences_map`: Chunk 到句子索引的映射
  - `restore_from_checkpoint` (552-655): 恢复字幕状态
    - 重建 `SentenceSegment` 对象
    - 恢复 `sentence_count`，确保新句子索引不会冲突
    - 恢复字级时间戳（`WordTimestamp`）

- `backend/app/services/job/checkpoint_manager.py` (CheckpointV37)
  - `TranscriptionState` 新增字段 (约88行)
    - `sentences_snapshot`: List[Dict[str, Any]]
    - `sentence_count`: int
    - `chunk_sentences_map`: Dict[int, List[int]]

### 线程安全和并发控制

- `backend/app/services/sse_service.py`
  - `broadcast_sync` (195-249): 使用 `asyncio.run_coroutine_threadsafe` 从后台线程调度协程
  - 主事件循环引用 (51, 185-193): 在应用启动时设置 `self.loop`
  - 容量检查 (165-169): 队列接近满时跳过更新，避免阻塞

- `backend/app/services/streaming_subtitle.py`
  - V3.8 锁保护 (49-52, 302, 384): 使用 `threading.RLock()` 保护字幕管理器
  - 深拷贝句子对象 (305, 467): 避免共享引用导致的竞态条件
  - 锁外推送 SSE (328-339, 403-429, 495-507): 避免长时间持锁

### Checkpoint 集成

- `backend/app/services/job_queue_service.py` (恢复流程)
  - `_run_dual_alignment_pipeline` 恢复点 (约745行)
    - 加载 checkpoint
    - 恢复 `ProgressEventEmitter` 状态
    - 恢复 `StreamingSubtitleManager` 字幕快照
    - 推送 SSE 事件通知前端恢复完成

- `backend/app/pipelines/preprocessing_pipeline.py` (v3.1.0.2)
  - 保存 `chunks_metadata` 到 checkpoint (约170行)
  - 从 `chunks_metadata` 恢复 chunks，跳过 VAD (约95-177行)

---

## Report (The Answers)

### result

#### 1. SSE 管理核心架构

**SSEManager (后端核心)**

video_to_srt_gpu 项目实现了一个**统一的 SSE 连接管理器**，支持：

- **多频道隔离**: 使用频道 ID (`global`, `job:{job_id}`, `models`) 隔离不同的事件流
- **连接池管理**: `connections: Dict[str, List[asyncio.Queue]]` - 每个频道维护多个客户端连接
- **心跳保活**: 每 10 秒发送 `ping` 事件，超时自动断开
- **线程安全**: 支持从后台线程（转录任务、模型下载）向 SSE 推送消息
  - 使用 `asyncio.run_coroutine_threadsafe` 调度协程到主事件循环
  - 保存主事件循环引用 `self.loop`（在应用启动时设置）
- **非阻塞推送**: 使用 `queue.put_nowait()` 避免阻塞，队列满时跳过
- **初始状态推送**: 连接建立时可选推送 `initial_state`，支持断线重连后全量拉取

**SSEChannelManager (前端核心)**

前端实现了一个**基于 EventEmitter 的频道管理器**，支持：

- **频道复用**: 使用 Map 存储频道连接，避免重复连接
- **自动重连**: 连接失败时指数退避重连（最大 5 次，最长 30 秒）
- **智能断开检测**: 检测 HTTP 404 错误，停止对不存在任务的重连
- **事件路由**: 根据事件类型（`progress.*`, `signal.*`, `subtitle.*`）分发到不同处理器
- **延迟取消订阅**: 给一个短暂的缓冲期，支持标签页快速切换时复用连接

#### 2. 事件类型系统（命名空间化）

项目采用**命名空间化事件类型**（v3.1.0+），解决事件冲突和分类问题：

**进度事件** (`progress.*`)
- `progress.overall` - 总体进度（v3.1.0.1+ 包含 `detail` 字段：fast/slow/align/preprocess）
- `progress.extract`, `progress.vad`, `progress.demucs` - 预处理阶段
- `progress.sensevoice`, `progress.whisper` - 转录阶段
- `progress.fast`, `progress.slow`, `progress.align` - v3.1.0.2+ 双流流水线专用

**字幕流式事件** (`subtitle.*`)
- `subtitle.draft` - 草稿字幕（快流/SenseVoice，双模态架构）
- `subtitle.finalized` - 定稿字幕（极速模式）
- `subtitle.replace_chunk` - 替换 Chunk（慢流/Whisper，双模态架构）
- `subtitle.restored` - 恢复字幕（断点续传后，v3.1.0.3）
- `subtitle.sv_sentence`, `subtitle.whisper_patch`, `subtitle.llm_proof`, `subtitle.llm_trans` - 旧版兼容

**信号事件** (`signal.*`)
- `signal.job_start`, `signal.job_complete`, `signal.job_failed` - 任务生命周期
- `signal.job_paused`, `signal.job_resumed`, `signal.job_canceled` - 任务控制
- `signal.pause_pending` - v3.1.0.2+ 暂停挂起信号（正在等待流水线响应）
- `signal.circuit_breaker`, `signal.model_upgrade` - 熔断和模型升级

**设计优势**：
1. **避免冲突**: 不同模块的事件不会相互干扰
2. **清晰分类**: 前端可根据前缀快速过滤事件（如 `progress.*` 用于进度条）
3. **易于扩展**: 新功能可添加新的命名空间（如 `video.*` 视频转码事件）
4. **向后兼容**: 保留旧事件类型，逐步迁移

#### 3. 后端 SSE 发送链路

**ProgressEventEmitter (v3.1.0.1+)**

项目引入了**统一进度发射器**，解决双流流水线进度不同步的问题：

- **三种模式**: `SENSEVOICE_ONLY` (极速模式)、`WHISPER_PATCH` (复核模式)、`DUAL_STREAM` (双流模式)
- **阶段权重**: 预处理 10%、快流 50%、慢流 30%、对齐 10%（双流模式）
- **加权进度计算**: `total = preprocess * 0.1 + fast * 0.5 + slow * 0.3 + align * 0.1`
- **防倒退保护**: 只允许进度增加，防止节流导致的阶段不同步（v3.1.0.2）
- **节流推送**: 0.5 秒节流间隔，关键节点强制推送（阶段完成时）
- **双层推送**: 同时推送阶段进度（`progress.fast`）和总体进度（`progress.overall`）

**StreamingSubtitleManager (字幕流式推送)**

项目实现了**流式字幕管理器**，支持多阶段增量更新：

- **双模态架构** (V3.5+):
  - `add_draft_sentences()` - 推送草稿字幕（快流/SenseVoice）
  - `replace_chunk()` - 替换 Chunk（慢流/Whisper）
  - `add_finalized_sentences()` - 推送定稿字幕（极速模式）
- **Chunk 级别映射**: `chunk_sentences[chunk_index] = [sentence_index_1, ...]`
- **SSE 事件推送**: 每个操作都推送对应的 SSE 事件（`subtitle.draft`, `subtitle.replace_chunk`, `subtitle.finalized`）
- **线程安全** (V3.8): 使用 `threading.RLock()` 保护字幕管理器，深拷贝句子对象避免竞态条件
- **锁外推送**: 在锁外推送 SSE 事件，避免长时间持锁导致死锁

**转录服务中的事件发送点**

- **预处理阶段**: `emitter.update_preprocess(percent, stage)`
  - 音频提取、VAD 切分、频谱分诊、人声分离
- **FastWorker 循环**: `emitter.update_fast(i + 1, total_chunks)`
  - SenseVoice 推理和立即分句
- **SlowWorker 循环**: `emitter.update_slow(i + 1, total_chunks)`
  - Whisper 复核和上文更新
- **AlignmentWorker 循环**: `emitter.update_align(i + 1, total_chunks)`
  - 字级时间戳对齐

#### 4. 前端 SSE 接收架构

**SSEChannelManager (频道管理器)**

前端通过 **SSEChannelManager** 统一管理所有 SSE 连接：

- **频道隔离**: `global` 频道（所有任务）、`job:{job_id}` 频道（单个任务）、`models` 频道（模型下载）
- **事件分发**: 根据事件类型分发到不同处理器（`onProgress`, `onSignal`, `onSubtitleUpdate`）
- **自动重连**: 连接失败时指数退避重连（1s → 2s → 4s → 8s → 16s → 30s）
- **智能停止重连**: 检测 HTTP 404 错误（任务不存在），停止无效重连
- **延迟取消订阅**: 标签页切换时不立即关闭连接，给一个短暂的缓冲期

**事件处理器映射**

前端实现了**完整的事件处理器映射**：

```javascript
// v3.1.0.2: 区分总体进度和阶段进度
'progress.overall': handleOverallProgress,  // 只有这个更新主进度条
'progress.extract': handlePhaseProgress,    // 仅日志，不更新主进度条
'progress.fast': handlePhaseProgress,
'progress.slow': handlePhaseProgress,
// ... 其他阶段

// 信号事件
'signal.job_complete': handleSignal,
'signal.job_paused': handleSignal,
'signal.pause_pending': handleSignal,  // v3.1.0.2+ 暂停挂起

// 字幕流式事件（双模态架构）
'subtitle.draft': (data) => { handlers.onDraft?.(data) },
'subtitle.replace_chunk': (data) => { handlers.onReplaceChunk?.(data) },
'subtitle.finalized': (data) => { handlers.onFinalized?.(data) },
'subtitle.restored': (data) => { handlers.onRestored?.(data) },
```

**多层进度条的更新策略**

前端从 SSE 提取双流进度（而非字幕数计算）：

```javascript
// v3.1.0.2 修复: 从 SSE 后端推送更新双流进度
onProgress(data) {
  if (data.detail) {
    // 提取 detail 字段中的真实进度
    projectStore.updateDualStreamProgressFromSSE({
      fastStream: Math.round(data.detail.fast || 0),
      slowStream: Math.round(data.detail.slow || 0),
      totalChunks: data.total || projectStore.dualStreamProgress.totalChunks
    })
  }
}
```

#### 5. 双流进度推送机制（v3.1.0+）

**后端设计**

v3.1.0+ 引入了**双流独立进度推送**，解决进度卡顿问题：

- **FastWorker 独立进度**: `emitter.update_fast(processed, total)` - 基于实际 Chunk 处理数
- **SlowWorker 独立进度**: `emitter.update_slow(processed, total)` - 基于实际 Chunk 处理数
- **AlignmentWorker 独立进度**: `emitter.update_align(processed, total)` - 基于实际 Chunk 对齐数
- **progress.overall 中的 detail 字段**:
  ```json
  {
    "percent": 45.2,
    "phase": "sensevoice",
    "status": "processing",
    "total": 20,
    "detail": {
      "preprocess": 100,
      "fast": 60,
      "slow": 30,
      "align": 15
    }
  }
  ```

**前端提取策略**

前端从 SSE 事件中提取双流进度（而非字幕数计算）：

- **为什么要从 detail 而非字幕数计算**:
  1. **准确性**: 后端基于实际 Chunk 处理数计算，而非字幕条数（一个 Chunk 可能产生多条字幕）
  2. **独立性**: 快流和慢流处理进度独立，前端字幕数无法分别计算
  3. **实时性**: 后端推送真实进度，前端无需等待字幕完全生成

**v3.1.0.2 进度抖动修复**

问题：后端同时推送 `progress.overall`（总进度）和 `progress.{phase}`（阶段进度），前端将所有事件都更新主进度条，导致从 64% 突然降到 20%。

修复：区分事件处理器
- `handleOverallProgress`: 总体进度处理函数（更新主进度条）
- `handlePhaseProgress`: 阶段进度处理函数（仅日志，不更新主进度条）
- 前端防倒退保护: 检测进度倒退 > 5%，忽略异常事件

#### 6. 暂停/恢复的 SSE 支持（v3.1.0+）

**暂停机制完整覆盖**

v3.1.0+ 实现了**完整的暂停机制**，覆盖所有阶段：

- **音频提取 (FFmpeg)**: 独立原子区域，等待完成后检查暂停
- **VAD 切分**: 独立原子区域（快速 < 1 秒）
- **频谱分诊**: 每 Chunk 检查点
- **人声分离 (Demucs)**: 全局模式为原子区域，按需模式每 Chunk 检查点
- **预处理→转录过渡点** (v3.1.0.2): 新增检查点，防止暂停信号被忽略
- **FastWorker (极速模式)** (v3.1.0.2): 每 Chunk 检查点
- **SlowWorker**: 每 Chunk 检查点
- **AlignmentWorker**: 每 Chunk 检查点

**SSE 信号事件**

- `signal.pause_pending` (v3.1.0.2): 暂停挂起信号
  - 用户点击暂停 → 推送 `pause_pending` → 前端显示"正在暂停..."
  - 流水线响应后 → 推送 `job_paused` → 前端显示"已暂停"
- `signal.job_paused`: 任务已暂停
- `signal.job_resumed`: 任务已恢复

**前端 UI 状态同步**

前端通过 SSE 事件同步暂停/恢复状态：

```javascript
'signal.pause_pending': (data) => {
  taskStatus.value = 'pausing'  // 显示"正在暂停..."
},
'signal.job_paused': (data) => {
  taskStatus.value = 'paused'   // 显示"已暂停"
},
'signal.job_resumed': (data) => {
  taskStatus.value = 'processing'  // 显示"处理中"
}
```

**断点续传后的进度恢复**

恢复时从 checkpoint 加载进度并推送 SSE：

1. 加载 checkpoint 数据
2. 恢复 `ProgressEventEmitter` 状态（`restore_from_checkpoint`）
3. 立即推送 `progress.overall` 事件（`force=True`），让前端恢复到正确进度
4. 恢复 `StreamingSubtitleManager` 字幕快照
5. 推送 `subtitle.restored` 事件，让前端重建字幕列表

#### 7. 字幕实时流式传输（双模态架构）

**draft/finalized/chunk_replace 事件**

项目实现了**双模态字幕流式传输**，支持草稿和定稿分离：

- **草稿字幕** (`subtitle.draft`):
  - FastWorker 处理完 Chunk 后立即推送
  - 前端显示为草稿状态（可能有误差）
  - 事件数据: `{ index, chunk_index, sentence, is_draft: true }`

- **定稿字幕** (`subtitle.finalized`):
  - 极速模式（SENSEVOICE_ONLY）下 FastWorker 直接输出定稿
  - 事件数据: `{ index, chunk_index, sentence, is_finalized: true }`

- **Chunk 替换** (`subtitle.replace_chunk`):
  - SlowWorker (Whisper) 处理完 Chunk 后批量替换
  - 删除旧的草稿句子，添加新的定稿句子
  - 事件数据: `{ chunk_index, old_indices, new_indices, sentences }`

**草稿字幕与定稿字幕的前端渲染策略**

前端通过不同的事件处理器渲染草稿和定稿：

```javascript
'subtitle.draft': (data) => {
  // 添加草稿字幕（半透明显示）
  subtitles.value.push({
    ...data.sentence,
    isDraft: true,
    opacity: 0.6
  })
},
'subtitle.replace_chunk': (data) => {
  // 删除旧的草稿句子
  data.old_indices.forEach(idx => {
    const index = subtitles.value.findIndex(s => s.index === idx)
    if (index > -1) subtitles.value.splice(index, 1)
  })

  // 添加新的定稿句子
  data.sentences.forEach(sentence => {
    subtitles.value.push({
      ...sentence,
      isDraft: false,
      isFinalized: true
    })
  })
}
```

**批量更新优化**

- **Chunk 级别批量推送**: 一次推送整个 Chunk 的所有句子，减少 SSE 事件数量
- **深拷贝句子对象** (V3.8): 避免共享引用导致的竞态条件
- **锁外推送 SSE**: 在锁外推送 SSE 事件，避免长时间持锁导致死锁

#### 8. 字幕实时持久化（v3.1.0.3）

**问题**: 暂停后恢复时字幕索引从 0 重新开始，导致新字幕覆盖旧字幕

**解决方案**: Checkpoint 保存字幕快照和全局计数器

- **sentences_snapshot**: 所有已生成句子的序列化数据
- **sentence_count**: 全局句子计数器（恢复时从此值继续）
- **chunk_sentences_map**: Chunk 到句子索引的映射

**持久化流程**:

1. FastWorker/SlowWorker 处理 Chunk 后调用 `subtitle_manager.to_checkpoint_data()`
2. Checkpoint 保存字幕快照
3. 暂停任务
4. 恢复任务时调用 `subtitle_manager.restore_from_checkpoint(checkpoint_data)`
5. 重建 `SentenceSegment` 对象，恢复 `sentence_count`
6. 推送 `subtitle.restored` 事件到前端
7. 前端重建字幕列表

**关键设计**:

- **sentence_count 保留**: 确保新句子索引不会与已有句子冲突
- **字级时间戳恢复**: 恢复 `WordTimestamp` 对象，支持字幕切分
- **Chunk 映射恢复**: 恢复 `chunk_sentences` 映射，支持按 Chunk 恢复

---

## conclusions

### 1. SSE 架构的独特设计

1. **统一频道管理**: 使用频道 ID 隔离不同的事件流（全局、任务、模型），避免事件混淆
2. **命名空间化事件**: 使用 `progress.*`, `signal.*`, `subtitle.*` 前缀，清晰分类事件类型
3. **线程安全推送**: 支持从后台线程（转录任务、模型下载）向 SSE 推送消息，使用 `asyncio.run_coroutine_threadsafe`
4. **非阻塞推送**: 使用 `queue.put_nowait()` 避免阻塞，队列满时跳过，保证系统响应性
5. **初始状态推送**: 连接建立时可选推送 `initial_state`，支持断线重连后全量拉取
6. **心跳保活**: 每 10 秒发送 `ping` 事件，超时自动断开，保证连接活跃
7. **双层推送**: 同时推送阶段进度（`progress.fast`）和总体进度（`progress.overall`），前端灵活选择

### 2. 与 WebSocket 方案的差异化优势

| 特性 | SSE | WebSocket |
|------|-----|-----------|
| **单向通信** | 服务器→客户端 | 双向通信 |
| **协议** | HTTP/1.1 长连接 | 独立协议（ws://） |
| **自动重连** | 浏览器原生支持 | 需要手动实现 |
| **事件类型** | 原生支持事件类型 | 需要手动解析消息 |
| **代理兼容性** | 良好（HTTP） | 较差（需要代理支持） |
| **消息格式** | 文本（JSON） | 二进制或文本 |
| **适用场景** | 单向推送（进度、日志） | 双向交互（聊天、游戏） |

**video_to_srt_gpu 选择 SSE 的原因**:
1. **单向推送需求**: 转录进度、字幕流式传输只需要服务器→客户端推送，不需要双向通信
2. **浏览器原生支持**: `EventSource` API 自动处理重连、事件类型，减少前端代码复杂度
3. **HTTP 兼容性**: SSE 基于 HTTP/1.1，与现有 FastAPI 框架无缝集成，无需额外配置
4. **事件类型天然支持**: SSE 原生支持事件类型（`event: progress.overall`），无需手动解析消息

### 3. 性能优化和工程实践

1. **节流推送**: 0.5 秒节流间隔，关键节点强制推送（阶段完成时），减少 SSE 事件频率
2. **队列容量检查**: 队列接近满（95%）时跳过更新，避免阻塞
3. **批量推送**: Chunk 级别批量推送字幕（`subtitle.replace_chunk`），减少 SSE 事件数量
4. **深拷贝句子对象**: 避免共享引用导致的竞态条件（V3.8）
5. **锁外推送 SSE**: 在锁外推送 SSE 事件，避免长时间持锁导致死锁
6. **防倒退保护**: 只允许进度增加，防止节流导致的阶段不同步（v3.1.0.2）
7. **智能停止重连**: 检测 HTTP 404 错误（任务不存在），停止无效重连

### 4. 容错机制和边界情况处理

1. **断线重连**: 前端指数退避重连（1s → 2s → 4s → 8s → 16s → 30s），最大 5 次
2. **初始状态推送**: 连接建立时推送 `initial_state`，支持断线重连后全量拉取
3. **心跳保活**: 每 10 秒发送 `ping` 事件，超时自动断开
4. **线程安全**: 使用 `asyncio.run_coroutine_threadsafe` 从后台线程调度协程，避免竞态条件
5. **锁保护**: 使用 `threading.RLock()` 保护字幕管理器，避免并发修改
6. **降级轮询**: SSE 不可用时降级到轮询模式（`sseService.js`）
7. **暂停挂起检测**: 推送 `pause_pending` 信号，前端显示"正在暂停..."，避免用户误操作
8. **Checkpoint 持久化**: 字幕快照保存到 checkpoint，恢复时重建字幕列表，避免字幕丢失

### 5. 创新点和闪光点

1. **双模态字幕流式传输**: 草稿字幕（快流/SenseVoice）和定稿字幕（慢流/Whisper）分离，用户可实时看到草稿并等待定稿
2. **统一进度发射器**: `ProgressEventEmitter` 统一管理进度计算、SSE 推送、Checkpoint 持久化，解决双流流水线进度不同步问题
3. **命名空间化事件**: 使用 `progress.*`, `signal.*`, `subtitle.*` 前缀，清晰分类事件类型，易于扩展
4. **双流独立进度推送**: FastWorker、SlowWorker、AlignmentWorker 独立进度，前端从 SSE 提取真实进度（而非字幕数计算）
5. **字幕实时持久化**: Checkpoint 保存字幕快照和全局计数器，恢复时不会出现字幕索引冲突
6. **暂停机制完整覆盖**: 覆盖所有阶段（音频提取、VAD、频谱分诊、人声分离、转录、对齐），响应时间 < 5 秒
7. **线程安全推送**: 支持从后台线程向 SSE 推送消息，使用 `asyncio.run_coroutine_threadsafe`，保证系统稳定性

---

## relations

### 1. 后端核心关系

```
SSEManager (统一 SSE 连接管理器)
    ↓
    ├─ ProgressEventEmitter (统一进度发射器)
    │   ├─ 调用 SSEManager.broadcast_sync() 推送进度事件
    │   ├─ 支持三种模式: SENSEVOICE_ONLY, WHISPER_PATCH, DUAL_STREAM
    │   ├─ 计算加权总进度并推送 progress.overall 事件
    │   └─ 推送阶段进度事件 (progress.fast, progress.slow, progress.align)
    │
    ├─ StreamingSubtitleManager (流式字幕管理器)
    │   ├─ 调用 SSEManager.broadcast_sync() 推送字幕事件
    │   ├─ 双模态架构: add_draft_sentences(), replace_chunk(), add_finalized_sentences()
    │   ├─ 推送 subtitle.draft, subtitle.replace_chunk, subtitle.finalized 事件
    │   └─ v3.1.0.3 持久化: to_checkpoint_data(), restore_from_checkpoint()
    │
    └─ push_progress_event, push_subtitle_event, push_signal_event (辅助函数)
        └─ 封装 SSEManager.broadcast_sync() 调用
```

### 2. 前端核心关系

```
SSEChannelManager (频道管理器)
    ↓
    ├─ 创建 EventSource 连接
    │   ├─ subscribeGlobal() - 订阅全局事件流
    │   ├─ subscribeJob() - 订阅单个任务事件流
    │   └─ subscribeModels() - 订阅模型下载事件流
    │
    ├─ 事件分发
    │   ├─ handleOverallProgress - 总体进度处理函数（更新主进度条）
    │   ├─ handlePhaseProgress - 阶段进度处理函数（仅日志）
    │   ├─ handleSignal - 信号事件处理函数（job_complete, job_paused, etc.）
    │   └─ 字幕事件处理器 (onDraft, onReplaceChunk, onFinalized, onRestored)
    │
    └─ 自动重连机制
        ├─ _createConnection() - 创建 EventSource 连接
        ├─ _scheduleReconnect() - 计划重连（指数退避）
        └─ 智能停止重连 (HTTP 404 检测)
```

### 3. 双流进度推送关系

```
后端 ProgressEventEmitter
    ↓
    ├─ update_fast(processed, total) - 更新快流进度
    ├─ update_slow(processed, total) - 更新慢流进度
    ├─ update_align(processed, total) - 更新对齐进度
    └─ _push_overall() - 推送 progress.overall 事件
        └─ data.detail = { fast, slow, align, preprocess }
            ↓
前端 SSEChannelManager
    ↓
    ├─ handleOverallProgress(data) - 接收 progress.overall 事件
    │   └─ 提取 data.detail.fast, data.detail.slow
    │       ↓
    └─ projectStore.updateDualStreamProgressFromSSE()
        └─ 更新 dualStreamProgress = { fastStream, slowStream, totalChunks }
            ↓
EditorHeader.vue
    └─ 渲染双层进度条 (SenseVoice 进度条, Whisper 进度条)
```

### 4. 暂停/恢复关系

```
用户点击暂停
    ↓
后端 JobQueueService.pause_job()
    ↓
    ├─ token.pause() - 触发暂停信号
    ├─ 推送 signal.pause_pending - 前端显示"正在暂停..."
    └─ 等待流水线响应
        ↓
流水线 (FastWorker/SlowWorker/AlignmentWorker)
    ↓
    ├─ token.check_and_save() - 检查暂停信号
    ├─ 保存 Checkpoint（包含进度、字幕快照）
    └─ 抛出 PausedException
        ↓
后端 JobQueueService
    ↓
    ├─ 捕获 PausedException
    ├─ 设置 job.status = "paused"
    └─ 推送 signal.job_paused - 前端显示"已暂停"
        ↓
前端 SSEChannelManager
    ↓
    └─ handleSignal('job_paused') - 更新 UI 状态
```

### 5. 字幕实时持久化关系

```
后端 FastWorker/SlowWorker 处理 Chunk
    ↓
StreamingSubtitleManager.add_draft_sentences() / replace_chunk()
    ↓
    ├─ 更新 sentences 字典
    ├─ 推送 SSE 事件 (subtitle.draft / subtitle.replace_chunk)
    └─ checkpoint_manager.save_checkpoint()
        └─ 调用 subtitle_manager.to_checkpoint_data()
            └─ 保存 sentences_snapshot, sentence_count, chunk_sentences_map
                ↓
用户暂停任务
    ↓
用户恢复任务
    ↓
后端 JobQueueService.resume_job_internal()
    ↓
    ├─ checkpoint_manager.load_checkpoint_v37()
    ├─ subtitle_manager.restore_from_checkpoint(checkpoint_data)
    │   └─ 重建 SentenceSegment 对象，恢复 sentence_count
    └─ 推送 subtitle.restored 事件
        ↓
前端 SSEChannelManager
    ↓
    └─ handleRestored(data) - 重建字幕列表
        └─ projectStore.restoreSubtitles(data.sentences)
```

### 6. Checkpoint 与 SSE 集成关系

```
后端 Checkpoint 保存
    ↓
    ├─ ProgressEventEmitter.to_checkpoint_data() - 导出进度数据
    ├─ StreamingSubtitleManager.to_checkpoint_data() - 导出字幕快照
    └─ CheckpointManagerV37.save_checkpoint() - 保存到文件
        ↓
后端 Checkpoint 恢复
    ↓
    ├─ CheckpointManagerV37.load_checkpoint_v37() - 从文件加载
    ├─ ProgressEventEmitter.restore_from_checkpoint() - 恢复进度状态
    │   └─ _push_overall(force=True) - 立即推送进度，让前端恢复显示
    ├─ StreamingSubtitleManager.restore_from_checkpoint() - 恢复字幕状态
    │   └─ push_restored_subtitles_to_frontend() - 推送 subtitle.restored 事件
    └─ 继续任务执行
```
