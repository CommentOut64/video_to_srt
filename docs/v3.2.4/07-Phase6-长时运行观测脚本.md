# Phase 6：长时运行观测脚本（24h）

## 目标

在**不改业务行为**前提下，为 Lite/Full 共用代码提供可重复执行的长时观测能力，用于发现：

- 队列内部缓存是否持续增长（`pending_cancel/orphan/runner/sse_publisher`）
- 前端长时间编辑时监听器/定时器是否回收
- 运行过程中是否出现“计数只增不减”的泄漏迹象

## 后端观测接口

- 新增接口：`GET /api/projects/tasks/runtime-diagnostics`
- 返回字段（节选）：
  - `queue_length`
  - `jobs_count`
  - `pending_cancel_count`
  - `orphan_execution_count`
  - `runner_thread_count`
  - `runner_alive_count`
  - `pending_physical_delete_count`
  - `logically_removed_count`
  - `cancellation_token_count`
  - `sse_publisher_cache_count`
  - `is_gpu_busy_override`
  - `is_runner_gate_blocking`

## 后端采样脚本

脚本：`scripts/observe_queue_runtime_diagnostics.py`

### 常用命令

```bash
# 1 小时采样（每 10 秒）
python scripts/observe_queue_runtime_diagnostics.py

# 24 小时采样（每 30 秒）
python scripts/observe_queue_runtime_diagnostics.py \
  --interval-seconds 30 \
  --duration-seconds 86400 \
  --output logs/queue_runtime_diag_24h.jsonl
```

### 输出说明

- 输出文件为 JSONL，每行一条快照，包含采集时间与完整返回体。
- 脚本结束会打印各核心计数字段峰值，便于快速判断是否存在异常增长。

## 前端运行健康采样（默认关闭）

- 新增诊断模块：`frontend/src/composables/runtimeHealthDiagnostics.js`
- 默认关闭，不影响正常运行。

### 控制台启用

```javascript
window.__AF_RUNTIME_HEALTH_DIAG__ = {
  enabled: true,
  sampleIntervalMs: 10000,
  maxSamples: 2000,
  logToConsole: false,
}
```

### 控制台读取状态

```javascript
window.__AF_RUNTIME_HEALTH__.getState()
```

关注字段（WaveformTimeline 采样）：

- `heapUsedMb`
- `hasLoadRetryTimer`
- `hasLateReadyRenderTimer`
- `hasMediaKeydownListener`
- `hasScrollListener`
- `subtitlesCount`

计数器（`counters`）：

- `waveform.media_keydown.bind/unbind`
- `waveform.scroll_listener.bind/unbind`
- `waveform.retry_timer.*`
- `waveform.late_ready_timer.*`

## 判定建议

满足以下特征可判定“无明显泄漏趋势”：

1. 长跑后 `pending_cancel_count/orphan_execution_count/sse_publisher_cache_count` 能回落或稳定在低水位。
2. 前端 `bind` 与 `unbind` 计数长期接近平衡。
3. `heapUsedMb` 在业务波动后可回落，不出现单调上升。
