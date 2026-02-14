# M4 S-Epoch 与 W-Epoch 性能优化方案

> Type: Architecture | Status: Active
> Version: V3.2.0+dev.20260213.06
> Scope: 双 Epoch 调度、吞吐、首条延迟与资源占用优化

## 1. Summary

* **Goal**: 在不牺牲切分质量的前提下，降低首条可见延迟、提升吞吐，并控制 GPU 峰值占用。
* **核心原则**:
  - 质量优先：不以牺牲关键切分约束换吞吐。
  - 增量优先：只重算受影响窗口，不做全量重复计算。
  - 非阻塞优先：LLM 复核绝不阻塞主链完成。

## 2. Diagram

* `Preprocess` -> `Fast Loop` -> `S-Epoch` -> `Bridge` -> `W-Epoch` -> `Finalize`
* `Fast/Slow结果差异` -> `Affected Window Detector` -> `局部重算`
* `主链输出` + `LLM异步复核` -> `修订事件`

## 3. Key Components

* `backend/app/pipelines/async_dual_pipeline.py`: 乱序执行、顺序提交主编排。
* `backend/app/services/timeline/speaker_timeline_service.py`: S-Epoch 时间线构建。
* `backend/app/services/bridge/turn_group_builder.py`: Bridge 组批与 flush 触发。
* `backend/app/services/segmentation/segmentation_processor.py`: 后处理切分裁决。

## 4. 优化目标与指标

1. 首条可见延迟（TTFS）下降或持平。
2. 单位时间处理块数（throughput）提升。
3. GPU 峰值显存不高于现状上限。
4. `deferred` 最终收敛时间可控（避免尾部拖延过长）。

## 5. 调度优化策略

### 5.1 SuperBlock 参数收敛

1. 默认块长建议 `6-10 分钟`。
2. 首块采用 `60-90 秒` 小块先行，降低首条等待时间。
3. 后续块按稳定长度滚动处理。

### 5.2 GPU 占用策略

1. 严格串行：`S-Epoch -> W-Epoch`，避免并载显存冲突。
2. 快流维持 CPU/ONNX 并行，不与 GPU 阶段争抢。
3. 配合模型常驻策略，减少重复加载抖动。

### 5.3 Bridge 背压优化

1. 触发 flush 条件显式化：语言切换、稳定 speaker_change、超时。
2. 控制批次上限，避免慢流长尾。
3. 对高风险组批优先处理，降低 deferred 堆积。

## 6. 后处理与复核性能优化

### 6.1 局部重算

1. 只重算受影响窗口，不重跑整个句段。
2. 受影响判定复用 M2 规则。

### 6.2 缓存复用

1. 复用对齐中间结果（词级映射、锚点评分）。
2. 复用 speaker 证据轨快照，避免重复融合。

### 6.3 deferred 队列治理

1. 队列上限：`MAX_DEFERRED_WINDOWS = 3`。
2. 超限策略：强制处理最旧窗口，打 `overflow_forced`。
3. 统计 `deferred_age_p95` 作为调参依据。

### 6.4 LLM 异步复核

1. 只复核高风险窗口，不全量复核。
2. 通过 SSE 下发修订，不阻塞主链。
3. 设置超时与预算，超限直接回退标点备用链路。

## 7. 压测与验收

1. 压测场景：
  - 单人长讲解。
  - 双人快速对话（高切换频率）。
  - 三人以上交替发言（复杂窗口堆积场景）。
2. 验收阈值：
  - TTFS 不回退。
  - 吞吐提升或持平且质量不下降。
  - GPU 峰值在预算内。
  - `deferred_forced` 比例可控。

## 8. 风险与回退

1. 风险：过度追求吞吐导致分句质量下滑。  
回退：恢复保守参数，优先保障 P0/P1。
2. 风险：局部重算遗漏导致脏状态。  
回退：窗口校验失败时退化为块级重算。
3. 风险：LLM 队列拥堵。  
回退：降级到仅标点备用，不阻塞交付。

