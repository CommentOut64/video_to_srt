# V3.2.0+dev.20260119.01 统一模型管理补充开发文档

> 目标：补齐统一模型管理在**运行参数 API**、**硬件检测接入**、**智能显存管理**、**去除预热**、**SSE 下载进度**方面的缺口。  
> 本文档基于现有实现与代码调查结果，给出可落地的文件/代码级改造方案。

---

## 0. 调查结论（现状与缺口）

### 0.1 统一模型管理现状
- ModelManagerV2 入口：`backend/app/services/model_manager_v2.py`  
- ModelSpec / Registry：`backend/app/core/asr/model_spec.py`、`backend/app/core/asr/registry.py`  
- 下载器：`backend/app/services/model_downloader.py`  
- SSE 设施：`backend/app/services/sse_service.py`（已有模型频道）  
- 资源治理旧实现：`backend/app/core/resource_manager.py`（未接入 V2）

### 0.2 明确缺口
1) **缺少运行参数 API**：前端无法读取/修改模型运行参数（device/compute_type/cpu_threads 等）。  
2) **硬件检测未接入**：旧硬件检测/优化散落在 `hardware_service.py`、`cpu_affinity_service.py`、`cpu_optimizer.py`，未与 V2 整合。  
3) **显存管理过于简单**：已补齐智能显存策略与动态预算（见第 3 节）。  
4) **预热逻辑仍存在**：`ModelSpec.warmup` + `ModelLoader.warmup` 仍保留。  
5) **SSE 下载进度未接入**：下载进度未向前端实时推送。

---

## 1. 运行参数 API（前端可读写）

### 1.1 运行参数清单（必须可查询/可修改）
**全局参数**（作用于全部模型，允许覆盖硬件推荐）：
- `global.device_preference`：`auto/cuda/cpu`
- `global.allow_download`：是否允许自动下载（默认 false）
- `global.max_vram_mb` / `global.reserved_vram_mb` / `global.max_models`
- `global.cpu_threads` / `global.cpu_affinity_strategy`
- `global.onnx_intra_threads` / `global.onnx_inter_threads`

**单模型参数**（覆盖默认行为）：
- `device`：`auto/cuda/cpu`
- `compute_type`：如 `int8/float16/int8_float16`
- `cpu_threads`：ONNX/CPU 推理线程
- `keep_resident`：是否常驻（影响显存策略）
- `evict_priority`：驱逐优先级（低优先级先被驱逐）
- `max_concurrency`：模型并发上限（可选）

**单模型独立配置（必须支持）**：
- 每个 `model_id` 独立保存运行参数覆盖（per_model 覆盖层）。
- 支持“批量任务共享参数”：同一批次任务可绑定统一运行参数快照（见 3.3.1）。

> 注意：参数来源优先级必须明确：**前端/用户配置 > .env > 硬件推荐 > models.yaml 默认值**。

### 1.2 数据模型（新增）
新增文件：`backend/app/models/model_runtime_models.py`

```python
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal

DeviceType = Literal["auto", "cuda", "cpu"]

@dataclass
class ModelRuntimeOverride:
    device: Optional[DeviceType] = None
    compute_type: Optional[str] = None
    cpu_threads: Optional[int] = None
    keep_resident: Optional[bool] = None
    evict_priority: Optional[int] = None

@dataclass
class GlobalRuntimeConfig:
    device_preference: DeviceType = "auto"
    allow_download: bool = False
    max_vram_mb: Optional[int] = None
    reserved_vram_mb: Optional[int] = None
    max_models: Optional[int] = None
    cpu_threads: Optional[int] = None
    cpu_affinity_strategy: Optional[str] = None
    onnx_intra_threads: Optional[int] = None
    onnx_inter_threads: Optional[int] = None

@dataclass
class ModelRuntimeConfig:
    global_config: GlobalRuntimeConfig = field(default_factory=GlobalRuntimeConfig)
    per_model: Dict[str, ModelRuntimeOverride] = field(default_factory=dict)
```

### 1.3 运行参数服务（新增）
新增文件：`backend/app/services/model_runtime_config_service.py`

- 负责读取/写入 `model_runtime_config.json`
- 合并 `.env` 覆盖
- 输出 `ModelRuntimeConfig` 供 ModelManagerV2 使用

### 1.4 API 设计（新增）
新增路由文件：`backend/app/api/routes/model_runtime_routes.py`

```
GET  /api/models/runtime            # 获取全量运行参数（global + per_model）
GET  /api/models/runtime/{model_id} # 获取单模型运行参数
PUT  /api/models/runtime            # 更新全局参数（部分更新）
PUT  /api/models/runtime/{model_id} # 更新单模型参数（部分更新）
POST /api/models/runtime/apply      # 触发参数应用（必要时卸载/重载）
GET  /api/models/params             # 返回各模型可调参数与范围（来自 ModelSpec）
```

**返回字段应包含**：
- 当前生效值（合并后的最终值）
- 原始来源（env/user/hardware/default）
- 可选范围（由 ModelSpec 与硬件能力决定）
- 单模型独立配置快照（per_model 覆盖）

---

## 2. 硬件检测接入（解耦但松耦合）

### 2.1 现有硬件相关模块
- `backend/app/services/hardware_service.py`：硬件检测与优化建议  
- `backend/app/services/cpu_affinity_service.py`：CPU 亲和性  
- `backend/app/utils/cpu_optimizer.py`：ONNX 线程策略  
> 以上模块需要**抽象成“硬件能力提供者”**，由 ModelManagerV2 调用，但保持解耦。

### 2.2 新增“硬件能力提供者”
新增文件：`backend/app/services/hardware_profile_service.py`

```python
class HardwareProfileProvider:
    def get_hardware_info(self) -> HardwareInfo: ...
    def get_optimization_config(self, hw: HardwareInfo) -> OptimizationConfig: ...
    def get_onnx_thread_budget(self, hw: HardwareInfo) -> int: ...
```

### 2.3 ModelManagerV2 接入点
修改文件：`backend/app/services/model_manager_v2.py`

- 初始化时获取 `HardwareInfo`  
- 计算 `ResourceBudget` 默认值  
- 生成模型默认运行参数（device/compute_type/threads）  
- `OnnxLoader` 线程参数统一由 ModelManagerV2 传入 `LoadPlan.runtime`

### 2.4 CPU 亲和性与 cpu_optimizer 归一化
- `CPUAffinityManager` 与 `ONNXThreadOptimizer` 迁入 `HardwareProfileService`  
- ModelManagerV2 仅调用接口，不直接依赖实现  
- CPU 亲和性应用时机：**任务开始 / 高负载阶段**，而非模型加载阶段  
- ONNX 线程使用 `HardwareProfileService.get_onnx_thread_budget()` 计算后写入 `LoadPlan`

### 2.6 自动生成最佳运行参数（必须实现）
参考旧逻辑：
- `backend/app/services/hardware_service.py::CoreOptimizer.get_optimization_config`
- `backend/app/utils/cpu_optimizer.py::ONNXThreadOptimizer.calculate_optimal_threads`

实现目标：
- `HardwareProfileService` 输出 **最佳运行参数建议**（device/compute_type/cpu_threads/onnx_threads）。
- ModelManagerV2 在加载前自动计算并应用建议值（可被前端/.env 覆盖）。
- 保留“来源标记”，向前端返回参数来源（auto vs user）。

### 2.5 旧代码归档清单（实施后移动到 archive）
- `backend/app/services/hardware_service.py`
- `backend/app/services/cpu_affinity_service.py`
- `backend/app/utils/cpu_optimizer.py`
- `backend/app/core/resource_manager.py`

归档路径建议：`archive/hardware_legacy/`、`archive/resource_manager_legacy/`

---

## 3. 智能显存管理与动态预算（已实现）

### 3.1 现状（更新）
- ModelManagerV2 已接入策略引擎与动态预算，不再仅依赖简单 LRU。
- 旧 `ResourceManager` 仍未接入 V2（后续阶段统一归档）。

### 3.2 显存策略引擎（已接入）
新增文件：`backend/app/services/model_residency_policy.py`

**驱逐评分（示例）**：
```
score = priority_weight * keep_resident
      + freq_weight * normalized_use_count
      + recency_weight * recency_score
      - size_weight * normalized_vram
```

**策略要点**：
- **队列驱动**：由 JobQueueService 提供 `QueueSnapshot`，决定是否强制卸载大模型。  
- **体积驱动**：模型显存占用低且频繁使用，可保持常驻。  
- **频率驱动**：近 N 次任务使用频率高的模型优先保留。  

### 3.3 常驻/可卸载策略建议
新增文件：`backend/app/config/model_residency.yaml`

建议默认策略：
- 常驻：`sensevoice-small`、`silero-vad`
- 任务期常驻：`brouhaha-quality`（若频谱分诊开启）
- 按需加载：`whisper-medium`、`demucs-htdemucs`、`punct-*`、`langid-voxlingua`
- 任务结束即卸载：`demucs-htdemucs`（显存大、长时间空闲）

### 3.3.1 批量任务参数复用与 Whisper 暂时常驻
场景：批量加入队列的任务通常使用**相同运行参数**。  
策略要求：
- 若任务配置开启 Whisper，且显存允许且不影响 Demucs 等流程，则 Whisper 可在**批处理窗口**内常驻。  
- 常驻条件示例：
  - 队列中近 N 个任务 `use_whisper=True`
  - 当前显存余量 ≥ `whisper-medium` + `demucs-htdemucs` 预算
  - `demucs` 处于 idle 或未加载状态
- 退出条件：
  - 队列空闲时间 > T
  - 任务配置切换为 `sensevoice_only`
  - 显存压力触发（优先卸载 Whisper）

该策略需与 `JobQueueService` 的队列快照联动，建议在 `ModelResidencyPolicy` 中实现。

### 3.4 动态预算调整（已接入）
修改文件：`backend/app/services/model_manager_v2.py`

- 加载前读取 GPU 空闲显存（`torch.cuda.mem_get_info`），按比例计算动态预算  
- 结合队列空闲状态调整 `max_models`（空闲期自动下调常驻模型数）  
- 当预算不足且无可驱逐模型时，加载直接拒绝并抛出错误

### 3.5 强制常驻列表（已接入）
涉及文件：  
- `backend/app/services/model_runtime_config_service.py`（`resident_models` 持久化）  
- `backend/app/api/routes/model_runtime_routes.py`（`/api/models/resident`）  
- `backend/app/services/model_manager_v2.py`（强制常驻列表应用与驱逐保护）

---

## 4. 移除预热逻辑（完全删除）

### 4.1 需修改位置
- `backend/app/core/asr/model_spec.py`：删除 `warmup` 字段  
- `backend/app/core/asr/loader_base.py`：删除 `warmup()` 抽象  
- `backend/app/services/model_manager_v2.py`：移除 `loader.warmup` 调用  
- `backend/app/config/models.yaml`：删除 `warmup:` 配置项  
- 相关测试：移除 warmup 相关断言

### 4.2 影响说明
- 预热时间减少，启动更快  
- 推理首帧延迟略升，但整体收益更大  
- 所有模型加载逻辑更简化、可预测

---

## 5. SSE 模型下载进度推送

### 5.1 SSE 事件定义
新增文件：`backend/app/services/model_download_event_bus.py`

事件统一在 `models` 频道推送：

```
model.download.start
model.download.progress
model.download.complete
model.download.error
model.download.cache_hit
```

### 5.2 Downloader 改造
修改文件：`backend/app/services/model_downloader.py`

- 在 `_download_from_hf` 中注入进度回调  
- 进度来源：
  - `huggingface_hub.snapshot_download` + `tqdm_class` 自定义  
  - 或 `hf_hub_download` 的 `progress_callback`

**伪代码**：
```python
bus.emit("model.download.start", {...})
snapshot_download(..., tqdm_class=ModelDownloadTqdm)
bus.emit("model.download.complete", {...})
```

### 5.3 API 接口
修改文件：`backend/app/api/routes/model_routes.py`

```
GET /api/models/events    # SSE 订阅 models 频道
```

`initial_state` 返回当前下载中模型列表与进度快照。

---

## 6. 实施步骤与测试

### 6.1 实施步骤（推荐顺序）
1) 新增 `model_runtime_models.py` / `model_runtime_config_service.py`  
2) 新增 `model_runtime_routes.py` 并挂载到 FastAPI  
3) 新增 `hardware_profile_service.py` 并接入 ModelManagerV2  
4) 新增 `model_residency_policy.py` 与 `model_residency.yaml`  
5) 删除 warmup 字段与调用  
6) 接入 SSE 下载进度  
7) 归档旧硬件/资源管理模块

### 6.2 测试清单
- 单元：
  - RuntimeConfig 合并优先级  
  - ResidencyPolicy 评分/驱逐  
  - HardwareProfileProvider 输出稳定性  
  - SSE 事件格式校验
- 集成：
  - 双流模式加载 SenseVoice/Whisper 路径  
  - 队列变化触发模型卸载策略  
  - 模型下载触发 SSE 推送  

---

## 7. 归档策略（执行后必做）

归档路径建议：`archive/model_manager_legacy/` 与 `archive/hardware_legacy/`

待归档文件：
- `backend/app/core/resource_manager.py`
- `backend/app/services/hardware_service.py`
- `backend/app/services/cpu_affinity_service.py`
- `backend/app/utils/cpu_optimizer.py`

---

## 8. 与版本规划的对齐
- 与 `docs/v3.2.0+/version-roadmap.md` 中 ModelManager V2 要求一致  
- 本文为 v3.2.0 阶段补齐项，完成后可进入 v3.2.1 LangID 基础设施落地
