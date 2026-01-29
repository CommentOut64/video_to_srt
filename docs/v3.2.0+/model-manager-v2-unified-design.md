# 统一模型管理系统（ModelManager V2）一次性交付方案（完全替换旧逻辑）

> 目标：一次性交付完整的 ModelManager V2，旧模型管理代码全部归档停用；直接支持 v3.2.2 的标点模型扩展，无需分阶段灰度。仅保留与模型管理相关的设计和落地步骤。

## 0. 现状与必须淘汰的散点
- `backend/app/services/model_manager_service.py`：Whisper 下载/校验单一逻辑，需归档，不再调用。  
- `backend/app/services/model_preload_manager.py`：自带缓存/预热，与 V2 重叠，迁移后删除。  
- `backend/app/services/whisper_service.py`、`sensevoice_onnx_service.py`、`brouhaha_service.py`：包含下载/加载分散实现，迁移后仅保留推理前后处理，加载交由 V2。  
- 其他散点：Silero VAD 直接文件引用、Demucs 直接加载 checkpoint、`archive/legacy/resource_manager/resource_manager.py` 已归档（旧资源管理未与加载绑定）。  
- 要求：上述旧实现统一移动至 `archive/model_manager_legacy/`（或同级 archive 目录），主代码树不得再引用。

## 1. 一次性交付的目标能力
- 统一注册表 + 类型适配：以 `ModelSpec` 描述所有模型（ASR/标点/VAD/分离/质量/LangID）。  
- 下载/校验流水线：多源镜像、断点续传、哈希/文件列表校验，随包模型仅校验不重复下载。  
- 生命周期 + 资源治理：`load/unload/reload` 统一入口，显存/CPU 预算驱动 compute_type 降级或 LRU 驱逐。  
- 配置与特性暴露：语言/标点/时间戳能力、默认设备与可选 compute_type、输入输出维度统一查询。  
- 观测性：下载/加载耗时、使用频率、当前资源占用，Prometheus + 管理 API。  
- 可扩展：新增模型 = 注册表配置 + 对应 Loader 支持，无需改核心代码。  
- 直接支持 v3.2.2：标点模型按语言注册，Manager 提供按语言选择和缺失 fallback 建议。

## 2. 架构设计（模型管理专用）

### 2.1 ModelSpec 与注册表
- 新增 `backend/app/core/asr/model_spec.py`：`ModelSpec/ModelSource`（框架 `onnx/torch/ctranslate2/external/wfst`，kind 覆盖 asr/vad/separation/quality/langid/punct）。  
- 新增注册表 `backend/app/config/models.yaml`（可扩展 `models.d/*.yaml`）：  
  - `sensevoice-small`（ONNX）、`whisper-medium`（ctranslate2）、`brouhaha-quality`（torch，env.pip 锁版本）、`langid-voxlingua`、`silero-vad`、`demucs-htdemucs`。  
- 标点模型：`punct-ct-transformer-zh`, `punct-edge-punct-en`, `punct-pcs-47lang`（含 `punct-char-bert-ja` 备用），声明语言、框架、资源估算与 fallback 优先级。  

### 2.2 Loader 抽象
- `backend/app/core/asr/loader_base.py`：`ModelLoader.load/unload` 抽象。  
- 实现文件：`onnx_loader.py`（ort+线程策略）、`torch_loader.py`（torch+autocast+env 校验）、`ct2_loader.py`（faster-whisper）、`external_loader.py`（外部进程/服务句柄）、`wfst_loader.py`（WeTextProcessing）。  
- Loader 需读取 `spec.env` 设定环境/校验依赖，缺失时返回可安装列表。  

### 2.3 ModelManager V2 主体
- 新建 `backend/app/services/model_manager_v2.py`：  
  - `ModelRegistry`：加载/合并 `models.yaml`，按 kind/framework/device 过滤。  
  - `Downloader`：多镜像（HF/Mirror/ModelScope/本地包）+ 断点续传 + `ModelValidator` 校验。  
  - `ResourcePool`：基于 `ResourceBudget`，加载前评估显存/CPU，支持 compute_type 降级与 LRU 驱逐。  
  - `Lifecycle`：`acquire/release/load/unload/reload`；缓存存放 LoadedModel 句柄。  
  - `Metrics`：Prometheus 指标（加载/下载耗时、命中率、驱逐、显存/内存占用），统一日志。  
  - `ConfigBridge`：暴露模型特性（语言/标点/时间戳/默认参数范围），供 API/前端查询。  
  - 单例入口 `get_model_manager_v2()`，不保留旧开关，默认启用。  

### 2.4 生命周期调用示例
```python
# backend/app/services/model_manager_v2.py 片段
class ModelManagerV2:
    def acquire(self, model_id: str, device="auto", compute_type="auto"):
        spec = self.registry.get(model_id)
        plan = self._plan_device(spec, device, compute_type)   # 结合资源预算/设备可用性
        cached = self.cache.get(plan.key)
        if cached:
            return cached
        path = self.downloader.ensure_local(spec)
        loader = self._select_loader(spec.framework)
        handle = loader.load(spec, plan.with_path(path))
        self.cache.put(plan.key, handle)
        self.metrics.record_load(spec, handle, plan)
        return handle
```

### 2.5 观测与控制
- Prometheus：`/metrics/model_manager` 暴露加载/下载/热身耗时、缓存命中、驱逐、显存占用。  
- 管理 API：`GET /api/models/state`（当前加载/设备/占用）、`POST /api/models/reload`、`POST /api/models/evict`。  
- 日志：统一记录镜像/本地来源、降级策略、耗时、错误原因。  

## 3. 一次性落地步骤（无灰度）
1) 建立注册表与类型系统：新增 `model_spec.py`、`models.yaml`，补齐所有模型（ASR/标点/VAD/分离/质量/LangID）。  
2) 实现 Loader 全套（onnx/torch/ct2/external），支持 env 校验。  
3) 实现 `model_manager_v2.py`（Registry/Downloader/ResourcePool/Lifecycle/Metrics/ConfigBridge），默认替换旧 Manager。  
4) 适配现有服务：`whisper_service.py`、`sensevoice_onnx_service.py`、`brouhaha_service.py`、`demucs_service.py`、VAD/LangID/标点调用统一从 `ModelManagerV2.acquire()` 获取；移除内部下载/加载逻辑。  
5) 清理与归档：将旧文件整体迁移至 `archive/model_manager_legacy/`，主代码树删除引用；删除旧开关 `USE_MODEL_MANAGER_V2`，确保唯一入口。  
6) 标点直接扩展（对齐 v3.2.2 要求）：在 `models.yaml` 注册语言特定标点模型；Manager 支持按语言选择和缺失 fallback；调用方通过 Manager 获取模型句柄，无需额外阶段。  
7) 观测与验证：开启 Prometheus 指标与管理 API，替换旧 SSE 下载进度为 Manager 统一事件；验证显存预算驱逐/降级路径。  

## 4. 清理范围（必须执行）
- 归档并停止引用：`backend/app/services/model_manager_service.py`、`model_manager_service.py.backup`、`model_preload_manager.py`。  
- 剥离下载/加载：`whisper_service.py`、`sensevoice_onnx_service.py`、`brouhaha_service.py`、`demucs_service.py` 内仅保留推理/前后处理。  
- 配置收敛：移除硬编码模型表、镜像开关，统一读取 `models.yaml`。  
- 资源治理：用 `ResourcePool` 替换 `archive/legacy/resource_manager/resource_manager.py` 的零散估算，实现强绑定加载流程。  

## 5. 测试与验收（聚焦模型管理）
- 单元：ModelSpec 解析；Downloader 多镜像/断点/校验；ResourcePool 驱逐与 compute_type 降级；各 Loader 的 load/unload。  
- 集成：SenseVoice/Whisper/Brouhaha/Demucs/VAD/LangID/标点 在 V2 下的加载/卸载/缓存命中/显存预算触发。  
- 回归：下载/加载耗时对比旧版（±10%）、缓存命中率、驱逐正确性；随包模型仅校验不重复下载。  
- 观测：Prometheus 指标暴露正确，降级/驱逐日志完整；管理 API 返回模型状态与资源占用。  

---
一次性交付后，所有模型实例必须通过 `ModelManagerV2.acquire()` 取得，旧代码全部归档停用；标点模型已在注册表层准备好扩展，无需额外阶段。 
