"""
统一模型管理系统（ModelManager V2）核心实现骨架。
一次性交付版本的基础设施：注册表、下载占位、资源预算、生命周期管理。
后续阶段可在此基础上扩展真实下载、显存策略与监控。
V3.2.0+dev.20260114.01
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, List

from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, generate_latest

from app.core.config import config
from app.core.asr.loader_base import LoadPlan, ModelLoader
from app.core.asr.model_spec import ModelSpec
from app.core.asr.registry import ModelRegistry
from app.services.model_validator import ModelValidator
from app.services.model_downloader import ModelDownloader
from app.services.model_runtime_config_service import get_model_runtime_config_service
from app.services.hardware_profile_service import get_hardware_profile_provider
from app.services.model_residency_policy import (
    ModelResidencyPolicy,
    ModelResidencyEntry,
    QueueSnapshot,
)

try:
    from app.core.asr.loaders.onnx_loader import OnnxLoader
    from app.core.asr.loaders.torch_loader import TorchLoader
    from app.core.asr.loaders.ct2_loader import CT2Loader
    from app.core.asr.loaders.external_loader import ExternalLoader
    from app.core.asr.loaders.demucs_loader import DemucsLoader
except Exception as exc:  # pragma: no cover
    # 部分依赖可能缺失，延迟注册
    logger = logging.getLogger(__name__)
    logger.debug("加载默认 Loader 失败: %s", exc)

logger = logging.getLogger(__name__)


@dataclass
class ResourceBudget:
    """资源预算配置，用于预估显存/线程。"""

    max_models: int = 3
    max_vram_mb: int = 8000
    reserved_vram_mb: int = 500

    @property
    def available_vram_mb(self) -> int:
        return max(self.max_vram_mb - self.reserved_vram_mb, 0)


class _LRUCache:
    """简单线程安全 LRU，用于模型句柄缓存，可主动驱逐。"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._data: Dict[str, Any] = {}
        self._order: Dict[str, None] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._data:
                return None
            self._order.pop(key, None)
            self._order[key] = None
            return self._data[key]

    def put(self, key: str, value: Any, auto_evict: bool = True) -> Optional[tuple[str, Any]]:
        """返回被驱逐的 (key, handle)（如果有）。"""
        evicted = None
        with self._lock:
            if key in self._data:
                self._order.pop(key, None)
            self._data[key] = value
            self._order[key] = None
            if auto_evict and len(self._data) > self.capacity:
                oldest_key = next(iter(self._order))
                self._order.pop(oldest_key, None)
                handle = self._data.pop(oldest_key, None)
                evicted = (oldest_key, handle)
        return evicted

    def evict_oldest(self) -> Optional[tuple[str, Any]]:
        """手动驱逐最旧项。"""
        with self._lock:
            if not self._order:
                return None
            oldest_key = next(iter(self._order))
            self._order.pop(oldest_key, None)
            handle = self._data.pop(oldest_key, None)
            return (oldest_key, handle)

    def pop(self, key: str) -> Optional[Any]:
        """按 key 移除缓存。"""
        with self._lock:
            self._order.pop(key, None)
            return self._data.pop(key, None)

    def touch(self, key: str) -> None:
        """刷新 LRU 顺序。"""
        with self._lock:
            if key not in self._data:
                return
            self._order.pop(key, None)
            self._order[key] = None

    def keys(self) -> List[str]:
        """返回缓存 key 列表。"""
        with self._lock:
            return list(self._data.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class ModelManagerV2:
    """
    ModelManager V2 核心：注册表 + 下载校验占位 + 生命周期 + 缓存。
    真实下载/显存策略可在后续阶段扩展。
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        loaders: Optional[Dict[str, ModelLoader]] = None,
        budget: Optional[ResourceBudget] = None,
        downloader: Optional[ModelDownloader] = None,
        hardware_profile_provider=None,
    ):
        base = Path(config.BASE_DIR)
        self.hardware_profile_provider = hardware_profile_provider or get_hardware_profile_provider()
        self._runtime_service = get_model_runtime_config_service()
        global_effective = self._runtime_service.get_effective_global()
        effective = global_effective.get("effective", {})
        # V3.2.0+dev.20260116.02: 传递 base_dir 给 Registry 和 Downloader
        self.registry = registry or ModelRegistry(
            config_path=base / "backend" / "app" / "config" / "models.yaml",
            extra_dir=base / "backend" / "app" / "config" / "models.d",
            base_dir=base,  # 传递项目根目录作为基准路径
        )
        self.loaders = loaders or self._default_loaders()
        if budget is None:
            self.budget = ResourceBudget(
                max_models=int(effective.get("max_models") or 3),
                max_vram_mb=int(effective.get("max_vram_mb") or 8000),
                reserved_vram_mb=int(effective.get("reserved_vram_mb") or 500),
            )
        else:
            self.budget = budget
        self.cache = _LRUCache(self.budget.max_models)
        self._cache_lock = threading.RLock()
        self._usage_stats: Dict[str, ModelResidencyEntry] = {}
        self._residency_policy = ModelResidencyPolicy.from_yaml(
            base / "backend" / "app" / "config" / "model_residency.yaml"
        )
        self._force_resident_models: set[str] = set()
        self._last_queue_activity_at = time.time()
        self._usage_vram: Dict[str, int] = {}
        hardware_info = self.hardware_profile_provider.get_hardware_info()
        self._hardware_info = hardware_info
        # 为防止意外下载，默认仅使用本地文件
        allow_download = bool(effective.get("allow_download")) if downloader is None else downloader.allow_download
        self.downloader = downloader or ModelDownloader(
            allow_download=allow_download,
            local_files_only=True,
            base_dir=base,  # V3.2.0+dev.20260116.02: 传递基准路径
        )
        # Metrics
        self._registry = CollectorRegistry()
        self._metric_load_time = Histogram(
            "model_manager_load_seconds",
            "模型加载耗时",
            ["model_id", "framework"],
            registry=self._registry,
        )
        self._metric_cache_evicts = Counter(
            "model_manager_cache_evict_total",
            "模型缓存驱逐次数",
            ["reason"],
            registry=self._registry,
        )
        self._metric_vram = Gauge(
            "model_manager_vram_usage_mb",
            "模型缓存显存估算",
            registry=self._registry,
        )
        self.refresh_runtime_config()
        logger.info(
            "ModelManagerV2 初始化完成: registry=%s budget(max_models=%d,max_vram=%d,reserve=%d)",
            self.registry.config_path,
            self.budget.max_models,
            self.budget.max_vram_mb,
            self.budget.reserved_vram_mb,
        )
        logger.info(
            "ModelManagerV2 硬件概况: gpu=%s cpu_cores=%s cpu_threads=%s memory=%sMB",
            hardware_info.gpu_count,
            hardware_info.cpu_cores,
            hardware_info.cpu_threads,
            hardware_info.memory_total_mb,
        )

    def _default_loaders(self) -> Dict[str, ModelLoader]:
        """按可用依赖注册默认 Loader。"""
        result: Dict[str, ModelLoader] = {}
        for name, cls in [
            ("onnx", "OnnxLoader"),
            ("torch", "TorchLoader"),
            ("ctranslate2", "CT2Loader"),
            ("external", "ExternalLoader"),
            ("demucs", "DemucsLoader"),
        ]:
            try:
                loader_cls = globals()[cls]
                result[name] = loader_cls()
            except Exception:  # pragma: no cover
                continue
        return result

    def _select_loader(self, framework: str) -> ModelLoader:
        if framework not in self.loaders:
            raise KeyError(f"未注册框架对应的 Loader: {framework}")
        return self.loaders[framework]

    def _estimate_vram(self, spec: ModelSpec) -> int:
        """估算显存占用，缺省返回 0 表示未知。"""
        return int(spec.resources.get("vram_mb", 0))

    def refresh_runtime_config(self) -> Dict[str, Any]:
        """刷新运行参数与强制常驻列表。"""
        global_effective = self._runtime_service.get_effective_global()
        effective = global_effective.get("effective", {})

        max_models = int(effective.get("max_models") or self.budget.max_models)
        self.budget.max_models = max(1, max_models)
        self.cache.capacity = self.budget.max_models

        max_vram = effective.get("max_vram_mb")
        if max_vram is not None:
            self.budget.max_vram_mb = int(max_vram)
        reserved_vram = effective.get("reserved_vram_mb")
        if reserved_vram is not None:
            self.budget.reserved_vram_mb = int(reserved_vram)
        allow_download = effective.get("allow_download")
        if allow_download is not None:
            self.downloader.allow_download = bool(allow_download)

        resident = self._runtime_service.get_resident_models().get("models", [])
        self.set_force_resident_models(resident)
        self._refresh_residency_overrides()

        return global_effective

    def get_force_resident_models(self) -> List[str]:
        """返回强制常驻模型列表。"""
        return sorted(self._force_resident_models)

    def set_force_resident_models(self, model_ids: List[str]) -> None:
        """设置强制常驻模型列表。"""
        self._force_resident_models = {model_id for model_id in model_ids if model_id}
        for entry in self._usage_stats.values():
            entry.is_force_resident = entry.model_id in self._force_resident_models
        self._apply_residency_policy(request_vram_mb=0, incoming_models=0)

    def _refresh_residency_overrides(self) -> None:
        """刷新已加载模型的常驻/驱逐配置。"""
        for entry in self._usage_stats.values():
            try:
                spec = self.registry.get(entry.model_id)
            except KeyError:
                continue
            effective = self._runtime_service.get_effective_model(spec).get("effective", {})
            entry.is_keep_resident = bool(effective.get("keep_resident"))
            entry.evict_priority = int(effective.get("evict_priority") or 0)
            entry.is_force_resident = entry.model_id in self._force_resident_models

    def _get_queue_snapshot(self) -> Optional[QueueSnapshot]:
        """获取任务队列快照（失败时返回 None）。"""
        try:
            from app.services.job_queue_service import get_queue_service
            queue_service = get_queue_service()
        except Exception:
            return None

        try:
            with queue_service.lock:
                queued = list(queue_service.queue)
                running_job_id = queue_service.running_job_id
        except Exception:
            return None

        now = time.time()
        is_idle = not running_job_id and not queued
        if not is_idle:
            self._last_queue_activity_at = now
        idle_seconds = now - self._last_queue_activity_at if is_idle else 0.0

        return QueueSnapshot(
            running_job_id=running_job_id,
            queued_job_ids=queued,
            pending_jobs=len(queued),
            is_idle=is_idle,
            idle_seconds=idle_seconds,
        )

    def _resolve_available_vram_mb(self) -> int:
        """计算动态可用显存预算。"""
        base_budget = self.budget.available_vram_mb
        try:
            import torch

            if torch.cuda.is_available():
                free_bytes, _ = torch.cuda.mem_get_info()
                free_mb = int(free_bytes / (1024 * 1024))
                ratio = self._residency_policy.config.dynamic_vram_ratio
                safety_mb = self._residency_policy.config.dynamic_vram_safety_mb
                dynamic_limit = int(free_mb * ratio) - safety_mb
                return max(0, min(base_budget, dynamic_limit))
        except Exception as exc:
            logger.debug("动态显存预算获取失败: %s", exc)

        return base_budget

    def _build_residency_entries(self) -> List[ModelResidencyEntry]:
        return list(self._usage_stats.values())

    def _apply_residency_policy(
        self,
        request_vram_mb: int,
        incoming_models: int = 1,
        apply_vram_budget: bool = True,
    ) -> None:
        """应用智能显存策略，必要时驱逐缓存。"""
        entries = self._build_residency_entries()
        if not entries and request_vram_mb <= 0:
            return

        snapshot = self._get_queue_snapshot()
        max_models = self._residency_policy.resolve_max_models(self.budget.max_models, snapshot)
        if apply_vram_budget:
            available_vram = self._resolve_available_vram_mb()
            current_vram = sum(entry.vram_mb for entry in entries)
        else:
            available_vram = 1_000_000_000
            current_vram = 0
            request_vram_mb = 0

        plan = self._residency_policy.select_evictions(
            entries=entries,
            required_vram_mb=request_vram_mb,
            max_models=max_models,
            available_vram_mb=available_vram,
            current_vram_mb=current_vram,
            snapshot=snapshot,
            incoming_models=incoming_models,
        )

        if plan.keys:
            self._evict_keys(plan.keys, plan.reason)

        freed_vram = sum(
            entry.vram_mb for entry in entries if entry.plan_key in plan.keys
        )
        projected_vram = current_vram - freed_vram + max(request_vram_mb, 0)
        projected_models = len(entries) - len(plan.keys) + incoming_models

        if incoming_models > 0:
            if projected_models > max_models:
                raise RuntimeError("模型缓存容量不足，无法加载模型")
            if apply_vram_budget and projected_vram > available_vram:
                raise RuntimeError("显存预算不足，无法加载模型")

    def _resolve_effective_device(self, spec: ModelSpec, device: str) -> str:
        """解析模型最终设备，用于预算判断。"""
        device_lower = (device or "auto").lower()
        if device_lower in {"cpu", "cuda"}:
            return device_lower
        if device_lower != "auto":
            return device_lower

        framework = spec.framework
        if framework == "onnx":
            return "cpu"

        if self._hardware_info and getattr(self._hardware_info, "cuda_available", False):
            return "cuda"
        return "cpu"

    def _evict_keys(self, keys: List[str], reason: str) -> None:
        """按 key 驱逐模型。"""
        for key in keys:
            handle = self.cache.pop(key)
            entry = self._usage_stats.pop(key, None)
            self._usage_vram.pop(key, None)
            if handle and hasattr(handle, "unload"):
                try:
                    handle.unload()
                except Exception as exc:  # pragma: no cover
                    logger.warning("缓存驱逐卸载失败: %s", exc)
            if entry:
                logger.info(
                    "模型驱逐: %s reason=%s vram=%sMB",
                    entry.model_id,
                    reason,
                    entry.vram_mb,
                )
            self._metric_cache_evicts.labels(reason=reason).inc()
        self._metric_vram.set(sum(self._usage_vram.values()))

    def _touch_usage(
        self,
        plan_key: str,
        spec: ModelSpec,
        model_runtime: Dict[str, Any],
        vram_mb: int,
    ) -> None:
        """更新模型使用统计。"""
        now = time.time()
        entry = self._usage_stats.get(plan_key)
        if entry:
            entry.last_used_at = now
            entry.use_count += 1
            entry.evict_priority = int(model_runtime.get("evict_priority") or 0)
            entry.is_keep_resident = bool(model_runtime.get("keep_resident"))
            entry.is_force_resident = spec.id in self._force_resident_models
            if vram_mb > 0:
                entry.vram_mb = vram_mb
            return

        self._usage_stats[plan_key] = ModelResidencyEntry(
            plan_key=plan_key,
            model_id=spec.id,
            vram_mb=vram_mb,
            loaded_at=now,
            last_used_at=now,
            use_count=1,
            evict_priority=int(model_runtime.get("evict_priority") or 0),
            is_keep_resident=bool(model_runtime.get("keep_resident")),
            is_force_resident=spec.id in self._force_resident_models,
        )

    def _register_usage(
        self,
        plan_key: str,
        spec: ModelSpec,
        model_runtime: Dict[str, Any],
        vram_mb: int,
    ) -> None:
        """注册新加载模型的统计信息。"""
        now = time.time()
        self._usage_stats[plan_key] = ModelResidencyEntry(
            plan_key=plan_key,
            model_id=spec.id,
            vram_mb=vram_mb,
            loaded_at=now,
            last_used_at=now,
            use_count=1,
            evict_priority=int(model_runtime.get("evict_priority") or 0),
            is_keep_resident=bool(model_runtime.get("keep_resident")),
            is_force_resident=spec.id in self._force_resident_models,
        )

    def acquire(self, model_id: str, device: str = "auto", compute_type: str = "auto") -> Any:
        spec = self.registry.get(model_id)
        model_runtime = self._runtime_service.get_effective_model(spec).get("effective", {})
        global_runtime = self._runtime_service.get_effective_global().get("effective", {})

        resolved_device = device if device != "auto" else model_runtime.get("device", spec.default_device)
        resolved_compute_type = compute_type if compute_type != "auto" else model_runtime.get(
            "compute_type",
            spec.compute_type,
        )
        cpu_threads = model_runtime.get("cpu_threads")

        runtime_payload: Dict[str, Any] = {}
        if spec.framework == "onnx":
            if cpu_threads:
                runtime_payload["cpu_threads"] = int(cpu_threads)
            if global_runtime.get("onnx_intra_threads") is not None:
                runtime_payload["onnx_intra_threads"] = int(global_runtime["onnx_intra_threads"])
            if global_runtime.get("onnx_inter_threads") is not None:
                runtime_payload["onnx_inter_threads"] = int(global_runtime["onnx_inter_threads"])

        plan = LoadPlan(
            device=resolved_device,
            compute_type=resolved_compute_type,
            local_path=self.downloader.ensure_local(spec),
            runtime=runtime_payload,
        )
        logger.info(
            "ModelManagerV2 接到加载请求: %s framework=%s device=%s compute_type=%s path=%s",
            spec.id,
            spec.framework,
            plan.device,
            plan.compute_type,
            plan.local_path,
        )
        effective_device = self._resolve_effective_device(spec, resolved_device)
        request_vram = self._estimate_vram(spec) if effective_device == "cuda" else 0
        logger.info(
            "ModelManagerV2 预算设备解析: %s resolved=%s effective=%s vram=%sMB",
            spec.id,
            resolved_device,
            effective_device,
            request_vram,
        )
        with self._cache_lock:
            self.refresh_runtime_config()
            cached = self.cache.get(plan.key)
            if cached:
                self._touch_usage(plan.key, spec, model_runtime, request_vram)
                logger.info(
                    "ModelManagerV2 命中缓存: %s device=%s compute_type=%s path=%s",
                    spec.id,
                    plan.device,
                    plan.compute_type,
                    plan.local_path,
                )
                return cached
            self._apply_residency_policy(
                request_vram_mb=request_vram,
                incoming_models=1,
                apply_vram_budget=(effective_device == "cuda"),
            )

        runtime_resources = dict(spec.resources)
        if cpu_threads:
            runtime_resources["cpu_threads"] = int(cpu_threads)
        runtime_spec = spec
        if runtime_resources != spec.resources:
            runtime_spec = replace(spec, resources=runtime_resources)

        loader = self._select_loader(spec.framework)
        start = time.time()
        handle = loader.load(runtime_spec, plan)

        with self._cache_lock:
            cached = self.cache.get(plan.key)
            if cached:
                try:
                    loader.unload(handle)
                except Exception as exc:  # pragma: no cover
                    logger.warning("重复加载句柄卸载失败: %s", exc)
                self._touch_usage(plan.key, spec, model_runtime, request_vram)
                return cached

            self.cache.put(plan.key, handle, auto_evict=False)
            self._register_usage(plan.key, spec, model_runtime, request_vram)

        if request_vram > 0:
            self._usage_vram[plan.key] = request_vram
            self._metric_vram.set(sum(self._usage_vram.values()))

        duration = time.time() - start
        logger.info(
            "ModelManagerV2 加载完成: %s framework=%s device=%s compute_type=%s path=%s 耗时=%.2fs",
            spec.id,
            spec.framework,
            plan.device,
            plan.compute_type,
            plan.local_path,
            duration,
        )
        self._metric_load_time.labels(model_id=spec.id, framework=spec.framework).observe(duration)
        return handle

    # ========== 辅助接口 ==========

    def model_status(self, model_id: str) -> Dict[str, Any]:
        """返回模型状态，不触发下载。"""
        spec = self.registry.get(model_id)
        local = self.downloader.find_local(spec)
        status = "missing"
        missing_files: list = []
        local_path = None

        if local:
            local_path = str(local)
            try:
                self.downloader._validate_files(spec, local)
                status = "ready"
            except FileNotFoundError as exc:
                status = "incomplete"
                # 从异常消息中提取缺失文件
                text = str(exc)
                if "缺少" in text:
                    missing_files = [text]
        return {
            "id": model_id,
            "kind": spec.kind,
            "framework": spec.framework,
            "status": status,
            "local_path": local_path,
            "missing": missing_files,
            "features": spec.features,
            "resources": spec.resources,
        }

    def list_status(self, kind: Optional[str] = None) -> Dict[str, Any]:
        specs = self.registry.list(kind=kind) if kind else self.registry.list()
        return {spec.id: self.model_status(spec.id) for spec in specs}

    def ensure_available(self, model_id: str) -> str:
        """确保模型可用，必要时下载。"""
        spec = self.registry.get(model_id)
        local_path = self.downloader.ensure_local(spec)
        logger.info(
            "ModelManagerV2 ensure_available 成功: %s framework=%s path=%s",
            spec.id,
            spec.framework,
            local_path,
        )
        return local_path

    def delete_model(self, model_id: str) -> bool:
        """删除本地模型目录（若存在）。"""
        import shutil

        spec = self.registry.get(model_id)
        local = self.downloader.find_local(spec)
        if not local:
            return False
        try:
            shutil.rmtree(local, ignore_errors=False)
            # 清理缓存记录
            self._usage_vram = {k: v for k, v in self._usage_vram.items() if local.as_posix() not in k}
            return True
        except Exception as exc:
            logger.error("删除模型失败 %s: %s", model_id, exc)
            return False

    def unload_all(self) -> None:
        """卸载缓存中的所有模型。"""
        while True:
            evicted = self.cache.evict_oldest()
            if not evicted:
                break
            _, handle = evicted
            if handle and hasattr(handle, "unload"):
                try:
                    handle.unload()
                except Exception as exc:  # pragma: no cover
                    logger.warning("卸载模型失败: %s", exc)
        self._usage_vram.clear()
        self._usage_stats.clear()
        self._metric_vram.set(0)

    # ========== 标点模型选择 ==========
    def select_punct_model(self, language: str) -> Optional[str]:
        """按语言选择标点模型，使用 features.languages 匹配，找不到则按 fallback。"""
        candidates = self.registry.list(kind="punct")
        # 直接匹配
        for spec in candidates:
            langs = spec.features.get("languages") if spec.features else []
            if language in langs:
                return spec.id
        # fallback 链
        for spec in candidates:
            if spec.fallback:
                for fb in spec.fallback:
                    try:
                        fb_spec = self.registry.get(fb)
                        langs = fb_spec.features.get("languages") if fb_spec.features else []
                        if language in langs:
                            return fb_spec.id
                    except KeyError:
                        continue
        return None

    # ========== Metrics ==========
    def metrics_text(self) -> str:
        """导出 Prometheus 文本格式。"""
        return generate_latest(self._registry).decode("utf-8")


_model_manager_v2: Optional[ModelManagerV2] = None
_model_manager_lock = threading.Lock()


def get_model_manager_v2() -> ModelManagerV2:
    """单例访问入口。"""
    global _model_manager_v2
    created = False
    if _model_manager_v2 is None:
        with _model_manager_lock:
            if _model_manager_v2 is None:
                _model_manager_v2 = ModelManagerV2()
                created = True
    # 如果注册表为空，说明可能路径配置不正确，强制重建以确保加载 models.yaml
    if _model_manager_v2 and not _model_manager_v2.registry.list():
        with _model_manager_lock:
            _model_manager_v2 = ModelManagerV2()
            created = True
    logger.info("get_model_manager_v2 调用: created=%s", created)
    return _model_manager_v2
