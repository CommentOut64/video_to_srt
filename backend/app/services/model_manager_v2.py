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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, generate_latest

from app.core.config import config
from app.core.asr.loader_base import LoadPlan, ModelLoader
from app.core.asr.model_spec import ModelSpec
from app.core.asr.registry import ModelRegistry
from app.services.model_validator import ModelValidator
from app.services.model_downloader import ModelDownloader

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

    def put(self, key: str, value: Any) -> Optional[tuple[str, Any]]:
        """返回被驱逐的 (key, handle)（如果有）。"""
        evicted = None
        with self._lock:
            if key in self._data:
                self._order.pop(key, None)
            self._data[key] = value
            self._order[key] = None
            if len(self._data) > self.capacity:
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
    ):
        base = Path(config.BASE_DIR)
        # V3.2.0+dev.20260116.02: 传递 base_dir 给 Registry 和 Downloader
        self.registry = registry or ModelRegistry(
            config_path=base / "backend" / "app" / "config" / "models.yaml",
            extra_dir=base / "backend" / "app" / "config" / "models.d",
            base_dir=base,  # 传递项目根目录作为基准路径
        )
        self.loaders = loaders or self._default_loaders()
        self.budget = budget or ResourceBudget()
        self.cache = _LRUCache(self.budget.max_models)
        self._usage_vram: Dict[str, int] = {}
        # 为防止意外下载，默认仅使用本地文件
        self.downloader = downloader or ModelDownloader(
            allow_download=False,
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
        logger.info(
            "ModelManagerV2 初始化完成: registry=%s budget(max_models=%d,max_vram=%d,reserve=%d)",
            self.registry.config_path,
            self.budget.max_models,
            self.budget.max_vram_mb,
            self.budget.reserved_vram_mb,
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

    def _evict_for_budget(self, request_vram: int) -> None:
        """当预计显存不足时驱逐 LRU，直到满足预算。"""
        if request_vram <= 0:
            return
        while True:
            total = sum(self._usage_vram.values())
            if total + request_vram <= self.budget.available_vram_mb:
                return
            evicted = self.cache.evict_oldest()
            if not evicted:
                raise RuntimeError("无法驱逐缓存以满足显存预算")
            evicted_key, handle = evicted
            self._usage_vram.pop(evicted_key, None)
            if handle and hasattr(handle, "unload"):
                try:
                    handle.unload()
                except Exception as exc:  # pragma: no cover
                    logger.warning("缓存驱逐卸载失败: %s", exc)

    def acquire(self, model_id: str, device: str = "auto", compute_type: str = "auto") -> Any:
        spec = self.registry.get(model_id)
        plan = LoadPlan(
            device=device if device != "auto" else spec.default_device,
            compute_type=compute_type if compute_type != "auto" else spec.compute_type,
            local_path=self.downloader.ensure_local(spec),
        )
        logger.info(
            "ModelManagerV2 接到加载请求: %s framework=%s device=%s compute_type=%s path=%s",
            spec.id,
            spec.framework,
            plan.device,
            plan.compute_type,
            plan.local_path,
        )
        request_vram = self._estimate_vram(spec)
        self._evict_for_budget(request_vram)

        cached = self.cache.get(plan.key)
        if cached:
            logger.info(
                "ModelManagerV2 命中缓存: %s device=%s compute_type=%s path=%s",
                spec.id,
                plan.device,
                plan.compute_type,
                plan.local_path,
            )
            return cached

        loader = self._select_loader(spec.framework)
        start = time.time()
        handle = loader.load(spec, plan)
        try:
            loader.warmup(handle, spec)
        except Exception as exc:  # pragma: no cover
            logger.warning("模型预热失败（忽略继续）: %s", exc)

        evicted = self.cache.put(plan.key, handle)
        if evicted:
            evicted_key, evicted_handle = evicted
            self._usage_vram.pop(evicted_key, None)
            if evicted_handle and hasattr(evicted_handle, "unload"):
                try:
                    evicted_handle.unload()
                except Exception as exc:  # pragma: no cover
                    logger.warning("缓存自动卸载失败: %s", exc)
            self._metric_cache_evicts.labels(reason="capacity").inc()

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
