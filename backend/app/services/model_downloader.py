"""
模型下载与校验流水线。
默认优先使用本地路径；当允许下载且存在 repo_id 时调用 Hugging Face snapshot_download。
V3.2.4+dev.20260304.14: 移除“pretrained 必须随包内置”限制，缺失时可按策略自动修复。
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from app.core.asr.model_spec import ModelSpec
from app.services.model_validator import ModelValidator
from app.core.config import config
from app.services.model_download_event_bus import get_model_download_event_bus

logger = logging.getLogger(__name__)


class ModelDownloader:
    """统一的模型获取入口。"""

    def __init__(self, allow_download: bool = True, local_files_only: bool = False, base_dir: Optional[Path] = None):
        """
        Args:
            allow_download: 是否允许从远程仓库下载。
            local_files_only: 调用 HF 时仅使用本地缓存（测试环境可开启）。
            base_dir: 基准目录，用于解析相对路径（V3.2.0+dev.20260116.02）。
        """
        self.allow_download = allow_download
        self.local_files_only = local_files_only
        self.base_dir = base_dir  # V3.2.0+dev.20260116.02: 添加上下文支持

    def ensure_local(self, spec: ModelSpec) -> str:
        """
        确保模型在本地可用，必要时触发下载。

        规则：
        - 本地完整 -> 直接返回；
        - 本地不完整 -> 若可下载则自动修复，否则报错；
        - 本地缺失 -> 若可下载则下载，否则报错。

        Returns: 本地路径字符串。
        """
        event_bus = get_model_download_event_bus()
        local = self.find_local(spec)
        if local:
            try:
                self._validate_files(spec, local)
            except FileNotFoundError as exc:
                logger.warning("本地模型不完整，将尝试自动修复: model=%s error=%s", spec.id, exc)
                if not spec.source.repo_id:
                    event_bus.error(spec.id, str(exc), spec.source.repo_id)
                    raise
                local = None
            else:
                event_bus.cache_hit(spec.id, str(local), spec.source.repo_id)
                return str(local)

        if not spec.source.repo_id:
            event_bus.error(spec.id, "模型未提供本地路径或仓库", spec.source.repo_id)
            raise FileNotFoundError(f"模型未提供本地路径或仓库: {spec.id}")

        if not self.allow_download:
            event_bus.error(spec.id, "模型缺失且未允许下载", spec.source.repo_id)
            raise FileNotFoundError(f"模型缺失且未允许下载: {spec.id} repo={spec.source.repo_id}")

        path = self._download_from_hf(spec)
        self._validate_files(spec, path)
        event_bus.complete(spec.id, str(path), spec.source.repo_id)
        return str(path)

    def find_local(self, spec: ModelSpec) -> Optional[Path]:
        """尝试查找本地已有路径，不触发下载。"""
        local_path = self._resolve_local_path(spec)
        if local_path and local_path.exists():
            return local_path

        # install_mode=mirror/project 表示 HF 仅作为下载源，最终安装位仍是 local_path。
        # 此时不回退到 HF cache/snapshot，避免运行时看到与 pretrained 目录不同的结构。
        if local_path and spec.source.install_mode != "cache":
            return None

        if spec.source.repo_id:
            return self._resolve_cached_snapshot(spec)
        return None

    def _validate_files(self, spec: ModelSpec, base: Path) -> None:
        """按文件列表校验；Whisper 复用 ModelValidator。"""
        missing = []
        for fname in spec.source.files:
            if not (base / fname).exists():
                missing.append(fname)
        if missing:
            raise FileNotFoundError(f"模型缺失文件: {spec.id} 缺少 {missing}")

        # Whisper 专用校验（非 Whisper 模型忽略异常）
        try:
            ModelValidator.validate_whisper_model(base)
        except Exception:
            pass

    def _download_from_hf(self, spec: ModelSpec) -> Path:
        """使用 huggingface_hub snapshot_download 下载。"""
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover
            raise ImportError("缺少 huggingface_hub 依赖，无法下载模型") from exc

        event_bus = get_model_download_event_bus()
        logger.info("下载模型: %s", spec.id)
        event_bus.start(spec.id, spec.source.repo_id)

        allow_patterns = list(spec.source.allow_patterns or [])
        if not allow_patterns and spec.source.install_mode != "cache":
            allow_patterns = list(self._build_file_projection(spec).keys())

        download_kwargs = {
            "repo_id": spec.source.repo_id,
            "local_files_only": self.local_files_only,
            "max_workers": 8,
            "tqdm_class": _ModelDownloadTqdmFactory(event_bus, spec.id, spec.source.repo_id),
        }
        if allow_patterns:
            download_kwargs["allow_patterns"] = allow_patterns
        if spec.source.ignore_patterns:
            download_kwargs["ignore_patterns"] = list(spec.source.ignore_patterns)
        if spec.source.revision:
            download_kwargs["revision"] = spec.source.revision
        hf_token = self._resolve_hf_token()
        if hf_token:
            download_kwargs["token"] = hf_token

        local_target = self._resolve_local_path(spec)
        if spec.source.install_mode != "cache" and local_target is not None:
            return self._download_with_endpoint_fallback(
                spec,
                snapshot_download,
                download_kwargs,
                lambda endpoint: self._download_to_local_target(spec, local_target, snapshot_download, download_kwargs, endpoint),
            )

        cache_dir = Path(config.HF_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        download_kwargs["cache_dir"] = cache_dir.as_posix()
        return self._download_with_endpoint_fallback(
            spec,
            snapshot_download,
            download_kwargs,
            lambda endpoint: self._snapshot_download_with_compat(snapshot_download, download_kwargs, endpoint),
        )

    def _resolve_local_path(self, spec: ModelSpec) -> Optional[Path]:
        if not spec.source.local_path:
            return None
        path = Path(spec.source.local_path)
        if not path.is_absolute() and self.base_dir:
            path = (self.base_dir / path).resolve()
        return path

    def _resolve_hf_token(self) -> Optional[str]:
        for env_name in ("HUGGING_FACE_HUB_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            token = str(os.getenv(env_name, "") or "").strip()
            if token:
                return token
        return None

    def _resolve_cached_snapshot(self, spec: ModelSpec) -> Optional[Path]:
        cache_dir = Path(config.HF_CACHE_DIR)
        repo_name = spec.source.repo_id.replace("/", "--")
        repo_root = cache_dir / f"models--{repo_name}"
        snapshots = repo_root / "snapshots"
        if snapshots.exists():
            candidates = [p for p in snapshots.iterdir() if p.is_dir()]
            if candidates:
                return max(candidates, key=lambda path: path.stat().st_mtime)
        return None

    def _build_file_projection(self, spec: ModelSpec) -> Dict[str, str]:
        if spec.source.file_map:
            return dict(spec.source.file_map)
        return {name: name for name in spec.source.files}

    def _download_to_local_target(
        self,
        spec: ModelSpec,
        local_target: Path,
        snapshot_download,
        download_kwargs: Dict,
        endpoint: Optional[str] = None,
    ) -> Path:
        staging_root = Path(config.TEMP_DIR) / "model-downloads" / spec.id
        staging_dir = staging_root / uuid4().hex
        staging_dir.mkdir(parents=True, exist_ok=True)
        download_kwargs = dict(download_kwargs)
        download_kwargs["local_dir"] = staging_dir.as_posix()
        download_kwargs["local_dir_use_symlinks"] = False

        try:
            snapshot_root = self._snapshot_download_with_compat(snapshot_download, download_kwargs, endpoint)
            return self._install_snapshot_to_local(spec, snapshot_root, local_target)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def _download_with_endpoint_fallback(self, spec: ModelSpec, snapshot_download, download_kwargs: Dict, executor) -> Path:
        errors: List[str] = []
        endpoints = self._resolve_download_endpoints(spec)
        for endpoint in endpoints:
            try:
                if endpoint:
                    logger.info("尝试下载模型: model=%s repo=%s endpoint=%s", spec.id, spec.source.repo_id, endpoint)
                return executor(endpoint)
            except Exception as exc:
                endpoint_label = endpoint or "default"
                errors.append(f"{endpoint_label}: {exc}")
                if endpoint != endpoints[-1]:
                    logger.warning(
                        "模型下载失败，准备切换下一个源: model=%s repo=%s endpoint=%s error=%s",
                        spec.id,
                        spec.source.repo_id,
                        endpoint_label,
                        exc,
                    )
                    continue
                event_bus = get_model_download_event_bus()
                joined = " | ".join(errors)
                event_bus.error(spec.id, joined, spec.source.repo_id)
                raise RuntimeError(
                    f"模型下载失败，已尝试源 {endpoints}: {joined}"
                ) from exc
        raise RuntimeError(f"模型下载失败，未解析到可用下载源: {spec.id}")

    def _resolve_download_endpoints(self, spec: ModelSpec) -> List[Optional[str]]:
        endpoints: List[Optional[str]] = []
        for endpoint in list(spec.source.mirrors or []):
            normalized = self._normalize_hf_endpoint(endpoint)
            if normalized and normalized not in endpoints:
                endpoints.append(normalized)

        configured = self._normalize_hf_endpoint(os.getenv("HF_ENDPOINT") or getattr(config, "HF_ENDPOINT", None))
        if configured and configured not in endpoints:
            endpoints.append(configured)

        official = self._normalize_hf_endpoint("https://huggingface.co")
        if official and official not in endpoints:
            endpoints.append(official)

        return endpoints or [None]

    @staticmethod
    def _normalize_hf_endpoint(endpoint: Optional[str]) -> Optional[str]:
        value = str(endpoint or "").strip().rstrip("/")
        return value or None

    def _snapshot_download_with_compat(
        self,
        snapshot_download,
        download_kwargs: Dict,
        endpoint: Optional[str] = None,
    ) -> Path:
        kwargs = dict(download_kwargs)
        original_endpoint = os.environ.get("HF_ENDPOINT")
        endpoint_lock = _ModelDownloadTqdmFactory.get_lock()
        with endpoint_lock:
            try:
                if endpoint:
                    os.environ["HF_ENDPOINT"] = endpoint
                elif original_endpoint is None and "HF_ENDPOINT" in os.environ:
                    del os.environ["HF_ENDPOINT"]
                try:
                    return Path(snapshot_download(**kwargs))
                except TypeError:
                    if "token" in kwargs and "use_auth_token" not in kwargs:
                        kwargs["use_auth_token"] = kwargs.pop("token")
                    kwargs.pop("max_workers", None)
                try:
                    return Path(snapshot_download(**kwargs))
                except TypeError:
                    kwargs.pop("local_dir_use_symlinks", None)
                    return Path(snapshot_download(**kwargs))
            finally:
                if original_endpoint is None:
                    os.environ.pop("HF_ENDPOINT", None)
                else:
                    os.environ["HF_ENDPOINT"] = original_endpoint

    def _install_snapshot_to_local(self, spec: ModelSpec, snapshot_root: Path, local_target: Path) -> Path:
        projection = self._build_file_projection(spec)
        if not projection:
            raise FileNotFoundError(f"模型未声明需安装的文件: {spec.id}")

        local_target.parent.mkdir(parents=True, exist_ok=True)
        staging_target = local_target.with_name(local_target.name + ".downloading")
        self._remove_path(staging_target)
        staging_target.mkdir(parents=True, exist_ok=True)

        try:
            for source_relative, target_relative in projection.items():
                source_path = snapshot_root / source_relative
                if not source_path.exists():
                    raise FileNotFoundError(f"模型下载结果缺失文件: {spec.id} 缺少 {source_relative}")
                target_path = staging_target / target_relative
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if source_path.is_dir():
                    shutil.copytree(source_path, target_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(source_path, target_path)

            self._remove_path(local_target)
            staging_target.replace(local_target)
            return local_target
        except Exception:
            self._remove_path(staging_target)
            raise

    @staticmethod
    def _remove_path(path: Path) -> None:
        if not path.exists():
            return
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=False)
            return
        path.unlink()


class _ModelDownloadTqdmFactory:
    """为 snapshot_download 注入进度回调。"""

    _lock = threading.RLock()

    def __init__(self, event_bus, model_id: str, repo_id: Optional[str]):
        self._event_bus = event_bus
        self._model_id = model_id
        self._repo_id = repo_id

    def __call__(self, *args, **kwargs):
        return _ModelDownloadTqdm(self._event_bus, self._model_id, self._repo_id, *args, **kwargs)

    @classmethod
    def get_lock(cls):
        return cls._lock

    @classmethod
    def set_lock(cls, lock):
        cls._lock = lock


class _ModelDownloadTqdm:
    """huggingface_hub 兼容的 tqdm 轻量适配器。"""

    def __init__(self, event_bus, model_id: str, repo_id: Optional[str], *args, **kwargs):
        self._event_bus = event_bus
        self._model_id = model_id
        self._repo_id = repo_id
        self._iterable = args[0] if args else None
        self.total = kwargs.get("total")
        self.desc = kwargs.get("desc")
        self.n = 0
        self._last_emit = 0.0
        self._emit_interval = 0.2

    def __iter__(self):
        if self._iterable is None:
            return iter(())
        for item in self._iterable:
            self.update(1)
            yield item

    def update(self, n: int = 1):
        self.n += n
        now = time.time()
        if now - self._last_emit >= self._emit_interval or (self.total and self.n >= self.total):
            self._event_bus.progress(
                self._model_id,
                downloaded_bytes=self.n,
                total_bytes=self.total,
                file=self.desc,
                repo_id=self._repo_id,
            )
            self._last_emit = now

    def set_description(self, desc: str, refresh: bool = True):
        self.desc = desc

    def set_description_str(self, desc: str, refresh: bool = True):
        self.desc = desc

    def refresh(self):
        return None

    def close(self):
        self._event_bus.progress(
            self._model_id,
            downloaded_bytes=self.n,
            total_bytes=self.total,
            file=self.desc,
            repo_id=self._repo_id,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
