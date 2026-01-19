"""
模型下载与校验流水线。
默认优先使用本地路径；当允许下载且存在 repo_id 时调用 Hugging Face snapshot_download。
V3.2.0+dev.20260116.04: 区分自带模型和需下载模型
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

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

    def _is_bundled_model(self, spec: ModelSpec) -> bool:
        """
        判断模型是否为自带模型（无需下载）。

        V3.2.0+dev.20260116.04: 区分自带模型和需下载模型
        - backend/models/pretrained/ 下的模型是自带的
        - 根目录 models/ 下的模型需要下载

        Args:
            spec: 模型规格

        Returns:
            bool: True 表示自带模型，False 表示需下载
        """
        if not spec.source.local_path:
            return False

        local_path = spec.source.local_path

        # 判断是否在 backend/models/pretrained/ 下
        if local_path.startswith("backend/models/pretrained/") or \
           local_path.startswith("backend\\models\\pretrained\\"):
            return True

        return False

    def ensure_local(self, spec: ModelSpec) -> str:
        """
        确保模型在本地可用，必要时触发下载。

        V3.2.0+dev.20260116.04: 自带模型缺失时报错，不尝试下载

        Returns: 本地路径字符串。
        """
        event_bus = get_model_download_event_bus()
        local = self.find_local(spec)
        if local:
            self._validate_files(spec, local)
            event_bus.cache_hit(spec.id, str(local), spec.source.repo_id)
            return str(local)

        # V3.2.0+dev.20260116.04: 自带模型缺失时直接报错
        if self._is_bundled_model(spec):
            event_bus.error(spec.id, "自带模型缺失", spec.source.repo_id)
            raise FileNotFoundError(
                f"自带模型缺失: {spec.id} 路径={spec.source.local_path}\n"
                f"请确保模型文件已正确放置在整合包中"
            )

        if not spec.source.repo_id:
            event_bus.error(spec.id, "模型未提供本地路径或仓库", spec.source.repo_id)
            raise FileNotFoundError(f"模型未提供本地路径或仓库: {spec.id}")

        # V3.2.0+dev.20260116.03: Whisper 模型允许自动下载
        is_whisper = spec.kind == "asr" and "whisper" in spec.id.lower()
        if not self.allow_download and not is_whisper:
            event_bus.error(spec.id, "模型缺失且未允许下载", spec.source.repo_id)
            raise FileNotFoundError(f"模型缺失且未允许下载: {spec.id} repo={spec.source.repo_id}")

        path = self._download_from_hf(spec)
        self._validate_files(spec, path)
        event_bus.complete(spec.id, str(path), spec.source.repo_id)
        return str(path)

    def find_local(self, spec: ModelSpec) -> Optional[Path]:
        """尝试查找本地已有路径，不触发下载。"""
        if spec.source.local_path:
            path = Path(spec.source.local_path)
            # V3.2.0+dev.20260116.02: 如果是相对路径且有 base_dir，则相对于 base_dir 解析
            if not path.is_absolute() and self.base_dir:
                path = (self.base_dir / path).resolve()
            if path.exists():
                return path
        if spec.source.repo_id:
            cache_dir = Path(config.HF_CACHE_DIR)
            repo_name = spec.source.repo_id.replace("/", "--")
            repo_root = cache_dir / f"models--{repo_name}"
            snapshots = repo_root / "snapshots"
            if snapshots.exists():
                # 选择最新快照
                candidates = [p for p in snapshots.iterdir() if p.is_dir()]
                if candidates:
                    latest = max(candidates, key=lambda p: p.stat().st_mtime)
                    return latest
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
        cache_dir = Path(config.HF_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            local_path = snapshot_download(
                repo_id=spec.source.repo_id,
                cache_dir=cache_dir.as_posix(),
                resume_download=True,
                local_files_only=self.local_files_only,
                tqdm_class=_ModelDownloadTqdmFactory(event_bus, spec.id, spec.source.repo_id),
            )
        except Exception as exc:
            event_bus.error(spec.id, str(exc), spec.source.repo_id)
            raise
        return Path(local_path)


class _ModelDownloadTqdmFactory:
    """为 snapshot_download 注入进度回调。"""

    def __init__(self, event_bus, model_id: str, repo_id: Optional[str]):
        self._event_bus = event_bus
        self._model_id = model_id
        self._repo_id = repo_id

    def __call__(self, *args, **kwargs):
        return _ModelDownloadTqdm(self._event_bus, self._model_id, self._repo_id, *args, **kwargs)


class _ModelDownloadTqdm:
    """huggingface_hub 兼容的 tqdm 轻量适配器。"""

    def __init__(self, event_bus, model_id: str, repo_id: Optional[str], *args, **kwargs):
        self._event_bus = event_bus
        self._model_id = model_id
        self._repo_id = repo_id
        self.total = kwargs.get("total")
        self.desc = kwargs.get("desc")
        self.n = 0
        self._last_emit = 0.0
        self._emit_interval = 0.2

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
