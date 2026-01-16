"""
模型下载与校验流水线。
默认优先使用本地路径；当允许下载且存在 repo_id 时调用 Hugging Face snapshot_download。
V3.2.0+dev.20260116.04: 区分自带模型和需下载模型
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.core.asr.model_spec import ModelSpec
from app.services.model_validator import ModelValidator
from app.core.config import config

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
        local = self.find_local(spec)
        if local:
            self._validate_files(spec, local)
            return str(local)

        # V3.2.0+dev.20260116.04: 自带模型缺失时直接报错
        if self._is_bundled_model(spec):
            raise FileNotFoundError(
                f"自带模型缺失: {spec.id} 路径={spec.source.local_path}\n"
                f"请确保模型文件已正确放置在整合包中"
            )

        if not spec.source.repo_id:
            raise FileNotFoundError(f"模型未提供本地路径或仓库: {spec.id}")

        # V3.2.0+dev.20260116.03: Whisper 模型允许自动下载
        is_whisper = spec.kind == "asr" and "whisper" in spec.id.lower()
        if not self.allow_download and not is_whisper:
            raise FileNotFoundError(f"模型缺失且未允许下载: {spec.id} repo={spec.source.repo_id}")

        path = self._download_from_hf(spec)
        self._validate_files(spec, path)
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

        logger.info("下载模型: %s", spec.id)
        cache_dir = Path(config.HF_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_path = snapshot_download(
            repo_id=spec.source.repo_id,
            cache_dir=cache_dir.as_posix(),
            resume_download=True,
            local_files_only=self.local_files_only,
        )
        return Path(local_path)
