"""
模型边界判定工具。

职责：
1. 统一判断模型是否属于 backend/models/pretrained 预置集合；
2. 统一判断模型应由启动自愈处理，还是在运行时按需下载。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.core.asr.model_spec import ModelSpec
from app.core.config import config


def get_pretrained_root() -> Path:
    """返回预置模型根目录的绝对路径。"""
    return (Path(config.BASE_DIR) / "backend" / "models" / "pretrained").resolve(strict=False)


def resolve_model_local_path(local_path: Optional[str]) -> Optional[Path]:
    """将模型 local_path 规范化为绝对路径。"""
    if not local_path:
        return None
    candidate = Path(str(local_path))
    if not candidate.is_absolute():
        candidate = Path(config.BASE_DIR) / candidate
    return candidate.resolve(strict=False)


def is_pretrained_local_path(local_path: Optional[str]) -> bool:
    """判断 local_path 是否位于 backend/models/pretrained 下。"""
    resolved = resolve_model_local_path(local_path)
    if resolved is None:
        return False
    root = get_pretrained_root()
    root_parts = root.parts
    cand_parts = resolved.parts
    return len(cand_parts) >= len(root_parts) and cand_parts[: len(root_parts)] == root_parts


def is_bootstrap_required_model(spec: ModelSpec) -> bool:
    """判断模型是否属于启动阶段应自愈的预置模型。"""
    return is_pretrained_local_path(getattr(getattr(spec, "source", None), "local_path", None))


def is_on_demand_model(spec: ModelSpec) -> bool:
    """判断模型是否应在首次真正使用时按需下载。"""
    return not is_bootstrap_required_model(spec)
