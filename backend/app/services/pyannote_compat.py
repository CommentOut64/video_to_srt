"""
Pyannote 兼容适配层。

目标：
1. 统一处理 pyannote 3.x/未来 4.x 的鉴权参数差异；
2. 在 huggingface_hub 新版本下兼容旧的 use_auth_token 调用；
3. 为业务模块提供单一加载入口，降低后续升级改造成本。
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)
_PATCH_FLAG = "_pyannote_hf_hub_patched"
_TORCH_WEIGHTS_ONLY_FLAG = "_pyannote_torch_weights_only_patched"


def _resolve_local_checkpoint_path(
    checkpoint: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """解析本地 checkpoint 路径，兼容 pyannote 3.x 仅接受文件路径的限制。"""
    active_logger = logger or LOGGER
    checkpoint_path = Path(checkpoint)

    if checkpoint_path.is_file():
        return str(checkpoint_path)

    if not checkpoint_path.is_dir():
        return checkpoint

    candidate_files = (
        "pytorch_model.bin",
        "model.safetensors",
        "model.ckpt",
        "weights.ckpt",
    )
    for file_name in candidate_files:
        candidate_path = checkpoint_path / file_name
        if candidate_path.is_file():
            return str(candidate_path)

    for pattern in ("*.bin", "*.safetensors", "*.ckpt", "*.pt", "*.pth"):
        matched_files = sorted(checkpoint_path.glob(pattern))
        if matched_files:
            return str(matched_files[0])

    active_logger.warning(
        "pyannote 本地目录未找到可识别权重文件，保持原路径: %s",
        checkpoint,
    )
    return checkpoint


def _ensure_torch_weights_only_compat(
    logger: Optional[logging.Logger] = None,
) -> None:
    """兼容 torch>=2.6 默认 `weights_only=True` 导致的旧 checkpoint 加载失败。"""
    active_logger = logger or LOGGER

    if os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD"):
        return

    if os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"):
        return

    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    if not getattr(_ensure_torch_weights_only_compat, _TORCH_WEIGHTS_ONLY_FLAG, False):
        active_logger.debug(
            "已启用 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1，兼容 pyannote 旧 checkpoint 加载"
        )
        setattr(_ensure_torch_weights_only_compat, _TORCH_WEIGHTS_ONLY_FLAG, True)


def patch_huggingface_hub_for_pyannote(
    logger: Optional[logging.Logger] = None,
) -> bool:
    """对 huggingface_hub 做一次性兼容补丁。"""
    active_logger = logger or LOGGER
    try:
        import huggingface_hub.file_download as hf_download
    except Exception as exc:
        active_logger.debug(
            "huggingface_hub 不可用，跳过 pyannote 兼容补丁: %s",
            exc,
        )
        return False

    if getattr(hf_download, _PATCH_FLAG, False):
        return True

    original_hf_hub_download = hf_download.hf_hub_download

    def patched_hf_hub_download(*args: Any, **kwargs: Any) -> Any:
        # 兼容旧调用：use_auth_token -> token
        if "use_auth_token" in kwargs:
            if "token" not in kwargs:
                kwargs["token"] = kwargs["use_auth_token"]
            kwargs.pop("use_auth_token", None)
        return original_hf_hub_download(*args, **kwargs)

    hf_download.hf_hub_download = patched_hf_hub_download
    setattr(hf_download, _PATCH_FLAG, True)
    active_logger.debug("已应用 pyannote/huggingface_hub 兼容补丁")
    return True


def _load_pyannote_object(
    object_type: str,
    checkpoint: str,
    token: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    **kwargs: Any,
) -> Any:
    """统一加载 pyannote Pipeline/Model，并兼容 token 参数差异。"""
    active_logger = logger or LOGGER
    patch_huggingface_hub_for_pyannote(active_logger)

    # V3.2.0+dev.20260218.01: 抑制 pyannote/lightning/torch 加载时的第三方库警告
    # 必须包裹 import 语句，因为 torchcodec 警告在 pyannote 模块导入时就触发
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if object_type == "pipeline":
            from pyannote.audio import Pipeline

            pyannote_cls = Pipeline
        elif object_type == "model":
            from pyannote.audio import Model

            pyannote_cls = Model
        else:
            raise ValueError(f"不支持的 pyannote 对象类型: {object_type}")

        call_kwargs = dict(kwargs)
        if token:
            call_kwargs["token"] = token

        _ensure_torch_weights_only_compat(active_logger)

        if object_type == "pipeline":
            # Why:
            # - pyannote 4.x 的 community pipeline 入口是目录（含 config.yaml 与子模型目录）；
            # - 若把目录误解析为单一权重文件，会破坏 pipeline 级加载。
            resolved_checkpoint = checkpoint
        else:
            resolved_checkpoint = _resolve_local_checkpoint_path(
                checkpoint=checkpoint,
                logger=active_logger,
            )

        try:
            return pyannote_cls.from_pretrained(resolved_checkpoint, **call_kwargs)
        except TypeError as exc:
            # 兼容极端场景：若某版本仍要求 use_auth_token，则自动回退。
            should_retry_with_legacy_token = (
                token is not None
                and "token" in str(exc)
                and "unexpected keyword" in str(exc)
            )
            if not should_retry_with_legacy_token:
                raise

            legacy_kwargs = dict(kwargs)
            legacy_kwargs["use_auth_token"] = token
            active_logger.warning(
                "pyannote from_pretrained 不接受 token 参数，回退 use_auth_token"
            )
            return pyannote_cls.from_pretrained(resolved_checkpoint, **legacy_kwargs)


def load_pyannote_pipeline(
    checkpoint: str,
    token: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    **kwargs: Any,
) -> Any:
    """加载 pyannote Pipeline（VAD/分割等）。"""
    return _load_pyannote_object(
        object_type="pipeline",
        checkpoint=checkpoint,
        token=token,
        logger=logger,
        **kwargs,
    )


def load_pyannote_model(
    checkpoint: str,
    token: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    **kwargs: Any,
) -> Any:
    """加载 pyannote Model（如 Brouhaha 权重）。"""
    return _load_pyannote_object(
        object_type="model",
        checkpoint=checkpoint,
        token=token,
        logger=logger,
        **kwargs,
    )

