"""
标点调试输出工具。
V3.2.0+dev.20260203.04
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


_TRUE_VALUES = {"1", "true", "yes", "on", "y", "t"}


def is_debug_punctuation_enabled(
    *,
    env_value: Optional[str],
    config_value: bool,
) -> bool:
    """判断是否启用标点调试输出。"""
    if env_value is None:
        return bool(config_value)
    return str(env_value).strip().lower() in _TRUE_VALUES


def append_debug_punctuation_line(
    job_dir: Optional[Path],
    payload: Dict[str, Any],
    *,
    logger: Optional[logging.Logger] = None,
) -> None:
    """追加调试输出到 debug/punctuation.jsonl。"""
    if not job_dir:
        return
    log = logger or logging.getLogger(__name__)
    try:
        debug_dir = job_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        file_path = debug_dir / "punctuation.jsonl"
        line = dict(payload)
        line.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
        with file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception as exc:
        log.debug("写入标点调试输出失败（忽略）: %s", exc)


def append_debug_whisper_line(
    job_dir: Optional[Path],
    payload: Dict[str, Any],
    *,
    logger: Optional[logging.Logger] = None,
) -> None:
    """追加 Whisper 调试输出到 debug/whisper_debug.jsonl。"""
    if not job_dir:
        return
    log = logger or logging.getLogger(__name__)
    try:
        debug_dir = job_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        file_path = debug_dir / "whisper_debug.jsonl"
        line = dict(payload)
        line.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
        with file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception as exc:
        log.debug("写入 Whisper 调试输出失败（忽略）: %s", exc)
