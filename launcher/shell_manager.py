# -*- coding: utf-8 -*-
"""
Electron Shell 管理模块
V3.2.4+dev.20260303.04
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional
from urllib import error, request

logger = logging.getLogger("launcher.shell")


def wait_backend_ready(base_url: str, timeout_sec: int = 60) -> bool:
    """
    轮询后端 ready 探针。

    设计取舍：
    - 失败不抛异常，返回 False 由调用方决定降级策略（回退浏览器）。
    - 探测频率固定为 1 秒，避免短时间高频请求造成日志噪音。
    """
    ready_url = f"{base_url.rstrip('/')}/api/system/ready"
    deadline = time.time() + timeout_sec

    while time.time() < deadline:
        try:
            req = request.Request(ready_url, method="GET")
            with request.urlopen(req, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if bool(payload.get("success")) and bool(payload.get("ready")):
                logger.info("后端就绪探针通过: %s", ready_url)
                return True
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            logger.debug("后端就绪探针未通过: %s", exc)
        time.sleep(1)

    logger.error("等待后端就绪超时: timeout=%ss url=%s", timeout_sec, ready_url)
    return False


def launch_electron(shell_path: Path) -> Optional[subprocess.Popen]:
    """
    拉起 Electron Shell。

    设计取舍：
    - 失败时返回 None，不中断启动器主流程，交由调用方决定回退行为。
    """
    if not shell_path.exists():
        logger.error("Electron Shell 不存在: %s", shell_path)
        return None

    try:
        process = subprocess.Popen(
            [str(shell_path)],
            cwd=str(shell_path.parent),
        )
        logger.info("Electron Shell 已启动: %s (PID=%s)", shell_path, process.pid)
        return process
    except OSError as exc:
        logger.error("启动 Electron Shell 失败: %s", exc)
        return None


def open_browser_fallback(url: str) -> bool:
    """在 Shell 拉起失败时回退浏览器打开。"""
    try:
        if os.name == "nt":
            subprocess.Popen(
                ["cmd", "/c", "start", "", url],
                shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            import webbrowser

            webbrowser.open(url, new=2)
        logger.warning("已回退浏览器打开: %s", url)
        return True
    except OSError as exc:
        logger.error("浏览器回退失败: %s", exc)
        return False


def resolve_shell_path(project_root: Path, configured_shell_path: Optional[Path]) -> Path:
    """解析 Shell 可执行文件路径（环境/配置优先）。"""
    if configured_shell_path:
        return configured_shell_path
    return project_root / "core" / "shell" / "AnchorFluxShell.exe"
