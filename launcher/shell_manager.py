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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, request

logger = logging.getLogger("launcher.shell")


def _windows_hidden_subprocess_kwargs() -> Dict[str, Any]:
    """返回 Windows 下无窗口子进程参数。"""
    if os.name != "nt":
        return {}

    kwargs: Dict[str, Any] = {}
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if creationflags:
        kwargs["creationflags"] = creationflags

    startupinfo_cls = getattr(subprocess, "STARTUPINFO", None)
    startf_use_showwindow = getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
    sw_hide = getattr(subprocess, "SW_HIDE", 0)
    if startupinfo_cls and startf_use_showwindow:
        startupinfo = startupinfo_cls()
        startupinfo.dwFlags |= startf_use_showwindow
        startupinfo.wShowWindow = sw_hide
        kwargs["startupinfo"] = startupinfo

    return kwargs


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


def _resolve_shell_log_dir(shell_path: Path) -> Path:
    """
    解析 Electron Shell 日志目录。

    设计取舍：
    - 优先允许外部通过环境变量显式指定，便于排障时临时重定向。
    - 未指定时按 `core/shell/AnchorFluxShell.exe` 反推应用根目录，统一落到 `<root>/logs`。
    """
    explicit_log_dir = os.environ.get("ANCHORFLUX_SHELL_LOG_DIR", "").strip()
    if explicit_log_dir:
        return Path(explicit_log_dir).resolve()

    # 打包路径约定：<root>/core/shell/AnchorFluxShell.exe
    app_root = shell_path.parent.parent.parent
    return (app_root / "logs").resolve()


def _build_chromium_log_path(log_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return log_dir / f"electron-chromium-{timestamp}.log"


def launch_electron(shell_path: Path) -> Optional[subprocess.Popen]:
    """
    拉起 Electron Shell。

    设计取舍：
    - 失败时返回 None，不中断启动器主流程，交由调用方决定回退行为。
    """
    if not shell_path.exists():
        logger.error("Electron Shell 不存在: %s", shell_path)
        return None

    log_dir = _resolve_shell_log_dir(shell_path)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("创建 Shell 日志目录失败，回退默认行为: %s", exc)
        log_dir = shell_path.parent
    chromium_log_path = _build_chromium_log_path(log_dir)

    cmd = [
        str(shell_path),
        "--enable-logging",
        f"--log-file={chromium_log_path}",
    ]
    child_env = os.environ.copy()
    child_env["ANCHORFLUX_SHELL_LOG_DIR"] = str(log_dir)
    child_env["ANCHORFLUX_SHELL_CHROMIUM_LOG"] = str(chromium_log_path)

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(shell_path.parent),
            env=child_env,
        )
        logger.info(
            "Electron Shell 已启动: %s (PID=%s, chromium_log=%s)",
            shell_path,
            process.pid,
            chromium_log_path,
        )
        return process
    except OSError as exc:
        logger.error("启动 Electron Shell 失败: %s", exc)
        return None


def open_browser_fallback(url: str) -> bool:
    """在 Shell 拉起失败时回退浏览器打开。"""
    try:
        if os.name == "nt":
            startfile = getattr(os, "startfile", None)
            if callable(startfile):
                startfile(url)
            else:
                subprocess.Popen(
                    ["cmd", "/c", "start", "", url],
                    shell=False,
                    **_windows_hidden_subprocess_kwargs(),
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
