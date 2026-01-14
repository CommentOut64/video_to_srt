#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AnchorFlux 离线一键升级工具
--------------------------------
将离线更新包解压后，双击运行本工具即可自动完成与在线升级相同的流程。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def log(message: str) -> None:
    print(f"[OneClickUpgrade] {message}", flush=True)


def get_current_dir() -> Path:
    """兼容源码与 PyInstaller 打包后的运行路径"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def validate_install_dir(path: Path) -> bool:
    return (path / "AnchorFlux.exe").exists()


def guess_install_dir(source_dir: Path) -> Optional[Path]:
    """尝试根据常见路径推断安装目录，减少用户交互"""
    candidates = []

    env_path = os.environ.get("ANCHORFLUX_INSTALL")
    if env_path:
        candidates.append(Path(env_path))

    # 约定：若用户将离线包放在 AnchorFlux 根目录下，则父目录即为安装路径
    candidates.append(source_dir.parent)
    # 兼容再次套了一层目录的情况（例如 Desktop\AnchorFlux_update_v3.1.2\payload）
    candidates.append(source_dir.parent.parent)

    for candidate in candidates:
        if candidate and validate_install_dir(candidate):
            return candidate.resolve()
    return None


def ask_install_dir() -> Optional[Path]:
    """无法自动识别时请求用户手动选择"""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="请选择 AnchorFlux 的安装目录")
        root.destroy()
        if folder:
            candidate = Path(folder)
            if validate_install_dir(candidate):
                return candidate.resolve()
            log("所选目录下未找到 AnchorFlux.exe，请确认路径是否正确")
    except Exception as exc:  # noqa: BLE001
        log(f"调用目录选择对话框失败: {exc}")

    try:
        user_input = input("请输入 AnchorFlux 安装目录的完整路径: ").strip('" ').strip()
        if user_input:
            candidate = Path(user_input)
            if validate_install_dir(candidate):
                return candidate.resolve()
            log("输入的目录中未找到 AnchorFlux.exe")
    except KeyboardInterrupt:
        return None

    return None


def run_update(helper: Path, source_dir: Path, target_dir: Path) -> bool:
    """调用 update_helper 完成实际的文件覆盖"""
    cmd = [
        str(helper),
        "--source",
        str(source_dir),
        "--target",
        str(target_dir),
        "--exe-path",
        str(target_dir / "AnchorFlux.exe"),
        "--cleanup",
    ]
    log("开始执行更新，请确保 AnchorFlux 已关闭...")
    proc = subprocess.run(cmd, capture_output=False, text=False)
    return proc.returncode == 0


def main():
    source_dir = get_current_dir()
    helper_path = source_dir / "update_helper.exe"
    if not helper_path.exists():
        log("未找到 update_helper.exe，请确认离线包是否完整")
        input("按回车退出...")
        sys.exit(1)

    install_dir = guess_install_dir(source_dir)
    if not install_dir:
        log("未能自动识别 AnchorFlux 安装目录，将提示用户手动选择")
        install_dir = ask_install_dir()

    if not install_dir:
        log("未提供有效的安装目录，无法继续更新")
        input("按回车退出...")
        sys.exit(1)

    log(f"将更新安装目录: {install_dir}")
    if not run_update(helper_path, source_dir, install_dir):
        log("更新失败，请检查日志后重试")
        input("按回车退出...")
        sys.exit(1)

    log("更新已完成，程序即将自动重新启动，可关闭此窗口。")


if __name__ == "__main__":
    main()
