#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AnchorFlux Update Helper
------------------------
独立执行的更新辅助工具，用于在主程序退出后完成文件替换并重新启动应用。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Set, Tuple


DEFAULT_EXCLUDE_DIRS = {
    "jobs",
    "models",
    "temp",
    "output",
    "input",
    "logs",
    "tools",
    ".venv",
    "node_modules",
    ".git",
    "backend/models",
    "backend/app/assets",
}

DEFAULT_EXCLUDE_FILES = {".env", ".env_installed", ".req_hash"}


def log(msg: str):
    print(f"[UpdateHelper] {msg}", flush=True)


def load_update_config(target_dir: Path) -> Tuple[Set[str], Set[str]]:
    config_path = target_dir / "backend" / "update_config.json"
    if not config_path.exists():
        return DEFAULT_EXCLUDE_DIRS.copy(), DEFAULT_EXCLUDE_FILES.copy()

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        dirs = set(data.get("exclude_dirs", DEFAULT_EXCLUDE_DIRS))
        files = set(data.get("exclude_files", DEFAULT_EXCLUDE_FILES))
        return dirs, files
    except Exception as exc:
        log(f"Failed to read update_config.json: {exc}, using defaults")
        return DEFAULT_EXCLUDE_DIRS.copy(), DEFAULT_EXCLUDE_FILES.copy()


def normalized(path: str) -> str:
    return path.replace("\\", "/")


def should_skip_dir(rel_path: str, exclude_dirs: Set[str]) -> bool:
    rel_norm = normalized(rel_path)
    for rule in exclude_dirs:
        rule_norm = normalized(rule)
        if rel_norm == rule_norm or rel_norm.startswith(f"{rule_norm}/"):
            return True
    return False


def should_skip_file(rel_path: str, exclude_files: Set[str]) -> bool:
    rel_norm = normalized(rel_path)
    name = rel_norm.split("/")[-1]
    for rule in exclude_files:
        rule_norm = normalized(rule)
        if rel_norm == rule_norm or name == rule_norm:
            return True
    return False


def wait_for_file_release(exe_path: Path, timeout: int = 300) -> bool:
    """等待指定文件可写，最多 timeout 秒"""
    end_time = time.time() + timeout
    while time.time() < end_time:
        try:
            with open(exe_path, "ab"):
                return True
        except PermissionError:
            time.sleep(1)
    return False


def copy_with_excludes(source: Path, target: Path, exclude_dirs: Set[str], exclude_files: Set[str]):
    for item in source.rglob("*"):
        rel_path = item.relative_to(source)
        rel_str = normalized(str(rel_path))

        if item.is_dir():
            if should_skip_dir(rel_str, exclude_dirs):
                continue
            (target / rel_path).mkdir(parents=True, exist_ok=True)
        else:
            if should_skip_file(rel_str, exclude_files):
                continue
            dest = target / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)


def schedule_cleanup(source: Path):
    """通过临时 BAT 删除 staging 目录"""
    cleanup_dir = source.parent
    script = cleanup_dir / "cleanup_update.bat"
    bat_content = f"""@echo off
ping 127.0.0.1 -n 5 >nul
rmdir /s /q "{source}"
del "%~f0"
"""
    script.write_text(bat_content, encoding="utf-8")
    subprocess.Popen(["cmd", "/c", "start", "", str(script)], cwd=str(cleanup_dir))


def restart_application(exe_path: Path):
    subprocess.Popen([str(exe_path)], cwd=str(exe_path.parent))


def main():
    parser = argparse.ArgumentParser(description="AnchorFlux Update Helper")
    parser.add_argument("--source", required=True, help="已解压的新版本目录")
    parser.add_argument("--target", required=True, help="程序安装目录")
    parser.add_argument("--exe-path", required=True, help="AnchorFlux.exe 的完整路径")
    parser.add_argument("--cleanup", action="store_true", help="完成后删除 staging 目录")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    target = Path(args.target).resolve()
    exe_path = Path(args.exe_path).resolve()

    log(f"Staging directory: {source}")
    log(f"Installing to: {target}")
    log("Waiting for running application to exit...")

    if not wait_for_file_release(exe_path):
        log("ERROR: Timed out waiting for application to exit")
        sys.exit(2)

    log("Applying update...")
    exclude_dirs, exclude_files = load_update_config(target)
    copy_with_excludes(source, target, exclude_dirs, exclude_files)

    if args.cleanup:
        log("Scheduling cleanup...")
        schedule_cleanup(source)

    log("Restarting application...")
    restart_application(exe_path)
    log("Update completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"Fatal error: {exc}")
        sys.exit(1)
