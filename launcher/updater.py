# -*- coding: utf-8 -*-
"""
自更新系统模块
V3.2.0+dev.20260209.01

支持：
- 启动器自身更新（利用 Windows 重命名运行中 exe 的特性）
- 业务代码更新
- 排除列表保护用户数据
"""

import os
import sys
import json
import hashlib
import shutil
import zipfile
import tempfile
import subprocess
import logging
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger("launcher.updater")


@dataclass
class UpdateInfo:
    """更新信息"""
    version: str
    download_url: str
    changelog: str = ""
    size: int = 0
    profile: str = ""
    flavor: str = ""
    channel: str = ""
    sha256: str = ""
    manifest_url: str = ""
    is_launcher_update: bool = False


@dataclass
class UpdateResult:
    """更新结果"""
    success: bool
    message: str
    needs_restart: bool = False


class SelfUpdater:
    """
    自更新管理器

    利用 Windows 允许重命名运行中 exe 的特性实现自更新：
    1. 下载新版本到临时目录
    2. 重命名当前 exe 为 _old.exe
    3. 移动新版本到原位置
    4. 启动新版本并退出
    """

    # 默认排除列表（保护用户数据）
    DEFAULT_EXCLUDE_DIRS = {
        'jobs', 'models', 'temp', 'output', 'input', 'logs',
        'tools/python', '.venv', 'node_modules', '.git',
        'backend/models', 'backend/app/assets', 'data'
    }
    DEFAULT_EXCLUDE_FILES = {'.env', 'uv.lock'}

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.signal_file = project_root / "update_signal.json"
        self.config_file = project_root / "backend" / "update_config.json"
        self.pending_dir = project_root / "core" / "pending"

    def check_signal(self) -> Optional[UpdateInfo]:
        """检查更新信号文件"""
        if not self.signal_file.exists():
            return None

        try:
            with open(self.signal_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return UpdateInfo(
                version=data.get('version', 'unknown'),
                download_url=data.get('download_url', ''),
                changelog=data.get('changelog', ''),
                size=data.get('size', 0),
                profile=data.get('profile', ''),
                flavor=data.get('flavor', ''),
                channel=data.get('channel', ''),
                sha256=data.get('sha256', ''),
                manifest_url=data.get('manifest_url', ''),
                is_launcher_update=data.get('is_launcher_update', False)
            )
        except Exception as e:
            logger.error(f"读取更新信号失败: {e}")
            return None

    def clear_signal(self):
        """清除更新信号"""
        if self.signal_file.exists():
            try:
                self.signal_file.unlink()
                logger.info("更新信号已清除")
            except Exception as e:
                logger.error(f"清除更新信号失败: {e}")

    def _load_exclude_config(self, update_info: Optional[UpdateInfo] = None) -> Tuple[Set[str], Set[str]]:
        """加载排除配置"""
        exclude_dirs = self.DEFAULT_EXCLUDE_DIRS.copy()
        exclude_files = self.DEFAULT_EXCLUDE_FILES.copy()

        if not self.config_file.exists():
            return exclude_dirs, exclude_files

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            if 'exclude_dirs' in config:
                exclude_dirs = set(config['exclude_dirs'])
            if 'exclude_files' in config:
                exclude_files = set(config['exclude_files'])

            return self._apply_profile_exclude_policy(exclude_dirs, exclude_files, update_info)
        except Exception as e:
            logger.warning(f"加载排除配置失败: {e}")
            return self._apply_profile_exclude_policy(
                self.DEFAULT_EXCLUDE_DIRS.copy(),
                self.DEFAULT_EXCLUDE_FILES.copy(),
                update_info,
            )

    def _apply_profile_exclude_policy(
        self,
        exclude_dirs: Set[str],
        exclude_files: Set[str],
        update_info: Optional[UpdateInfo],
    ) -> Tuple[Set[str], Set[str]]:
        """
        根据 profile 调整排除规则。

        设计取舍：
        - 默认行为保持不变（仍排除 `.venv/tools`），确保旧更新包兼容。
        - 当信号文件携带 profile/channel 时，按“分发系统重构”策略允许运行时目录更新，
          解决 Lite 依赖变更无法生效的问题。
        """
        if update_info is None:
            return exclude_dirs, exclude_files

        profile = str(update_info.profile or "").strip().lower()
        channel = str(update_info.channel or "").strip().lower()
        if profile == "" and channel == "":
            return exclude_dirs, exclude_files

        for path in (".venv", "tools", "tools/python"):
            exclude_dirs.discard(path)
        exclude_files.discard("uv.lock")
        logger.info(
            "按 profile 更新排除策略已启用: profile=%s channel=%s，允许更新 .venv/tools/uv.lock",
            profile or "<unknown>",
            channel or "<unknown>",
        )
        return exclude_dirs, exclude_files

    def _verify_download_integrity(self, zip_path: Path, update_info: UpdateInfo) -> bool:
        """按 signal 中的 size/sha256 执行可选完整性校验。"""
        expected_size = int(update_info.size or 0)
        expected_sha256 = str(update_info.sha256 or "").strip().lower()
        actual_size = zip_path.stat().st_size
        if expected_size > 0 and actual_size != expected_size:
            logger.error("更新包大小校验失败: expected=%s actual=%s", expected_size, actual_size)
            return False

        if expected_sha256:
            hasher = hashlib.sha256()
            with open(zip_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    hasher.update(chunk)
            actual_sha256 = hasher.hexdigest().lower()
            if actual_sha256 != expected_sha256:
                logger.error(
                    "更新包哈希校验失败: expected=%s actual=%s",
                    expected_sha256,
                    actual_sha256,
                )
                return False
        return True

    def download_update(self, update_info: UpdateInfo, progress_callback=None) -> Optional[Path]:
        """
        下载更新包

        Args:
            update_info: 更新信息（包含下载地址与可选校验信息）
            progress_callback: 进度回调 (message, progress)

        Returns:
            下载文件路径，失败返回 None
        """
        temp_dir = Path(tempfile.mkdtemp())
        zip_path = temp_dir / "update.zip"

        try:
            if progress_callback:
                progress_callback("正在下载更新包...", 0.1)

            # 下载文件
            def report_hook(block_num, block_size, total_size):
                if progress_callback and total_size > 0:
                    progress = min(0.1 + (block_num * block_size / total_size) * 0.5, 0.6)
                    downloaded = block_num * block_size / 1024 / 1024
                    total = total_size / 1024 / 1024
                    progress_callback(f"下载中: {downloaded:.1f}/{total:.1f} MB", progress)

            urllib.request.urlretrieve(update_info.download_url, str(zip_path), report_hook)

            if not self._verify_download_integrity(zip_path, update_info):
                logger.error("更新包完整性校验失败，终止应用更新")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return None

            if progress_callback:
                progress_callback("下载完成，正在解压...", 0.6)

            logger.info(f"下载完成: {zip_path.stat().st_size} bytes")
            return zip_path

        except Exception as e:
            logger.error(f"下载更新失败: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None

    def extract_update(self, zip_path: Path, progress_callback=None) -> Optional[Path]:
        """解压更新包"""
        try:
            extract_dir = zip_path.parent / "extracted"

            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(str(extract_dir))

            if progress_callback:
                progress_callback("解压完成", 0.7)

            # 处理单层目录结构
            items = list(extract_dir.iterdir())
            if len(items) == 1 and items[0].is_dir():
                return items[0]

            return extract_dir

        except Exception as e:
            logger.error(f"解压更新失败: {e}")
            return None

    def apply_update(
        self,
        source_dir: Path,
        update_info: Optional[UpdateInfo] = None,
        progress_callback=None,
    ) -> UpdateResult:
        """
        应用更新

        Args:
            source_dir: 更新源目录
            progress_callback: 进度回调

        Returns:
            UpdateResult
        """
        if progress_callback:
            progress_callback("正在应用更新...", 0.8)

        exclude_dirs, exclude_files = self._load_exclude_config(update_info)

        try:
            copied, skipped = self._copy_with_excludes(
                source_dir, self.project_root, exclude_dirs, exclude_files
            )

            logger.info(f"更新完成: 复制 {copied} 个文件, 跳过 {skipped} 个项目")

            if progress_callback:
                progress_callback("更新完成", 1.0)

            return UpdateResult(
                success=True,
                message=f"更新成功: {copied} 个文件已更新",
                needs_restart=True
            )

        except Exception as e:
            logger.error(f"应用更新失败: {e}")
            return UpdateResult(success=False, message=f"应用更新失败: {e}")

    def _copy_with_excludes(self, source: Path, dest: Path,
                            exclude_dirs: Set[str], exclude_files: Set[str],
                            prefix: str = "") -> Tuple[int, int]:
        """递归复制，支持排除列表"""
        copied = 0
        skipped = 0

        for item in source.iterdir():
            rel_path = f"{prefix}/{item.name}" if prefix else item.name
            rel_path_norm = rel_path.replace('\\', '/')

            if item.is_dir():
                # 检查目录排除
                is_excluded = any(
                    rel_path_norm == ex.replace('\\', '/') or
                    rel_path_norm.startswith(ex.replace('\\', '/') + '/')
                    for ex in exclude_dirs
                )

                if is_excluded:
                    skipped += 1
                    continue

                dest_subdir = dest / item.name
                dest_subdir.mkdir(parents=True, exist_ok=True)

                sub_copied, sub_skipped = self._copy_with_excludes(
                    item, dest_subdir, exclude_dirs, exclude_files, rel_path
                )
                copied += sub_copied
                skipped += sub_skipped
            else:
                # 检查文件排除
                is_excluded = any(
                    rel_path_norm == ex.replace('\\', '/') or item.name == ex
                    for ex in exclude_files
                )

                if is_excluded:
                    skipped += 1
                    continue

                dest_file = dest / item.name
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest_file)
                copied += 1

        return copied, skipped

    def self_update_launcher(self, new_launcher_path: Path) -> UpdateResult:
        """
        自更新启动器

        利用 Windows 允许重命名运行中 exe 的特性：
        1. 重命名当前 exe 为 _old.exe
        2. 移动新版本到原位置
        3. 启动新版本
        4. 退出当前进程
        """
        if not getattr(sys, 'frozen', False):
            # 源码运行模式，直接复制文件
            logger.info("源码运行模式，直接更新文件")
            return UpdateResult(success=True, message="源码模式无需自更新")

        current_exe = Path(sys.executable)
        old_exe = current_exe.with_name(current_exe.stem + "_old.exe")

        try:
            # Step 1: 删除旧的备份（如果存在）
            if old_exe.exists():
                old_exe.unlink()

            # Step 2: 重命名当前 exe（Windows 允许！）
            logger.info(f"重命名 {current_exe} -> {old_exe}")
            os.rename(current_exe, old_exe)

            # Step 3: 移动新版本到原位置
            logger.info(f"移动 {new_launcher_path} -> {current_exe}")
            shutil.move(str(new_launcher_path), str(current_exe))

            # Step 4: 启动新版本
            logger.info("启动新版本启动器...")
            subprocess.Popen(
                [str(current_exe), "--cleanup-old"],
                cwd=str(current_exe.parent)
            )

            # Step 5: 退出当前进程
            logger.info("退出当前进程...")
            os._exit(0)

        except Exception as e:
            # 回滚：尝试恢复原文件名
            logger.error(f"自更新失败: {e}")
            if old_exe.exists() and not current_exe.exists():
                try:
                    os.rename(old_exe, current_exe)
                    logger.info("已回滚")
                except Exception:
                    pass

            return UpdateResult(success=False, message=f"自更新失败: {e}")

    def cleanup_old_launcher(self):
        """清理旧版本启动器"""
        if not getattr(sys, 'frozen', False):
            return

        current_exe = Path(sys.executable)
        old_exe = current_exe.with_name(current_exe.stem + "_old.exe")

        if old_exe.exists():
            # 等待旧进程完全退出
            import time
            for _ in range(10):
                try:
                    old_exe.unlink()
                    logger.info("旧版本启动器已清理")
                    break
                except PermissionError:
                    time.sleep(0.5)

    def stage_launcher_update(self, new_launcher_path: Path) -> bool:
        """
        将启动器更新暂存到 pending 目录
        供 Go Stub 在下次启动时应用
        """
        try:
            self.pending_dir.mkdir(parents=True, exist_ok=True)

            dest = self.pending_dir / "launcher.exe"
            shutil.copy2(new_launcher_path, dest)

            logger.info(f"启动器更新已暂存: {dest}")
            return True

        except Exception as e:
            logger.error(f"暂存启动器更新失败: {e}")
            return False

    def execute_update(self, update_info: UpdateInfo, progress_callback=None) -> UpdateResult:
        """执行完整的更新流程"""
        # 1. 下载
        zip_path = self.download_update(update_info, progress_callback)
        if not zip_path:
            return UpdateResult(success=False, message="下载更新失败")

        try:
            # 2. 解压
            source_dir = self.extract_update(zip_path, progress_callback)
            if not source_dir:
                return UpdateResult(success=False, message="解压更新失败")

            # 3. 检查是否包含启动器更新
            new_launcher = source_dir / "core" / "launcher.exe"
            if new_launcher.exists() and update_info.is_launcher_update:
                # 暂存启动器更新，由 Go Stub 在下次启动时应用
                if self.stage_launcher_update(new_launcher):
                    logger.info("启动器更新已暂存，将在下次启动时应用")

            # 4. 应用其他更新
            result = self.apply_update(source_dir, update_info, progress_callback)

            # 5. 清理
            self.clear_signal()
            shutil.rmtree(zip_path.parent, ignore_errors=True)

            return result

        except Exception as e:
            logger.error(f"更新流程异常: {e}")
            return UpdateResult(success=False, message=f"更新异常: {e}")
