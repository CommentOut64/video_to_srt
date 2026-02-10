# -*- coding: utf-8 -*-
"""
uv 依赖管理模块
V3.2.0+dev.20260209.01

使用 uv 替代 pip 进行依赖管理，特性：
- uv sync --check 智能检测依赖状态
- 开发模式使用 --all-extras
- 生产模式使用基础依赖
- 无需标志文件（.env_installed, .req_hash）
"""

import os
import subprocess
import shutil
import logging
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("launcher.uv")


@dataclass
class UvResult:
    """uv 命令执行结果"""
    success: bool
    message: str
    needs_sync: bool = False


class UvManager:
    """uv 依赖管理器"""

    def __init__(self, project_root: Path, dev_mode: bool = False):
        self.project_root = project_root
        self.dev_mode = dev_mode
        self.uv_exec = self._find_uv()

    def _find_uv(self) -> Optional[Path]:
        """查找 uv 可执行文件"""
        # 1. 项目内置的 uv
        tools_uv = self.project_root / "tools" / "uv.exe"
        if tools_uv.exists():
            return tools_uv

        # 2. 系统 PATH 中的 uv
        uv_path = shutil.which("uv")
        if uv_path:
            return Path(uv_path)

        return None

    def is_available(self) -> bool:
        """检查 uv 是否可用"""
        return self.uv_exec is not None

    def check_sync_status(self) -> UvResult:
        """
        检查依赖同步状态

        使用 uv sync --check 快速检测：
        - 返回码 0：环境已同步，无需操作
        - 返回码 1：环境过期，需要运行 uv sync
        """
        if not self.is_available():
            return UvResult(
                success=False,
                message="uv 未安装或不可用",
                needs_sync=True
            )

        try:
            cmd = [str(self.uv_exec), "sync", "--check", "--quiet"]

            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                timeout=60
            )

            if result.returncode == 0:
                return UvResult(
                    success=True,
                    message="依赖已同步，无需更新",
                    needs_sync=False
                )
            else:
                return UvResult(
                    success=True,
                    message="依赖需要同步",
                    needs_sync=True
                )

        except subprocess.TimeoutExpired:
            return UvResult(
                success=False,
                message="检查依赖状态超时",
                needs_sync=True
            )
        except Exception as e:
            return UvResult(
                success=False,
                message=f"检查依赖状态失败: {e}",
                needs_sync=True
            )

    def sync_dependencies(self, progress_callback=None) -> UvResult:
        """
        同步依赖

        Args:
            progress_callback: 进度回调函数 (message: str, progress: float)

        Returns:
            UvResult: 同步结果
        """
        if not self.is_available():
            return UvResult(
                success=False,
                message="uv 未安装或不可用"
            )

        # 构建命令
        cmd = [str(self.uv_exec), "sync"]

        if self.dev_mode:
            # 开发模式：安装所有可选依赖
            cmd.append("--all-extras")
            logger.info("开发模式：同步所有依赖（包括开发工具）")
        else:
            # 生产模式：仅安装核心依赖 + launcher 依赖
            cmd.extend(["--extra", "launcher"])
            logger.info("生产模式：同步核心依赖")

        if progress_callback:
            progress_callback("正在同步依赖...", 0.1)

        try:
            # 实时输出进度
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            output_lines = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    line = line.strip()
                    output_lines.append(line)
                    logger.debug(line)

                    # 解析进度信息
                    if progress_callback:
                        if "Resolved" in line:
                            progress_callback("正在解析依赖...", 0.3)
                        elif "Prepared" in line or "Downloaded" in line:
                            progress_callback("正在下载包...", 0.5)
                        elif "Installed" in line:
                            progress_callback("正在安装包...", 0.7)
                        elif "Uninstalled" in line:
                            progress_callback("正在清理旧包...", 0.6)

            if process.returncode == 0:
                if progress_callback:
                    progress_callback("依赖同步完成", 1.0)
                return UvResult(
                    success=True,
                    message="依赖同步完成"
                )
            else:
                error_msg = "\n".join(output_lines[-10:])  # 最后10行
                return UvResult(
                    success=False,
                    message=f"依赖同步失败:\n{error_msg}"
                )

        except Exception as e:
            return UvResult(
                success=False,
                message=f"依赖同步异常: {e}"
            )

    def get_python_path(self) -> Optional[Path]:
        """获取 uv 管理的 Python 路径"""
        venv_python = self.project_root / ".venv" / "Scripts" / "python.exe"
        if venv_python.exists():
            return venv_python

        # Linux/macOS
        venv_python_unix = self.project_root / ".venv" / "bin" / "python"
        if venv_python_unix.exists():
            return venv_python_unix

        return None

    def get_site_packages(self) -> Optional[Path]:
        """获取 site-packages 路径"""
        # Windows
        site_packages = self.project_root / ".venv" / "Lib" / "site-packages"
        if site_packages.exists():
            return site_packages

        # Linux/macOS (需要检测 Python 版本)
        venv_lib = self.project_root / ".venv" / "lib"
        if venv_lib.exists():
            for item in venv_lib.iterdir():
                if item.name.startswith("python"):
                    sp = item / "site-packages"
                    if sp.exists():
                        return sp

        return None

    def ensure_venv(self) -> UvResult:
        """确保虚拟环境存在"""
        venv_dir = self.project_root / ".venv"

        if venv_dir.exists():
            return UvResult(success=True, message="虚拟环境已存在")

        if not self.is_available():
            return UvResult(
                success=False,
                message="uv 未安装，无法创建虚拟环境"
            )

        try:
            cmd = [str(self.uv_exec), "venv", str(venv_dir)]
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                timeout=120
            )

            if result.returncode == 0:
                return UvResult(success=True, message="虚拟环境创建成功")
            else:
                stderr = result.stderr.decode('utf-8', errors='replace')
                return UvResult(success=False, message=f"创建虚拟环境失败: {stderr}")

        except Exception as e:
            return UvResult(success=False, message=f"创建虚拟环境异常: {e}")


def fix_pytorch_dll(site_packages: Path):
    """
    修复 PyTorch 在 Windows 上的 DLL 依赖问题

    PyTorch 2.x+cu121 的 fbgemm.dll 依赖 libomp140.x86_64.dll (LLVM OpenMP)，
    但 Windows 系统默认不包含此 DLL。
    """
    import shutil as shutil_mod

    torch_lib = site_packages / "torch" / "lib"
    source_dll = torch_lib / "libiomp5md.dll"
    target_dll = torch_lib / "libomp140.x86_64.dll"

    if not torch_lib.exists():
        return  # PyTorch 未安装

    if target_dll.exists():
        return  # 已修复

    if source_dll.exists():
        logger.info("修复 PyTorch DLL 依赖...")
        shutil_mod.copy(source_dll, target_dll)
        logger.info("DLL 修复完成")
