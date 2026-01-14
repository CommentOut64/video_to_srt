#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AnchorFlux Bootloader - Unified Launcher
V3.1.0+dev.20260104.01

Core Features:
1. Lifecycle loop (Guardian mode): Detect signal file after backend exit
2. Dev/Prod mode switch: Via DEV_MODE in .env
3. Update support: Detect update_signal.json and execute update
4. Cross-environment: Support venv and embedded Python

Usage:
- Double-click to run (production mode)
- Command line: python bootloader.py [--dev]
"""

import os
import sys
import time
import json
import signal
import shutil
import zipfile
import tempfile
import subprocess
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# ========================================
# Constants
# ========================================
VERSION = "3.1.1+dev.20260105.05"
APP_NAME = "AnchorFlux"
DEFAULT_BACKEND_PORT = 8000
DEFAULT_FRONTEND_PORT = 5173
UPDATE_HELPER_NAME = "update_helper.exe"

# Signal files
UPDATE_SIGNAL_FILE = "update_signal.json"
SHUTDOWN_SIGNAL_FILE = ".shutdown_signal"

# Dependency marker files
MARKER_FILE = ".env_installed"
REQ_HASH_FILE = ".req_hash"

# Log formats
LOG_FORMAT_DEV = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_FORMAT_PROD = "%(asctime)s [%(levelname)s] %(message)s"


@dataclass
class BootloaderConfig:
    """Bootloader configuration"""
    project_root: Path
    dev_mode: bool = False
    backend_port: int = DEFAULT_BACKEND_PORT
    frontend_port: int = DEFAULT_FRONTEND_PORT

    # Python paths
    python_exec: Optional[Path] = None
    python_mode: str = "unknown"  # "embedded", "venv", "system"
    site_packages: Optional[Path] = None

    # Tool paths
    tools_dir: Optional[Path] = None
    ffmpeg_path: Optional[Path] = None

    # Environment variables
    pypi_mirror: str = "https://pypi.tuna.tsinghua.edu.cn/simple"
    hf_mirror: bool = True

    # Log level
    log_level: str = "INFO"


class BootloaderLogger:
    """Bootloader logger"""

    def __init__(self, dev_mode: bool = False):
        self.dev_mode = dev_mode
        self.logger = logging.getLogger("bootloader")
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging"""
        level = logging.DEBUG if self.dev_mode else logging.INFO
        fmt = LOG_FORMAT_DEV if self.dev_mode else LOG_FORMAT_PROD

        self.logger.setLevel(level)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(fmt))
        self.logger.addHandler(console_handler)

    def info(self, msg: str):
        self.logger.info(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)


class ProcessManager:
    """Process manager"""

    def __init__(self, logger: BootloaderLogger):
        self.logger = logger
        self.backend_process: Optional[subprocess.Popen] = None
        self.frontend_process: Optional[subprocess.Popen] = None
        self._shutdown_requested = False

    def cleanup_old_processes(self, backend_port: int, frontend_port: int):
        """Clean up residual processes"""
        self.logger.info("Checking for old processes...")

        if os.name != 'nt':
            return

        try:
            import psutil

            # Clean up port occupancy
            for port in [backend_port, frontend_port]:
                for conn in psutil.net_connections(kind='inet'):
                    if conn.laddr.port == port and conn.status == 'LISTEN':
                        try:
                            proc = psutil.Process(conn.pid)
                            self.logger.info(f"Killing process on port {port}: PID={proc.pid}")
                            proc.terminate()
                            proc.wait(timeout=3)
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                            pass

            # Clean up FFmpeg processes
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    name = proc.info['name']
                    if name and name.lower() in ['ffmpeg.exe', 'ffprobe.exe']:
                        proc.terminate()
                        self.logger.debug(f"Terminated FFmpeg process: PID={proc.info['pid']}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            self.logger.info("Old process cleanup completed")

        except ImportError:
            self.logger.warning("psutil not available, using fallback cleanup")
            self._fallback_cleanup(backend_port, frontend_port)

        # Wait for cleanup to complete
        time.sleep(2)

    def _fallback_cleanup(self, backend_port: int, frontend_port: int):
        """Fallback cleanup (without psutil)"""
        for port in [backend_port, frontend_port]:
            try:
                result = subprocess.run(
                    f'netstat -ano | findstr ":{port}" | findstr "LISTENING"',
                    shell=True, capture_output=True, timeout=5
                )
                if result.returncode == 0 and result.stdout:
                    lines = result.stdout.decode('utf-8', errors='replace').strip().split('\n')
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 5:
                            pid = parts[-1]
                            subprocess.run(['taskkill', '/F', '/PID', pid],
                                         capture_output=True, timeout=3)
            except Exception:
                pass

        # Clean up FFmpeg
        try:
            subprocess.run(['taskkill', '/F', '/IM', 'ffmpeg.exe'],
                         capture_output=True, timeout=3)
            subprocess.run(['taskkill', '/F', '/IM', 'ffprobe.exe'],
                         capture_output=True, timeout=3)
        except Exception:
            pass

    def start_backend(self, config: BootloaderConfig) -> bool:
        """Start backend service"""
        self.logger.info(f"Starting backend service on port {config.backend_port}...")

        backend_dir = config.project_root / "backend"

        # Build environment variables
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

        # V3.1.1+dev.20260105.05: 传递 DEV_MODE 给后端，控制是否托管静态文件
        env['DEV_MODE'] = 'true' if config.dev_mode else 'false'

        # Set HuggingFace mirror
        if config.hf_mirror:
            env['HF_ENDPOINT'] = 'https://hf-mirror.com'

        # Set PATH (including PyTorch DLLs and tools)
        if config.site_packages:
            torch_lib = config.site_packages / "torch" / "lib"
            nvidia_cudnn = config.site_packages / "nvidia" / "cudnn" / "bin"
            nvidia_cublas = config.site_packages / "nvidia" / "cublas" / "bin"

            path_additions = []
            for p in [torch_lib, nvidia_cudnn, nvidia_cublas]:
                if p.exists():
                    path_additions.append(str(p))

            if config.tools_dir and config.tools_dir.exists():
                path_additions.append(str(config.tools_dir))

            if path_additions:
                env['PATH'] = ';'.join(path_additions) + ';' + env.get('PATH', '')

        # Start command
        cmd = [
            str(config.python_exec),
            '-m', 'uvicorn',
            'app.main:app',
            '--host', '0.0.0.0',
            '--port', str(config.backend_port)
        ]

        try:
            # V3.1.0+dev.20260104.02: Run backend in same console to see output
            # In production mode, run in same window; in dev mode, create new window
            if config.dev_mode and os.name == 'nt':
                # Dev mode: create new window for backend
                creationflags = subprocess.CREATE_NEW_CONSOLE
                self.backend_process = subprocess.Popen(
                    cmd,
                    cwd=str(backend_dir),
                    env=env,
                    creationflags=creationflags
                )
            else:
                # Production mode: run in same console, inherit stdout/stderr
                self.backend_process = subprocess.Popen(
                    cmd,
                    cwd=str(backend_dir),
                    env=env
                )

            self.logger.info(f"Backend started: PID={self.backend_process.pid}")

            # V3.1.0+dev.20260104.02: Wait a moment and check if backend crashed immediately
            time.sleep(3)
            if self.backend_process.poll() is not None:
                exit_code = self.backend_process.returncode
                self.logger.error(f"Backend crashed immediately with exit code: {exit_code}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Failed to start backend: {e}")
            return False

    def start_frontend(self, config: BootloaderConfig) -> bool:
        """Start frontend service (dev mode only)"""
        if not config.dev_mode:
            self.logger.debug("Production mode, skipping frontend startup")
            return True

        frontend_dir = config.project_root / "frontend"

        # Check node_modules
        if not (frontend_dir / "node_modules").exists():
            self.logger.warning("node_modules not found, running npm install...")
            try:
                subprocess.run('npm install', cwd=str(frontend_dir),
                             check=True, timeout=300, shell=True)
            except Exception as e:
                self.logger.error(f"npm install failed: {e}")
                return False

        self.logger.info(f"Starting frontend service on port {config.frontend_port}...")

        try:
            creationflags = 0
            if os.name == 'nt':
                creationflags = subprocess.CREATE_NEW_CONSOLE

            # V3.1.1+dev.20260105.05: 使用 shell=True 确保能找到 npm
            self.frontend_process = subprocess.Popen(
                'npm run dev',
                cwd=str(frontend_dir),
                creationflags=creationflags,
                shell=True
            )

            self.logger.info(f"Frontend started: PID={self.frontend_process.pid}")

            # V3.1.1+dev.20260105.05: 等待几秒让前端启动
            time.sleep(3)
            if self.frontend_process.poll() is not None:
                exit_code = self.frontend_process.returncode
                self.logger.error(f"Frontend crashed immediately with exit code: {exit_code}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Failed to start frontend: {e}")
            return False

    def wait_for_backend_exit(self) -> int:
        """Wait for backend to exit, return exit code"""
        if self.backend_process is None:
            return -1

        self.logger.info("Waiting for backend to exit...")
        return self.backend_process.wait()

    def is_backend_running(self) -> bool:
        """Check if backend is still running"""
        if self.backend_process is None:
            return False
        return self.backend_process.poll() is None

    def terminate_all(self):
        """Terminate all processes"""
        self._shutdown_requested = True

        if self.frontend_process and self.frontend_process.poll() is None:
            self.logger.info("Terminating frontend process...")
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()

        if self.backend_process and self.backend_process.poll() is None:
            self.logger.info("Terminating backend process...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()


class UpdateManager:
    """Update manager"""

    # V3.1.1+dev.20260105.04: 默认排除列表（配置文件不存在时使用）
    DEFAULT_EXCLUDE_DIRS = {'jobs', 'models', 'temp', 'output', 'input', 'logs',
                           'tools', '.venv', 'node_modules', '.git',
                           'backend/models', 'backend/app/assets'}
    DEFAULT_EXCLUDE_FILES = {'.env', '.env_installed', '.req_hash'}

    def __init__(self, config: BootloaderConfig, logger: BootloaderLogger):
        self.config = config
        self.logger = logger
        self.signal_file = config.project_root / UPDATE_SIGNAL_FILE
        self.update_config_file = config.project_root / "backend" / "update_config.json"

    def _load_update_config(self) -> Tuple[set, set]:
        """
        V3.1.1+dev.20260105.03: 从配置文件加载更新排除列表
        配置文件位于 backend/update_config.json，可通过更新进行更新
        """
        exclude_dirs = self.DEFAULT_EXCLUDE_DIRS.copy()
        exclude_files = self.DEFAULT_EXCLUDE_FILES.copy()

        if not self.update_config_file.exists():
            self.logger.debug(f"Update config not found, using defaults: {self.update_config_file}")
            return exclude_dirs, exclude_files

        try:
            with open(self.update_config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # 读取目录排除列表
            if 'exclude_dirs' in config_data:
                exclude_dirs = set(config_data['exclude_dirs'])
                self.logger.info(f"Loaded {len(exclude_dirs)} exclude dirs from config")

            # 读取文件排除列表
            if 'exclude_files' in config_data:
                exclude_files = set(config_data['exclude_files'])
                self.logger.info(f"Loaded {len(exclude_files)} exclude files from config")

            return exclude_dirs, exclude_files

        except Exception as e:
            self.logger.warning(f"Failed to load update config: {e}, using defaults")
            return self.DEFAULT_EXCLUDE_DIRS.copy(), self.DEFAULT_EXCLUDE_FILES.copy()

    def _copy_with_excludes(self, source_dir: Path, dest_dir: Path,
                            exclude_dirs: set, exclude_files: set,
                            prefix: str = "") -> Tuple[int, int]:
        """
        V3.1.1+dev.20260105.04: 递归复制，支持任意深度的排除路径

        Args:
            source_dir: 源目录
            dest_dir: 目标目录
            exclude_dirs: 排除的目录路径集合（支持相对路径如 "backend/models"）
            exclude_files: 排除的文件路径集合（支持相对路径）
            prefix: 当前路径前缀（用于构建相对路径）

        Returns:
            (copied_count, skipped_count) 复制和跳过的项目数
        """
        copied_count = 0
        skipped_count = 0

        for item in source_dir.iterdir():
            # 构建相对路径（用于匹配排除规则）
            if prefix:
                rel_path = f"{prefix}/{item.name}"
            else:
                rel_path = item.name

            # 统一使用正斜杠，便于跨平台匹配
            rel_path_normalized = rel_path.replace('\\', '/')

            if item.is_dir():
                # 检查目录是否在排除列表中
                # 支持精确匹配和前缀匹配（如 "backend/models" 匹配 "backend/models/xxx"）
                is_excluded = False
                for exclude_dir in exclude_dirs:
                    exclude_normalized = exclude_dir.replace('\\', '/')
                    if rel_path_normalized == exclude_normalized or \
                       rel_path_normalized.startswith(exclude_normalized + '/'):
                        is_excluded = True
                        break

                if is_excluded:
                    self.logger.debug(f"{prefix}[Skip] Excluded dir: {rel_path}")
                    skipped_count += 1
                    continue

                # 目录未排除，递归处理
                dest_subdir = dest_dir / item.name

                # 确保目标目录存在
                dest_subdir.mkdir(parents=True, exist_ok=True)

                # 递归复制子目录内容
                sub_copied, sub_skipped = self._copy_with_excludes(
                    item, dest_subdir, exclude_dirs, exclude_files, rel_path
                )
                copied_count += sub_copied
                skipped_count += sub_skipped

            else:
                # 文件处理
                # 检查文件是否在排除列表中（支持相对路径和文件名匹配）
                is_excluded = False
                for exclude_file in exclude_files:
                    exclude_normalized = exclude_file.replace('\\', '/')
                    # 支持精确路径匹配和纯文件名匹配
                    if rel_path_normalized == exclude_normalized or \
                       item.name == exclude_file:
                        is_excluded = True
                        break

                if is_excluded:
                    self.logger.debug(f"{prefix}[Skip] Excluded file: {rel_path}")
                    skipped_count += 1
                    continue

                # 文件未排除，直接覆盖复制
                dest_file = dest_dir / item.name
                shutil.copy2(item, dest_file)
                copied_count += 1

        return copied_count, skipped_count

    def _stage_update_package(self, download_url: str, version: str) -> Path:
        import urllib.request

        temp_dir = Path(tempfile.mkdtemp())
        zip_path = temp_dir / "update.zip"

        self.logger.info("Downloading update package...")
        urllib.request.urlretrieve(download_url, str(zip_path))
        self.logger.info(f"Downloaded update: {zip_path.stat().st_size} bytes")

        extract_dir = temp_dir / "extracted"
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(str(extract_dir))

        extracted_items = list(extract_dir.iterdir())
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            source_dir = extracted_items[0]
        else:
            source_dir = extract_dir

        staged_root = self.config.project_root / "temp" / "staged_updates"
        staged_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        staged_dir = staged_root / f"{version.replace('/', '_')}_{timestamp}"
        if staged_dir.exists():
            shutil.rmtree(staged_dir, ignore_errors=True)

        shutil.copytree(source_dir, staged_dir, dirs_exist_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)

        self.logger.info(f"Update staged at: {staged_dir}")
        return staged_dir

    def _launch_update_helper(self, staged_dir: Path):
        if not getattr(sys, 'frozen', False):
            self.logger.info("Running from source, applying update directly...")
            exclude_dirs, exclude_files = self._load_update_config()
            copied, skipped = self._copy_with_excludes(
                staged_dir, self.config.project_root, exclude_dirs, exclude_files
            )
            self.logger.info(f"Copied {copied} files, skipped {skipped} items")
            shutil.rmtree(staged_dir, ignore_errors=True)
            return

        helper_path = staged_dir / UPDATE_HELPER_NAME
        if not helper_path.exists():
            fallback = self.config.project_root / UPDATE_HELPER_NAME
            if fallback.exists():
                shutil.copy2(fallback, helper_path)
            else:
                raise FileNotFoundError(f"{UPDATE_HELPER_NAME} not found in update package")

        exe_path = Path(sys.executable)
        cmd = [
            str(helper_path),
            "--source", str(staged_dir),
            "--target", str(self.config.project_root),
            "--exe-path", str(exe_path),
            "--cleanup"
        ]

        self.logger.info("Launching update helper and exiting current process...")
        subprocess.Popen(cmd, cwd=str(staged_dir))
        os._exit(0)

    def check_update_signal(self) -> Optional[Dict[str, Any]]:
        """Check for update signal file"""
        if not self.signal_file.exists():
            return None

        try:
            with open(self.signal_file, 'r', encoding='utf-8') as f:
                signal_data = json.load(f)

            self.logger.info(f"Update signal detected: {signal_data.get('version', 'unknown')}")
            return signal_data

        except Exception as e:
            self.logger.error(f"Failed to read update signal: {e}")
            return None

    def clear_update_signal(self):
        """Clear update signal file"""
        if self.signal_file.exists():
            try:
                self.signal_file.unlink()
                self.logger.info("Update signal cleared")
            except Exception as e:
                self.logger.error(f"Failed to clear update signal: {e}")

    def execute_update(self, signal_data: Dict[str, Any], headless: bool = False) -> bool:
        """
        Execute update by staging files and launching helper executable.
        """
        download_url = signal_data.get('download_url')
        version = signal_data.get('version', 'unknown')

        if not download_url:
            self.logger.error("No download URL in update signal")
            return False

        self.logger.info(f"Starting update to version {version}...")

        try:
            staged_dir = self._stage_update_package(download_url, version)
        except Exception as e:
            self.logger.error(f"Failed to stage update: {e}")
            return False

        # 清理信号，避免重复执行
        self.clear_update_signal()

        try:
            self._launch_update_helper(staged_dir)
        except Exception as e:
            self.logger.error(f"Failed to launch update helper: {e}")
            return False

        return True



def get_project_root() -> Path:
    """
    获取项目根目录

    V3.1.1+dev.20260105.02: 支持打包后的 EXE 运行环境
    - 打包后: sys.executable 指向 EXE 文件，使用其所在目录
    - 源码运行: __file__ 指向 bootloader.py，使用其所在目录
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后，sys.executable 是 EXE 文件路径
        return Path(sys.executable).parent.resolve()
    else:
        # 源码运行
        return Path(__file__).parent.resolve()


class Bootloader:
    """Main bootloader class"""

    def __init__(self):
        self.project_root = get_project_root()
        self.config: Optional[BootloaderConfig] = None
        self.logger: Optional[BootloaderLogger] = None
        self.process_manager: Optional[ProcessManager] = None
        self.update_manager: Optional[UpdateManager] = None
        self._running = True

    def _detect_dev_mode(self) -> bool:
        """Detect development mode"""
        # 1. Command line argument
        if '--dev' in sys.argv:
            return True

        # 2. Environment variable
        if os.environ.get('DEV_MODE', '').lower() in ('true', '1', 'yes'):
            return True

        # 3. .env file
        env_file = self.project_root / '.env'
        if env_file.exists():
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('#') or '=' not in line:
                            continue
                        key, value = line.split('=', 1)
                        if key.strip() == 'DEV_MODE':
                            return value.strip().lower() in ('true', '1', 'yes')
            except Exception:
                pass

        return False

    def _detect_python(self, dev_mode: bool = False) -> Tuple[Optional[Path], str, Optional[Path]]:
        """
        Detect Python environment
        V3.1.0+dev.20260105.01: 根据 DEV_MODE 调整检测优先级
        - DEV_MODE=true: 优先使用 .venv（开发环境）
        - DEV_MODE=false: 优先使用 embedded Python（生产环境）
        """
        tools_dir = self.project_root / "tools"
        embed_python = tools_dir / "python" / "python.exe"
        embed_site_packages = tools_dir / "python" / "Lib" / "site-packages"
        venv_python = self.project_root / ".venv" / "Scripts" / "python.exe"
        venv_site_packages = self.project_root / ".venv" / "Lib" / "site-packages"

        if dev_mode:
            # 开发模式: 优先使用 .venv
            if venv_python.exists():
                return venv_python, "venv", venv_site_packages
            if embed_python.exists():
                return embed_python, "embedded", embed_site_packages
        else:
            # 生产模式: 优先使用 embedded Python
            if embed_python.exists():
                return embed_python, "embedded", embed_site_packages
            if venv_python.exists():
                return venv_python, "venv", venv_site_packages

        # 回退: System Python
        try:
            result = subprocess.run(['python', '--version'],
                                  capture_output=True, timeout=5)
            if result.returncode == 0:
                return Path('python'), "system", None
        except Exception:
            pass

        return None, "not_found", None

    def _check_embedded_dependencies(self, python_exec: Path, site_packages: Path) -> bool:
        """
        检查 Python 环境是否已安装必要依赖
        V3.1.0+dev.20260105.01: 智能检测，支持哈希值比对
        """
        marker_file = self.project_root / MARKER_FILE
        req_hash_file = self.project_root / REQ_HASH_FILE
        req_file = self.project_root / "requirements.txt"

        # 计算当前 requirements.txt 的哈希值
        current_hash = self._get_file_hash(req_file)
        saved_hash = req_hash_file.read_text().strip() if req_hash_file.exists() else ""

        # 如果哈希值相同且标记文件存在，说明无变化
        if current_hash == saved_hash and marker_file.exists():
            return True

        # 额外检查核心包是否存在
        core_packages = ['fastapi', 'uvicorn', 'pydub', 'torch']
        for pkg in core_packages:
            pkg_dir = site_packages / pkg
            pkg_dist = site_packages / f"{pkg.replace('-', '_')}.dist-info"
            if not pkg_dir.exists() and not pkg_dist.exists():
                return False

        return marker_file.exists()

    def _get_file_hash(self, filepath: Path) -> str:
        """计算文件的 MD5 哈希值"""
        if not filepath.exists():
            return ""
        with open(filepath, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _fix_pytorch_dll(self, site_packages: Path):
        """
        修复 PyTorch 在 Windows 上的 DLL 依赖问题

        PyTorch 2.x+cu118 的 fbgemm.dll 依赖 libomp140.x86_64.dll (LLVM OpenMP)，
        但 Windows 系统默认不包含此 DLL。解决方案是将 PyTorch 自带的
        libiomp5md.dll (Intel OpenMP) 复制为 libomp140.x86_64.dll，
        两者 API 兼容。
        """
        torch_lib = site_packages / "torch" / "lib"
        source_dll = torch_lib / "libiomp5md.dll"
        target_dll = torch_lib / "libomp140.x86_64.dll"

        if not torch_lib.exists():
            return  # PyTorch 未安装

        if target_dll.exists():
            return  # 已修复

        if source_dll.exists():
            self.logger.info("Fixing PyTorch DLL dependency (fbgemm.dll -> libomp140.x86_64.dll)...")
            shutil.copy(source_dll, target_dll)
            self.logger.info("DLL fix applied")

    def _fix_onnxruntime_conflict(self, python_exec: Path, site_packages: Path, pypi_mirror: str):
        """
        修复 onnxruntime 版本冲突

        funasr-onnx 会自动安装 onnxruntime (CPU版), 但我们只需要 onnxruntime-gpu
        onnxruntime-gpu 完全兼容 CPU 推理, 可以满足所有依赖
        """
        # 卸载 CPU 版本
        subprocess.run(
            [str(python_exec), '-m', 'pip', 'uninstall', 'onnxruntime', '-y'],
            capture_output=True
        )

        # 清理残留目录
        onnx_dir = site_packages / "onnxruntime"
        if onnx_dir.exists():
            self.logger.info("Cleaning residual onnxruntime directory...")
            shutil.rmtree(onnx_dir, ignore_errors=True)

        # 重新安装 GPU 版本
        self.logger.info("Reinstalling onnxruntime-gpu to ensure integrity...")
        cmd = [
            str(python_exec), '-m', 'pip', 'install',
            '--force-reinstall', '--no-deps',
            'onnxruntime-gpu==1.18.0'
        ]
        if pypi_mirror:
            cmd.extend(['-i', pypi_mirror])
        subprocess.run(cmd, capture_output=True)

    def _install_embedded_dependencies(self, python_exec: Path, site_packages: Path, pypi_mirror: str) -> bool:
        """
        为 Python 环境安装依赖
        V3.1.0+dev.20260105.01: 恢复原有的完整安装逻辑
        """
        req_file = self.project_root / "requirements.txt"
        marker_file = self.project_root / MARKER_FILE
        req_hash_file = self.project_root / REQ_HASH_FILE

        if not req_file.exists():
            self.logger.error(f"requirements.txt not found: {req_file}")
            return False

        self.logger.info("=" * 50)
        self.logger.info("Installing dependencies...")
        self.logger.info("This may take 10-30 minutes on first run...")
        self.logger.info("=" * 50)

        # Step 1: 升级 pip
        self.logger.info("Step 1/4: Upgrading pip...")
        cmd = [str(python_exec), '-m', 'pip', 'install', '--upgrade', 'pip']
        if pypi_mirror:
            cmd.extend(['-i', pypi_mirror])
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            self.logger.warning("Failed to upgrade pip, continuing anyway...")

        # Step 2: 安装依赖
        self.logger.info("Step 2/4: Installing dependencies from requirements.txt...")
        cmd = [
            str(python_exec), '-m', 'pip', 'install',
            '-r', str(req_file)
        ]
        if pypi_mirror:
            cmd.extend(['-i', pypi_mirror])

        try:
            # 实时显示安装进度
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=str(self.project_root)
            )

            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    line = line.strip()
                    # 显示关键信息
                    if any(kw in line.lower() for kw in ['installing', 'successfully', 'requirement', 'downloading', 'error', 'warning', 'collecting']):
                        self.logger.info(f"  {line[:120]}")

            if process.returncode != 0:
                self.logger.error(f"pip install failed with exit code: {process.returncode}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to install dependencies: {e}")
            return False

        # Step 3: 修复 onnxruntime 冲突
        self.logger.info("Step 3/4: Fixing onnxruntime version conflict...")
        self._fix_onnxruntime_conflict(python_exec, site_packages, pypi_mirror)
        self.logger.info("onnxruntime-gpu installed (CPU version removed)")

        # Step 4: 修复 PyTorch DLL
        self.logger.info("Step 4/4: Fixing PyTorch DLL dependencies...")
        self._fix_pytorch_dll(site_packages)

        # 保存标记文件和哈希值
        current_hash = self._get_file_hash(req_file)
        req_hash_file.write_text(current_hash)
        marker_file.touch()

        self.logger.info("=" * 50)
        self.logger.info("Dependencies installed successfully!")
        self.logger.info("=" * 50)
        return True

    def _load_env_config(self) -> Dict[str, str]:
        """Load .env configuration"""
        config = {}
        env_file = self.project_root / '.env'

        if not env_file.exists():
            return config

        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or '=' not in line:
                        continue
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        except Exception:
            pass

        return config

    def initialize(self) -> bool:
        """Initialize bootloader"""
        # Detect dev mode
        dev_mode = self._detect_dev_mode()

        # Initialize logger
        self.logger = BootloaderLogger(dev_mode)

        self.logger.info("=" * 50)
        self.logger.info(f"{APP_NAME} Bootloader v{VERSION}")
        self.logger.info(f"Mode: {'Development' if dev_mode else 'Production'}")
        self.logger.info("=" * 50)

        # Load .env config (需要在 Python 检测前加载，获取 PYPI_MIRROR)
        env_config = self._load_env_config()
        pypi_mirror = env_config.get('PYPI_MIRROR', 'https://pypi.tuna.tsinghua.edu.cn/simple')

        # V3.1.0+dev.20260105.01: 根据 dev_mode 检测 Python
        python_exec, python_mode, site_packages = self._detect_python(dev_mode)

        if python_exec is None:
            self.logger.error("Python not found! Please install Python 3.10+ or place embedded Python in tools/python/")
            return False

        self.logger.info(f"Python Mode: {python_mode}")
        self.logger.info(f"Python Path: {python_exec}")

        # V3.1.0+dev.20260105.01: 检查 Python 依赖，必要时自动安装
        if site_packages:
            if not self._check_embedded_dependencies(python_exec, site_packages):
                self.logger.warning("Python environment missing dependencies, starting installation...")
                if not self._install_embedded_dependencies(python_exec, site_packages, pypi_mirror):
                    self.logger.error("Failed to install dependencies")
                    self.logger.error("Press Enter to exit...")
                    try:
                        input()
                    except:
                        pass
                    return False
            else:
                self.logger.info("Dependencies verified (fast boot mode)")

        # Create config
        self.config = BootloaderConfig(
            project_root=self.project_root,
            dev_mode=dev_mode,
            python_exec=python_exec,
            python_mode=python_mode,
            site_packages=site_packages,
            tools_dir=self.project_root / "tools",
            pypi_mirror=pypi_mirror,
            hf_mirror=env_config.get('USE_HF_MIRROR', 'true').lower() == 'true',
            log_level='DEBUG' if dev_mode else 'INFO'
        )

        # Initialize managers
        self.process_manager = ProcessManager(self.logger)
        self.update_manager = UpdateManager(self.config, self.logger)

        # Set signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        return True

    def _signal_handler(self, signum, frame):
        """Signal handler"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self._running = False
        if self.process_manager:
            self.process_manager.terminate_all()

    def run(self) -> int:
        """Run main loop"""
        if not self.initialize():
            return 1

        while self._running:
            # Clean up old processes
            self.process_manager.cleanup_old_processes(
                self.config.backend_port,
                self.config.frontend_port
            )

            # Start backend
            if not self.process_manager.start_backend(self.config):
                self.logger.error("Failed to start backend")
                self.logger.error("Press Enter to exit...")
                try:
                    input()
                except:
                    pass
                return 1

            # Start frontend (dev mode)
            if self.config.dev_mode:
                if not self.process_manager.start_frontend(self.config):
                    self.logger.warning("Failed to start frontend, continuing anyway...")

            # Show startup info
            self.logger.info("")
            self.logger.info("=" * 50)
            self.logger.info("Services Started Successfully!")
            self.logger.info("=" * 50)

            if self.config.dev_mode:
                self.logger.info(f"Frontend: http://localhost:{self.config.frontend_port}")
            self.logger.info(f"Application: http://localhost:{self.config.backend_port}")
            self.logger.info(f"API Docs: http://localhost:{self.config.backend_port}/docs")
            self.logger.info("")
            self.logger.info("Press Ctrl+C to stop, or use 'Exit System' button in the app")
            self.logger.info("=" * 50)

            # Wait for backend to exit
            exit_code = self.process_manager.wait_for_backend_exit()
            self.logger.info(f"Backend exited with code: {exit_code}")

            # Check update signal
            update_signal = self.update_manager.check_update_signal()

            if update_signal:
                # Execute update
                self.logger.info("Update signal detected, starting update process...")

                # Terminate frontend
                self.process_manager.terminate_all()

                # Execute update
                if self.update_manager.execute_update(update_signal):
                    self.logger.info("Update successful, restarting...")
                    self.update_manager.clear_update_signal()
                    # Continue loop, restart services
                    time.sleep(2)
                    continue
                else:
                    self.logger.error("Update failed, restarting with current version...")
                    self.update_manager.clear_update_signal()
                    time.sleep(2)
                    continue
            else:
                # Normal exit
                self.logger.info("No update signal, normal shutdown")
                self._running = False

        # Cleanup
        if self.process_manager:
            self.process_manager.terminate_all()

        self.logger.info("Bootloader shutdown complete")
        return 0


def main():
    """Main entry"""
    # Set console encoding
    if os.name == 'nt':
        os.system('chcp 65001 >nul 2>&1')

    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'

    bootloader = Bootloader()
    sys.exit(bootloader.run())


if __name__ == "__main__":
    main()
