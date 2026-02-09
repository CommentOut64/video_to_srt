# -*- coding: utf-8 -*-
"""
启动器配置模块
V3.2.0+dev.20260209.01
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


# ========================================
# 常量
# ========================================
VERSION = "3.2.0+dev.20260209.01"
APP_NAME = "AnchorFlux"
DEFAULT_BACKEND_PORT = 8000
DEFAULT_FRONTEND_PORT = 5173

# 信号文件
UPDATE_SIGNAL_FILE = "update_signal.json"
SHUTDOWN_SIGNAL_FILE = ".shutdown_signal"

# 日志格式
LOG_FORMAT_DEV = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_FORMAT_PROD = "%(asctime)s [%(levelname)s] %(message)s"


def get_project_root() -> Path:
    """
    获取项目根目录

    支持多种运行环境：
    - 源码运行：__file__ 所在目录的父目录
    - PyInstaller onedir：sys.executable 所在目录
    - Go Stub 启动：通过环境变量传递
    """
    # 优先使用环境变量（Go Stub 会设置）
    if "ANCHORFLUX_ROOT" in os.environ:
        return Path(os.environ["ANCHORFLUX_ROOT"]).resolve()

    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后
        # onedir 模式：exe 在 core/ 目录下，项目根目录是其父目录
        exe_dir = Path(sys.executable).parent
        if exe_dir.name == "core":
            return exe_dir.parent.resolve()
        return exe_dir.resolve()
    else:
        # 源码运行：launcher/ 目录的父目录
        return Path(__file__).parent.parent.resolve()


@dataclass
class LauncherConfig:
    """启动器配置"""
    project_root: Path
    dev_mode: bool = False
    backend_port: int = DEFAULT_BACKEND_PORT
    frontend_port: int = DEFAULT_FRONTEND_PORT

    # Python 环境配置
    python_exec: Optional[Path] = None
    python_mode: str = "unknown"  # "embedded", "venv", "system"
    site_packages: Optional[Path] = None

    # uv 配置
    uv_exec: Optional[Path] = None

    # 工具路径
    tools_dir: Optional[Path] = None
    ffmpeg_path: Optional[Path] = None

    # 环境变量
    hf_mirror: bool = True

    # 日志级别
    log_level: str = "INFO"

    # 更新配置
    update_url: str = ""
    check_update_on_start: bool = True

    def __post_init__(self):
        """初始化后处理"""
        if self.tools_dir is None:
            self.tools_dir = self.project_root / "tools"


def load_env_config(project_root: Path) -> Dict[str, str]:
    """加载 .env 配置文件"""
    config = {}
    env_file = project_root / '.env'

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


def detect_dev_mode(project_root: Path) -> bool:
    """检测开发模式"""
    # 1. 命令行参数
    if '--dev' in sys.argv:
        return True

    # 2. 环境变量
    if os.environ.get('DEV_MODE', '').lower() in ('true', '1', 'yes'):
        return True

    # 3. .env 文件
    env_config = load_env_config(project_root)
    if env_config.get('DEV_MODE', '').lower() in ('true', '1', 'yes'):
        return True

    return False
