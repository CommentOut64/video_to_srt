# -*- coding: utf-8 -*-
"""
启动器配置模块
V3.2.4+dev.20260303.04
"""

import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict


# ========================================
# 常量
# ========================================
VERSION = "3.2.4+dev.20260303.04"
APP_NAME = "AnchorFlux"
DEFAULT_BACKEND_PORT = 8000
DEFAULT_FRONTEND_PORT = 5173
DEFAULT_UI_MODE = "browser"
DEFAULT_RUNTIME_POLICY = "offline"
DEFAULT_FLAVOR = "full"
DEFAULT_MEDIA_PROFILE = "electron_native"
DEFAULT_GPU_MODE = "auto"
VALID_UI_MODES = ("browser", "electron", "none")
VALID_RUNTIME_POLICIES = ("offline", "hybrid")
VALID_FLAVORS = ("full", "lite")
VALID_MEDIA_PROFILES = ("browser_compat", "electron_native", "lite_safe")
VALID_GPU_MODES = ("auto", "prefer_dgpu", "prefer_igpu", "safe")

# 信号文件
UPDATE_SIGNAL_FILE = "update_signal.json"
SHUTDOWN_SIGNAL_FILE = ".shutdown_signal"

# 日志格式
LOG_FORMAT_DEV = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_FORMAT_PROD = "%(asctime)s [%(levelname)s] %(message)s"
logger = logging.getLogger("launcher.config")


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

    # UI / 运行时策略配置
    ui_mode: str = DEFAULT_UI_MODE
    runtime_policy: str = DEFAULT_RUNTIME_POLICY
    flavor: str = DEFAULT_FLAVOR
    shell_path: Optional[Path] = None
    media_profile: str = DEFAULT_MEDIA_PROFILE
    gpu_mode: str = DEFAULT_GPU_MODE

    def __post_init__(self):
        """初始化后处理"""
        if self.tools_dir is None:
            self.tools_dir = self.project_root / "tools"
        self.ui_mode = normalize_ui_mode(self.ui_mode)
        self.runtime_policy = normalize_runtime_policy(self.runtime_policy)
        self.flavor = normalize_flavor(self.flavor)
        self.media_profile = normalize_media_profile(self.media_profile)
        self.gpu_mode = normalize_gpu_mode(self.gpu_mode)


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
    except OSError as exc:
        logger.warning("读取 .env 失败: %s", exc)

    return config


def normalize_ui_mode(raw_value: str) -> str:
    """归一化 UI 模式值。"""
    normalized = str(raw_value or "").strip().lower()
    if normalized in VALID_UI_MODES:
        return normalized
    return DEFAULT_UI_MODE


def normalize_runtime_policy(raw_value: str) -> str:
    """归一化运行策略值。"""
    normalized = str(raw_value or "").strip().lower()
    if normalized in VALID_RUNTIME_POLICIES:
        return normalized
    return DEFAULT_RUNTIME_POLICY


def normalize_flavor(raw_value: str) -> str:
    """归一化产品形态值。"""
    normalized = str(raw_value or "").strip().lower()
    if normalized in VALID_FLAVORS:
        return normalized
    return DEFAULT_FLAVOR


def normalize_media_profile(raw_value: str) -> str:
    """归一化媒体 profile。"""
    normalized = str(raw_value or "").strip().lower()
    if normalized in VALID_MEDIA_PROFILES:
        return normalized
    return DEFAULT_MEDIA_PROFILE


def normalize_gpu_mode(raw_value: str) -> str:
    """归一化 GPU 模式。"""
    normalized = str(raw_value or "").strip().lower()
    legacy_aliases = {
        "prefer_hardware": "prefer_dgpu",
        "off": "safe",
    }
    normalized = legacy_aliases.get(normalized, normalized)
    if normalized in VALID_GPU_MODES:
        return normalized
    return DEFAULT_GPU_MODE


def resolve_media_profile(ui_mode: str, flavor: str, dev_mode: bool, raw_value: str = "") -> str:
    """根据运行模式解析默认媒体 profile。"""
    normalized = normalize_media_profile(raw_value)
    if str(raw_value or "").strip():
        return normalized
    if dev_mode or normalize_ui_mode(ui_mode) != "electron":
        return "browser_compat"
    if normalize_flavor(flavor) == "lite":
        return "lite_safe"
    return "electron_native"


def resolve_gpu_mode(ui_mode: str, flavor: str, media_profile: str, raw_value: str = "") -> str:
    """根据运行模式解析默认 GPU 模式。"""
    normalized = normalize_gpu_mode(raw_value)
    if str(raw_value or "").strip():
        return normalized
    if normalize_ui_mode(ui_mode) != "electron":
        return "auto"
    # 设计取舍：Lite 默认通过 lite_safe 媒体链降低解码压力，GPU 模式仍保持 auto，
    # 只有用户显式选择 safe 时才彻底关闭硬件加速，避免默认策略过于保守。
    return "auto"


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
