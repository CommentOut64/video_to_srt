# -*- coding: utf-8 -*-
"""
根级 pytest 配置。

职责:
- 注册自定义 markers
- 注册 CLI 选项 (--test-video / --real-engines / --reference)
- CI 环境 stub 重型模块
- 共享路径与 sys.path 设置
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

# ============================================================
# 路径设置
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"

# 确保 backend 在 sys.path 中
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ============================================================
# CI 环境 stub（统一收编，替代各测试文件散落的 sys.modules hack）
# ============================================================
_CI_STUB_MODULES = [
    "torch",
    "torchaudio",
    "soundfile",
    "onnxruntime",
    "demucs",
    "demucs.pretrained",
    "demucs.apply",
    "librosa",
    "brouhaha",
    "brouhaha.pipeline",
    "speechbrain",
    "pyannote",
    "pyannote.audio",
    "silero_vad",
    "av",
]

_INTEGRATION_PROFILE = os.environ.get("INTEGRATION_PROFILE", "").lower()


def _is_module_available(module_name: str) -> bool:
    """检测模块是否可导入（不触发真实导入）。"""
    return importlib.util.find_spec(module_name) is not None


def _install_ci_stubs() -> None:
    """在 CI 环境中，为不可用的重型模块安装空 stub。"""
    for mod_name in _CI_STUB_MODULES:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    # fastapi 特殊处理：仅在环境未安装 fastapi 时注入最小 stub，避免覆盖真实 FastAPI
    if "fastapi" not in sys.modules and not _is_module_available("fastapi"):
        fastapi_stub = types.ModuleType("fastapi")

        class DummyRequest:
            async def is_disconnected(self) -> bool:
                return False

        fastapi_stub.Request = DummyRequest  # type: ignore[attr-defined]
        sys.modules["fastapi"] = fastapi_stub

    # soundfile stub 需要空的 read/write
    sf_mod = sys.modules.get("soundfile")
    if sf_mod and not hasattr(sf_mod, "read"):
        sf_mod.read = lambda *a, **kw: (None, 16000)  # type: ignore[attr-defined]
        sf_mod.write = lambda *a, **kw: None  # type: ignore[attr-defined]


if _INTEGRATION_PROFILE == "ci":
    _install_ci_stubs()


# ============================================================
# CLI 选项注册
# ============================================================
def pytest_addoption(parser: pytest.Parser) -> None:
    """注册自定义命令行选项。"""
    parser.addoption(
        "--test-video",
        action="store",
        default=None,
        help="测试视频路径（启用真实预处理测试）",
    )
    parser.addoption(
        "--real-engines",
        action="store_true",
        default=False,
        help="使用真实 ASR 引擎而非 Mock（需要 GPU + 模型）",
    )
    parser.addoption(
        "--reference",
        action="store",
        default=None,
        help="参考 SRT 文件路径（质量评估用）",
    )
    parser.addoption(
        "--hypothesis",
        action="store",
        default=None,
        help="待评估 SRT 文件路径（质量评估用）",
    )
    parser.addoption(
        "--profile",
        action="store",
        default="sv_whisper_dual",
        help="转录模式 (sensevoice_only / sv_whisper_patch / sv_whisper_dual)",
    )


# ============================================================
# Fixture: 测试配置
# ============================================================
@pytest.fixture
def test_video_path(request: pytest.FixtureRequest) -> Path | None:
    """获取测试视频路径。"""
    val = request.config.getoption("--test-video")
    return Path(val) if val else None


@pytest.fixture
def use_real_engines(request: pytest.FixtureRequest) -> bool:
    """是否使用真实 ASR 引擎。"""
    return bool(request.config.getoption("--real-engines"))


@pytest.fixture
def reference_srt_path(request: pytest.FixtureRequest) -> Path | None:
    """获取参考 SRT 文件路径。"""
    val = request.config.getoption("--reference")
    return Path(val) if val else None


@pytest.fixture
def hypothesis_srt_path(request: pytest.FixtureRequest) -> Path | None:
    """获取待评估 SRT 文件路径。"""
    val = request.config.getoption("--hypothesis")
    return Path(val) if val else None


@pytest.fixture
def transcription_profile(request: pytest.FixtureRequest) -> str:
    """获取转录模式。"""
    return str(request.config.getoption("--profile"))
