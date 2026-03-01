# -*- coding: utf-8 -*-
"""
集成测试 Harness 框架。

提供统一的测试基础设施，包括：
- SRT 解析和比较
- Mock 引擎工厂
- Mock 服务集合
- 音频测试数据工厂
- 报告构建器
- 集成测试运行器
"""
import sys
from pathlib import Path

# 确保 backend 目录在 Python 路径中
_project_root = Path(__file__).resolve().parent.parent.parent.parent
_backend_path = _project_root / "backend"
if str(_backend_path) not in sys.path:
    sys.path.insert(0, str(_backend_path))

from .srt_parser import SRTParser, SRTEntry
from .srt_comparator import SRTComparator, ComparisonResult
from .mock_engines import MockEngineFactory, RecordingWhisperEngine, ConfigurableEngine
from .mock_services import (
    StubPunctuationService,
    StubAligner,
    StubScheduler,
    NoOpProgressEmitter,
)
from .audio_factory import AudioFactory
from .report_builder import ReportBuilder, StageReport

# test_runner 依赖完整后端运行时（prometheus_client 等），延迟加载以避免轻量 CI 环境报错
def __getattr__(name: str):
    if name in ("IntegrationTestRunner", "PipelineTestConfig", "PipelineTestResult"):
        from .test_runner import IntegrationTestRunner, PipelineTestConfig, PipelineTestResult
        return {"IntegrationTestRunner": IntegrationTestRunner,
                "PipelineTestConfig": PipelineTestConfig,
                "PipelineTestResult": PipelineTestResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SRTParser",
    "SRTEntry",
    "SRTComparator",
    "ComparisonResult",
    "MockEngineFactory",
    "RecordingWhisperEngine",
    "ConfigurableEngine",
    "StubPunctuationService",
    "StubAligner",
    "StubScheduler",
    "NoOpProgressEmitter",
    "AudioFactory",
    "ReportBuilder",
    "StageReport",
    "IntegrationTestRunner",
    "PipelineTestConfig",
    "PipelineTestResult",
]
