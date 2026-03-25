# -*- coding: utf-8 -*-
"""
本地全流程集成测试。

覆盖三种场景：
1. TestMinimalDummyPipeline - 无模型依赖的最小测试
2. TestFullDummyPipeline - 全阶段 Dummy 测试
3. TestRealVideoIntegration - 真实视频测试（需 --test-video）

使用命令：
    # 最小 Dummy 流程
    pytest tests/integration/test_local_pipeline.py -v

    # 指定视频
    pytest tests/integration/test_local_pipeline.py -v --test-video="path/to/video.mp4"

    # 完整流程（需 GPU + 模型）
    pytest tests/integration/test_local_pipeline.py -v --test-video="path/to/video.mp4" --real-engines
"""
from __future__ import annotations

import pytest

from ci_tests.integration.harness import (
    AudioFactory,
    IntegrationTestRunner,
    MockEngineFactory,
    PipelineTestConfig,
    ReportBuilder,
    SRTComparator,
    SRTParser,
    StageReport,
)


# ============================================================
# TestMinimalDummyPipeline - 无模型依赖
# ============================================================
class TestMinimalDummyPipeline:
    """最小 Dummy 流水线测试，无模型依赖。"""

    @pytest.mark.integration_ci
    @pytest.mark.asyncio
    async def test_sensevoice_only_profile(self):
        """极速模式：仅快流。"""
        config = PipelineTestConfig(
            transcription_profile="sensevoice_only",
            chunk_count=2,
            chunk_duration=1.0,
            draft_text="极速模式测试",
            enable_bridge=False,
        )
        runner = IntegrationTestRunner(config)
        result = await runner.run_full_pipeline()

        assert result.success, f"流水线失败: {result.error}"
        assert len(result.contexts) == 2
        assert result.srt_content, "SRT 内容不应为空"

    @pytest.mark.integration_ci
    @pytest.mark.asyncio
    async def test_sv_whisper_patch_profile(self):
        """复核模式：快流 + 补刀。"""
        config = PipelineTestConfig(
            transcription_profile="sv_whisper_patch",
            chunk_count=2,
            chunk_duration=1.0,
            draft_text="复核模式测试",
            slow_segments=[
                {"start": 0.0, "end": 0.5, "text": "补刀A"},
                {"start": 0.5, "end": 1.0, "text": "补刀B"},
            ],
            enable_bridge=False,
        )
        runner = IntegrationTestRunner(config)
        result = await runner.run_full_pipeline()

        assert result.success, f"流水线失败: {result.error}"
        assert len(result.contexts) == 1
        ready_window = result.contexts[0].ready_slow_window
        assert ready_window is not None
        assert tuple(ready_window.source_chunk_indices) == (0, 1)

    @pytest.mark.integration_ci
    @pytest.mark.asyncio
    async def test_sv_whisper_dual_profile(self):
        """双流模式：快流 + Bridge + 慢流。"""
        config = PipelineTestConfig(
            transcription_profile="sv_whisper_dual",
            chunk_count=3,
            chunk_duration=2.0,
            draft_text="双流模式测试",
            slow_segments=[
                {"start": 0.0, "end": 1.0, "text": "慢流A"},
                {"start": 1.0, "end": 2.0, "text": "慢流B"},
            ],
            enable_bridge=True,
        )
        runner = IntegrationTestRunner(config)
        result = await runner.run_full_pipeline()

        assert result.success, f"流水线失败: {result.error}"
        assert len(result.contexts) == 1
        ready_window = result.contexts[0].ready_slow_window
        assert ready_window is not None
        assert tuple(ready_window.source_chunk_indices) == (0, 1, 2)


# ============================================================
# TestFullDummyPipeline - 全阶段 Dummy 测试
# ============================================================
class TestFullDummyPipeline:
    """全阶段 Dummy 流水线测试。"""

    @pytest.mark.integration_ci
    @pytest.mark.asyncio
    async def test_dual_stream_with_bridge(self):
        """双流模式：快流 -> Bridge -> 慢流完整流程。"""
        config = PipelineTestConfig(
            transcription_profile="sv_whisper_dual",
            chunk_count=4,
            chunk_duration=2.0,
            draft_text="快流草稿文本",
            slow_segments=[
                {"start": 0.0, "end": 1.0, "text": "慢流定稿A"},
                {"start": 1.0, "end": 2.0, "text": "慢流定稿B"},
            ],
            enable_bridge=True,
        )
        runner = IntegrationTestRunner(config)
        result = await runner.run_full_pipeline()

        assert result.success, f"流水线失败: {result.error}"
        assert len(result.contexts) == 1
        ready_window = result.contexts[0].ready_slow_window
        assert ready_window is not None
        assert tuple(ready_window.source_chunk_indices) == (0, 1, 2, 3)
        assert "pipeline_run" in result.stage_timings

    @pytest.mark.integration_ci
    @pytest.mark.asyncio
    async def test_srt_output_format(self):
        """SRT 格式校验。"""
        config = PipelineTestConfig(
            transcription_profile="sensevoice_only",
            chunk_count=3,
            chunk_duration=1.5,
            draft_text="SRT格式测试",
            enable_bridge=False,
        )
        runner = IntegrationTestRunner(config)
        result = await runner.run_full_pipeline()

        assert result.success, f"流水线失败: {result.error}"
        assert result.srt_content, "SRT 内容不应为空"

        # 解析 SRT 验证格式
        entries = SRTParser.parse_content(result.srt_content)
        assert len(entries) > 0, "SRT 应包含至少一个条目"

        for entry in entries:
            assert entry.start >= 0, "开始时间应非负"
            assert entry.end > entry.start, "结束时间应大于开始时间"
            assert entry.text, "文本不应为空"

    @pytest.mark.integration_ci
    @pytest.mark.asyncio
    async def test_report_generation(self):
        """测试报告生成功能。"""
        config = PipelineTestConfig(
            transcription_profile="sensevoice_only",
            chunk_count=2,
            chunk_duration=1.0,
            draft_text="报告测试",
        )
        runner = IntegrationTestRunner(config)
        result = await runner.run_full_pipeline()

        # 构建报告
        stages = [
            StageReport(stage_name="audio_setup", status="passed", duration_ms=10.0),
            StageReport(stage_name="pipeline_run", status="passed", duration_ms=100.0),
        ]
        report = ReportBuilder.build_integration_report(result, config, stages)

        assert report["success"] is True
        assert "stages" in report
        assert len(report["stages"]) == 2

        # 格式化文本报告
        text_report = ReportBuilder.format_text_report(report)
        assert "集成测试报告" in text_report


# ============================================================
# TestRealVideoIntegration - 真实视频测试
# ============================================================
class TestRealVideoIntegration:
    """真实视频集成测试，需要 --test-video 参数。"""

    @pytest.mark.slow
    @pytest.mark.local_integration
    @pytest.mark.asyncio
    async def test_real_preprocessing_dummy_transcription(self, test_video_path):
        """真实预处理 + Dummy 转录。"""
        if test_video_path is None:
            pytest.skip("需要 --test-video 参数")

        # 此测试需要真实预处理实现
        pytest.skip("真实预处理尚未集成到 TestRunner")

    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.local_integration
    @pytest.mark.asyncio
    async def test_real_full_pipeline(self, test_video_path, use_real_engines):
        """完整真实流水线（需 GPU + 模型）。"""
        if test_video_path is None:
            pytest.skip("需要 --test-video 参数")
        if not use_real_engines:
            pytest.skip("需要 --real-engines 参数")
        config = PipelineTestConfig(
            transcription_profile="sv_whisper_dual",
            use_real_engines=True,
            video_path=test_video_path,
            chunk_count=1,
            chunk_duration=2.0,
            enable_bridge=True,
            language="zh",
        )
        runner = IntegrationTestRunner(config)
        result = await runner.run_full_pipeline()

        assert result.success, f"真实流水线失败: {result.error}"
        assert len(result.contexts) == config.chunk_count
        assert result.srt_content, "真实流水线应输出非空 SRT"
        ctx = result.contexts[0]
        finalization_metrics = getattr(ctx, "finalization_metrics", {}) or {}
        # 新四层执行证据：集合层/评分层/裁决层/输出层
        assert "fact_word_count" in finalization_metrics
        assert "evidence_pause_anchor_count" in finalization_metrics
        assert "split_decision_count" in finalization_metrics
        assert "l7_error_count" in finalization_metrics
