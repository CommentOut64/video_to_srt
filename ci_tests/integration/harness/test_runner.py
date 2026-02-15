# -*- coding: utf-8 -*-
"""
集成测试运行器。

核心运行器，直接实例化 AsyncDualPipeline 绕过重依赖链，
提供无模型和有模型两种运行模式。
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.engines.dummy_engine import DummyEngine
from app.engines.factory import ASREngineFactory
from app.models.sensevoice_models import SentenceSegment
from app.pipelines.async_dual_pipeline import AsyncDualPipeline
from app.services.audio.chunk_engine import AudioChunk
from app.services.bridge.bridge_controller import BridgeController
from app.services.bridge.config import BridgeConfig
from app.services.streaming_subtitle import (
    get_streaming_subtitle_manager,
    remove_streaming_subtitle_manager,
)
from app.utils.text_utils import segments_to_srt

from .audio_factory import AudioFactory
from .mock_engines import MockEngineFactory, RecordingWhisperEngine
from .mock_services import (
    NoOpProgressEmitter,
    StubAligner,
    StubPunctuationService,
)


@dataclass
class PipelineTestConfig:
    """流水线测试配置。"""

    transcription_profile: str = "sv_whisper_dual"
    use_real_preprocessing: bool = False
    use_real_engines: bool = False
    video_path: Optional[Path] = None
    draft_text: str = "测试文本"
    slow_segments: Optional[List[dict]] = None
    chunk_count: int = 3
    chunk_duration: float = 2.0
    enable_bridge: bool = True
    language: str = "zh"
    sample_rate: int = 16000


@dataclass
class PipelineTestResult:
    """流水线测试结果。"""

    success: bool = False
    contexts: List[Any] = field(default_factory=list)
    sentences: List[SentenceSegment] = field(default_factory=list)
    srt_content: str = ""
    stage_timings: Dict[str, float] = field(default_factory=dict)
    stage_details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None


class IntegrationTestRunner:
    """集成测试运行器。

    直接实例化 AsyncDualPipeline，绕过 PipelineOrchestrator 的重依赖链。
    """

    def __init__(self, config: PipelineTestConfig, work_dir: Optional[Path] = None) -> None:
        self.config = config
        self.work_dir = work_dir
        self._job_id = f"test-{int(time.time() * 1000)}"

    async def run_full_pipeline(self) -> PipelineTestResult:
        """运行完整测试流水线。"""
        result = PipelineTestResult()
        t_start = time.perf_counter()

        try:
            # 构建音频 chunks
            t0 = time.perf_counter()
            chunks, full_audio, sr = self._build_audio()
            result.stage_timings["audio_setup"] = (time.perf_counter() - t0) * 1000

            # 构建引擎
            t0 = time.perf_counter()
            draft_engine, patch_engine = self._build_engines()
            result.stage_timings["engine_setup"] = (time.perf_counter() - t0) * 1000

            # 构建流水线
            t0 = time.perf_counter()
            pipeline = self._build_pipeline(draft_engine, patch_engine)
            result.stage_timings["pipeline_setup"] = (time.perf_counter() - t0) * 1000

            # 运行流水线
            t0 = time.perf_counter()
            contexts = await pipeline.run(
                audio_chunks=chunks,
                full_audio_array=full_audio,
                full_audio_sr=sr,
            )
            result.stage_timings["pipeline_run"] = (time.perf_counter() - t0) * 1000

            result.contexts = contexts

            # 收集句子
            t0 = time.perf_counter()
            sentences: List[SentenceSegment] = []
            for ctx in contexts:
                # 优先从 final_sentences 获取
                if getattr(ctx, "final_sentences", []):
                    sentences.extend(ctx.final_sentences)
                # sensevoice_only 模式下从 sv_result 构建句子
                elif self.config.transcription_profile == "sensevoice_only" and ctx.sv_result:
                    sentences.extend(self._build_sentences_from_sv_result(ctx))
            result.sentences = sentences
            result.stage_timings["sentence_collect"] = (time.perf_counter() - t0) * 1000

            # 生成 SRT
            t0 = time.perf_counter()
            result.srt_content = self._build_srt(sentences)
            result.stage_timings["srt_build"] = (time.perf_counter() - t0) * 1000

            result.success = True

        except Exception as e:
            result.error = e
            result.success = False

        finally:
            # 清理
            try:
                remove_streaming_subtitle_manager(self._job_id)
            except Exception:
                pass
            result.stage_timings["total"] = (time.perf_counter() - t_start) * 1000

        return result

    def _build_audio(self):
        """构建测试音频数据。"""
        if self.config.use_real_preprocessing and self.config.video_path:
            # 真实预处理路径 — 需要外部实现
            raise NotImplementedError("真实预处理需要 FFmpeg/VAD，暂未集成到 TestRunner")

        # Dummy 模式：直接构造静音 chunks
        chunks = AudioFactory.create_silent_chunks(
            count=self.config.chunk_count,
            duration=self.config.chunk_duration,
            sample_rate=self.config.sample_rate,
            language=self.config.language,
        )
        full_audio, sr = AudioFactory.create_full_audio_from_chunks(chunks)
        return chunks, full_audio, sr

    def _build_engines(self):
        """构建 ASR 引擎。"""
        if self.config.use_real_engines:
            # Why: 真实集成测试需要验证生产级引擎装配路径，直接复用工厂注册避免测试侧分叉实现。
            draft_engine = ASREngineFactory.create(
                "sensevoice",
                language=self.config.language,
            )
            patch_engine = None
            if self.config.transcription_profile != "sensevoice_only":
                patch_engine = ASREngineFactory.create(
                    "whisper",
                    device="cuda",
                )
            return draft_engine, patch_engine

        # Dummy 模式
        draft_engine = MockEngineFactory.create_dummy_draft(
            text=self.config.draft_text,
            latency_ms=0,
        )

        patch_engine = None
        if self.config.transcription_profile != "sensevoice_only":
            slow_segments = self.config.slow_segments or [
                {"start": 0.0, "end": 1.0, "text": "慢流文本A"},
                {"start": 1.0, "end": 2.0, "text": "慢流文本B"},
            ]
            patch_engine = MockEngineFactory.create_recording_whisper(slow_segments)

        return draft_engine, patch_engine

    def _build_pipeline(self, draft_engine, patch_engine) -> AsyncDualPipeline:
        """构建 AsyncDualPipeline 实例。"""
        punctuation_service = StubPunctuationService()

        # Bridge 配置
        bridge_controller = None
        if self.config.enable_bridge and self.config.transcription_profile == "sv_whisper_dual":
            max_wait_ms = max(50, int(self.config.chunk_duration * 1000))
            high_watermark = max(2, int(self.config.chunk_count))
            low_watermark = max(1, min(high_watermark - 1, 2))
            bridge_controller = BridgeController(
                config=BridgeConfig(
                    # Why: BridgeConfig 已收敛为 Phase 3 精简字段，这里按等价语义映射旧测试参数。
                    is_enable_dynamic_batch=False,
                    dynamic_batch_max_wait_ms=max_wait_ms,
                    queue_low_watermark=low_watermark,
                    queue_high_watermark=high_watermark,
                    force_flush_on_gpu_idle=True,
                )
            )

        pipeline = AsyncDualPipeline(
            job_id=self._job_id,
            transcription_profile=self.config.transcription_profile,
            draft_engine=draft_engine,
            patch_engine=patch_engine,
            punctuation_service=punctuation_service,
            # Why: 真实引擎集成测试需要覆盖 L4-L7 后处理链路，关闭语义缓冲以强制进入队列与对齐阶段。
            enable_semantic_buffer=not self.config.use_real_engines,
            bridge_controller=bridge_controller,
        )
        # 注入 stub 对齐器
        pipeline.aligner = StubAligner()

        return pipeline

    def _build_sentences_from_sv_result(self, ctx) -> List[SentenceSegment]:
        """从 sv_result 构建句子（sensevoice_only 模式）。"""
        sv_result = ctx.sv_result
        chunk = ctx.audio_chunk
        if not sv_result:
            return []

        text = sv_result.get("text_clean") or sv_result.get("raw_text") or ""
        if not text:
            return []

        return [
            SentenceSegment(
                text=text,
                text_clean=text,
                start=chunk.start,
                end=chunk.end,
            )
        ]

    def _build_srt(self, sentences: List[SentenceSegment]) -> str:
        """从句子列表构建 SRT 字符串。"""
        if not sentences:
            return ""
        segments = [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text_clean or s.text,
            }
            for s in sentences
        ]
        return segments_to_srt(segments)
