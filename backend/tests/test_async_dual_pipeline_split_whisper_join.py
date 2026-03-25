"""
AsyncDualPipeline: Whisper 批次结果按 Chunk 拆分时的文本拼接测试。

该用例用于回归：
- 拆分时若使用 "" 直接拼接 segment.text，会导致段间空格丢失；
- 在英文场景会出现 "lamp.Who" 这类错误粘连，后续可能被误当作缩写点保护而放大。
"""

import sys
import types
import asyncio
from pathlib import Path
from unittest.mock import Mock

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

try:
    import fastapi  # noqa: F401
except ImportError:
    fastapi = None

if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")
if "soundfile" not in sys.modules:
    sys.modules["soundfile"] = types.ModuleType("soundfile")
if "fastapi" not in sys.modules:
    fastapi_stub = types.ModuleType("fastapi")

    class Request:  # noqa: D401 - 测试占位
        """FastAPI Request 占位类型。"""

    fastapi_stub.Request = Request
    sys.modules["fastapi"] = fastapi_stub

import numpy as np

from app.engines.dummy_engine import DummyEngine
from app.models.sensevoice_models import SentenceSegment
from app.pipelines.async_dual_pipeline import AsyncDualPipeline
from app.services.audio.chunk_engine import AudioChunk
from app.services.punctuation.semantic_buffer import SemanticChunk
from app.services.timeanchored_alignment.slow_window.contracts import ReadySlowWindow


def test_split_whisper_result_by_chunks_join_with_space() -> None:
    """确保拆分 Chunk 时 segment 文本以空格拼接，避免 lamp.Who 粘连。"""
    pipeline = AsyncDualPipeline(
        job_id="test_split_whisper_join",
        draft_engine=DummyEngine(response_text="draft", latency_ms=0),
        patch_engine=DummyEngine(response_text="patch", latency_ms=0),
        transcription_profile="sv_whisper_dual",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=False,
        punctuation_service=Mock(),
    )

    audio = np.zeros(16000, dtype=np.float32)
    chunk = AudioChunk(index=0, start=0.0, end=2.0, audio=audio, sample_rate=16000, language="en")
    pipeline._audio_chunks_by_index = {0: chunk}

    whisper_result = {
        "confidence": 0.5,
        "language": "en",
        "raw_result": {
            "segments": [
                {"start": 0.0, "end": 0.5, "text": "lamp.", "avg_logprob": -0.1, "no_speech_prob": 0.0, "words": []},
                {"start": 0.5, "end": 1.0, "text": "Who.", "avg_logprob": -0.1, "no_speech_prob": 0.0, "words": []},
            ]
        },
    }

    results = pipeline._split_whisper_result_by_chunks(
        whisper_result,
        chunk_indices=[0],
        batch_start=0.0,
        language_override="en",
    )

    assert 0 in results
    assert results[0]["text"] == "lamp. Who."
    assert results[0]["raw_text"] == "lamp. Who."
    assert results[0]["min_clean_text"] == "lamp. Who."


def test_split_whisper_result_by_chunks_word_level_cross_chunk_boundary() -> None:
    """跨 Chunk 的同一 segment 应按词级时间戳拆分，避免前缀串入下一 Chunk。"""
    pipeline = AsyncDualPipeline(
        job_id="test_split_whisper_word_level_boundary",
        draft_engine=DummyEngine(response_text="draft", latency_ms=0),
        patch_engine=DummyEngine(response_text="patch", latency_ms=0),
        transcription_profile="sv_whisper_dual",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=False,
        punctuation_service=Mock(),
    )

    audio = np.zeros(16000, dtype=np.float32)
    chunk_1 = AudioChunk(index=1, start=10.0, end=20.0, audio=audio, sample_rate=16000, language="en")
    chunk_2 = AudioChunk(index=2, start=20.0, end=30.0, audio=audio, sample_rate=16000, language="en")
    pipeline._audio_chunks_by_index = {1: chunk_1, 2: chunk_2}

    whisper_result = {
        "confidence": 0.5,
        "language": "en",
        "raw_result": {
            "segments": [
                {
                    "start": 0.0,
                    "end": 15.0,
                    "text": "Who is my favorite DDLC character? probably Monica, she has the right idea.",
                    "avg_logprob": -0.1,
                    "no_speech_prob": 0.0,
                    "words": [
                        {"word": "Who", "start": 1.0, "end": 1.2},
                        {"word": " is", "start": 1.2, "end": 1.35},
                        {"word": " my", "start": 1.35, "end": 1.5},
                        {"word": " favorite", "start": 1.5, "end": 1.9},
                        {"word": " DDLC", "start": 1.9, "end": 2.2},
                        {"word": " character?", "start": 2.2, "end": 2.7},
                        {"word": " probably", "start": 10.0, "end": 10.3},
                        {"word": " Monica,", "start": 10.3, "end": 10.7},
                        {"word": " she", "start": 10.7, "end": 10.95},
                        {"word": " has", "start": 10.95, "end": 11.2},
                        {"word": " the", "start": 11.2, "end": 11.35},
                        {"word": " right", "start": 11.35, "end": 11.65},
                        {"word": " idea.", "start": 11.65, "end": 12.0},
                    ],
                }
            ]
        },
    }

    results = pipeline._split_whisper_result_by_chunks(
        whisper_result,
        chunk_indices=[1, 2],
        batch_start=10.0,
        language_override="en",
    )

    assert results[1]["raw_text"] == "Who is my favorite DDLC character?"
    assert results[2]["raw_text"] == "probably Monica, she has the right idea."
    assert results[1]["text"] == "Who is my favorite DDLC character?"
    assert results[2]["text"] == "probably Monica, she has the right idea."


def test_split_whisper_result_by_chunks_keeps_cjk_tail_on_original_chunk() -> None:
    """慢流整窗回切只按时间重叠拆分，不在这里做中文尾词重平衡。"""
    pipeline = AsyncDualPipeline(
        job_id="test_split_whisper_cjk_keep_original_chunk",
        draft_engine=DummyEngine(response_text="draft", latency_ms=0),
        patch_engine=DummyEngine(response_text="patch", latency_ms=0),
        transcription_profile="sv_whisper_dual",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=False,
        punctuation_service=Mock(),
    )

    audio = np.zeros(16000, dtype=np.float32)
    chunk_1 = AudioChunk(index=1, start=100.0, end=105.0, audio=audio, sample_rate=16000, language="zh")
    chunk_2 = AudioChunk(index=2, start=105.0, end=110.0, audio=audio, sample_rate=16000, language="zh")
    pipeline._audio_chunks_by_index = {1: chunk_1, 2: chunk_2}

    whisper_result = {
        "confidence": 0.6,
        "language": "zh",
        "raw_result": {
            "segments": [
                {
                    "start": 0.0,
                    "end": 7.0,
                    "text": "非常难喝于是赶紧用自来水冲洗口腔",
                    "avg_logprob": -0.1,
                    "no_speech_prob": 0.0,
                    "words": [
                        {"word": "非常", "start": 0.2, "end": 0.7},
                        {"word": "难喝", "start": 0.7, "end": 1.4},
                        {"word": "于", "start": 4.82, "end": 4.98},
                        {"word": "是", "start": 5.02, "end": 5.16},
                        {"word": "赶紧", "start": 5.16, "end": 5.8},
                        {"word": "用", "start": 5.8, "end": 6.0},
                        {"word": "自来水", "start": 6.0, "end": 6.6},
                        {"word": "冲洗口腔", "start": 6.6, "end": 6.95},
                    ],
                }
            ]
        },
    }

    results = pipeline._split_whisper_result_by_chunks(
        whisper_result,
        chunk_indices=[1, 2],
        batch_start=100.0,
        language_override="zh",
    )

    assert results[1]["raw_text"] == "非常难喝于"
    assert results[2]["raw_text"].startswith("是赶紧用自来水冲洗口腔")


def test_flush_semantic_buffer_still_flushes_window_builder_when_semantic_tail_empty() -> None:
    """语义缓冲尾部无新增 chunk 时，也必须执行 window-first EOF flush。"""
    pipeline = AsyncDualPipeline(
        job_id="test_semantic_tail_empty_bridge_flush",
        draft_engine=DummyEngine(response_text="draft", latency_ms=0),
        patch_engine=DummyEngine(response_text="patch", latency_ms=0),
        transcription_profile="sv_whisper_dual",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=True,
        punctuation_service=Mock(),
    )

    pending_chunk = SemanticChunk(
        chunk_id="chunk-6",
        text="尾部测试",
        sentences=[
            SentenceSegment(
                text="尾部测试",
                text_clean="尾部测试",
                start=6.0,
                end=6.8,
            )
        ],
        punctuation_result=None,
        punctuation_decision=None,
        pending_tail="",
        audio_range=(6.0, 6.8),
        language="zh",
        source_chunks=["chunk-6"],
        speaker_id="spk-6",
    )
    audio = np.zeros(16000, dtype=np.float32)
    pipeline._audio_chunks_by_index = {
        6: AudioChunk(index=6, start=6.0, end=6.8, audio=audio, sample_rate=16000, language="zh"),
    }

    asyncio.run(pipeline._ingest_bridge_chunks([pending_chunk]))

    pipeline.semantic_buffer.flush = Mock(return_value=[])  # type: ignore[method-assign]

    async def _run_and_get_payload() -> ReadySlowWindow:
        await pipeline._flush_semantic_buffer(chunk_index=6)
        assert pipeline.queue_inter.qsize() == 1
        payload = await pipeline.queue_inter.get()
        assert isinstance(payload, ReadySlowWindow)
        return payload

    ready_window = asyncio.run(_run_and_get_payload())

    pipeline.semantic_buffer.flush.assert_called_once_with(reason="pipeline_end")
    assert ready_window.flush_reason == "eof_flush"
    assert ready_window.source_chunk_ids == ("chunk-6",)



