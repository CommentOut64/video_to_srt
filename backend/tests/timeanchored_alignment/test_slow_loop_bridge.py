from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from app.engines.dummy_engine import DummyEngine
from app.pipelines.async_dual_pipeline import AsyncDualPipeline
from app.pipelines.dual_pipeline.services.slow_loop_service import SlowLoopService
from app.schemas.pipeline_context import ProcessingContext
from app.services.bridge.turn_group_builder import TurnGroupEnvelope
from app.services.bridge.turn_group_models import TurnGroup
from app.services.punctuation.semantic_buffer import SemanticChunk
from app.models.sensevoice_models import SentenceSegment
from app.services.timeanchored_alignment.slow_window.contracts import ReadySlowWindow


def _build_semantic_chunk(
    *,
    chunk_id: str,
    text: str,
    language: str,
    start: float,
    end: float,
    speaker_id: str = "spk-1",
) -> SemanticChunk:
    return SemanticChunk(
        chunk_id=chunk_id,
        text=text,
        sentences=[SentenceSegment(text=text, text_clean=text, start=start, end=end)],
        punctuation_result=None,
        punctuation_decision=None,
        pending_tail="",
        audio_range=(start, end),
        language=language,
        source_chunks=[chunk_id],
        speaker_id=speaker_id,
    )


@pytest.mark.asyncio
async def test_hetero_window_path_enqueues_ready_slow_window() -> None:
    pipeline = AsyncDualPipeline(
        job_id="job-phase2-hetero",
        draft_engine=DummyEngine(response_text="draft", latency_ms=0),
        patch_engine=DummyEngine(response_text="patch", latency_ms=0),
        transcription_profile="sv_whisper_dual",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=True,
        punctuation_service=Mock(),
    )
    chunk = _build_semantic_chunk(
        chunk_id="chunk-0",
        text="你好 world",
        language="mixed",
        start=0.0,
        end=2.2,
    )

    await pipeline._ingest_bridge_chunks([chunk])
    await pipeline._flush_bridge_controller()

    payload = await asyncio.wait_for(pipeline.queue_inter.get(), timeout=1.0)
    assert isinstance(payload, ReadySlowWindow)
    assert payload.window_id
    assert payload.source_chunk_ids == ("chunk-0",)


@pytest.mark.asyncio
async def test_bridge_runtime_path_builds_explicit_ingress_before_ready_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = AsyncDualPipeline(
        job_id="job-phase2-explicit-ingress",
        draft_engine=DummyEngine(response_text="draft", latency_ms=0),
        patch_engine=DummyEngine(response_text="patch", latency_ms=0),
        transcription_profile="sv_whisper_dual",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=True,
        punctuation_service=Mock(),
    )
    chunk = _build_semantic_chunk(
        chunk_id="chunk-5",
        text="你好",
        language="zh",
        start=0.0,
        end=4.3,
    )
    captured: dict[str, tuple[int, ...]] = {}

    original_adapt = pipeline._slow_window_ingress_adapter.adapt

    def _capture_adapt(*args, **kwargs):
        captured["source_chunk_indices"] = tuple(kwargs.get("source_chunk_indices") or ())
        return original_adapt(*args, **kwargs)

    monkeypatch.setattr(pipeline._slow_window_ingress_adapter, "adapt", _capture_adapt)
    assert not hasattr(pipeline, "_turn_group_builder")

    await pipeline._ingest_bridge_chunks([chunk])
    await pipeline._flush_bridge_controller()

    payload = await asyncio.wait_for(pipeline.queue_inter.get(), timeout=1.0)
    assert isinstance(payload, ReadySlowWindow)
    assert captured["source_chunk_indices"] == (5,)


@pytest.mark.asyncio
async def test_legacy_turn_group_envelope_path_is_rejected() -> None:
    host = SimpleNamespace(
        cancellation_token=None,
        queue_inter=asyncio.Queue(),
        queue_final=asyncio.Queue(),
        bridge_controller=None,
        _flush_window_assembler_idle=lambda now: None,
        logger=Mock(),
        progress_emitter=None,
        debug_punctuation=False,
        job_id="job-phase2-legacy",
        errors=[],
        pause_exception=None,
        _slow_processed_indices=set(),
        _last_slow_chunk_index=-1,
        _context_cache={},
    )

    envelope = TurnGroupEnvelope(
        group=TurnGroup(group_id="tg-legacy-1", speaker_id="spk-1", source_chunks=["chunk-0"]),
        sentences=[],
        punctuation_decision=None,
    )
    await host.queue_inter.put(envelope)
    await host.queue_inter.put(
        ProcessingContext(
            job_id="job-phase2-legacy",
            chunk_index=-1,
            audio_chunk=None,
            is_end=True,
        )
    )

    service = SlowLoopService(host=host)
    await service.run(job_dir=None, total_chunks=0)

    assert len(host.errors) == 1
    assert isinstance(host.errors[0], RuntimeError)
    assert "不再接受 TurnGroupEnvelope" in str(host.errors[0])
    terminal_ctx = await asyncio.wait_for(host.queue_final.get(), timeout=1.0)
    assert terminal_ctx.is_end is True
    assert isinstance(terminal_ctx.error, RuntimeError)
