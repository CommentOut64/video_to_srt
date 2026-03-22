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
async def test_hetero_window_path_can_enqueue_turn_group_envelope() -> None:
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
    assert isinstance(payload, TurnGroupEnvelope)
    assert payload.group.metadata.get("window_id")


@pytest.mark.asyncio
async def test_legacy_turn_group_envelope_path_still_works() -> None:
    processed_payloads: list[TurnGroupEnvelope] = []

    async def _process_turn_group(*args, **kwargs):
        processed_payloads.append(args[0])
        return False

    host = SimpleNamespace(
        cancellation_token=None,
        queue_inter=asyncio.Queue(),
        queue_final=asyncio.Queue(),
        bridge_controller=None,
        _turn_group_builder=SimpleNamespace(flush_idle=lambda now: None),
        _flush_window_assembler_idle=lambda now: None,
        _is_turn_group_committed=lambda _group_id: False,
        _record_turn_group_unit=lambda **kwargs: None,
        _process_turn_group=_process_turn_group,
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

    assert len(processed_payloads) == 1
    assert processed_payloads[0].group.group_id == "tg-legacy-1"


@pytest.mark.asyncio
async def test_mixed_window_group_is_safely_fallback_in_process_turn_group() -> None:
    pipeline = AsyncDualPipeline(
        job_id="job-phase2-mixed-fallback",
        draft_engine=DummyEngine(response_text="draft", latency_ms=0),
        patch_engine=DummyEngine(response_text="patch", latency_ms=0),
        transcription_profile="sv_whisper_dual",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=False,
        punctuation_service=Mock(),
    )
    pipeline.slow_worker.process_turn_group = AsyncMock(  # type: ignore[assignment]
        side_effect=AssertionError("mixed fallback 不应调用 slow_worker")
    )

    audio = np.zeros(16000, dtype=np.float32)
    ctx0 = ProcessingContext(job_id="job-phase2-mixed-fallback", chunk_index=0, audio_chunk=None)
    ctx1 = ProcessingContext(job_id="job-phase2-mixed-fallback", chunk_index=1, audio_chunk=None)
    pipeline._context_cache = {0: ctx0, 1: ctx1}
    pipeline._audio_chunks_by_index = {
        0: SimpleNamespace(index=0, start=0.0, end=1.0, audio=audio, sample_rate=16000, language="mixed"),
        1: SimpleNamespace(index=1, start=1.0, end=2.0, audio=audio, sample_rate=16000, language="mixed"),
    }

    envelope = TurnGroupEnvelope(
        group=TurnGroup(
            group_id="sw-000001",
            speaker_id="spk-1",
            source_chunks=["chunk-0", "chunk-1"],
            flush_reason="eof_flush",
            language="mixed",
            metadata={"window_id": "sw-000001", "is_mixed_window": True, "route": "fallback"},
        ),
        sentences=[],
        punctuation_decision=None,
    )

    paused = await pipeline._process_turn_group(
        envelope,
        job_dir=None,
        total_chunks=0,
        slow_processed_indices=set(),
        token=None,
    )

    assert paused is False
    assert pipeline.queue_final.qsize() == 2
    assert ctx0.whisper_skipped is True
    assert ctx1.whisper_skipped is True
    assert getattr(ctx0, "slow_window_is_mixed", False) is True
    assert getattr(ctx1, "slow_window_is_mixed", False) is True
