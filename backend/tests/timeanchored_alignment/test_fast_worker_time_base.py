from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from app.core.asr.enums import ASRCapability, TimestampPrecision
from app.core.asr.models import ASRMetadata, ASRResult, Segment, WordTimestamp
from app.pipelines.workers.fast_worker import FastWorker
from app.schemas.pipeline_context import ProcessingContext
from app.services.audio.chunk_engine import AudioChunk


@dataclass
class _DummyRuntimeService:
    payload: dict[str, Any]

    def get_effective_runtime_global(self) -> dict[str, Any]:
        return self.payload


class _DummyDraftEngine:
    def __init__(self, *, include_ctc_logits: bool = True) -> None:
        self._include_ctc_logits = include_ctc_logits

    async def transcribe(self, *args: Any, **kwargs: Any) -> ASRResult:
        logits = np.array(
            [
                [2.1, 5.0, 1.0],
                [1.9, 4.8, 1.2],
                [5.5, 1.0, 1.0],
                [1.8, 1.1, 5.2],
                [1.7, 1.2, 5.0],
            ],
            dtype=np.float32,
        )
        raw_tokens = [
            {
                "word": "你",
                "start": 0.0,
                "end": 0.12,
                "confidence": 0.9,
                "is_pseudo": False,
                "top_candidates": [
                    {"text": "你", "score": 0.51, "token_id": 1},
                    {"text": "好", "score": 0.48, "token_id": 2},
                ],
            },
            {
                "word": "好",
                "start": 0.12,
                "end": 0.24,
                "confidence": 0.92,
                "is_pseudo": False,
            },
        ]
        compact_trace = {
            "blank_ratio": 0.30,
            "avg_max_prob": 0.78,
            "low_prob_ratio": 0.10,
            "raw_tokens": raw_tokens,
            "top_k": 3,
        }
        raw_tags: dict[str, Any] = {
            "raw_tokens": raw_tokens,
            "sv_language_info": {"language": "zh", "confidence": 0.85},
            "ctc_compact_trace": compact_trace,
            "ctc_frame_stride": 0.06,
        }
        if self._include_ctc_logits:
            raw_tags["ctc_logits"] = logits

        metadata = ASRMetadata(
            engine="unit",
            source="unit",
            timestamp_precision=TimestampPrecision.WORD,
            capabilities=[ASRCapability.WORD_TIMESTAMPS],
            raw_tags=raw_tags,
        )
        return ASRResult(
            text="你好",
            text_clean="你好",
            segments=[Segment(start=0.0, end=0.24, text="你好")],
            words=[
                WordTimestamp(word="你", start=0.0, end=0.12, confidence=0.9),
                WordTimestamp(word="好", start=0.12, end=0.24, confidence=0.92),
            ],
            raw_tokens=raw_tokens,
            confidence=0.9,
            language="zh",
            metadata=metadata,
        )


def _make_ctx() -> ProcessingContext:
    chunk = AudioChunk(
        index=0,
        start=0.0,
        end=1.0,
        audio=np.zeros(16000, dtype=np.float32),
        language=None,
    )
    return ProcessingContext(job_id="job-time-base", chunk_index=0, audio_chunk=chunk)


@pytest.mark.asyncio
async def test_fast_worker_writes_time_base_chunk_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.pipelines.workers.fast_worker as fast_worker_module

    runtime_payload = {
        "effective": {
            "sensevoice": {"use_itn": True},
            "alignment_pipeline": {"version": "timeanchored", "mode": "shadow"},
        },
        "override": {
            "alignment_pipeline": {"version": "timeanchored", "mode": "shadow"},
        },
    }
    monkeypatch.setattr(
        fast_worker_module,
        "get_model_runtime_config_service",
        lambda: _DummyRuntimeService(runtime_payload),
    )

    worker = FastWorker(job_id="job-time-base", draft_engine=_DummyDraftEngine(include_ctc_logits=True))
    ctx = _make_ctx()
    await worker.process(ctx)

    assert ctx.time_base_chunk is not None
    assert "ctc_logits" not in (ctx.sv_result or {})


@pytest.mark.asyncio
async def test_fast_worker_keeps_legacy_path_without_time_base(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.pipelines.workers.fast_worker as fast_worker_module

    runtime_payload = {
        "effective": {
            "sensevoice": {"use_itn": True},
        },
        "override": {
            "alignment_pipeline": {"version": "legacy", "mode": "off"},
        },
    }
    monkeypatch.setattr(
        fast_worker_module,
        "get_model_runtime_config_service",
        lambda: _DummyRuntimeService(runtime_payload),
    )

    worker = FastWorker(job_id="job-legacy", draft_engine=_DummyDraftEngine(include_ctc_logits=True))
    ctx = _make_ctx()
    await worker.process(ctx)

    assert ctx.time_base_chunk is None
    assert "ctc_logits" in (ctx.sv_result or {})
