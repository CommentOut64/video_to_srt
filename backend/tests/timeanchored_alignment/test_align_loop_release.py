from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.pipelines.dual_pipeline.services.align_loop_service import AlignLoopService
from app.schemas.pipeline_context import ProcessingContext
from app.services.timeanchored_alignment.contracts import (
    LanguageRun,
    PhoneUnit,
    PronunciationPackage,
    ProtectedSpan,
    TokenToPhoneSpan,
    TokenUnit,
    TextTruthPackage,
    TextTruthQuality,
    TextTruthUnit,
)


def _build_text_truth() -> TextTruthPackage:
    return TextTruthPackage(
        units=(
            TextTruthUnit(
                text="你好",
                normalized_text="你好",
                confidence=0.9,
                language="zh",
            ),
        ),
        quality=TextTruthQuality(hallucination_risk=0.0, repetition_ratio=0.0, length_ratio=1.0),
        language="zh",
        raw_text="你好",
        normalized_text="你好",
    )


def _build_pronunciation() -> PronunciationPackage:
    token = TokenUnit(token_text="你", language="zh", char_start=0, char_end=1)
    phone = PhoneUnit(phone_text="ni3", language="zh")
    span = TokenToPhoneSpan(token_index=0, phone_start=0, phone_end=0)
    return PronunciationPackage(
        token_units=(token,),
        phone_units=(phone,),
        token_to_phone_spans=(span,),
        frontend_source="homophone_tokenizer",
        dependency_mode={"zh": "pypinyin"},
        language="zh",
    )


@dataclass
class _DummyHost:
    queue_final: asyncio.Queue
    _run_alignment_stage: object
    _is_finalize_batch_committed: object
    _record_finalize_batch_unit: object
    cancellation_token: object = None
    _runtime_checkpoint_service: object = None
    _finalized_indices: set | None = None
    _last_align_chunk_index: int = -1
    progress_emitter: object = None
    subtitle_manager: object = None
    pause_exception: object = None
    errors: list = None
    job_id: str = "job-align-release"
    logger: object = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []
        if self.logger is None:
            self.logger = Mock()


@pytest.mark.asyncio
async def test_align_loop_releases_preparation_artifacts_after_alignment_stage() -> None:
    queue = asyncio.Queue()
    ctx = ProcessingContext(
        job_id="job-align-release",
        chunk_index=0,
        audio_chunk=SimpleNamespace(language="zh"),
        sv_result={
            "text": "你好",
            "ctc_logits": [[0.1, 0.9]],
            "top_candidates": [{"token": "你"}],
        },
        alignment_preparation=object(),
        text_truth=_build_text_truth(),
        protected_spans=[ProtectedSpan(start=0, end=2, kind="decimal", text="3.14")],
        language_runs=[LanguageRun(run_text="你好", run_language="zh", char_start=0, char_end=2)],
        pronunciation_package=_build_pronunciation(),
        slow_window_meta={"window_id": "sw-0001"},
        pronunciation_report={"cache_size": 12},
    )
    await queue.put(ctx)
    await queue.put(
        ProcessingContext(
            job_id="job-align-release",
            chunk_index=-1,
            audio_chunk=None,
            is_end=True,
        )
    )

    async def _run_alignment_stage(target_ctx: ProcessingContext) -> None:
        target_ctx.final_sentences = []

    host = _DummyHost(
        queue_final=queue,
        _run_alignment_stage=_run_alignment_stage,
        _is_finalize_batch_committed=lambda _idx: False,
        _record_finalize_batch_unit=lambda **kwargs: None,
    )
    service = AlignLoopService(host=host)
    results: list[ProcessingContext] = []

    await service.run(results=results, job_dir=None, total_chunks=1)

    assert len(results) == 1
    assert ctx.text_truth is None
    assert ctx.alignment_preparation is None
    assert ctx.protected_spans == []
    assert ctx.language_runs == []
    assert ctx.pronunciation_package is None
    assert ctx.slow_window_meta == {}
    assert ctx.pronunciation_report == {}
    assert ctx.sv_result is not None
    assert "ctc_logits" not in ctx.sv_result
    assert "top_candidates" not in ctx.sv_result


@pytest.mark.asyncio
async def test_align_loop_releases_preparation_artifacts_when_batch_already_committed() -> None:
    queue = asyncio.Queue()
    ctx = ProcessingContext(
        job_id="job-align-release",
        chunk_index=8,
        audio_chunk=SimpleNamespace(language="zh"),
        sv_result={
            "text": "你好",
            "ctc_logits": [[0.1, 0.9]],
            "top_candidates": [{"token": "你"}],
        },
        alignment_preparation=object(),
        text_truth=_build_text_truth(),
        protected_spans=[ProtectedSpan(start=0, end=2, kind="decimal", text="3.14")],
        language_runs=[LanguageRun(run_text="你好", run_language="zh", char_start=0, char_end=2)],
        pronunciation_package=_build_pronunciation(),
        slow_window_meta={"window_id": "sw-0008"},
        pronunciation_report={"cache_size": 99},
    )
    await queue.put(ctx)
    await queue.put(
        ProcessingContext(
            job_id="job-align-release",
            chunk_index=-1,
            audio_chunk=None,
            is_end=True,
        )
    )

    run_called = False

    async def _run_alignment_stage(_target_ctx: ProcessingContext) -> None:
        nonlocal run_called
        run_called = True

    host = _DummyHost(
        queue_final=queue,
        _run_alignment_stage=_run_alignment_stage,
        _is_finalize_batch_committed=lambda _idx: True,
        _record_finalize_batch_unit=lambda **kwargs: None,
    )
    service = AlignLoopService(host=host)
    results: list[ProcessingContext] = []

    await service.run(results=results, job_dir=None, total_chunks=1)

    assert run_called is False
    assert len(results) == 1
    assert ctx.text_truth is None
    assert ctx.alignment_preparation is None
    assert ctx.pronunciation_package is None
    assert "ctc_logits" not in (ctx.sv_result or {})
