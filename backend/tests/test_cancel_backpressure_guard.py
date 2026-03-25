"""
取消链路背压防卡死回归测试。
"""
from __future__ import annotations

import asyncio
import sys
import types
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


backend_path = Path(__file__).parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))


if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    class _CudaStub:
        @staticmethod
        def is_available() -> bool:
            return False

    torch_stub.cuda = _CudaStub()
    sys.modules["torch"] = torch_stub


from app.schemas.pipeline_context import ProcessingContext  # noqa: E402
from app.pipelines.dual_pipeline.services.full_pipeline_orchestrator_service import (  # noqa: E402
    FullPipelineOrchestratorService,
)
from app.pipelines.dual_pipeline.services.fast_loop_service import FastLoopService  # noqa: E402
from app.pipelines.dual_pipeline.services.slow_loop_service import SlowLoopService  # noqa: E402
from app.pipelines.workers.slow_worker import SlowWorker  # noqa: E402
from app.engines.whisper_engine import WhisperEngine  # noqa: E402
from app.utils.cancellation_token import (  # noqa: E402
    CancelledException,
    create_cancellation_token,
)


@pytest.mark.asyncio
async def test_slow_loop_queue_final_full_cancel_exits_quickly() -> None:
    """queue_final 背压 + 已取消时，SlowLoop 应快速退出且不阻塞。"""
    token = create_cancellation_token("job_backpressure")
    token.cancel()

    host = SimpleNamespace(
        cancellation_token=token,
        queue_inter=asyncio.Queue(maxsize=1),
        queue_final=asyncio.Queue(maxsize=1),
        bridge_controller=None,
        _turn_group_builder=SimpleNamespace(flush_idle=lambda now: None),
        _is_turn_group_committed=lambda _group_id: False,
        _record_turn_group_unit=lambda **kwargs: None,
        _process_turn_group=lambda *args, **kwargs: asyncio.sleep(0),
        debug_punctuation=False,
        job_id="job_backpressure",
        errors=[],
        pause_exception=None,
        logger=Mock(),
        progress_emitter=None,
    )

    await host.queue_final.put(
        ProcessingContext(
            job_id="job_backpressure",
            chunk_index=999,
            audio_chunk=None,
        )
    )
    await host.queue_inter.put(
        ProcessingContext(
            job_id="job_backpressure",
            chunk_index=-1,
            audio_chunk=None,
            is_end=True,
            error=CancelledException("job_backpressure"),
        )
    )

    service = SlowLoopService(host=host)
    start = time.perf_counter()
    await asyncio.wait_for(
        service.run(job_dir=None, total_chunks=0),
        timeout=2.0,
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 1.5
    assert host.queue_final.qsize() == 1


@pytest.mark.asyncio
async def test_fast_loop_end_signal_not_dropped_when_backpressure_is_temporary() -> None:
    """正常路径下，end_ctx 在短时背压期间不应被丢弃。"""
    token = create_cancellation_token("job_fast_end_signal")
    host = SimpleNamespace(
        queue_inter=asyncio.Queue(maxsize=1),
        logger=Mock(),
        errors=[],
    )
    await host.queue_inter.put(
        ProcessingContext(
            job_id="job_fast_end_signal",
            chunk_index=1,
            audio_chunk=None,
        )
    )

    service = FastLoopService(host=host)
    end_ctx = ProcessingContext(
        job_id="job_fast_end_signal",
        chunk_index=-1,
        audio_chunk=None,
        is_end=True,
    )

    async def _drain_queue_later() -> None:
        await asyncio.sleep(0.2)
        await host.queue_inter.get()

    drain_task = asyncio.create_task(_drain_queue_later())
    result = await asyncio.wait_for(
        service._put_queue_inter_with_guard(
            payload=end_ctx,
            token=token,
            timeout_seconds=0.05,
            max_retry=2,
            allow_drop=True,
            reason="end_ctx",
        ),
        timeout=1.0,
    )
    await drain_task
    assert result is True
    queued = await asyncio.wait_for(host.queue_inter.get(), timeout=0.3)
    assert queued.is_end is True


@pytest.mark.asyncio
async def test_slow_loop_end_signal_not_dropped_when_backpressure_is_temporary() -> None:
    """正常路径下，SlowLoop 的 end/error 信号在短时背压期间不应丢弃。"""
    token = create_cancellation_token("job_slow_end_signal")
    host = SimpleNamespace(
        queue_final=asyncio.Queue(maxsize=1),
        logger=Mock(),
        errors=[],
    )
    await host.queue_final.put(
        ProcessingContext(
            job_id="job_slow_end_signal",
            chunk_index=1,
            audio_chunk=None,
        )
    )

    service = SlowLoopService(host=host)
    end_ctx = ProcessingContext(
        job_id="job_slow_end_signal",
        chunk_index=-1,
        audio_chunk=None,
        is_end=True,
    )

    async def _drain_queue_later() -> None:
        await asyncio.sleep(0.2)
        await host.queue_final.get()

    drain_task = asyncio.create_task(_drain_queue_later())
    result = await asyncio.wait_for(
        service._put_queue_final_with_guard(
            payload=end_ctx,
            token=token,
            timeout_seconds=0.05,
            max_retry=2,
            allow_drop=True,
            reason="upstream_end_or_error",
        ),
        timeout=1.0,
    )
    await drain_task
    assert result is True
    queued = await asyncio.wait_for(host.queue_final.get(), timeout=0.3)
    assert queued.is_end is True


@pytest.mark.asyncio
async def test_fast_loop_terminal_signal_not_dropped_when_host_has_errors() -> None:
    """即使 host.errors 非空，终止信号也必须在短时背压后送达。"""
    token = create_cancellation_token("job_fast_terminal_must_deliver")
    host = SimpleNamespace(
        queue_inter=asyncio.Queue(maxsize=1),
        logger=Mock(),
        errors=[RuntimeError("upstream_error_already_recorded")],
    )
    await host.queue_inter.put(
        ProcessingContext(
            job_id="job_fast_terminal_must_deliver",
            chunk_index=1,
            audio_chunk=None,
        )
    )

    service = FastLoopService(host=host)
    end_ctx = ProcessingContext(
        job_id="job_fast_terminal_must_deliver",
        chunk_index=-1,
        audio_chunk=None,
        is_end=True,
    )

    async def _drain_queue_later() -> None:
        await asyncio.sleep(0.2)
        await host.queue_inter.get()

    drain_task = asyncio.create_task(_drain_queue_later())
    result = await asyncio.wait_for(
        service._put_queue_inter_with_guard(
            payload=end_ctx,
            token=token,
            timeout_seconds=0.05,
            max_retry=2,
            allow_drop=True,
            reason="end_ctx",
        ),
        timeout=1.0,
    )
    await drain_task
    assert result is True
    queued = await asyncio.wait_for(host.queue_inter.get(), timeout=0.3)
    assert queued.is_end is True


@pytest.mark.asyncio
async def test_fast_loop_allow_drop_when_terminal_request_is_active() -> None:
    """取消请求生效时，允许丢弃终止载荷以保证快速退出。"""
    token = create_cancellation_token("job_fast_drop_on_cancel")
    token.cancel()
    host = SimpleNamespace(
        queue_inter=asyncio.Queue(maxsize=1),
        logger=Mock(),
        errors=[],
    )
    await host.queue_inter.put(
        ProcessingContext(
            job_id="job_fast_drop_on_cancel",
            chunk_index=1,
            audio_chunk=None,
        )
    )

    service = FastLoopService(host=host)
    end_ctx = ProcessingContext(
        job_id="job_fast_drop_on_cancel",
        chunk_index=-1,
        audio_chunk=None,
        is_end=True,
    )

    start = time.perf_counter()
    result = await asyncio.wait_for(
        service._put_queue_inter_with_guard(
            payload=end_ctx,
            token=token,
            timeout_seconds=0.05,
            max_retry=2,
            allow_drop=True,
            reason="end_ctx",
        ),
        timeout=1.0,
    )
    elapsed = time.perf_counter() - start
    assert result is False
    assert elapsed < 0.5


@pytest.mark.asyncio
async def test_full_pipeline_prefers_canceled_over_generic_error() -> None:
    """cancellation_token 已取消时，编排器应优先抛出 CancelledException。"""
    token = create_cancellation_token("job_orchestrator_cancel")
    token.cancel()

    async def _prepare_timeline_domain(**kwargs):
        return None

    async def _fast_loop(*args, **kwargs):
        host.errors.append(RuntimeError("generic_error_should_not_win"))
        return None

    async def _slow_loop(*args, **kwargs):
        return None

    async def _align_loop(*args, **kwargs):
        return None

    host = SimpleNamespace(
        job_id="job_orchestrator_cancel",
        cancellation_token=token,
        errors=[],
        pause_exception=None,
        _is_dual_time_experiment_enabled=False,
        _dual_time_compare_accumulator={},
        _dual_time_legacy_sentences_by_chunk={},
        _dual_time_experiment_sentences_by_chunk={},
        _decision_processor=SimpleNamespace(reset_state=lambda: None),
        _soft_cut_pending_deferred_by_stream={},
        _prepare_timeline_domain=_prepare_timeline_domain,
        _fast_loop=_fast_loop,
        _slow_loop=_slow_loop,
        _align_loop=_align_loop,
        logger=Mock(),
        _force_save_pause_checkpoint=lambda _job_dir, _total_chunks: True,
        _finalized_indices=set(),
    )

    service = FullPipelineOrchestratorService(host=host)
    with pytest.raises(CancelledException):
        await service.run(audio_chunks=[])


@pytest.mark.asyncio
async def test_slow_worker_infer_is_interruptible_by_cancel_checker() -> None:
    """慢流推理等待期间收到取消时，应快速退出。"""

    class _BlockingPatchEngine:
        async def transcribe(self, *args, **kwargs):
            await asyncio.sleep(5.0)
            raise AssertionError("测试不应等待到推理完成")

    token = create_cancellation_token("job_slow_cancel_guard")
    worker = SlowWorker(
        patch_engine=_BlockingPatchEngine(),
        whisper_language="en",
        logger=Mock(),
    )

    async def _trigger_cancel() -> None:
        await asyncio.sleep(0.3)
        token.cancel()

    loop = asyncio.get_running_loop()
    start = loop.time()
    cancel_task = asyncio.create_task(_trigger_cancel())

    with pytest.raises(CancelledException):
        await asyncio.wait_for(
            worker.infer(
                audio=np.zeros(160, dtype=np.float32),
                initial_prompt="test",
                cancel_checker=token.raise_if_canceled,
            ),
            timeout=2.0,
        )

    await cancel_task
    elapsed = loop.time() - start
    assert elapsed < 1.5


@pytest.mark.asyncio
async def test_whisper_engine_transcribe_not_block_event_loop() -> None:
    """WhisperEngine.transcribe 应通过线程卸载避免阻塞事件循环。"""

    class _StubWhisperService:
        def __init__(self) -> None:
            self.model = object()

        def load_model(self, *args, **kwargs) -> None:
            self.model = object()

        def transcribe(self, *args, **kwargs):
            time.sleep(0.5)
            return {"text": "ok", "segments": [], "language": "en"}

    engine = WhisperEngine(model_name=None)
    engine.service = _StubWhisperService()

    ticked = False

    async def _ticker() -> None:
        nonlocal ticked
        await asyncio.sleep(0.1)
        ticked = True

    ticker_task = asyncio.create_task(_ticker())
    transcribe_task = asyncio.create_task(
        engine.transcribe(np.zeros(320, dtype=np.float32), language="en")
    )

    await asyncio.wait_for(ticker_task, timeout=0.3)
    assert ticked is True
    await asyncio.wait_for(transcribe_task, timeout=2.0)
