from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.schemas.pipeline_context import ProcessingContext
from app.services.bridge.turn_group_builder import TurnGroupEnvelope
from app.services.bridge.turn_group_models import TurnGroup
from app.services.language_policy import resolve_language_tag
from app.services.sensevoice_onnx_service import SenseVoiceLanguageInfo, SenseVoiceONNXService
from app.services.textflow.output_layer import OutputLayerProcessor
from app.services.alignment.types import OutputLayerInput
from app.models.sensevoice_models import SentenceSegment
from app.pipelines.dual_pipeline.services.slow_loop_service import SlowLoopService


@dataclass
class _DummyMergeResult:
    words: list[dict]
    raw_tokens: list[dict]


class _DummyNormalizer:
    def extract_tags(self, _text: str) -> dict:
        return {"language": "zh"}

    def process(self, text: str, extract_info: bool = True, language: str | None = None) -> dict:
        return {
            "text_clean": text.replace("<|zh|>", "").strip(),
            "tags": {"emotion": None, "event": None},
        }


def _build_stub_sensevoice_service() -> SenseVoiceONNXService:
    service = SenseVoiceONNXService.__new__(SenseVoiceONNXService)
    service.is_loaded = True
    service.config = SimpleNamespace(language="auto", use_itn=True, ban_emo_unk=False)
    service.logger = SimpleNamespace(debug=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
    service.time_stride = 0.06
    service._apply_runtime_params = lambda: {}
    service._preprocess_audio = lambda _audio, _sr: np.zeros((1, 4, 560), dtype=np.float32)
    service._run_inference = lambda _features, language="auto", use_itn=True: np.zeros((4, 5), dtype=np.float32)
    service._clean_ctc_word_timestamps = lambda words: words
    service._strip_unknown_emotion_tags = lambda text: text
    service.decoder = SimpleNamespace(
        vocab={0: "<blank>", 1: "你", 2: "好", 3: "<|zh|>", 4: "<unk>"},
        blank_id=0,
        decode=lambda _logits, _stride: (
            "<|zh|>你好",
            [
                {"word": "你", "start": 0.06, "end": 0.12, "confidence": 0.9, "is_pseudo": False},
                {"word": "好", "start": 0.12, "end": 0.18, "confidence": 0.92, "is_pseudo": False},
            ],
            0.91,
            SenseVoiceLanguageInfo(language="zh", confidence=0.88),
        )
    )
    return service


def test_phase0_reality_gate_baseline_sensevoice_result_default_no_ctc_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    """阶段性现实基线：默认结果不暴露 ctc_logits（Phase1 仅在 flag 开启时保留）。

    退役条件：当系统决定默认也输出完整矩阵时，该测试应在对应 Phase 显式迁移。
    """
    import app.services.text_normalizer as text_normalizer_module
    import app.services.token_merge_service as token_merge_service_module

    monkeypatch.setattr(text_normalizer_module, "get_text_normalizer", lambda: _DummyNormalizer())
    monkeypatch.setattr(
        token_merge_service_module,
        "merge_tokens",
        lambda tokens, language=None: _DummyMergeResult(words=list(tokens), raw_tokens=[dict(item) for item in tokens]),
    )

    service = _build_stub_sensevoice_service()
    result = service.transcribe_audio_array(np.zeros(16000, dtype=np.float32), sample_rate=16000)

    assert "ctc_logits" not in result


def test_phase0_reality_gate_sensevoice_result_keeps_ctc_logits_only_with_explicit_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.model_runtime_config_service as runtime_config_module
    import app.services.text_normalizer as text_normalizer_module
    import app.services.token_merge_service as token_merge_service_module

    class _DummyRuntimeService:
        @staticmethod
        def get_effective_runtime_global() -> dict:
            return {
                "effective": {
                    "alignment_pipeline": {
                        "retain_ctc_logits": True,
                    }
                },
                "override": {
                    "alignment_pipeline": {
                        "retain_ctc_logits": True,
                    }
                },
            }

    monkeypatch.setattr(text_normalizer_module, "get_text_normalizer", lambda: _DummyNormalizer())
    monkeypatch.setattr(
        token_merge_service_module,
        "merge_tokens",
        lambda tokens, language=None: _DummyMergeResult(words=list(tokens), raw_tokens=[dict(item) for item in tokens]),
    )
    monkeypatch.setattr(
        runtime_config_module,
        "get_model_runtime_config_service",
        lambda: _DummyRuntimeService(),
    )

    service = _build_stub_sensevoice_service()
    result = service.transcribe_audio_array(np.zeros(16000, dtype=np.float32), sample_rate=16000)

    assert "ctc_logits" in result


def test_phase0_reality_gate_language_policy_currently_aliases_mixed_to_zh() -> None:
    """阶段性现实基线：language_policy 仍存在 mixed -> zh 归一化。"""
    assert resolve_language_tag(language_hint="mixed", fallback="en") == "zh"


@pytest.mark.asyncio
async def test_phase0_reality_gate_slow_loop_consumes_turn_group_envelope() -> None:
    """现实基线：SlowLoopService 当前显式消费 TurnGroupEnvelope。"""
    import asyncio

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
        _is_turn_group_committed=lambda _group_id: False,
        _record_turn_group_unit=lambda **kwargs: None,
        _process_turn_group=_process_turn_group,
        logger=Mock(),
        progress_emitter=None,
        debug_punctuation=False,
        job_id="job-phase0-gate",
        errors=[],
        pause_exception=None,
        _slow_processed_indices=set(),
        _last_slow_chunk_index=-1,
        _context_cache={},
    )

    envelope = TurnGroupEnvelope(
        group=TurnGroup(group_id="tg-000001", speaker_id="spk-1", source_chunks=["chunk-0"]),
        sentences=[],
        punctuation_decision=None,
    )
    await host.queue_inter.put(envelope)
    await host.queue_inter.put(
        ProcessingContext(
            job_id="job-phase0-gate",
            chunk_index=-1,
            audio_chunk=None,
            is_end=True,
        )
    )

    service = SlowLoopService(host=host)
    await service.run(job_dir=None, total_chunks=0)

    assert len(processed_payloads) == 1
    assert isinstance(processed_payloads[0], TurnGroupEnvelope)


def test_phase0_reality_gate_output_layer_uses_output_layer_input_to_replace_chunk() -> None:
    """现实基线：OutputLayerProcessor 通过 OutputLayerInput -> replace_chunk() 收口。"""

    class _DummySubtitleManager:
        def __init__(self) -> None:
            self.last_call = None

        def replace_chunk(self, chunk_index: int, sentences: list[SentenceSegment]) -> list[int]:
            self.last_call = (chunk_index, list(sentences))
            return list(range(len(sentences)))

    subtitle_manager = _DummySubtitleManager()
    processor = OutputLayerProcessor(subtitle_manager=subtitle_manager)

    sentence = SentenceSegment(
        text="你好",
        text_clean="你好",
        start=0.0,
        end=0.8,
        display_confidence=0.95,
    )
    output = processor.process(
        OutputLayerInput(
            chunk_index=3,
            sentence_segments=[sentence],
            language="zh",
        )
    )

    assert subtitle_manager.last_call is not None
    assert subtitle_manager.last_call[0] == 3
    assert len(subtitle_manager.last_call[1]) == 1
    assert output.output_payload.get("transport_meta", {}).get("channels")
