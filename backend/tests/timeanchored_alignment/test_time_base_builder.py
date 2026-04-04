from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from app.services.timeanchored_alignment.adapters.fast.sensevoice_time_adapter import SenseVoiceTimeAdapter
from app.services.timeanchored_alignment.contracts import TimeBasePackage, TimeBaseQuality, TimeBaseUnit
from app.services.timeanchored_alignment.time_base_builder import TimeBaseBuilder


class MockFastTimeAdapter:
    @property
    def can_decode_ctc(self) -> bool:
        return False

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def build_time_base(
        self,
        *,
        ctc_logits,
        language: str,
        frame_stride: float = 0.06,
        encoder_out_lens: int | None = None,
        compact_acoustic_trace: dict | None = None,
        raw_tokens=None,
    ) -> TimeBasePackage:
        self.calls.append(
            {
                "ctc_logits": ctc_logits,
                "language": language,
                "frame_stride": frame_stride,
                "encoder_out_lens": encoder_out_lens,
                "compact_acoustic_trace": dict(compact_acoustic_trace or {}),
                "raw_tokens": list(raw_tokens or []),
            }
        )
        unit = TimeBaseUnit(text="你", start=0.0, end=0.1, confidence=0.93, token_type="word")
        return TimeBasePackage(
            raw_units=(unit,),
            word_units=(replace(unit, text="你好", end=0.2),),
            quality=TimeBaseQuality(blank_ratio=0.0, avg_max_prob=0.93, low_prob_ratio=0.0),
            language=language,
            source="mock_fast_adapter",
        )


def _make_sv_result_with_logits(zh_ctc_logits_fixture: np.ndarray, compact_trace_fixture: dict) -> dict:
    return {
        "language": "zh",
        "ctc_frame_stride": 0.06,
        "ctc_logits": np.array(zh_ctc_logits_fixture, copy=True),
        "raw_tokens": list(compact_trace_fixture["raw_tokens"]),
        "ctc_compact_trace": dict(compact_trace_fixture),
    }


def test_build_from_processing_context(
    zh_ctc_logits_fixture: np.ndarray,
    zh_vocab: dict[int, str],
    compact_trace_fixture: dict,
) -> None:
    builder = TimeBaseBuilder(adapter=SenseVoiceTimeAdapter(vocab=zh_vocab, blank_id=0))
    ctx = SimpleNamespace(
        chunk_index=0,
        sv_result=_make_sv_result_with_logits(zh_ctc_logits_fixture, compact_trace_fixture),
    )

    pkg = builder.build(ctx)
    assert pkg is not None
    assert pkg.language == "zh"


def test_ctc_logits_released_after_build(
    zh_ctc_logits_fixture: np.ndarray,
    zh_vocab: dict[int, str],
    compact_trace_fixture: dict,
) -> None:
    builder = TimeBaseBuilder(adapter=SenseVoiceTimeAdapter(vocab=zh_vocab, blank_id=0))
    sv_result = _make_sv_result_with_logits(zh_ctc_logits_fixture, compact_trace_fixture)
    ctx = SimpleNamespace(chunk_index=1, sv_result=sv_result)

    pkg = builder.build(ctx)
    assert pkg is not None
    assert "ctc_logits" not in sv_result


def test_empty_sv_result_returns_none(zh_vocab: dict[int, str]) -> None:
    builder = TimeBaseBuilder(adapter=SenseVoiceTimeAdapter(vocab=zh_vocab, blank_id=0))
    ctx = SimpleNamespace(chunk_index=2, sv_result={})
    assert builder.build(ctx) is None


def test_build_from_compact_trace_without_logits(compact_trace_fixture: dict) -> None:
    builder = TimeBaseBuilder(adapter=SenseVoiceTimeAdapter(vocab=None, blank_id=0))
    ctx = SimpleNamespace(
        chunk_index=3,
        sv_result={
            "language": "zh",
            "raw_tokens": list(compact_trace_fixture["raw_tokens"]),
            "ctc_compact_trace": dict(compact_trace_fixture),
            "ctc_frame_stride": 0.06,
        },
    )
    pkg = builder.build(ctx)
    assert pkg is not None
    assert pkg.quality.blank_ratio == compact_trace_fixture["blank_ratio"]


def test_build_from_mock_fast_adapter_without_sensevoice_decoder(
    zh_ctc_logits_fixture: np.ndarray,
    compact_trace_fixture: dict,
) -> None:
    adapter = MockFastTimeAdapter()
    builder = TimeBaseBuilder(adapter=adapter)
    sv_result = _make_sv_result_with_logits(zh_ctc_logits_fixture, compact_trace_fixture)
    sv_result["encoder_out_lens"] = 4
    sv_result["blank_track"] = [0.71, 0.18]
    sv_result["sparse_logits"] = [{"frame": 0, "token_id": 1, "score": 0.9}]
    ctx = SimpleNamespace(chunk_index=4, sv_result=sv_result)

    pkg = builder.build(ctx)

    assert pkg is not None
    assert pkg.source == "mock_fast_adapter"
    assert len(adapter.calls) == 1
    assert adapter.calls[0]["ctc_logits"] is None
    assert adapter.calls[0]["encoder_out_lens"] == 4
    assert pkg.metadata["encoder_out_lens"] == 4
    assert pkg.metadata["blank_track"] == [0.71, 0.18]
    assert pkg.metadata["sparse_logits"] == [{"frame": 0, "token_id": 1, "score": 0.9}]
