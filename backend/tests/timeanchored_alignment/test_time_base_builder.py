from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.services.timeanchored_alignment.adapters.fast.sensevoice_time_adapter import SenseVoiceTimeAdapter
from app.services.timeanchored_alignment.time_base_builder import TimeBaseBuilder


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
