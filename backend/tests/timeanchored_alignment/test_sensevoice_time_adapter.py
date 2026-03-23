from __future__ import annotations

import numpy as np

from app.services.timeanchored_alignment.adapters.fast.sensevoice_time_adapter import SenseVoiceTimeAdapter
from app.services.timeanchored_alignment.contracts import TimeBasePackage


def test_build_time_base_from_ctc_logits_zh(
    zh_ctc_logits_fixture: np.ndarray,
    zh_vocab: dict[int, str],
) -> None:
    adapter = SenseVoiceTimeAdapter(vocab=zh_vocab, blank_id=0)
    pkg = adapter.build_time_base(
        ctc_logits=zh_ctc_logits_fixture,
        language="zh",
        frame_stride=0.06,
    )
    assert isinstance(pkg, TimeBasePackage)
    assert pkg.contract_version == "1.0"
    assert pkg.language == "zh"
    assert len(pkg.raw_units) > 0
    assert len(pkg.word_units) > 0
    assert pkg.quality.blank_ratio >= 0.0
    assert pkg.source == "sensevoice"


def test_build_time_base_from_ctc_logits_en(
    en_ctc_logits_fixture: np.ndarray,
    en_vocab: dict[int, str],
) -> None:
    adapter = SenseVoiceTimeAdapter(vocab=en_vocab, blank_id=0)
    pkg = adapter.build_time_base(
        ctc_logits=en_ctc_logits_fixture,
        language="en",
        frame_stride=0.06,
    )
    assert len(pkg.word_units) > 0
    for word_unit in pkg.word_units:
        assert not word_unit.text.startswith("▁")


def test_quality_stats_calculation(
    zh_ctc_logits_fixture: np.ndarray,
    zh_vocab: dict[int, str],
) -> None:
    adapter = SenseVoiceTimeAdapter(vocab=zh_vocab, blank_id=0)
    pkg = adapter.build_time_base(
        ctc_logits=zh_ctc_logits_fixture,
        language="zh",
    )
    assert 0.0 <= pkg.quality.blank_ratio <= 1.0
    assert 0.0 <= pkg.quality.avg_max_prob <= 1.0
    assert 0.0 <= pkg.quality.low_prob_ratio <= 1.0


def test_top_candidates_retention_and_max_k(
    ambiguous_ctc_logits_fixture: np.ndarray,
    zh_vocab: dict[int, str],
) -> None:
    adapter = SenseVoiceTimeAdapter(vocab=zh_vocab, blank_id=0, top_k=7)
    pkg = adapter.build_time_base(
        ctc_logits=ambiguous_ctc_logits_fixture,
        language="zh",
    )

    ambiguous_units = [unit for unit in pkg.raw_units if unit.top_candidates]
    assert ambiguous_units
    assert all(len(unit.top_candidates) <= 3 for unit in ambiguous_units)


def test_build_time_base_from_compact_trace_without_logits(compact_trace_fixture: dict) -> None:
    adapter = SenseVoiceTimeAdapter(vocab=None, blank_id=0)
    pkg = adapter.build_time_base(
        ctc_logits=None,
        compact_acoustic_trace=compact_trace_fixture,
        raw_tokens=compact_trace_fixture["raw_tokens"],
        language="zh",
    )
    assert len(pkg.raw_units) == len(compact_trace_fixture["raw_tokens"])
    assert pkg.quality.blank_ratio == compact_trace_fixture["blank_ratio"]


def test_build_time_base_falls_back_to_compact_trace_when_logits_invalid(
    compact_trace_fixture: dict,
    zh_vocab: dict[int, str],
) -> None:
    adapter = SenseVoiceTimeAdapter(vocab=zh_vocab, blank_id=0)
    invalid_logits = np.array([[np.nan, np.nan, np.nan]], dtype=np.float32)
    pkg = adapter.build_time_base(
        ctc_logits=invalid_logits,
        compact_acoustic_trace=compact_trace_fixture,
        raw_tokens=compact_trace_fixture["raw_tokens"],
        language="zh",
    )

    assert len(pkg.raw_units) == len(compact_trace_fixture["raw_tokens"])
    assert pkg.metadata.get("decoder") == "compact_trace"


def test_build_time_base_from_compact_trace_with_nan_quality_values() -> None:
    adapter = SenseVoiceTimeAdapter(vocab=None, blank_id=0)
    compact_trace = {
        "blank_ratio": np.nan,
        "avg_max_prob": np.inf,
        "low_prob_ratio": -np.inf,
        "raw_tokens": [
            {"word": "你", "start": 0.0, "end": 0.1, "confidence": np.nan},
            {"word": "好", "start": 0.12, "end": 0.24, "confidence": np.inf},
        ],
    }

    pkg = adapter.build_time_base(
        ctc_logits=None,
        compact_acoustic_trace=compact_trace,
        raw_tokens=compact_trace["raw_tokens"],
        language="zh",
    )

    assert 0.0 <= pkg.quality.blank_ratio <= 1.0
    assert 0.0 <= pkg.quality.avg_max_prob <= 1.0
    assert 0.0 <= pkg.quality.low_prob_ratio <= 1.0
    assert all(0.0 <= unit.confidence <= 1.0 for unit in pkg.raw_units)
