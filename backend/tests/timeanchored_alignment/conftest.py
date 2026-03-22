from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture
def zh_vocab() -> dict[int, str]:
    return {
        0: "<blank>",
        1: "你",
        2: "好",
        3: "世",
        4: "界",
    }


@pytest.fixture
def en_vocab() -> dict[int, str]:
    return {
        0: "<blank>",
        1: "▁hel",
        2: "lo",
        3: "▁world",
        4: "!",
    }


@pytest.fixture
def zh_ctc_logits_fixture(zh_vocab: dict[int, str]) -> np.ndarray:
    vocab_size = max(zh_vocab.keys()) + 1
    logits = np.full((8, vocab_size), -8.0, dtype=np.float32)
    logits[:, 0] = 1.5

    # blank, 你, 你, blank, 好, 好, blank, blank
    logits[1, 1] = 6.0
    logits[2, 1] = 5.8
    logits[4, 2] = 6.2
    logits[5, 2] = 6.0
    return logits


@pytest.fixture
def en_ctc_logits_fixture(en_vocab: dict[int, str]) -> np.ndarray:
    vocab_size = max(en_vocab.keys()) + 1
    logits = np.full((10, vocab_size), -8.0, dtype=np.float32)
    logits[:, 0] = 1.0

    # blank, ▁hel, ▁hel, lo, lo, blank, ▁world, ▁world, blank, !
    logits[1, 1] = 6.0
    logits[2, 1] = 5.9
    logits[3, 2] = 5.8
    logits[4, 2] = 5.7
    logits[6, 3] = 6.1
    logits[7, 3] = 6.0
    logits[9, 4] = 5.5
    return logits


@pytest.fixture
def ambiguous_ctc_logits_fixture(zh_vocab: dict[int, str]) -> np.ndarray:
    vocab_size = max(zh_vocab.keys()) + 1
    logits = np.full((7, vocab_size), -8.0, dtype=np.float32)
    logits[:, 0] = 1.5

    # 在第一个有效单元制造 top1/top2 接近，触发 top_candidates 保留
    logits[1, 1] = 3.20
    logits[1, 2] = 3.10
    logits[2, 1] = 3.05
    logits[2, 2] = 3.00

    logits[4, 2] = 6.10
    logits[5, 2] = 5.90
    return logits


@pytest.fixture
def compact_trace_fixture() -> dict:
    raw_tokens = [
        {
            "word": "你",
            "start": 0.06,
            "end": 0.18,
            "confidence": 0.88,
            "is_pseudo": False,
            "top_candidates": [
                {"text": "你", "score": 0.52, "token_id": 1},
                {"text": "好", "score": 0.47, "token_id": 2},
            ],
        },
        {
            "word": "好",
            "start": 0.24,
            "end": 0.36,
            "confidence": 0.91,
            "is_pseudo": False,
        },
    ]
    return {
        "blank_ratio": 0.34,
        "avg_max_prob": 0.79,
        "low_prob_ratio": 0.11,
        "raw_tokens": raw_tokens,
        "top_k": 3,
    }
