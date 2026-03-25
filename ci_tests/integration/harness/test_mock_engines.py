from __future__ import annotations

import asyncio

import numpy as np

from ci_tests.integration.harness.mock_engines import MockEngineFactory


def test_create_dummy_draft_emits_minimal_time_base_trace() -> None:
    engine = MockEngineFactory.create_dummy_draft(text="测试文本", latency_ms=0)

    result = asyncio.run(engine.transcribe(np.zeros(1600, dtype=np.float32), language="zh"))

    raw_tags = result.metadata.raw_tags
    assert isinstance(raw_tags.get("raw_tokens"), list)
    assert raw_tags["raw_tokens"]
    assert raw_tags.get("ctc_compact_trace") is not None
    assert raw_tags["ctc_compact_trace"]["raw_tokens"] == raw_tags["raw_tokens"]
    assert raw_tags.get("ctc_frame_stride") == 0.06
