from __future__ import annotations

from app.models.sensevoice_models import SentenceSegment
from app.services.punctuation.semantic_buffer import SemanticChunk
from app.services.timeanchored_alignment.slow_window_assembler import (
    SlowWindowAssembler,
    SlowWindowAssemblerConfig,
)


def _build_chunk(
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


def _build_assembler() -> SlowWindowAssembler:
    return SlowWindowAssembler(
        config=SlowWindowAssemblerConfig(
            first_window_target_sec=4.0,
            steady_window_target_sec=8.0,
            long_pause_cut_sec=1.8,
        )
    )


def test_slow_window_assembler_hard_flush_on_speaker_change() -> None:
    assembler = _build_assembler()
    first = _build_chunk(chunk_id="chunk-1", text="你好。", language="zh", start=0.0, end=1.5, speaker_id="spk-a")
    second = _build_chunk(chunk_id="chunk-2", text="继续。", language="zh", start=1.5, end=3.0, speaker_id="spk-b")

    assert assembler.add_chunk(first, speaker_id="spk-a", turn_id="turn-a") == []
    outputs = assembler.add_chunk(second, speaker_id="spk-b", turn_id="turn-b")

    assert len(outputs) == 1
    assert outputs[0].window.metadata["flush_reason"] == "speaker_change"


def test_slow_window_assembler_hard_flush_on_language_change() -> None:
    assembler = _build_assembler()
    first = _build_chunk(chunk_id="chunk-1", text="这是中文。", language="zh", start=0.0, end=1.5)
    second = _build_chunk(chunk_id="chunk-2", text="This is English.", language="en", start=1.5, end=3.0)

    assert assembler.add_chunk(first, speaker_id="spk-a", turn_id="turn-a") == []
    outputs = assembler.add_chunk(second, speaker_id="spk-a", turn_id="turn-a")

    assert len(outputs) == 1
    assert outputs[0].window.metadata["flush_reason"] == "language_change"


def test_slow_window_assembler_hard_flush_on_long_pause() -> None:
    assembler = _build_assembler()
    first = _build_chunk(chunk_id="chunk-1", text="测试。", language="zh", start=0.0, end=1.0)
    second = _build_chunk(chunk_id="chunk-2", text="继续测试。", language="zh", start=3.2, end=4.0)

    assert assembler.add_chunk(first, speaker_id="spk-a", turn_id="turn-a") == []
    outputs = assembler.add_chunk(second, speaker_id="spk-a", turn_id="turn-a")

    assert len(outputs) == 1
    assert outputs[0].window.metadata["flush_reason"] == "long_pause_cut"


def test_slow_window_assembler_switches_bootstrap_to_steady_window() -> None:
    assembler = _build_assembler()
    chunks = [
        _build_chunk(chunk_id="chunk-1", text="第一段。", language="zh", start=0.0, end=2.0),
        _build_chunk(chunk_id="chunk-2", text="第二段。", language="zh", start=2.0, end=4.0),
        _build_chunk(chunk_id="chunk-3", text="第三段。", language="zh", start=4.0, end=6.0),
        _build_chunk(chunk_id="chunk-4", text="第四段。", language="zh", start=6.0, end=8.0),
        _build_chunk(chunk_id="chunk-5", text="第五段。", language="zh", start=8.0, end=10.0),
        _build_chunk(chunk_id="chunk-6", text="第六段。", language="zh", start=10.0, end=12.0),
    ]

    first_outputs = []
    for chunk in chunks[:2]:
        first_outputs.extend(assembler.add_chunk(chunk, speaker_id="spk-a", turn_id="turn-a"))
    assert len(first_outputs) == 1
    assert first_outputs[0].window.metadata["window_mode"] == "bootstrap"

    steady_outputs = []
    for chunk in chunks[2:]:
        steady_outputs.extend(assembler.add_chunk(chunk, speaker_id="spk-a", turn_id="turn-a"))
    assert len(steady_outputs) == 1
    assert steady_outputs[0].window.metadata["window_mode"] == "steady"


def test_slow_window_assembler_mixed_window_not_enter_main_chain() -> None:
    assembler = _build_assembler()
    chunks = [
        _build_chunk(chunk_id="chunk-1", text="你好 world", language="mixed", start=0.0, end=2.0),
        _build_chunk(chunk_id="chunk-2", text="こんにちは hello", language="mixed", start=2.0, end=4.0),
    ]
    outputs = []
    for chunk in chunks:
        outputs.extend(assembler.add_chunk(chunk, speaker_id="spk-a", turn_id="turn-a"))

    envelope = outputs[0] if outputs else assembler.flush(reason="eof_flush")
    assert envelope is not None
    assert envelope.window.metadata["is_mixed_window"] is True
    assert envelope.window.metadata["route"] == "fallback"
    assert assembler.can_enter_main_chain(envelope) is False
