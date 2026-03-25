from __future__ import annotations

from app.models.sensevoice_models import SentenceSegment
from app.services.punctuation.semantic_buffer import SemanticChunk
from app.services.timeanchored_alignment.slow_window.ingress_adapter import SlowWindowIngressAdapter
from app.services.timeanchored_alignment.slow_window.builder import SlowWindowBuilder, SlowWindowBuilderConfig
from app.services.timeanchored_alignment.slow_window.contracts import ReadySlowWindow, SlowWindowIngressUnit


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


def _build_builder() -> SlowWindowBuilder:
    return SlowWindowBuilder(
        config=SlowWindowBuilderConfig(
            first_window_target_sec=4.0,
            steady_window_target_sec=8.0,
            hard_max_window_sec=12.0,
            tail_idle_sec=1.0,
        )
    )


def _build_ingress(
    *,
    chunk_id: str,
    text: str,
    language: str,
    start: float,
    end: float,
    speaker_id: str = "spk-1",
    turn_id: str = "turn-a",
) -> SlowWindowIngressUnit:
    adapter = SlowWindowIngressAdapter()
    return adapter.adapt(
        _build_chunk(
            chunk_id=chunk_id,
            text=text,
            language=language,
            start=start,
            end=end,
            speaker_id=speaker_id,
        ),
        speaker_id=speaker_id,
        turn_id=turn_id,
        source_chunk_indices=(int(chunk_id.split("-")[-1]),),
        arrived_at=start,
    )


def test_candidate_cuts_do_not_force_flush_by_default() -> None:
    builder = _build_builder()

    assert builder.add_chunk(_build_ingress(chunk_id="chunk-1", text="你好", language="zh", start=0.0, end=1.0, speaker_id="spk-a", turn_id="turn-a")) == []
    assert builder.add_chunk(_build_ingress(chunk_id="chunk-2", text="继续", language="zh", start=2.9, end=3.6, speaker_id="spk-a", turn_id="turn-a")) == []
    assert builder.add_chunk(_build_ingress(chunk_id="chunk-3", text="hello", language="en", start=3.6, end=3.9, speaker_id="spk-b", turn_id="turn-b")) == []
    assert builder.pending_candidate_cut_reasons == ("long_pause", "language_shift", "speaker_change")

    ready = builder.flush(reason="eof_flush")

    assert ready is not None
    assert ready.flush_reason == "eof_flush"


def test_builder_outputs_ready_slow_window_instead_of_turn_group_envelope() -> None:
    builder = _build_builder()

    outputs = builder.add_chunk(
        _build_ingress(chunk_id="chunk-7", text="第一段。第二段。", language="zh", start=0.0, end=4.2, speaker_id="spk-a", turn_id="turn-a"),
    )

    assert len(outputs) == 1
    assert isinstance(outputs[0], ReadySlowWindow)


def test_ready_window_keeps_complete_source_chunk_ids_and_indices() -> None:
    builder = _build_builder()

    builder.add_chunk(
        _build_ingress(chunk_id="chunk-11", text="第一段", language="zh", start=0.0, end=2.0, speaker_id="spk-a", turn_id="turn-a"),
    )
    outputs = builder.add_chunk(
        _build_ingress(chunk_id="chunk-12", text="第二段", language="zh", start=2.0, end=4.4, speaker_id="spk-a", turn_id="turn-a"),
    )

    ready = outputs[0]
    assert ready.source_chunk_ids == ("chunk-11", "chunk-12")
    assert ready.source_chunk_indices == (11, 12)
    assert tuple(binding.chunk_index for binding in ready.coverage.chunk_bindings) == (11, 12)


def test_ready_window_coverage_marks_edge_context_as_guards() -> None:
    builder = SlowWindowBuilder(
        config=SlowWindowBuilderConfig(
            first_window_target_sec=6.0,
            steady_window_target_sec=8.0,
            hard_max_window_sec=12.0,
            tail_idle_sec=1.0,
        )
    )

    assert builder.add_chunk(
        _build_ingress(
            chunk_id="chunk-1",
            text="前文",
            language="zh",
            start=0.0,
            end=1.0,
            speaker_id="spk-a",
            turn_id="turn-a",
        )
    ) == []
    assert builder.add_chunk(
        _build_ingress(
            chunk_id="chunk-2",
            text="主体内容",
            language="zh",
            start=1.0,
            end=5.0,
            speaker_id="spk-a",
            turn_id="turn-a",
        )
    ) == []
    outputs = builder.add_chunk(
        _build_ingress(
            chunk_id="chunk-3",
            text="后文",
            language="zh",
            start=5.0,
            end=6.0,
            speaker_id="spk-a",
            turn_id="turn-a",
        )
    )

    assert len(outputs) == 1
    ready = outputs[0]
    roles = {binding.chunk_index: binding.role for binding in ready.coverage.chunk_bindings}
    assert roles == {1: "left_guard", 2: "owner", 3: "right_guard"}
    assert ready.coverage.left_guard_sec == 1.0
    assert ready.coverage.right_guard_sec == 1.0
    assert ready.coverage.core_segments == ((1.0, 5.0),)


def test_fragmented_dialogue_uses_shorter_steady_target_than_single_speaker() -> None:
    builder = SlowWindowBuilder(
        config=SlowWindowBuilderConfig(
            first_window_target_sec=4.0,
            steady_window_target_sec=12.0,
            hard_max_window_sec=16.0,
            tail_idle_sec=1.0,
        )
    )

    first = builder.add_chunk(
        _build_ingress(chunk_id="chunk-0", text="起", language="zh", start=0.0, end=4.2, speaker_id="spk-a", turn_id="turn-a"),
    )
    assert len(first) == 1
    assert first[0].window_mode == "bootstrap"

    assert builder.add_chunk(
        _build_ingress(chunk_id="chunk-1", text="甲", language="zh", start=4.2, end=5.3, speaker_id="spk-a", turn_id="turn-a"),
    ) == []
    assert builder.add_chunk(
        _build_ingress(chunk_id="chunk-2", text="乙", language="zh", start=5.3, end=6.4, speaker_id="spk-b", turn_id="turn-b"),
    ) == []
    assert builder.add_chunk(
        _build_ingress(chunk_id="chunk-3", text="丙", language="zh", start=6.4, end=7.5, speaker_id="spk-a", turn_id="turn-c"),
    ) == []
    outputs = builder.add_chunk(
        _build_ingress(chunk_id="chunk-4", text="丁", language="zh", start=7.5, end=9.4, speaker_id="spk-b", turn_id="turn-d"),
    )

    assert len(outputs) == 1
    assert outputs[0].window_mode == "steady"
    assert outputs[0].dialogue_shape.shape == "ping_pong_fragmented"


def test_builder_runtime_path_prefers_explicit_ingress_without_chunk_id_reparse() -> None:
    builder = SlowWindowBuilder(
        config=SlowWindowBuilderConfig(
            first_window_target_sec=2.0,
            steady_window_target_sec=8.0,
            hard_max_window_sec=12.0,
            tail_idle_sec=1.0,
        )
    )

    ingress = SlowWindowIngressUnit(
        semantic_chunk_id="semantic-1",
        text="你好",
        sentences=(),
        punctuation_decision=None,
        audio_range=(0.0, 2.2),
        language="zh",
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        speaker_id="spk-a",
        turn_id="turn-a",
        arrived_at=0.0,
    )

    outputs = builder.add_chunk(ingress)

    assert len(outputs) == 1
    assert outputs[0].source_chunk_indices == (1,)


def test_steady_window_batch_hint_reflects_ready_queue_pressure() -> None:
    def _emit_second_window(*, ready_queue_depth: int) -> ReadySlowWindow:
        builder = SlowWindowBuilder(
            config=SlowWindowBuilderConfig(
                first_window_target_sec=2.0,
                steady_window_target_sec=4.0,
                hard_max_window_sec=8.0,
                tail_idle_sec=1.0,
            )
        )

        first = builder.add_chunk(
            _build_ingress(
                chunk_id="chunk-0",
                text="起",
                language="zh",
                start=0.0,
                end=2.2,
                speaker_id="spk-a",
                turn_id="turn-a",
            ),
            ready_queue_depth=0,
        )
        assert len(first) == 1
        builder.mark_ready_window_dequeued()

        second = builder.add_chunk(
            _build_ingress(
                chunk_id="chunk-1",
                text="继续",
                language="zh",
                start=2.2,
                end=6.4,
                speaker_id="spk-a",
                turn_id="turn-a",
            ),
            ready_queue_depth=ready_queue_depth,
        )
        assert len(second) == 1
        return second[0]

    low_pressure = _emit_second_window(ready_queue_depth=0)
    high_pressure = _emit_second_window(ready_queue_depth=2)

    assert low_pressure.window_mode == "steady"
    assert high_pressure.window_mode == "steady"
    assert high_pressure.batch_hint.queue_priority > low_pressure.batch_hint.queue_priority
