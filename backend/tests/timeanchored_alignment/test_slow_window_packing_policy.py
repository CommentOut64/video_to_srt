from __future__ import annotations

from app.services.timeanchored_alignment.slow_window.packing_policy import PackingPolicy


def test_packing_policy_fragmented_dialogue_can_emit_before_single_speaker_target() -> None:
    policy = PackingPolicy(
        first_window_target_sec=4.0,
        steady_window_target_sec=12.0,
        steady_two_party_target_sec=9.0,
        steady_fragmented_target_sec=5.0,
        hard_max_window_sec=16.0,
        ready_queue_low_watermark=0,
        ready_queue_target_depth=1,
        ready_queue_high_watermark=2,
    )

    single = policy.evaluate(
        window_count=1,
        duration_sec=5.1,
        dialogue_shape="single_speaker",
        has_candidate_cut=True,
        ready_queue_depth=0,
    )
    fragmented = policy.evaluate(
        window_count=1,
        duration_sec=5.1,
        dialogue_shape="ping_pong_fragmented",
        has_candidate_cut=True,
        ready_queue_depth=0,
    )

    assert single.should_emit is False
    assert fragmented.should_emit is True


def test_packing_policy_uses_queue_pressure_to_promote_candidate_cut_window() -> None:
    policy = PackingPolicy(
        first_window_target_sec=4.0,
        steady_window_target_sec=12.0,
        steady_two_party_target_sec=9.0,
        steady_fragmented_target_sec=5.0,
        hard_max_window_sec=16.0,
        ready_queue_low_watermark=0,
        ready_queue_target_depth=1,
        ready_queue_high_watermark=2,
    )

    low_pressure = policy.evaluate(
        window_count=1,
        duration_sec=10.2,
        dialogue_shape="single_speaker",
        has_candidate_cut=True,
        ready_queue_depth=0,
    )
    high_pressure = policy.evaluate(
        window_count=1,
        duration_sec=10.2,
        dialogue_shape="single_speaker",
        has_candidate_cut=True,
        ready_queue_depth=2,
    )

    assert low_pressure.should_emit is False
    assert high_pressure.should_emit is True
