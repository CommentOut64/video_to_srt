from __future__ import annotations

import importlib.util

import app.services.timeanchored_alignment as timeanchored_alignment


def test_legacy_slow_window_entrypoints_are_not_exported() -> None:
    assert not hasattr(timeanchored_alignment, "SlowWindowAssembler")
    assert not hasattr(timeanchored_alignment, "SlowWindowAssemblerConfig")
    assert not hasattr(timeanchored_alignment, "TurnGroupAdapter")


def test_legacy_slow_window_modules_are_removed() -> None:
    assert importlib.util.find_spec("app.services.timeanchored_alignment.slow_window_assembler") is None
    assert importlib.util.find_spec("app.services.timeanchored_alignment.turn_group_adapter") is None
