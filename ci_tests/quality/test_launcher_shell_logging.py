from __future__ import annotations

from pathlib import Path

from launcher import shell_manager


def test_launch_electron_injects_chromium_logging_args(tmp_path: Path, monkeypatch) -> None:
    shell_path = tmp_path / "core" / "shell" / "AnchorFluxShell.exe"
    shell_path.parent.mkdir(parents=True, exist_ok=True)
    shell_path.write_bytes(b"stub")

    captured: dict[str, object] = {}

    class DummyProcess:
        pid = 9527

    def fake_popen(cmd, cwd=None, env=None):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        return DummyProcess()

    monkeypatch.setattr(shell_manager.subprocess, "Popen", fake_popen)

    proc = shell_manager.launch_electron(shell_path)
    assert proc is not None

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "--enable-logging" in cmd

    log_arg = next((item for item in cmd if isinstance(item, str) and item.startswith("--log-file=")), None)
    assert log_arg is not None
    log_file_path = Path(str(log_arg).split("=", 1)[1])

    expected_log_dir = tmp_path / "logs"
    assert log_file_path.parent == expected_log_dir
    assert log_file_path.name.startswith("electron-chromium-")
    assert log_file_path.suffix == ".log"

    env = captured["env"]
    assert isinstance(env, dict)
    assert env.get("ANCHORFLUX_SHELL_LOG_DIR") == str(expected_log_dir)
