from __future__ import annotations

from app.services.timeanchored_alignment.language_run_frontend import LanguageRunFrontend


def test_language_run_frontend_zh_with_en_islands() -> None:
    frontend = LanguageRunFrontend()
    result = frontend.build_runs(
        text="今天发布 v3.3.0 的 U.S. state-of-the-art 模型，don't 拆碎。",
        language_hint="zh",
    )

    assert result.dominant_language == "zh"
    assert result.window_kind == "dominant_with_islands"
    assert result.foreign_run_ratio > 0.0

    run_texts = [run.run_text for run in result.runs]
    assert "v3.3.0" in run_texts
    assert "U.S." in run_texts
    assert "state-of-the-art" in run_texts
    assert "don't" in run_texts

    for item in result.runs:
        if item.run_text in {"v3.3.0", "U.S.", "state-of-the-art", "don't"}:
            assert item.is_protected is True
            assert item.run_language == "en"
            assert item.is_foreign_island is True


def test_language_run_frontend_ja_with_en_island_and_middle_dot() -> None:
    frontend = LanguageRunFrontend()
    result = frontend.build_runs(
        text="今日は OpenAI・API について話す",
        language_hint="ja",
    )

    assert result.dominant_language == "ja"
    assert result.window_kind == "dominant_with_islands"

    run_map = {run.run_text: run for run in result.runs}
    assert run_map["OpenAI"].run_language == "en"
    assert run_map["OpenAI"].is_foreign_island is True
    assert run_map["・"].is_protected is True


def test_language_run_frontend_true_mixed_window() -> None:
    frontend = LanguageRunFrontend()
    result = frontend.build_runs(
        text="Hello 你好 こんにちは",
        language_hint="auto",
    )

    assert result.window_kind == "true_mixed"
    assert result.dominant_language == "mixed"
