from __future__ import annotations

from pathlib import Path

import pytest

from app.utils.text_utils import parse_srt_content


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_JOB_DIR = _PROJECT_ROOT / "jobs" / "p-20260323-232740-tr-p01-80-mtz4"
_SRT_PATH = _JOB_DIR / "日本毒可乐连环杀人案，无差别随机杀人事件，日本历史著名悬案-p01-80.srt"


def _load_srt_segments() -> dict[int, dict]:
    content = _SRT_PATH.read_text(encoding="utf-8")
    return {int(item["index"]): item for item in parse_srt_content(content)}


@pytest.mark.xfail(strict=True, reason="#17-#19 当前仍存在连续重复字幕，先冻结为回归样本")
def test_job_p20260323_entries_17_to_19_should_not_repeat_same_text_three_times() -> None:
    segments = _load_srt_segments()

    texts = [str(segments[idx]["text"]) for idx in (17, 18, 19)]
    assert len(set(texts)) == 3


@pytest.mark.xfail(strict=True, reason="#77 当前仍存在重复弱标点")
def test_job_p20260323_entry_77_should_not_contain_double_comma() -> None:
    segments = _load_srt_segments()

    assert "，，" not in str(segments[77]["text"])


@pytest.mark.xfail(strict=True, reason="#79 当前仍存在小数点丢失与重复弱标点")
def test_job_p20260323_entry_79_should_preserve_decimal_points() -> None:
    segments = _load_srt_segments()
    text = str(segments[79]["text"])

    assert "0.15" in text
    assert "0.2" in text
    assert "，，" not in text


@pytest.mark.xfail(strict=True, reason="#81 当前仍存在异常连句，第三瓶描述未正确切开")
def test_job_p20260323_entry_81_should_not_swallow_next_sentence_start() -> None:
    segments = _load_srt_segments()
    text = str(segments[81]["text"])

    assert "第三瓶" not in text
