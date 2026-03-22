from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.services.language_policy import build_language_policy_snapshot


def _load_language_literal_values() -> set[str]:
    tokenizers_path = Path(__file__).resolve().parents[2] / "app" / "services" / "homophone" / "tokenizers.py"
    module = ast.parse(tokenizers_path.read_text(encoding="utf-8"), filename=str(tokenizers_path))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "Language":
                    value = node.value
                    if isinstance(value, ast.Subscript) and getattr(value.value, "id", "") == "Literal":
                        literal_values: set[str] = set()
                        elements = []
                        if isinstance(value.slice, ast.Tuple):
                            elements = list(value.slice.elts)
                        else:
                            elements = [value.slice]
                        for item in elements:
                            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                                literal_values.add(item.value)
                        return literal_values
    raise AssertionError("未在 tokenizers.py 中找到 Language = Literal[...] 定义")


def test_scope_contract_first_release_languages_are_zh_ja_en() -> None:
    supported = _load_language_literal_values()
    assert supported == {"zh", "ja", "en"}


@pytest.mark.xfail(
    reason="Phase 3/7 目标：mixed 作为显式状态治理，不应静默回落到 zh",
    strict=False,
)
def test_scope_contract_mixed_should_not_enter_primary_language_chain() -> None:
    snapshot = build_language_policy_snapshot(language_hint="mixed")
    assert snapshot.language_tag == "mixed"


@pytest.mark.xfail(
    reason="Phase 3 目标：yue/ko 不能默认映射到 zh 文本/发音主链",
    strict=False,
)
@pytest.mark.parametrize("lang", ["yue", "ko"])
def test_scope_contract_yue_ko_should_not_fallback_to_zh(lang: str) -> None:
    snapshot = build_language_policy_snapshot(language_hint=lang)
    assert snapshot.language_tag == lang
