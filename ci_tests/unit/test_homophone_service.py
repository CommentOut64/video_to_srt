"""同音服务核心单元测试。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from app.services.homophone.db import GlobalTermRule, HomophoneDb
from app.services.homophone.service import HomophoneService, SentenceRecord


@dataclass(frozen=True)
class _Token:
    token_text: str
    reading_key: str
    reading_key_fuzzy: str
    reading_key_no_punct: str
    reading_key_fuzzy_no_punct: str
    char_start: int
    char_end: int


class _FakeTokenizer:
    def __init__(self, mapping: Dict[str, List[_Token]]) -> None:
        self._mapping = mapping

    def tokenize(self, text: str, language: str) -> List[_Token]:
        return list(self._mapping.get(text, []))

    def build_query_key(self, query_text: str, language: str, is_fuzzy: bool) -> str:
        tokens = self.tokenize(query_text, language)
        if not tokens:
            return ""
        token = tokens[0]
        return token.reading_key_fuzzy if is_fuzzy else token.reading_key


def _build_service(tmp_path: Path, mapping: Dict[str, List[_Token]]) -> HomophoneService:
    db = HomophoneDb(tmp_path / "homophone_test.db")
    tokenizer = _FakeTokenizer(mapping)
    return HomophoneService(db=db, tokenizer=tokenizer)


def test_index_and_search_strict_returns_expected_match(tmp_path: Path) -> None:
    mapping = {
        "甲": [_Token("甲", "jia", "jia", "jia", "jia", 0, 1)],
        "乙": [_Token("乙", "yi", "yi", "yi", "yi", 0, 1)],
    }
    service = _build_service(tmp_path, mapping)
    service.index_chunk(
        project_id="project-1",
        revision=1,
        chunk_index=0,
        language="zh",
        sentences=[SentenceRecord(index=0, text="甲"), SentenceRecord(index=1, text="乙")],
    )

    status = service.get_index_status("project-1")
    assert status is not None
    assert status.status == "ready"
    assert status.last_committed_chunk == 0

    matches = service.search_homophone(
        project_id="project-1",
        revision=1,
        language="zh",
        query_text="甲",
        mode="homophone_strict",
        is_ignore_punctuation=False,
        limit=10,
    )
    assert len(matches) == 1
    assert matches[0].sentence_index == 0
    assert matches[0].token_text == "甲"


def test_search_never_matches_across_sentences(tmp_path: Path) -> None:
    mapping = {
        "甲": [_Token("甲", "jia", "jia", "jia", "jia", 0, 1)],
        "乙": [_Token("乙", "yi", "yi", "yi", "yi", 0, 1)],
        "甲乙": [
            _Token("甲", "jia", "jia", "jia", "jia", 0, 1),
            _Token("乙", "yi", "yi", "yi", "yi", 1, 2),
        ],
    }
    service = _build_service(tmp_path, mapping)
    service.index_chunk(
        project_id="project-2",
        revision=1,
        chunk_index=0,
        language="zh",
        sentences=[SentenceRecord(index=0, text="甲"), SentenceRecord(index=1, text="乙")],
    )

    matches = service.search_homophone(
        project_id="project-2",
        revision=1,
        language="zh",
        query_text="甲乙",
        mode="homophone_strict",
        is_ignore_punctuation=False,
        limit=10,
    )
    assert matches == []


def test_global_terms_replace_list_and_apply(tmp_path: Path) -> None:
    mapping: Dict[str, List[_Token]] = {}
    service = _build_service(tmp_path, mapping)
    rules = [
        GlobalTermRule(
            language="auto",
            source_text="foo",
            target_text="bar",
            match_mode="exact",
            priority=10,
            is_enabled=True,
            note="",
        ),
        GlobalTermRule(
            language="zh",
            source_text="A\\s+B",
            target_text="AB",
            match_mode="regex",
            priority=20,
            is_enabled=True,
            note="",
        ),
    ]
    service.replace_global_terms(rules)

    listed = service.list_global_terms()
    assert len(listed) == 2
    assert listed[0].source_text == "foo"
    assert listed[1].source_text == "A\\s+B"

    assert service.apply_global_terms(text="foo hello", language="en", is_modified=False) == "bar hello"
    assert service.apply_global_terms(text="A   B", language="zh", is_modified=False) == "AB"
    assert service.apply_global_terms(text="foo", language="en", is_modified=True) == "foo"
