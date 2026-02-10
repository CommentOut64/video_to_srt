"""
同音索引业务服务。

V3.2.0+dev.20260210.02
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
import logging
import re
from typing import Dict, Iterable, List, Literal, Optional

from .db import GlobalTermRule, HomophoneDb, IndexState, PostingRecord
from .tokenizers import HomophoneTokenizer, Language


SearchMode = Literal["literal", "regex", "homophone_strict", "homophone_fuzzy"]


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HomophoneMatch:
    sentence_index: int
    token_index: int
    token_text: str
    char_start: int
    char_end: int
    reading_key: str
    cluster_id: str
    reading_label: str


@dataclass(frozen=True)
class SentenceRecord:
    index: int
    text: str


class HomophoneService:
    """同音检索服务（Phase 1）。"""

    def __init__(self, db: HomophoneDb, tokenizer: Optional[HomophoneTokenizer] = None) -> None:
        self._db = db
        self._tokenizer = tokenizer or HomophoneTokenizer()

    def index_chunk(
        self,
        *,
        job_id: str,
        revision: int,
        chunk_index: int,
        language: Language,
        sentences: Iterable[SentenceRecord],
    ) -> None:
        now = self._now_iso()
        self._db.upsert_index_state(
            IndexState(
                job_id=job_id,
                revision=revision,
                status="building",
                last_committed_chunk=max(-1, chunk_index - 1),
                heartbeat_at=now,
                updated_at=now,
            )
        )

        records: List[PostingRecord] = []
        for sentence in sentences:
            token_readings = self._tokenizer.tokenize(sentence.text, language)
            for token_index, token in enumerate(token_readings):
                records.append(
                    PostingRecord(
                        job_id=job_id,
                        revision=revision,
                        language=language,
                        chunk_index=chunk_index,
                        sentence_index=sentence.index,
                        token_index=token_index,
                        token_text=token.token_text,
                        reading_key=token.reading_key,
                        reading_key_fuzzy=token.reading_key_fuzzy,
                        reading_key_no_punct=token.reading_key_no_punct,
                        reading_key_fuzzy_no_punct=token.reading_key_fuzzy_no_punct,
                        char_start=token.char_start,
                        char_end=token.char_end,
                    )
                )

        self._db.replace_chunk_postings(records)
        now_done = self._now_iso()
        self._db.upsert_index_state(
            IndexState(
                job_id=job_id,
                revision=revision,
                status="ready",
                last_committed_chunk=chunk_index,
                heartbeat_at=now_done,
                updated_at=now_done,
            )
        )

    def search_homophone(
        self,
        *,
        job_id: str,
        revision: int,
        language: Language,
        query_text: str,
        mode: SearchMode,
        is_ignore_punctuation: bool,
        limit: int = 500,
    ) -> List[HomophoneMatch]:
        is_fuzzy = mode == "homophone_fuzzy"
        key_value = self._tokenizer.build_query_key(
            query_text=query_text,
            language=language,
            is_fuzzy=is_fuzzy,
        )
        if not key_value:
            return []

        rows = self._db.query_postings(
            job_id=job_id,
            revision=revision,
            language=language,
            key_value=key_value,
            is_fuzzy=is_fuzzy,
            is_ignore_punctuation=is_ignore_punctuation,
            limit=limit,
        )
        return [
            HomophoneMatch(
                sentence_index=int(row["sentence_index"]),
                token_index=int(row["token_index"]),
                token_text=str(row["token_text"]),
                char_start=int(row["char_start"]),
                char_end=int(row["char_end"]),
                reading_key=str(row["reading_key"]),
                cluster_id=self._cluster_id(str(row["reading_key"])),
                reading_label=str(row["reading_key"]),
            )
            for row in rows
        ]

    def apply_global_terms(
        self,
        *,
        text: str,
        language: str,
        is_modified: bool,
    ) -> str:
        """应用全局术语替换；用户已编辑句子默认跳过。"""
        if is_modified:
            return text
        rules = self._db.list_global_terms()
        output = text
        for rule in sorted(
            [r for r in rules if r.is_enabled and r.language in ("auto", language)],
            key=lambda item: (item.priority, -len(item.source_text)),
        ):
            if not rule.source_text:
                continue
            if rule.match_mode == "regex":
                try:
                    output = re.sub(rule.source_text, rule.target_text, output)
                except re.error:
                    logger.warning("全局术语正则无效，已跳过: %s", rule.source_text)
                    continue
            else:
                output = output.replace(rule.source_text, rule.target_text)
        return output

    def replace_global_terms(self, items: List[GlobalTermRule]) -> None:
        self._db.replace_global_terms(items, now_iso=self._now_iso())

    def list_global_terms(self) -> List[GlobalTermRule]:
        return self._db.list_global_terms()

    def get_index_status(self, job_id: str) -> Optional[IndexState]:
        return self._db.get_latest_index_state(job_id)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _cluster_id(reading_key: str) -> str:
        digest = sha1(reading_key.encode("utf-8")).hexdigest()
        return f"cluster_{digest[:10]}"
