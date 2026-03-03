"""
同音服务运行时编排层。

V3.2.0+dev.20260210.03
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
import logging
from pathlib import Path
import re
import threading
from typing import Iterable, List, Optional

from app.core.config import config

from .db import HomophoneDb, IndexState
from .service import HomophoneService, SentenceRecord
from .tokenizers import Language


logger = logging.getLogger(__name__)


class HomophoneRuntime:
    """同音服务运行时封装（单例 + 异步索引侧车）。"""

    def __init__(
        self,
        *,
        db_path: Optional[Path] = None,
        max_workers: int = 2,
    ) -> None:
        resolved_db_path = db_path or (config.JOBS_DIR / "_shared" / "homophone.db")
        self._db = HomophoneDb(resolved_db_path)
        self._service = HomophoneService(self._db)
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="homophone-index",
        )

    @property
    def service(self) -> HomophoneService:
        return self._service

    def index_chunk_async(
        self,
        *,
        project_id: str,
        revision: int,
        chunk_index: int,
        language_hint: str,
        sentences: Iterable[SentenceRecord],
    ) -> Future[None]:
        sentence_list = list(sentences)
        language = self._normalize_language(language_hint, sentence_list)
        return self._executor.submit(
            self._index_chunk_safe,
            project_id,
            revision,
            chunk_index,
            language,
            sentence_list,
        )

    def _index_chunk_safe(
        self,
        project_id: str,
        revision: int,
        chunk_index: int,
        language: Language,
        sentences: List[SentenceRecord],
    ) -> None:
        try:
            self._service.index_chunk(
                project_id=project_id,
                revision=revision,
                chunk_index=chunk_index,
                language=language,
                sentences=sentences,
            )
        except Exception:
            now = self._now_iso()
            # 设计说明：此处显式标记失败状态，避免索引中断后出现“ready”假象。
            self._db.upsert_index_state(
                IndexState(
                    project_id=project_id,
                    revision=revision,
                    status="failed",
                    last_committed_chunk=max(-1, chunk_index - 1),
                    heartbeat_at=now,
                    updated_at=now,
                )
            )
            logger.exception(
                "同音索引异步构建失败: project_id=%s revision=%s chunk_index=%s",
                project_id,
                revision,
                chunk_index,
            )

    @staticmethod
    def _normalize_language(language_hint: str, sentences: List[SentenceRecord]) -> Language:
        normalized = (language_hint or "").strip().lower()
        if normalized in {"zh", "ja", "en"}:
            return normalized  # type: ignore[return-value]
        detected = HomophoneRuntime._detect_language(sentences)
        if detected in {"zh", "ja", "en"}:
            return detected
        return "zh"

    @staticmethod
    def _detect_language(sentences: List[SentenceRecord]) -> str:
        if not sentences:
            return "zh"
        text = "\n".join(sentence.text for sentence in sentences if sentence.text)
        if not text:
            return "zh"
        if re.search(r"[ぁ-んァ-ン]", text):
            return "ja"
        if re.search(r"[\u4e00-\u9fff]", text):
            return "zh"
        if re.search(r"[A-Za-z]", text):
            return "en"
        return "zh"

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


_runtime_lock = threading.Lock()
_runtime_instance: Optional[HomophoneRuntime] = None


def get_homophone_runtime() -> HomophoneRuntime:
    global _runtime_instance
    if _runtime_instance is None:
        with _runtime_lock:
            if _runtime_instance is None:
                _runtime_instance = HomophoneRuntime()
    return _runtime_instance


def get_homophone_service() -> HomophoneService:
    return get_homophone_runtime().service


def index_chunk_async(
    *,
    project_id: str,
    revision: int,
    chunk_index: int,
    language_hint: str,
    sentences: Iterable[SentenceRecord],
) -> Future[None]:
    return get_homophone_runtime().index_chunk_async(
        project_id=project_id,
        revision=revision,
        chunk_index=chunk_index,
        language_hint=language_hint,
        sentences=sentences,
    )

