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
from typing import Any, Dict, Iterable, List, Literal, Optional, Set, Tuple

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
        query_key_sequence, _ = self._build_query_sequence(
            query_text=query_text,
            language=language,
            is_fuzzy=is_fuzzy,
            is_ignore_punctuation=is_ignore_punctuation,
        )
        if not query_key_sequence:
            return []

        candidate_sentence_indices = self._collect_candidate_sentence_indices(
            job_id=job_id,
            revision=revision,
            language=language,
            query_key_sequence=query_key_sequence,
            is_fuzzy=is_fuzzy,
            is_ignore_punctuation=is_ignore_punctuation,
            limit=limit,
        )
        if not candidate_sentence_indices:
            return []

        sentence_rows = self._db.list_sentence_postings(
            job_id=job_id,
            revision=revision,
            language=language,
            sentence_indices=sorted(candidate_sentence_indices),
        )
        grouped_rows = self._group_rows_by_sentence(sentence_rows)

        matches: List[HomophoneMatch] = []
        for sentence_index in sorted(grouped_rows.keys()):
            token_rows = grouped_rows[sentence_index]
            sentence_matches = self._match_sequence_in_sentence(
                sentence_index=sentence_index,
                token_rows=token_rows,
                query_key_sequence=query_key_sequence,
                language=language,
                is_fuzzy=is_fuzzy,
                is_ignore_punctuation=is_ignore_punctuation,
            )
            matches.extend(sentence_matches)
            if len(matches) >= limit:
                break
        return matches[:limit]

    def _collect_candidate_sentence_indices(
        self,
        *,
        job_id: str,
        revision: int,
        language: Language,
        query_key_sequence: List[str],
        is_fuzzy: bool,
        is_ignore_punctuation: bool,
        limit: int,
    ) -> Set[int]:
        """多种子召回候选句子，降低漏召回风险。"""
        candidate_sentence_indices: Set[int] = set()
        seed_keys = [item for item in dict.fromkeys(query_key_sequence) if item]
        per_seed_limit = max(limit * 8, 1000)
        for seed_key in seed_keys:
            rows = self._db.query_postings(
                job_id=job_id,
                revision=revision,
                language=language,
                key_value=seed_key,
                is_fuzzy=is_fuzzy,
                is_ignore_punctuation=is_ignore_punctuation,
                limit=per_seed_limit,
            )
            candidate_sentence_indices.update(int(row["sentence_index"]) for row in rows)

        # 中文 fuzzy 兜底：按读音前缀扩召，覆盖 qing/qin、hua/huai 等近音误差。
        if is_fuzzy and language == "zh":
            prefix_limit = max(limit * 12, 2000)
            for key_prefix in self._build_fuzzy_prefixes(seed_keys):
                rows = self._db.query_postings_by_prefix(
                    job_id=job_id,
                    revision=revision,
                    language=language,
                    key_prefix=key_prefix,
                    is_fuzzy=is_fuzzy,
                    is_ignore_punctuation=is_ignore_punctuation,
                    limit=prefix_limit,
                )
                candidate_sentence_indices.update(int(row["sentence_index"]) for row in rows)

        return candidate_sentence_indices

    @staticmethod
    def _build_fuzzy_prefixes(keys: List[str]) -> List[str]:
        """构建中文 fuzzy 的前缀召回键。"""
        prefixes: Set[str] = set()
        for key in keys:
            normalized_key = re.sub(r"[^a-z]", "", str(key).lower())
            if len(normalized_key) < 2:
                continue
            prefix_length = 3 if len(normalized_key) >= 3 else 2
            prefixes.add(normalized_key[:prefix_length])
        return sorted(prefixes)

    @staticmethod
    def _group_rows_by_sentence(rows: List[Any]) -> Dict[int, List[Any]]:
        grouped: Dict[int, List[Any]] = {}
        for row in rows:
            sentence_index = int(row["sentence_index"])
            grouped.setdefault(sentence_index, []).append(row)
        return grouped

    def _build_query_sequence(
        self,
        *,
        query_text: str,
        language: Language,
        is_fuzzy: bool,
        is_ignore_punctuation: bool,
    ) -> Tuple[List[str], List[str]]:
        """构建查询读音序列（支持文字、拼音/罗马音、标点）。"""
        normalized_query = str(query_text or "").strip()
        if not normalized_query:
            return [], []

        lowered_query = normalized_query.lower()

        # 中文拼音输入：例如 zhong4 / zhong4 qi3。
        if language == "zh" and re.fullmatch(r"[a-z1-5\s]+", lowered_query):
            syllables = [item for item in re.split(r"\s+", lowered_query) if item]
            if syllables:
                if is_fuzzy:
                    syllables = [re.sub(r"[1-5]$", "", item) for item in syllables]
                return syllables, syllables

        # 日语罗马音/假名输入优先按整体词处理，保证 kouen / コウエン 可命中「公園/講演」。
        if language == "ja" and re.fullmatch(r"[A-Za-zぁ-んァ-ンー\-']+", normalized_query):
            whole_key = self._tokenizer.build_query_key(
                query_text=normalized_query,
                language=language,
                is_fuzzy=is_fuzzy,
            )
            if whole_key:
                return [whole_key], [normalized_query]

        query_tokens = self._tokenizer.tokenize(normalized_query, language)
        if not query_tokens:
            return [], []

        key_sequence: List[str] = []
        surface_sequence: List[str] = []
        for token in query_tokens:
            key_value = self._resolve_token_key(
                row=token,
                is_fuzzy=is_fuzzy,
                is_ignore_punctuation=is_ignore_punctuation,
            )
            if not key_value:
                continue
            key_sequence.append(key_value)
            surface_sequence.append(token.token_text)

        return key_sequence, surface_sequence

    @staticmethod
    def _read_field(row: Any, field_name: str) -> str:
        """读取 token 字段，兼容 dataclass 与 sqlite Row。"""
        if hasattr(row, field_name):
            return str(getattr(row, field_name))
        try:
            return str(row[field_name])
        except Exception:
            return ""

    @staticmethod
    def _resolve_token_key(
        *,
        row: Any,
        is_fuzzy: bool,
        is_ignore_punctuation: bool,
    ) -> str:
        """统一解析 token 对应的读音键。"""
        if is_fuzzy:
            if is_ignore_punctuation:
                return HomophoneService._read_field(row, "reading_key_fuzzy_no_punct")
            return HomophoneService._read_field(row, "reading_key_fuzzy")
        if is_ignore_punctuation:
            return HomophoneService._read_field(row, "reading_key_no_punct")
        return HomophoneService._read_field(row, "reading_key")

    def _match_sequence_in_sentence(
        self,
        *,
        sentence_index: int,
        token_rows: List[Any],
        query_key_sequence: List[str],
        language: Language,
        is_fuzzy: bool,
        is_ignore_punctuation: bool,
    ) -> List[HomophoneMatch]:
        """在单句中执行词级连续序列匹配（禁止跨字幕）。"""
        if not token_rows or not query_key_sequence:
            return []

        normalized_tokens: List[Any] = []
        for row in token_rows:
            key_value = self._resolve_token_key(
                row=row,
                is_fuzzy=is_fuzzy,
                is_ignore_punctuation=is_ignore_punctuation,
            )
            if not key_value:
                continue
            normalized_tokens.append(row)

        if not normalized_tokens:
            return []

        sentence_key_sequence = [
            self._resolve_token_key(
                row=row,
                is_fuzzy=is_fuzzy,
                is_ignore_punctuation=is_ignore_punctuation,
            )
            for row in normalized_tokens
        ]

        sentence_surface_sequence = [str(row["token_text"]) for row in normalized_tokens]

        result: List[HomophoneMatch] = []
        sequence_length = len(query_key_sequence)
        if sequence_length == 1:
            target_key = query_key_sequence[0]
            return [
                HomophoneMatch(
                    sentence_index=sentence_index,
                    token_index=int(row["token_index"]),
                    token_text=str(row["token_text"]),
                    char_start=int(row["char_start"]),
                    char_end=int(row["char_end"]),
                    reading_key=str(row["reading_key"]),
                    cluster_id=self._cluster_id(str(row["reading_key"])),
                    reading_label=str(row["reading_key"]),
                )
                for row in normalized_tokens
                if self._is_token_key_match(
                    sentence_key=self._resolve_token_key(
                        row=row,
                        is_fuzzy=is_fuzzy,
                        is_ignore_punctuation=is_ignore_punctuation,
                    ),
                    query_key=target_key,
                    language=language,
                    is_fuzzy=is_fuzzy,
                )
            ]

        for start in range(0, len(normalized_tokens) - sequence_length + 1):
            end = start + sequence_length
            window_keys = sentence_key_sequence[start:end]
            if window_keys != query_key_sequence:
                continue

            begin_row = normalized_tokens[start]
            end_row = normalized_tokens[end - 1]
            reading_label = "|".join(window_keys)
            result.append(
                HomophoneMatch(
                    sentence_index=sentence_index,
                    token_index=int(begin_row["token_index"]),
                    token_text="".join(sentence_surface_sequence[start:end]),
                    char_start=int(begin_row["char_start"]),
                    char_end=int(end_row["char_end"]),
                    reading_key=reading_label,
                    cluster_id=self._cluster_id(reading_label),
                    reading_label=reading_label,
                )
            )
        if result or not is_fuzzy:
            return result

        # fuzzy 兜底：仅允许同长度序列的近音替代。
        tolerant_result = self._match_fuzzy_with_tolerance(
            sentence_index=sentence_index,
            normalized_tokens=normalized_tokens,
            sentence_key_sequence=sentence_key_sequence,
            sentence_surface_sequence=sentence_surface_sequence,
            query_key_sequence=query_key_sequence,
            language=language,
        )
        return tolerant_result

    def _match_fuzzy_with_tolerance(
        self,
        *,
        sentence_index: int,
        normalized_tokens: List[Any],
        sentence_key_sequence: List[str],
        sentence_surface_sequence: List[str],
        query_key_sequence: List[str],
        language: Language,
    ) -> List[HomophoneMatch]:
        query_len = len(query_key_sequence)
        if query_len <= 1:
            return []

        matches: List[HomophoneMatch] = []
        # 说明：仅保留“近音替代”路径，停用“丢字/多字”容错窗口。
        # 若后续需要恢复，可复用 _is_sequence_match_with_single_gap。
        window_lengths = [query_len]
        seen_spans: Set[Tuple[int, int]] = set()
        for window_len in window_lengths:
            if window_len > len(sentence_key_sequence):
                continue
            for start in range(0, len(sentence_key_sequence) - window_len + 1):
                end = start + window_len
                window_keys = sentence_key_sequence[start:end]
                if not self._is_sequence_match_same_length(
                    sentence_window_keys=window_keys,
                    query_key_sequence=query_key_sequence,
                    language=language,
                ):
                    continue
                span = (start, end)
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                begin_row = normalized_tokens[start]
                end_row = normalized_tokens[end - 1]
                reading_label = "|".join(window_keys)
                matches.append(
                    HomophoneMatch(
                        sentence_index=sentence_index,
                        token_index=int(begin_row["token_index"]),
                        token_text="".join(sentence_surface_sequence[start:end]),
                        char_start=int(begin_row["char_start"]),
                        char_end=int(end_row["char_end"]),
                        reading_key=reading_label,
                        cluster_id=self._cluster_id(reading_label),
                        reading_label=reading_label,
                    )
                )
        return matches

    def _is_sequence_match_same_length(
        self,
        *,
        sentence_window_keys: List[str],
        query_key_sequence: List[str],
        language: Language,
    ) -> bool:
        """同长度序列匹配：允许近音替代，不允许丢字/多字。"""
        if len(sentence_window_keys) != len(query_key_sequence):
            return False
        return all(
            self._is_token_key_match(
                sentence_key=sentence_window_keys[index],
                query_key=query_key_sequence[index],
                language=language,
                is_fuzzy=True,
            )
            for index in range(len(query_key_sequence))
        )

    def _is_sequence_match_with_single_gap(
        self,
        *,
        sentence_window_keys: List[str],
        query_key_sequence: List[str],
        language: Language,
    ) -> bool:
        sentence_len = len(sentence_window_keys)
        query_len = len(query_key_sequence)
        if abs(sentence_len - query_len) > 1:
            return False

        if sentence_len == query_len:
            return all(
                self._is_token_key_match(
                    sentence_key=sentence_window_keys[index],
                    query_key=query_key_sequence[index],
                    language=language,
                    is_fuzzy=True,
                )
                for index in range(query_len)
            )

        if sentence_len == query_len - 1:
            for skip_index in range(query_len):
                candidate_query = (
                    query_key_sequence[:skip_index] + query_key_sequence[skip_index + 1:]
                )
                if all(
                    self._is_token_key_match(
                        sentence_key=sentence_window_keys[index],
                        query_key=candidate_query[index],
                        language=language,
                        is_fuzzy=True,
                    )
                    for index in range(sentence_len)
                ):
                    return True
            return False

        for skip_index in range(sentence_len):
            candidate_sentence = (
                sentence_window_keys[:skip_index] + sentence_window_keys[skip_index + 1:]
            )
            if all(
                self._is_token_key_match(
                    sentence_key=candidate_sentence[index],
                    query_key=query_key_sequence[index],
                    language=language,
                    is_fuzzy=True,
                )
                for index in range(query_len)
            ):
                return True
        return False

    def _is_token_key_match(
        self,
        *,
        sentence_key: str,
        query_key: str,
        language: Language,
        is_fuzzy: bool,
    ) -> bool:
        if sentence_key == query_key:
            return True
        if not is_fuzzy:
            return False
        if language != "zh":
            return False
        return self._is_zh_fuzzy_key_close(sentence_key=sentence_key, query_key=query_key)

    @staticmethod
    def _is_zh_fuzzy_key_close(*, sentence_key: str, query_key: str) -> bool:
        normalized_sentence = re.sub(r"[^a-z]", "", sentence_key.lower())
        normalized_query = re.sub(r"[^a-z]", "", query_key.lower())
        if not normalized_sentence or not normalized_query:
            return False
        if normalized_sentence[0] != normalized_query[0]:
            return False
        if abs(len(normalized_sentence) - len(normalized_query)) > 1:
            return False
        return HomophoneService._levenshtein_distance(normalized_sentence, normalized_query) <= 1

    @staticmethod
    def _levenshtein_distance(left: str, right: str) -> int:
        if left == right:
            return 0
        if not left:
            return len(right)
        if not right:
            return len(left)

        previous_row = list(range(len(right) + 1))
        for left_index, left_char in enumerate(left, start=1):
            current_row = [left_index]
            for right_index, right_char in enumerate(right, start=1):
                substitution_cost = 0 if left_char == right_char else 1
                current_row.append(
                    min(
                        previous_row[right_index] + 1,
                        current_row[right_index - 1] + 1,
                        previous_row[right_index - 1] + substitution_cost,
                    )
                )
            previous_row = current_row
        return previous_row[-1]

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
