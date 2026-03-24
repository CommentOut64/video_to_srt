"""
流式字幕管理系统

核心职责：
1. 管理字幕句子列表（支持原地更新）
2. 协调 SSE 事件推送（统一 Tag）
3. 支持多阶段增量更新（SV → Whisper → LLM）
"""
import copy
import hashlib
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.models.sensevoice_models import SentenceSegment, TextSource
from app.services.sse_service import get_sse_manager
from app.services.subtitle_visibility import is_hidden_unknown_sentence
from app.services.textflow.contracts import SubtitleBatch, SubtitleItem
import logging

logger = logging.getLogger(__name__)


def push_subtitle_event(
    sse_manager,
    job_id: str,
    event_type: str,
    data: dict,
    project_id: Optional[str] = None,
):
    """
    推送字幕事件（统一封装）

    Args:
        sse_manager: SSE 管理器
        job_id: 运行态任务 ID（兼容字段）
        event_type: 事件类型
        data: 事件数据
    """
    channel_identifier = str(project_id or job_id)
    sse_manager.broadcast_sync(
        f"project:{channel_identifier}",
        f"subtitle.{event_type}",
        data,
    )


class StreamingSubtitleManager:
    """流式字幕管理器"""

    def __init__(self, job_id: str, project_id: Optional[str] = None):
        self.job_id = job_id
        self.project_id = project_id
        self.sentences: Dict[int, SentenceSegment] = {}  # key = sentence_index
        self.sentence_count = 0
        self.sse_manager = get_sse_manager()

        # Phase 4: Chunk 级别的句子索引映射
        # chunk_sentences[chunk_index] = [sentence_index_1, sentence_index_2, ...]
        self.chunk_sentences: Dict[Any, List[int]] = {}

        # V3.8: 添加锁保护，防止 remove_marked_sentences 在错误时机执行
        self._lock = threading.RLock()
        # V3.8: 标记是否允许删除句子
        self._deletion_enabled = False
        # V3.2.4+dev.20260228.01: 运行态字幕真源持久化（project runtime_state.db）。
        self._runtime_checkpoint_service: Optional[Any] = None
        self._runtime_checkpoint_db_path: Optional[Path] = None

    def bind_runtime_checkpoint_service(self, job_dir: Path) -> None:
        """绑定 runtime checkpoint 服务，用于持久化运行态字幕真源。"""
        from app.services.checkpoint import RuntimeCheckpointService

        normalized_job_dir = Path(job_dir)
        runtime_db_path = normalized_job_dir / "runtime_state.db"
        if (
            self._runtime_checkpoint_service is not None
            and self._runtime_checkpoint_db_path == runtime_db_path
        ):
            return
        self._runtime_checkpoint_service = RuntimeCheckpointService(job_dir=normalized_job_dir)
        self._runtime_checkpoint_db_path = runtime_db_path
        self._persist_runtime_subtitle_state(reason="bind_runtime_service")

    @staticmethod
    def _normalize_chunk_ref(chunk_ref: Any) -> Any:
        """标准化 chunk 引用（支持 int chunk_index 和 string chunk_id）。"""
        if isinstance(chunk_ref, int):
            return chunk_ref
        if isinstance(chunk_ref, str):
            normalized = chunk_ref.strip()
            if normalized.lstrip("-").isdigit():
                try:
                    return int(normalized)
                except ValueError:
                    return normalized
            if normalized.startswith("chunk-"):
                suffix = normalized.split("-")[-1]
                if suffix.lstrip("-").isdigit():
                    try:
                        return int(suffix)
                    except ValueError:
                        return normalized
            return normalized
        if chunk_ref is None:
            return "chunk:unknown"
        return str(chunk_ref)

    @staticmethod
    def _try_parse_chunk_index(chunk_ref: Any) -> Optional[int]:
        """尝试从 chunk_ref 解析 legacy chunk_index。"""
        if isinstance(chunk_ref, int):
            return chunk_ref
        chunk_ref_text = str(chunk_ref or "").strip()
        if not chunk_ref_text:
            return None
        if chunk_ref_text.lstrip("-").isdigit():
            try:
                return int(chunk_ref_text)
            except ValueError:
                return None
        if chunk_ref_text.startswith("chunk-"):
            try:
                return int(chunk_ref_text.split("-")[-1])
            except ValueError:
                return None
        return None

    def _iter_chunk_alias_keys(self, chunk_ref: Any) -> List[Any]:
        """
        返回与 chunk_ref 等价的所有映射键（含历史别名与语义后缀键）。
        """
        normalized_ref = self._normalize_chunk_ref(chunk_ref)
        alias_keys: List[Any] = []
        seen = set()

        def _add_key(key: Any) -> None:
            marker = (type(key).__name__, str(key))
            if marker in seen:
                return
            seen.add(marker)
            alias_keys.append(key)

        _add_key(normalized_ref)
        _add_key(str(normalized_ref))

        parsed_index = self._try_parse_chunk_index(normalized_ref)
        if parsed_index is not None:
            _add_key(parsed_index)
            _add_key(str(parsed_index))
            _add_key(f"chunk-{parsed_index}")

            for existing_key in list(self.chunk_sentences.keys()):
                key_text = str(existing_key)
                if key_text.startswith(f"{parsed_index}#"):
                    _add_key(existing_key)
                    continue
                if key_text.startswith(f"chunk-{parsed_index}#"):
                    _add_key(existing_key)
                    continue
                if key_text.startswith(f"chunk-{parsed_index}+"):
                    _add_key(existing_key)
                    continue
                if f"+chunk-{parsed_index}" in key_text:
                    _add_key(existing_key)

        return alias_keys

    def _remove_chunk_alias_mappings(self, chunk_ref: Any) -> None:
        """删除 chunk_ref 对应的所有别名映射，避免旧键残留。"""
        for alias_key in self._iter_chunk_alias_keys(chunk_ref):
            if alias_key in self.chunk_sentences:
                del self.chunk_sentences[alias_key]

    def _build_sentence_uid(
        self,
        *,
        chunk_ref: Any,
        sentence: SentenceSegment,
    ) -> str:
        """构建稳定 sentence_uid。"""
        chunk_key = str(self._normalize_chunk_ref(chunk_ref))
        stable_text = str(sentence.text_clean or sentence.text or "")
        seed = (
            f"{self.project_id or self.job_id}|{chunk_key}|"
            f"{float(sentence.start):.3f}|{float(sentence.end):.3f}|{stable_text}"
        )
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
        return f"seg-{digest}"

    def _ensure_sentence_identity(
        self,
        sentence: SentenceSegment,
        *,
        chunk_ref: Any,
    ) -> None:
        """确保句子具备稳定身份字段。"""
        normalized_chunk_ref = self._normalize_chunk_ref(chunk_ref)
        if not getattr(sentence, "chunk_uid", None):
            sentence.chunk_uid = str(normalized_chunk_ref)
        if not getattr(sentence, "sentence_uid", None):
            sentence.sentence_uid = self._build_sentence_uid(
                chunk_ref=normalized_chunk_ref,
                sentence=sentence,
            )
        if not getattr(sentence, "segment_id", None):
            sentence.segment_id = sentence.sentence_uid

    @staticmethod
    def _resolve_subtitle_item_source(source: Optional[str]) -> TextSource:
        source_text = str(source or "").strip().lower()
        if source_text in {"fast", TextSource.SENSEVOICE.value}:
            return TextSource.SENSEVOICE
        if source_text in {"manual", TextSource.MANUAL.value}:
            return TextSource.MANUAL
        if source_text in {
            "slow",
            "aligned",
            "render_core",
            TextSource.WHISPER_PATCH.value,
            TextSource.LLM_CORRECTION.value,
            TextSource.LLM_TRANSLATION.value,
        }:
            return TextSource.WHISPER_PATCH
        return TextSource.WHISPER_PATCH

    def _build_sentence_from_subtitle_item(self, item: SubtitleItem) -> SentenceSegment:
        trace = dict(item.trace or {})
        sentence = SentenceSegment(
            text=str(item.text),
            text_clean=str(item.text),
            start=float(item.start),
            end=float(item.end),
            source=self._resolve_subtitle_item_source(item.source),
            is_draft=str(item.status or "").lower() == "draft",
            is_finalized=str(item.status or "").lower() != "draft",
            speaker_id=item.speaker_id,
            turn_id=item.turn_id,
            sentence_uid=item.segment_id,
            segment_id=item.segment_id,
            chunk_uid=item.chunk_id,
        )
        sentence.split_reason = str(trace.get("split_reason", "") or "")
        sentence.split_risk = str(trace.get("split_risk", "") or "")
        sentence.window_id = str(trace.get("window_id", "") or "")
        sentence.pyannote_frame_time = trace.get("pyannote_frame_time")
        sentence.mapped_cut_time = trace.get("mapped_cut_time")
        sentence.mapping_quality = str(trace.get("mapping_quality", "") or "")
        sentence.mapping_reason = str(trace.get("mapping_reason", "") or "")
        return sentence

    def _build_subtitle_snapshot_payload(
        self,
        sentence: SentenceSegment,
        *,
        index: int,
    ) -> Dict[str, Any]:
        payload = self._build_sentence_payload(
            sentence,
            index=index,
            is_draft=getattr(sentence, "is_draft", False),
            is_finalized=getattr(sentence, "is_finalized", False),
        )
        segment_id = str(
            payload.get("segment_id")
            or payload.get("sentence_uid")
            or getattr(sentence, "segment_id", None)
            or getattr(sentence, "sentence_uid", None)
            or f"seg-{index}"
        )
        chunk_id = str(
            payload.get("chunk_id")
            or payload.get("chunk_uid")
            or getattr(sentence, "chunk_uid", None)
            or f"chunk:transcribe:{index}"
        )
        source_value = payload.get("source")
        if hasattr(source_value, "value"):
            source_value = source_value.value
        is_draft = bool(payload.get("is_draft", getattr(sentence, "is_draft", False)))
        is_finalized = bool(
            payload.get("is_finalized", getattr(sentence, "is_finalized", not is_draft))
        )
        status = "draft" if is_draft and not is_finalized else "final"
        trace: Dict[str, Any] = {}
        raw_trace = payload.get("trace")
        if isinstance(raw_trace, dict):
            trace.update(raw_trace)
        for key in (
            "split_reason",
            "split_risk",
            "window_id",
            "pyannote_frame_time",
            "mapped_cut_time",
            "mapping_quality",
            "mapping_reason",
        ):
            value = payload.get(key)
            if value is None:
                continue
            trace.setdefault(key, value)

        return {
            "segment_id": segment_id,
            "sentence_uid": segment_id,
            "chunk_id": chunk_id,
            "chunk_uid": chunk_id,
            "text": str(payload.get("text", "") or ""),
            "start": float(payload.get("start", 0.0) or 0.0),
            "end": float(payload.get("end", payload.get("start", 0.0)) or payload.get("start", 0.0)),
            "status": status,
            "source": str(source_value or "transcribe"),
            "speaker_id": payload.get("speaker_id"),
            "turn_id": payload.get("turn_id"),
            "trace": trace,
            "legacy_index": int(index),
            "source_type": str(source_value or "transcribe"),
            "is_modified": bool(payload.get("is_modified", getattr(sentence, "is_modified", False))),
            "original_text": payload.get("original_text"),
        }

    def _persist_runtime_subtitle_state(self, *, reason: str) -> None:
        """持久化运行态字幕快照到 runtime_state.db。"""
        if self._runtime_checkpoint_service is None:
            return
        try:
            payload = self.to_checkpoint_data()
            payload["updated_reason"] = reason
            payload["updated_at"] = payload.get("updated_at") or time.time()
            self._runtime_checkpoint_service.save_subtitle_runtime(payload)
        except Exception as exc:
            logger.warning(
                "运行态字幕快照写入失败: job_id=%s reason=%s error=%s",
                self.job_id,
                reason,
                exc,
            )

    @staticmethod
    def _remove_speaker_fields_for_draft(payload: Dict[str, Any]) -> Dict[str, Any]:
        """草稿事件强制移除 speaker 相关字段。"""
        for key in (
            "speaker_id",
            "turn_id",
            "speaker_label",
            "speaker_color_key",
            "binding_source",
        ):
            payload.pop(key, None)
        return payload

    @staticmethod
    def _is_hidden_unknown_sentence(sentence: SentenceSegment) -> bool:
        """统一 unknown 字幕可见性判断。"""
        return is_hidden_unknown_sentence(sentence)

    def _filter_unknown_sentences(
        self,
        sentences: List[SentenceSegment],
    ) -> List[SentenceSegment]:
        return [
            sentence for sentence in list(sentences or [])
            if not self._is_hidden_unknown_sentence(sentence)
        ]

    def _build_sentence_payload(
        self,
        sentence: SentenceSegment,
        *,
        index: Optional[int] = None,
        is_draft: Optional[bool] = None,
        is_finalized: Optional[bool] = None,
        sanitize_draft_speaker: bool = False,
    ) -> Dict[str, Any]:
        """统一序列化句子 payload，避免多处字段漂移。"""
        payload = sentence.to_dict() if hasattr(sentence, "to_dict") else {
            "text": sentence.text_clean or sentence.text,
            "start": sentence.start,
            "end": sentence.end,
            "confidence": sentence.confidence,
            "confidence_display_raw": getattr(sentence, "confidence_display_raw", None),
            "display_confidence": getattr(sentence, "display_confidence", None),
            "confidence_source": getattr(sentence, "confidence_source", None),
            "source": sentence.source.value if hasattr(sentence.source, "value") else str(sentence.source),
            "is_modified": getattr(sentence, "is_modified", False),
            "original_text": getattr(sentence, "original_text", None),
            "warning_type": getattr(getattr(sentence, "warning_type", None), "value", "none"),
            "words": [w.to_dict() if hasattr(w, "to_dict") else w for w in getattr(sentence, "words", [])],
            "speaker_id": getattr(sentence, "speaker_id", None),
            "turn_id": getattr(sentence, "turn_id", None),
            "speaker_label": getattr(sentence, "speaker_label", None),
            "speaker_color_key": getattr(sentence, "speaker_color_key", None),
            "binding_source": getattr(sentence, "binding_source", None),
            "sentence_uid": getattr(sentence, "sentence_uid", None),
            "segment_id": getattr(sentence, "segment_id", None),
            "chunk_uid": getattr(sentence, "chunk_uid", None),
        }
        if index is not None:
            payload["index"] = index
        if is_draft is not None:
            payload["is_draft"] = bool(is_draft)
        if is_finalized is not None:
            payload["is_finalized"] = bool(is_finalized)
        if sanitize_draft_speaker:
            self._remove_speaker_fields_for_draft(payload)
        return payload

    def _emit_subtitle_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """统一推送字幕事件（project 单频道）。"""
        push_subtitle_event(
            self.sse_manager,
            self.job_id,
            event_type,
            data,
            project_id=self.project_id,
        )

    def add_sentence(self, sentence: SentenceSegment) -> int:
        """
        添加新句子（SenseVoice 阶段）

        Args:
            sentence: 句子段落

        Returns:
            int: 句子索引
        """
        if self._is_hidden_unknown_sentence(sentence):
            logger.info("跳过 unknown 字幕写入: job_id=%s", self.job_id)
            return -1

        index = self.sentence_count
        self._ensure_sentence_identity(sentence, chunk_ref="chunk:legacy")
        self.sentences[index] = sentence
        self.sentence_count += 1

        # 推送 SSE 事件（使用清洗后的文本）
        # V3.1.2+dev.20260113.01: fallback 字典补充 display_confidence 和 confidence_source
        sentence_dict = sentence.to_dict() if hasattr(sentence, 'to_dict') else {
            "index": index,
            "text": sentence.text_clean or sentence.text,  # 优先使用清洗后的文本
            "start": sentence.start,
            "end": sentence.end,
            "confidence": sentence.confidence,
            "confidence_display_raw": getattr(sentence, 'confidence_display_raw', None),
            "display_confidence": getattr(sentence, 'display_confidence', None),  # V3.1.2: 映射后准确率
            "confidence_source": getattr(sentence, 'confidence_source', None),    # V3.1.2: 置信度来源
            "source": sentence.source.value if hasattr(sentence.source, 'value') else str(sentence.source),
            "words": [w.to_dict() if hasattr(w, 'to_dict') else w for w in getattr(sentence, 'words', [])]  # 确保包含 words
        }

        self._emit_subtitle_event(
            "sv_sentence",
            {
                "index": index,
                "sentence": sentence_dict,
                "source": "sensevoice",
            },
        )

        logger.debug(f"添加句子 {index}: {sentence.text[:30]}...")
        self._persist_runtime_subtitle_state(reason="add_sentence")
        return index

    def update_sentence(
        self,
        index: int,
        new_text: str,
        source: TextSource,
        confidence: float = None,
        perplexity: float = None,
        confidence_source: str = None  # V3.1.2: 新增，指定置信度来源
    ):
        """
        更新已有句子（Whisper 复核或 LLM 校对）

        V3.1.2+dev.20260111.01: 新增 confidence_source 参数，用于正确计算 display_confidence

        Args:
            index: 句子索引
            new_text: 新文本
            source: 文本来源
            confidence: 新置信度（可选）
            perplexity: LLM 困惑度（可选）
            confidence_source: 置信度来源（可选，如 "whisper"）
        """
        if index not in self.sentences:
            logger.warning(f"句子 {index} 不存在，无法更新")
            return

        sentence = self.sentences[index]

        # V3.2.0+dev.20260124.01: 保护用户编辑，防止 AI 覆盖
        if getattr(sentence, 'is_modified', False):
            logger.info(
                f"[V3.2.0+dev.20260124.01] 跳过更新用户已编辑的句子: "
                f"index={index}, source={source.value}"
            )
            return

        # 应用伪对齐
        from .pseudo_alignment import PseudoAlignment
        PseudoAlignment.apply_to_sentence(sentence, new_text, source)

        # V3.1.2: 更新置信度时，同时更新来源并重新计算 display_confidence
        if confidence is not None:
            # 确定置信度来源：如果没有明确指定，根据 source 推断
            if confidence_source is None:
                if source == TextSource.WHISPER_PATCH:
                    confidence_source = "whisper"
                else:
                    confidence_source = source.value

            sentence.update_confidence(confidence, confidence_source)
        else:
            # 无置信度（例如 Whisper 未返回可靠值）时，清空显示，避免沿用旧值
            sentence.confidence = None
            sentence.confidence_source = source.value
            sentence.confidence_display_raw = None
            sentence.display_confidence = None

        if perplexity is not None:
            sentence.perplexity = perplexity
            sentence.warning_type = sentence.compute_warning_type()

        if self._is_hidden_unknown_sentence(sentence):
            self.remove_sentence_by_index(index)
            logger.info("移除更新后变为 unknown 的字幕: job_id=%s index=%s", self.job_id, index)
            return

        # 推送 SSE 事件
        event_type = {
            TextSource.WHISPER_PATCH: "whisper_patch",
            TextSource.LLM_CORRECTION: "llm_proof",
            TextSource.LLM_TRANSLATION: "llm_trans",
        }.get(source, "batch_update")

        self._emit_subtitle_event(
            event_type,
            {
                "index": index,
                "sentence": sentence.to_dict(),
                "source": source.value,
                "is_update": True,
            },
        )

        logger.debug(f"更新句子 {index} ({source.value}): {new_text[:30]}...")
        self._persist_runtime_subtitle_state(reason="update_sentence")

    def set_translation(self, index: int, translation: str, confidence: float = None):
        """
        设置翻译结果

        Args:
            index: 句子索引
            translation: 翻译文本
            confidence: 翻译置信度
        """
        if index not in self.sentences:
            return

        sentence = self.sentences[index]
        sentence.translation = translation
        if confidence is not None:
            sentence.translation_confidence = confidence

        self._emit_subtitle_event(
            "llm_trans",
            {
                "index": index,
                "translation": translation,
                "confidence": confidence,
            },
        )

    def mark_for_deletion(self, index: int, reason: str = "garbage"):
        """
        标记句子为待删除（Whisper 仲裁后确认为垃圾）

        Args:
            index: 句子索引
            reason: 删除原因
        """
        if index not in self.sentences:
            return

        sentence = self.sentences[index]
        sentence.marked_for_deletion = True
        sentence.deletion_reason = reason

        # 推送 SSE 事件通知前端
        self._emit_subtitle_event(
            "sentence_deleted",
            {
                "index": index,
                "reason": reason,
            },
        )
        logger.info(f"标记删除句子 {index}: {reason}")

    def remove_marked_sentences(self) -> int:
        """
        物理删除被标记为垃圾的句子

        V3.8: 添加锁保护和删除开关，防止在流水线运行期间误删

        Returns:
            int: 删除的句子数量
        """
        # V3.8: 检查删除开关
        if not self._deletion_enabled:
            logger.warning("remove_marked_sentences: 删除功能未启用，跳过删除操作")
            return 0

        with self._lock:
            marked_indices = [
                idx for idx, s in self.sentences.items()
                if getattr(s, 'marked_for_deletion', False)
            ]

            for idx in marked_indices:
                del self.sentences[idx]

            if marked_indices:
                logger.info(f"已删除 {len(marked_indices)} 个垃圾句子: {marked_indices}")
                self._persist_runtime_subtitle_state(reason="remove_marked_sentences")

            return len(marked_indices)

    def enable_deletion(self):
        """V3.8: 启用删除功能（在任务完成后调用）"""
        self._deletion_enabled = True
        logger.info(f"StreamingSubtitleManager: 删除功能已启用 job_id={self.job_id}")

    def disable_deletion(self):
        """V3.8: 禁用删除功能（在任务开始时调用）"""
        self._deletion_enabled = False
        logger.info(f"StreamingSubtitleManager: 删除功能已禁用 job_id={self.job_id}")

    def get_all_sentences(self) -> List[SentenceSegment]:
        """获取所有句子（按时间排序，排除已标记删除的）"""
        sentences = [
            s for s in self.sentences.values()
            if not getattr(s, 'marked_for_deletion', False)
            and not self._is_hidden_unknown_sentence(s)
        ]
        sentences.sort(key=lambda s: s.start)
        # V3.8 调试日志：导出时记录句子数量
        logger.debug(
            f"get_all_sentences: job_id={self.job_id}, "
            f"total_in_dict={len(self.sentences)}, "
            f"after_filter={len(sentences)}, "
            f"chunk_count={len(self.chunk_sentences)}"
        )
        return sentences

    def get_context_window(self, index: int, window_size: int = 3) -> str:
        """
        获取上下文窗口（用于 LLM 校对和 Whisper 复核）

        Args:
            index: 当前句子索引
            window_size: 上下文窗口大小

        Returns:
            str: 上下文文本（清洗后的文本，避免传递 SenseVoice 原始 token）
        """
        context_indices = range(max(0, index - window_size), index)
        context_texts = [
            self.sentences[i].text_clean or self.sentences[i].text  # 优先使用清洗后的文本，避免传递下划线等原始 token
            for i in context_indices
            if i in self.sentences
        ]
        return " ".join(context_texts)

    def get_chunk_sentence_indices(self, chunk_ref: Any) -> List[int]:
        """获取指定 chunk 引用下的句子索引。"""
        resolved: List[int] = []
        seen_indices = set()
        for alias_key in self._iter_chunk_alias_keys(chunk_ref):
            indices = self.chunk_sentences.get(alias_key)
            if not isinstance(indices, list):
                continue
            for idx in indices:
                if idx in seen_indices:
                    continue
                seen_indices.add(idx)
                resolved.append(idx)
        return resolved

    # ========== Phase 4: 双流对齐专用方法 ==========

    def add_draft_sentences(
        self,
        chunk_ref: Any,
        sentences: List[SentenceSegment]
    ) -> List[int]:
        """
        添加草稿句子（快流推送）

        Phase 4 双流对齐专用方法。
        推送多个草稿句子，并记录 Chunk 级别的索引映射。

        V3.8: 深拷贝句子对象，避免共享引用导致的竞态条件

        Args:
            chunk_ref: Chunk 引用（支持 int chunk_index 或 string chunk_id）
            sentences: 句子列表

        Returns:
            List[int]: 句子索引列表
        """
        visible_sentences = self._filter_unknown_sentences(sentences)
        chunk_key = self._normalize_chunk_ref(chunk_ref)
        dropped_unknown_count = max(0, len(list(sentences or [])) - len(visible_sentences))
        if dropped_unknown_count > 0:
            logger.info(
                "草稿链路过滤 unknown 字幕: job_id=%s chunk=%s count=%s",
                self.job_id,
                chunk_key,
                dropped_unknown_count,
            )

        sentence_indices = []
        sentences_to_push = []  # V3.8: 收集待推送的句子数据

        with self._lock:
            old_indices = self.get_chunk_sentence_indices(chunk_key)
            protected_indices: List[int] = []
            for old_index in old_indices:
                old_sentence = self.sentences.get(old_index)
                if old_sentence is None:
                    continue
                if getattr(old_sentence, "is_modified", False):
                    protected_indices.append(old_index)
                    continue
                del self.sentences[old_index]

            for sentence in visible_sentences:
                # V3.8 修复：深拷贝句子对象，避免共享引用
                sentence_copy = copy.deepcopy(sentence)
                sentence_copy.is_draft = True
                sentence_copy.is_finalized = False

                index = self.sentence_count
                self._ensure_sentence_identity(sentence_copy, chunk_ref=chunk_key)
                self.sentences[index] = sentence_copy
                self.sentence_count += 1
                sentence_indices.append(index)

                # 草稿事件口径：仅文本/时间字段，不透传 speaker 标签
                sentence_dict = self._build_sentence_payload(
                    sentence_copy,
                    index=index,
                    is_draft=True,
                    is_finalized=False,
                    sanitize_draft_speaker=True,
                )
                sentences_to_push.append((index, sentence_dict))

            # 记录 Chunk 级别的索引映射
            self._remove_chunk_alias_mappings(chunk_key)
            self.chunk_sentences[chunk_key] = sorted(protected_indices + sentence_indices)

        # V3.8: 在锁外推送 SSE 事件，避免长时间持锁
        chunk_index_for_event = self._try_parse_chunk_index(chunk_key)
        for index, sentence_dict in sentences_to_push:
            event_payload = {
                "index": index,
                "chunk_index": chunk_index_for_event if chunk_index_for_event is not None else chunk_key,
                "chunk_uid": str(chunk_key),
                "sentence": sentence_dict,
            }
            self._emit_subtitle_event(
                "draft",
                event_payload,
            )

        # V3.8 调试日志：确认草稿已添加到管理器
        logger.debug(
            f"add_draft_sentences: Chunk {chunk_key} 添加 {len(visible_sentences)} 个草稿, "
            f"索引 {sentence_indices}, 当前总句子数={len(self.sentences)}"
        )

        self._persist_runtime_subtitle_state(reason="add_draft_sentences")
        return sentence_indices

    def replace_chunk(
        self,
        chunk_ref: Any,
        sentences: List[SentenceSegment]
    ) -> List[int]:
        """
        替换 Chunk 的所有句子（慢流推送）

        Phase 4 双流对齐专用方法。
        用定稿句子替换整个 Chunk 的草稿句子。

        V3.8: 添加锁保护，防止竞态条件

        流程：
        1. 删除旧的草稿句子
        2. 添加新的定稿句子
        3. 更新 Chunk 索引映射
        4. 推送 replace_chunk 事件

        Args:
            chunk_ref: Chunk 引用（支持 int chunk_index 或 string chunk_id）
            sentences: 定稿句子列表

        Returns:
            List[int]: 新的句子索引列表
        """
        visible_sentences = self._filter_unknown_sentences(sentences)
        chunk_key = self._normalize_chunk_ref(chunk_ref)
        dropped_unknown_count = max(0, len(list(sentences or [])) - len(visible_sentences))
        if dropped_unknown_count > 0:
            logger.info(
                "定稿替换过滤 unknown 字幕: job_id=%s chunk=%s count=%s",
                self.job_id,
                chunk_key,
                dropped_unknown_count,
            )

        # 防御性处理：空定稿也要完成 chunk 收口，避免前端草稿永久停留“生成中”。
        if not visible_sentences:
            with self._lock:
                old_indices = self.get_chunk_sentence_indices(chunk_key)
                protected_sentences = {}
                for old_index in old_indices:
                    if old_index not in self.sentences:
                        continue
                    old_sentence = self.sentences[old_index]
                    if getattr(old_sentence, 'is_modified', False):
                        protected_sentences[old_index] = old_sentence
                        logger.info(
                            f"[V3.2.0+dev.20260216.11] 空定稿保留用户编辑: "
                            f"chunk={chunk_key}, index={old_index}"
                        )
                    else:
                        del self.sentences[old_index]

                new_indices = sorted(protected_sentences.keys())
                self._remove_chunk_alias_mappings(chunk_key)
                self.chunk_sentences[chunk_key] = new_indices

            chunk_index_for_event = self._try_parse_chunk_index(chunk_key)
            self._emit_subtitle_event(
                "replace_chunk",
                {
                    "chunk_index": chunk_index_for_event if chunk_index_for_event is not None else chunk_key,
                    "chunk_uid": str(chunk_key),
                    "old_indices": old_indices,
                    "new_indices": new_indices,
                    "sentences": [],
                },
            )
            logger.warning(
                f"replace_chunk: Chunk {chunk_key} 的定稿句子为空，"
                f"已清理 {max(len(old_indices) - len(new_indices), 0)} 个草稿并推送空替换事件，"
                f"保留用户编辑 {len(new_indices)} 条"
            )
            self._persist_runtime_subtitle_state(reason="replace_chunk_empty")
            return new_indices

        # V3.8: 使用锁保护整个替换过程
        with self._lock:
            # 删除旧的草稿句子（V3.2.0+dev.20260124.01: 保护用户编辑）
            old_indices = self.get_chunk_sentence_indices(chunk_key)
            protected_sentences = {}  # 保存被保护的用户编辑句子
            new_sentence_pairs: list[tuple[int, SentenceSegment]] = []

            for old_index in old_indices:
                if old_index in self.sentences:
                    old_sentence = self.sentences[old_index]
                    # 如果用户已编辑，保护该句子，不删除
                    if getattr(old_sentence, 'is_modified', False):
                        protected_sentences[old_index] = old_sentence
                        logger.info(
                            f"[V3.2.0+dev.20260124.01] 保护用户编辑: "
                            f"chunk={chunk_key}, index={old_index}"
                        )
                    else:
                        del self.sentences[old_index]

            # 添加新的定稿句子
            new_indices = []
            for sentence in visible_sentences:
                sentence_copy = copy.deepcopy(sentence)
                sentence_copy.is_draft = False
                sentence_copy.is_finalized = True
                index = self.sentence_count
                self._ensure_sentence_identity(sentence_copy, chunk_ref=chunk_key)
                self.sentences[index] = sentence_copy
                self.sentence_count += 1
                new_indices.append(index)
                new_sentence_pairs.append((index, sentence_copy))

            # 合并保护的句子（保持原索引）
            for protected_index, protected_sentence in protected_sentences.items():
                self.sentences[protected_index] = protected_sentence
                new_indices.append(protected_index)

            # 更新 Chunk 索引映射
            self._remove_chunk_alias_mappings(chunk_key)
            self.chunk_sentences[chunk_key] = sorted(new_indices)

        # 推送 SSE 事件（批量替换）- 在锁外推送，避免死锁
        sentences_data: List[Dict[str, Any]] = []
        for sentence_index, sentence in new_sentence_pairs:
            sentences_data.append(
                self._build_sentence_payload(
                    sentence,
                    index=sentence_index,
                    is_draft=False,
                    is_finalized=True,
                    sanitize_draft_speaker=False,
                )
            )

        chunk_index_for_event = self._try_parse_chunk_index(chunk_key)
        self._emit_subtitle_event(
            "replace_chunk",
            {
                "chunk_index": chunk_index_for_event if chunk_index_for_event is not None else chunk_key,
                "chunk_uid": str(chunk_key),
                "old_indices": old_indices,
                "new_indices": new_indices,
                "sentences": sentences_data,
            },
        )

        # V3.8 调试日志：确认替换成功
        logger.debug(
            f"replace_chunk: Chunk {chunk_key} 替换完成 - "
            f"删除 {len(old_indices)} 个草稿 {old_indices}, "
            f"添加 {len(visible_sentences)} 个定稿 {new_indices}, "
            f"当前总句子数={len(self.sentences)}"
        )

        self._persist_runtime_subtitle_state(reason="replace_chunk")
        return new_indices

    def replace_chunk_batch(self, subtitle_batch: SubtitleBatch) -> List[int]:
        """按统一字幕 DTO 替换 Chunk。"""
        subtitle_items = list(subtitle_batch.items or ())
        sentences = [
            self._build_sentence_from_subtitle_item(item)
            for item in subtitle_items
        ]
        return self.replace_chunk(subtitle_batch.chunk_id, sentences)

    def add_finalized_sentences(
        self,
        chunk_ref: Any,
        sentences: List[SentenceSegment]
    ) -> List[int]:
        """
        添加定稿句子（极速模式专用）

        V3.5 新增: 极速模式下 FastWorker 直接输出定稿，不经过 SlowWorker。
        与 add_draft_sentences 不同，这里直接推送定稿事件。

        V3.8: 添加锁保护和深拷贝，防止竞态条件

        Args:
            chunk_ref: Chunk 引用（支持 int chunk_index 或 string chunk_id）
            sentences: 定稿句子列表

        Returns:
            List[int]: 句子索引列表
        """
        visible_sentences = self._filter_unknown_sentences(sentences)
        chunk_key = self._normalize_chunk_ref(chunk_ref)
        dropped_unknown_count = max(0, len(list(sentences or [])) - len(visible_sentences))
        if dropped_unknown_count > 0:
            logger.info(
                "极速定稿过滤 unknown 字幕: job_id=%s chunk=%s count=%s",
                self.job_id,
                chunk_key,
                dropped_unknown_count,
            )

        sentence_indices = []
        sentences_to_push = []  # V3.8: 收集待推送的句子数据

        with self._lock:
            old_indices = self.get_chunk_sentence_indices(chunk_key)
            for old_index in old_indices:
                existing_sentence = self.sentences.get(old_index)
                if existing_sentence is None:
                    continue
                if getattr(existing_sentence, "is_modified", False):
                    continue
                del self.sentences[old_index]

            for sentence in visible_sentences:
                # V3.8 修复：深拷贝句子对象，避免共享引用
                sentence_copy = copy.deepcopy(sentence)

                # 确保句子标记为定稿
                sentence_copy.is_draft = False
                sentence_copy.is_finalized = True

                index = self.sentence_count
                self._ensure_sentence_identity(sentence_copy, chunk_ref=chunk_key)
                self.sentences[index] = sentence_copy
                self.sentence_count += 1
                sentence_indices.append(index)

                # 定稿事件允许透传 speaker 字段
                sentence_dict = self._build_sentence_payload(
                    sentence_copy,
                    index=index,
                    is_draft=False,
                    is_finalized=True,
                    sanitize_draft_speaker=False,
                )
                sentences_to_push.append((index, sentence_dict))

            # 记录 Chunk 级别的索引映射
            self._remove_chunk_alias_mappings(chunk_key)
            self.chunk_sentences[chunk_key] = sentence_indices

        # V3.8: 在锁外推送 SSE 事件，避免死锁
        chunk_index_for_event = self._try_parse_chunk_index(chunk_key)
        for index, sentence_dict in sentences_to_push:
            self._emit_subtitle_event(
                "finalized",
                {
                    "index": index,
                    "chunk_index": chunk_index_for_event if chunk_index_for_event is not None else chunk_key,
                    "chunk_uid": str(chunk_key),
                    "sentence": sentence_dict,
                    "mode": "sensevoice_only",
                },
            )

        logger.debug(
            f"添加定稿句子 [极速模式]: Chunk {chunk_key}, "
            f"{len(visible_sentences)} 个句子, 索引 {sentence_indices}"
        )

        self._persist_runtime_subtitle_state(reason="add_finalized_sentences")
        return sentence_indices

    # ========== V3.1.0: 字幕持久化方法 ==========

    def to_checkpoint_data(self) -> dict:
        """
        V3.1.0: 导出字幕快照用于 Checkpoint 保存

        返回完整的字幕状态，包括：
        - sentences_snapshot: 所有句子的序列化数据
        - sentence_count: 全局句子计数器
        - chunk_sentences_map: Chunk 到句子索引的映射

        Returns:
            dict: 可直接保存到 Checkpoint 的字幕数据
        """
        sentences_snapshot = []
        subtitle_items_snapshot = []
        visible_indices = set()
        for idx, sentence in self.sentences.items():
            if self._is_hidden_unknown_sentence(sentence):
                continue
            # Checkpoint 持久化保留全量字段（包括 speaker），供恢复与最终导出使用
            sentence_dict = self._build_sentence_payload(sentence)
            # 添加索引信息
            sentence_dict["_index"] = idx
            sentence_dict["_is_draft"] = getattr(sentence, 'is_draft', False)
            sentence_dict["_is_finalized"] = getattr(sentence, 'is_finalized', False)
            sentences_snapshot.append(sentence_dict)
            subtitle_items_snapshot.append(
                self._build_subtitle_snapshot_payload(sentence, index=idx)
            )
            visible_indices.add(idx)

        chunk_sentences_map: Dict[str, List[int]] = {}
        for chunk_ref, indices in self.chunk_sentences.items():
            chunk_key = str(chunk_ref)
            chunk_sentences_map[chunk_key] = [
                idx for idx in indices if idx in visible_indices
            ]

        return {
            "subtitle_items_snapshot": subtitle_items_snapshot,
            "sentences_snapshot": sentences_snapshot,
            "sentence_count": self.sentence_count,
            "chunk_sentences_map": chunk_sentences_map,
            "version": "3.2.4+dev.20260228.01",
            "updated_at": time.time(),
        }

    def restore_from_checkpoint(self, checkpoint_data: dict) -> bool:
        """
        V3.1.0: 从 Checkpoint 恢复字幕状态

        V3.1.2+dev.20260111.01: 恢复时计算 display_confidence（兼容旧数据）

        恢复所有已保存的句子，并恢复索引计数器状态。
        恢复后，新添加的句子会从正确的索引继续编号，不会与已有句子冲突。

        Args:
            checkpoint_data: Checkpoint 中的字幕数据，包含：
                - sentences_snapshot: 句子快照列表
                - sentence_count: 句子计数器
                - chunk_sentences_map: Chunk 映射

        Returns:
            bool: 恢复是否成功
        """
        from app.models.sensevoice_models import SentenceSegment, TextSource, WarningType, WordTimestamp

        try:
            sentences_snapshot = checkpoint_data.get("sentences_snapshot", [])
            sentence_count = checkpoint_data.get("sentence_count", 0)
            chunk_sentences_map = checkpoint_data.get("chunk_sentences_map", {})

            if not sentences_snapshot:
                logger.info(f"[V3.1.0] 无字幕快照需要恢复: job_id={self.job_id}")
                return True

            # 恢复句子
            restored_count = 0
            for sentence_dict in sentences_snapshot:
                idx = sentence_dict.get("_index")
                if idx is None:
                    continue

                # V3.1.2: 读取置信度相关字段（兼容旧数据）
                raw_confidence = sentence_dict.get("confidence", 1.0)
                display_confidence = sentence_dict.get("display_confidence")  # 可能为 None（旧数据）
                display_raw = sentence_dict.get("confidence_display_raw")
                confidence_source = sentence_dict.get("confidence_source")    # 可能为 None（旧数据）

                # 从字典重建 SentenceSegment
                sentence = SentenceSegment(
                    text=sentence_dict.get("original_text") or sentence_dict.get("text", ""),
                    text_clean=sentence_dict.get("text", ""),
                    start=sentence_dict.get("start", 0.0),
                    end=sentence_dict.get("end", 0.0),
                    confidence=raw_confidence,
                    confidence_display_raw=display_raw,
                    display_confidence=display_confidence,  # V3.1.2: 新增
                    confidence_source=confidence_source,    # V3.1.2: 新增
                )

                # 恢复来源
                source_str = sentence_dict.get("source", "sensevoice")
                try:
                    sentence.source = TextSource(source_str)
                except ValueError:
                    sentence.source = TextSource.SENSEVOICE

                # V3.1.2: 如果旧数据没有 display_confidence，立即计算
                # __post_init__ 已经会自动计算，但这里确保 confidence_source 正确
                if sentence.confidence_source is None:
                    sentence.confidence_source = source_str
                if sentence.display_confidence is None:
                    sentence._compute_display_confidence()

                # 恢复警告类型
                warning_str = sentence_dict.get("warning_type", "none")
                try:
                    sentence.warning_type = WarningType(warning_str)
                except ValueError:
                    sentence.warning_type = WarningType.NONE

                # 恢复其他字段
                sentence.is_modified = sentence_dict.get("is_modified", False)
                sentence.original_text = sentence_dict.get("original_text")
                sentence.whisper_alternative = sentence_dict.get("whisper_alternative")
                sentence.perplexity = sentence_dict.get("perplexity")
                sentence.translation = sentence_dict.get("translation")
                sentence.translation_confidence = sentence_dict.get("translation_confidence")
                restored_is_draft = bool(sentence_dict.get("_is_draft", False))
                restored_is_finalized = sentence_dict.get("_is_finalized")
                if restored_is_finalized is None:
                    restored_is_finalized = not restored_is_draft
                sentence.is_draft = restored_is_draft
                sentence.is_finalized = bool(restored_is_finalized)
                sentence.sentence_uid = sentence_dict.get("sentence_uid") or sentence_dict.get("segment_id")
                sentence.segment_id = sentence_dict.get("segment_id") or sentence.sentence_uid
                sentence.chunk_uid = sentence_dict.get("chunk_uid")
                sentence.speaker_id = sentence_dict.get("speaker_id")
                sentence.turn_id = sentence_dict.get("turn_id")
                sentence.speaker_label = sentence_dict.get("speaker_label")
                sentence.speaker_color_key = sentence_dict.get("speaker_color_key")
                sentence.binding_source = sentence_dict.get("binding_source")

                # 恢复字级时间戳
                words_data = sentence_dict.get("words", [])
                sentence.words = []
                for word_dict in words_data:
                    # V3.1.2: confidence 可能为 None（Whisper 复核后的伪对齐）
                    word_confidence = word_dict.get("confidence")  # 不设置默认值，保持 None
                    word = WordTimestamp(
                        word=word_dict.get("word", ""),
                        start=word_dict.get("start", 0.0),
                        end=word_dict.get("end", 0.0),
                        confidence=word_confidence,
                        confidence_raw=word_dict.get("confidence_raw"),
                        confidence_display_raw=word_dict.get("confidence_display_raw"),
                        token_type=word_dict.get("token_type"),
                        is_pseudo=word_dict.get("is_pseudo", False)
                    )
                    sentence.words.append(word)

                if self._is_hidden_unknown_sentence(sentence):
                    continue
                self._ensure_sentence_identity(
                    sentence,
                    chunk_ref=sentence.chunk_uid or "chunk:restored",
                )
                self.sentences[idx] = sentence
                restored_count += 1

            # 恢复计数器（关键：确保新句子索引不会冲突）
            self.sentence_count = max(sentence_count, restored_count)

            # 恢复 Chunk 映射
            # JSON 反序列化后键是 str，需要转换为 int
            if chunk_sentences_map:
                normalized_map: Dict[Any, List[int]] = {}
                for raw_key, raw_indices in chunk_sentences_map.items():
                    parsed_key = self._try_parse_chunk_index(raw_key)
                    normalized_key: Any = parsed_key if parsed_key is not None else str(raw_key)
                    if not isinstance(raw_indices, list):
                        continue
                    normalized_map[normalized_key] = [
                        int(item) for item in raw_indices
                        if isinstance(item, int) or str(item).lstrip("-").isdigit()
                    ]
                self.chunk_sentences = normalized_map
                self.chunk_sentences = {
                    chunk_index: [
                        idx for idx in indices
                        if idx in self.sentences and not self._is_hidden_unknown_sentence(self.sentences[idx])
                    ]
                    for chunk_index, indices in self.chunk_sentences.items()
                }

            logger.info(
                f"[V3.1.0] 字幕恢复成功: job_id={self.job_id}, "
                f"恢复了 {restored_count} 个句子, "
                f"sentence_count={self.sentence_count}, "
                f"chunk_count={len(self.chunk_sentences)}"
            )
            self._persist_runtime_subtitle_state(reason="restore_from_checkpoint")
            return True

        except Exception as e:
            logger.error(f"[V3.1.0] 字幕恢复失败: job_id={self.job_id}, error={e}", exc_info=True)
            return False

    def apply_user_edits(self, edits: Dict[int, Dict[str, Any]]) -> int:
        """
        V3.2.0+dev.20260124.02: 将用户编辑叠加到内存字幕（防止恢复后被覆盖）。

        Args:
            edits: 用户编辑映射（sentence_index -> edit data）

        Returns:
            int: 成功应用的编辑数量
        """
        if not edits:
            return 0

        updated_count = 0
        for index, edit in edits.items():
            sentence = self.sentences.get(index)
            if not sentence:
                continue

            if edit.get("text") is not None:
                if not sentence.is_modified and not sentence.original_text:
                    sentence.original_text = sentence.text
                sentence.text = edit["text"]
                sentence.text_clean = edit["text"]
                sentence.update_confidence(None, "manual")

            if edit.get("start") is not None:
                sentence.start = edit["start"]
            if edit.get("end") is not None:
                sentence.end = edit["end"]

            sentence.is_modified = True
            updated_count += 1

        if updated_count:
            logger.info(
                "[V3.2.0+dev.20260124.02] 已应用用户编辑到字幕管理器: "
                "job_id=%s, count=%s",
                self.job_id,
                updated_count
            )
            self._persist_runtime_subtitle_state(reason="apply_user_edits")
        return updated_count

    def add_manual_sentence(self, index: int, text: str, start: float, end: float) -> None:
        """
        V3.2.0+dev.20260124.02: 添加用户手动字幕（不影响 sentence_count）。
        """
        with self._lock:
            sentence = SentenceSegment(
                text=text,
                text_clean=text,
                start=start,
                end=end,
                confidence=None
            )
            sentence.source = TextSource.MANUAL
            sentence.is_modified = True
            self._ensure_sentence_identity(sentence, chunk_ref="chunk:manual")
            self.sentences[index] = sentence
        self._persist_runtime_subtitle_state(reason="add_manual_sentence")

    def remove_sentence_by_index(self, index: int) -> bool:
        """
        V3.2.0+dev.20260124.02: 删除指定索引的句子，并维护 chunk 映射。
        """
        removed = False
        with self._lock:
            if index in self.sentences:
                del self.sentences[index]
                removed = True

            # 从 chunk 映射中移除
            for chunk_index, indices in list(self.chunk_sentences.items()):
                if index in indices:
                    new_indices = [item for item in indices if item != index]
                    if new_indices:
                        self.chunk_sentences[chunk_index] = new_indices
                    else:
                        del self.chunk_sentences[chunk_index]

        if removed:
            self._persist_runtime_subtitle_state(reason="remove_sentence_by_index")
        return removed

    def apply_user_deletions(self, deleted_indices: List[int]) -> int:
        """
        V3.2.0+dev.20260124.02: 应用用户删除列表，防止恢复时回补。
        """
        if not deleted_indices:
            return 0
        removed_count = 0
        for index in deleted_indices:
            if self.remove_sentence_by_index(index):
                removed_count += 1
        if removed_count:
            logger.info(
                "[V3.2.0+dev.20260124.02] 已应用用户删除: job_id=%s, count=%s",
                self.job_id,
                removed_count
            )
        return removed_count

    def apply_manual_entries(self, edits: Dict[int, Dict[str, Any]]) -> int:
        """
        V3.2.0+dev.20260124.02: 将手动新增字幕叠加到内存字幕。
        """
        if not edits:
            return 0
        added_count = 0
        for index, entry in edits.items():
            if index >= 0:
                continue
            if index in self.sentences:
                continue
            text = entry.get("text")
            if text is None:
                continue
            self.add_manual_sentence(
                index=index,
                text=text,
                start=entry.get("start", 0),
                end=entry.get("end", 0)
            )
            added_count += 1
        if added_count:
            logger.info(
                "[V3.2.0+dev.20260124.02] 已追加手动字幕: job_id=%s, count=%s",
                self.job_id,
                added_count
            )
            self._persist_runtime_subtitle_state(reason="apply_manual_entries")
        return added_count

    def push_restored_subtitles_to_frontend(self):
        """
        V3.1.0: 恢复后推送所有字幕到前端

        在恢复字幕后调用此方法，将已恢复的字幕通过 SSE 推送到前端，
        确保前端状态与后端同步。
        """
        # 按 Chunk 分组推送
        for chunk_index, sentence_indices in self.chunk_sentences.items():
            sentences_data = []
            for idx in sentence_indices:
                if idx in self.sentences:
                    sentence = self.sentences[idx]
                    if self._is_hidden_unknown_sentence(sentence):
                        continue
                    is_draft = bool(getattr(sentence, "is_draft", False))
                    sentence_dict = self._build_sentence_payload(
                        sentence,
                        index=idx,
                        is_draft=is_draft,
                        is_finalized=bool(getattr(sentence, "is_finalized", True)),
                        sanitize_draft_speaker=is_draft,
                    )
                    sentences_data.append(sentence_dict)

            if sentences_data:
                # 推送恢复事件（使用新的事件类型，避免与实时推送混淆）
                self._emit_subtitle_event(
                    "restored",
                    {
                        "chunk_index": chunk_index,
                        "sentences": sentences_data,
                        "is_restore": True,
                    },
                )

        # 推送手动新增字幕（不在 chunk 映射内）
        manual_sentences = []
        for idx, sentence in self.sentences.items():
            if idx >= 0:
                continue
            if self._is_hidden_unknown_sentence(sentence):
                continue
            is_draft = bool(getattr(sentence, "is_draft", False))
            sentence_dict = self._build_sentence_payload(
                sentence,
                index=idx,
                is_draft=is_draft,
                is_finalized=bool(getattr(sentence, "is_finalized", True)),
                sanitize_draft_speaker=is_draft,
            )
            manual_sentences.append(sentence_dict)

        if manual_sentences:
            self._emit_subtitle_event(
                "restored",
                {
                    "chunk_index": "manual",
                    "sentences": manual_sentences,
                    "is_restore": True,
                },
            )

        logger.info(
            f"[V3.1.0] 已推送恢复的字幕到前端: job_id={self.job_id}, "
            f"chunks={len(self.chunk_sentences)}, "
            f"total_sentences={len(self.sentences)}"
        )


# ========== 单例工厂 ==========

_subtitle_managers: Dict[str, StreamingSubtitleManager] = {}


def get_streaming_subtitle_manager(
    job_id: str,
    project_id: Optional[str] = None,
) -> StreamingSubtitleManager:
    """获取或创建流式字幕管理器"""
    global _subtitle_managers
    if job_id not in _subtitle_managers:
        _subtitle_managers[job_id] = StreamingSubtitleManager(
            job_id=job_id,
            project_id=project_id,
        )
    elif project_id:
        _subtitle_managers[job_id].project_id = project_id
    return _subtitle_managers[job_id]


def get_streaming_subtitle_manager_if_exists(job_id: str) -> Optional[StreamingSubtitleManager]:
    """仅在存在时返回字幕管理器，避免完成态无意义实例化。"""
    return _subtitle_managers.get(job_id)


def remove_streaming_subtitle_manager(job_id: str):
    """移除流式字幕管理器"""
    global _subtitle_managers
    if job_id in _subtitle_managers:
        del _subtitle_managers[job_id]
