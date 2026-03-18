"""
同音检索路由（纯 Project 语义）。

设计模式：
- Facade（门面模式）：统一封装同音检索/替换 API，屏蔽底层索引与存储细节。
- Adapter（适配器模式）：将 Project 字幕真源适配为同音服务需要的 sentence 列表。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.homophone.db import GlobalTermRule
from app.services.homophone.project_sync import (
    detect_language_from_segments,
    sync_project_index_delta,
)
from app.services.homophone.runtime import get_homophone_service
from app.services.homophone.service import SentenceRecord
from app.services.project_id_resolver import get_project_id_resolver
from app.services.sse_service import get_sse_manager, push_subtitle_event
from app.services.streaming_subtitle import get_streaming_subtitle_manager_if_exists
from app.services.subtitle_doc_service import get_subtitle_doc_service
from app.services.subtitle_edit_store import (
    apply_deletions_to_segments,
    apply_edits_to_segments,
    apply_edits_to_sentences_snapshot,
    build_manual_segments,
    load_deleted_indices,
    load_edits,
    save_edit,
)
from app.services.subtitle_visibility import (
    filter_hidden_unknown_sentences,
    is_hidden_unknown_sentence,
)

logger = logging.getLogger(__name__)


class HomophoneFindRequest(BaseModel):
    """同音检索请求模型。"""

    mode: Literal["homophone_strict", "homophone_fuzzy"] = Field(
        default="homophone_strict",
        description="同音检索模式",
    )
    query_text: str = Field(..., min_length=1, description="待检索文本或读音输入")
    language: Literal["zh", "ja", "en"] = Field(default="zh", description="语言")
    is_ignore_punctuation: bool = Field(default=False, description="是否忽略标点")
    limit: int = Field(default=500, ge=1, le=5000, description="最大返回数量")


class BatchReplaceRequest(BaseModel):
    """批量替换请求模型。"""

    mode: Literal["literal", "regex", "homophone_strict", "homophone_fuzzy"] = Field(
        default="literal",
        description="替换模式",
    )
    query_text: str = Field(..., min_length=1, description="查找文本")
    replace_text: str = Field(default="", description="替换文本")
    language: Literal["zh", "ja", "en"] = Field(default="zh", description="语言")
    is_ignore_punctuation: bool = Field(default=False, description="是否忽略标点")
    selected_sentence_indices: List[int] = Field(
        default_factory=list,
        description="用户勾选的句子索引列表",
    )


class GlobalTermItem(BaseModel):
    """全局术语配置项。"""

    language: Literal["auto", "zh", "ja", "en"] = Field(default="auto")
    source_text: str = Field(..., min_length=1)
    target_text: str = Field(default="")
    match_mode: Literal["exact", "regex", "homophone_strict", "homophone_fuzzy"] = Field(default="exact")
    priority: int = Field(default=100, ge=0, le=10000)
    is_enabled: bool = Field(default=True)
    note: str = Field(default="")


class GlobalTermSyncRequest(BaseModel):
    """全量同步全局术语请求。"""

    items: List[GlobalTermItem] = Field(default_factory=list)


def create_homophone_router() -> APIRouter:
    """创建同音检索路由（纯 project_id 语义）。"""
    router = APIRouter(prefix="/api", tags=["homophone"])
    sse_manager = get_sse_manager()

    def _resolve_project_identity_or_404(project_id: str):
        normalized_project_id = str(project_id or "").strip()
        if not normalized_project_id:
            raise HTTPException(status_code=404, detail="项目未找到")
        try:
            return get_project_id_resolver().resolve_or_fail(normalized_project_id)
        except Exception as exc:
            raise HTTPException(status_code=404, detail="项目未找到") from exc

    def _collect_segments_from_runtime_snapshot(project_dir: Path) -> List[Dict[str, Any]]:
        """
        从 runtime 快照读取字幕（转录项目优先路径）。

        说明：
        - 该路径与转录链路输出保持一致，可覆盖实时写入 checkpoint 的场景；
        - 纯导入项目通常没有 runtime 快照，调用方会回退到 subtitle_doc。
        """
        checkpoint_path = project_dir / "checkpoint.json"
        snapshot_path = project_dir / "transcription_text.json"

        data: Optional[Dict[str, Any]] = None
        transcription_data: Optional[Dict[str, Any]] = None

        try:
            if checkpoint_path.exists():
                with open(checkpoint_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    transcription_data = data.get("transcription", {})
            elif snapshot_path.exists():
                with open(snapshot_path, "r", encoding="utf-8") as file:
                    transcription_data = json.load(file)
                    data = {"transcription": transcription_data}
            else:
                return []
        except Exception as exc:
            logger.warning("读取同音 runtime 快照失败: project_dir=%s error=%s", project_dir, exc)
            return []

        transcription = transcription_data or {}
        sentences_snapshot = transcription.get("sentences_snapshot", [])
        edits = load_edits(project_dir)
        deleted_indices = load_deleted_indices(project_dir)

        all_segments: List[Dict[str, Any]] = []
        if sentences_snapshot:
            if edits:
                apply_edits_to_sentences_snapshot(sentences_snapshot, edits)
            for sentence in sentences_snapshot:
                if is_hidden_unknown_sentence(sentence):
                    continue
                all_segments.append(
                    {
                        "id": int(sentence.get("_index", sentence.get("index", 0))),
                        "start": float(sentence.get("start", 0.0)),
                        "end": float(sentence.get("end", 0.0)),
                        "text": str(sentence.get("text", "")),
                        "is_modified": bool(sentence.get("is_modified", False)),
                    }
                )
        else:
            unaligned_results = (data or {}).get("unaligned_results", [])
            for result in unaligned_results:
                all_segments.extend(result.get("segments", []))
            all_segments.sort(key=lambda item: item.get("start", 0))
            for index, segment in enumerate(all_segments):
                segment["id"] = int(segment.get("id", index))
            if edits:
                apply_edits_to_segments(all_segments, edits)

        if deleted_indices:
            all_segments = apply_deletions_to_segments(all_segments, deleted_indices)

        manual_segments = build_manual_segments(edits, deleted_indices)
        if manual_segments:
            all_segments.extend(manual_segments)
        all_segments = filter_hidden_unknown_sentences(all_segments)

        all_segments.sort(
            key=lambda item: (
                float(item.get("start", 0.0)),
                float(item.get("end", 0.0)),
                int(item.get("id", 0)),
            )
        )
        return all_segments

    def _collect_segments_from_subtitle_doc(project_dir: Path) -> List[Dict[str, Any]]:
        """从 subtitle_doc 读取字幕（导入项目主路径）。"""
        subtitle_doc_service = get_subtitle_doc_service()
        segments = subtitle_doc_service.load_segments(project_dir)
        normalized: List[Dict[str, Any]] = []
        for item in segments:
            legacy_index = item.get("legacy_index")
            if legacy_index is None:
                continue
            normalized.append(
                {
                    "id": int(legacy_index),
                    "start": float(item.get("start", 0.0)),
                    "end": float(item.get("end", 0.0)),
                    "text": str(item.get("text", "")),
                    "is_modified": bool(item.get("is_modified", False)),
                }
            )
        normalized.sort(
            key=lambda segment: (
                float(segment.get("start", 0.0)),
                float(segment.get("end", 0.0)),
                int(segment.get("id", 0)),
            )
        )
        return normalized

    def _collect_segments_for_project(project_dir: Path) -> List[Dict[str, Any]]:
        """
        统一读取同音检索源字幕。

        优先级：
        1. runtime 快照（转录任务）；
        2. subtitle_doc（导入/仅编辑项目）。
        """
        runtime_segments = _collect_segments_from_runtime_snapshot(project_dir)
        if runtime_segments:
            return runtime_segments
        return _collect_segments_from_subtitle_doc(project_dir)

    def _rebuild_homophone_index(
        *,
        project_id: str,
        requested_language: str,
        project_dir: Path,
        revision: int,
    ) -> Optional[Any]:
        """基于当前字幕真源重建同音索引。"""
        homophone_service = get_homophone_service()
        segments = _collect_segments_for_project(project_dir)
        if not segments:
            return None

        detected_language = detect_language_from_segments(segments)
        language = requested_language if requested_language in {"zh", "ja", "en"} else detected_language
        if language not in {"zh", "ja", "en"}:
            language = detected_language

        records = [
            SentenceRecord(
                index=int(item.get("id", 0)),
                text=str(item.get("text", "")),
            )
            for item in segments
        ]
        homophone_service.index_chunk(
            project_id=project_id,
            revision=max(1, int(revision)),
            chunk_index=0,
            language=language,
            sentences=records,
        )
        return homophone_service.get_index_status(project_id)

    def _ensure_homophone_index_ready(
        *,
        project_id: str,
        requested_language: str,
        project_dir: Path,
    ) -> Optional[Any]:
        """确保同音索引可查询。"""
        homophone_service = get_homophone_service()
        state = homophone_service.get_index_status(project_id)
        if state is not None and state.status == "ready":
            return state

        target_revision = state.revision if state is not None else 1
        return _rebuild_homophone_index(
            project_id=project_id,
            requested_language=requested_language,
            project_dir=project_dir,
            revision=target_revision,
        )

    @router.post("/projects/{project_id}/homophone/find")
    async def project_homophone_find(project_id: str, payload: HomophoneFindRequest):
        """Project 语义同音检索接口。"""
        identity = _resolve_project_identity_or_404(project_id)
        normalized_project_id = identity.project_id

        homophone_service = get_homophone_service()
        state = _ensure_homophone_index_ready(
            project_id=normalized_project_id,
            requested_language=payload.language,
            project_dir=identity.project_dir,
        )
        if state is None:
            return {
                "success": True,
                "data": {
                    "project_id": normalized_project_id,
                    "matches": [],
                    "index_status": "missing",
                    "message": "索引尚未构建完成",
                },
            }

        request_language = payload.language if payload.language in {"zh", "ja", "en"} else "zh"
        indexed_language = request_language

        matches = homophone_service.search_homophone(
            project_id=normalized_project_id,
            revision=state.revision,
            language=request_language,
            query_text=payload.query_text,
            mode=payload.mode,
            is_ignore_punctuation=payload.is_ignore_punctuation,
            limit=payload.limit,
        )

        if not matches:
            for fallback_language in ("zh", "ja", "en"):
                if fallback_language == request_language:
                    continue
                fallback_matches = homophone_service.search_homophone(
                    project_id=normalized_project_id,
                    revision=state.revision,
                    language=fallback_language,
                    query_text=payload.query_text,
                    mode=payload.mode,
                    is_ignore_punctuation=payload.is_ignore_punctuation,
                    limit=payload.limit,
                )
                if fallback_matches:
                    matches = fallback_matches
                    indexed_language = fallback_language
                    break

        if not matches:
            segments = _collect_segments_for_project(identity.project_dir)
            if segments:
                detected_language = detect_language_from_segments(segments)
                if detected_language in {"zh", "ja", "en"} and detected_language != indexed_language:
                    refreshed_state = _rebuild_homophone_index(
                        project_id=normalized_project_id,
                        requested_language=detected_language,
                        project_dir=identity.project_dir,
                        revision=state.revision,
                    )
                    if refreshed_state is not None:
                        state = refreshed_state

                        fallback_matches = homophone_service.search_homophone(
                            project_id=normalized_project_id,
                            revision=state.revision,
                            language=detected_language,
                            query_text=payload.query_text,
                            mode=payload.mode,
                            is_ignore_punctuation=payload.is_ignore_punctuation,
                            limit=payload.limit,
                        )
                        if fallback_matches:
                            matches = fallback_matches
                            indexed_language = detected_language

        return {
            "success": True,
            "data": {
                "project_id": normalized_project_id,
                "index_status": state.status,
                "revision": state.revision,
                "query": payload.query_text,
                "mode": payload.mode,
                "language": indexed_language,
                "is_ignore_punctuation": payload.is_ignore_punctuation,
                "matches": [
                    {
                        "sentence_index": item.sentence_index,
                        "token_index": item.token_index,
                        "token_text": item.token_text,
                        "char_start": item.char_start,
                        "char_end": item.char_end,
                        "cluster_id": item.cluster_id,
                        "reading_label": item.reading_label,
                    }
                    for item in matches
                ],
            },
        }

    @router.get("/projects/{project_id}/homophone/index-status")
    async def project_homophone_index_status(project_id: str):
        """Project 语义同音索引状态接口。"""
        identity = _resolve_project_identity_or_404(project_id)
        normalized_project_id = identity.project_id

        homophone_service = get_homophone_service()
        state = homophone_service.get_index_status(normalized_project_id)
        if state is None:
            return {
                "success": True,
                "data": {
                    "status": "missing",
                    "project_id": normalized_project_id,
                },
            }
        return {
            "success": True,
            "data": {
                "project_id": normalized_project_id,
                "revision": state.revision,
                "status": state.status,
                "last_committed_chunk": state.last_committed_chunk,
                "heartbeat_at": state.heartbeat_at,
                "updated_at": state.updated_at,
            },
        }

    @router.get("/settings/homophone/global-terms")
    async def get_homophone_global_terms():
        """读取全局专有名词表。"""
        homophone_service = get_homophone_service()
        terms = homophone_service.list_global_terms()
        return {
            "success": True,
            "data": {
                "items": [
                    {
                        "language": item.language,
                        "source_text": item.source_text,
                        "target_text": item.target_text,
                        "match_mode": item.match_mode,
                        "priority": item.priority,
                        "is_enabled": item.is_enabled,
                        "note": item.note,
                    }
                    for item in terms
                ]
            },
        }

    @router.put("/settings/homophone/global-terms")
    async def put_homophone_global_terms(payload: GlobalTermSyncRequest):
        """全量覆盖全局专有名词表。"""
        homophone_service = get_homophone_service()
        rules = [
            GlobalTermRule(
                language=item.language,
                source_text=item.source_text,
                target_text=item.target_text,
                match_mode=item.match_mode,
                priority=item.priority,
                is_enabled=item.is_enabled,
                note=item.note,
            )
            for item in payload.items
        ]
        homophone_service.replace_global_terms(rules)
        return {
            "success": True,
            "data": {
                "count": len(rules),
            },
        }

    @router.post("/projects/{project_id}/homophone/batch-replace")
    async def project_homophone_batch_replace(project_id: str, payload: BatchReplaceRequest):
        """Project 语义同音/正则/精确批量替换接口。"""
        identity = _resolve_project_identity_or_404(project_id)
        normalized_project_id = identity.project_id

        if not payload.selected_sentence_indices:
            return {
                "success": True,
                "data": {
                    "project_id": normalized_project_id,
                    "updated_count": 0,
                    "updated_indices": [],
                },
            }

        selected_indices: Set[int] = {int(item) for item in payload.selected_sentence_indices}
        all_segments = _collect_segments_for_project(identity.project_dir)
        segment_map: Dict[int, Dict[str, Any]] = {
            int(segment.get("id", 0)): segment
            for segment in all_segments
            if int(segment.get("id", 0)) in selected_indices
        }

        homophone_service = get_homophone_service()
        state = _ensure_homophone_index_ready(
            project_id=normalized_project_id,
            requested_language=payload.language,
            project_dir=identity.project_dir,
        )
        homophone_matches: Dict[int, List[Any]] = {}
        if payload.mode in {"homophone_strict", "homophone_fuzzy"} and state is not None:
            query_matches = homophone_service.search_homophone(
                project_id=normalized_project_id,
                revision=state.revision,
                language=payload.language,
                query_text=payload.query_text,
                mode=payload.mode,
                is_ignore_punctuation=payload.is_ignore_punctuation,
                limit=10000,
            )
            for match in query_matches:
                if match.sentence_index in selected_indices:
                    homophone_matches.setdefault(match.sentence_index, []).append(match)

        subtitle_manager = get_streaming_subtitle_manager_if_exists(normalized_project_id)
        updated_indices: List[int] = []

        for sentence_index in sorted(selected_indices):
            source_text = str(segment_map.get(sentence_index, {}).get("text", ""))
            if not source_text:
                continue

            replaced_text = source_text
            if payload.mode == "literal":
                replaced_text = source_text.replace(payload.query_text, payload.replace_text)
            elif payload.mode == "regex":
                try:
                    replaced_text = re.sub(payload.query_text, payload.replace_text, source_text)
                except re.error as exc:
                    raise HTTPException(status_code=400, detail=f"正则表达式无效: {exc}") from exc
            else:
                matched_items = homophone_matches.get(sentence_index, [])
                if not matched_items:
                    continue
                chars = list(source_text)
                for match in sorted(matched_items, key=lambda item: item.char_start, reverse=True):
                    start = max(0, int(match.char_start))
                    end = min(len(chars), int(match.char_end))
                    if start >= end:
                        continue
                    chars[start:end] = list(payload.replace_text)
                replaced_text = "".join(chars)

            if replaced_text == source_text:
                continue

            save_edit(identity.project_dir, sentence_index, {"text": replaced_text}, original_text=source_text)
            if subtitle_manager and sentence_index in subtitle_manager.sentences:
                sentence = subtitle_manager.sentences[sentence_index]
                if not sentence.is_modified:
                    sentence.original_text = sentence.text
                sentence.text = replaced_text
                sentence.text_clean = replaced_text
                sentence.is_modified = True

            push_subtitle_event(
                sse_manager,
                normalized_project_id,
                "edited",
                {
                    "index": sentence_index,
                    "sentence": {
                        "index": sentence_index,
                        "text": replaced_text,
                        "is_modified": True,
                        "original_text": source_text,
                    },
                    "source": "homophone_batch_replace",
                    "is_update": True,
                },
            )
            updated_indices.append(sentence_index)

        if updated_indices:
            refreshed_segments = _collect_segments_for_project(identity.project_dir)
            sync_project_index_delta(
                project_id=normalized_project_id,
                segments=refreshed_segments,
                updated_sentence_indices=updated_indices,
            )

        return {
            "success": True,
            "data": {
                "project_id": normalized_project_id,
                "updated_count": len(updated_indices),
                "updated_indices": updated_indices,
            },
        }

    return router
