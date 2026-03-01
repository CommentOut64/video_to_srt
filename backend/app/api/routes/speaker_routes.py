"""
Speaker API 路由。

职责：
1. 提供说话人 profile 查询与编辑接口。
2. 提供句级 speaker 改绑接口（后端为唯一真源）。
3. 写入成功后通过 SSE 回推前端，避免多端状态分叉。
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.project_id_resolver import get_project_id_resolver
from app.services.speaker_store import SpeakerStoreService
from app.services.sse_service import get_sse_manager, push_subtitle_event
from app.services.streaming_subtitle import get_streaming_subtitle_manager_if_exists
from app.services.transcription_service import TranscriptionService


class SpeakerProfilePatchRequest(BaseModel):
    """说话人 profile 更新请求。"""

    display_name: str | None = Field(default=None, description="展示名称")
    color_key: str | None = Field(default=None, description="颜色键")
    is_locked: bool | None = Field(default=None, description="是否锁定")
    status: str | None = Field(default=None, description="状态 candidate/confirmed/merged")


class SubtitleSpeakerPatchRequest(BaseModel):
    """句级 speaker 改绑请求。"""

    speaker_id: str = Field(..., min_length=1, description="目标说话人 ID")


class SpeakerMergeRequest(BaseModel):
    """说话人合并请求。"""

    source_speaker_id: str = Field(..., min_length=1, description="源 speaker")
    target_speaker_id: str = Field(..., min_length=1, description="目标 speaker")


def create_speaker_router(transcription_service: TranscriptionService) -> APIRouter:
    """创建 speaker 路由。"""
    router = APIRouter(prefix="/api/speakers", tags=["speakers"])
    sse_manager = get_sse_manager()

    def _resolve_job(project_id: str) -> Any:
        normalized_identifier = str(project_id or "").strip()
        job = transcription_service.get_job(normalized_identifier)
        if job:
            if not getattr(job, "project_id", None):
                job.project_id = normalized_identifier
            return job

        try:
            identity = get_project_id_resolver().resolve_or_fail(normalized_identifier)
        except Exception:
            raise HTTPException(status_code=404, detail="任务未找到")

        for identifier in (identity.project_id, identity.legacy_job_id):
            normalized = str(identifier or "").strip()
            if not normalized:
                continue
            job = transcription_service.get_job(normalized)
            if job:
                job.project_id = identity.project_id
                return job

        runtime_jobs = getattr(getattr(transcription_service, "job_lifecycle", None), "jobs", None)
        if isinstance(runtime_jobs, dict):
            for runtime_job in runtime_jobs.values():
                if str(getattr(runtime_job, "project_id", "") or "").strip() != identity.project_id:
                    continue
                return runtime_job

        if not job:
            raise HTTPException(status_code=404, detail="任务未找到")
        return job

    def _resolve_speaker_service(project_id: str) -> SpeakerStoreService:
        job = _resolve_job(project_id)
        return SpeakerStoreService(job_dir=Path(job.dir))

    def _build_sentence_payload(
        *,
        sentence_index: int,
        link_row: dict[str, object],
        subtitle_manager: Any,
    ) -> dict[str, object]:
        """构建 subtitle.revised 的句子 payload。"""
        sentence_payload: dict[str, object] = {
            "index": int(sentence_index),
            "speaker_id": str(link_row.get("speaker_id") or "unknown"),
            "turn_id": link_row.get("turn_id"),
            "speaker_color_key": str(link_row.get("speaker_color_key") or "speaker-01"),
            "binding_source": str(link_row.get("binding_source") or "user"),
            "speaker_label": str(link_row.get("speaker_label") or link_row.get("speaker_id") or "unknown"),
            "start": float(link_row.get("start") or 0.0),
            "end": float(link_row.get("end") or 0.0),
        }

        if subtitle_manager and sentence_index in subtitle_manager.sentences:
            sentence = subtitle_manager.sentences[sentence_index]
            sentence.speaker_id = sentence_payload["speaker_id"]
            sentence.turn_id = sentence_payload["turn_id"]
            sentence.speaker_color_key = sentence_payload["speaker_color_key"]
            sentence.binding_source = sentence_payload["binding_source"]
            sentence.speaker_label = sentence_payload["speaker_label"]
            sentence_payload = sentence.to_dict()
            sentence_payload["index"] = sentence_index
        return sentence_payload

    @router.get("/{project_id}/profiles")
    async def list_profiles(project_id: str) -> dict[str, object]:
        """查询说话人 profile 列表。"""
        speaker_service = _resolve_speaker_service(project_id)
        profiles = speaker_service.list_speaker_profiles()
        return {
            "project_id": project_id,
            "profiles": profiles,
        }

    @router.get("/{project_id}/profiles/{speaker_id}/subtitles")
    async def list_profile_subtitles(project_id: str, speaker_id: str) -> dict[str, object]:
        """查询指定说话人的句子绑定列表。"""
        speaker_service = _resolve_speaker_service(project_id)
        links = speaker_service.list_subtitle_speaker_links(speaker_id=speaker_id)
        return {
            "project_id": project_id,
            "speaker_id": speaker_id,
            "subtitles": links,
        }

    @router.patch("/{project_id}/profiles/{speaker_id}")
    async def patch_profile(
        project_id: str,
        speaker_id: str,
        payload: SpeakerProfilePatchRequest,
    ) -> dict[str, object]:
        """更新说话人 profile。"""
        speaker_service = _resolve_speaker_service(project_id)
        update_payload = payload.dict(exclude_none=True)
        if not update_payload:
            raise HTTPException(status_code=400, detail="更新内容为空")

        updated = speaker_service.update_speaker_profile(
            speaker_id=speaker_id,
            display_name=payload.display_name,
            color_key=payload.color_key,
            is_locked=payload.is_locked,
            status=payload.status,
        )
        if updated is None:
            raise HTTPException(status_code=404, detail="说话人不存在")

        revision_id = uuid.uuid4().hex
        updated_at = float(updated.get("updated_at") or 0.0)
        speaker_service.append_audit_log(
            action="speaker.profile.patch",
            speaker_id=speaker_id,
            payload={
                "request": update_payload,
                "result": updated,
                "revision_id": revision_id,
            },
        )

        push_subtitle_event(
            sse_manager,
            project_id,
            "speaker_profiles",
            {
                "revision_id": revision_id,
                "updated_at": updated_at,
                "source": "speaker_profile_patch",
                "profiles": [updated],
                "changed_speaker_ids": [speaker_id],
            },
        )
        return {
            "success": True,
            "data": updated,
            "revision_id": revision_id,
            "updated_at": updated_at,
        }

    @router.patch("/{project_id}/subtitles/{sentence_index}")
    async def patch_subtitle_speaker(
        project_id: str,
        sentence_index: int,
        payload: SubtitleSpeakerPatchRequest,
    ) -> dict[str, object]:
        """改绑句级 speaker。"""
        speaker_service = _resolve_speaker_service(project_id)
        updated_link = speaker_service.rebind_subtitle_speaker(
            sentence_index=sentence_index,
            speaker_id=payload.speaker_id,
            binding_source="user",
        )
        if updated_link is None:
            raise HTTPException(status_code=404, detail="句子绑定不存在")

        revision_id = uuid.uuid4().hex
        updated_at = float(updated_link.get("updated_at") or 0.0)
        subtitle_manager = get_streaming_subtitle_manager_if_exists(project_id)
        sentence_payload = _build_sentence_payload(
            sentence_index=sentence_index,
            link_row=updated_link,
            subtitle_manager=subtitle_manager,
        )

        speaker_service.append_audit_log(
            action="speaker.subtitle.rebind",
            speaker_id=str(updated_link.get("speaker_id") or payload.speaker_id),
            sentence_index=sentence_index,
            payload={
                "speaker_id": payload.speaker_id,
                "result": updated_link,
                "revision_id": revision_id,
            },
        )

        push_subtitle_event(
            sse_manager,
            project_id,
            "revised",
            {
                "index": sentence_index,
                "sentence": sentence_payload,
                "revision_id": revision_id,
                "updated_at": updated_at,
                "source": "speaker_rebind",
            },
        )
        return {
            "success": True,
            "data": sentence_payload,
            "revision_id": revision_id,
            "updated_at": updated_at,
        }

    @router.post("/{project_id}/profiles/merge")
    async def merge_profiles(project_id: str, payload: SpeakerMergeRequest) -> dict[str, object]:
        """合并 speaker。"""
        source_speaker_id = payload.source_speaker_id.strip()
        target_speaker_id = payload.target_speaker_id.strip()
        if source_speaker_id == target_speaker_id:
            raise HTTPException(status_code=400, detail="源 speaker 与目标 speaker 不能相同")

        speaker_service = _resolve_speaker_service(project_id)
        affected_before = speaker_service.list_subtitle_speaker_links(
            speaker_id=source_speaker_id
        )
        merge_result = speaker_service.merge_speakers(
            source_speaker_id=source_speaker_id,
            target_speaker_id=target_speaker_id,
        )
        if merge_result is None:
            raise HTTPException(status_code=404, detail="源或目标说话人不存在")

        revision_id = uuid.uuid4().hex
        updated_at = float(
            merge_result.get("target_profile", {}).get("updated_at")
            or 0.0
        )
        speaker_service.append_audit_log(
            action="speaker.profile.merge",
            speaker_id=target_speaker_id,
            payload={
                "source_speaker_id": source_speaker_id,
                "target_speaker_id": target_speaker_id,
                "merge_result": merge_result,
                "revision_id": revision_id,
            },
        )

        affected_sentence_indices = [
            int(item["sentence_index"]) for item in affected_before
        ]
        revised_sentences: list[dict[str, object]] = []
        subtitle_manager = get_streaming_subtitle_manager_if_exists(project_id)
        if affected_sentence_indices:
            affected_after_map = speaker_service.get_subtitle_speaker_map(
                sentence_indices=affected_sentence_indices
            )
            for sentence_index in affected_sentence_indices:
                link_row = affected_after_map.get(sentence_index)
                if not link_row:
                    continue
                revised_sentences.append(
                    _build_sentence_payload(
                        sentence_index=sentence_index,
                        link_row=link_row,
                        subtitle_manager=subtitle_manager,
                    )
                )

        if revised_sentences:
            push_subtitle_event(
                sse_manager,
                project_id,
                "revised",
                {
                    "sentences": revised_sentences,
                    "revision_id": revision_id,
                    "updated_at": updated_at,
                    "source": "speaker_merge",
                },
            )

        profiles = speaker_service.list_speaker_profiles()
        push_subtitle_event(
            sse_manager,
            project_id,
            "speaker_profiles",
            {
                "revision_id": revision_id,
                "updated_at": updated_at,
                "source": "speaker_merge",
                "profiles": profiles,
                "changed_speaker_ids": [source_speaker_id, target_speaker_id],
            },
        )
        return {
            "success": True,
            "data": {
                **merge_result,
                "affected_sentence_indices": affected_sentence_indices,
            },
            "revision_id": revision_id,
            "updated_at": updated_at,
        }

    return router
