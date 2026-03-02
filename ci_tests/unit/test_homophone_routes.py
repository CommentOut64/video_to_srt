"""同音路由接口单元测试（纯 Project 语义）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.homophone_routes import create_homophone_router
from app.services.homophone.db import GlobalTermRule, IndexState


@dataclass
class _FakeIdentity:
    project_id: str
    project_dir: Path
    legacy_job_id: str | None = None


class _FakeProjectIdResolver:
    def __init__(self, project_dir: Path, project_id: str = "project-1") -> None:
        self._project_dir = project_dir
        self._project_id = project_id

    def resolve_or_fail(self, identifier: str) -> _FakeIdentity:
        if str(identifier) != self._project_id:
            raise FileNotFoundError(f"项目不存在: {identifier}")
        return _FakeIdentity(
            project_id=self._project_id,
            project_dir=self._project_dir,
            legacy_job_id=None,
        )


@dataclass
class _FakeMatch:
    sentence_index: int
    token_index: int
    token_text: str
    char_start: int
    char_end: int
    cluster_id: str
    reading_label: str


class _FakeHomophoneService:
    def __init__(self) -> None:
        self._state = IndexState(
            job_id="project-1",
            revision=1,
            status="ready",
            last_committed_chunk=0,
            heartbeat_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        self._terms: List[GlobalTermRule] = [
            GlobalTermRule(
                language="auto",
                source_text="旧词",
                target_text="新词",
                match_mode="exact",
                priority=100,
                is_enabled=True,
                note="",
            )
        ]

    def get_index_status(self, job_id: str) -> IndexState | None:
        if job_id != "project-1":
            return None
        return self._state

    def index_chunk(
        self,
        *,
        job_id: str,
        revision: int,
        chunk_index: int,
        language: str,
        sentences: List[Any],
    ) -> None:
        return

    def search_homophone(
        self,
        *,
        job_id: str,
        revision: int,
        language: str,
        query_text: str,
        mode: str,
        is_ignore_punctuation: bool,
        limit: int,
    ) -> List[_FakeMatch]:
        if query_text == "空":
            return []
        return [
            _FakeMatch(
                sentence_index=0,
                token_index=0,
                token_text="同音",
                char_start=0,
                char_end=2,
                cluster_id="cluster_x",
                reading_label="tongyin",
            )
        ]

    def list_global_terms(self) -> List[GlobalTermRule]:
        return list(self._terms)

    def replace_global_terms(self, rules: List[GlobalTermRule]) -> None:
        self._terms = list(rules)


def _build_client(tmp_path: Path, monkeypatch) -> tuple[TestClient, _FakeHomophoneService]:
    project_dir = tmp_path / "project-1"
    project_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = project_dir / "checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "transcription": {
                    "sentences_snapshot": [
                        {
                            "_index": 0,
                            "start": 0.0,
                            "end": 1.0,
                            "text": "foo test",
                            "is_modified": False,
                        }
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    fake_homophone_service = _FakeHomophoneService()
    fake_resolver = _FakeProjectIdResolver(project_dir=project_dir, project_id="project-1")

    monkeypatch.setattr(
        "app.api.routes.homophone_routes.get_homophone_service",
        lambda: fake_homophone_service,
    )
    monkeypatch.setattr(
        "app.api.routes.homophone_routes.get_project_id_resolver",
        lambda: fake_resolver,
    )
    monkeypatch.setattr(
        "app.api.routes.homophone_routes.push_subtitle_event",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "app.api.routes.homophone_routes.get_streaming_subtitle_manager_if_exists",
        lambda _project_id: None,
    )

    app = FastAPI()
    app.include_router(create_homophone_router())
    return TestClient(app), fake_homophone_service


def test_homophone_find_and_index_status_api(tmp_path: Path, monkeypatch) -> None:
    client, _ = _build_client(tmp_path, monkeypatch)

    find_resp = client.post(
        "/api/projects/project-1/homophone/find",
        json={
            "mode": "homophone_strict",
            "query_text": "同音",
            "language": "zh",
            "is_ignore_punctuation": False,
            "limit": 10,
        },
    )
    assert find_resp.status_code == 200
    payload = find_resp.json()
    assert payload["success"] is True
    assert payload["data"]["project_id"] == "project-1"
    assert payload["data"]["index_status"] == "ready"
    assert len(payload["data"]["matches"]) == 1

    status_resp = client.get("/api/projects/project-1/homophone/index-status")
    assert status_resp.status_code == 200
    status_payload = status_resp.json()
    assert status_payload["success"] is True
    assert status_payload["data"]["status"] == "ready"
    assert status_payload["data"]["project_id"] == "project-1"


def test_homophone_global_terms_api(tmp_path: Path, monkeypatch) -> None:
    client, fake_service = _build_client(tmp_path, monkeypatch)

    get_resp = client.get("/api/settings/homophone/global-terms")
    assert get_resp.status_code == 200
    assert get_resp.json()["data"]["items"][0]["source_text"] == "旧词"

    put_resp = client.put(
        "/api/settings/homophone/global-terms",
        json={
            "items": [
                {
                    "language": "auto",
                    "source_text": "旧",
                    "target_text": "新",
                    "match_mode": "exact",
                    "priority": 1,
                    "is_enabled": True,
                    "note": "",
                }
            ]
        },
    )
    assert put_resp.status_code == 200
    assert len(fake_service.list_global_terms()) == 1
    assert fake_service.list_global_terms()[0].source_text == "旧"


def test_homophone_batch_replace_api(tmp_path: Path, monkeypatch) -> None:
    client, _ = _build_client(tmp_path, monkeypatch)

    resp = client.post(
        "/api/projects/project-1/homophone/batch-replace",
        json={
            "mode": "literal",
            "query_text": "foo",
            "replace_text": "bar",
            "language": "zh",
            "is_ignore_punctuation": False,
            "selected_sentence_indices": [0],
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert payload["data"]["project_id"] == "project-1"
    assert payload["data"]["updated_count"] == 1
    assert payload["data"]["updated_indices"] == [0]
