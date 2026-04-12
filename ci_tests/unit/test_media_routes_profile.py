"""媒体路由 profile 分流单元测试。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.api.routes import media_routes
from app.core.config import config


@dataclass
class _FakeIdentity:
    project_id: str
    project_dir: Path
    legacy_job_id: str | None = None


class _FakeProjectIdResolver:
    def __init__(self, project_id: str, project_dir: Path) -> None:
        self._project_id = project_id
        self._project_dir = project_dir

    def resolve_or_fail(self, identifier: str) -> _FakeIdentity:
        if str(identifier).strip() != self._project_id:
            raise FileNotFoundError(f"任务不存在: {identifier}")
        return _FakeIdentity(
            project_id=self._project_id,
            project_dir=self._project_dir,
            legacy_job_id=None,
        )


class _FakeMediaPrepService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.full_task_status: dict[str, Any] | None = None
        self.proxy_status: dict[str, Any] | None = None
        self.preview_status: dict[str, Any] | None = None
        self.remux_status: dict[str, Any] | None = None
        self.normalize_status: dict[str, Any] | None = None

    def enqueue_proxy(self, job_id: str, video_path: Path, output_path: Path, priority: int = 10) -> bool:
        self.calls.append(
            {
                "method": "enqueue_proxy",
                "job_id": job_id,
                "video_path": video_path.name,
                "output_path": output_path.name,
                "priority": priority,
            }
        )
        return True

    def enqueue_preview(self, job_id: str, video_path: Path, output_path: Path, priority: int = 5) -> bool:
        self.calls.append(
            {
                "method": "enqueue_preview",
                "job_id": job_id,
                "video_path": video_path.name,
                "output_path": output_path.name,
                "priority": priority,
            }
        )
        return True

    def enqueue_remux(self, job_id: str, video_path: Path, output_path: Path, priority: int = 3) -> bool:
        self.calls.append(
            {
                "method": "enqueue_remux",
                "job_id": job_id,
                "video_path": video_path.name,
                "output_path": output_path.name,
                "priority": priority,
            }
        )
        return True

    def enqueue_normalize_h264(
        self,
        job_id: str,
        video_path: Path,
        output_path: Path,
        priority: int = 4,
    ) -> bool:
        self.calls.append(
            {
                "method": "enqueue_normalize_h264",
                "job_id": job_id,
                "video_path": video_path.name,
                "output_path": output_path.name,
                "priority": priority,
            }
        )
        return True

    def get_proxy_status(self, _job_id: str) -> dict[str, Any] | None:
        return self.proxy_status

    def get_preview_status(self, _job_id: str) -> dict[str, Any] | None:
        return self.preview_status

    def get_remux_status(self, _job_id: str) -> dict[str, Any] | None:
        return self.remux_status

    def get_normalize_status(self, _job_id: str) -> dict[str, Any] | None:
        return self.normalize_status

    def get_full_task_status(self, _job_id: str) -> dict[str, Any] | None:
        return self.full_task_status

    @staticmethod
    def _is_transcription_queue_busy() -> bool:
        return False


class _FakeScheduler:
    def request(self, *args, **kwargs) -> dict[str, Any]:
        return {"accepted": True, "reason": "queued_for_check"}

    def get_state(self, _project_id: str) -> dict[str, Any]:
        return {}

    def ensure_tracked(self, *args, **kwargs) -> None:
        return None


def _build_client(tmp_path: Path, monkeypatch, profile: str = "electron_native") -> tuple[TestClient, _FakeMediaPrepService, Path]:
    project_id = "project-1"
    project_dir = tmp_path / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    fake_resolver = _FakeProjectIdResolver(project_id=project_id, project_dir=project_dir)
    fake_media_prep = _FakeMediaPrepService()

    monkeypatch.setattr("app.api.routes.media_routes.get_project_id_resolver", lambda: fake_resolver)
    monkeypatch.setattr("app.services.media_prep_service.get_media_prep_service", lambda: fake_media_prep)
    monkeypatch.setattr("app.services.proxy_720_scheduler.get_proxy_scheduler", lambda: _FakeScheduler())

    monkeypatch.setattr("app.api.routes.media_routes._is_browser_compat_profile", lambda: profile == "browser_compat")
    monkeypatch.setattr("app.api.routes.media_routes._is_electron_native_profile", lambda: profile == "electron_native")
    monkeypatch.setattr(config, "MEDIA_PROFILE", profile, raising=False)

    # 避免文件流断言复杂化：用轻量 JSON 响应替代实际 Range 文件响应。
    monkeypatch.setattr(
        "app.api.routes.media_routes._serve_file_with_range",
        lambda file_path, _request, _media_type, job_id=None: JSONResponse(
            {"served_name": Path(file_path).name, "job_id": job_id}
        ),
    )

    app = FastAPI()
    app.include_router(media_routes.router)
    return TestClient(app), fake_media_prep, project_dir


def test_get_video_in_electron_profile_prefers_normalized_h264(tmp_path: Path, monkeypatch) -> None:
    client, _, project_dir = _build_client(tmp_path, monkeypatch, profile="electron_native")
    (project_dir / "normalized_h264.mp4").write_bytes(b"normalized")

    resp = client.get("/api/media/project-1/video")

    assert resp.status_code == 200
    assert resp.json()["served_name"] == "normalized_h264.mp4"


def test_get_video_in_browser_profile_prefers_ready_proxy_720p(tmp_path: Path, monkeypatch) -> None:
    client, fake_media_prep, project_dir = _build_client(tmp_path, monkeypatch, profile="browser_compat")
    (project_dir / "source.mp4").write_bytes(b"source")
    (project_dir / "preview_360p.mp4").write_bytes(b"preview")
    (project_dir / "remux.mp4").write_bytes(b"remux")
    (project_dir / "proxy_720p.mp4").write_bytes(b"proxy")
    fake_media_prep.proxy_status = {"status": "completed", "progress": 100}

    resp = client.get("/api/media/project-1/video")

    assert resp.status_code == 200
    assert resp.json()["served_name"] == "proxy_720p.mp4"


def test_get_video_in_electron_profile_prefers_remux_over_preview(tmp_path: Path, monkeypatch) -> None:
    client, _, project_dir = _build_client(tmp_path, monkeypatch, profile="electron_native")
    (project_dir / "source.mp4").write_bytes(b"source")
    (project_dir / "preview_360p.mp4").write_bytes(b"preview")
    (project_dir / "remux.mp4").write_bytes(b"remux")

    resp = client.get("/api/media/project-1/video")

    assert resp.status_code == 200
    assert resp.json()["served_name"] == "remux.mp4"


def test_proxy_status_in_electron_profile_enqueues_normalize_h264(tmp_path: Path, monkeypatch) -> None:
    client, fake_media_prep, project_dir = _build_client(tmp_path, monkeypatch, profile="electron_native")
    (project_dir / "source.mp4").write_bytes(b"video")

    monkeypatch.setattr(
        "app.api.routes.media_routes._analyze_transcode_requirement",
        lambda _video: (True, "编码不兼容", "transcode_full"),
    )

    resp = client.get("/api/media/project-1/proxy-status")
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["state"] == "analyzing"
    assert payload["media_profile"] == "electron_native"
    assert any(call["method"] == "enqueue_normalize_h264" for call in fake_media_prep.calls)
    assert not any(call["method"] == "enqueue_proxy" for call in fake_media_prep.calls)


def test_progressive_status_in_electron_profile_enqueues_normalize_h264(tmp_path: Path, monkeypatch) -> None:
    client, fake_media_prep, project_dir = _build_client(tmp_path, monkeypatch, profile="electron_native")
    (project_dir / "source.mp4").write_bytes(b"video")

    monkeypatch.setattr(
        "app.api.routes.media_routes._analyze_transcode_requirement",
        lambda _video: (True, "编码不兼容", "transcode_full"),
    )

    resp = client.get("/api/media/project-1/status/progressive")
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["media_profile"] == "electron_native"
    assert any(call["method"] == "enqueue_normalize_h264" for call in fake_media_prep.calls)
    assert not any(call["method"] == "enqueue_preview" for call in fake_media_prep.calls)
    assert payload["normalize_h264"]["exists"] is False
    assert payload["current_resolution"] is None


def test_progressive_status_in_browser_profile_keeps_preview_pipeline(tmp_path: Path, monkeypatch) -> None:
    client, fake_media_prep, project_dir = _build_client(tmp_path, monkeypatch, profile="browser_compat")
    (project_dir / "source.mp4").write_bytes(b"video")

    monkeypatch.setattr(
        "app.api.routes.media_routes._analyze_transcode_requirement",
        lambda _video: (True, "编码不兼容", "transcode_full"),
    )

    resp = client.get("/api/media/project-1/status/progressive")
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["media_profile"] == "browser_compat"
    assert any(call["method"] == "enqueue_preview" for call in fake_media_prep.calls)
    assert not any(call["method"] == "enqueue_normalize_h264" for call in fake_media_prep.calls)


def test_upgrade_720p_is_disabled_by_electron_profile(tmp_path: Path, monkeypatch) -> None:
    client, _, _ = _build_client(tmp_path, monkeypatch, profile="electron_native")

    resp = client.post("/api/media/project-1/upgrade-720p")
    payload = resp.json()

    assert resp.status_code == 400
    assert payload["reason"] == "disabled_by_profile"
    assert payload["media_profile"] == "electron_native"


def test_post_process_in_electron_profile_uses_normalize_h264(tmp_path: Path, monkeypatch) -> None:
    client, fake_media_prep, project_dir = _build_client(tmp_path, monkeypatch, profile="electron_native")
    (project_dir / "source.mp4").write_bytes(b"video")
    (project_dir / "audio.wav").write_bytes(b"audio")

    monkeypatch.setattr(
        "app.api.routes.media_routes._analyze_transcode_requirement",
        lambda _video: (True, "编码不兼容", "transcode_full"),
    )

    async def _fake_audio_peaks(_identifier: str, _samples: int = 0, _method: str = "auto"):
        return JSONResponse({"ok": True})

    async def _fake_thumbnails(_identifier: str, _count: int = 10, _sprite: bool = True):
        return JSONResponse({"ok": True})

    monkeypatch.setattr("app.api.routes.media_routes.get_audio_peaks", _fake_audio_peaks)
    monkeypatch.setattr("app.api.routes.media_routes.get_thumbnails", _fake_thumbnails)

    resp = client.post("/api/media/project-1/post-process")
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["proxy_needed"] is True
    assert payload["proxy"] is True
    assert any(call["method"] == "enqueue_normalize_h264" for call in fake_media_prep.calls)
    assert not any(call["method"] == "enqueue_proxy" for call in fake_media_prep.calls)


def test_post_process_in_browser_profile_still_uses_proxy_pipeline(tmp_path: Path, monkeypatch) -> None:
    client, fake_media_prep, project_dir = _build_client(tmp_path, monkeypatch, profile="browser_compat")
    (project_dir / "source.mp4").write_bytes(b"video")
    (project_dir / "audio.wav").write_bytes(b"audio")

    monkeypatch.setattr(
        "app.api.routes.media_routes._analyze_transcode_requirement",
        lambda _video: (True, "编码不兼容", "transcode_full"),
    )

    async def _fake_audio_peaks(_identifier: str, _samples: int = 0, _method: str = "auto"):
        return JSONResponse({"ok": True})

    async def _fake_thumbnails(_identifier: str, _count: int = 10, _sprite: bool = True):
        return JSONResponse({"ok": True})

    monkeypatch.setattr("app.api.routes.media_routes.get_audio_peaks", _fake_audio_peaks)
    monkeypatch.setattr("app.api.routes.media_routes.get_thumbnails", _fake_thumbnails)

    resp = client.post("/api/media/project-1/post-process")
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["proxy_needed"] is True
    assert payload["proxy"] is True
    assert any(call["method"] == "enqueue_proxy" for call in fake_media_prep.calls)
    assert not any(call["method"] == "enqueue_normalize_h264" for call in fake_media_prep.calls)


def test_media_info_in_electron_profile_enqueues_normalize_h264(tmp_path: Path, monkeypatch) -> None:
    client, fake_media_prep, project_dir = _build_client(tmp_path, monkeypatch, profile="electron_native")
    (project_dir / "source.mp4").write_bytes(b"video")
    (project_dir / "thumbnail.jpg").write_bytes(b"thumb")
    (project_dir / "peaks_2000.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "app.api.routes.media_routes._analyze_transcode_requirement",
        lambda _video: (True, "编码不兼容", "transcode_full"),
    )
    monkeypatch.setattr("app.api.routes.media_routes._has_audio_stream", lambda _video: True)
    monkeypatch.setattr("app.api.routes.media_routes._get_video_codec", lambda _video: "hevc")

    resp = client.get("/api/media/project-1/info")
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["video"]["needs_proxy"] is True
    assert payload["video"]["media_profile"] == "electron_native"
    assert any(call["method"] == "enqueue_normalize_h264" for call in fake_media_prep.calls)
    assert not any(call["method"] == "enqueue_proxy" for call in fake_media_prep.calls)


def test_media_info_uses_normalize_progress_as_primary(tmp_path: Path, monkeypatch) -> None:
    client, fake_media_prep, project_dir = _build_client(tmp_path, monkeypatch, profile="electron_native")
    (project_dir / "source.mp4").write_bytes(b"video")

    fake_media_prep.proxy_status = {"status": "processing", "progress": 88}
    fake_media_prep.normalize_status = {"status": "processing", "progress": 31}

    monkeypatch.setattr(
        "app.api.routes.media_routes._analyze_transcode_requirement",
        lambda _video: (True, "编码不兼容", "transcode_full"),
    )
    monkeypatch.setattr("app.api.routes.media_routes._has_audio_stream", lambda _video: True)
    monkeypatch.setattr("app.api.routes.media_routes._get_video_codec", lambda _video: "hevc")

    resp = client.get("/api/media/project-1/info?retry_missing=false")
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["video"]["proxy_progress"] == 31
    assert payload["video"]["normalize_generating"] is True
    assert payload["video"]["normalize_progress"] == 31
