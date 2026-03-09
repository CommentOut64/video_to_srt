from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from app.core.config import (
    MEDIA_PROFILE_BROWSER_COMPAT,
    MEDIA_PROFILE_ELECTRON_NATIVE,
    MEDIA_PROFILE_LITE_SAFE,
    config,
)


@dataclass(frozen=True)
class RuntimeMediaPolicy:
    profile: str
    browser_preview_enabled: bool
    auto_trigger_720p: bool
    variant_priority: Tuple[str, ...]


class RuntimeMediaPolicyService:
    """运行时媒体策略服务。

    设计取舍：
    - 把媒体 profile 对“可播放产物优先级”的判断从路由层抽离，避免 Phase 4 后继续在
      `media_routes.py` 堆叠分支。
    - `lite_safe` 的目标不是追求最高画质，而是尽快拿到低负载可播放版本，因此优先级与
      `browser_compat`、`electron_native` 都不同。
    """

    def get_policy(self, profile: Optional[str] = None) -> RuntimeMediaPolicy:
        resolved_profile = str(profile or config.MEDIA_PROFILE or MEDIA_PROFILE_ELECTRON_NATIVE).strip().lower()
        if resolved_profile == MEDIA_PROFILE_BROWSER_COMPAT:
            return RuntimeMediaPolicy(
                profile=MEDIA_PROFILE_BROWSER_COMPAT,
                browser_preview_enabled=True,
                auto_trigger_720p=True,
                variant_priority=("proxy_720p", "remux", "preview_360p", "source"),
            )
        if resolved_profile == MEDIA_PROFILE_LITE_SAFE:
            return RuntimeMediaPolicy(
                profile=MEDIA_PROFILE_LITE_SAFE,
                browser_preview_enabled=True,
                auto_trigger_720p=False,
                variant_priority=("preview_360p", "normalized_h264", "remux", "source"),
            )
        return RuntimeMediaPolicy(
            profile=MEDIA_PROFILE_ELECTRON_NATIVE,
            browser_preview_enabled=False,
            auto_trigger_720p=False,
            variant_priority=("normalized_h264", "remux", "source"),
        )

    def is_browser_preview_profile(self, profile: Optional[str] = None) -> bool:
        return self.get_policy(profile).browser_preview_enabled

    def select_best_variant(
        self,
        variants: Dict[str, Optional[Path]],
        profile: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Path]]:
        policy = self.get_policy(profile)
        for variant in policy.variant_priority:
            path = variants.get(variant)
            if path is not None:
                return variant, path
        return None, None

    def build_variant_url(self, project_id: str, variant: Optional[str]) -> Optional[str]:
        if not project_id or not variant:
            return None
        if variant == "preview_360p":
            return f"/api/media/{project_id}/video/preview"
        if variant == "source":
            return f"/api/media/{project_id}/video/source"
        return f"/api/media/{project_id}/video"

    def describe_variant_resolution(self, variant: Optional[str]) -> Optional[str]:
        mapping = {
            "preview_360p": "360p",
            "proxy_720p": "720p",
            "normalized_h264": "normalized_h264",
            "remux": "remux",
            "source": "source",
        }
        return mapping.get(variant)


_runtime_media_policy_service: Optional[RuntimeMediaPolicyService] = None


def get_runtime_media_policy_service() -> RuntimeMediaPolicyService:
    global _runtime_media_policy_service
    if _runtime_media_policy_service is None:
        _runtime_media_policy_service = RuntimeMediaPolicyService()
    return _runtime_media_policy_service
