"""
语言适配层公共入口。
"""

from app.services.language_policy.compiler import build_language_policy_snapshot
from app.services.language_policy.loader import (
    get_language_policy_config,
    reset_language_policy_cache,
)
from app.services.language_policy.registry import (
    get_language_policy_registry,
    reset_language_policy_registry,
)
from app.services.language_policy.resolver import resolve_language_tag
from app.services.language_policy.types import LanguagePolicySnapshot

__all__ = [
    "LanguagePolicySnapshot",
    "build_language_policy_snapshot",
    "get_language_policy_config",
    "get_language_policy_registry",
    "resolve_language_tag",
    "reset_language_policy_cache",
    "reset_language_policy_registry",
]
