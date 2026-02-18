"""
soft-cut CutPlan 生产器契约与装配入口。
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from app.services.alignment.types import AlignedFacts, AnnotatedWord, FusedEvidence


@runtime_checkable
class SoftCutPlanProvider(Protocol):
    """CutPlan 生产器协议。"""

    def build_plan(
        self,
        *,
        annotated_words: Sequence[AnnotatedWord],
        stream_id: str,
        block_id: str,
        is_last_chunk: bool,
        aligned_facts: Optional[AlignedFacts] = None,
        fused_evidence: Optional[FusedEvidence] = None,
    ) -> Optional[Any]:
        """为裁决层生成 CutPlan。"""


class M1InternalSoftCutPlanProvider:
    """
    默认 M1 provider（Adapter Pattern）。

    Why:
    - 通过统一 provider 协议包装现有 M1 逻辑
    - 后续 M2 仅新增 provider 并切换配置，不再改 Pipeline 主链
    """

    def __init__(self, *, pipeline: Any) -> None:
        self._pipeline = pipeline

    def build_plan(
        self,
        *,
        annotated_words: Sequence[AnnotatedWord],
        stream_id: str,
        block_id: str,
        is_last_chunk: bool,
        aligned_facts: Optional[AlignedFacts] = None,
        fused_evidence: Optional[FusedEvidence] = None,
    ) -> Optional[Any]:
        return self._pipeline._build_soft_cut_plan_for_decision_m1(
            annotated_words=annotated_words,
            stream_id=stream_id,
            block_id=block_id,
            is_last_chunk=is_last_chunk,
            aligned_facts=aligned_facts,
            fused_evidence=fused_evidence,
        )


def resolve_soft_cut_plan_provider(
    *,
    pipeline: Any,
    provider_name: str,
    provider_class_path: str,
    logger: Optional[logging.Logger] = None,
) -> SoftCutPlanProvider:
    """
    解析并实例化 CutPlan provider。

    规则：
    1. 若配置了 `provider_class_path`，优先按动态类加载
    2. 否则按 `provider_name` 选择内置 provider
    3. 失败时回退到 `M1InternalSoftCutPlanProvider`
    """
    active_logger = logger or logging.getLogger(__name__)
    class_path = str(provider_class_path or "").strip()
    if class_path:
        try:
            provider = _build_provider_from_class_path(
                pipeline=pipeline,
                class_path=class_path,
            )
            active_logger.info(f"soft-cut provider 已加载: class={class_path}")
            return provider
        except Exception as exc:
            active_logger.warning(f"soft-cut provider 动态加载失败，回退 M1: class={class_path} error={exc}")

    normalized_name = str(provider_name or "m1_internal").strip().lower()
    if normalized_name in {"m1_internal", "m1", "default"}:
        return M1InternalSoftCutPlanProvider(pipeline=pipeline)

    active_logger.warning(f"soft-cut provider 名称未知，回退 M1: provider={normalized_name}")
    return M1InternalSoftCutPlanProvider(pipeline=pipeline)


def _build_provider_from_class_path(
    *,
    pipeline: Any,
    class_path: str,
) -> SoftCutPlanProvider:
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    provider_cls = getattr(module, class_name)
    try:
        provider = provider_cls(pipeline=pipeline)
    except TypeError:
        provider = provider_cls(pipeline)
    if not isinstance(provider, SoftCutPlanProvider):
        raise TypeError(f"provider 未实现 SoftCutPlanProvider 协议: {class_path}")
    return provider


__all__ = [
    "M1InternalSoftCutPlanProvider",
    "SoftCutPlanProvider",
    "resolve_soft_cut_plan_provider",
]
