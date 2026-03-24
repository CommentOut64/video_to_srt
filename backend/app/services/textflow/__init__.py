"""
四层文本后处理统一入口。
V3.2.0+dev.20260215.26
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MAP = {
    "CollectionAlignmentProcessor": ("app.services.textflow.collection_layer", "CollectionAlignmentProcessor"),
    "CollectionFactBuilder": ("app.services.textflow.collection_layer", "CollectionFactBuilder"),
    "CollectionFactBuilderConfig": ("app.services.textflow.collection_layer", "CollectionFactBuilderConfig"),
    "ScoringEvidenceFusion": ("app.services.textflow.scoring_layer", "ScoringEvidenceFusion"),
    "ScoringEvidenceFusionConfig": ("app.services.textflow.scoring_layer", "ScoringEvidenceFusionConfig"),
    "ScoringSemanticInjectionProcessor": (
        "app.services.textflow.scoring_layer",
        "ScoringSemanticInjectionProcessor",
    ),
    "DecisionSegmentationProcessor": ("app.services.textflow.decision_layer", "DecisionSegmentationProcessor"),
    "OutputLayerProcessor": ("app.services.textflow.output_dispatch_adapter", "OutputLayerProcessor"),
    "AlignmentProcessor": ("app.services.textflow.collection_layer", "CollectionAlignmentProcessor"),
    "FactBuilder": ("app.services.textflow.collection_layer", "CollectionFactBuilder"),
    "FactBuilderConfig": ("app.services.textflow.collection_layer", "CollectionFactBuilderConfig"),
    "SemanticInjectionProcessor": (
        "app.services.textflow.scoring_layer",
        "ScoringSemanticInjectionProcessor",
    ),
    "SegmentationProcessor": ("app.services.textflow.decision_layer", "DecisionSegmentationProcessor"),
    "OutputProcessor": ("app.services.textflow.output_dispatch_adapter", "OutputLayerProcessor"),
}

__all__ = list(_EXPORT_MAP.keys())


def __getattr__(name: str) -> Any:
    target = _EXPORT_MAP.get(name)
    if target is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module_name, attr_name = target
    module = import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(__all__)
