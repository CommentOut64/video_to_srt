"""
四层文本后处理统一入口。
V3.2.0+dev.20260215.26
"""

from app.services.textflow.collection_layer import (
    CollectionAlignmentProcessor,
    CollectionFactBuilder,
    CollectionFactBuilderConfig,
)
from app.services.textflow.scoring_layer import (
    ScoringEvidenceFusion,
    ScoringEvidenceFusionConfig,
    ScoringSemanticInjectionProcessor,
)
from app.services.textflow.decision_layer import DecisionSegmentationProcessor
from app.services.textflow.output_layer import OutputLayerProcessor

# 稳定 API：对外暴露短名，避免调用方依赖具体层文件路径。
AlignmentProcessor = CollectionAlignmentProcessor
FactBuilder = CollectionFactBuilder
FactBuilderConfig = CollectionFactBuilderConfig
SemanticInjectionProcessor = ScoringSemanticInjectionProcessor
SegmentationProcessor = DecisionSegmentationProcessor
OutputProcessor = OutputLayerProcessor

__all__ = [
    "CollectionAlignmentProcessor",
    "CollectionFactBuilder",
    "CollectionFactBuilderConfig",
    "ScoringEvidenceFusion",
    "ScoringEvidenceFusionConfig",
    "ScoringSemanticInjectionProcessor",
    "DecisionSegmentationProcessor",
    "OutputLayerProcessor",
    "AlignmentProcessor",
    "FactBuilder",
    "FactBuilderConfig",
    "SemanticInjectionProcessor",
    "SegmentationProcessor",
    "OutputProcessor",
]
