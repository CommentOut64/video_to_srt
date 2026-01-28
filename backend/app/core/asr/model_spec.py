"""
模型规格与来源定义（ModelSpec）
用于统一描述各类模型（ASR/标点/VAD/分离/质量/LangID/声纹等），支持多框架与扩展。
V3.2.0+dev.20260114.01
V3.2.0+dev.20260127.01: 增加 speaker 模型类型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Any

Framework = Literal["onnx", "torch", "ctranslate2", "external"]
ModelKind = Literal["asr", "punct", "vad", "separation", "quality", "langid", "speaker"]


@dataclass
class ModelSource:
    """模型来源信息，支持本地、远程仓库与镜像。"""

    repo_id: Optional[str] = None
    local_path: Optional[str] = None
    files: List[str] = field(default_factory=list)
    hash: Optional[str] = None
    mirrors: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelSource":
        return cls(
            repo_id=data.get("repo_id"),
            local_path=data.get("local_path"),
            files=data.get("files") or [],
            hash=data.get("hash"),
            mirrors=data.get("mirrors") or [],
        )


@dataclass
class ModelSpec:
    """模型统一描述，供注册表与 ModelManager V2 使用。"""

    id: str
    kind: ModelKind
    framework: Framework
    format: str
    default_device: Literal["cuda", "cpu", "auto"] = "auto"
    compute_type: str = "auto"
    source: ModelSource = field(default_factory=ModelSource)
    resources: Dict[str, int] = field(default_factory=dict)  # 如 vram_mb、cpu_threads
    features: Dict[str, Any] = field(default_factory=dict)   # 语言、时间戳类型、标点等
    env: Dict[str, Any] = field(default_factory=dict)        # 依赖或环境变量
    fallback: List[str] = field(default_factory=list)        # 可选，标点等按语言兜底

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelSpec":
        return cls(
            id=data["id"],
            kind=data["kind"],
            framework=data["framework"],
            format=data["format"],
            default_device=data.get("default_device", "auto"),
            compute_type=data.get("compute_type", "auto"),
            source=ModelSource.from_dict(data.get("source", {})),
            resources=data.get("resources") or {},
            features=data.get("features") or {},
            env=data.get("env") or {},
            fallback=data.get("fallback") or [],
        )

