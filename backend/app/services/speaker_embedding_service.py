"""
SpeakerEmbeddingService - 声纹提取服务

V3.2.0+dev.20260127.06: 新增声纹批量推理与缓存元数据构建。
V3.2.0+dev.20260127.08: 设备选择与批处理大小自适应。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from app.services.audio.chunk_engine import AudioChunk


@dataclass(frozen=True)
class SpeakerEmbeddingResult:
    """声纹向量结果（保留向量本体）。"""
    embedding: List[float]


SpeakerEmbeddingCallback = Optional[
    Callable[[Sequence[AudioChunk], Sequence[SpeakerEmbeddingResult]], None]
]


class SpeakerEmbeddingProvider(Protocol):
    """
    声纹提取提供者接口（策略模式扩展点，便于后续接入其他模型）。
    """

    def extract_embeddings(
        self,
        chunks: Sequence[AudioChunk],
        device: str = "auto",
        mode: Optional[str] = None,
        on_embeddings: SpeakerEmbeddingCallback = None,
    ) -> Dict[int, SpeakerEmbeddingResult]:
        """批量提取声纹向量，返回 {chunk_index: embedding}。"""
        raise NotImplementedError


class SpeakerEmbeddingService(SpeakerEmbeddingProvider):
    """
    SpeechBrain 声纹提取服务

    说明：
    - 使用 Provider 接口作为策略模式扩展点，便于后续替换模型实现。
    - 默认 CPU 推理，若语言检测为 precise 且设备为 auto，优先 GPU。
    """

    DEFAULT_MODEL_ID = "speaker-ecapa-voxceleb"
    TARGET_DURATION_SECONDS = 4.0
    EMBEDDING_DIM = 192

    BATCH_SIZE_CPU = 32
    BATCH_SIZE_GPU = 64

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model_id = model_id
        self.logger = logger or logging.getLogger(__name__)
        self._classifiers: Dict[str, Any] = {}

    def extract_embeddings(
        self,
        chunks: Sequence[AudioChunk],
        device: str = "auto",
        mode: Optional[str] = None,
        on_embeddings: SpeakerEmbeddingCallback = None,
    ) -> Dict[int, SpeakerEmbeddingResult]:
        """批量提取声纹向量。"""
        if not chunks:
            return {}

        resolved_device, cpu_threads, batch_size = self._resolve_runtime(mode, device)
        self._apply_cpu_threads(resolved_device, cpu_threads)

        results: Dict[int, SpeakerEmbeddingResult] = {}
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start:start + batch_size]
            embeddings = self._encode_batch(batch_chunks, resolved_device)
            self._emit_embeddings(on_embeddings, batch_chunks, embeddings)
            for chunk, embedding in zip(batch_chunks, embeddings):
                results[chunk.index] = embedding

        return results

    def build_speaker_cache_metadata(
        self,
        mode: Optional[str],
        device: str,
    ) -> Dict[str, Any]:
        """构建 Speaker 缓存元数据，确保命中一致性。"""
        resolved_device, _, batch_size = self._resolve_runtime(mode, device)

        model_repo = None
        model_hash = None
        try:
            from app.services.model_manager_v2 import get_model_manager_v2

            spec = get_model_manager_v2().registry.get(self.model_id)
            model_repo = spec.source.repo_id
            model_hash = spec.source.hash
        except Exception as exc:
            self.logger.debug("Speaker 缓存元数据获取模型信息失败: %s", exc)

        torch_version = None
        speechbrain_version = None
        try:
            import torch

            torch_version = getattr(torch, "__version__", None)
        except Exception:
            pass
        try:
            import speechbrain

            speechbrain_version = getattr(speechbrain, "__version__", None)
        except Exception:
            pass

        return {
            "schema_version": "1.0",
            "model_id": self.model_id,
            "model_repo": model_repo,
            "model_hash": model_hash,
            "embedding_dim": self.EMBEDDING_DIM,
            "target_duration_seconds": self.TARGET_DURATION_SECONDS,
            "batch_size": batch_size,
            "resolved_device": resolved_device,
            "torch_version": torch_version,
            "speechbrain_version": speechbrain_version,
        }

    @staticmethod
    def _emit_embeddings(
        on_embeddings: SpeakerEmbeddingCallback,
        chunks: Sequence[AudioChunk],
        embeddings: Sequence[SpeakerEmbeddingResult],
    ) -> None:
        if on_embeddings:
            on_embeddings(chunks, embeddings)

    def _encode_batch(
        self,
        chunks: Sequence[AudioChunk],
        device: str,
    ) -> List[SpeakerEmbeddingResult]:
        if not chunks:
            return []

        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        classifier = self._get_classifier(EncoderClassifier, device)
        batch = self._prepare_batch(chunks)
        if device != "cpu":
            batch = batch.to(device)

        with torch.inference_mode():
            embeddings = classifier.encode_batch(batch)

        vectors = self._normalize_embeddings(embeddings)
        results: List[SpeakerEmbeddingResult] = []
        for vector in vectors:
            results.append(SpeakerEmbeddingResult(embedding=vector))
        return results

    def _get_classifier(self, classifier_cls: Any, device: str) -> Any:
        if device in self._classifiers:
            return self._classifiers[device]

        from app.services.model_manager_v2 import get_model_manager_v2

        manager = get_model_manager_v2()
        model_dir = Path(manager.ensure_available(self.model_id))
        try:
            classifier = classifier_cls.from_hparams(
                source=model_dir.as_posix(),
                savedir=model_dir.as_posix(),
                run_opts={"device": device},
            )
        except Exception as exc:
            raise RuntimeError(f"Speaker 模型加载失败: {model_dir} -> {exc}") from exc

        self._classifiers[device] = classifier
        return classifier

    def _prepare_batch(self, chunks: Sequence[AudioChunk]) -> Any:
        import torch

        arrays = [
            self._center_crop_and_pad(chunk.audio, chunk.sample_rate)
            for chunk in chunks
        ]
        batch = np.stack(arrays, axis=0).astype(np.float32, copy=False)
        return torch.from_numpy(batch)

    def _normalize_embeddings(self, embeddings: Any) -> List[List[float]]:
        import torch
        import torch.nn.functional as F

        if not isinstance(embeddings, torch.Tensor):
            raise RuntimeError("Speaker embedding 输出格式异常")

        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        if embeddings.dim() == 3 and embeddings.shape[1] == 1:
            embeddings = embeddings.squeeze(1)
        if embeddings.dim() != 2:
            raise RuntimeError(f"Speaker embedding 维度异常: {embeddings.shape}")

        embeddings = embeddings.float()
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        if embeddings.shape[-1] != self.EMBEDDING_DIM:
            self.logger.warning(
                "Speaker embedding 维度异常: expect=%s, got=%s",
                self.EMBEDDING_DIM,
                embeddings.shape[-1],
            )

        vectors = embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
        return [vector.tolist() for vector in vectors]

    def _resolve_runtime(
        self,
        mode: Optional[str],
        device_preference: str,
    ) -> Tuple[str, Optional[int], int]:
        cpu_threads: Optional[int] = None
        runtime_device: Optional[str] = None
        hardware_info = None
        max_vram_mb: Optional[int] = None

        try:
            from app.services.model_manager_v2 import get_model_manager_v2
            from app.services.model_runtime_config_service import get_model_runtime_config_service

            manager = get_model_manager_v2()
            spec = manager.registry.get(self.model_id)
            runtime = get_model_runtime_config_service().get_effective_model(spec).get("effective", {})
            cpu_threads = runtime.get("cpu_threads")
            runtime_device = runtime.get("device")
        except Exception as exc:
            self.logger.warning("Speaker 运行参数回退: %s", exc)

        try:
            from app.services.hardware_profile_service import get_hardware_profile_provider

            hardware_info = get_hardware_profile_provider().get_hardware_info()
        except Exception as exc:
            self.logger.debug("Speaker 硬件信息获取失败: %s", exc)

        if cpu_threads is None and hardware_info:
            base_threads = hardware_info.cpu_threads or hardware_info.cpu_cores or 1
            cpu_threads = max(1, int(base_threads) - 4)

        if hardware_info and hardware_info.gpu_memory_mb:
            max_vram_mb = max(hardware_info.gpu_memory_mb)

        if device_preference != "auto":
            resolved_device = device_preference
        elif runtime_device and runtime_device != "auto":
            resolved_device = runtime_device
        elif mode == "precise" and self._cuda_available():
            resolved_device = "cuda"
        else:
            resolved_device = "cpu"

        if resolved_device.startswith("cuda") and not self._cuda_available():
            self.logger.warning("Speaker 选择 GPU 失败，回退 CPU")
            resolved_device = "cpu"

        batch_size = self._resolve_batch_size(resolved_device, cpu_threads, max_vram_mb)

        return resolved_device, cpu_threads, batch_size

    @classmethod
    def _resolve_batch_size(
        cls,
        device: str,
        cpu_threads: Optional[int],
        max_vram_mb: Optional[int],
    ) -> int:
        if device == "cpu":
            batch_size = cls.BATCH_SIZE_CPU
            if cpu_threads and cpu_threads < 4:
                batch_size = max(8, int(cpu_threads) * 4)
            return batch_size

        if max_vram_mb is None:
            return cls.BATCH_SIZE_GPU
        if max_vram_mb < 2000:
            return 16
        if max_vram_mb < 4000:
            return 32
        return cls.BATCH_SIZE_GPU

    def _apply_cpu_threads(self, device: str, cpu_threads: Optional[int]) -> None:
        if device != "cpu" or not cpu_threads:
            return
        try:
            import torch

            threads = max(1, int(cpu_threads))
            torch.set_num_threads(threads)
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            os.environ["OMP_NUM_THREADS"] = str(threads)
            os.environ["MKL_NUM_THREADS"] = str(threads)
        except Exception as exc:
            self.logger.warning("Speaker 线程配置失败: %s", exc)

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    @classmethod
    def _center_crop_and_pad(
        cls,
        audio: Optional[np.ndarray],
        sample_rate: int,
    ) -> np.ndarray:
        if audio is None or audio.size == 0:
            sample_rate = sample_rate or 16000
            target = int(cls.TARGET_DURATION_SECONDS * sample_rate)
            return np.zeros(target, dtype=np.float32)

        sr = sample_rate or 16000
        target_length = int(cls.TARGET_DURATION_SECONDS * sr)
        data = np.asarray(audio, dtype=np.float32)

        if data.shape[0] > target_length:
            center = data.shape[0] // 2
            half = target_length // 2
            start = max(0, center - half)
            end = start + target_length
            data = data[start:end]
        elif data.shape[0] < target_length:
            pad = target_length - data.shape[0]
            data = np.pad(data, (0, pad), mode="constant")

        return data


_speaker_embedding_service: Optional[SpeakerEmbeddingService] = None


def get_speaker_embedding_service(
    logger: Optional[logging.Logger] = None,
) -> SpeakerEmbeddingService:
    """获取 SpeakerEmbeddingService 单例。"""
    global _speaker_embedding_service
    if _speaker_embedding_service is None:
        _speaker_embedding_service = SpeakerEmbeddingService(logger=logger)
    elif logger is not None:
        _speaker_embedding_service.logger = logger
    return _speaker_embedding_service
