"""
SpeechAnalysisService - LangID 语言检测服务

V3.2.0+dev.20260127.02: 实现 LangID 三模式检测与中心裁剪批推理。
V3.2.0+dev.20260127.03: 支持 Logit Bias + 动态白名单。
V3.2.0+dev.20260127.04: 解析 LangID 标签前缀代码并用于白名单匹配。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.services.audio.chunk_engine import AudioChunk

LangIDMode = str


@dataclass(frozen=True)
class LangIDPrediction:
    """LangID 预测结果（仅保留关键字段）"""
    language: str
    confidence: float


@dataclass(frozen=True)
class LangIDGroupDecision:
    """分组决策信息，用于边界微扫判断"""
    language: str
    mean_confidence: float
    decision: str  # broadcast | full_scan


class SpeechAnalysisService:
    """
    LangID 服务（策略模式：fast/balanced/precise 按模式分派）

    说明：
    - 采用策略模式是为了隔离三种推理路径，便于后续扩展与性能调优。
    - 默认 CPU 推理，精准模式优先 GPU。
    """

    DEFAULT_MODEL_ID = "langid-voxlingua"
    DEFAULT_LANGUAGE_WHITELIST = ("zh", "ja", "en")
    DEFAULT_LOGIT_BIAS_SCORE = 2.5

    TARGET_DURATION_SECONDS = 4.0
    GROUP_WINDOW_SECONDS = 60.0

    FAST_SAMPLE_MIN = 8
    FAST_SAMPLE_MAX = 15
    FAST_SAMPLE_DIVISOR = 12
    FAST_CONSENSUS_RATIO = 0.9
    FAST_CONFIDENCE_THRESHOLD = 0.8

    PROBE_RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)
    BOUNDARY_SCAN_RATIO = 0.25
    BOUNDARY_CONFIDENCE_THRESHOLD = 0.75

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

    def detect_languages(
        self,
        chunks: Sequence[AudioChunk],
        mode: LangIDMode = "balanced",
        device: str = "auto",
        whitelist: Optional[Sequence[str]] = None,
        logit_bias_score: Optional[float] = None,
    ) -> Dict[int, LangIDPrediction]:
        """检测所有 Chunk 的语言标签。"""
        if not chunks:
            return {}

        resolved_mode = mode if mode in ("fast", "balanced", "precise") else "balanced"
        resolved_device, cpu_threads = self._resolve_runtime(resolved_mode, device)
        self._apply_cpu_threads(resolved_device, cpu_threads)
        normalized_whitelist = self._normalize_language_whitelist(whitelist)
        bias_score = self._normalize_logit_bias_score(logit_bias_score)

        if resolved_mode == "fast":
            return self._detect_fast(chunks, resolved_device, normalized_whitelist, bias_score)
        if resolved_mode == "precise":
            return self._detect_precise(chunks, resolved_device, normalized_whitelist, bias_score)
        return self._detect_balanced(chunks, resolved_device, normalized_whitelist, bias_score)

    def _detect_fast(
        self,
        chunks: Sequence[AudioChunk],
        device: str,
        whitelist: Optional[Sequence[str]],
        logit_bias_score: float,
    ) -> Dict[int, LangIDPrediction]:
        sample_indices = self._select_fast_sample_indices(len(chunks))
        if len(sample_indices) < self.FAST_SAMPLE_MIN:
            self.logger.info("LangID fast 模式样本不足，降级为 balanced")
            return self._detect_balanced(chunks, device, whitelist, logit_bias_score)

        sample_chunks = [chunks[i] for i in sample_indices]
        predictions = self._classify_chunks_in_batches(sample_chunks, device, whitelist, logit_bias_score)
        if not predictions:
            raise RuntimeError("LangID fast 模式推理结果为空")

        dominant, ratio, mean_confidence = self._calculate_consensus(predictions)
        if ratio >= self.FAST_CONSENSUS_RATIO and mean_confidence >= self.FAST_CONFIDENCE_THRESHOLD:
            return {
                chunk.index: LangIDPrediction(dominant, mean_confidence)
                for chunk in chunks
            }

        self.logger.info("LangID fast 模式一致性不足，降级为 balanced")
        return self._detect_balanced(chunks, device, whitelist, logit_bias_score)

    def _detect_balanced(
        self,
        chunks: Sequence[AudioChunk],
        device: str,
        whitelist: Optional[Sequence[str]],
        logit_bias_score: float,
    ) -> Dict[int, LangIDPrediction]:
        groups = self._group_chunks_by_time(chunks, self.GROUP_WINDOW_SECONDS)
        results: Dict[int, LangIDPrediction] = {}
        decisions: List[Optional[LangIDGroupDecision]] = [None] * len(groups)

        probes: List[AudioChunk] = []
        probe_map: List[Tuple[int, int]] = []

        for group_index, group in enumerate(groups):
            probe_chunks = self._select_group_probes(group)
            if len(probe_chunks) < 3:
                group_predictions = self._classify_chunks_in_batches(group, device, whitelist, logit_bias_score)
                for chunk, prediction in zip(group, group_predictions):
                    results[chunk.index] = prediction
                decisions[group_index] = self._build_group_decision(group_predictions, "full_scan")
                continue

            for chunk in probe_chunks:
                probes.append(chunk)
                probe_map.append((group_index, chunk.index))

        probe_predictions = self._classify_chunks_in_batches(
            probes,
            device,
            whitelist,
            logit_bias_score,
        ) if probes else []
        group_probe_results: Dict[int, List[LangIDPrediction]] = {}
        for (group_index, _), prediction in zip(probe_map, probe_predictions):
            group_probe_results.setdefault(group_index, []).append(prediction)

        for group_index, group in enumerate(groups):
            if decisions[group_index] is not None:
                continue

            probe_results = group_probe_results.get(group_index, [])
            if not probe_results:
                group_predictions = self._classify_chunks_in_batches(group, device, whitelist, logit_bias_score)
                for chunk, prediction in zip(group, group_predictions):
                    results[chunk.index] = prediction
                decisions[group_index] = self._build_group_decision(group_predictions, "full_scan")
                continue

            dominant, ratio, mean_confidence = self._calculate_consensus(probe_results)
            if ratio == 1.0:
                for chunk in group:
                    results[chunk.index] = LangIDPrediction(dominant, mean_confidence)
                decisions[group_index] = LangIDGroupDecision(
                    language=dominant,
                    mean_confidence=mean_confidence,
                    decision="broadcast",
                )
            else:
                group_predictions = self._classify_chunks_in_batches(group, device, whitelist, logit_bias_score)
                for chunk, prediction in zip(group, group_predictions):
                    results[chunk.index] = prediction
                decisions[group_index] = self._build_group_decision(group_predictions, "full_scan")

        self._refine_boundaries(
            groups,
            decisions,
            results,
            device,
            whitelist,
            logit_bias_score,
        )
        return results

    def _detect_precise(
        self,
        chunks: Sequence[AudioChunk],
        device: str,
        whitelist: Optional[Sequence[str]],
        logit_bias_score: float,
    ) -> Dict[int, LangIDPrediction]:
        predictions = self._classify_chunks_in_batches(chunks, device, whitelist, logit_bias_score)
        return {
            chunk.index: prediction
            for chunk, prediction in zip(chunks, predictions)
        }

    def _refine_boundaries(
        self,
        groups: Sequence[Sequence[AudioChunk]],
        decisions: Sequence[Optional[LangIDGroupDecision]],
        results: Dict[int, LangIDPrediction],
        device: str,
        whitelist: Optional[Sequence[str]],
        logit_bias_score: float,
    ) -> None:
        for index in range(len(groups) - 1):
            left = decisions[index]
            right = decisions[index + 1]
            if not left or not right:
                continue
            if not self._should_micro_scan(left, right):
                continue

            tail_chunks = self._select_boundary_chunks(groups[index], tail=True)
            head_chunks = self._select_boundary_chunks(groups[index + 1], tail=False)
            boundary_chunks = tail_chunks + head_chunks
            if not boundary_chunks:
                continue

            boundary_predictions = self._classify_chunks_in_batches(
                boundary_chunks,
                device,
                whitelist,
                logit_bias_score,
            )
            for chunk, prediction in zip(boundary_chunks, boundary_predictions):
                results[chunk.index] = prediction

    def _classify_chunks_in_batches(
        self,
        chunks: Sequence[AudioChunk],
        device: str,
        whitelist: Optional[Sequence[str]],
        logit_bias_score: float,
    ) -> List[LangIDPrediction]:
        if not chunks:
            return []

        batch_size = self.BATCH_SIZE_GPU if device != "cpu" else self.BATCH_SIZE_CPU
        predictions: List[LangIDPrediction] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            predictions.extend(self._classify_batch(batch, device, whitelist, logit_bias_score))
        return predictions

    def _classify_batch(
        self,
        chunks: Sequence[AudioChunk],
        device: str,
        whitelist: Optional[Sequence[str]],
        logit_bias_score: float,
    ) -> List[LangIDPrediction]:
        if not chunks:
            return []

        import torch
        from speechbrain.inference import EncoderClassifier

        classifier = self._get_classifier(EncoderClassifier, device)
        batch = self._prepare_batch(chunks)
        if device != "cpu":
            batch = batch.to(device)

        # V3.2.0+dev.20260127.10: 添加调试日志
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"_classify_batch: 调用前 batch 形状={batch.shape}, 范围 min={batch.min().item():.6f}, max={batch.max().item():.6f}")

        with torch.inference_mode():
            result = classifier.classify_batch(batch)

        # V3.2.0+dev.20260127.10: 检查 result[0] 的内容
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"_classify_batch: result 类型={type(result)}, 长度={len(result)}")
            self.logger.debug(f"_classify_batch: result[0] 形状={result[0].shape}, 范围 min={result[0].min().item():.6f}, max={result[0].max().item():.6f}")

        return self._parse_predictions(
            result,
            classifier,
            len(chunks),
            whitelist=whitelist,
            logit_bias_score=logit_bias_score,
        )

    def _get_classifier(self, classifier_cls: Any, device: str) -> Any:
        if device in self._classifiers:
            return self._classifiers[device]

        from app.services.model_manager_v2 import get_model_manager_v2

        manager = get_model_manager_v2()
        model_dir = Path(manager.ensure_available(self.model_id))
        try:
            # V3.2.0+dev.20260127.11: 修复模型加载，添加 savedir 参数
            classifier = classifier_cls.from_hparams(
                source=model_dir.as_posix(),
                savedir=model_dir.as_posix(),  # 添加 savedir 参数
                run_opts={"device": device},
            )
        except Exception as exc:
            raise RuntimeError(f"LangID 模型加载失败: {model_dir} -> {exc}") from exc

        self._classifiers[device] = classifier
        return classifier

    def _prepare_batch(self, chunks: Sequence[AudioChunk]) -> Any:
        import torch

        arrays = [
            self._center_crop_and_pad(chunk.audio, chunk.sample_rate)
            for chunk in chunks
        ]
        batch = np.stack(arrays, axis=0).astype(np.float32, copy=False)

        # V3.2.0+dev.20260127.09: 添加调试日志
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"_prepare_batch: batch 形状={batch.shape}, dtype={batch.dtype}")
            self.logger.debug(f"_prepare_batch: batch 范围 min={batch.min():.6f}, max={batch.max():.6f}")

        return torch.from_numpy(batch)

    def _parse_predictions(
        self,
        result: Any,
        classifier: Any,
        batch_size: int,
        whitelist: Optional[Sequence[str]] = None,
        logit_bias_score: Optional[float] = None,
    ) -> List[LangIDPrediction]:
        import torch

        if not isinstance(result, tuple) or not result:
            raise RuntimeError("LangID 推理返回结果为空")

        probs = result[0]
        if not isinstance(probs, torch.Tensor):
            raise RuntimeError("LangID 推理输出格式异常")

        if probs.dim() == 1:
            probs = probs.unsqueeze(0)

        # V3.2.0+dev.20260127.08: 检查并处理异常值
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"LangID probs 形状: {probs.shape}, dtype: {probs.dtype}")
            self.logger.debug(f"LangID probs 范围: min={probs.min().item():.4f}, max={probs.max().item():.4f}")
            # 检查是否有 inf 或 nan
            if torch.isinf(probs).any() or torch.isnan(probs).any():
                self.logger.warning("LangID probs 包含 inf 或 nan 值")

        normalized_whitelist = self._normalize_language_whitelist(whitelist)
        whitelist_set = set(normalized_whitelist) if normalized_whitelist else set()
        bias_score = self._normalize_logit_bias_score(logit_bias_score)
        label_to_index = self._build_label_to_index_map(classifier)

        # V3.2.0+dev.20260127.05: 添加调试日志
        self.logger.debug(f"LangID 白名单: {normalized_whitelist}")
        self.logger.debug(f"LangID 标签映射表: {label_to_index}")

        bias_indices = [
            label_to_index[label]
            for label in (normalized_whitelist or [])
            if label in label_to_index
        ]

        self.logger.debug(f"LangID bias 索引: {bias_indices}, bias 分数: {bias_score}")
        use_bias = bool(bias_indices) and bias_score != 0.0

        if use_bias:
            # V3.2.0+dev.20260127.06: 修复 logit bias 的数学实现
            # probs 是 log probabilities，需要转换为 unnormalized logits 再加 bias
            # 将 log probs 视为 unnormalized logits，加 bias 后重新 softmax
            logits = probs.clone()

            # V3.2.0+dev.20260127.07: 添加调试日志查看 bias 前后的值
            if self.logger.isEnabledFor(logging.DEBUG):
                original_max_idx = torch.argmax(logits, dim=-1)
                original_max_val = logits[0, original_max_idx[0]].item()
                self.logger.debug(f"LangID bias 前: 最大值索引={original_max_idx[0].item()}, 值={original_max_val:.4f}")
                for idx in bias_indices[:3]:  # 只显示前3个
                    self.logger.debug(f"  白名单索引 {idx}: 原始值={logits[0, idx].item():.4f}")

            for index in bias_indices:
                logits[:, index] = logits[:, index] + bias_score

            # V3.2.0+dev.20260127.07: 添加调试日志查看 bias 后的值
            if self.logger.isEnabledFor(logging.DEBUG):
                biased_max_idx = torch.argmax(logits, dim=-1)
                biased_max_val = logits[0, biased_max_idx[0]].item()
                self.logger.debug(f"LangID bias 后: 最大值索引={biased_max_idx[0].item()}, 值={biased_max_val:.4f}")
                for idx in bias_indices[:3]:  # 只显示前3个
                    self.logger.debug(f"  白名单索引 {idx}: bias后值={logits[0, idx].item():.4f}")

            # 重新计算 softmax 和置信度
            indices = torch.argmax(logits, dim=-1)
            confidences = torch.softmax(logits, dim=-1).max(dim=-1).values.clamp(0.0, 1.0)
            labels = self._extract_labels(
                result,
                classifier,
                indices,
                batch_size,
                allow_result_labels=False,
            )
        else:
            indices = torch.argmax(probs, dim=-1)
            scores = result[1] if len(result) >= 2 else None
            if isinstance(scores, torch.Tensor):
                if scores.dim() == 0:
                    scores = scores.unsqueeze(0)
                if scores.shape[0] == probs.shape[0]:
                    confidences = torch.exp(scores).clamp(0.0, 1.0)
                else:
                    confidences = torch.exp(probs).max(dim=-1).values.clamp(0.0, 1.0)
            else:
                confidences = torch.exp(probs).max(dim=-1).values.clamp(0.0, 1.0)
            labels = self._extract_labels(
                result,
                classifier,
                indices,
                batch_size,
                allow_result_labels=True,
            )

        predictions: List[LangIDPrediction] = []
        for i in range(batch_size):
            label = labels[i] if labels and i < len(labels) else "auto"
            language = self._extract_label_code(label)

            # V3.2.0+dev.20260127.05: 添加详细调试日志
            self.logger.debug(
                f"LangID 样本 {i}: 原始标签={label}, 提取语言={language}, "
                f"置信度={confidences[i].item():.4f}, 白名单={whitelist_set}"
            )

            if whitelist_set and language not in whitelist_set:
                self.logger.debug(f"LangID 样本 {i}: 语言 '{language}' 不在白名单中，回退到 auto")
                predictions.append(LangIDPrediction(language="auto", confidence=0.0))
                continue
            predictions.append(
                LangIDPrediction(
                    language=language,
                    confidence=float(confidences[i].item()),
                )
            )
        return predictions

    def _extract_labels(
        self,
        result: Tuple[Any, ...],
        classifier: Any,
        indices: Any,
        batch_size: int,
        allow_result_labels: bool = True,
    ) -> Optional[List[str]]:
        labels: Any = None
        if allow_result_labels and len(result) >= 4:
            labels = result[3]
        if labels is None and allow_result_labels and len(result) >= 3:
            labels = self._decode_indices(classifier, result[2])
        if labels is None:
            labels = self._decode_indices(classifier, indices)
        return self._normalize_labels(labels, batch_size)

    @classmethod
    def _normalize_language_whitelist(
        cls,
        whitelist: Optional[Sequence[str]],
    ) -> Optional[List[str]]:
        if whitelist is None:
            return [lang.lower() for lang in cls.DEFAULT_LANGUAGE_WHITELIST]
        normalized: List[str] = []
        for item in whitelist:
            if not item:
                continue
            text = str(item).strip().lower()
            if text:
                normalized.append(text)
        if not normalized:
            return []
        return list(dict.fromkeys(normalized))

    @classmethod
    def _normalize_logit_bias_score(cls, logit_bias_score: Optional[float]) -> float:
        if logit_bias_score is None:
            return cls.DEFAULT_LOGIT_BIAS_SCORE
        try:
            return float(logit_bias_score)
        except (TypeError, ValueError):
            return cls.DEFAULT_LOGIT_BIAS_SCORE

    @staticmethod
    def _extract_label_code(label: Any) -> str:
        if label is None:
            return "auto"
        text = str(label).strip().lower()
        if not text:
            return "auto"
        if ":" in text:
            code = text.split(":", 1)[0].strip()
            if code:
                return code
        return text

    @staticmethod
    def _build_label_to_index_map(classifier: Any) -> Dict[str, int]:
        encoder = getattr(getattr(classifier, "hparams", None), "label_encoder", None)
        if not encoder:
            return {}
        lab2ind = getattr(encoder, "lab2ind", None)
        if isinstance(lab2ind, dict):
            mapping: Dict[str, int] = {}
            for label, index in lab2ind.items():
                text = str(label).strip().lower()
                if not text:
                    continue
                mapping.setdefault(text, int(index))
                code = SpeechAnalysisService._extract_label_code(text)
                if code and code != "auto":
                    mapping.setdefault(code, int(index))
            return mapping
        ind2lab = getattr(encoder, "ind2lab", None)
        if isinstance(ind2lab, dict):
            mapping = {}
            for index, label in ind2lab.items():
                text = str(label).strip().lower()
                if not text:
                    continue
                mapping.setdefault(text, int(index))
                code = SpeechAnalysisService._extract_label_code(text)
                if code and code != "auto":
                    mapping.setdefault(code, int(index))
            return mapping
        if isinstance(ind2lab, (list, tuple)):
            mapping = {}
            for index, label in enumerate(ind2lab):
                text = str(label).strip().lower()
                if not text:
                    continue
                mapping.setdefault(text, int(index))
                code = SpeechAnalysisService._extract_label_code(text)
                if code and code != "auto":
                    mapping.setdefault(code, int(index))
            return mapping
        return {}

    def _decode_indices(self, classifier: Any, indices: Any) -> Optional[Any]:
        encoder = getattr(getattr(classifier, "hparams", None), "label_encoder", None)
        if not encoder or not hasattr(encoder, "decode_torch"):
            return None
        try:
            return encoder.decode_torch(indices)
        except Exception as exc:
            self.logger.warning("LangID 标签解码失败: %s", exc)
            return None

    @staticmethod
    def _normalize_labels(labels: Any, batch_size: int) -> Optional[List[str]]:
        if labels is None:
            return None
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        if isinstance(labels, tuple):
            labels = list(labels)
        if isinstance(labels, list):
            if len(labels) == batch_size and all(isinstance(item, str) for item in labels):
                return [SpeechAnalysisService._extract_label_code(item) for item in labels]
            if len(labels) == batch_size and all(isinstance(item, (list, tuple)) for item in labels):
                return [
                    SpeechAnalysisService._extract_label_code(item[0]) if item else "auto"
                    for item in labels
                ]
        return None

    def _resolve_runtime(
        self,
        mode: LangIDMode,
        device_preference: str,
    ) -> Tuple[str, Optional[int]]:
        cpu_threads: Optional[int] = None

        try:
            from app.services.model_manager_v2 import get_model_manager_v2
            from app.services.model_runtime_config_service import get_model_runtime_config_service

            manager = get_model_manager_v2()
            spec = manager.registry.get(self.model_id)
            runtime = get_model_runtime_config_service().get_effective_model(spec).get("effective", {})
            cpu_threads = runtime.get("cpu_threads")
        except Exception as exc:
            self.logger.warning("LangID 运行参数回退: %s", exc)

        if device_preference != "auto":
            resolved_device = device_preference
        elif mode == "precise" and self._cuda_available():
            resolved_device = "cuda"
        else:
            resolved_device = "cpu"

        if resolved_device == "cuda" and not self._cuda_available():
            self.logger.warning("LangID 选择 GPU 失败，回退 CPU")
            resolved_device = "cpu"

        return resolved_device, cpu_threads

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
                # interop 线程只允许设置一次，失败可接受
                pass
            os.environ["OMP_NUM_THREADS"] = str(threads)
            os.environ["MKL_NUM_THREADS"] = str(threads)
        except Exception as exc:
            self.logger.warning("LangID 线程配置失败: %s", exc)

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    @classmethod
    def _select_fast_sample_indices(cls, total: int) -> List[int]:
        if total <= 0:
            return []
        step = max(1, total // cls.FAST_SAMPLE_DIVISOR)
        indices = list(range(0, total, step))
        return indices[: cls.FAST_SAMPLE_MAX]

    @staticmethod
    def _group_chunks_by_time(
        chunks: Sequence[AudioChunk],
        window_seconds: float,
    ) -> List[List[AudioChunk]]:
        groups: List[List[AudioChunk]] = []
        if not chunks:
            return groups

        current: List[AudioChunk] = []
        window_start = chunks[0].start
        for chunk in chunks:
            if current and (chunk.start - window_start) >= window_seconds:
                groups.append(current)
                current = []
                window_start = chunk.start
            current.append(chunk)

        if current:
            groups.append(current)
        return groups

    @classmethod
    def _select_group_probes(cls, group: Sequence[AudioChunk]) -> List[AudioChunk]:
        if not group:
            return []

        total_duration = sum(chunk.duration for chunk in group)
        if total_duration <= 0:
            return [group[0]]

        probes: List[AudioChunk] = []
        seen: set[int] = set()

        for ratio in cls.PROBE_RATIOS:
            target = total_duration * ratio
            cumulative = 0.0
            selected = group[-1]
            for chunk in group:
                cumulative += chunk.duration
                if cumulative >= target:
                    selected = chunk
                    break
            if selected.index not in seen:
                probes.append(selected)
                seen.add(selected.index)

        return probes

    @classmethod
    def _select_boundary_chunks(
        cls,
        group: Sequence[AudioChunk],
        tail: bool,
    ) -> List[AudioChunk]:
        if not group:
            return []
        count = max(1, int(len(group) * cls.BOUNDARY_SCAN_RATIO))
        return list(group[-count:] if tail else group[:count])

    @classmethod
    def _should_micro_scan(
        cls,
        left: LangIDGroupDecision,
        right: LangIDGroupDecision,
    ) -> bool:
        if left.decision != "broadcast" or right.decision != "broadcast":
            return False
        if left.language == right.language:
            return False
        return (
            left.mean_confidence < cls.BOUNDARY_CONFIDENCE_THRESHOLD
            or right.mean_confidence < cls.BOUNDARY_CONFIDENCE_THRESHOLD
        )

    @staticmethod
    def _calculate_consensus(
        predictions: Sequence[LangIDPrediction],
    ) -> Tuple[str, float, float]:
        if not predictions:
            return "auto", 0.0, 0.0

        from collections import Counter

        counts = Counter(pred.language for pred in predictions)
        dominant, count = counts.most_common(1)[0]
        ratio = count / len(predictions)
        mean_confidence = sum(pred.confidence for pred in predictions) / len(predictions)
        return dominant, ratio, mean_confidence

    @staticmethod
    def _build_group_decision(
        predictions: Sequence[LangIDPrediction],
        decision: str,
    ) -> LangIDGroupDecision:
        dominant, _, mean_confidence = SpeechAnalysisService._calculate_consensus(predictions)
        return LangIDGroupDecision(
            language=dominant,
            mean_confidence=mean_confidence,
            decision=decision,
        )

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


_speech_analysis_service: Optional[SpeechAnalysisService] = None


def get_speech_analysis_service(
    logger: Optional[logging.Logger] = None,
) -> SpeechAnalysisService:
    """获取 LangID 服务单例。"""
    global _speech_analysis_service
    if _speech_analysis_service is None:
        _speech_analysis_service = SpeechAnalysisService(logger=logger)
    elif logger is not None:
        _speech_analysis_service.logger = logger
    return _speech_analysis_service
