"""
ONNX 模型加载器
根据设备选择执行提供者，支持自动选择第一个 .onnx 文件。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.core.asr.loader_base import LoadPlan, ModelLoader
from app.core.asr.model_spec import ModelSpec


class OnnxLoader(ModelLoader):
    """使用 onnxruntime 的加载器。"""

    def _select_model_file(self, spec: ModelSpec, plan: LoadPlan) -> Path:
        """选择可用的 .onnx 文件，便于测试独立于 ort。"""
        base = Path(plan.local_path)
        # 优先 features 指定
        filename = spec.features.get("model_file") if spec.features else None
        if filename:
            candidate = base / filename
            if candidate.exists():
                return candidate
        # 其次 source.files 中的 onnx
        for fname in spec.source.files:
            if fname.endswith(".onnx"):
                candidate = base / fname
                if candidate.exists():
                    return candidate
        # 最后扫描目录
        for f in base.glob("*.onnx"):
            return f
        raise FileNotFoundError(f"未找到 ONNX 模型文件: {base}")

    def load(self, spec: ModelSpec, plan: LoadPlan):
        import onnxruntime as ort  # 延迟导入，避免无依赖环境阻塞

        model_file = self._select_model_file(spec, plan)
        providers = self._select_providers(plan.device)

        sess_options = ort.SessionOptions()
        # 依据资源配置优化线程数
        intra_threads = spec.resources.get("cpu_threads")
        if intra_threads:
            sess_options.intra_op_num_threads = intra_threads
            sess_options.inter_op_num_threads = 1

        session = ort.InferenceSession(
            model_file.as_posix(),
            sess_options=sess_options,
            providers=providers,
        )
        return session

    def unload(self, handle) -> None:
        """onnxruntime Session 依赖 GC，无需额外操作。"""
        return

    @staticmethod
    def _select_providers(device: str):
        device_lower = device.lower()
        if device_lower == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
