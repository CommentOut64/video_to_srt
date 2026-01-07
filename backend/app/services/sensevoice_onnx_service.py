"""
SenseVoice ONNX 推理服务（纯 ONNX Runtime 实现）

核心特性：
1. 移除 funasr-onnx 依赖，使用纯 ONNX Runtime
2. 自研 CTC 解码器，从 Logits 提取字级时间戳
3. 支持多语言语音识别、情感识别和事件检测

模型来源：
- ModelScope: https://www.modelscope.cn/models/iic/SenseVoiceSmall
- HuggingFace: https://huggingface.co/FunAudioLLM/SenseVoiceSmall

依赖：
- onnxruntime / onnxruntime-gpu
- numpy
- librosa (用于音频预处理)
"""
import os
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Union, Tuple
import threading
import json

logger = logging.getLogger(__name__)


class CTCDecoder:
    """
    CTC 解码器（自研）

    从 SenseVoice 的 Logits 输出中提取：
    1. 文本序列
    2. 字级时间戳
    3. 置信度
    """

    def __init__(self, vocab: Dict[int, str], blank_id: int = 0):
        """
        初始化 CTC 解码器

        Args:
            vocab: 词汇表 {token_id: token_str}
            blank_id: CTC blank token ID
        """
        self.vocab = vocab
        self.blank_id = blank_id
        self.logger = logging.getLogger(__name__)

    def decode(
        self,
        logits: np.ndarray,
        time_stride: float = 0.06
    ) -> Tuple[str, List[Dict], float]:
        """
        CTC 解码（贪心算法 + 字级时间戳提取）

        Args:
            logits: 模型输出的 logits，形状 [time_steps, vocab_size]
            time_stride: 时间步长（秒），SenseVoice 默认 60ms

        Returns:
            Tuple[str, List[Dict], float]:
                - text: 解码后的文本
                - word_timestamps: 字级时间戳列表
                - confidence: 平均置信度
        """
        # 1. Softmax 转换为概率
        probs = self._softmax(logits)

        # 2. 贪心解码：选择每个时间步的最大概率 token
        token_ids = np.argmax(probs, axis=-1)  # [time_steps]
        token_probs = np.max(probs, axis=-1)   # [time_steps]

        # 3. CTC 去重 + 提取字级时间戳
        text_chars = []
        word_timestamps = []
        prev_token = self.blank_id
        char_start_time = None
        char_probs = []

        for t, (token_id, prob) in enumerate(zip(token_ids, token_probs)):
            current_time = t * time_stride

            if token_id == self.blank_id:
                # 遇到 blank，结束当前字符
                if char_start_time is not None:
                    # 保存字符和时间戳
                    char_end_time = current_time
                    avg_prob = np.mean(char_probs) if char_probs else prob

                    word_timestamps.append({
                        "word": self.vocab.get(prev_token, "<unk>"),
                        "start": round(char_start_time, 3),
                        "end": round(char_end_time, 3),
                        "confidence": round(float(avg_prob), 3),
                        "is_pseudo": False
                    })

                    char_start_time = None
                    char_probs = []
                prev_token = self.blank_id

            elif token_id != prev_token:
                # 新字符开始
                if char_start_time is not None:
                    # 保存上一个字符
                    char_end_time = current_time
                    avg_prob = np.mean(char_probs) if char_probs else token_probs[t-1]

                    word_timestamps.append({
                        "word": self.vocab.get(prev_token, "<unk>"),
                        "start": round(char_start_time, 3),
                        "end": round(char_end_time, 3),
                        "confidence": round(float(avg_prob), 3),
                        "is_pseudo": False
                    })

                # 开始新字符
                char_start_time = current_time
                char_probs = [prob]
                prev_token = token_id

            else:
                # 同一字符持续
                if char_start_time is not None:
                    char_probs.append(prob)

        # 处理最后一个字符
        if char_start_time is not None and prev_token != self.blank_id:
            char_end_time = len(token_ids) * time_stride
            avg_prob = np.mean(char_probs) if char_probs else token_probs[-1]

            word_timestamps.append({
                "word": self.vocab.get(prev_token, "<unk>"),
                "start": round(char_start_time, 3),
                "end": round(char_end_time, 3),
                "confidence": round(float(avg_prob), 3),
                "is_pseudo": False
            })

        # 【新增】CTC 重叠去重：移除前缀重复（如 "W" + "Would" => "Would"）
        word_timestamps = self._remove_overlap_duplicates(word_timestamps)

        # 4. 拼接文本
        text = "".join([wt["word"] for wt in word_timestamps])

        # 5. 计算平均置信度
        avg_confidence = np.mean([wt["confidence"] for wt in word_timestamps]) if word_timestamps else 0.0

        return text, word_timestamps, float(avg_confidence)

    def _remove_overlap_duplicates(self, word_timestamps: List[Dict]) -> List[Dict]:
        """
        移除 CTC 解码产生的重叠前缀词（口吃去重）

        典型场景：
        - "W" (start=0.0, end=0.06, conf=0.572) + "Would" (start=0.06, end=0.24, conf=0.886)
        - 原因：CTC 在上一帧输出了 "W"，下一帧输出了完整的 "Would"
        - 修正：删除 "W"，只保留 "Would"

        去重条件（同时满足）：
        1. 当前词是下一个词的前缀（忽略大小写）
        2. 时间间隔 <= 0.1s（避免误判真正独立的单词）
        3. 当前词长度 <= 2（避免误删正常短词）

        Args:
            word_timestamps: 原始字级时间戳列表

        Returns:
            List[Dict]: 去重后的字级时间戳列表
        """
        if len(word_timestamps) < 2:
            return word_timestamps

        cleaned = []
        i = 0

        while i < len(word_timestamps):
            current = word_timestamps[i]
            current_word = current["word"].strip().lower()

            # 检查是否应该与下一个词合并
            if i + 1 < len(word_timestamps):
                next_word_info = word_timestamps[i + 1]
                next_word = next_word_info["word"].strip().lower()

                # 计算时间间隔
                time_gap = next_word_info["start"] - current["end"]

                # 判断是否为前缀重复
                is_prefix = (
                    len(current_word) <= 2 and  # 当前词极短
                    len(next_word) > len(current_word) and  # 下一个词更长
                    next_word.startswith(current_word) and  # 下一个词包含当前词作为前缀
                    time_gap <= 0.1  # 时间紧挨着或重叠
                )

                if is_prefix:
                    # 跳过当前词（前缀），保留下一个完整词
                    # logger.debug(
                    #     f"CTC 去重: 移除前缀 '{current_word}' -> 保留 '{next_word}' "
                    #     f"(gap={time_gap:.3f}s)"
                    # )
                    i += 1
                    continue

            # 保留当前词
            cleaned.append(current)
            i += 1

        return cleaned

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Softmax 函数"""
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


class SenseVoiceONNXService:
    """SenseVoice ONNX 推理服务（纯 ONNX Runtime 实现）"""

    # 语言 ID 映射
    LID_DICT = {
        "auto": 0,
        "zh": 3,
        "en": 4,
        "yue": 7,
        "ja": 11,
        "ko": 12,
        "nospeech": 13
    }

    # 文本规范化 ID 映射
    TEXTNORM_DICT = {
        "withitn": 14,  # 带逆文本规范化
        "woitn": 15     # 不带逆文本规范化
    }

    # 前端参数（来自 config.yaml）
    SAMPLE_RATE = 16000
    N_MELS = 80
    FRAME_LENGTH_MS = 25  # 窗长 25ms
    FRAME_SHIFT_MS = 10   # 帧移 10ms
    LFR_M = 7  # LFR 窗口大小
    LFR_N = 6  # LFR 步长

    def __init__(self, config=None):
        """
        初始化 SenseVoice 服务

        Args:
            config: SenseVoice 配置
        """
        if config is None:
            from app.models.sensevoice_models import SenseVoiceConfig
            config = SenseVoiceConfig()

        self.config = config
        self.session = None
        self.decoder = None
        self.is_loaded = False
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

        # 模型路径
        self.model_path = self._resolve_model_path()

        # 计算帧参数
        self.n_fft = int(self.SAMPLE_RATE * self.FRAME_LENGTH_MS / 1000)  # 400
        self.hop_length = int(self.SAMPLE_RATE * self.FRAME_SHIFT_MS / 1000)  # 160

        # 时间步长：LFR 后每帧代表 LFR_N * FRAME_SHIFT_MS = 60ms
        self.time_stride = self.LFR_N * self.FRAME_SHIFT_MS / 1000  # 0.06s

    def _resolve_model_path(self) -> str:
        """
        解析模型路径

        优先级：
        1. 项目预置模型（backend/models/pretrained/sensevoice/）
        2. 本地路径（如果 model_dir 是绝对路径）
        3. ModelScope 缓存路径
        4. HuggingFace 缓存路径
        5. 项目 models 目录（backend/models/sensevoice/）

        Returns:
            模型路径
        """
        from ..core.config import config as project_config

        # 优先使用项目预置模型（类似 Demucs 的方式）
        pretrained_model_path = Path(__file__).parent.parent.parent / "models" / "pretrained" / "sensevoice"
        if pretrained_model_path.exists() and (pretrained_model_path / "model_quant.onnx").exists():
            self.logger.info(f"使用预置模型: {pretrained_model_path}")
            return str(pretrained_model_path)

        model_dir = self.config.model_dir

        # 检查是否为本地路径（绝对路径）
        if os.path.isabs(model_dir) and os.path.exists(model_dir):
            self.logger.info(f"使用本地模型: {model_dir}")
            return model_dir

        # 检查 ModelScope 缓存
        home = Path.home()
        modelscope_cache = home / ".cache" / "modelscope" / "hub" / model_dir
        if modelscope_cache.exists():
            self.logger.info(f"使用 ModelScope 缓存: {modelscope_cache}")
            return str(modelscope_cache)

        # 检查 HuggingFace 缓存
        hf_cache = home / ".cache" / "huggingface" / "hub" / f"models--{model_dir.replace('/', '--')}"
        if hf_cache.exists():
            self.logger.info(f"使用 HuggingFace 缓存: {hf_cache}")
            return str(hf_cache)

        # 检查项目内的 models 目录
        project_model_path = project_config.MODELS_DIR / "sensevoice"
        if project_model_path.exists():
            self.logger.info(f"使用项目模型: {project_model_path}")
            return str(project_model_path)

        raise FileNotFoundError(
            f"未找到 SenseVoice 模型。请下载模型到以下任一位置：\n"
            f"1. {pretrained_model_path} (推荐，随项目发布)\n"
            f"2. {modelscope_cache}\n"
            f"3. {hf_cache}\n"
            f"4. {project_model_path}\n"
            f"或设置 model_dir 为绝对路径"
        )

    def load_model(self):
        """加载 SenseVoice ONNX 模型"""
        with self._lock:
            if self.is_loaded:
                self.logger.info("SenseVoice 模型已加载")
                return

            try:
                self.logger.info("开始加载 SenseVoice ONNX 模型...")
                self.logger.info(f"模型路径: {self.model_path}")

                # 导入 ONNX Runtime
                try:
                    import onnxruntime as ort
                except ImportError as e:
                    raise ImportError(
                        "onnxruntime 未安装。请运行: pip install onnxruntime-gpu 或 pip install onnxruntime"
                    ) from e

                # 查找 ONNX 模型文件
                model_file = self._find_onnx_model()
                self.logger.info(f"ONNX 模型文件: {model_file}")

                # 配置 ONNX Runtime
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                # V3.9: 智能 CPU 线程配置（根据 Intel/AMD 架构优化）
                # 原理：
                # - Intel 混合架构：仅使用 P-Core，避免 E-Core 拖慢推理
                # - AMD 全大核：使用物理核心数的 50-60%，避免跨 CCX 开销
                # - 使用物理核心而非逻辑核心，避免超线程带来的缓存竞争
                from app.utils.cpu_optimizer import ONNXThreadOptimizer, CPUArchitectureDetector

                vendor = CPUArchitectureDetector.detect_cpu_vendor()
                physical_cores = CPUArchitectureDetector.get_physical_cores()
                optimal_threads, thread_info = ONNXThreadOptimizer.calculate_optimal_threads(
                    vendor=vendor,
                    physical_cores=physical_cores,
                    usage_ratio=0.6  # 使用 60% 的核心，避免功耗墙
                )

                sess_options.intra_op_num_threads = optimal_threads  # 算子内部并行度
                sess_options.inter_op_num_threads = 1  # 算子间并行度

                # 通用 CPU 优化（适用于 Intel/AMD）
                try:
                    # 1. 关闭线程自旋等待，降低空转功耗
                    sess_options.add_session_config_entry('session.intra_op.allow_spinning', '0')
                    sess_options.add_session_config_entry('session.inter_op.allow_spinning', '0')

                    # 2. 将极小浮点数（denormal）视为 0，提升性能
                    sess_options.add_session_config_entry('session.set_denormal_as_zero', '1')

                    self.logger.debug("CPU 优化配置已启用: allow_spinning=0, denormal_as_zero=1")
                except Exception as e:
                    self.logger.debug(f"CPU 优化配置失败: {e}")

                self.logger.info(
                    f"CPU 线程配置: 厂商={vendor.upper()}, "
                    f"物理核心={physical_cores}, "
                    f"ONNX 使用={optimal_threads} 线程"
                )
                self.logger.info(f"配置策略: {thread_info['strategy']}")

                # 选择执行提供者
                providers = self._get_execution_providers()
                self.logger.info(f"执行提供者: {providers}")

                # 加载模型
                self.session = ort.InferenceSession(
                    model_file,
                    sess_options=sess_options,
                    providers=providers
                )

                # 加载词汇表
                vocab = self._load_vocab()
                self.decoder = CTCDecoder(vocab, blank_id=0)

                self.is_loaded = True
                self.logger.info("SenseVoice ONNX 模型加载成功")

                # 打印模型信息
                self._print_model_info()

            except Exception as e:
                self.logger.error(f"加载 SenseVoice 模型失败: {e}", exc_info=True)
                self.is_loaded = False
                raise

    def _find_onnx_model(self) -> str:
        """
        查找 ONNX 模型文件

        根据环境变量 SENSEVOICE_MODEL_TYPE 选择模型:
        - quantized (默认): 使用 INT8 量化模型 (model_quant.onnx), CPU 下速度最快
        - fp32: 使用 FP32 完整模型 (model.onnx), 支持 GPU 推理

        注意: CUDA 对 INT8 量化算子支持不完整, 使用 GPU 时建议选择 fp32
        """
        import os
        model_path = Path(self.model_path)

        # 从环境变量读取模型类型配置
        model_type = os.getenv('SENSEVOICE_MODEL_TYPE', 'quantized').lower()

        if model_type == 'fp32':
            # FP32 模型优先 (支持 GPU)
            candidates = [
                "model.onnx",                 # FP32 优先
                "sensevoice_small.onnx",      # FP32 备选
                "model_quant.onnx",           # INT8 回退
                "sensevoice_small_int8.onnx"
            ]
            self.logger.info("模型类型: FP32 (支持 GPU)")
        else:
            # 量化模型优先 (默认, CPU 速度最快)
            candidates = [
                "model_quant.onnx",           # INT8 优先
                "sensevoice_small_int8.onnx", # INT8 备选
                "model.onnx",                 # FP32 回退
                "sensevoice_small.onnx"
            ]
            self.logger.info("模型类型: INT8 量化 (CPU 优化)")

        for candidate in candidates:
            model_file = model_path / candidate
            if model_file.exists():
                self.logger.info(f"选择模型文件: {candidate}")
                return str(model_file)

        # 搜索所有 .onnx 文件
        onnx_files = list(model_path.glob("*.onnx"))
        if onnx_files:
            return str(onnx_files[0])

        raise FileNotFoundError(f"在 {model_path} 中未找到 ONNX 模型文件")

    def _load_vocab(self) -> Dict[int, str]:
        """加载词汇表"""
        model_path = Path(self.model_path)

        # 查找词汇表文件
        vocab_files = [
            "tokens.json",
            "vocab.json",
            "tokens.txt"
        ]

        for vocab_file in vocab_files:
            vocab_path = model_path / vocab_file
            if vocab_path.exists():
                self.logger.info(f"加载词汇表: {vocab_path}")

                if vocab_file.endswith(".json"):
                    with open(vocab_path, "r", encoding="utf-8") as f:
                        vocab_data = json.load(f)
                        # 转换为 {id: token} 格式
                        if isinstance(vocab_data, dict):
                            return {int(k): v for k, v in vocab_data.items()}
                        elif isinstance(vocab_data, list):
                            return {i: token for i, token in enumerate(vocab_data)}

                elif vocab_file.endswith(".txt"):
                    with open(vocab_path, "r", encoding="utf-8") as f:
                        tokens = [line.strip() for line in f]
                        return {i: token for i, token in enumerate(tokens)}

        self.logger.warning("未找到词汇表文件，使用默认词汇表")
        # 返回空词汇表（解码时会使用 <unk>）
        return {}

    def _get_execution_providers(self) -> List[str]:
        """
        获取执行提供者

        从环境变量 SENSEVOICE_DEVICE 读取设备配置:
        - cpu (默认): 强制使用 CPU, 配合 INT8 量化模型速度最优
        - cuda: 强制使用 GPU, 需要 FP32 模型
        - auto: 自动选择, 检测到 CUDA 则用 GPU, 否则用 CPU

        注意: CUDA 对 INT8 量化算子支持不完整, 使用 GPU 时建议:
              SENSEVOICE_DEVICE=cuda + SENSEVOICE_MODEL_TYPE=fp32
        """
        import os
        import onnxruntime as ort

        available_providers = ort.get_available_providers()
        self.logger.info(f"可用的执行提供者: {available_providers}")

        # 从环境变量读取设备配置 (默认 cpu)
        device = os.getenv('SENSEVOICE_DEVICE', 'cpu').lower()
        self.logger.info(f"SENSEVOICE_DEVICE 配置: {device}")

        if device == "auto":
            # 自动选择: GPU 优先, 降级 CPU
            if "CUDAExecutionProvider" in available_providers:
                self.logger.info("设备选择: auto -> CUDA (GPU 可用)")
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self.logger.info("设备选择: auto -> CPU (无可用 GPU)")
                return ["CPUExecutionProvider"]

        elif device == "cuda":
            # 强制 GPU
            if "CUDAExecutionProvider" in available_providers:
                self.logger.info("设备选择: CUDA (用户指定)")
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self.logger.warning("请求 CUDA 但不可用, 降级到 CPU")
                return ["CPUExecutionProvider"]

        else:
            # CPU 模式 (默认)
            self.logger.info("设备选择: CPU (默认, 配合 INT8 量化模型)")
            return ["CPUExecutionProvider"]

    def _print_model_info(self):
        """打印模型信息"""
        if self.session is None:
            return

        self.logger.info("=== 模型信息 ===")
        self.logger.info(f"输入: {[inp.name for inp in self.session.get_inputs()]}")
        self.logger.info(f"输出: {[out.name for out in self.session.get_outputs()]}")

    def unload_model(self):
        """卸载模型，释放内存"""
        with self._lock:
            if self.session is not None:
                del self.session
                self.session = None
                self.decoder = None
                self.is_loaded = False
                self.logger.info("SenseVoice 模型已卸载")

    def transcribe_audio_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
        language: str = None,
        use_itn: bool = None
    ) -> Dict:
        """
        转录音频数组（内存中的音频数据）

        Args:
            audio_array: 音频数组 (numpy array)，单声道
            sample_rate: 采样率（必须是 16000）
            language: 语言代码
            use_itn: 是否使用逆文本正则化

        Returns:
            转录结果字典，包含：
            - text: 原始文本（带标签）
            - text_clean: 清洗后的文本
            - words: 字级时间戳列表
            - confidence: 平均置信度
            - language: 检测到的语言
            - emotion: 情感标签
            - event: 事件标签
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        if sample_rate != 16000:
            raise ValueError(f"SenseVoice 要求采样率为 16000 Hz，当前为 {sample_rate} Hz")

        try:
            # 设置默认值
            if language is None:
                language = "auto"
            if use_itn is None:
                use_itn = True

            # 1. 音频预处理（提取 Fbank + LFR）
            audio_features = self._preprocess_audio(audio_array, sample_rate)

            self.logger.debug(f"特征形状: {audio_features.shape}")  # [batch, time, 560]

            # 2. 模型推理（传递完整的 4 个输入）
            logits = self._run_inference(audio_features, language=language, use_itn=use_itn)

            # 3. CTC 解码
            text, word_timestamps, confidence = self.decoder.decode(logits, self.time_stride)

            # 【阶段一】过滤特殊标记，并补偿时间偏移
            # 特殊标记格式：<|xxx|>，如 <|en|>, <|EMO_UNKNOWN|>, <|Speech|>, <|withitn|>
            # 这些标记占用时间轴（每个约60ms），过滤时需要累计其持续时间并从后续词中扣除
            clean_word_timestamps = []
            removed_duration = 0.0  # 累计被删除标记的持续时间

            for w in word_timestamps:
                word = w.get("word", "")
                if word.startswith("<|") and word.endswith("|>"):
                    # 累计被删除标记的持续时间
                    removed_duration += w.get("end", 0) - w.get("start", 0)
                    continue

                # 从后续词的时间戳中扣除累计的偏移，补偿标记占用的时间
                adjusted_w = w.copy()
                adjusted_w["start"] = max(0.0, w.get("start", 0) - removed_duration)
                adjusted_w["end"] = max(0.0, w.get("end", 0) - removed_duration)
                clean_word_timestamps.append(adjusted_w)

            word_timestamps = clean_word_timestamps

            # 【新增】Token 合并：将 BPE/SentencePiece Subword Tokens 合并为完整单词
            # 修复 "laval" 被拆分为 " la" + "val" 的问题
            word_timestamps = self._merge_tokens_to_words(word_timestamps)

            # 4. 提取标签信息并使用语言自适应标点归一化
            from ..services.text_normalizer import get_text_normalizer
            normalizer = get_text_normalizer()

            # 先提取语言标签
            tags = normalizer.extract_tags(text)
            detected_language = tags.get("language") if tags else language

            # 使用检测到的语言进行文本清洗和标点归一化
            process_result = normalizer.process(text, extract_info=True, language=detected_language)

            # 5. 构建结果
            result = {
                "text": text,
                "text_clean": process_result["text_clean"],
                "words": word_timestamps,
                "confidence": confidence,
                "language": detected_language,
                "emotion": process_result["tags"]["emotion"] if process_result["tags"] else None,
                "event": process_result["tags"]["event"] if process_result["tags"] else None
            }

            self.logger.debug(f"转录完成: {len(word_timestamps)} 个字符, 置信度 {confidence:.3f}")
            return result

        except Exception as e:
            self.logger.error(f"转录失败: {e}", exc_info=True)
            raise

    def _preprocess_audio(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        音频预处理：提取 Fbank 特征并做 LFR 处理

        Args:
            audio_array: 音频数组
            sample_rate: 采样率

        Returns:
            预处理后的特征 [batch, time, 560]
        """
        import librosa

        # 确保单声道
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)

        # 归一化
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        # 确保范围在 [-1, 1]
        max_val = np.abs(audio_array).max()
        if max_val > 1.0:
            audio_array = audio_array / max_val

        # 1. 提取 Mel 频谱特征 (Fbank)
        # 使用 librosa 提取 mel 频谱，与 kaldi fbank 类似
        mel_spec = librosa.feature.melspectrogram(
            y=audio_array,
            sr=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.N_MELS,
            window='hamming',
            center=False,
            power=1.0  # 使用幅度谱而非功率谱
        )

        # 转换为对数刻度 (log mel)
        fbank = np.log(np.maximum(mel_spec, 1e-10)).T  # [time, n_mels]

        # 2. 应用 LFR (Low Frame Rate) 处理
        # 将连续的 LFR_M 帧堆叠，步长为 LFR_N
        fbank_lfr = self._apply_lfr(fbank)  # [time_lfr, n_mels * LFR_M]

        # 3. 添加 batch 维度
        features = fbank_lfr[np.newaxis, :, :]  # [1, time, 560]

        return features.astype(np.float32)

    def _apply_lfr(self, fbank: np.ndarray) -> np.ndarray:
        """
        应用 LFR (Low Frame Rate) 处理

        将连续的 LFR_M 帧堆叠成一帧，步长为 LFR_N

        Args:
            fbank: 原始 Fbank 特征 [time, n_mels]

        Returns:
            LFR 后的特征 [time_lfr, n_mels * LFR_M]
        """
        T, D = fbank.shape  # time, n_mels (80)
        lfr_m = self.LFR_M  # 7
        lfr_n = self.LFR_N  # 6

        # 计算 LFR 后的时间步数
        # 公式：T_lfr = (T - lfr_m) // lfr_n + 1
        T_lfr = (T - lfr_m) // lfr_n + 1

        if T_lfr <= 0:
            # 如果音频太短，至少返回一帧（用 padding）
            # 用最后一帧填充
            padded = np.zeros((lfr_m, D), dtype=fbank.dtype)
            padded[:T] = fbank[:min(T, lfr_m)]
            if T < lfr_m:
                padded[T:] = fbank[-1] if T > 0 else 0
            return padded.reshape(1, -1)  # [1, 560]

        # 堆叠帧
        lfr_features = np.zeros((T_lfr, D * lfr_m), dtype=fbank.dtype)
        for i in range(T_lfr):
            start = i * lfr_n
            # 取 lfr_m 帧并拼接
            lfr_features[i] = fbank[start:start + lfr_m].reshape(-1)

        return lfr_features  # [time_lfr, 560]

    def _run_inference(
        self,
        audio_features: np.ndarray,
        language: str = "auto",
        use_itn: bool = True
    ) -> np.ndarray:
        """
        运行模型推理

        Args:
            audio_features: 音频特征 [batch, time, 560]
            language: 语言代码 (auto, zh, en, yue, ja, ko, nospeech)
            use_itn: 是否使用逆文本规范化

        Returns:
            logits: [time_steps, vocab_size]
        """
        batch_size = audio_features.shape[0]
        feats_length = audio_features.shape[1]

        # 准备 4 个输入
        # 1. speech: [batch, time, 560] float32
        speech = audio_features.astype(np.float32)

        # 2. speech_lengths: [batch] int32
        speech_lengths = np.array([feats_length] * batch_size, dtype=np.int32)

        # 3. language: [batch] int32
        lang_id = self.LID_DICT.get(language, self.LID_DICT["auto"])
        language_input = np.array([lang_id] * batch_size, dtype=np.int32)

        # 4. textnorm: [batch] int32
        textnorm_id = self.TEXTNORM_DICT["withitn" if use_itn else "woitn"]
        textnorm_input = np.array([textnorm_id] * batch_size, dtype=np.int32)

        # 运行推理
        outputs = self.session.run(
            None,
            {
                "speech": speech,
                "speech_lengths": speech_lengths,
                "language": language_input,
                "textnorm": textnorm_input
            }
        )

        # 提取 logits（第一个输出是 ctc_logits）
        logits = outputs[0]  # [batch, time_steps, vocab_size]

        # 移除 batch 维度
        logits = logits[0]  # [time_steps, vocab_size]

        return logits

    def transcribe(
        self,
        audio_path: Union[str, List[str]],
        language: str = None,
        use_itn: bool = None,
        ban_emo_unk: bool = None
    ) -> List[Dict]:
        """
        转录音频文件

        Args:
            audio_path: 音频文件路径或路径列表
            language: 语言代码
            use_itn: 是否使用逆文本正则化
            ban_emo_unk: 是否禁用未知情感标签

        Returns:
            转录结果列表
        """
        import librosa

        # 确保输入为列表
        if isinstance(audio_path, str):
            audio_paths = [audio_path]
        else:
            audio_paths = audio_path

        results = []
        for path in audio_paths:
            # 加载音频
            audio_array, sr = librosa.load(path, sr=16000, mono=True)

            # 转录
            result = self.transcribe_audio_array(audio_array, sr, language, use_itn)
            result["audio_path"] = path
            results.append(result)

        return results

    def _merge_tokens_to_words(self, tokens: List[Dict]) -> List[Dict]:
        """
        将 BPE/SentencePiece Subword Tokens 合并为完整单词

        典型场景：
        - " la" (0.900) + "val" (0.687) => " laval" (min=0.687)
        - 原因：SenseVoice 使用 Subword 分词器（BPE/SentencePiece）
        - 目标：将字符级时间戳聚合为词级时间戳

        合并规则：
        1. Token 以空格/▁ 开头 => 新词开始
        2. Token 不以空格开头 且 是字母数字 => 词的延续，合并到前一个词
        3. 标点符号 => 独立成词（视情况）

        置信度策略：取最小值（木桶效应），利于触发补刀

        Args:
            tokens: 原始字符级时间戳列表

        Returns:
            List[Dict]: 合并后的词级时间戳列表
        """
        if not tokens:
            return []

        merged_words = []
        current_word = None

        for token in tokens:
            text = token["word"]

            # 判断是否是新词的开始
            # 规则1：以空格开头（如 " la"）或 SentencePiece 符号 ▁ 开头
            # 规则2：当前没有正在构建的词
            # 规则3：标点符号通常独立（可选，这里暂不处理）
            is_new_word = (
                text.startswith(" ") or
                text.startswith("▁") or  # SentencePiece 词边界标记 (U+2581)
                current_word is None
            )

            if is_new_word:
                # 归档上一个词
                if current_word is not None:
                    # 清理 SentencePiece 符号 ▁ 和前导空格
                    current_word["word"] = current_word["word"].lstrip("▁ ")
                    merged_words.append(current_word)

                # 开始新词
                current_word = token.copy()
            else:
                # 合并到当前词
                # 1. 文本拼接
                current_word["word"] += text

                # 2. 时间延展（结束时间更新为当前 token 的结束时间）
                current_word["end"] = token["end"]

                # 3. 置信度：取最小值（木桶效应），反映该词最弱环节的可信度
                # 这样有利于触发 Whisper 补刀
                current_word["confidence"] = min(
                    current_word["confidence"],
                    token["confidence"]
                )

                # 4. 标记伪对齐状态
                if token.get("is_pseudo"):
                    current_word["is_pseudo"] = True

        # 归档最后一个词（同样清理前缀）
        if current_word is not None:
            current_word["word"] = current_word["word"].lstrip("▁ ")
            merged_words.append(current_word)

        return merged_words

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_name": "SenseVoice-Small",
            "model_path": self.model_path,
            "is_loaded": self.is_loaded,
            "device": self.config.device,
            "time_stride": self.time_stride,
            "supported_languages": ["zh", "en", "yue", "ja", "ko"],
            "features": [
                "多语言识别",
                "情感识别",
                "事件检测",
                "字级时间戳",
                "CTC 解码"
            ]
        }


# ========== 单例模式 ==========

_sensevoice_service_instance: Optional[SenseVoiceONNXService] = None
_instance_lock = threading.Lock()


def get_sensevoice_service(config=None) -> SenseVoiceONNXService:
    """
    获取 SenseVoice 服务单例

    Args:
        config: SenseVoice 配置（仅在首次创建时使用）

    Returns:
        SenseVoiceONNXService 实例
    """
    global _sensevoice_service_instance

    if _sensevoice_service_instance is None:
        with _instance_lock:
            if _sensevoice_service_instance is None:
                _sensevoice_service_instance = SenseVoiceONNXService(config)
                logger.info("SenseVoice 服务单例已创建")

    return _sensevoice_service_instance


def reset_sensevoice_service():
    """重置 SenseVoice 服务单例（用于测试）"""
    global _sensevoice_service_instance

    with _instance_lock:
        if _sensevoice_service_instance is not None:
            _sensevoice_service_instance.unload_model()
            _sensevoice_service_instance = None
            logger.info("SenseVoice 服务单例已重置")
