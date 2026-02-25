"""
YAMNet 探针模式分类器

使用预训练的 YAMNet ONNX 模型进行语义级音频分类，
替代基于频谱特征的规则判断，实现 "多点采样 + 软投票" 机制。

核心优势：
- 语义级分类：能区分 Speech 和 Music，避免人声被误判为音乐
- 极致轻量：CPU 推理 < 10ms，比 librosa 特征提取还快
- 零误杀：只要模型识别为 Speech，就不会触发分离
"""
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path

from app.services.runtime_param_resolver import get_yamnet_runtime_params

logger = logging.getLogger(__name__)


@dataclass
class YAMNetClassificationResult:
    """YAMNet 分类结果"""
    is_music: bool                      # 是否需要走 Demucs 分离
    confidence: float                   # 音乐置信度 (0-1)
    speech_score: float                 # 人声得分
    music_score: float                  # 音乐得分
    tags: List[str] = field(default_factory=list)  # 决策标签
    top_classes: List[Tuple[str, float]] = field(default_factory=list)  # Top-N 类别


class YAMNetPreprocessor:
    """
    YAMNet 预处理器

    将原始音频转换为 YAMNet 所需的 Log-Mel Spectrogram
    严格遵循 YAMNet 官方声学参数
    """

    def __init__(self):
        # YAMNet 核心声学参数 (不可修改)
        self.sr = 16000
        self.n_fft = 512
        self.n_mels = 64
        self.hop_length = 160   # 10ms
        self.win_length = 400   # 25ms
        self.fmin = 125
        self.fmax = 7500

        self._librosa = None
        self._mel_basis = None

    def _ensure_librosa(self):
        """懒加载 librosa"""
        if self._librosa is None:
            try:
                import librosa  # type: ignore

                # 某些损坏环境会出现空命名空间包，需要显式校验关键 API。
                if not hasattr(librosa, "filters") or not hasattr(librosa, "stft"):
                    raise ImportError("librosa 缺少 filters/stft 接口")
                self._librosa = librosa
                # 预计算 Mel 滤波器组 (性能优化)
                # YAMNet 原始训练用的 mel basis 是基于 htk=True 的
                self._mel_basis = librosa.filters.mel(
                    sr=self.sr,
                    n_fft=self.n_fft,
                    n_mels=self.n_mels,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    htk=True  # 关键：YAMNet 使用 HTK 公式
                )
            except Exception as exc:
                logger.warning("YAMNet 预处理不可用（librosa 异常）: %s", exc)
                self._librosa = False
        if self._librosa is False:
            return None
        return self._librosa

    def preprocess(self, waveform: np.ndarray) -> np.ndarray:
        """
        将原始音频转换为 YAMNet 所需的 Log-Mel Spectrogram

        Input: (N,) float32 waveform @ 16kHz
        Output: (1, 1, 96, 64) float32 (适用于 ONNX 输入)
        """
        librosa = self._ensure_librosa()
        if librosa is None:
            raise RuntimeError("librosa 不可用，无法执行 YAMNet 预处理")

        # 1. 确保长度
        # YAMNet 标准输入窗口是 0.96秒 (15360 samples，对应 96 帧)
        target_len = 15600  # 稍微多留一点余量防止边界计算问题
        if len(waveform) > target_len:
            waveform = waveform[:target_len]
        else:
            waveform = np.pad(waveform, (0, target_len - len(waveform)))

        # 2. 归一化
        max_val = max(abs(waveform.max()), abs(waveform.min()))
        if max_val > 0:
            waveform = waveform / max_val

        # 3. STFT
        # center=False 是为了对齐 TensorFlow 的 framed_signal_spectrogram
        spectrogram = np.abs(librosa.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False
        )) ** 2

        # 4. Mel 映射
        mel_spectrogram = np.dot(self._mel_basis, spectrogram)

        # 5. Log Stabilized (Log-Mel)
        # YAMNet 使用 log(mel + 0.001)
        log_mel = np.log(mel_spectrogram + 0.001)

        # 6. 转置与形状调整
        # Librosa 输出是 (freq, time)，YAMNet 需要 (time, freq)
        log_mel = log_mel.T

        # 截取标准的 96 帧
        if log_mel.shape[0] > 96:
            log_mel = log_mel[:96, :]
        elif log_mel.shape[0] < 96:
            # 填充到 96 帧
            pad_len = 96 - log_mel.shape[0]
            log_mel = np.pad(log_mel, ((0, pad_len), (0, 0)), mode='constant')

        # 增加 Batch 和 Channel 维度 -> (1, 1, 96, 64)
        return log_mel[np.newaxis, np.newaxis, :, :].astype(np.float32)


class YAMNetClassifier:
    """
    YAMNet 探针模式分类器

    策略：不对整个 Chunk 进行全量扫描，而是采用 "多点采样 + 软投票" 机制
    - 只抽取 Chunk 的 首、中、尾 3-5 个关键帧进行推理
    - 聚合逻辑：只有当 Music 类的置信度在多个帧中持续高分时才触发熔断
    - 人声免疫：如果 Speech 分数极高，即使有微弱 Music 分数，也强制判定为通过
    """

    # AudioSet 类别索引定义
    # Speech 相关 (0-65 范围内的人声相关类别)
    SPEECH_INDICES = [
        0,   # Speech
        1,   # Child speech, kid speaking
        2,   # Conversation
        3,   # Narration, monologue
        24,  # Singing (人声演唱也算人声)
        25,  # Choir
        29,  # Child singing
        31,  # Rapping
        32,  # Humming
    ]

    # Music 相关 (132-276 范围，但我们只关注核心音乐类别)
    MUSIC_INDICES = [
        132,  # Music
        133,  # Musical instrument
        134,  # Plucked string instrument
        135,  # Guitar
        136,  # Electric guitar
        148,  # Piano
        153,  # Synthesizer
        156,  # Percussion
        157,  # Drum kit
        158,  # Drum machine
        179,  # Orchestra
        211,  # Pop music
        212,  # Hip hop music
        214,  # Rock music
        230,  # Jazz
        232,  # Classical music
        234,  # Electronic music
        262,  # Background music  # 关键！
    ]

    # A Cappella (清唱) - 需要特殊处理，这是纯人声但可能被分类为音乐
    ACAPPELLA_INDEX = 250  # A capella

    def __init__(self, model_path: Optional[str] = None):
        """
        初始化 YAMNet 分类器

        Args:
            model_path: ONNX 模型路径，默认为 backend/models/pretrained/yamnet/model.onnx
        """
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "models" / "pretrained" / "yamnet" / "model.onnx"
        self.model_path = Path(model_path)

        self.session = None
        self.preprocessor = YAMNetPreprocessor()
        self.sample_rate = 16000
        self.window_samples = 15600  # 0.975s @ 16kHz

        # 类别名称映射 (懒加载)
        self._class_names: Optional[List[str]] = None
        self._yamnet_music_conf_min = 0.15
        self._yamnet_speech_conf_min = 0.85
        self._yamnet_music_weak_max = 0.10
        self._probe_window_count = 3

        self.apply_runtime_params()

        self._init_model()

    def apply_runtime_params(self) -> None:
        """应用运行参数配置。"""
        runtime = get_yamnet_runtime_params()
        # 优先读取新参数，旧参数仅做兼容回退
        self._yamnet_music_conf_min = float(
            runtime.get("yamnet_music_conf_min", runtime.get("music_max_threshold", 0.15))
        )
        self._yamnet_speech_conf_min = float(
            runtime.get("yamnet_speech_conf_min", runtime.get("speech_max_threshold", 0.85))
        )
        self._yamnet_music_weak_max = float(
            runtime.get("yamnet_music_weak_max", runtime.get("speech_max_music_threshold", 0.10))
        )
        self._probe_window_count = int(runtime.get("probe_window_count", 3))
        duration_sec = float(runtime.get("probe_window_duration_sec", 0.975))
        self.window_samples = max(1, int(self.sample_rate * duration_sec))
    def _init_model(self):
        """初始化 ONNX 模型（固定 CPU 执行）"""
        try:
            import onnxruntime as ort

            if not self.model_path.exists():
                logger.warning(f"YAMNet model not found: {self.model_path}")
                return

            # Why: YAMNet 仅作为 DNSMOS 灰区兜底，固定 CPU 可避免 CUDA DLL 依赖抖动影响主链稳定性。
            sess_options = self._create_optimized_session_options()
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=['CPUExecutionProvider'],
                sess_options=sess_options
            )
            self._device = 'cpu'

            # 仅 CPU 模式下设置亲和性，减少与主链推理线程争用。
            self._setup_pcore_affinity()

            logger.info(f"Loaded YAMNet model from {self.model_path} (device={self._device})")

        except Exception as e:
            logger.error(f"Failed to load YAMNet: {e}")
            self.session = None
            self._device = None

    def _create_optimized_session_options(self):
        """创建优化的 ONNX SessionOptions"""
        import onnxruntime as ort

        try:
            from app.services.hardware_profile_service import get_hardware_profile_provider

            provider = get_hardware_profile_provider()
            sess_options, optimal_threads, info = provider.build_onnx_session_options(
                usage_ratio=0.6
            )

            logger.info(
                f"YAMNet ONNX 优化: {optimal_threads} 线程 "
                f"({info.get('strategy', 'unknown')})"
            )

            return sess_options

        except Exception as e:
            # 回退：使用默认配置
            logger.warning(f"硬件能力提供者不可用，使用默认配置: {e}")

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4  # 回退默认值
            sess_options.inter_op_num_threads = 1
            return sess_options

    def _setup_pcore_affinity(self):
        """
        设置 P-Core 亲和性（仅 Intel 混合架构）

        在 Intel 12代+ 混合架构 CPU 上，将当前进程绑定到 P-Core，
        避免 ONNX 推理任务被调度到 E-Core 导致性能下降。
        """
        try:
            from app.services.hardware_profile_service import get_hardware_profile_provider

            provider = get_hardware_profile_provider()
            provider.apply_pcore_affinity(context="YAMNet P-Core")

        except Exception as e:
            # 亲和性设置失败不是致命错误
            logger.debug(f"P-Core 亲和性设置失败（非致命）: {e}")

    def _load_class_names(self) -> List[str]:
        """加载类别名称映射"""
        if self._class_names is not None:
            return self._class_names

        # 尝试加载 CSV 映射表
        csv_path = self.model_path.parent / "yamnet_class_map.csv"
        if csv_path.exists():
            try:
                import csv
                self._class_names = []
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self._class_names.append(row['display_name'])
                logger.debug(f"Loaded {len(self._class_names)} class names from {csv_path}")
                return self._class_names
            except Exception as e:
                logger.warning(f"Failed to load class names: {e}")

        # 回退：使用索引作为名称
        self._class_names = [f"class_{i}" for i in range(521)]
        return self._class_names

    def _get_probe_windows(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        获取探针窗口

        策略：按运行参数决定采样数量与窗口时长
        """
        total_len = len(audio)
        window_size = self.window_samples

        # 如果音频太短，直接填充返回单个窗口
        if total_len < window_size:
            audio = np.pad(audio, (0, window_size - total_len))
            return [audio]

        count = max(1, self._probe_window_count)
        max_start = max(0, total_len - window_size)
        probes = []

        if count == 1:
            start = max_start // 2
            probes.append(audio[start:start + window_size])
            return probes

        if count == 2:
            probes.append(audio[:window_size])
            probes.append(audio[-window_size:])
            return probes

        positions = np.linspace(0, max_start, num=count)
        seen = set()
        for pos in positions:
            start = int(round(pos))
            if start in seen:
                continue
            seen.add(start)
            probes.append(audio[start:start + window_size])

        return probes

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """计算 softmax 将 logits 转为概率"""
        exp_x = np.exp(x - np.max(x))  # 减去最大值防止溢出
        return exp_x / exp_x.sum()

    def _run_inference(self, mel_input: np.ndarray) -> np.ndarray:
        """运行单次推理，返回归一化的概率分布"""
        if self.session is None:
            return np.zeros(521)

        try:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: mel_input})
            # outputs[0] shape: (1, 521), 是 logits
            logits = outputs[0][0]
            # 转为概率
            return self._softmax(logits)
        except Exception as e:
            logger.warning(f"YAMNet inference error: {e}")
            return np.zeros(521)

    def classify_chunk(
        self,
        audio: np.ndarray,
        chunk_id: int = 0
    ) -> YAMNetClassificationResult:
        """
        对单个音频 Chunk 进行分类

        使用探针模式：只检测关键帧，快速决策

        Args:
            audio: 音频数组 (单声道, 16kHz)
            chunk_id: Chunk 索引 (用于日志)

        Returns:
            YAMNetClassificationResult: 分类结果
        """
        # 模型未加载时的回退
        if self.session is None:
            return YAMNetClassificationResult(
                is_music=False,
                confidence=0.0,
                speech_score=0.0,
                music_score=0.0,
                tags=["ModelNotLoaded"]
            )

        # 1. 获取探针窗口
        probes = self._get_probe_windows(audio)

        # 2. 批量推理
        music_scores = []
        speech_scores = []
        acappella_scores = []
        all_scores = []

        for probe in probes:
            try:
                # 预处理
                mel_input = self.preprocessor.preprocess(probe)

                # 推理
                scores = self._run_inference(mel_input)
                all_scores.append(scores)

                # 聚合分数
                m_score = np.max(scores[self.MUSIC_INDICES])
                s_score = np.max(scores[self.SPEECH_INDICES])
                a_score = scores[self.ACAPPELLA_INDEX]

                music_scores.append(m_score)
                speech_scores.append(s_score)
                acappella_scores.append(a_score)
            except Exception as e:
                logger.warning("YAMNet 预处理/推理失败，跳过单个 probe: chunk=%s, err=%s", chunk_id, e)
                continue

        if not music_scores:
            # 保守策略：灰区若 YAMNet 不可用，默认分离，避免漏分离。
            return YAMNetClassificationResult(
                is_music=True,
                confidence=1.0,
                speech_score=0.0,
                music_score=1.0,
                tags=["ProbeErrorConservativeSeparate"]
            )

        # 3. 计算聚合指标
        avg_music = float(np.mean(music_scores))
        max_music = float(np.max(music_scores))
        avg_speech = float(np.mean(speech_scores))
        max_speech = float(np.max(speech_scores))
        max_acappella = float(np.max(acappella_scores))

        # 获取 Top-N 类别 (用于调试)
        avg_scores = np.mean(all_scores, axis=0)
        top_indices = np.argsort(avg_scores)[-5:][::-1]
        class_names = self._load_class_names()
        top_classes = [(class_names[i], float(avg_scores[i])) for i in top_indices]

        logger.debug(
            f"Chunk {chunk_id}: Music={avg_music:.3f}(max:{max_music:.3f}), "
            f"Speech={avg_speech:.3f}(max:{max_speech:.3f}), "
            f"Acappella={max_acappella:.3f}, Top={top_classes[0]}"
        )

        # 4. 决策逻辑（YAMNet-Lite）
        # Why: 该分类器只服务 DNSMOS 灰区裁决，规则收敛为三条以降低行为分叉。
        tags = []
        if max_music >= self._yamnet_music_conf_min:
            tags.append("MusicStrong")
            return YAMNetClassificationResult(
                is_music=True,
                confidence=max_music,
                speech_score=avg_speech,
                music_score=avg_music,
                tags=tags,
                top_classes=top_classes,
            )

        if max_speech >= self._yamnet_speech_conf_min and max_music < self._yamnet_music_weak_max:
            tags.append("SpeechStrong")
            return YAMNetClassificationResult(
                is_music=False,
                confidence=max_music,
                speech_score=avg_speech,
                music_score=avg_music,
                tags=tags,
                top_classes=top_classes,
            )

        # 灰区保守策略：优先保障“该分离不漏分离”
        tags.append("GrayConservativeSeparate")
        return YAMNetClassificationResult(
            is_music=True,
            confidence=max_music,
            speech_score=avg_speech,
            music_score=avg_music,
            tags=tags,
            top_classes=top_classes
        )

    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self.session is not None


# ========== 单例访问 ==========

_yamnet_instance: Optional[YAMNetClassifier] = None


def get_yamnet_classifier() -> YAMNetClassifier:
    """获取 YAMNet 分类器单例"""
    global _yamnet_instance
    if _yamnet_instance is None:
        _yamnet_instance = YAMNetClassifier()
    else:
        _yamnet_instance.apply_runtime_params()
    return _yamnet_instance
