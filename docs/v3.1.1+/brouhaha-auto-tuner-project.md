# Brouhaha 自动调参项目开发文档

**版本**: V1.0.0
**日期**: 2026-01-08
**状态**: 设计文档

---

## 1. 项目概述

### 1.1 项目目标

创建一个独立的自动调参项目，用于优化 `video_to_srt_gpu` 项目中 Brouhaha SNR+C50 频谱分诊系统的阈值参数。

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| **离线运行** | 使用本地 Brouhaha 模型，无需联网 |
| **自动生成样本** | Edge-TTS 生成中英日三语人声 |
| **流式噪声** | HuggingFace Streaming 零磁盘占用 |
| **贝叶斯优化** | Optuna 自动搜索最佳参数 |
| **业务导向** | 漏报惩罚 > 误报惩罚（保质量优先） |

### 1.3 输出产物

给出完整的参数表

---

## 2. 待优化参数清单

### 2.1 Layer 1 核心阈值（高优先级）

| 参数 | 当前值 | 搜索范围 | 物理意义 |
|------|--------|----------|----------|
| `snr_high_threshold` | 25.0 dB | [20, 35] | SNR >= 此值直接放行 |
| `snr_low_threshold` | 12.0 dB | [8, 18] | SNR < 此值强制分离 |
| `c50_good_threshold` | 5.0 dB | [0, 15] | C50 >= 此值视为良好 |
| `c50_bad_threshold` | -5.0 dB | [-15, 0] | C50 < 此值视为严重混响 |

### 2.2 Layer 2 频谱阈值（中优先级）

| 参数 | 当前值 | 搜索范围 | 物理意义 |
|------|--------|----------|----------|
| `spectral_contrast_low` | 15.0 dB | [10, 25] | 低对比度警戒线 |
| `spectral_contrast_critical` | 12.0 dB | [8, 18] | 极低对比度分离线 |
| `spectral_flatness_high` | 0.4 | [0.2, 0.6] | 高平坦度阈值 |

### 2.3 综合判定阈值（低优先级）

| 参数 | 当前值 | 搜索范围 | 物理意义 |
|------|--------|----------|----------|
| `music_score_threshold` | 0.35 | [0.2, 0.5] | 音乐得分分离线 |
| `noise_score_threshold` | 0.45 | [0.3, 0.6] | 噪音得分分离线 |

---

## 3. 项目结构

```
brouhaha-auto-tuner/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖清单
├── config.py                    # 配置文件
│
├── models/
│   └── brouhaha/
│       └── pytorch_model.bin    # 从主项目复制 (47MB)
│
├── assets/
│   └── clean_samples/           # Edge-TTS 生成的纯净人声
│       ├── zh/                  # 中文样本
│       ├── en/                  # 英文样本
│       └── ja/                  # 日文样本
│
├── core/
│   ├── __init__.py
│   ├── brouhaha_detector.py     # Brouhaha 推理（照搬主项目）
│   ├── spectrum_features.py     # 频谱特征提取（照搬主项目）
│   └── decision_logic.py        # 三层决策逻辑（照搬主项目）
│
├── data/
│   ├── __init__.py
│   ├── tts_generator.py         # Edge-TTS 样本生成器
│   ├── noise_streamer.py        # HuggingFace 噪声流
│   └── audio_mixer.py           # 音频混合器
│
├── tuner/
│   ├── __init__.py
│   ├── objective.py             # Optuna 目标函数
│   ├── search_space.py          # 参数搜索空间
│   └── evaluator.py             # 决策评估器
│
├── scripts/
│   ├── 01_generate_samples.py   # 步骤1: 生成 TTS 样本
│   ├── 02_run_tuning.py         # 步骤2: 执行调参
│   └── 03_export_results.py     # 步骤3: 导出结果
│
└── results/
    ├── study.db                 # Optuna 数据库
    └── best_params.json         # 最佳参数输出
```

---

## 4. 依赖清单

### 4.1 requirements.txt

```txt
# PyTorch (与主项目一致)
--extra-index-url https://download.pytorch.org/whl/cu118
torch==2.4.0+cu118
torchaudio==2.4.0+cu118

# 数值计算
numpy==1.26.4
scipy

# Brouhaha 模型
pyannote.audio>=3.1.0
brouhaha @ https://github.com/marianne-m/brouhaha-vad/archive/main.zip

# 音频处理
librosa
soundfile

# TTS 样本生成
edge-tts

# 噪声数据流
datasets

# 优化引擎
optuna
optuna-dashboard  # 可选：可视化

# 工具
tqdm
click
```

---

## 5. 核心代码实现

### 5.1 配置文件 (config.py)

```python
"""
自动调参配置
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

@dataclass
class TunerConfig:
    """调参配置"""

    # 模型路径
    brouhaha_model_path: Path = PROJECT_ROOT / "models" / "brouhaha" / "pytorch_model.bin"

    # 样本配置
    sample_dir: Path = PROJECT_ROOT / "assets" / "clean_samples"
    samples_per_language: int = 10  # 每种语言生成的样本数
    sample_duration: float = 10.0   # 每个样本时长（秒）

    # TTS 配置
    tts_voices: dict = field(default_factory=lambda: {
        "zh": ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"],
        "en": ["en-US-JennyNeural", "en-US-GuyNeural"],
        "ja": ["ja-JP-NanamiNeural", "ja-JP-KeitaNeural"]
    })

    # 噪声数据集
    noise_dataset: str = "mathias/musan"
    noise_batch_size: int = 20

    # Optuna 配置
    n_trials: int = 100             # 搜索轮次
    n_samples_per_trial: int = 50   # 每轮验证样本数
    study_name: str = "brouhaha_threshold_tuning"
    storage: str = f"sqlite:///{PROJECT_ROOT}/results/study.db"

    # 惩罚系数（业务导向）
    false_negative_penalty: float = 10.0  # 漏报惩罚（应分离但未分离）
    false_positive_penalty: float = 1.0   # 误报惩罚（不应分离但分离了）

    # 采样率
    target_sr: int = 16000


@dataclass
class SearchSpace:
    """参数搜索空间"""

    # Layer 1: SNR/C50 阈值
    snr_high_threshold: tuple = (20.0, 35.0)
    snr_low_threshold: tuple = (8.0, 18.0)
    c50_good_threshold: tuple = (0.0, 15.0)
    c50_bad_threshold: tuple = (-15.0, 0.0)

    # Layer 2: 频谱特征阈值
    spectral_contrast_low: tuple = (10.0, 25.0)
    spectral_contrast_critical: tuple = (8.0, 18.0)
    spectral_flatness_high: tuple = (0.2, 0.6)


# 默认配置
DEFAULT_CONFIG = TunerConfig()
DEFAULT_SEARCH_SPACE = SearchSpace()
```

### 5.2 TTS 样本生成器 (data/tts_generator.py)

```python
"""
Edge-TTS 多语言样本生成器

生成中英日三语的纯净人声样本作为 Ground Truth
"""
import asyncio
import edge_tts
from pathlib import Path
import soundfile as sf
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

# 示例文本（覆盖不同语音特征）
SAMPLE_TEXTS = {
    "zh": [
        "今天的天气真不错，阳光明媚，适合出门散步。",
        "人工智能技术正在改变我们的生活方式。",
        "请问这个产品的价格是多少？有优惠吗？",
        "我们需要在下周三之前完成这个项目。",
        "这部电影讲述了一个感人的故事。",
        "学习外语需要持之以恒的努力和练习。",
        "欢迎来到我们的直播间，今天给大家带来好物推荐。",
        "根据最新的研究报告显示，这项技术已经取得重大突破。",
        "春节是中国最重要的传统节日之一。",
        "请您稍等，我马上为您处理这个问题。",
    ],
    "en": [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming industries worldwide.",
        "Could you please tell me the price of this product?",
        "We need to complete this project by next Wednesday.",
        "This movie tells a touching story about friendship.",
        "Learning a new language requires consistent effort and practice.",
        "Welcome to our live stream, today we have great recommendations.",
        "According to the latest research, this technology has made breakthroughs.",
        "Christmas is one of the most important holidays in Western culture.",
        "Please wait a moment, I will handle this issue for you.",
    ],
    "ja": [
        "今日の天気はとても良いですね。散歩に出かけましょう。",
        "人工知能技術は私たちの生活を変えています。",
        "この商品の価格はいくらですか？割引はありますか？",
        "来週の水曜日までにこのプロジェクトを完成させる必要があります。",
        "この映画は感動的な物語を描いています。",
        "外国語を学ぶには継続的な努力と練習が必要です。",
        "ライブ配信へようこそ。今日は良い商品をご紹介します。",
        "最新の研究報告によると、この技術は大きな進歩を遂げました。",
        "お正月は日本で最も重要な伝統的な祝日の一つです。",
        "少々お待ちください。すぐにこの問題を処理いたします。",
    ]
}


class TTSGenerator:
    """Edge-TTS 样本生成器"""

    def __init__(self, output_dir: Path, target_sr: int = 16000):
        self.output_dir = Path(output_dir)
        self.target_sr = target_sr

    async def generate_sample(
        self,
        text: str,
        voice: str,
        output_path: Path
    ) -> bool:
        """生成单个 TTS 样本"""
        try:
            communicate = edge_tts.Communicate(text, voice)

            # 临时保存为 MP3
            temp_mp3 = output_path.with_suffix(".mp3")
            await communicate.save(str(temp_mp3))

            # 转换为 16kHz WAV
            import torchaudio
            waveform, sr = torchaudio.load(str(temp_mp3))

            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = resampler(waveform)

            # 转单声道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # 保存 WAV
            torchaudio.save(str(output_path), waveform, self.target_sr)

            # 删除临时文件
            temp_mp3.unlink()

            return True

        except Exception as e:
            logger.error(f"TTS 生成失败: {e}")
            return False

    async def generate_all_samples(
        self,
        voices: dict,
        texts: dict = SAMPLE_TEXTS
    ) -> List[Path]:
        """生成所有语言的样本"""
        generated = []

        for lang, voice_list in voices.items():
            lang_dir = self.output_dir / lang
            lang_dir.mkdir(parents=True, exist_ok=True)

            lang_texts = texts.get(lang, [])

            for i, text in enumerate(lang_texts):
                # 轮流使用不同声音
                voice = voice_list[i % len(voice_list)]
                output_path = lang_dir / f"sample_{i:03d}.wav"

                success = await self.generate_sample(text, voice, output_path)
                if success:
                    generated.append(output_path)
                    logger.info(f"生成: {output_path.name} ({voice})")

        return generated


async def main():
    """生成样本入口"""
    from config import DEFAULT_CONFIG

    generator = TTSGenerator(
        output_dir=DEFAULT_CONFIG.sample_dir,
        target_sr=DEFAULT_CONFIG.target_sr
    )

    samples = await generator.generate_all_samples(
        voices=DEFAULT_CONFIG.tts_voices
    )

    print(f"共生成 {len(samples)} 个样本")


if __name__ == "__main__":
    asyncio.run(main())
```

### 5.3 噪声流加载器 (data/noise_streamer.py)

```python
"""
HuggingFace 流式噪声加载器

零磁盘占用，实时从云端加载噪声样本
"""
import torch
import torchaudio
from datasets import load_dataset
from typing import List, Iterator
import logging

logger = logging.getLogger(__name__)


class CloudNoiseStreamer:
    """云端噪声流式加载器"""

    def __init__(
        self,
        dataset_name: str = "mathias/musan",
        target_sr: int = 16000,
        min_duration: float = 3.0
    ):
        self.dataset_name = dataset_name
        self.target_sr = target_sr
        self.min_samples = int(min_duration * target_sr)
        self._iterator: Iterator = None
        self._dataset = None

    def _init_stream(self):
        """初始化 HuggingFace 流式连接"""
        logger.info(f"连接 HuggingFace ({self.dataset_name})...")
        try:
            self._dataset = load_dataset(
                self.dataset_name,
                split="train",
                streaming=True,
                trust_remote_code=True
            )
            self._iterator = iter(self._dataset)
            logger.info("HuggingFace 连接成功")
        except Exception as e:
            logger.error(f"连接 HF 失败: {e}")
            raise

    def _process_audio(self, item: dict) -> torch.Tensor:
        """处理单个音频项"""
        # 提取音频数据
        wav = torch.tensor(item['audio']['array']).float()
        src_sr = item['audio']['sampling_rate']

        # 重采样到目标采样率
        if src_sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(src_sr, self.target_sr)
            wav = resampler(wav)

        # 转单声道
        if wav.ndim > 1:
            wav = wav.mean(dim=0)

        return wav

    def get_noise_batch(self, batch_size: int = 10) -> List[torch.Tensor]:
        """获取一批噪声样本"""
        if self._iterator is None:
            self._init_stream()

        noises = []
        attempts = 0
        max_attempts = batch_size * 5  # 最多尝试次数

        while len(noises) < batch_size and attempts < max_attempts:
            attempts += 1
            try:
                item = next(self._iterator)
                wav = self._process_audio(item)

                # 过滤太短的片段
                if wav.shape[0] < self.min_samples:
                    continue

                noises.append(wav)

            except StopIteration:
                # 数据集遍历完毕，重新开始
                logger.info("噪声数据集遍历完毕，重新开始")
                self._init_stream()

        if len(noises) < batch_size:
            logger.warning(f"仅获取到 {len(noises)}/{batch_size} 个噪声样本")

        return noises


# 单元测试
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    streamer = CloudNoiseStreamer()
    batch = streamer.get_noise_batch(3)

    for i, noise in enumerate(batch):
        print(f"噪声 {i}: shape={noise.shape}, duration={noise.shape[0]/16000:.2f}s")
```

### 5.4 音频混合器 (data/audio_mixer.py)

```python
"""
音频混合器

按照精确的 SNR 值混合人声和噪声，生成 Ground Truth
"""
import torch
import numpy as np
from typing import Tuple


def calculate_rms(audio: torch.Tensor) -> torch.Tensor:
    """计算均方根能量"""
    return torch.sqrt(torch.mean(audio ** 2) + 1e-9)


def mix_audio(
    clean_speech: torch.Tensor,
    noise: torch.Tensor,
    target_snr_db: float
) -> torch.Tensor:
    """
    按照目标 SNR 混合人声和噪声

    SNR = 20 * log10(Speech_RMS / Noise_RMS)

    Args:
        clean_speech: 纯净人声张量
        noise: 噪声张量
        target_snr_db: 目标信噪比 (dB)

    Returns:
        混合后的音频张量
    """
    # 1. 长度对齐：将噪声切割或循环以匹配人声
    speech_len = clean_speech.shape[0]
    noise_len = noise.shape[0]

    if noise_len < speech_len:
        # 循环噪声直到足够长
        repeat_times = (speech_len // noise_len) + 1
        noise = noise.repeat(repeat_times)

    noise = noise[:speech_len]

    # 2. 计算当前能量
    speech_rms = calculate_rms(clean_speech)
    noise_rms = calculate_rms(noise)

    # 3. 计算目标噪声增益
    # SNR = 20 * log10(speech / noise)
    # noise_target = speech / 10^(SNR/20)
    target_noise_rms = speech_rms / (10 ** (target_snr_db / 20))
    gain = target_noise_rms / (noise_rms + 1e-9)

    # 4. 混合
    mixed = clean_speech + noise * gain

    # 5. 归一化防爆音
    max_val = torch.max(torch.abs(mixed))
    if max_val > 0.99:
        mixed = mixed * 0.99 / max_val

    return mixed


def generate_test_case(
    clean_speech: torch.Tensor,
    noise: torch.Tensor,
    target_snr_db: float
) -> Tuple[torch.Tensor, bool, str]:
    """
    生成测试用例

    Args:
        clean_speech: 纯净人声
        noise: 噪声
        target_snr_db: 目标 SNR

    Returns:
        (mixed_audio, should_separate, description)
    """
    mixed = mix_audio(clean_speech, noise, target_snr_db)

    # 根据 SNR 确定 Ground Truth
    # 这里使用主项目的阈值作为基准
    if target_snr_db >= 25:
        should_separate = False
        desc = f"高质量 (SNR={target_snr_db}dB)"
    elif target_snr_db < 12:
        should_separate = True
        desc = f"低质量 (SNR={target_snr_db}dB)"
    else:
        # 警戒区：根据具体值决定
        should_separate = target_snr_db < 18
        desc = f"警戒区 (SNR={target_snr_db}dB)"

    return mixed, should_separate, desc
```

### 5.5 Brouhaha 检测器 (core/brouhaha_detector.py)

```python
"""
Brouhaha 检测器

从主项目复制的核心推理逻辑，支持离线运行
"""
import os
import warnings
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class BrouhahaResult:
    """Brouhaha 检测结果"""
    snr: float          # 信噪比 (dB)
    c50: float          # 清晰度指数 (dB)
    vad: float          # 语音活动概率 (0-1)
    is_valid: bool      # 结果是否有效


class BrouhahaDetector:
    """
    Brouhaha SNR+C50 检测器

    直接从本地 checkpoint 文件加载模型，无需联网
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "auto"
    ):
        self.model_path = Path(model_path)
        self.model = None
        self._device = None

        # 设备选择
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._load_model()

    def _load_model(self):
        """从本地加载模型"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        try:
            from pyannote.audio import Model

            logger.info(f"从本地加载 Brouhaha 模型: {self.model_path}")

            # 抑制版本警告
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Model was trained with")
                warnings.filterwarnings("ignore", message="You are using `torch.load`")
                warnings.filterwarnings("ignore", message="Lightning automatically upgraded")

                self.model = Model.from_pretrained(
                    str(self.model_path.absolute()),
                    strict=False
                )

            self.model = self.model.to(self._device)
            self.model.eval()

            logger.info(f"模型加载成功，设备: {self._device}")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def detect(
        self,
        audio: np.ndarray,
        sr: int = 16000
    ) -> BrouhahaResult:
        """
        检测音频的 SNR 和 C50

        Args:
            audio: 音频数组 (numpy)
            sr: 采样率

        Returns:
            BrouhahaResult
        """
        if self.model is None:
            return BrouhahaResult(snr=0, c50=0, vad=0, is_valid=False)

        try:
            # 转换为张量
            if isinstance(audio, np.ndarray):
                audio_tensor = torch.from_numpy(audio).float()
            else:
                audio_tensor = audio.float()

            # 确保形状正确: (batch, channel, time)
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
            elif audio_tensor.ndim == 2:
                audio_tensor = audio_tensor.unsqueeze(0)

            audio_tensor = audio_tensor.to(self._device)

            # 推理
            with torch.no_grad():
                output = self.model(audio_tensor)

            # 解析输出 (Brouhaha 输出: [vad, snr, c50])
            # shape: (batch, frames, 3)
            if isinstance(output, dict):
                vad = output.get('vad', output.get('segmentation', None))
                snr = output.get('snr', None)
                c50 = output.get('c50', None)
            else:
                # 直接输出张量
                vad = output[:, :, 0].mean().item()
                snr = output[:, :, 1].mean().item()
                c50 = output[:, :, 2].mean().item()

            # 确保是标量
            if isinstance(snr, torch.Tensor):
                snr = snr.mean().item()
            if isinstance(c50, torch.Tensor):
                c50 = c50.mean().item()
            if isinstance(vad, torch.Tensor):
                vad = vad.mean().item()

            return BrouhahaResult(
                snr=float(snr),
                c50=float(c50),
                vad=float(vad),
                is_valid=True
            )

        except Exception as e:
            logger.error(f"Brouhaha 检测失败: {e}")
            return BrouhahaResult(snr=0, c50=0, vad=0, is_valid=False)

    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self.model is not None
```

### 5.6 决策逻辑 (core/decision_logic.py)

```python
"""
三层决策逻辑

照搬主项目 audio_spectrum_classifier.py 的决策逻辑
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import librosa


@dataclass
class TriageThresholds:
    """分诊阈值（待优化参数）"""

    # Layer 1: SNR/C50
    snr_high_threshold: float = 25.0
    snr_low_threshold: float = 12.0
    c50_good_threshold: float = 5.0
    c50_bad_threshold: float = -5.0

    # Layer 2: 频谱特征
    spectral_contrast_low: float = 15.0
    spectral_contrast_critical: float = 12.0
    spectral_flatness_high: float = 0.4


@dataclass
class TriageResult:
    """分诊结果"""
    need_separation: bool
    layer: int  # 1, 2, 或 3
    reason: str
    snr: float
    c50: float
    snr_level: str
    c50_level: str


class DecisionLogic:
    """三层决策逻辑"""

    def __init__(self, thresholds: TriageThresholds = None):
        self.th = thresholds or TriageThresholds()

    def update_thresholds(self, **kwargs):
        """动态更新阈值（用于 Optuna 搜索）"""
        for key, value in kwargs.items():
            if hasattr(self.th, key):
                setattr(self.th, key, value)

    def classify_snr_level(self, snr: float) -> str:
        """分类 SNR 级别"""
        if snr >= self.th.snr_high_threshold:
            return "high"
        elif snr >= self.th.snr_low_threshold:
            return "warn"
        else:
            return "low"

    def classify_c50_level(self, c50: float) -> str:
        """分类 C50 级别"""
        if c50 >= self.th.c50_good_threshold:
            return "good"
        elif c50 >= self.th.c50_bad_threshold:
            return "warn"
        else:
            return "bad"

    def layer1_decision(
        self,
        snr: float,
        c50: float,
        snr_level: str,
        c50_level: str
    ) -> Optional[Tuple[bool, str]]:
        """
        Layer 1: SNR+C50 快速筛选

        Returns:
            None 表示进入下一层
        """
        # 高质量：直接放行
        if snr_level == "high" and c50_level == "good":
            return (False, f"[L1] 高SNR({snr:.1f})+良好C50({c50:.1f})")

        # 低质量：强制分离
        if snr_level == "low":
            return (True, f"[L1] 低SNR({snr:.1f})")

        if c50_level == "bad":
            return (True, f"[L1] 严重混响C50({c50:.1f})")

        # 警戒区
        return None

    def layer2_decision(
        self,
        audio: np.ndarray,
        sr: int,
        snr_level: str,
        c50_level: str
    ) -> Optional[Tuple[bool, str]]:
        """
        Layer 2: 频谱特征补充判断
        """
        # 提取频谱对比度和平坦度
        contrast = self._extract_spectral_contrast(audio, sr)
        flatness = self._extract_spectral_flatness(audio, sr)

        # 极低对比度：强制分离
        if contrast < self.th.spectral_contrast_critical:
            return (True, f"[L2] 极低对比度({contrast:.1f}dB)")

        # 低对比度 + 高平坦度：分离
        if contrast < self.th.spectral_contrast_low and flatness > self.th.spectral_flatness_high:
            return (True, f"[L2] 低对比+高平坦({contrast:.1f}dB, {flatness:.2f})")

        # 高 SNR + 中 C50 + 良好对比度：放行
        if snr_level == "high" and c50_level == "warn":
            if contrast >= self.th.spectral_contrast_low:
                return (False, f"[L2] 高SNR+良好对比度({contrast:.1f}dB)")

        # 进入 Layer 3
        return None

    def layer3_decision(self, snr: float, c50: float) -> Tuple[bool, str]:
        """
        Layer 3: 简化版兜底决策

        在调参场景下，Layer 3 使用简单规则
        """
        # 综合评分
        score = (snr / 25.0) * 0.6 + ((c50 + 10) / 20.0) * 0.4

        if score >= 0.6:
            return (False, f"[L3] 综合评分通过({score:.2f})")
        else:
            return (True, f"[L3] 综合评分不足({score:.2f})")

    def triage(
        self,
        audio: np.ndarray,
        snr: float,
        c50: float,
        sr: int = 16000
    ) -> TriageResult:
        """
        执行完整的三层分诊
        """
        snr_level = self.classify_snr_level(snr)
        c50_level = self.classify_c50_level(c50)

        # Layer 1
        result = self.layer1_decision(snr, c50, snr_level, c50_level)
        if result is not None:
            need_sep, reason = result
            return TriageResult(
                need_separation=need_sep,
                layer=1,
                reason=reason,
                snr=snr,
                c50=c50,
                snr_level=snr_level,
                c50_level=c50_level
            )

        # Layer 2
        result = self.layer2_decision(audio, sr, snr_level, c50_level)
        if result is not None:
            need_sep, reason = result
            return TriageResult(
                need_separation=need_sep,
                layer=2,
                reason=reason,
                snr=snr,
                c50=c50,
                snr_level=snr_level,
                c50_level=c50_level
            )

        # Layer 3
        need_sep, reason = self.layer3_decision(snr, c50)
        return TriageResult(
            need_separation=need_sep,
            layer=3,
            reason=reason,
            snr=snr,
            c50=c50,
            snr_level=snr_level,
            c50_level=c50_level
        )

    def _extract_spectral_contrast(self, audio: np.ndarray, sr: int) -> float:
        """提取频谱对比度"""
        try:
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            return float(np.mean(contrast))
        except:
            return 15.0  # 默认值

    def _extract_spectral_flatness(self, audio: np.ndarray, sr: int) -> float:
        """提取频谱平坦度"""
        try:
            flatness = librosa.feature.spectral_flatness(y=audio)
            return float(np.mean(flatness))
        except:
            return 0.3  # 默认值
```

### 5.7 Optuna 目标函数 (tuner/objective.py)

```python
"""
Optuna 目标函数

定义参数搜索空间和损失计算逻辑
"""
import optuna
import torch
import torchaudio
import numpy as np
import random
from pathlib import Path
from typing import List
import logging

from config import TunerConfig, SearchSpace
from core.brouhaha_detector import BrouhahaDetector
from core.decision_logic import DecisionLogic, TriageThresholds
from data.noise_streamer import CloudNoiseStreamer
from data.audio_mixer import mix_audio

logger = logging.getLogger(__name__)


class TuningObjective:
    """调参目标函数"""

    def __init__(
        self,
        config: TunerConfig,
        search_space: SearchSpace,
        clean_samples: List[Path]
    ):
        self.config = config
        self.search_space = search_space
        self.clean_samples = clean_samples

        # 初始化组件
        self.detector = BrouhahaDetector(config.brouhaha_model_path)
        self.noise_streamer = CloudNoiseStreamer(
            dataset_name=config.noise_dataset,
            target_sr=config.target_sr
        )

        # 预加载纯净样本
        self._preload_samples()

    def _preload_samples(self):
        """预加载纯净人声样本"""
        self.loaded_samples = []

        for path in self.clean_samples:
            try:
                waveform, sr = torchaudio.load(str(path))
                if sr != self.config.target_sr:
                    resampler = torchaudio.transforms.Resample(sr, self.config.target_sr)
                    waveform = resampler(waveform)

                # 转单声道
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0)
                else:
                    waveform = waveform.squeeze(0)

                self.loaded_samples.append(waveform)

            except Exception as e:
                logger.warning(f"加载样本失败 {path}: {e}")

        logger.info(f"已加载 {len(self.loaded_samples)} 个纯净样本")

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Optuna 调用的目标函数

        返回值越小越好
        """
        # 1. 从搜索空间采样参数
        params = self._suggest_params(trial)

        # 2. 创建决策逻辑实例
        thresholds = TriageThresholds(**params)
        decision_logic = DecisionLogic(thresholds)

        # 3. 获取噪声批次
        noises = self.noise_streamer.get_noise_batch(self.config.noise_batch_size)

        # 4. 计算总惩罚
        total_penalty = 0.0

        for i in range(self.config.n_samples_per_trial):
            # 随机选择一个纯净样本
            clean = random.choice(self.loaded_samples)
            noise = random.choice(noises)

            # 随机生成目标 SNR（覆盖各种情况）
            # 50% 生成需要分离的难例，50% 生成不需要分离的易例
            if i % 2 == 0:
                # 难例：低 SNR
                target_snr = random.uniform(5, 15)
                ground_truth_separate = True
            else:
                # 易例：高 SNR
                target_snr = random.uniform(20, 35)
                ground_truth_separate = False

            # 混合音频
            mixed = mix_audio(clean, noise, target_snr)
            mixed_np = mixed.numpy()

            # Brouhaha 检测
            result = self.detector.detect(mixed_np, self.config.target_sr)

            if not result.is_valid:
                continue

            # 执行分诊
            triage = decision_logic.triage(
                audio=mixed_np,
                snr=result.snr,
                c50=result.c50,
                sr=self.config.target_sr
            )

            # 计算惩罚
            if ground_truth_separate and not triage.need_separation:
                # 漏报：应该分离但没分离（最严重）
                total_penalty += self.config.false_negative_penalty

            elif not ground_truth_separate and triage.need_separation:
                # 误报：不应分离但分离了
                total_penalty += self.config.false_positive_penalty

        return total_penalty

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        """从搜索空间采样参数"""
        ss = self.search_space

        return {
            # Layer 1
            "snr_high_threshold": trial.suggest_float(
                "snr_high_threshold", ss.snr_high_threshold[0], ss.snr_high_threshold[1]
            ),
            "snr_low_threshold": trial.suggest_float(
                "snr_low_threshold", ss.snr_low_threshold[0], ss.snr_low_threshold[1]
            ),
            "c50_good_threshold": trial.suggest_float(
                "c50_good_threshold", ss.c50_good_threshold[0], ss.c50_good_threshold[1]
            ),
            "c50_bad_threshold": trial.suggest_float(
                "c50_bad_threshold", ss.c50_bad_threshold[0], ss.c50_bad_threshold[1]
            ),
            # Layer 2
            "spectral_contrast_low": trial.suggest_float(
                "spectral_contrast_low", ss.spectral_contrast_low[0], ss.spectral_contrast_low[1]
            ),
            "spectral_contrast_critical": trial.suggest_float(
                "spectral_contrast_critical", ss.spectral_contrast_critical[0], ss.spectral_contrast_critical[1]
            ),
            "spectral_flatness_high": trial.suggest_float(
                "spectral_flatness_high", ss.spectral_flatness_high[0], ss.spectral_flatness_high[1]
            ),
        }
```

### 5.8 主执行脚本 (scripts/02_run_tuning.py)

```python
"""
步骤 2: 执行自动调参

使用 Optuna 搜索最佳阈值参数
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
import json
import logging
from datetime import datetime

from config import DEFAULT_CONFIG, DEFAULT_SEARCH_SPACE
from tuner.objective import TuningObjective

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_clean_samples(sample_dir: Path) -> list:
    """收集所有纯净样本"""
    samples = []
    for lang_dir in sample_dir.iterdir():
        if lang_dir.is_dir():
            for wav_file in lang_dir.glob("*.wav"):
                samples.append(wav_file)
    return samples


def main():
    config = DEFAULT_CONFIG
    search_space = DEFAULT_SEARCH_SPACE

    # 收集纯净样本
    clean_samples = collect_clean_samples(config.sample_dir)
    if len(clean_samples) == 0:
        logger.error("未找到纯净样本，请先运行 01_generate_samples.py")
        return

    logger.info(f"找到 {len(clean_samples)} 个纯净样本")

    # 创建目标函数
    objective = TuningObjective(
        config=config,
        search_space=search_space,
        clean_samples=clean_samples
    )

    # 创建 Optuna Study
    study = optuna.create_study(
        study_name=config.study_name,
        storage=config.storage,
        direction="minimize",
        load_if_exists=True
    )

    # 开始搜索
    logger.info(f"开始自动调参，计划运行 {config.n_trials} 轮...")
    study.optimize(
        objective,
        n_trials=config.n_trials,
        show_progress_bar=True
    )

    # 输出结果
    print("\n" + "=" * 60)
    print("调参完成！")
    print("=" * 60)
    print(f"最佳惩罚值: {study.best_value:.2f}")
    print(f"最佳参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.4f}")

    # 保存结果
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials)
    }

    output_path = results_dir / "best_params.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_path}")

    # 生成可复制的代码
    print("\n" + "=" * 60)
    print("请将以下代码复制到主项目的 spectrum_thresholds.py:")
    print("=" * 60)
    print("""
@dataclass
class SpectrumThresholds:
    # ... 其他字段保持不变 ...

    # ========== Brouhaha SNR+C50 分诊阈值 (自动调参结果) ==========
""")
    for key, value in study.best_params.items():
        print(f"    {key}: float = {value:.2f}")


if __name__ == "__main__":
    main()
```

---

## 6. 使用指南

### 6.1 环境准备

```bash
# 1. 创建项目目录
mkdir brouhaha-auto-tuner
cd brouhaha-auto-tuner

# 2. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 复制 Brouhaha 模型
mkdir -p models/brouhaha
copy "F:\video_to_srt_gpu\backend\models\pretrained\brouhaha\pytorch_model.bin" models/brouhaha/
```

### 6.2 执行流程

```bash
# 步骤 1: 生成 TTS 样本（需要联网）
python scripts/01_generate_samples.py

# 步骤 2: 执行调参（需要联网加载噪声）
python scripts/02_run_tuning.py

# 步骤 3: 导出结果
python scripts/03_export_results.py
```

### 6.3 可视化监控（可选）

```bash
# 启动 Optuna Dashboard
optuna-dashboard sqlite:///results/study.db
# 浏览器访问 http://localhost:8080
```

---

## 7. 注意事项

### 7.1 网络要求

| 步骤 | 网络需求 |
|------|----------|
| 生成 TTS 样本 | 需要（Edge-TTS 云端） |
| 加载噪声数据 | 需要（HuggingFace Streaming） |
| Brouhaha 推理 | 不需要（本地模型） |

### 7.2 硬件要求

| 资源 | 推荐配置 |
|------|----------|
| GPU | CUDA 11.8+，4GB+ 显存 |
| RAM | 8GB+ |
| 磁盘 | ~100MB（模型 + 样本） |

### 7.3 调参时间估计

| 配置 | 预计时间 |
|------|----------|
| 50 trials, 50 samples/trial | ~30 分钟 |
| 100 trials, 100 samples/trial | ~2 小时 |

---

## 8. 输出示例

```json
{
  "timestamp": "2026-01-08T15:30:00",
  "best_value": 12.0,
  "best_params": {
    "snr_high_threshold": 23.45,
    "snr_low_threshold": 13.82,
    "c50_good_threshold": 4.21,
    "c50_bad_threshold": -6.15,
    "spectral_contrast_low": 14.33,
    "spectral_contrast_critical": 11.27,
    "spectral_flatness_high": 0.38
  },
  "n_trials": 100
}
```

---
