# SenseVoice 集成开发计划

> 基于现有 video_to_srt_gpu 项目架构，集成 SenseVoice 作为主要转录引擎，实现高速、高精度的语音识别能力。

---

## 一、项目背景与目标

### 1.1 当前架构

| 组件 | 当前实现 | 说明 |
|------|----------|------|
| 转录引擎 | Faster-Whisper | 基于 Whisper，支持时间戳 |
| VAD | Silero VAD / Pyannote | 语音活动检测 |
| 人声分离 | Demucs (htdemucs) | 可选的背景音乐分离 |
| 模型管理 | ModelPreloadManager | LRU 缓存，最多 3 个模型 |

### 1.2 SenseVoice 优势

| 特性 | SenseVoice | Whisper |
|------|-----------|---------|
| **推理速度** | 10s 音频仅 70ms | 10s 音频约 1000ms |
| **中文识别** | 优于 Whisper | 一般 |
| **情感识别** | 原生支持 | 不支持 |
| **事件检测** | 支持 BGM/掌声/笑声等 | 不支持 |
| **时间戳** | 原生字级时间戳 | 支持 word_timestamps |

### 1.3 集成目标

1. **SenseVoice 作为主要转录引擎**：使用 ONNX Runtime (INT8 量化) 部署
2. **Whisper 作为复核引擎**：仅处理低置信度片段（<0.6）
3. **智能熔断机制**：人声分离与 ASR 复核协调联动
4. **统一模型管理**：复用现有 LRU 缓存机制
5. **向后兼容**：支持用户选择 Faster-Whisper 或 SenseVoice

---

## 二、现有设计问题分析

### 2.1 进度条逻辑僵化

**问题描述**：当前设计给转录阶段分配固定权重（如 70%），无法区分不同场景：
- "仅运行 SenseVoice" vs "SenseVoice + Whisper 复核"
- 开启复核时进度条可能停滞或回跳
- 关闭复核时进度条可能过早走完

**解决方案**：动态权重调整机制，根据实际需要复核的片段数量调整权重分配。

### 2.2 缺乏句级切分（语义粒度缺失）

**问题描述**：直接将 VAD 切分的片段（可能长达 30 秒）作为一个字幕块输出。SenseVoice 虽然有字级时间戳，但缺失将这些字根据标点和停顿组合成短句的逻辑，导致用户看到的是"一大坨"文字。

**解决方案**：新增分句算法，基于标点、停顿、长度进行智能切分。

### 2.3 熔断机制割裂

**问题描述**：人声分离（Demucs）的升级逻辑与 ASR（Whisper）的复核逻辑是分离的。系统可能在背景噪音极大的情况下直接让 Whisper 复核（效果依然差），而不是先升级分离模型去除噪音。

**解决方案**：整合智能熔断机制，熔断升级优先于 ASR 复核。

### 2.4 VAD 角色定位模糊

**问题描述**：在 SenseVoice 自带时间戳的情况下，VAD 容易被误解为多余。实际上 VAD 仍有重要作用。

**解决方案**：重新定义 VAD 职责为物理切分，字幕的语义粒度由分句算法负责。

---

## 三、技术架构设计

### 3.1 整体架构（三层漏斗模型 - 改进版）

```
                    +-------------------+
                    |   视频/音频输入    |
                    +-------------------+
                            |
    ======================== 物理层 ========================
                            |
                    +-------------------+
                    | FFmpeg: 提取音频   |
                    | 输出: 16kHz WAV    |
                    +-------------------+
                            |
                    +-------------------+
                    | Silero VAD: 物理切分|
                    | 输出: 15-30s Chunk  |
                    | (防幻觉+显存保护)   |
                    +-------------------+
                            |
    ======================== 智能熔断层 ========================
                            |
                    +-------------------+
                    | 频谱指纹预判 (CPU)  |
                    | - ZCR方差检测      |
                    | - 滑动窗口惯性     |
                    +-------------------+
                            |
                   /        |        \
           clean片段   可疑片段    噪音片段
                |           |           |
           直接转录    按需分离     升级分离
                            |
                    +-------------------+
                    | Demucs: 人声分离   |
                    | (htdemucs/mdx_extra)|
                    +-------------------+
                            |
    ======================== 声学层 ========================
                            |
                    +-------------------+
                    | SenseVoice-ONNX   |
                    | (INT8量化)         |
                    | + 字级时间戳       |
                    +-------------------+
                            |
                    +-------------------+
                    | 分句算法 (CPU)     |
                    | - 标点切分         |
                    | - 停顿检测(>0.4s)  |
                    | - 长度限制(<30字)  |
                    +-------------------+
                            |
                +------------------------+
                |    置信度门控 (0.6)    |
                +------------------------+
               /                          \
        >= 0.6                           < 0.6
              |                              |
    +-------------------+          +-------------------+
    |   直接进入结果池   |          | Whisper: 二次识别 |
    +-------------------+          +-------------------+
              |                              |
              +-------------+----------------+
                            |
    ======================== 输出层 ========================
                            |
                    +-------------------+
                    | SSE 流式推送       |
                    | (句级粒度实时输出)  |
                    +-------------------+
                            |
                    +-------------------+
                    | SRT/ASS 字幕生成   |
                    +-------------------+
```

### 3.2 VAD 职责重新定义（物理切分层）

#### 3.2.1 VAD 的新定义

**不再负责**：决定字幕的长相和边界

**核心职责**：
1. **防幻觉**：过滤纯静音段，防止 SenseVoice 在无语音处产生幻觉
2. **显存保护**：将长音频切分为 15-30s 的 Chunk，防止 OOM
3. **并行加速**：为 GPU 提供 Batch 推理的原材料

#### 3.2.2 VAD 配置参数

```python
@dataclass
class VADConfig:
    """VAD 物理切分配置"""
    min_speech_duration_ms: int = 250     # 最小语音段长度
    min_silence_duration_ms: int = 100    # 最小静音段长度
    max_segment_duration_s: float = 30.0  # 最大片段长度（显存保护）
    target_segment_duration_s: float = 15.0  # 目标片段长度
    padding_ms: int = 200                 # 前后 padding
```

### 3.3 分句算法设计（语义切分层）

#### 3.3.1 算法概述

分句算法将 SenseVoice 输出的字级时间戳列表，根据语义规则切分为用户可读的短句字幕。

#### 3.3.2 切分条件（优先级从高到低）

| 优先级 | 条件 | 说明 |
|--------|------|------|
| 1 | 标点符号 | 遇到 `。？！` 必切 |
| 2 | 长停顿 | 相邻字间隔 > 0.4s 必切 |
| 3 | 强制长度 | 单句超过 5秒 或 30个字 强制切分 |

#### 3.3.3 核心实现

**文件位置**：`backend/app/services/sentence_splitter.py`

```python
from dataclasses import dataclass
from typing import List, Optional
import re


@dataclass
class WordTimestamp:
    """字级时间戳"""
    word: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class SentenceSegment:
    """句级字幕段"""
    text: str
    start: float
    end: float
    words: List[WordTimestamp]


class SentenceSplitter:
    """
    分句算法 - 将字级时间戳切分为句级字幕

    嵌入位置: _transcribe_segment_sensevoice 返回结果之前
    """

    def __init__(self, config: dict = None):
        self.config = config or self._default_config()

        # 强制切分的标点符号
        self.terminal_punctuation = set('。？！?!')
        # 弱切分的标点符号（仅在句子过长时考虑）
        self.weak_punctuation = set('，,、；;：:')

    def _default_config(self) -> dict:
        return {
            'pause_threshold': 0.4,       # 停顿切分阈值（秒）
            'max_duration': 5.0,          # 最大时长（秒）
            'max_chars': 30,              # 最大字符数
            'min_chars': 2,               # 最小字符数（避免单字成句）
        }

    def split(self, words: List[WordTimestamp]) -> List[SentenceSegment]:
        """
        将字级时间戳列表切分为句级字幕列表

        Args:
            words: SenseVoice 输出的字级时间戳列表

        Returns:
            List[SentenceSegment]: 句级字幕列表
        """
        if not words:
            return []

        sentences = []
        current_words = []
        current_start = words[0].start

        for i, word in enumerate(words):
            current_words.append(word)

            # 判断是否需要切分
            should_split, reason = self._should_split(
                current_words,
                word,
                words[i + 1] if i + 1 < len(words) else None
            )

            if should_split:
                # 生成句子
                sentence = self._create_sentence(current_words, current_start)
                if sentence:
                    sentences.append(sentence)

                # 重置
                current_words = []
                if i + 1 < len(words):
                    current_start = words[i + 1].start

        # 处理剩余的词
        if current_words:
            sentence = self._create_sentence(current_words, current_start)
            if sentence:
                sentences.append(sentence)

        return sentences

    def _should_split(
        self,
        current_words: List[WordTimestamp],
        current_word: WordTimestamp,
        next_word: Optional[WordTimestamp]
    ) -> tuple:
        """
        判断是否应该在当前词后切分

        Returns:
            (should_split: bool, reason: str)
        """
        text = ''.join(w.word for w in current_words)
        duration = current_word.end - current_words[0].start

        # 规则1: 终结标点符号必切
        if current_word.word in self.terminal_punctuation:
            return True, 'terminal_punctuation'

        # 规则2: 长停顿必切
        if next_word:
            pause = next_word.start - current_word.end
            if pause > self.config['pause_threshold']:
                return True, f'long_pause_{pause:.2f}s'

        # 规则3: 超过最大时长或字数，强制切分
        if duration >= self.config['max_duration']:
            return True, f'max_duration_{duration:.2f}s'

        if len(text) >= self.config['max_chars']:
            # 尝试在弱标点处切分
            for j in range(len(current_words) - 1, -1, -1):
                if current_words[j].word in self.weak_punctuation:
                    return True, 'max_chars_weak_punct'
            return True, 'max_chars_forced'

        return False, ''

    def _create_sentence(
        self,
        words: List[WordTimestamp],
        start: float
    ) -> Optional[SentenceSegment]:
        """创建句子段落"""
        if not words:
            return None

        text = ''.join(w.word for w in words)

        # 过滤过短的句子
        if len(text.strip()) < self.config['min_chars']:
            return None

        return SentenceSegment(
            text=text,
            start=start,
            end=words[-1].end,
            words=words
        )
```

### 3.4 智能熔断机制（协调联动）

详见：`docs/SenseVoice/最终方案/智能熔断机制集成改进方案.md`

#### 3.4.1 熔断决策流程

```
                    片段输入
                       |
            +-----------------------+
            |   频谱指纹预判 (CPU)   |
            |   - ZCR方差           |
            |   - 谱质心标准差       |
            +-----------------------+
                       |
         +-------------+-------------+
         |             |             |
      clean        suspicious     noisy
      (直接)       (需验证)      (需分离)
         |             |             |
         |      +------+------+      |
         |      |             |      |
         |   Demucs验证    直接分离   |
         |      |             |      |
         |      +------+------+      |
         |             |             |
         +-------------+-------------+
                       |
            +-----------------------+
            |  SenseVoice 转录      |
            +-----------------------+
                       |
         +-------------+-------------+
         |                           |
     高置信度                     低置信度
     (>=0.6)                      (<0.6)
         |                           |
    直接采用              +----------+----------+
         |                |                     |
         |           检查熔断状态          直接Whisper
         |                |                     |
         |     已升级?────────未升级             |
         |        |              |              |
         |   Whisper复核    升级分离模型        |
         |                       |              |
         |                  重新转录            |
         |                       |              |
         +----------+------------+              |
                    |                           |
               采用结果 <-----------------------+
```

#### 3.4.2 核心决策逻辑

```python
class FuseBreaker:
    """
    熔断决策器 - 协调人声分离与ASR复核

    决策原则：
    1. 熔断升级优先于ASR复核
    2. 已升级后才允许Whisper复核
    3. 避免在噪音环境下盲目复核
    """

    def __init__(self, config: dict = None):
        self.config = config or self._default_config()
        self.fuse_state = {}  # 片段熔断状态记录

    def _default_config(self) -> dict:
        return {
            'confidence_threshold': 0.6,
            'upgrade_threshold': 0.4,      # 低于此值触发模型升级
            'noise_event_tags': ['BGM', 'Noise', 'Cough'],
        }

    def decide_action(
        self,
        segment_id: int,
        confidence: float,
        event_tag: Optional[str],
        separation_level: str  # 'none' | 'htdemucs' | 'mdx_extra'
    ) -> str:
        """
        决策下一步动作

        Returns:
            'accept': 接受结果
            'upgrade_separation': 升级分离模型
            'whisper_retry': Whisper复核
        """
        # 高置信度：直接接受
        if confidence >= self.config['confidence_threshold']:
            return 'accept'

        # 检测到噪音事件且未升级：优先升级分离
        if event_tag in self.config['noise_event_tags']:
            if separation_level != 'mdx_extra':
                return 'upgrade_separation'

        # 极低置信度且未升级：优先升级分离
        if confidence < self.config['upgrade_threshold']:
            if separation_level != 'mdx_extra':
                return 'upgrade_separation'

        # 已升级或无需升级：Whisper复核
        return 'whisper_retry'
```

### 3.5 动态权重调整机制

#### 3.5.1 场景分析

| 场景 | 转录权重 | 复核权重 | 分离权重 |
|------|----------|----------|----------|
| 纯净语音 | 80% | 0% | 0% |
| 轻度BGM | 70% | 5% | 5% |
| 重度BGM | 50% | 10% | 20% |
| 低质量音频 | 40% | 30% | 10% |

#### 3.5.2 实现方案

```python
def calculate_dynamic_weights(
    total_segments: int,
    segments_to_separate: int,
    segments_to_retry: int,
    engine: str = 'sensevoice'
) -> Dict[str, int]:
    """
    动态计算阶段权重

    Args:
        total_segments: 总片段数
        segments_to_separate: 需要分离的片段数
        segments_to_retry: 需要复核的片段数（预估）
        engine: 转录引擎

    Returns:
        动态权重字典
    """
    # 基础权重
    base_weights = {
        'extract': 5,
        'vad': 5,
        'bgm_detect': 2,
        'demucs': 0,
        'transcribe': 70,
        'retry': 0,
        'sentence_split': 3,
        'srt': 10,
        'complete': 5
    }

    # 计算分离比例
    sep_ratio = segments_to_separate / total_segments if total_segments > 0 else 0
    retry_ratio = segments_to_retry / total_segments if total_segments > 0 else 0

    # 动态调整
    base_weights['demucs'] = int(15 * sep_ratio)  # 最多15%
    base_weights['retry'] = int(20 * retry_ratio)  # 最多20%

    # 从转录阶段扣除
    used = base_weights['demucs'] + base_weights['retry']
    base_weights['transcribe'] = max(40, 70 - used)

    return base_weights
```

### 3.6 新流式推送机制（句级 SSE）

#### 3.6.1 后端处理流程

```
VAD 切出 Chunk (15s)
         |
         v
SenseVoice 转录
         |
         v
分句算法切成 3 个短句
         |
         v
SSE 逐句推送
```

#### 3.6.2 SSE 事件结构

```python
# 句级推送事件（新增）
{
    "event": "sentence",
    "data": {
        "sentence_id": 1,
        "text": "这是第一个句子。",
        "start": 0.5,
        "end": 2.3,
        "confidence": 0.85,
        "is_final": True,  # False 表示可能被后续修正
        "source": "sensevoice"
    }
}

# 批次完成事件
{
    "event": "batch_complete",
    "data": {
        "batch_id": 1,
        "segment_start": 0.0,
        "segment_end": 15.0,
        "sentence_count": 5,
        "avg_confidence": 0.82
    }
}

# 复核事件
{
    "event": "retry_started",
    "data": {
        "sentence_id": 3,
        "reason": "low_confidence",
        "original_confidence": 0.45
    }
}
```

---

## 四、SenseVoice ONNX 部署方案

### 4.1 模型要求

**重要**：SenseVoice 必须使用 ONNX Runtime (INT8 量化) 部署，以确保：
1. 跨平台兼容性
2. 更低的显存占用
3. CPU 推理可用性

### 4.2 部署配置

```python
@dataclass
class SenseVoiceONNXConfig:
    """SenseVoice ONNX 配置"""
    model_path: str = "models/sensevoice_small_int8.onnx"
    use_gpu: bool = True                  # 优先使用 GPU
    fallback_to_cpu: bool = True          # GPU 不可用时回退 CPU
    num_threads: int = 4                  # CPU 推理线程数
    batch_size: int = 1                   # 批处理大小

    # INT8 量化参数
    quantization: str = "int8"            # int8 | fp16 | fp32

    # 推理优化
    enable_graph_optimization: bool = True
    optimization_level: int = 99          # ORT_ENABLE_ALL
```

### 4.3 服务实现

```python
import onnxruntime as ort
import numpy as np
from typing import List, Optional


class SenseVoiceONNXService:
    """
    SenseVoice ONNX 推理服务
    """

    def __init__(self, config: SenseVoiceONNXConfig):
        self.config = config
        self.session = None
        self.device = None

    def load_model(self) -> bool:
        """加载 ONNX 模型"""
        try:
            # 配置 SessionOptions
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_options.intra_op_num_threads = self.config.num_threads

            # 选择执行提供者
            providers = self._get_providers()

            self.session = ort.InferenceSession(
                self.config.model_path,
                sess_options=sess_options,
                providers=providers
            )

            self.device = self.session.get_providers()[0]
            return True

        except Exception as e:
            if self.config.fallback_to_cpu:
                return self._load_cpu_fallback()
            raise

    def _get_providers(self) -> List[str]:
        """获取执行提供者列表"""
        providers = []

        if self.config.use_gpu:
            # 检查 CUDA 可用性
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                providers.append('CUDAExecutionProvider')
            elif 'DmlExecutionProvider' in available:  # DirectML (Windows)
                providers.append('DmlExecutionProvider')

        providers.append('CPUExecutionProvider')
        return providers

    def _load_cpu_fallback(self) -> bool:
        """CPU 回退加载"""
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.config.num_threads

        self.session = ort.InferenceSession(
            self.config.model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        self.device = 'CPU'
        return True

    def transcribe(self, audio: np.ndarray) -> dict:
        """
        转录音频

        Args:
            audio: 16kHz float32 音频数组

        Returns:
            转录结果字典
        """
        # ONNX 推理...
        pass
```

---

## 五、硬件适配策略

### 5.1 硬件检测逻辑

```python
def detect_hardware_capability() -> dict:
    """
    检测硬件能力，决定功能配置

    Returns:
        {
            'has_gpu': bool,
            'gpu_memory_gb': float,
            'recommended_config': dict
        }
    """
    import torch

    result = {
        'has_gpu': False,
        'gpu_memory_gb': 0,
        'recommended_config': {}
    }

    if torch.cuda.is_available():
        result['has_gpu'] = True
        props = torch.cuda.get_device_properties(0)
        result['gpu_memory_gb'] = props.total_memory / (1024**3)

    # 生成推荐配置
    result['recommended_config'] = _generate_recommended_config(result)

    return result


def _generate_recommended_config(hardware: dict) -> dict:
    """根据硬件生成推荐配置"""

    if not hardware['has_gpu']:
        # 无 GPU 配置
        return {
            'sensevoice_device': 'cpu',
            'enable_demucs': False,  # 默认关闭人声分离
            'demucs_model': None,
            'max_concurrent_jobs': 1,
            'batch_size': 1,
            'note': '未检测到GPU，人声分离已禁用，转录使用CPU模式'
        }

    gpu_mem = hardware['gpu_memory_gb']

    if gpu_mem < 4:
        # 低显存配置 (< 4GB)
        return {
            'sensevoice_device': 'cuda',
            'enable_demucs': 'on_demand',  # 仅强BGM时启用
            'demucs_model': 'htdemucs',    # 最轻量模型
            'max_concurrent_jobs': 1,
            'batch_size': 1,
            'note': '低显存模式，仅在检测到强BGM时使用htdemucs'
        }

    elif gpu_mem < 8:
        # 中等显存配置 (4-8GB)
        return {
            'sensevoice_device': 'cuda',
            'enable_demucs': True,
            'demucs_model': 'htdemucs',
            'max_concurrent_jobs': 1,
            'batch_size': 2,
            'note': '标准配置，支持人声分离'
        }

    else:
        # 高显存配置 (>= 8GB)
        return {
            'sensevoice_device': 'cuda',
            'enable_demucs': True,
            'demucs_model': 'mdx_extra',   # 高质量模型
            'max_concurrent_jobs': 2,
            'batch_size': 4,
            'note': '高性能配置，支持并发处理'
        }
```

### 5.2 人声分离策略

| 硬件条件 | 默认行为 | BGM检测策略 |
|----------|----------|------------|
| 无 GPU | 禁用分离 | 仅频谱检测，不分离 |
| GPU < 4GB | 按需分离 | 检测到强BGM时用 htdemucs |
| GPU 4-8GB | 正常分离 | htdemucs 为主 |
| GPU >= 8GB | 高质量分离 | mdx_extra 可用 |

---

## 六、分阶段开发计划

### Phase 1: 基础能力搭建 (2-3天)

**目标**：实现 SenseVoice ONNX 服务和分句算法

#### 1.1 任务清单

| 任务 | 文件 | 优先级 |
|------|------|--------|
| SenseVoice ONNX 服务封装 | `sensevoice_onnx_service.py` | P0 |
| 分句算法实现 | `sentence_splitter.py` | P0 |
| 硬件检测模块 | `hardware_detector.py` | P1 |
| 配置数据类 | `sensevoice_models.py` | P0 |
| 基础单元测试 | `tests/test_sensevoice.py` | P1 |

#### 1.2 交付物

- [x] `backend/app/services/sensevoice_onnx_service.py` - ONNX 服务
- [x] `backend/app/services/sentence_splitter.py` - 分句算法
- [x] `backend/app/services/hardware_detector.py` - 硬件检测
- [x] `backend/app/models/sensevoice_models.py` - 数据模型
- [x] ONNX 模型下载脚本

#### 1.3 验收标准

- SenseVoice ONNX 可正常加载/卸载
- 分句算法可正确切分测试用例
- 硬件检测可正确识别 GPU 状态

---

### Phase 2: 智能熔断集成 (2-3天)

**目标**：实现频谱预判和熔断决策机制

#### 2.1 任务清单

| 任务 | 文件 | 优先级 |
|------|------|--------|
| 频谱指纹检测器 | `audio_circuit_breaker.py` | P0 |
| 熔断决策器 | `fuse_breaker.py` | P0 |
| 动态权重计算 | `config.py` (修改) | P1 |
| SSE 事件扩展 | `sse_service.py` (修改) | P1 |

#### 2.2 交付物

- [x] `backend/app/services/audio_circuit_breaker.py` - 频谱检测
- [x] `backend/app/services/fuse_breaker.py` - 熔断决策
- [x] 动态权重计算方法
- [x] 新增 SSE 事件类型

#### 2.3 验收标准

- 频谱预判可正确识别 BGM 片段
- 熔断决策逻辑正确（升级优先于复核）
- 动态权重可根据场景调整

---

### Phase 3: 转录服务重构 (3-4天)

**目标**：整合所有组件到转录服务

#### 3.1 任务清单

| 任务 | 文件 | 优先级 |
|------|------|--------|
| VAD 物理切分重构 | `transcription_service.py` | P0 |
| SenseVoice 转录流程 | `transcription_service.py` | P0 |
| 分句集成 | `transcription_service.py` | P0 |
| 熔断逻辑集成 | `transcription_service.py` | P0 |
| 句级 SSE 推送 | `transcription_service.py` | P1 |
| Whisper 复核机制 | `transcription_service.py` | P1 |
| 模型管理集成 | `model_preload_manager.py` | P1 |

#### 3.2 核心修改

```python
# transcription_service.py 主流程

async def _process_video_sensevoice(self, job: JobState):
    """SenseVoice 主处理流程"""

    # 1. 音频提取
    audio_path = await self._extract_audio(job)

    # 2. VAD 物理切分（15-30s chunks）
    vad_segments = await self._vad_physical_split(audio_path, job)

    # 3. 频谱预判
    circuit_decisions = await self._spectral_analysis(vad_segments, job)

    # 4. 按需人声分离
    processed_segments = await self._smart_separation(
        vad_segments, circuit_decisions, job
    )

    # 5. SenseVoice 转录 + 分句
    transcript_results = []
    for seg in processed_segments:
        # 转录
        raw_result = await self._transcribe_segment_sensevoice(seg, job)

        # 分句（关键！）
        sentences = self.sentence_splitter.split(raw_result.words)

        # 逐句 SSE 推送
        for sentence in sentences:
            await self._push_sse_sentence(job, sentence)
            transcript_results.append(sentence)

    # 6. 熔断检查 + Whisper 复核
    final_results = await self._fuse_and_retry(
        transcript_results, processed_segments, job
    )

    # 7. 生成字幕
    await self._generate_subtitle(final_results, job)
```

#### 3.3 交付物

- [x] 重构后的 `transcription_service.py`
- [x] VAD 物理切分逻辑
- [x] 句级 SSE 推送逻辑
- [x] 熔断+复核完整流程
- [x] 集成测试用例

#### 3.4 验收标准

- 完整流程可运行
- 字幕输出为句级粒度（非 VAD 粒度）
- 熔断机制正确触发
- 进度条合理（无停滞/回跳）

---

### Phase 4: 前端适配 (1-2天)

**目标**：更新前端界面支持新功能

#### 4.1 任务清单

| 任务 | 文件 | 优先级 |
|------|------|--------|
| 引擎选择器 | `TaskListView.vue` | P0 |
| 硬件状态显示 | `TaskListView.vue` | P1 |
| 实时字幕预览 | `TaskListView.vue` | P1 |
| 熔断状态指示 | `TaskListView.vue` | P2 |

#### 4.2 交付物

- [x] 引擎选择 UI
- [x] 硬件检测结果显示
- [x] 句级实时字幕预览
- [x] 前端 API 适配

---

### Phase 5: 测试与优化 (2-3天)

**目标**：全面测试和性能优化

#### 5.1 测试矩阵

| 场景 | 输入 | 预期结果 |
|------|------|----------|
| 纯净语音 | 无BGM视频 | 直接转录，无分离 |
| 轻度BGM | 背景音乐较轻 | 按需分离部分片段 |
| 重度BGM | 背景音乐较重 | 全面分离 |
| 低质量 | 噪音/模糊 | 触发熔断升级 |
| 无GPU | CPU only | 禁用分离，正常转录 |
| 低显存 | 4GB GPU | htdemucs 轻量分离 |

#### 5.2 性能指标

| 指标 | 目标值 |
|------|--------|
| 10分钟视频处理时间 | < 2分钟 (GPU) |
| 句级平均长度 | 10-20字 |
| 置信度准确率 | > 85% |
| 显存峰值 (8GB) | < 6GB |

---

## 七、文件变更清单

### 7.1 新增文件

| 路径 | 说明 |
|------|------|
| `backend/app/services/sensevoice_onnx_service.py` | ONNX 服务封装 |
| `backend/app/services/sentence_splitter.py` | 分句算法 |
| `backend/app/services/audio_circuit_breaker.py` | 频谱预判 |
| `backend/app/services/fuse_breaker.py` | 熔断决策器 |
| `backend/app/services/hardware_detector.py` | 硬件检测 |
| `backend/app/models/sensevoice_models.py` | 数据模型 |
| `tests/test_sensevoice_integration.py` | 集成测试 |

### 7.2 修改文件

| 路径 | 变更内容 |
|------|----------|
| `backend/app/services/transcription_service.py` | 主流程重构 |
| `backend/app/services/model_preload_manager.py` | SenseVoice 缓存 |
| `backend/app/core/config.py` | 动态权重配置 |
| `frontend/src/views/TaskListView.vue` | 引擎选择 UI |
| `requirements.txt` | 新增 onnxruntime |

---

## 八、里程碑与验收标准

### Milestone 1: 基础能力 (Phase 1-2)

- [ ] SenseVoice ONNX 可正常加载/推理
- [ ] 分句算法输出符合预期
- [ ] 频谱预判可正确识别 BGM

### Milestone 2: 核心功能 (Phase 3)

- [ ] 完整转录流程可用
- [ ] 字幕输出为句级粒度
- [ ] 熔断机制正确工作
- [ ] 进度条无停滞/回跳

### Milestone 3: 完整交付 (Phase 4-5)

- [ ] 前端 UI 完整可用
- [ ] 所有测试场景通过
- [ ] 性能指标达标

---

## 九、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| ONNX 模型转换失败 | 高 | 准备 PyTorch fallback |
| 分句算法不准确 | 中 | 预留参数调优空间 |
| 熔断逻辑过于敏感 | 中 | 可配置阈值 |
| 显存管理复杂 | 中 | 严格串行化模型加载 |

---

## 十、附录

### A. ONNX 模型转换命令

```bash
# 导出 SenseVoice 为 ONNX
python -m funasr.export \
    --model_name="iic/SenseVoiceSmall" \
    --export_dir="./models" \
    --quantize=int8
```

### B. 分句算法测试用例

```python
# 测试用例
test_cases = [
    # 标点切分
    ("你好。世界。", ["你好。", "世界。"]),
    # 停顿切分
    ([{"word": "你", "end": 0.5}, {"word": "好", "end": 1.5}], ["你", "好"]),
    # 长度切分
    ("一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十一", [...]),
]
```

### C. 熔断状态机

```
                    +---------+
                    |  INIT   |
                    +---------+
                         |
                         v
                    +---------+
                    | DETECT  |
                    +---------+
                    /    |    \
                   v     v     v
            +-------+ +-------+ +-------+
            | CLEAN | | LIGHT | | HEAVY |
            +-------+ +-------+ +-------+
                |         |         |
                v         v         v
            +-------+ +-------+ +-------+
            |TRANSCR| |SEP+TR | |SEP+TR |
            +-------+ +-------+ +-------+
                |         |         |
                +----+----+----+----+
                     |
                     v
               +-----------+
               | EVALUATE  |
               +-----------+
              /      |      \
             v       v       v
         +------+ +------+ +--------+
         |ACCEPT| |RETRY | |UPGRADE |
         +------+ +------+ +--------+
```

---

## 十一、总结

本计划解决了现有设计的四个核心问题：

1. **进度条僵化** → 动态权重调整
2. **句级切分缺失** → 分句算法
3. **熔断机制割裂** → 协调联动的熔断决策器
4. **VAD 定位模糊** → 物理切分 vs 语义切分分离

关键技术决策：

- **SenseVoice 必须使用 ONNX Runtime (INT8 量化)**
- **无 GPU 时默认关闭人声分离**
- **熔断升级优先于 Whisper 复核**
- **VAD 负责物理切分，分句算法负责语义切分**

实施建议：按照 Phase 1 → Phase 5 顺序开发，每阶段充分测试后再进入下一阶段。
