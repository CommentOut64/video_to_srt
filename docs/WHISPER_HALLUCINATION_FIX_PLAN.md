# Whisper 幻觉抑制与双流稳定化实施方案 V3.1.0

> 本文档基于项目实际代码结构细化，可直接执行。
> **实施状态: 已完成** (2024-12-13)
> **更新日志**: V3.1.0 新增 Whisper Buffer Pool 方案，支持 DEEP_LISTEN 模式

## 目录

1. [问题背景](#1-问题背景)
2. [架构设计](#2-架构设计)
3. [第一道防线：音频切片重叠](#3-第一道防线音频切片重叠)
4. [第二道防线：推理层静态抑制](#4-第二道防线推理层静态抑制)
5. [第三道防线：双流校验与置信度门控](#5-第三道防线双流校验与置信度门控)
6. [Whisper Buffer Pool 方案](#6-whisper-buffer-pool-方案)
7. [开发任务清单](#7-开发任务清单)
8. [测试验证](#8-测试验证)

---

## 1. 问题背景

### 1.1 现象描述

在双流架构（SenseVoice + Whisper）中，Whisper 在以下场景容易产生幻觉：

| 问题现象 | 具体表现 | 根因分析 |
|---------|---------|---------|
| 静音段幻觉 | 输出 `Questions 19...`, `Subtitles by...` | Whisper 自回归特性，缺乏上下文时"强行"生成 |
| 短切片幻觉 | 输出 `______`, `...` | VAD 切片过短（<1s），Whisper 无法正确解码 |
| 句中停顿 | 强行补全无关句子 | 缺乏前序音频的"预热"上下文 |
| 重复提示词 | 照抄 initial_prompt 内容 | 短音频 + 强 prompt 导致提示词增益失控 |

### 1.2 当前项目状态

**已有防御机制**（`transcription_service.py:4438-4461`）：
- 下划线比例检测（>30% 回退）
- 提示词重复检测（overlap_ratio > 80% 回退）

**不足之处**：
- 防御发生在 Whisper 推理**之后**，已消耗计算资源
- 缺乏输入层预处理
- 缺乏解码层物理封锁
- 双流对比逻辑不完善

---

## 2. 架构设计

采用 **"防御纵深" (Defense in Depth)** 策略：

```
┌──────────────────────────────────────────────────────────────┐
│                        音频输入                               │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  【第一道防线】音频切片重叠 (Input Overlap)                     │
│  - 位置: chunk_engine.py / transcription_service.py          │
│  - 动作: VAD 切片前向扩展 0.5s，消除"极短切片"诱因              │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  【第二道防线】推理层静态抑制 (Inference Suppression)           │
│  - 位置: whisper_service.py                                  │
│  - 动作: suppress_tokens 物理封锁已知幻觉词 Token               │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  【第三道防线】双流校验与置信度门控 (Dual-Stream Gating)         │
│  - 位置: transcription_service.py                            │
│  - 动作: avg_logprob + no_speech_prob + 双流长度对比           │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 第一道防线：音频切片重叠

### 3.1 原理

Whisper 是自回归模型，需要前序音频来"预热"注意力机制。对于 VAD 切出的短片段，如果直接送入会缺乏上下文。

### 3.2 修改位置

**文件**: `backend/app/services/transcription_service.py`

**目标方法**: `_whisper_text_patch_with_arbitration()` 和 `_whisper_text_patch()`

### 3.3 修改内容

```python
# ========== 修改前 (transcription_service.py:4417-4421) ==========
# 提取对应时间段的音频
sr = 16000
start_sample = int(sentence.start * sr)
end_sample = int(sentence.end * sr)
audio_segment = audio_array[start_sample:end_sample]

# ========== 修改后 ==========
# 提取对应时间段的音频（增加前向重叠，为 Whisper 提供上下文预热）
sr = 16000
WHISPER_OVERLAP_SEC = 0.5  # Whisper 上下文重叠时长（秒）

# 计算重叠后的起始位置（不小于0）
overlap_start = max(0.0, sentence.start - WHISPER_OVERLAP_SEC)
start_sample = int(overlap_start * sr)
end_sample = int(sentence.end * sr)
audio_segment = audio_array[start_sample:end_sample]

# 记录日志（调试用）
if overlap_start < sentence.start:
    self.logger.debug(
        f"Whisper 补刀添加 {sentence.start - overlap_start:.2f}s 前向重叠: "
        f"[{overlap_start:.2f}s, {sentence.end:.2f}s]"
    )
```

### 3.4 注意事项

- **不修改 VAD 原始时间戳**：重叠仅用于 Whisper 推理，对齐阶段会自动处理多余内容
- **首个 Chunk 特殊处理**：当 `sentence.start == 0` 时，无需前推
- **时间戳保持不变**：SenseVoice 确定的时间轴不可变，只是给 Whisper "听更多"

---

## 4. 第二道防线：推理层静态抑制

### 4.1 原理

通过 `suppress_tokens` 参数在 Beam Search 阶段直接禁止生成特定 Token，从模型解码层面"物理封锁"已知的幻觉词汇。

### 4.2 步骤一：获取 Token ID

**运行脚本**:
```bash
python scripts/extract_hallucination_tokens.py --model medium
```

**预期输出示例** (Token ID 需以实际运行结果为准):
```
'_'                       | [62]                           | 62       | 单个下划线
'Questions'               | [15048]                        | 15048    | Questions 幻觉
'Subtitles'               | [3735, 30909]                  | 3735     | Subtitles 幻觉
...
```

### 4.3 步骤二：更新配置文件

**文件**: `backend/app/config/model_config.py`

```python
# ========== 新增内容 ==========

# Whisper 幻觉抑制 Token ID 配置
# 通过 scripts/extract_hallucination_tokens.py 生成
# 注意: 不同模型的 Token ID 可能不同，需分别配置

WHISPER_SUPPRESS_TOKENS = {
    # Whisper Medium 模型的幻觉 Token ID
    # 运行 `python scripts/extract_hallucination_tokens.py --model medium` 获取
    "medium": [
        # 下划线类 (最常见的幻觉)
        62,     # '_' 单个下划线
        # YouTube 风格幻觉
        # ... (根据脚本输出填充)
    ],

    # Whisper Large-v3 模型的幻觉 Token ID (未来扩展)
    # 运行 `python scripts/extract_hallucination_tokens.py --model large-v3` 获取
    "large-v3": [
        # ... (根据脚本输出填充)
    ],
}


def get_whisper_suppress_tokens(model_name: str) -> list:
    """
    获取指定模型的幻觉抑制 Token ID 列表

    Args:
        model_name: 模型名称 (如 "medium", "large-v3")

    Returns:
        list: Token ID 列表，用于 suppress_tokens 参数
    """
    for key, tokens in WHISPER_SUPPRESS_TOKENS.items():
        if key in model_name.lower():
            return tokens
    return []
```

### 4.4 步骤三：修改 WhisperService

**文件**: `backend/app/services/whisper_service.py`

**修改 `transcribe()` 方法** (约第 393-445 行):

```python
# ========== 修改前 ==========
def transcribe(
    self,
    audio: Union[str, np.ndarray],
    language: str = None,
    initial_prompt: str = None,
    word_timestamps: bool = False,
    beam_size: int = 5,
    vad_filter: bool = True,
    vad_parameters: dict = None,
    temperature: float = 0.0,
    condition_on_previous_text: bool = True
) -> Dict[str, Any]:

# ========== 修改后 ==========
def transcribe(
    self,
    audio: Union[str, np.ndarray],
    language: str = None,
    initial_prompt: str = None,
    word_timestamps: bool = False,
    beam_size: int = 5,
    vad_filter: bool = True,
    vad_parameters: dict = None,
    temperature: float = 0.0,
    condition_on_previous_text: bool = True,
    suppress_tokens: list = None  # 新增参数
) -> Dict[str, Any]:
```

**修改 `transcribe()` 方法内部** (约第 434-445 行):

```python
# ========== 修改前 ==========
# 执行转录
segments_generator, info = self.model.transcribe(
    audio,
    language=language,
    initial_prompt=initial_prompt,
    word_timestamps=word_timestamps,
    beam_size=beam_size,
    vad_filter=vad_filter,
    vad_parameters=vad_parameters,
    temperature=temperature,
    condition_on_previous_text=condition_on_previous_text
)

# ========== 修改后 ==========
# 获取幻觉抑制 Token ID（如果未指定）
if suppress_tokens is None:
    from app.config.model_config import get_whisper_suppress_tokens
    suppress_tokens = get_whisper_suppress_tokens(self._model_name)
    if suppress_tokens:
        logger.debug(f"启用幻觉抑制: {len(suppress_tokens)} 个 Token ID")

# 执行转录
segments_generator, info = self.model.transcribe(
    audio,
    language=language,
    initial_prompt=initial_prompt,
    word_timestamps=word_timestamps,
    beam_size=beam_size,
    vad_filter=vad_filter,
    vad_parameters=vad_parameters,
    temperature=temperature,
    condition_on_previous_text=condition_on_previous_text,
    suppress_tokens=suppress_tokens if suppress_tokens else None  # 幻觉抑制
)
```

### 4.5 注意事项

- **下划线 `_` 是核心封杀目标**：绝大多数下划线幻觉由单个下划线 Token 重复生成
- **不封杀常见词汇**：如 "the", "a" 等，即使它们偶尔出现在幻觉中
- **分模型配置**：Medium 和 Large-v3 的 Token ID 可能不同

---

## 5. 第三道防线：双流校验与置信度门控

### 5.1 原理

利用 SenseVoice（非自回归，极少幻觉）的结果来验证 Whisper（自回归，易幻觉）的可靠性。

### 5.2 修改位置

**文件**: `backend/app/services/transcription_service.py`

**目标方法**: `_whisper_text_patch_with_arbitration()` (约第 4380 行)

### 5.3 步骤一：升级 TextNormalizer

**文件**: `backend/app/services/text_normalizer.py`

```python
# ========== 新增幻觉检测正则 ==========

import re

class TextNormalizer:
    """SenseVoice 文本后处理器"""

    # ... 现有属性 ...

    # 【新增】Whisper 幻觉检测模式
    # 重复子串检测: 长度>=4 且重复>=3次的子串
    REPEATED_PATTERN = re.compile(r'(.{4,})\1{2,}')

    # 特定幻觉句式（开头匹配）
    HALLUCINATION_PATTERNS = [
        re.compile(r'^Questions?\s+\d+', re.IGNORECASE),           # "Questions 19..."
        re.compile(r'^Subtitles?\s+by', re.IGNORECASE),            # "Subtitles by..."
        re.compile(r'^Copyright\s+', re.IGNORECASE),               # "Copyright 2024..."
        re.compile(r'^Thanks?\s+for\s+watching', re.IGNORECASE),   # "Thanks for watching"
        re.compile(r'^Please\s+subscribe', re.IGNORECASE),         # "Please subscribe"
    ]

    @classmethod
    def is_whisper_hallucination(cls, text: str) -> bool:
        """
        检测 Whisper 输出是否为幻觉文本

        Args:
            text: Whisper 输出的文本

        Returns:
            bool: True 表示检测到幻觉
        """
        if not text:
            return False

        text = text.strip()

        # 检测1: 重复子串模式
        if cls.REPEATED_PATTERN.search(text):
            return True

        # 检测2: 特定幻觉句式
        for pattern in cls.HALLUCINATION_PATTERNS:
            if pattern.match(text):
                return True

        # 检测3: 纯下划线/符号（清洗后为空）
        cleaned = cls.clean(text)
        if not cleaned or len(cleaned) < len(text) * 0.3:
            # 原文很长，但清洗后几乎没了，说明都是垃圾字符
            return True

        return False

    @classmethod
    def clean_whisper_output(cls, text: str) -> str:
        """
        清洗 Whisper 输出（比 SenseVoice 清洗更激进）

        Args:
            text: Whisper 原始输出

        Returns:
            str: 清洗后的文本，如果是幻觉则返回空字符串
        """
        if not text:
            return ""

        # 先用基础清洗
        cleaned = cls.clean(text)

        # 再检测是否为幻觉
        if cls.is_whisper_hallucination(text):
            return ""

        return cleaned
```

### 5.4 步骤二：升级置信度检查

**文件**: `backend/app/services/transcription_service.py`

**修改方法**: `_whisper_text_patch_with_arbitration()` (约第 4430 行后)

```python
# ========== 在 whisper_text = result.get('text', '').strip() 之后添加 ==========

# === 第三道防线: 置信度门控 ===
segments = result.get('segments', [])

# A. avg_logprob 检查（模型在瞎猜）
if segments:
    avg_logprob = sum(s.get('avg_logprob', -0.5) for s in segments) / len(segments)
    avg_no_speech = sum(s.get('no_speech_prob', 0.0) for s in segments) / len(segments)

    # 熔断条件1: avg_logprob 过低（模型不确信）
    if avg_logprob < -1.0:
        self.logger.warning(
            f"Whisper 熔断(avg_logprob={avg_logprob:.2f} < -1.0): "
            f"'{whisper_text[:50]}...', 回退到 SenseVoice"
        )
        return sentence

    # 熔断条件2: no_speech_prob 高但仍输出文本（静音段被强行翻译）
    if avg_no_speech > 0.6 and whisper_text:
        self.logger.warning(
            f"Whisper 熔断(no_speech={avg_no_speech:.2f} > 0.6): "
            f"'{whisper_text[:50]}...', 回退到 SenseVoice"
        )
        return sentence

# B. 幻觉正则检测（使用升级后的 TextNormalizer）
from app.services.text_normalizer import TextNormalizer

if TextNormalizer.is_whisper_hallucination(whisper_text):
    self.logger.warning(
        f"Whisper 熔断(幻觉检测): '{whisper_text[:50]}...', 回退到 SenseVoice"
    )
    return sentence

# C. 双流长度/内容对比（关键创新）
sensevoice_text = sentence.text_clean or sentence.text or ""
if sensevoice_text:
    len_sv = len(sensevoice_text)
    len_w = len(whisper_text)

    # 长度暴涨检测: Whisper 输出远超 SenseVoice
    # 公式: len(whisper) > 3 * len(sensevoice) + 10
    if len_w > 3 * len_sv + 10:
        # 特权放行: 如果 Whisper 置信度极高，可能是 SenseVoice 漏识别
        if avg_logprob > -0.5:
            self.logger.info(
                f"Whisper 长度暴涨但置信度高(logprob={avg_logprob:.2f}), 特权放行"
            )
        else:
            self.logger.warning(
                f"Whisper 熔断(长度暴涨 {len_w} > 3*{len_sv}+10): "
                f"'{whisper_text[:50]}...', 回退到 SenseVoice"
            )
            return sentence
```

### 5.5 SenseVoice 为空时的特权机制

```python
# ========== 在双流对比逻辑中添加 ==========

# 特殊情况: SenseVoice 为空，但 Whisper 有输出
if not sensevoice_text and whisper_text:
    # 如果 Whisper 置信度极高，给予特权放行
    if segments and avg_logprob > -0.5:
        self.logger.info(
            f"SenseVoice 为空，Whisper 高置信度(logprob={avg_logprob:.2f})特权放行: "
            f"'{whisper_text[:50]}...'"
        )
        # 继续正常流程，采纳 Whisper 结果
    else:
        # Whisper 置信度不够高，保守起见不采纳
        self.logger.warning(
            f"SenseVoice 为空，Whisper 置信度不足(logprob={avg_logprob:.2f}): "
            f"'{whisper_text[:50]}...', 保持 SenseVoice 空结果"
        )
        return sentence
```

---

## 6. Whisper Buffer Pool 方案

### 6.1 方案概述

**核心思想**: 累积多个短 Chunk，拼接后一次性送入 Whisper 推理，利用 Whisper 的长上下文能力从根源消除短音频幻觉。

**适用模式**: `DEEP_LISTEN` (高保真精听模式)

**优势对比**:

| 方案 | 适用场景 | 延迟 | 幻觉抑制能力 | GPU 利用率 |
|------|---------|------|-------------|-----------|
| 逐句补刀 (SMART_PATCH) | 快速处理 | 低 | 中等 | 低 |
| 缓冲池批量 (DEEP_LISTEN) | 高质量需求 | 较高 | 强 | 高 |

### 6.2 架构设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Whisper Buffer Pool                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐                           │
│  │ Chunk 1 │ + │ Chunk 2 │ + │ Chunk 3 │  ──────► 累积缓冲区       │
│  │  2.3s   │   │  1.8s   │   │  2.1s   │          (>5s 触发)       │
│  └─────────┘   └─────────┘   └─────────┘                           │
│                      │                                              │
│                      ▼                                              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  拼接音频 (6.2s) ──────► Whisper 单次推理 ──────► 长文本输出  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                      │                                              │
│                      ▼                                              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │            WhisperBufferAligner: 长文本回填对齐              │  │
│  │  "你好世界欢迎观看" ──► ["你好世界", "欢迎观看"]              │  │
│  │   Chunk 1-2            Chunk 1      Chunk 2                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.3 核心类设计

**文件**: `backend/app/services/whisper_buffer_pool.py`

```python
@dataclass
class WhisperBufferConfig:
    """缓冲池配置"""
    min_duration_sec: float = 5.0    # 累积 5s 后触发
    max_chunk_count: int = 3         # 或累积 3 个 Chunk 后触发
    silence_trigger_sec: float = 1.0 # 长静音也触发
    max_duration_sec: float = 30.0   # 安全上限


class WhisperBufferPool:
    """缓冲池管理器"""
    def add_chunk(index, start, end, audio, sv_text, sv_conf): ...
    def should_trigger(has_long_silence, is_eof) -> bool: ...
    def get_buffer_audio() -> np.ndarray: ...


class WhisperBufferAligner:
    """长文本回填对齐器"""
    def align_to_chunks(whisper_text, chunks) -> List[AlignedResult]: ...
    # 支持 word-level 和 segment-level 对齐


class WhisperBufferService:
    """服务层: 编排 Pool + Aligner"""
    def process_buffer(whisper_service, language, prompt) -> List[dict]: ...
    def flush_remaining(...) -> List[dict]: ...
```

### 6.4 触发条件

缓冲池在以下任一条件满足时触发处理:

1. **时长触发**: 累积音频 >= `min_duration_sec` (默认 5s)
2. **数量触发**: 累积 Chunk 数 >= `max_chunk_count` (默认 3 个)
3. **静音触发**: 与下一个 Chunk 间隔 > `silence_trigger_sec` (默认 1s)
4. **EOF 触发**: 音频结束时强制清空

### 6.5 长文本回填对齐算法

**问题**: Whisper 输出的是拼接后的完整文本，需要映射回各个原始 Chunk。

**方案**: 基于 SenseVoice 文本的模糊匹配 + 比例分配

```python
def align_to_chunks(self, whisper_text: str, chunks: List[BufferChunk]) -> List[dict]:
    """
    将 Whisper 长文本对齐到各个 Chunk

    算法:
    1. 尝试 word-level 匹配 (如果 Whisper 返回 word_timestamps)
    2. 回退到 segment-level 分割 (基于标点)
    3. 最终回退: 按字数比例分配
    """
    # 策略1: Word-level 精确对齐
    if word_timestamps:
        return self._align_by_words(whisper_words, chunks)

    # 策略2: Segment-level 分割
    if self._has_punctuation(whisper_text):
        return self._align_by_segments(whisper_text, chunks)

    # 策略3: 比例分配 (兜底)
    return self._align_by_ratio(whisper_text, chunks)
```

### 6.6 集成方式

**文件**: `backend/app/services/transcription_service.py`

**入口方法**: `_whisper_buffer_pool_process()`

```python
async def _whisper_buffer_pool_process(
    self,
    patch_queue: List[Dict],
    audio_array: np.ndarray,
    job: 'JobState',
    subtitle_manager: 'StreamingSubtitleManager',
    progress_tracker
):
    """
    使用 Whisper 缓冲池批量处理句子（DEEP_LISTEN 模式）
    """
    buffer_service = WhisperBufferService(WhisperBufferConfig())

    for idx, item in enumerate(patch_queue):
        # 添加到缓冲池
        buffer_service.add_chunk(...)

        # 检查触发条件
        if buffer_service.should_trigger(...):
            # 批量处理
            aligned_results = buffer_service.process_buffer(...)

            # 应用结果
            for result in aligned_results:
                subtitle_manager.update_sentence(...)
```

**模式路由** (在 `_post_process_enhancement()` 中):

```python
if solution_config.enhancement == EnhancementMode.DEEP_LISTEN:
    # 使用缓冲池批量处理
    await self._whisper_buffer_pool_process(...)
else:
    # SMART_PATCH: 逐句处理
    for item in patch_queue:
        await self._whisper_text_patch_with_arbitration(...)
```

### 6.7 与三道防线的协同

Buffer Pool 方案与三道防线**并非互斥**，而是**协同工作**:

| 防线 | Buffer Pool 模式下的作用 |
|------|-------------------------|
| 第一道 (Overlap) | 每个 Chunk 添加到缓冲池前已应用 0.5s 重叠 |
| 第二道 (suppress_tokens) | Whisper 推理时仍生效，物理封锁幻觉 Token |
| 第三道 (双流校验) | 对齐后的结果仍经过幻觉正则检测 + 长度对比 |

**额外优势**: 长上下文推理本身就能大幅降低幻觉概率，三道防线作为"安全网"。

---

## 7. 开发任务清单

### 7.1 Backend 开发

| 序号 | 任务 | 文件 | 优先级 | 状态 |
|-----|------|------|-------|------|
| 1 | 运行 Token ID 提取脚本 | `scripts/extract_hallucination_tokens.py` | P0 | **已完成** |
| 2 | 更新 model_config.py 添加 SUPPRESS_TOKENS | `backend/app/config/model_config.py` | P0 | **已完成** |
| 3 | 修改 WhisperService.transcribe() 注入 suppress_tokens | `backend/app/services/whisper_service.py` | P0 | **已完成** |
| 4 | 添加音频切片重叠逻辑 | `backend/app/services/transcription_service.py` | P1 | **已完成** |
| 5 | 升级 TextNormalizer 添加幻觉检测 | `backend/app/services/text_normalizer.py` | P1 | **已完成** |
| 6 | 增强 _whisper_text_patch_with_arbitration() | `backend/app/services/transcription_service.py` | P1 | **已完成** |
| 7 | 实现 WhisperBufferPool 类 | `backend/app/services/whisper_buffer_pool.py` | P0 | **已完成** |
| 8 | 集成 Buffer Pool 到 transcription_service | `backend/app/services/transcription_service.py` | P0 | **已完成** |

### 7.2 已完成的代码变更

```
scripts/
└── extract_hallucination_tokens.py    [新增] Token ID 提取脚本

backend/app/config/
└── model_config.py                    [修改] 添加 WHISPER_SUPPRESS_TOKENS

backend/app/services/
├── whisper_service.py                 [修改] transcribe() 支持 suppress_tokens
├── text_normalizer.py                 [修改] 添加 is_whisper_hallucination()
├── transcription_service.py           [修改] 0.5s 重叠 + 三道防线 + Buffer Pool 集成
└── whisper_buffer_pool.py             [新增] 缓冲池完整实现
```

### 7.3 配置的 Token ID (Whisper Medium)

```python
WHISPER_SUPPRESS_TOKENS = {
    "medium": [
        62,      # '_' 单个下划线
        10852,   # '__' 双下划线
        23757,   # '____' 四下划线
        485,     # '...' 省略号
        353,     # '..' 双点
        27738,   # ' Questions'
        8511,    # ' Subtitles'
        25653,   # ' Copyright'
        27917,   # 'Thanks for watching'
        16216,   # 'Please subscribe'
        2012,    # ' Amara'
        3961,    # 音乐符号
    ],
}

---

## 8. 测试验证

### 8.1 单元测试

**测试文件**: `backend/tests/test_whisper_hallucination.py`

```python
import pytest
from app.services.text_normalizer import TextNormalizer


class TestHallucinationDetection:
    """幻觉检测测试"""

    def test_underscore_hallucination(self):
        """下划线幻觉检测"""
        assert TextNormalizer.is_whisper_hallucination("______") == True
        assert TextNormalizer.is_whisper_hallucination("___text___") == True

    def test_questions_hallucination(self):
        """Questions 幻觉检测"""
        assert TextNormalizer.is_whisper_hallucination("Questions 19 through 25") == True
        assert TextNormalizer.is_whisper_hallucination("Question 1: What is...") == True

    def test_subtitles_hallucination(self):
        """Subtitles by 幻觉检测"""
        assert TextNormalizer.is_whisper_hallucination("Subtitles by Amara.org") == True

    def test_repeated_pattern(self):
        """重复模式检测"""
        assert TextNormalizer.is_whisper_hallucination("hahahahahahaha") == True
        assert TextNormalizer.is_whisper_hallucination("no no no no no no") == True

    def test_normal_text(self):
        """正常文本不应被误判"""
        assert TextNormalizer.is_whisper_hallucination("Hello, how are you?") == False
        assert TextNormalizer.is_whisper_hallucination("This is a normal sentence.") == False
        assert TextNormalizer.is_whisper_hallucination("Okay.") == False


class TestWhisperOverlap:
    """音频重叠测试"""

    def test_overlap_calculation(self):
        """重叠计算测试"""
        OVERLAP_SEC = 0.5

        # 正常情况
        start = 5.0
        overlap_start = max(0.0, start - OVERLAP_SEC)
        assert overlap_start == 4.5

        # 边界情况（首个 Chunk）
        start = 0.2
        overlap_start = max(0.0, start - OVERLAP_SEC)
        assert overlap_start == 0.0  # 不能小于 0
```

### 8.2 集成测试

```bash
# 1. 静音音频测试
# 期望: Whisper 不输出文本或输出被熔断

# 2. 短切片测试（<1s 的 VAD 切片）
# 期望: 不出现 ______ 或 Questions

# 3. 正常短句测试
# 期望: "Okay." 不被误杀

# 4. 长句停顿测试
# 期望: 句中停顿不产生额外幻觉
```

### 8.3 回归测试

确保以下场景不受影响：
- 正常中文转录
- 正常英文转录
- 混合语言转录
- 带 BGM 的音频

---

## 附录 A: 完整代码修改清单

### A.1 model_config.py 完整修改

```python
# backend/app/config/model_config.py

"""
模型预加载配置文件
"""

import os
from typing import List

# ... 现有代码 ...


# ========== 新增: Whisper 幻觉抑制配置 ==========

# Whisper 幻觉抑制 Token ID 配置
# 通过 scripts/extract_hallucination_tokens.py 生成
WHISPER_SUPPRESS_TOKENS = {
    # Whisper Medium 模型
    "medium": [
        # TODO: 运行脚本后填入实际 Token ID
        # 62,     # '_' 单个下划线
        # ...
    ],

    # Whisper Large-v3 模型 (未来扩展)
    "large-v3": [
        # TODO: 运行脚本后填入实际 Token ID
    ],
}


def get_whisper_suppress_tokens(model_name: str) -> list:
    """
    获取指定模型的幻觉抑制 Token ID 列表

    Args:
        model_name: 模型名称 (如 "medium", "large-v3")

    Returns:
        list: Token ID 列表，用于 suppress_tokens 参数
    """
    model_name_lower = model_name.lower()
    for key, tokens in WHISPER_SUPPRESS_TOKENS.items():
        if key in model_name_lower:
            return tokens
    return []
```

### A.2 whisper_service.py 完整修改

详见 [第4.4节](#44-步骤三修改-whisperservice)

### A.3 text_normalizer.py 完整修改

详见 [第5.3节](#53-步骤一升级-textnormalizer)

### A.4 transcription_service.py 完整修改

详见 [第3.3节](#33-修改内容) 和 [第5.4节](#54-步骤二升级置信度检查)

---

## 附录 B: 预期效果

| 问题现象 | 原始表现 | 修复后表现 | 触发机制 |
|---------|---------|-----------|---------|
| 静音段 | `Questions 19...` | 不输出 / 回退空 | `no_speech_prob` + 正则 |
| 短切片 | `______` | 不输出 / 回退 SV | `suppress_tokens` + 正则 |
| 句中停顿 | 强行补全无关句 | 正确拼接 / 回退 SV | Overlap + 双流对比 |
| 极低置信度 | 乱码 | 回退 SV | `avg_logprob` 阈值 |

---

## 附录 C: 维护指南

### C.1 发现新幻觉模式

1. **优先**: 添加到 `TextNormalizer.HALLUCINATION_PATTERNS`（不用重启模型）
2. **其次**: 添加到 `WHISPER_SUPPRESS_TOKENS`（需要重启模型）

### C.2 误杀排查

如果正常文本被误杀：
1. 检查 `is_whisper_hallucination()` 的正则是否过于激进
2. 检查双流长度对比的阈值（`3 * len + 10`）是否需要调整
3. 检查 `avg_logprob` 阈值（`-1.0`）是否需要放宽

### C.3 Token ID 更新

当更换 Whisper 模型版本时：
```bash
python scripts/extract_hallucination_tokens.py --model <新模型名>
```
然后更新 `model_config.py` 中对应的 Token ID 列表。
