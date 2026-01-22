# Phase 3: 转录服务重构（修订版 v2.1）

> 目标：整合所有组件到转录服务，实现完整的时空解耦转录流程
>
> 工期：3-4天
>
> 版本更新：整合 [06_转录层深度优化_时空解耦架构](./06_转录层深度优化_时空解耦架构.md) 设计

---

## ⚠️ 重要修订

### v2.1 新增（VAD优先 + 频谱分诊 + 概念澄清）

- ✅ **流程确认**：`音频提取 → VAD → 频谱分诊(Chunk级) → 按需分离 → 转录`
- ✅ **组件更新**：使用 `AudioSpectrumClassifier` 替代原有 `AudioCircuitBreaker`
- ✅ **新增**：Chunk级频谱分诊，每个VAD片段独立判断是否需要分离
- ✅ **概念澄清**：熔断 ≠ Whisper复核（详见下方说明）
- ✅ **新增**：熔断回溯机制（保留原始音频，升级分离模型）
- ✅ **新增**：后处理增强体系（Whisper复核、LLM校对、翻译）

### v2.0 新增（时空解耦架构）

- ✅ **核心修改**：Whisper 复核**仅取文本**，时间戳由 SenseVoice 确定
- ✅ **新增**：伪对齐算法集成（Phase 1 的 `pseudo_alignment.py`）
- ✅ **新增**：字幕流式输出系统 `streaming_subtitle.py`
- ✅ **新增**：智能进度追踪集成（Phase 1 的 `progress_tracker.py`）
- ✅ **新增**：统一 SSE 事件 Tag 设计

### v1.0 基础修订

- ❌ **删除**：不使用 `hardware_detector.py`
- ✅ **复用**：使用现有的 `hardware_service.py`
- ✅ **修正**：SSE 推送使用 `get_sse_manager()` 和 `broadcast_sync()`

---

## 一、任务清单（扩展）

| 任务 | 文件 | 优先级 | 状态 |
|------|------|--------|------|
| VAD 物理切分重构 | `transcription_service.py` | P0 | 修改 |
| **频谱分诊集成** | `transcription_service.py` | **P0** | **修改** |
| **时空解耦复核逻辑** | `transcription_service.py` | **P0** | **修改** |
| **流式字幕输出系统** | `streaming_subtitle.py` | **P0** | **新建** |
| 时间戳偏移修正 | `transcription_service.py` | P0 | 修改 |
| 熔断逻辑集成 | `transcription_service.py` | P0 | 修改 |
| 句级 SSE 推送 | `transcription_service.py` | P1 | 修改 |
| 模型管理集成 | `model_preload_manager.py` | P1 | 修改 |

---

## 二、整体流程（VAD优先 + 时空解耦版 v2.1）

### 2.1 概念澄清：熔断 vs 后处理增强

> **重要**：熔断和后处理增强是两个不同的概念，发生在不同阶段

| 概念 | 触发条件 | 发生时机 | 动作 |
|------|----------|----------|------|
| **熔断** | BGM/Noise标签 + 低置信度 | 单个Chunk转录后立即判断 | 回溯到原始音频，升级分离模型重做 |
| **后处理增强** | 用户配置启用 + 低置信度 | 所有Chunk转录完成后 | Whisper复核、LLM校对、翻译 |

### 2.2 完整流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    转录层完整流程（v2.1）                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 音频提取                                                     │
│     ↓                                                           │
│  2. VAD 物理切分（15-30s chunks）                                │
│     ↓                                                           │
│  3. 频谱分诊（AudioSpectrumClassifier，Chunk级别）              │
│     │  - 提取频谱特征（ZCR、谱质心、谐波比、能量方差）          │
│     │  - 判断：CLEAN / MUSIC / NOISE / MIXED                    │
│     │  - 推荐分离模型：htdemucs / mdx_extra / None              │
│     ↓                                                           │
│  4. 按需人声分离（仅对需要分离的Chunk）                         │
│     │  【关键】保存原始音频到 ChunkProcessState.original_audio  │
│     ↓                                                           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 5. 逐Chunk处理循环（转录层核心）                            ││
│  │    ┌─────────────────────────────────────────────────────┐  ││
│  │    │ for each chunk:                                     │  ││
│  │    │   5.1 SenseVoice 转录（时间领主）                    │  ││
│  │    │   5.2 分句算法                                       │  ││
│  │    │   5.3 置信度评估 + 检测事件标签（BGM/Noise等）       │  ││
│  │    │   5.4 熔断决策（FuseBreaker.should_fuse）            │  ││
│  │    │       │                                              │  ││
│  │    │       ├─ 高置信度 或 无BGM标签 → ACCEPT → 继续下一个  │  ││
│  │    │       │                                              │  ││
│  │    │       └─ 低置信度 + BGM/Noise → UPGRADE_SEPARATION   │  ││
│  │    │           │                                          │  ││
│  │    │           ↓                                          │  ││
│  │    │       ┌────────────────────────────────────────┐     │  ││
│  │    │       │ 熔断回溯：                              │     │  ││
│  │    │       │ 1. 检查 separation_level               │     │  ││
│  │    │       │ 2. 取回 original_audio（分离前）        │     │  ││
│  │    │       │ 3. 升级模型重新分离                     │     │  ││
│  │    │       │ 4. 更新 current_audio                   │     │  ││
│  │    │       │ 5. 回到 5.1 重新转录                    │     │  ││
│  │    │       │ 6. fuse_retry_count++                   │     │  ││
│  │    │       │ （止损点：max_retry=1）                 │     │  ││
│  │    │       └────────────────────────────────────────┘     │  ││
│  │    │                                                      │  ││
│  │    │   5.5 句级 SSE 推送（流式输出）                       │  ││
│  │    └─────────────────────────────────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────┘│
│     ↓                                                           │
│     【所有Chunk转录完成】                                        │
│     ↓                                                           │
│  ═══════════════════════════════════════════════════════════════│
│  │             后处理增强层（根据用户配置）                     ││
│  ═══════════════════════════════════════════════════════════════│
│     ↓                                                           │
│  6. [可选] Whisper 复核（仅对低置信度句子）                      │
│     │  - 检查用户配置：enhancement != OFF                       │
│     │  - 仅取文本，弃用时间戳                                   │
│     │  - 应用伪对齐生成字级时间戳                               │
│     ↓                                                           │
│  7. [可选] LLM 校对                                             │
│     ↓                                                           │
│  8. [可选] LLM 翻译                                             │
│     ↓                                                           │
│  9. 生成字幕文件                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、流式字幕输出系统（新增）

**路径**: `backend/app/services/streaming_subtitle.py`

```python
"""
流式字幕管理系统

核心职责：
1. 管理字幕句子列表（支持原地更新）
2. 协调 SSE 事件推送（统一 Tag）
3. 支持多阶段增量更新（SV → Whisper → LLM）
"""
from typing import Dict, List, Optional
from ..models.sensevoice_models import SentenceSegment, TextSource
from .sse_service import get_sse_manager, push_subtitle_event, push_progress_event
import logging

logger = logging.getLogger(__name__)


class StreamingSubtitleManager:
    """流式字幕管理器"""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.sentences: Dict[int, SentenceSegment] = {}  # key = sentence_index
        self.sentence_count = 0
        self.sse_manager = get_sse_manager()

    def add_sentence(self, sentence: SentenceSegment) -> int:
        """
        添加新句子（SenseVoice 阶段）

        Args:
            sentence: 句子段落

        Returns:
            int: 句子索引
        """
        index = self.sentence_count
        self.sentences[index] = sentence
        self.sentence_count += 1

        # 推送 SSE 事件
        push_subtitle_event(
            self.sse_manager,
            self.job_id,
            "sv_sentence",
            {
                "index": index,
                "sentence": sentence.to_dict(),
                "source": "sensevoice"
            }
        )

        logger.debug(f"添加句子 {index}: {sentence.text[:30]}...")
        return index

    def update_sentence(
        self,
        index: int,
        new_text: str,
        source: TextSource,
        confidence: float = None,
        perplexity: float = None
    ):
        """
        更新已有句子（Whisper 复核或 LLM 校对）

        Args:
            index: 句子索引
            new_text: 新文本
            source: 文本来源
            confidence: 新置信度（可选）
            perplexity: LLM 困惑度（可选）
        """
        if index not in self.sentences:
            logger.warning(f"句子 {index} 不存在，无法更新")
            return

        sentence = self.sentences[index]

        # 应用伪对齐
        from .pseudo_alignment import PseudoAlignment
        PseudoAlignment.apply_to_sentence(sentence, new_text, source)

        # 更新置信度和困惑度
        if confidence is not None:
            sentence.confidence = confidence
        if perplexity is not None:
            sentence.perplexity = perplexity
            sentence.warning_type = sentence.compute_warning_type()

        # 推送 SSE 事件
        event_type = {
            TextSource.WHISPER_PATCH: "whisper_patch",
            TextSource.LLM_CORRECTION: "llm_proof",
            TextSource.LLM_TRANSLATION: "llm_trans",
        }.get(source, "batch_update")

        push_subtitle_event(
            self.sse_manager,
            self.job_id,
            event_type,
            {
                "index": index,
                "sentence": sentence.to_dict(),
                "source": source.value,
                "is_update": True
            }
        )

        logger.debug(f"更新句子 {index} ({source.value}): {new_text[:30]}...")

    def set_translation(self, index: int, translation: str, confidence: float = None):
        """
        设置翻译结果

        Args:
            index: 句子索引
            translation: 翻译文本
            confidence: 翻译置信度
        """
        if index not in self.sentences:
            return

        sentence = self.sentences[index]
        sentence.translation = translation
        if confidence is not None:
            sentence.translation_confidence = confidence

        push_subtitle_event(
            self.sse_manager,
            self.job_id,
            "llm_trans",
            {
                "index": index,
                "translation": translation,
                "confidence": confidence
            }
        )

    def get_all_sentences(self) -> List[SentenceSegment]:
        """获取所有句子（按时间排序）"""
        sentences = list(self.sentences.values())
        sentences.sort(key=lambda s: s.start)
        return sentences

    def get_context_window(self, index: int, window_size: int = 3) -> str:
        """
        获取上下文窗口（用于 LLM 校对）

        Args:
            index: 当前句子索引
            window_size: 上下文窗口大小

        Returns:
            str: 上下文文本
        """
        context_indices = range(max(0, index - window_size), index)
        context_texts = [
            self.sentences[i].text
            for i in context_indices
            if i in self.sentences
        ]
        return " ".join(context_texts)


# ========== 单例工厂 ==========

_subtitle_managers: Dict[str, StreamingSubtitleManager] = {}


def get_streaming_subtitle_manager(job_id: str) -> StreamingSubtitleManager:
    """获取或创建流式字幕管理器"""
    global _subtitle_managers
    if job_id not in _subtitle_managers:
        _subtitle_managers[job_id] = StreamingSubtitleManager(job_id)
    return _subtitle_managers[job_id]


def remove_streaming_subtitle_manager(job_id: str):
    """移除流式字幕管理器"""
    global _subtitle_managers
    if job_id in _subtitle_managers:
        del _subtitle_managers[job_id]
```

---

## 四、熔断回溯机制（转录层核心）

> **注意**：熔断仅指"升级分离模型"，发生在单个 Chunk 转录后。不是 Whisper 复核。

### 4.1 Chunk 状态管理

每个 VAD Chunk 需要维护状态，关键是保留原始音频引用：

```python
# 使用 Phase 2 定义的 ChunkProcessState
from ..models.circuit_breaker_models import ChunkProcessState, SeparationLevel

def _init_chunk_states(
    self,
    vad_segments: List[dict],
    audio_array: np.ndarray
) -> List[ChunkProcessState]:
    """
    初始化所有 Chunk 的处理状态

    关键：保存原始音频引用，用于熔断回溯
    """
    states = []
    sr = 16000

    for i, seg in enumerate(vad_segments):
        start_sample = int(seg['start'] * sr)
        end_sample = int(seg['end'] * sr)
        chunk_audio = audio_array[start_sample:end_sample]

        state = ChunkProcessState(
            chunk_index=i,
            start_time=seg['start'],
            end_time=seg['end'],
            original_audio=chunk_audio.copy(),  # 关键：保存原始音频副本
            current_audio=chunk_audio,           # 当前使用的音频（可能被分离后替换）
            separation_level=SeparationLevel.NONE
        )
        states.append(state)

    return states
```

### 4.2 单 Chunk 转录 + 熔断循环

```python
async def _transcribe_chunk_with_fusing(
    self,
    chunk_state: ChunkProcessState,
    job: JobState,
    subtitle_manager: StreamingSubtitleManager,
    demucs_service
) -> List[SentenceSegment]:
    """
    单个 Chunk 的转录流程（含熔断回溯）

    流程：
    1. 使用 current_audio 进行 SenseVoice 转录
    2. 评估置信度和事件标签
    3. 熔断决策
    4. 如需熔断：回溯到 original_audio，升级分离，重新转录
    5. 止损点：max_retry=1
    """
    from .fuse_breaker import get_fuse_breaker, execute_fuse_upgrade, FuseAction

    fuse_breaker = get_fuse_breaker()

    while True:
        # 1. SenseVoice 转录
        sv_result = await self._sensevoice_transcribe(chunk_state.current_audio, job)

        # 2. 分句
        sentences = self._split_sentences(sv_result, chunk_state.start_time)

        # 3. 计算置信度和事件标签
        avg_confidence = sum(s.confidence for s in sentences) / len(sentences) if sentences else 0.0
        event_tag = sv_result.event  # SenseVoice 检测到的事件（BGM/Noise等）

        # 4. 熔断决策
        decision = fuse_breaker.should_fuse(
            chunk_state=chunk_state,
            confidence=avg_confidence,
            event_tag=event_tag
        )

        self.logger.debug(
            f"Chunk {chunk_state.chunk_index} 熔断决策: {decision.action.value}, "
            f"置信度={avg_confidence:.2f}, 事件={event_tag}"
        )

        # 5. 处理决策
        if decision.action == FuseAction.ACCEPT:
            # 接受结果，推送 SSE，返回
            for sent in sentences:
                subtitle_manager.add_sentence(sent)
            return sentences

        elif decision.action == FuseAction.UPGRADE_SEPARATION:
            # 熔断回溯：使用原始音频重新分离
            self.logger.info(
                f"Chunk {chunk_state.chunk_index} 触发熔断，"
                f"升级分离: {chunk_state.separation_level.value} → {decision.next_separation_level.value}"
            )

            chunk_state = execute_fuse_upgrade(
                chunk_state=chunk_state,
                next_level=decision.next_separation_level,
                demucs_service=demucs_service
            )

            # 继续循环，使用升级后的音频重新转录
            continue

        else:
            # 未知动作，接受当前结果
            for sent in sentences:
                subtitle_manager.add_sentence(sent)
            return sentences
```

### 4.3 熔断升级执行（Phase 2 定义）

```python
# 来自 Phase 2 的 fuse_breaker.py
def execute_fuse_upgrade(
    chunk_state: ChunkProcessState,
    next_level: SeparationLevel,
    demucs_service
) -> ChunkProcessState:
    """
    执行熔断升级：使用原始音频重新分离

    关键逻辑：
    1. 取回 original_audio（分离前的原始音频）
    2. 使用升级的模型进行分离
    3. 更新 current_audio
    4. 增加重试计数
    """
    logger.info(
        f"Chunk {chunk_state.chunk_index} 熔断升级: "
        f"{chunk_state.separation_level.value} → {next_level.value}"
    )

    # 关键：使用原始音频（分离前）进行重新分离
    original_audio = chunk_state.original_audio
    if original_audio is None:
        logger.error("熔断失败：原始音频引用丢失")
        return chunk_state

    # 选择模型
    model_name = "htdemucs" if next_level == SeparationLevel.HTDEMUCS else "mdx_extra"

    # 执行分离
    separated_audio = demucs_service.separate_chunk(
        audio=original_audio,
        model=model_name
    )

    # 更新状态
    chunk_state.current_audio = separated_audio
    chunk_state.separation_level = next_level
    chunk_state.separation_model_used = model_name
    chunk_state.fuse_retry_count += 1

    logger.info(f"Chunk {chunk_state.chunk_index} 熔断升级完成，重试次数: {chunk_state.fuse_retry_count}")

    return chunk_state
```

---

## 五、后处理增强层（Whisper复核等）

> **注意**：本节内容发生在**所有 Chunk 转录完成后**，根据用户配置执行。

### 5.1 Whisper 复核（仅取文本 + 伪对齐）

**修改位置**: `transcription_service.py`

```python
async def _whisper_text_patch(
    self,
    sentence: SentenceSegment,
    sentence_index: int,
    audio_array: np.ndarray,
    job: JobState,
    subtitle_manager: StreamingSubtitleManager
) -> SentenceSegment:
    """
    Whisper 复核（时空解耦版：仅取文本）

    核心原则：
    - SenseVoice 确定的时间轴（start/end）不可变
    - 仅使用 Whisper 的文本结果
    - 新文本使用伪对齐生成字级时间戳

    Args:
        sentence: 原始句子（由 SenseVoice 生成）
        sentence_index: 句子索引
        audio_array: 完整音频数组
        job: 任务状态
        subtitle_manager: 流式字幕管理器

    Returns:
        SentenceSegment: 更新后的句子
    """
    from .whisper_service import get_whisper_service

    whisper_service = get_whisper_service()

    # 提取对应时间段的音频（使用 SenseVoice 的时间窗口）
    sr = 16000
    start_sample = int(sentence.start * sr)
    end_sample = int(sentence.end * sr)
    audio_segment = audio_array[start_sample:end_sample]

    # 获取上下文提示
    context = subtitle_manager.get_context_window(sentence_index)

    # Whisper 转录（仅取文本，弃用时间戳）
    result = whisper_service.transcribe(
        audio=audio_segment,
        initial_prompt=context,
        language=job.settings.language,
        word_timestamps=False  # 不需要字级时间戳，使用伪对齐
    )

    whisper_text = result.get('text', '').strip()

    if not whisper_text:
        self.logger.warning(f"Whisper 复核返回空文本，保留原结果")
        return sentence

    # 保存 Whisper 备选文本
    sentence.whisper_alternative = whisper_text

    # 使用伪对齐更新句子
    subtitle_manager.update_sentence(
        index=sentence_index,
        new_text=whisper_text,
        source=TextSource.WHISPER_PATCH,
        confidence=self._estimate_whisper_confidence(result)
    )

    return self.sentences[sentence_index]


def _estimate_whisper_confidence(self, result: dict) -> float:
    """估算 Whisper 结果置信度"""
    segments = result.get('segments', [])
    if not segments:
        return 0.7

    # 基于 avg_logprob 和 no_speech_prob 计算
    total_logprob = sum(s.get('avg_logprob', -0.5) for s in segments)
    avg_logprob = total_logprob / len(segments)

    avg_no_speech = sum(s.get('no_speech_prob', 0.1) for s in segments) / len(segments)

    # 转换为 0-1 置信度
    confidence = min(1.0, max(0.0, 1.0 + avg_logprob))  # logprob 越接近 0 越好
    confidence *= (1.0 - avg_no_speech)  # no_speech 越低越好

    return round(confidence, 3)
```

### 5.2 后处理增强调度

```python
async def _post_process_enhancement(
    self,
    sentences: List[SentenceSegment],
    audio_array: np.ndarray,
    job: JobState,
    subtitle_manager: StreamingSubtitleManager,
    solution_config: SolutionConfig
) -> List[SentenceSegment]:
    """
    后处理增强层（所有 Chunk 转录完成后执行）

    根据用户配置执行：
    1. 低置信度句子 → Whisper 复核（仅文本 + 伪对齐）
    2. [可选] LLM 校对
    3. [可选] LLM 翻译

    注意：这不是熔断，熔断在转录阶段已经处理完成
    """
    from .progress_tracker import get_progress_tracker, ProcessPhase
    from .solution_matrix import EnhancementMode, ProofreadMode, TranslateMode
    from .thresholds import needs_whisper_patch

    progress_tracker = get_progress_tracker(job.job_id, solution_config.preset_id)

    # 1. 收集需要 Whisper 复核的句子（根据用户配置）
    patch_queue = []
    if solution_config.enhancement != EnhancementMode.OFF:
        for i, sentence in enumerate(sentences):
            if needs_whisper_patch(sentence.confidence):
                patch_queue.append((i, sentence))

    # 2. Whisper 复核阶段
    if patch_queue:
        progress_tracker.start_phase(ProcessPhase.WHISPER_PATCH, len(patch_queue), "Whisper 复核中...")

        for idx, (sent_idx, sentence) in enumerate(patch_queue):
            await self._whisper_text_patch(
                sentence, sent_idx, audio_array, job, subtitle_manager
            )
            progress_tracker.update_phase(ProcessPhase.WHISPER_PATCH, increment=1)

        progress_tracker.complete_phase(ProcessPhase.WHISPER_PATCH)

    # 3. [可选] LLM 校对
    if solution_config.proofread != ProofreadMode.OFF:
        await self._llm_proofread(sentences, job, subtitle_manager, solution_config)

    # 4. [可选] LLM 翻译
    if solution_config.translate != TranslateMode.OFF:
        await self._llm_translate(sentences, job, subtitle_manager, solution_config)

    return subtitle_manager.get_all_sentences()
```

---

## 六、主处理流程（v2.1 概念澄清版）

```python
async def _process_video_sensevoice(self, job: JobState):
    """
    SenseVoice 主处理流程（v2.1 概念澄清版）

    流程说明：
    1-4: 准备阶段（音频提取、VAD、频谱分诊、按需分离）
    5: 转录阶段（逐Chunk转录 + 熔断回溯）
    6-8: 后处理增强阶段（Whisper复核、LLM校对/翻译）
    9: 输出阶段（生成字幕）
    """
    from .streaming_subtitle import get_streaming_subtitle_manager, remove_streaming_subtitle_manager
    from .progress_tracker import get_progress_tracker, remove_progress_tracker, ProcessPhase
    from .solution_matrix import SolutionConfig
    from .audio_spectrum_classifier import get_spectrum_classifier
    from .demucs_service import get_demucs_service

    # 获取方案配置
    preset_id = getattr(job.settings, 'preset_id', 'default')
    solution_config = SolutionConfig.from_preset(preset_id)

    # 初始化管理器
    subtitle_manager = get_streaming_subtitle_manager(job.job_id)
    progress_tracker = get_progress_tracker(job.job_id, preset_id)

    try:
        # 1. 音频提取
        progress_tracker.start_phase(ProcessPhase.EXTRACT, 1, "提取音频...")
        audio_path, audio_array = await self._extract_audio_with_array(job)
        progress_tracker.complete_phase(ProcessPhase.EXTRACT)

        # 2. VAD 物理切分
        progress_tracker.start_phase(ProcessPhase.VAD, 1, "VAD 切分...")
        vad_segments = await self._vad_physical_split(audio_path, audio_array, job)
        progress_tracker.complete_phase(ProcessPhase.VAD)

        # 3. 频谱分诊（Chunk级别）
        progress_tracker.start_phase(ProcessPhase.BGM_DETECT, 1, "频谱分诊...")
        spectrum_classifier = get_spectrum_classifier()
        diagnoses = spectrum_classifier.diagnose_chunks(
            [(audio_array[int(s['start']*16000):int(s['end']*16000)], s['start'], s['end'])
             for s in vad_segments]
        )
        progress_tracker.complete_phase(ProcessPhase.BGM_DETECT)

        # 4. 初始化 Chunk 状态 + 按需人声分离
        # 【关键】保存原始音频到 ChunkProcessState.original_audio
        chunk_states = self._init_chunk_states(vad_segments, audio_array)
        demucs_service = get_demucs_service()

        chunks_to_separate = [(i, chunk_states[i], diagnoses[i])
                              for i in range(len(diagnoses))
                              if diagnoses[i].need_separation]

        if chunks_to_separate:
            progress_tracker.start_phase(ProcessPhase.DEMUCS, len(chunks_to_separate), "人声分离...")
            for sep_idx, (chunk_idx, chunk_state, diag) in enumerate(chunks_to_separate):
                # 执行分离，更新 current_audio
                separated_audio = await demucs_service.separate_chunk(
                    audio=chunk_state.original_audio,
                    model=diag.recommended_model
                )
                chunk_state.current_audio = separated_audio
                chunk_state.separation_level = (
                    SeparationLevel.HTDEMUCS if diag.recommended_model == "htdemucs"
                    else SeparationLevel.MDX_EXTRA
                )
                chunk_state.separation_model_used = diag.recommended_model
                progress_tracker.update_phase(ProcessPhase.DEMUCS, increment=1)
            progress_tracker.complete_phase(ProcessPhase.DEMUCS)

        # 5. 逐Chunk转录 + 熔断回溯（转录层核心）
        progress_tracker.start_phase(ProcessPhase.SENSEVOICE, len(chunk_states), "SenseVoice 转录...")
        all_sentences = []

        for chunk_state in chunk_states:
            # 单个 Chunk 转录（含熔断回溯循环）
            sentences = await self._transcribe_chunk_with_fusing(
                chunk_state=chunk_state,
                job=job,
                subtitle_manager=subtitle_manager,
                demucs_service=demucs_service
            )
            all_sentences.extend(sentences)
            progress_tracker.update_phase(ProcessPhase.SENSEVOICE, increment=1)

        progress_tracker.complete_phase(ProcessPhase.SENSEVOICE)

        # ═══════════════════════════════════════════════════════════════
        # 后处理增强层（所有 Chunk 转录完成后，根据用户配置执行）
        # ═══════════════════════════════════════════════════════════════

        # 6. 后处理增强（Whisper复核、LLM校对/翻译）
        final_results = await self._post_process_enhancement(
            all_sentences, audio_array, job, subtitle_manager, solution_config
        )

        # 7. 生成字幕
        progress_tracker.start_phase(ProcessPhase.SRT, 1, "生成字幕...")
        await self._generate_subtitle_from_sentences(final_results, job)
        progress_tracker.complete_phase(ProcessPhase.SRT)

        # 8. 完成
        job.status = 'completed'
        push_signal_event(get_sse_manager(), job.job_id, "job_complete", "处理完成")

    except Exception as e:
        self.logger.error(f"SenseVoice 处理失败: {e}", exc_info=True)
        job.status = 'failed'
        job.error = str(e)
        push_signal_event(get_sse_manager(), job.job_id, "job_failed", str(e))
        raise

    finally:
        # 清理资源
        remove_streaming_subtitle_manager(job.job_id)
        remove_progress_tracker(job.job_id)
```

---

## 七、验收标准（扩展）

### 基础功能

- [ ] 完整流程可运行
- [ ] VAD 物理切分正常工作
- [ ] 分句算法输出句级粒度字幕
- [ ] 字幕文件成功生成

### VAD优先 + 频谱分诊（v2.1核心）

- [ ] 流程顺序正确：`音频提取 → VAD → 频谱分诊 → 按需分离 → 转录`
- [ ] 频谱分诊在每个VAD Chunk上独立执行
- [ ] 纯净Chunk直接跳过分离，节省时间
- [ ] 分离模型根据分诊结果动态选择（htdemucs/mdx_extra）

### 熔断回溯机制（v2.1核心新增）

- [ ] `ChunkProcessState` 正确保存 `original_audio`（分离前的原始音频）
- [ ] 熔断触发时使用 `original_audio` 重新分离，而非 `current_audio`
- [ ] 熔断后 `separation_level` 正确升级（NONE→HTDEMUCS→MDX_EXTRA）
- [ ] 止损点机制生效（`max_fuse_retry=1`）
- [ ] 熔断仅在 BGM/Noise 标签 + 低置信度时触发
- [ ] 熔断不影响无 BGM 标签的低置信度句子（由后处理增强处理）

### 后处理增强层

- [ ] Whisper 复核发生在**所有 Chunk 转录完成后**
- [ ] 仅当用户配置 `enhancement != OFF` 时才执行 Whisper 复核
- [ ] Whisper 复核**仅使用文本**，时间戳不变
- [ ] 伪对齐正确生成字级时间戳
- [ ] `SentenceSegment.source` 正确标记来源

### 流式输出系统

- [ ] `StreamingSubtitleManager` 正确管理句子列表
- [ ] SSE 事件使用统一 Tag（`subtitle.sv_sentence`, `subtitle.whisper_patch` 等）
- [ ] 多阶段更新正确推送到前端

### 性能验证

- [ ] 连续处理 3 个视频不 OOM
- [ ] 进度条平滑无跳跃

---

## 七、下一步

完成 Phase 3（修订版 v2.1）后，进入 [Phase 4: 前端适配（修订版）](./04_Phase4_前端适配_修订版.md)
