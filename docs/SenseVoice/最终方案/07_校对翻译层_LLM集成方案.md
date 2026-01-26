# 校对翻译层 LLM 集成方案

> **文档版本**：v1.0
> **基于**：《新补充.md》LLM 稀疏校对策略设计
> **前置依赖**：[06_转录层深度优化_时空解耦架构.md](./06_转录层深度优化_时空解耦架构.md)
> **目标**：建立独立的 LLM 校对/翻译层，与转录层完美对接

---

## 一、架构概述

### 1.1 LLM 层定位

在时空解耦架构中，LLM 扮演**逻辑胶水（Semantic Refiner）**角色：

```
┌─────────────────────────────────────────────────────────────────┐
│                      完整处理流程                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    转录层（已完成）                        │  │
│  │  SenseVoice → [Whisper 复核] → 伪对齐 → 基础字幕          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                  │
│                     TranscriptionOutput                         │
│                              ↓                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    校对翻译层（本文档）                    │  │
│  │                                                           │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │  │
│  │  │  P1: 稀疏   │    │  P2: 全量   │    │  T1/T2:     │   │  │
│  │  │  校对       │    │  校对       │    │  翻译       │   │  │
│  │  │  (推荐)     │    │  (高质量)   │    │             │   │  │
│  │  └─────────────┘    └─────────────┘    └─────────────┘   │  │
│  │                                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                  │
│                        最终字幕输出                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 LLM 层职责

| 功能 | 描述 | 时间戳影响 |
|------|------|-----------|
| **错别字修正** | 基于上下文修正 ASR 错误 | 仅修改文本，时间戳不变 |
| **歧义仲裁** | 当 SenseVoice 和 Whisper 结果不一致时裁决 | 仅修改文本，时间戳不变 |
| **口语润色** | 将口语化表达转为书面语 | 仅修改文本，时间戳不变 |
| **翻译** | 将文本翻译为目标语言 | 仅修改文本，时间戳不变 |

**核心原则**：LLM 层**永远不修改时间戳**，只修改文本内容。

---

## 二、模块设计

### 2.1 文件结构

```
backend/app/services/llm/
├── __init__.py
├── llm_service.py          # LLM 服务基类
├── sparse_proofreader.py   # P1: 稀疏校对器
├── full_proofreader.py     # P2: 全量校对器
├── translator.py           # T1/T2: 翻译器
├── prompt_templates.py     # Prompt 模板
└── llm_models.py           # 数据模型

backend/app/models/
└── llm_models.py           # LLM 相关数据模型
```

### 2.2 数据模型

**文件**: `backend/app/models/llm_models.py`

```python
"""
LLM 层数据模型
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class LLMProvider(Enum):
    """LLM 提供商"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # 本地部署（如 Ollama）


class ProofreadAction(Enum):
    """校对动作"""
    KEEP = "keep"           # 保持原文
    CORRECT = "correct"     # 修正错误
    ARBITRATE = "arbitrate" # 仲裁（SV vs Whisper）


@dataclass
class ProofreadRequest:
    """校对请求"""
    index: int                      # 句子索引
    text: str                       # 原始文本
    confidence: float               # 置信度
    context_before: List[str]       # 前文上下文（2句）
    context_after: List[str]        # 后文上下文（2句）
    whisper_alternative: Optional[str] = None  # Whisper 备选


@dataclass
class ProofreadResult:
    """校对结果"""
    index: int
    original_text: str
    corrected_text: str
    action: ProofreadAction
    reason: str
    confidence_boost: float = 0.0   # 置信度提升


@dataclass
class TranslateRequest:
    """翻译请求"""
    sentences: List[Dict]           # 待翻译句子
    source_language: str            # 源语言
    target_language: str            # 目标语言
    context_window: int = 5         # 上下文窗口大小


@dataclass
class TranslateResult:
    """翻译结果"""
    index: int
    original_text: str
    translated_text: str
    source_language: str
    target_language: str


@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4o-mini"      # 默认使用经济型模型
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.3        # 校对任务使用低温度
    max_tokens: int = 1000
    timeout: int = 30

    # 稀疏校对配置
    sparse_confidence_threshold: float = 0.7  # 低于此值才校对
    sparse_context_size: int = 2              # 上下文句数

    # 全量校对配置
    full_window_size: int = 10                # 滑动窗口大小
    full_overlap: int = 2                     # 窗口重叠句数
```

---

## 三、P1：稀疏校对（Context-Aware Sparse Correction）

### 3.1 核心原理

**问题**：只传一句给 LLM 会导致"瞎改"（缺乏上下文）

**解决方案**：采用**三明治 Prompt 结构**

```
[前2句 Context] + <target>待修补低置信度句</target> + [后2句 Context]
```

### 3.2 实现代码

**文件**: `backend/app/services/llm/sparse_proofreader.py`

```python
"""
P1: 稀疏校对器

核心策略：
- 仅对低置信度句子进行校对
- 使用三明治 Prompt 结构提供上下文
- 节省 90%+ Token
"""
import logging
from typing import List, Optional
from dataclasses import dataclass

from ..models.sensevoice_models import SentenceSegment, TextSource
from ..models.llm_models import (
    ProofreadRequest, ProofreadResult, ProofreadAction, LLMConfig
)
from .llm_service import get_llm_service
from .prompt_templates import SPARSE_PROOFREAD_PROMPT
from ..pseudo_alignment import get_pseudo_alignment

logger = logging.getLogger(__name__)


class SparseProofreader:
    """稀疏校对器"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.llm_service = get_llm_service(self.config)

    def filter_candidates(
        self,
        sentences: List[SentenceSegment]
    ) -> List[int]:
        """
        筛选需要校对的句子索引

        筛选条件：
        1. 置信度低于阈值
        2. 被 Whisper 复核修改过
        3. 有 Whisper 备选文本（需要仲裁）

        Returns:
            需要校对的句子索引列表
        """
        candidates = []

        for i, sent in enumerate(sentences):
            # 条件1：低置信度
            if sent.confidence < self.config.sparse_confidence_threshold:
                candidates.append(i)
                continue

            # 条件2：被修改过
            if sent.is_modified:
                candidates.append(i)
                continue

            # 条件3：有备选文本需要仲裁
            if sent.whisper_alternative:
                candidates.append(i)
                continue

        logger.info(
            f"稀疏校对筛选: {len(candidates)}/{len(sentences)} 句需要校对 "
            f"(节省 {100 - len(candidates)/len(sentences)*100:.1f}% Token)"
        )

        return candidates

    def build_request(
        self,
        sentences: List[SentenceSegment],
        index: int
    ) -> ProofreadRequest:
        """
        构建校对请求（三明治结构）

        Args:
            sentences: 全部句子
            index: 目标句子索引

        Returns:
            校对请求
        """
        ctx_size = self.config.sparse_context_size
        target = sentences[index]

        # 前文上下文
        context_before = [
            sentences[i].text
            for i in range(max(0, index - ctx_size), index)
        ]

        # 后文上下文
        context_after = [
            sentences[i].text
            for i in range(index + 1, min(len(sentences), index + 1 + ctx_size))
        ]

        return ProofreadRequest(
            index=index,
            text=target.text,
            confidence=target.confidence,
            context_before=context_before,
            context_after=context_after,
            whisper_alternative=target.whisper_alternative
        )

    async def proofread_single(
        self,
        request: ProofreadRequest
    ) -> ProofreadResult:
        """
        校对单个句子

        Args:
            request: 校对请求

        Returns:
            校对结果
        """
        # 构建 Prompt
        prompt = self._build_prompt(request)

        # 调用 LLM
        response = await self.llm_service.complete(prompt)

        # 解析响应
        result = self._parse_response(request, response)

        return result

    def _build_prompt(self, request: ProofreadRequest) -> str:
        """构建三明治 Prompt"""
        # 组装上下文
        context_before_str = "\n".join(request.context_before) if request.context_before else "(无)"
        context_after_str = "\n".join(request.context_after) if request.context_after else "(无)"

        # 是否有 Whisper 备选
        whisper_section = ""
        if request.whisper_alternative:
            whisper_section = f"""
参考文本（Whisper 识别结果）：
{request.whisper_alternative}

请对比上述两个识别结果，选择更准确的一个，或融合两者优点。
"""

        prompt = SPARSE_PROOFREAD_PROMPT.format(
            context_before=context_before_str,
            target_text=request.text,
            context_after=context_after_str,
            whisper_section=whisper_section,
            confidence=request.confidence
        )

        return prompt

    def _parse_response(
        self,
        request: ProofreadRequest,
        response: str
    ) -> ProofreadResult:
        """解析 LLM 响应"""
        # 简单解析：提取修正后的文本
        # 实际实现可以使用 JSON 格式化输出
        corrected_text = response.strip()

        # 判断是否修改
        if corrected_text == request.text:
            action = ProofreadAction.KEEP
            reason = "LLM 认为无需修改"
        elif request.whisper_alternative and corrected_text == request.whisper_alternative:
            action = ProofreadAction.ARBITRATE
            reason = "LLM 选择了 Whisper 结果"
        else:
            action = ProofreadAction.CORRECT
            reason = "LLM 进行了修正"

        return ProofreadResult(
            index=request.index,
            original_text=request.text,
            corrected_text=corrected_text,
            action=action,
            reason=reason,
            confidence_boost=0.2 if action != ProofreadAction.KEEP else 0
        )

    async def proofread_batch(
        self,
        sentences: List[SentenceSegment]
    ) -> List[SentenceSegment]:
        """
        批量稀疏校对

        Args:
            sentences: 输入句子列表

        Returns:
            校对后的句子列表（原对象被修改）
        """
        # 1. 筛选候选
        candidates = self.filter_candidates(sentences)

        if not candidates:
            logger.info("稀疏校对：无需校对的句子")
            return sentences

        # 2. 逐句校对
        pseudo_alignment = get_pseudo_alignment()

        for idx in candidates:
            request = self.build_request(sentences, idx)
            result = await self.proofread_single(request)

            if result.action != ProofreadAction.KEEP:
                # 应用修正 + 伪对齐
                sentence = sentences[idx]
                pseudo_alignment.apply_to_sentence(
                    sentence=sentence,
                    new_text=result.corrected_text,
                    source=TextSource.LLM_CORRECTION
                )

                # 提升置信度
                sentence.confidence = min(1.0, sentence.confidence + result.confidence_boost)

                logger.info(
                    f"句子 {idx} 已校对: '{result.original_text}' -> '{result.corrected_text}'"
                )

        return sentences


# ========== 单例访问 ==========

_sparse_proofreader_instance = None


def get_sparse_proofreader(config: Optional[LLMConfig] = None) -> SparseProofreader:
    """获取稀疏校对器单例"""
    global _sparse_proofreader_instance
    if _sparse_proofreader_instance is None:
        _sparse_proofreader_instance = SparseProofreader(config)
    return _sparse_proofreader_instance
```

### 3.3 Prompt 模板

**文件**: `backend/app/services/llm/prompt_templates.py`

```python
"""
LLM Prompt 模板

设计原则：
1. 明确角色和任务
2. 提供充足上下文
3. 约束输出格式
4. 避免过度修改
"""

# ========== P1: 稀疏校对 Prompt ==========

SPARSE_PROOFREAD_PROMPT = """你是一个专业的字幕校对助手。你的任务是基于上下文修正语音识别错误。

## 上下文（前文）
{context_before}

## 待校对文本（置信度: {confidence:.2f}）
<target>{target_text}</target>

## 上下文（后文）
{context_after}

{whisper_section}

## 校对规则
1. 仅修正明显的语音识别错误（同音字、谐音字）
2. 保持原句意，不要改变表达风格
3. 如果原文正确，直接返回原文
4. 不要添加或删除内容
5. 不要修改标点符号风格

## 输出要求
直接输出校对后的文本，不要包含任何解释或标记。

校对后的文本："""


# ========== P2: 全量校对 Prompt ==========

FULL_PROOFREAD_PROMPT = """你是一个专业的字幕校对和润色助手。你的任务是校对并适度润色以下字幕片段。

## 字幕片段（共 {count} 句）
{sentences}

## 校对和润色规则
1. 修正语音识别错误（同音字、谐音字）
2. 适度润色口语化表达（但保留说话人风格）
3. 确保前后文逻辑通顺
4. 保持每句的大致长度
5. 不要合并或拆分句子

## 输出格式
按原有顺序输出校对后的文本，每句一行，格式如下：
1. [校对后的第1句]
2. [校对后的第2句]
...

校对结果："""


# ========== T1: 全量翻译 Prompt ==========

FULL_TRANSLATE_PROMPT = """你是一个专业的字幕翻译助手。请将以下{source_language}字幕翻译为{target_language}。

## 原文字幕（共 {count} 句）
{sentences}

## 翻译规则
1. 保持译文自然流畅
2. 适应字幕场景（简洁、易读）
3. 保持每句的大致长度比例
4. 保留专有名词的原文或通用译法
5. 不要合并或拆分句子

## 输出格式
按原有顺序输出翻译后的文本，每句一行，格式如下：
1. [翻译后的第1句]
2. [翻译后的第2句]
...

翻译结果："""


# ========== T2: 部分翻译 Prompt ==========

PARTIAL_TRANSLATE_PROMPT = """你是一个专业的字幕翻译助手。请将以下标记的{source_language}句子翻译为{target_language}。

## 上下文
{context}

## 待翻译句子
{target_sentences}

## 翻译规则
1. 保持译文自然流畅
2. 结合上下文确保翻译准确
3. 保持简洁（字幕场景）

翻译结果："""


# ========== 仲裁 Prompt ==========

ARBITRATE_PROMPT = """你是一个语音识别质量评估专家。请比较以下两个识别结果，选择更准确的一个。

## 上下文
前文：{context_before}
后文：{context_after}

## 识别结果 A（SenseVoice）
{sensevoice_text}

## 识别结果 B（Whisper）
{whisper_text}

## 评估标准
1. 与上下文的语义连贯性
2. 语法正确性
3. 常用表达的合理性

## 输出要求
1. 选择 A 或 B，或融合两者
2. 直接输出最终文本

最终文本："""
```

---

## 四、P2：全量校对（Full Proofread）

### 4.1 核心原理

**场景**：正式出版、文稿整理等高质量需求

**策略**：滑动窗口全量校对

```
┌─────────────────────────────────────────────────────────────────┐
│                    滑动窗口校对策略                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  句子序列：[1] [2] [3] [4] [5] [6] [7] [8] [9] [10] ...        │
│                                                                 │
│  窗口1：    [1] [2] [3] [4] [5]                                 │
│  窗口2：            [4] [5] [6] [7] [8]    (重叠2句)            │
│  窗口3：                    [7] [8] [9] [10] [11]               │
│  ...                                                            │
│                                                                 │
│  重叠区域用于保持上下文连贯性                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 实现代码

**文件**: `backend/app/services/llm/full_proofreader.py`

```python
"""
P2: 全量校对器

核心策略：
- 滑动窗口全量处理
- 窗口重叠保持连贯性
- 适合高质量输出场景
"""
import logging
from typing import List, Optional
from dataclasses import dataclass

from ..models.sensevoice_models import SentenceSegment, TextSource
from ..models.llm_models import LLMConfig
from .llm_service import get_llm_service
from .prompt_templates import FULL_PROOFREAD_PROMPT
from ..pseudo_alignment import get_pseudo_alignment

logger = logging.getLogger(__name__)


class FullProofreader:
    """全量校对器"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.llm_service = get_llm_service(self.config)

    def generate_windows(
        self,
        total_count: int
    ) -> List[tuple]:
        """
        生成滑动窗口

        Args:
            total_count: 总句数

        Returns:
            窗口列表 [(start, end), ...]
        """
        window_size = self.config.full_window_size
        overlap = self.config.full_overlap
        step = window_size - overlap

        windows = []
        start = 0

        while start < total_count:
            end = min(start + window_size, total_count)
            windows.append((start, end))

            if end >= total_count:
                break

            start += step

        logger.info(
            f"全量校对窗口: {len(windows)} 个窗口, "
            f"窗口大小={window_size}, 重叠={overlap}"
        )

        return windows

    async def proofread_window(
        self,
        sentences: List[SentenceSegment],
        start: int,
        end: int
    ) -> List[str]:
        """
        校对单个窗口

        Args:
            sentences: 全部句子
            start: 窗口起始索引
            end: 窗口结束索引

        Returns:
            校对后的文本列表
        """
        # 构建输入
        window_sentences = sentences[start:end]
        sentences_text = "\n".join([
            f"{i+1}. {s.text}"
            for i, s in enumerate(window_sentences)
        ])

        # 构建 Prompt
        prompt = FULL_PROOFREAD_PROMPT.format(
            count=len(window_sentences),
            sentences=sentences_text
        )

        # 调用 LLM
        response = await self.llm_service.complete(prompt)

        # 解析响应
        corrected_texts = self._parse_window_response(response, len(window_sentences))

        return corrected_texts

    def _parse_window_response(
        self,
        response: str,
        expected_count: int
    ) -> List[str]:
        """解析窗口响应"""
        lines = response.strip().split("\n")
        results = []

        for line in lines:
            # 去除序号前缀
            line = line.strip()
            if line and line[0].isdigit():
                # 移除 "1. " 格式的前缀
                parts = line.split(". ", 1)
                if len(parts) == 2:
                    results.append(parts[1])
                else:
                    results.append(line)
            elif line:
                results.append(line)

        # 确保数量匹配
        if len(results) != expected_count:
            logger.warning(
                f"窗口响应数量不匹配: 期望 {expected_count}, 实际 {len(results)}"
            )
            # 补齐或截断
            if len(results) < expected_count:
                results.extend([""] * (expected_count - len(results)))
            else:
                results = results[:expected_count]

        return results

    async def proofread_batch(
        self,
        sentences: List[SentenceSegment]
    ) -> List[SentenceSegment]:
        """
        批量全量校对

        Args:
            sentences: 输入句子列表

        Returns:
            校对后的句子列表
        """
        windows = self.generate_windows(len(sentences))
        pseudo_alignment = get_pseudo_alignment()

        # 记录每个句子的校对结果（处理重叠）
        correction_map = {}  # index -> corrected_text

        for start, end in windows:
            corrected_texts = await self.proofread_window(sentences, start, end)

            for i, text in enumerate(corrected_texts):
                global_idx = start + i

                # 重叠区域：后窗口的结果优先（上下文更完整）
                correction_map[global_idx] = text

        # 应用校对结果
        for idx, corrected_text in correction_map.items():
            sentence = sentences[idx]

            if corrected_text and corrected_text != sentence.text:
                pseudo_alignment.apply_to_sentence(
                    sentence=sentence,
                    new_text=corrected_text,
                    source=TextSource.LLM_CORRECTION
                )
                sentence.confidence = 1.0  # 全量校对后置信度设为最高

        logger.info(f"全量校对完成: {len(sentences)} 句")

        return sentences


# ========== 单例访问 ==========

_full_proofreader_instance = None


def get_full_proofreader(config: Optional[LLMConfig] = None) -> FullProofreader:
    """获取全量校对器单例"""
    global _full_proofreader_instance
    if _full_proofreader_instance is None:
        _full_proofreader_instance = FullProofreader(config)
    return _full_proofreader_instance
```

---

## 五、T1/T2：翻译层

### 5.1 翻译策略

| 方案 | 描述 | 场景 |
|------|------|------|
| **T1: 全量翻译** | 滑动窗口全文翻译 | 跨语言内容 |
| **T2: 部分翻译** | 仅翻译指定段落 | 教学重点标注 |

### 5.2 实现代码

**文件**: `backend/app/services/llm/translator.py`

```python
"""
翻译层

T1: 全量翻译 - 滑动窗口全文翻译
T2: 部分翻译 - 仅翻译指定段落
"""
import logging
from typing import List, Optional, Set
from dataclasses import dataclass

from ..models.sensevoice_models import SentenceSegment, TextSource
from ..models.llm_models import LLMConfig, TranslateRequest, TranslateResult
from .llm_service import get_llm_service
from .prompt_templates import FULL_TRANSLATE_PROMPT, PARTIAL_TRANSLATE_PROMPT
from ..pseudo_alignment import get_pseudo_alignment

logger = logging.getLogger(__name__)


class Translator:
    """翻译器"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.llm_service = get_llm_service(self.config)

    async def translate_full(
        self,
        sentences: List[SentenceSegment],
        source_language: str,
        target_language: str
    ) -> List[SentenceSegment]:
        """
        T1: 全量翻译

        Args:
            sentences: 输入句子列表
            source_language: 源语言
            target_language: 目标语言

        Returns:
            翻译后的句子列表
        """
        window_size = self.config.full_window_size
        pseudo_alignment = get_pseudo_alignment()

        # 滑动窗口翻译
        for start in range(0, len(sentences), window_size):
            end = min(start + window_size, len(sentences))
            window = sentences[start:end]

            # 构建输入
            sentences_text = "\n".join([
                f"{i+1}. {s.text}"
                for i, s in enumerate(window)
            ])

            # 构建 Prompt
            prompt = FULL_TRANSLATE_PROMPT.format(
                source_language=source_language,
                target_language=target_language,
                count=len(window),
                sentences=sentences_text
            )

            # 调用 LLM
            response = await self.llm_service.complete(prompt)

            # 解析响应
            translated_texts = self._parse_response(response, len(window))

            # 应用翻译结果
            for i, text in enumerate(translated_texts):
                global_idx = start + i
                sentence = sentences[global_idx]

                if text:
                    pseudo_alignment.apply_to_sentence(
                        sentence=sentence,
                        new_text=text,
                        source=TextSource.LLM_TRANSLATION
                    )

        logger.info(
            f"全量翻译完成: {len(sentences)} 句, "
            f"{source_language} -> {target_language}"
        )

        return sentences

    async def translate_partial(
        self,
        sentences: List[SentenceSegment],
        target_indexes: Set[int],
        source_language: str,
        target_language: str
    ) -> List[SentenceSegment]:
        """
        T2: 部分翻译

        Args:
            sentences: 输入句子列表
            target_indexes: 需要翻译的句子索引集合
            source_language: 源语言
            target_language: 目标语言

        Returns:
            翻译后的句子列表（仅指定句子被翻译）
        """
        pseudo_alignment = get_pseudo_alignment()
        ctx_size = self.config.sparse_context_size

        for idx in target_indexes:
            if idx >= len(sentences):
                continue

            sentence = sentences[idx]

            # 构建上下文
            context_start = max(0, idx - ctx_size)
            context_end = min(len(sentences), idx + ctx_size + 1)
            context = [sentences[i].text for i in range(context_start, context_end)]

            # 构建 Prompt
            prompt = PARTIAL_TRANSLATE_PROMPT.format(
                source_language=source_language,
                target_language=target_language,
                context="\n".join(context),
                target_sentences=sentence.text
            )

            # 调用 LLM
            response = await self.llm_service.complete(prompt)
            translated_text = response.strip()

            # 应用翻译
            if translated_text:
                pseudo_alignment.apply_to_sentence(
                    sentence=sentence,
                    new_text=translated_text,
                    source=TextSource.LLM_TRANSLATION
                )

        logger.info(
            f"部分翻译完成: {len(target_indexes)}/{len(sentences)} 句, "
            f"{source_language} -> {target_language}"
        )

        return sentences

    def _parse_response(
        self,
        response: str,
        expected_count: int
    ) -> List[str]:
        """解析翻译响应"""
        lines = response.strip().split("\n")
        results = []

        for line in lines:
            line = line.strip()
            if line and line[0].isdigit():
                parts = line.split(". ", 1)
                if len(parts) == 2:
                    results.append(parts[1])
                else:
                    results.append(line)
            elif line:
                results.append(line)

        # 补齐
        while len(results) < expected_count:
            results.append("")

        return results[:expected_count]


# ========== 单例访问 ==========

_translator_instance = None


def get_translator(config: Optional[LLMConfig] = None) -> Translator:
    """获取翻译器单例"""
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = Translator(config)
    return _translator_instance
```

---

## 六、LLM 服务基类

**文件**: `backend/app/services/llm/llm_service.py`

```python
"""
LLM 服务基类

支持多种 LLM 提供商：
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- 本地部署 (Ollama)
"""
import logging
from typing import Optional
from abc import ABC, abstractmethod

from ..models.llm_models import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


class BaseLLMService(ABC):
    """LLM 服务基类"""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    async def complete(self, prompt: str) -> str:
        """执行补全"""
        pass


class OpenAIService(BaseLLMService):
    """OpenAI 服务"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout
                )
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
        return self._client

    async def complete(self, prompt: str) -> str:
        """执行补全"""
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI 调用失败: {e}")
            raise


class AnthropicService(BaseLLMService):
    """Anthropic 服务"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(
                    api_key=self.config.api_key,
                    timeout=self.config.timeout
                )
            except ImportError:
                raise ImportError("请安装 anthropic: pip install anthropic")
        return self._client

    async def complete(self, prompt: str) -> str:
        """执行补全"""
        try:
            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic 调用失败: {e}")
            raise


class LocalService(BaseLLMService):
    """本地服务 (Ollama)"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"

    async def complete(self, prompt: str) -> str:
        """执行补全"""
        import aiohttp

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    result = await resp.json()
                    return result.get("response", "")
        except Exception as e:
            logger.error(f"本地 LLM 调用失败: {e}")
            raise


# ========== 工厂函数 ==========

def get_llm_service(config: LLMConfig) -> BaseLLMService:
    """
    获取 LLM 服务实例

    Args:
        config: LLM 配置

    Returns:
        对应的 LLM 服务实例
    """
    if config.provider == LLMProvider.OPENAI:
        return OpenAIService(config)
    elif config.provider == LLMProvider.ANTHROPIC:
        return AnthropicService(config)
    elif config.provider == LLMProvider.LOCAL:
        return LocalService(config)
    else:
        raise ValueError(f"不支持的 LLM 提供商: {config.provider}")
```

---

## 七、与转录层的对接

### 7.1 接口调用示例

```python
# 在 transcription_service.py 中

from .llm.sparse_proofreader import get_sparse_proofreader
from .llm.full_proofreader import get_full_proofreader
from .llm.translator import get_translator


async def _apply_llm_processing(
    self,
    sentences: List[SentenceSegment],
    config: SolutionConfig
) -> List[SentenceSegment]:
    """
    应用 LLM 处理（校对 + 翻译）

    Args:
        sentences: 转录层输出的句子列表
        config: 方案配置

    Returns:
        LLM 处理后的句子列表
    """
    # 1. 校对（P1 或 P2）
    if config.proofread == ProofreadMode.SPARSE:
        proofreader = get_sparse_proofreader()
        sentences = await proofreader.proofread_batch(sentences)

    elif config.proofread == ProofreadMode.FULL:
        proofreader = get_full_proofreader()
        sentences = await proofreader.proofread_batch(sentences)

    # 2. 翻译（T1 或 T2）
    if config.translate == TranslateMode.FULL:
        translator = get_translator()
        sentences = await translator.translate_full(
            sentences,
            source_language="zh",
            target_language=config.target_language
        )

    elif config.translate == TranslateMode.PARTIAL:
        # 部分翻译需要用户指定索引
        # 这里示例使用低置信度句子
        target_indexes = {
            i for i, s in enumerate(sentences)
            if s.confidence < 0.7
        }
        translator = get_translator()
        sentences = await translator.translate_partial(
            sentences,
            target_indexes=target_indexes,
            source_language="zh",
            target_language=config.target_language
        )

    return sentences
```

### 7.2 SSE 事件推送

```python
# LLM 处理过程中的 SSE 事件

# 校对开始
sse_manager.broadcast_sync(
    channel_id=f"job:{job.job_id}",
    event="llm_proofread_start",
    data={
        "mode": "sparse",  # 或 "full"
        "candidate_count": len(candidates),
        "message": f"开始 AI 校对（{len(candidates)} 句待处理）..."
    }
)

# 校对进度
sse_manager.broadcast_sync(
    channel_id=f"job:{job.job_id}",
    event="llm_proofread_progress",
    data={
        "current": i + 1,
        "total": len(candidates),
        "message": f"AI 校对中 ({i+1}/{len(candidates)})..."
    }
)

# 校对完成
sse_manager.broadcast_sync(
    channel_id=f"job:{job.job_id}",
    event="llm_proofread_complete",
    data={
        "corrected_count": corrected_count,
        "message": f"AI 校对完成（修正 {corrected_count} 句）"
    }
)

# 翻译事件
sse_manager.broadcast_sync(
    channel_id=f"job:{job.job_id}",
    event="llm_translate_start",
    data={
        "target_language": target_language,
        "message": f"开始翻译为 {target_language}..."
    }
)
```

---

## 八、前端适配

### 8.1 校对模式选择器

```vue
<template>
  <div class="proofread-selector">
    <label>AI 校对</label>
    <select v-model="settings.proofread">
      <option value="off">关闭</option>
      <option value="sparse">按需修复（推荐）</option>
      <option value="full">全文精修</option>
    </select>
    <p class="hint">{{ proofreadHint }}</p>
  </div>
</template>

<script>
export default {
  computed: {
    proofreadHint() {
      const hints = {
        off: '不进行 AI 校对，保留原始识别结果',
        sparse: '仅对低置信度句子进行校对，节省 90% 成本',
        full: '全文滑动窗口校对，适合正式出版'
      }
      return hints[this.settings.proofread]
    }
  }
}
</script>
```

### 8.2 翻译配置

```vue
<template>
  <div class="translate-settings" v-if="settings.translate !== 'off'">
    <label>目标语言</label>
    <select v-model="settings.targetLanguage">
      <option value="en">英语</option>
      <option value="ja">日语</option>
      <option value="ko">韩语</option>
      <option value="fr">法语</option>
      <option value="de">德语</option>
    </select>
  </div>
</template>
```

### 8.3 处理状态显示

```vue
<template>
  <div class="llm-status" v-if="llmStatus.active">
    <div class="status-icon">
      <span class="spinner"></span>
    </div>
    <div class="status-text">{{ llmStatus.message }}</div>
    <div class="status-progress" v-if="llmStatus.progress">
      {{ llmStatus.current }}/{{ llmStatus.total }}
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      llmStatus: {
        active: false,
        message: '',
        current: 0,
        total: 0,
        progress: false
      }
    }
  },
  methods: {
    setupSSE() {
      // ... 现有 SSE 代码 ...

      // 监听 LLM 事件
      eventSource.addEventListener('llm_proofread_start', (event) => {
        const data = JSON.parse(event.data)
        this.llmStatus = {
          active: true,
          message: data.message,
          current: 0,
          total: data.candidate_count,
          progress: true
        }
      })

      eventSource.addEventListener('llm_proofread_progress', (event) => {
        const data = JSON.parse(event.data)
        this.llmStatus.current = data.current
        this.llmStatus.message = data.message
      })

      eventSource.addEventListener('llm_proofread_complete', (event) => {
        const data = JSON.parse(event.data)
        this.llmStatus.active = false
        this.llmStatus.message = data.message
      })
    }
  }
}
</script>
```

---

## 九、配置与依赖

### 9.1 环境变量

```bash
# .env

# LLM 配置
LLM_PROVIDER=openai           # openai / anthropic / local
LLM_MODEL=gpt-4o-mini         # 模型名称
LLM_API_KEY=sk-xxx            # API Key
LLM_BASE_URL=                 # 可选：自定义 API 地址

# 校对配置
LLM_SPARSE_THRESHOLD=0.7      # 稀疏校对置信度阈值
LLM_CONTEXT_SIZE=2            # 上下文句数
LLM_WINDOW_SIZE=10            # 全量校对窗口大小
```

### 9.2 依赖安装

```bash
# requirements.txt

# LLM SDK（按需安装）
openai>=1.0.0        # OpenAI
anthropic>=0.5.0     # Anthropic
aiohttp>=3.8.0       # 本地 LLM 调用
```

---

## 十、验收标准

### 10.1 功能验收

- [ ] P1 稀疏校对正确筛选低置信度句子
- [ ] P1 三明治 Prompt 结构包含正确上下文
- [ ] P2 滑动窗口正确生成
- [ ] P2 窗口重叠正确处理
- [ ] T1 全量翻译覆盖所有句子
- [ ] T2 部分翻译仅处理指定句子
- [ ] 所有 LLM 处理后时间戳保持不变

### 10.2 性能验收

- [ ] P1 稀疏校对节省 > 80% Token
- [ ] P2 全量校对 100 句 < 30 秒
- [ ] T1 翻译 100 句 < 60 秒
- [ ] LLM 调用失败正确重试

### 10.3 对接验收

- [ ] 转录层输出可直接传入 LLM 层
- [ ] LLM 层输出保持 SentenceSegment 结构
- [ ] SSE 事件正确推送
- [ ] 前端正确显示 LLM 处理状态

---

## 十一、总结

本文档定义了独立的 LLM 校对翻译层，与转录层完美对接：

### 核心模块

| 模块 | 功能 | Token 效率 |
|------|------|-----------|
| **P1: 稀疏校对** | 仅处理低置信度句子 | 节省 90%+ |
| **P2: 全量校对** | 滑动窗口全文处理 | 适合高质量需求 |
| **T1: 全量翻译** | 全文翻译 | - |
| **T2: 部分翻译** | 指定段落翻译 | 按需使用 |

### 架构优势

1. **模块化**：校对/翻译层完全独立，可单独启用
2. **时空解耦**：LLM 仅修改文本，时间戳由 SenseVoice 确定
3. **成本控制**：P1 稀疏策略大幅降低 Token 消耗
4. **扩展性**：支持多种 LLM 提供商（OpenAI/Anthropic/本地）

### 相关文档

- 前置：[06_转录层深度优化_时空解耦架构.md](./06_转录层深度优化_时空解耦架构.md)
- 参考：[新补充.md](../新补充.md)
