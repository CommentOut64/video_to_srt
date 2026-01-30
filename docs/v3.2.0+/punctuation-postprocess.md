# 标点后处理架构（统一快流/双流）

> Type: Architecture | Status: Active

## 1. Summary

* **Goal**: 用统一的多层后处理模块完成快流与双流标点增强，提升可读性并严格控制幻觉与延迟。
* **接口入口**: `postprocess_punctuation(raw_text, words, language, mode, candidates, config) -> PunctuationPostprocessResult`
* **接口出口**: `final_text + final_positions + decision_log + metrics`
* **输入结构（拟）**:
  - `raw_text`: ASR 原文（可含标点）
  - `words`: 词/字级时间戳与置信度（ASR 置信度来源）
  - `language`: `zh/en/ja/...`
  - `mode`: `fast | dual`（仅调节阈值强度，不分裂逻辑）
  - `candidates`: 候选标点列表（可为空，由模型路补齐）
  - `config`: `PunctuationPostprocessConfig`（阈值/门控/窗口等）
* **输出结构（拟）**:
  - `final_text`: 合并后的最终文本
  - `final_positions`: `PuncPosition[]`（统一位置+置信度）
  - `decision_log`: 每个标点的决策与原因（可追溯）
  - `metrics`: 统计信息（保留/新增/删除计数，候选命中率）
* **双路来源**: `raw_text` 中的原标点（ASR路）+ `clean_text` 上的候选标点（模型路），两路并行进入合并。
* **最终文本基准**: 以 `clean_text` 为基准插入标点，避免 raw_text 原标点重复。
* **保留句中标点**: 快流即定稿，允许并保留句中标点；`mode` 仅控制阈值与规则强度，不切断中间标点。
* **核心原则**: 仅增不删为基准，只有当“低置信度 + 候选缺失 + 规则失败”同时成立时才允许删除异常标点。
* **候选低分过滤**: 候选进入决策前先过滤 `candidate_conf < config.candidate_min_conf`，避免错加标点。
* **问号门控**: 仅当疑问词/语气词命中或满足高置信度与停顿条件时允许 `? / ？`。
* **句末规则**: 句末标点需满足停顿阈值或 chunk 末尾条件，防止误断句。
* **候选一致性**: 若同位置候选存在且与原标点冲突，候选置信度高且原标点置信度低时以候选为准。
* **候选检测规则**: 若原标点附近 `±candidate_window_words` 无候选且原置信度低，则可触发异常删除；若存在候选则以候选为主。
* **异常删除条件（必须同时满足）**:
  - `raw_conf < drop_raw_*_conf`
  - `candidate_missing == True`
  - `question_gate == False`（当标点为问号时）
* **评分策略（可选）**: `score = 0.4*raw_conf + 0.3*candidate_conf + 0.2*pause_score + 0.1*question_gate`，当 `score < drop_threshold` 且无候选时允许删除。
* **延迟控制**: 快流仅一次候选推理 + O(n) 规则处理；双流仅在需要时启用增强参数，不新增额外推理必选项。
* **快流/双流统一**: `mode` 控制阈值与规则强度，逻辑同源、参数不同。
* **核心流程（简化伪代码）**:
  - `clean_text = strip_punct(raw_text)`
  - `candidates = infer_candidates(clean_text) -> filter_low_conf`
  - `raw_marks = extract_raw_marks(raw_text, words)`
  - `mapped = map_candidates_to_words(candidates, words)`
  - `decisions = rule_engine(raw_marks, mapped, words, config, mode)`
  - `final_positions = merge_add_only(decisions)`
  - `final_text = apply_positions(clean_text, final_positions)`
* **默认参数（建议值，可配置）**:
  - `candidate_min_conf`: `fast=0.30`, `dual=0.25`
  - `add_mid_conf`: `fast=0.70`, `dual=0.60`
  - `add_end_conf`: `fast=0.75`, `dual=0.65`
  - `keep_raw_mid_conf`: `fast=0.60`, `dual=0.55`
  - `keep_raw_end_conf`: `fast=0.70`, `dual=0.60`
  - `drop_raw_mid_conf`: `fast=0.35`, `dual=0.30`
  - `drop_raw_end_conf`: `fast=0.50`, `dual=0.40`
  - `question_gate_min_conf`: `fast=0.85`, `dual=0.75`
  - `pause_end_min_sec`: `fast=0.35`, `dual=0.25`
  - `candidate_window_words`: `±1`
  - `conflict_window_chars`: `1`
  - `max_repeat_punct`: `1`
  - `allowed_punct_zh`: `。？！；：，、`
  - `allowed_punct_en`: `.!?,;:`

## 1.1 Interface-Level Pseudo Implementation

```python
from dataclasses import dataclass
from typing import List, Dict, Optional, Literal

Mode = Literal["fast", "dual"]


@dataclass
class WordInfo:
    word: str
    start: float
    end: float
    confidence: float


@dataclass
class CandidatePunc:
    char_index: int
    punctuation: str
    confidence: float


@dataclass
class PunctuationPostprocessConfig:
    candidate_min_conf: float
    add_mid_conf: float
    add_end_conf: float
    keep_raw_mid_conf: float
    keep_raw_end_conf: float
    drop_raw_mid_conf: float
    drop_raw_end_conf: float
    question_gate_min_conf: float
    pause_end_min_sec: float
    candidate_window_words: int
    conflict_window_chars: int
    max_repeat_punct: int
    allowed_punct_zh: str
    allowed_punct_en: str


@dataclass
class Decision:
    char_index: int
    punctuation: str
    action: Literal["keep_raw", "add_candidate", "drop_raw"]
    reason: str
    raw_conf: Optional[float]
    cand_conf: Optional[float]


@dataclass
class PunctuationPostprocessResult:
    final_text: str
    final_positions: List[CandidatePunc]
    decision_log: List[Decision]
    metrics: Dict[str, int]


def postprocess_punctuation(
    raw_text: str,
    words: List[WordInfo],
    language: str,
    mode: Mode,
    candidates: Optional[List[CandidatePunc]],
    config: PunctuationPostprocessConfig,
) -> PunctuationPostprocessResult:
    # 1) 构建 clean_text 与索引映射
    clean_text, raw_to_clean = build_clean_text(raw_text)

    # 2) 候选推理（外部传入或此处调用）
    # candidates = infer_candidates(clean_text)  # 由 PunctuationService 提供
    filtered_candidates = filter_candidates(candidates or [], language, config)

    # 3) 映射候选到词尾索引
    candidate_map = map_candidates_to_words(filtered_candidates, words, raw_to_clean)

    # 4) 提取原标点（ASR路）
    raw_marks = extract_raw_marks(raw_text, words)

    # 5) 规则引擎：置信度 + 句末规则 + 问号门控 + 候选一致性
    decisions = rule_engine(
        raw_marks=raw_marks,
        candidate_map=candidate_map,
        words=words,
        language=language,
        mode=mode,
        config=config,
    )

    # 6) 只增不删合并（满足异常条件才删除）
    final_positions = merge_add_only(decisions)

    # 7) 规范化：重复/冲突清洗
    final_positions = normalize_positions(final_positions, config)

    # 8) 生成最终文本
    final_text = apply_positions(clean_text, final_positions)

    metrics = build_metrics(decisions, final_positions)
    return PunctuationPostprocessResult(final_text, final_positions, decisions, metrics)


def filter_candidates(
    candidates: List[CandidatePunc],
    language: str,
    config: PunctuationPostprocessConfig,
) -> List[CandidatePunc]:
    allowed = config.allowed_punct_zh if language in {"zh", "yue"} else config.allowed_punct_en
    return [
        c for c in candidates
        if c.confidence >= config.candidate_min_conf and c.punctuation in allowed
    ]


def rule_engine(raw_marks, candidate_map, words, language, mode, config) -> List[Decision]:
    decisions: List[Decision] = []
    for mark in raw_marks:
        raw_conf = mark.raw_conf
        cand = candidate_map.get(mark.word_index)
        is_question = mark.punctuation in {"?", "？"}
        question_ok = pass_question_gate(mark, language, raw_conf, config)
        candidate_missing = cand is None

        # 异常删除（必须满足）
        if raw_conf < drop_threshold(mark, config, mode) and candidate_missing and (not question_ok):
            decisions.append(Decision(mark.char_index, mark.punctuation, "drop_raw",
                                      "drop_low_conf_no_candidate", raw_conf, None))
            continue

        # 冲突处理：候选更可信则替换
        if cand and conflict(mark, cand) and cand.confidence >= add_threshold(mark, config, mode) and raw_conf < keep_threshold(mark, config, mode):
            decisions.append(Decision(cand.char_index, cand.punctuation, "add_candidate",
                                      "replace_by_candidate", raw_conf, cand.confidence))
            continue

        # 保留原标点
        decisions.append(Decision(mark.char_index, mark.punctuation, "keep_raw",
                                  "keep_raw", raw_conf, cand.confidence if cand else None))

    # 补充新增（候选路）
    for cand in candidate_map.values():
        if should_add_candidate(cand, words, language, config, mode):
            decisions.append(Decision(cand.char_index, cand.punctuation, "add_candidate",
                                      "add_candidate", None, cand.confidence))
    return decisions
```

## 2. Diagram

* `ASR(raw_text, words, language)` -> `CleanTextBuilder` -> `CandidateInference` -> `CandidateFilter` -> `CandidateMapper` -> `RuleEngine(置信度/句末/问号门控/候选一致性)` -> `MergeEngine(只增不删+异常剔除)` -> `Normalizer` -> `PunctuationPostprocessResult`

## 3. Key Components

* `backend/app/services/punctuation/postprocess.py`: 标点后处理主入口（统一快流/双流模式，包含接口定义、规则调度与决策日志）。
* `backend/app/services/punctuation/postprocess.py`: `PunctuationPostprocessInput`（raw_text/words/language/mode/candidates/config）与 `PunctuationPostprocessResult`（final_text/positions/decision_log/metrics）。
* `backend/app/services/punctuation/postprocess.py`: `DecisionCodes`（如 `keep_raw`, `add_candidate`, `drop_low_conf`, `drop_no_candidate`, `drop_question_gate`）。
* `backend/app/services/punctuation/base.py`: 复用 `PuncPosition`/`PunctuationResult`，输出位置与置信度一致化。
* `backend/app/services/punctuation/service.py`: 作为候选标点推理入口，提供 `PuncPosition` 列表。
* `backend/app/pipelines/workers/fast_worker.py`: 快流调用后处理并写入 `metadata.punctuation`。
* `backend/app/services/alignment/default_aligner.py`: 双流场景复用同一后处理模块，参数模式切换为 `dual`。
* `backend/app/config/punctuation.yaml`: 新增后处理参数（阈值/问号门控/候选过滤/句末规则/模式配置）。
* `llmdoc/reference/punctuation-config.md`: 参数说明与默认值来源。
* **规则子模块（postprocess.py 内部）**:
  - `RawMarkExtractor`: 基于 `raw_text + words` 提取原标点，并计算 `raw_punct_conf=word_conf`。
  - `CandidateFilter`: 过滤 `candidate_conf < candidate_min_conf`，并按语言过滤标点集合。
  - `CandidateMapper`: 将候选标点映射到最近词尾索引（`word_end_index`）。
  - `RuleEngine`: 多信号决策（置信度/句末停顿/问号门控/候选一致性）。
  - `MergeEngine`: 只增不删合并；满足异常条件才删除原标点。
  - `Normalizer`: 合并重复标点、冲突窗口清洗、句末规范化。
* **接口对接建议**:
  - 快流：`FastWorker._apply_punctuation()` 内部调用 `postprocess_punctuation()`，以 `mode=fast` 写入 `metadata.punctuation`。
  - 双流：`DefaultAligner` 或 `AsyncDualPipeline` 在定稿写出前调用同一入口，以 `mode=dual` 覆盖最终文本。
