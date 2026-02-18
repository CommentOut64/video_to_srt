# M3 LLM语义切分引入与标点降级方案

> Type: Architecture | Status: Active
> Version: V3.2.0+dev.20260213.05
> Scope: 引入 LLM 语义切分并渐进替代标点切分，不阻塞主流程

## 1. Summary

* **Goal**: 在保持系统实时性的前提下，将语义边界能力从“标点代理”升级到“LLM 主导”，并可量化决定是否移除标点主链。
* **核心原则**:
  - LLM 先旁路观测，再仲裁，再主用。
  - 标点先降级为备用，不立即删除。
  - 任意时刻主流程可用性优先于语义最优。

## 2. Diagram

* `Fast/Slow初版切分` -> `LLM语义边界复核` -> `修订事件SSE` -> `最终句段`
* `PunctuationProxySemanticProvider` -> `LLMSemanticProvider`（阶段替换）
* `LLM建议` + `speaker evidence` + `pause anchors` -> `DecisionEngine`

## 3. Key Components

* `backend/app/services/segmentation/segmentation_processor.py`: 消费语义锚点并裁决。
* `backend/app/services/alignment/types.py`: 扩展语义锚点契约。
* `backend/app/pipelines/async_dual_pipeline.py`: 承载异步修订事件推送。
* `frontend` SSE 消费层：展示“修订前/修订后”替换。

## 4. 语义锚点提供者设计

### 4.1 统一协议

```python
class SemanticAnchorProvider(Protocol):
    def get_anchors(self, text: str, words: list[Word]) -> list[SemanticAnchor]:
        ...
```

### 4.2 当前代理实现（标点 + 规则）

```python
class PunctuationProxySemanticProvider(SemanticAnchorProvider):
    def get_anchors(self, text, words):
        anchors = []
        for i, word in enumerate(words):
            if word.text.endswith(("。", "？", "！", ".", "?", "!")):
                anchors.append(SemanticAnchor(index=i, confidence=0.6, source="punctuation"))
            if word.text in ("但是", "然后", "所以", "不过"):
                anchors.append(SemanticAnchor(index=max(0, i - 1), confidence=0.4, source="conjunction_rule"))
        return anchors
```

### 4.3 LLM 实现（M3 目标）

```python
class LLMSemanticProvider(SemanticAnchorProvider):
    async def get_anchors(self, text, words):
        # 调用 LLM 输出语义边界候选及置信度
        ...
```

## 5. 分阶段替换策略

### 5.1 Stage A: Shadow（旁路）

1. LLM 只产出候选，不改线上结果。
2. 记录与现网切分差异，为下一阶段提供基线。

### 5.2 Stage B: Arbitration（仲裁）

1. LLM 与标点共同输入决策引擎。
2. 发生冲突时，遵循 P0 约束并保守处理。
3. `high speaker_change` 场景优先确保说话人边界不被跨越。

### 5.3 Stage C: LLM-Primary（主用）

1. LLM 成为语义锚点主来源。
2. 标点降级为 `fallback` 通道，仅在 LLM 不可用时兜底。

### 5.4 Stage D: Decide Removal（移除判定）

只有在连续任务集满足验收门槛时，才允许删除标点主链逻辑。

## 6. 实时性与异步机制

### 6.1 非阻塞要求

1. 快流先输出草稿切分，不等待 LLM。
2. 慢流产出后可先定稿，再由 LLM 异步修订。
3. LLM 超时不影响主链结束。

### 6.2 SSE 修订事件

1. 事件类型：`subtitle.segment_revision`。
2. 载荷建议字段：
  - `segment_id`
  - `old_start/old_end/old_text`
  - `new_start/new_end/new_text`
  - `revision_reason`（`llm_semantic` / `deferred_resolved`）
  - `revision_risk`

## 7. 与 M1/M2 的对接点

1. 复用 `FusedEvidence.semantic_anchors` 字段。
2. 复用 `CutPlan.deferred`，让 LLM 优先处理延迟窗口。
3. 复用 NW V2 的 `AlignmentPriorProvider`，将语义先验投喂对齐层。

## 8. 验收指标与标点移除门槛

1. 指标：
  - `semantic_boundary_accept_rate`
  - `mid_sentence_wrong_split_rate`
  - `extra_cut_count_in_gt_scope`
  - `revision_latency_p95`
2. 标点主链移除建议门槛（连续任务集）：
  - 错切率显著下降或持平。
  - 额外切分数不高于基线。
  - LLM 失败回退路径稳定。

## 9. 风险与应对

1. 风险：LLM 延迟高。  
应对：异步修订 + 超时回退 + 批量缓存。
2. 风险：LLM 边界过度激进。  
应对：P0/P1 约束先行，冲突时保守。
3. 风险：成本上升。  
应对：仅复核高风险窗口，不做全量逐句调用。

