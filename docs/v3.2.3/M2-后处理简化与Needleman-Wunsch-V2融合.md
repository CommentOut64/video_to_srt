# M2 后处理简化与 Needleman-Wunsch V2 融合

> Type: Architecture | Status: Active
> Version: V3.2.0+dev.20260213.04
> Scope: 后处理主链简化、NW V2 改造、时间轴映射契约补齐

## 1. Summary

* **Goal**: 把当前复杂后处理链路收敛为四模块主链，并在同一阶段完成 NW V2 升级，统一“切分决策”和“对齐质量”基础能力。
* **关键结果**:
  - 七层行为收口到四模块契约。
  - 双 NW 实现收敛为单内核，支持置信度与先验加权。
  - 明确 pyannote 帧时间到快流词级时间轴的映射契约。

## 2. Diagram

* `FactBuilder` -> `EvidenceFusion` -> `SoftCutDecisionEngine` -> `OutputTrace`
* `fast words + slow words + speaker facts` -> `NW V2 alignment` -> `词级统一时基`
* `pyannote frame time` -> `候选边界` -> `锚点吸附` -> `最终切分时间`

## 3. Key Components

* `backend/app/services/alignment/alignment_service.py`: NW 主实现（需升级为 V2）。
* `backend/app/services/whisper_buffer_pool.py`: 第二套 NW 实现（需收敛到同一内核）。
* `backend/app/services/segmentation/segmentation_processor.py`: 软切决策执行器。
* `backend/app/services/timeline/segmentation_service.py`: pyannote 帧时间源。
* `backend/app/services/alignment/types.py`: 四模块契约对象定义。

## 4. 后处理简化方案（7 -> 4）

### 4.1 四模块职责

1. `FactBuilder`：只生产事实，不做判断。  
输入：快慢流词、turn、基础时间戳。  
输出：`AlignedFacts`。

2. `EvidenceFusion`：把多源证据统一成可评分结构。  
输入：`AlignedFacts` + pyannote/speaker/pause/punctuation/semantic 信号。  
输出：`FusedEvidence`。

3. `SoftCutDecisionEngine`：唯一裁决层。  
输入：`AlignedFacts` + `FusedEvidence`。  
输出：`CutPlan`（含 deferred 队列）。

4. `OutputTrace`：输出句段与追踪信息。  
输入：`CutPlan`。  
输出：`FinalSegments` + `trace`。

### 4.2 与旧七层的映射原则

1. 旧层内部算法可暂时保留，但只能通过四模块契约连接。
2. 删除硬边界补丁直推链路，不允许旁路落刀。
3. 输出必须包含 `split_reason` 与 `split_risk`。

## 5. 快慢流“受影响窗口重算”规则

### 5.1 触发条件

1. 决策依赖的快流草稿被慢流覆盖。
2. 决策带 `deferred` 标签，且新锚点进入窗口。
3. speaker_change 分级变化（例如 mid 升级 high）。

### 5.2 参考伪代码

```python
def find_affected_windows(existing_decisions, slow_result):
    affected = []
    for decision in existing_decisions:
        if decision.depends_on_fast_draft and slow_result.covers(decision.time_range):
            affected.append(decision.window_id)
        if decision.risk == "deferred":
            new_anchors = slow_result.get_anchors_in_range(decision.time_range)
            if new_anchors:
                affected.append(decision.window_id)
    return sorted(set(affected))
```

## 6. Needleman-Wunsch V2（与 M2 同步交付）

### 6.1 现状问题

1. 当前 NW 主要使用固定分值（`match/mismatch/gap`），对词置信度不敏感。
2. 置信度多数在后验评分中使用，未进入 DP 路径决策。
3. 存在两套 NW 实现，维护与行为一致性成本高。

### 6.2 V2 改造目标

1. 单内核：`alignment_service` 与 `whisper_buffer_pool` 共用同一 NW V2。
2. 加权打分：将词级置信度纳入 `match/mismatch/gap`。
3. 先验接口：预留 LLM 语义先验矩阵输入。

### 6.3 建议打分公式

```python
match_score = base_match + alpha * conf_pair + beta * prior_pair
mismatch_penalty = base_mismatch * (0.5 + conf_pair)
delete_penalty = base_gap * (0.5 + conf_whisper_i)
insert_penalty = base_gap * (0.5 + conf_sv_j)
```

其中：

1. `conf_pair`：候选匹配对的综合置信度。
2. `prior_pair`：先验矩阵给出的语义配对概率。
3. `base_gap` 为负值，置信度越高，跳过代价越大。

### 6.4 LLM 预留接口

```python
class AlignmentPriorProvider(Protocol):
    def prior_matrix(
        self,
        whisper_tokens: list[str],
        sv_tokens: list[str],
    ) -> list[list[float]]:
        ...
```

默认实现返回零矩阵；M3 再引入语义先验实现。

## 7. Pyannote 帧时间到快流时间轴映射契约（补齐缺口）

### 7.1 现状事实

`segmentation_service` 当前使用：

```python
time_sec = frame_start + idx * frame_step
```

这是 pyannote 自身帧时间，不等于“快流可落刀时间”。

### 7.2 目标契约

1. 输入：`pyannote_frame_time`、`fast_words`、`pause_anchors`、`punctuation_anchors`。
2. 输出：`mapped_cut_time`、`mapping_quality`、`mapping_reason`。

### 7.3 映射流程

1. 生成候选边界：`t_raw = frame_start + idx * frame_step`。
2. 时间基校正：若属于块内时间，先加 `block_start_offset` 转为全局时间。
3. 锚点吸附：
  - 一级：窗口内最近停顿词边界（优先）。
  - 二级：语义锚点（M3 接入后启用）。
  - 三级：句末标点锚点。
4. 超窗处理：若无可接受锚点，创建 deferred，不直接硬切。
5. 追踪落盘：记录 `raw_time`、`snapped_time`、`delta_ms`、`anchor_type`、`risk`。

### 7.4 参数建议

1. `anchor_snap_tolerance_sec=0.22`
2. `timebase_offset_strategy=block_global`
3. `unresolved_frame_policy=defer_then_force`

## 8. 兼容与灰度策略

1. 新四模块与旧逻辑并行跑一段时间，进行结果比对。
2. NW V2 可配置开关，先对实验任务集启用。
3. 若指标不达标，可回退：
  - 后处理回退到 M1 兼容链路。
  - NW 回退到固定分值版本。

## 9. 验收标准

1. 四模块主链可独立跑通。
2. NW V2 单元测试覆盖匹配、错配、gap、先验注入场景。
3. pyannote 映射契约可追溯，无无因切分。
4. 指标目标：
  - `gt_cut_hit_rate` 提升。
  - `extra_cut_count_in_gt_scope` 不显著上升。
  - 对齐异常 gap 率下降。

