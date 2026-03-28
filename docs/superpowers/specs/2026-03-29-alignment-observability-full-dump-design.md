# Alignment 对齐层全量可观测日志系统设计

**Status:** Proposed

**Date:** 2026-03-29

**Owner:** wgh / Codex

## 1. 背景与问题定义

当前项目已经具备两类调试输出：

1. `jobs/{project}/debug/postprocess/chunk_xxxx/*.json`
   - 由 [postprocess_trace_writer.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/services/postprocess_trace_writer.py) 负责落盘。
   - 主要记录 `10/11/20/21/30/31/40/50/60` 等阶段级快照。
2. 分层诊断输出
   - 由 [diagnostic_trace_service.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/services/diagnostic_trace_service.py) 负责生成。
   - 主要服务 Whisper / Arbitration / 四层输出诊断。

这两套输出都不足以支撑当前对齐层故障排查，核心缺口有四个：

1. **缺少 hook / slot 级真实数据**
   - 当前阶段级 payload 只能看到汇总数量和部分对象快照，无法稳定追踪每个 `PreparedTokenUnit`、`FastHook`、`AnchorCandidate`、`LocalAlignmentBlock` 的真实状态。
2. **缺少结构化 diff 主键**
   - 当前 JSON 更像“人工读文件”，而不是“可自动回归比对”的审计日志。
3. **缺少统一字段契约**
   - 现有 trace 里混杂了原始字段、派生字段和诊断字段，难以区分什么是“真实对象字段”，什么是“collector 算出来的注记”。
4. **缺少完整对齐链闭环**
   - 当前很难从 `preparation -> anchor_mount -> decision_ingress -> decision -> output_projection -> output_dispatch` 整体回答“错误在哪里首次发生、后续是如何被放大或修正的”。

因此，需要设计一套新的、独立的 **Alignment Observability** 系统，用于完整记录后处理对齐链的真实运行数据。

## 2. 设计目标

新系统必须同时满足以下目标：

1. **人工排查可读**
   - 能快速查看每个阶段的摘要。
   - 能追到每个 hook、slot、candidate、block、envelope、boundary 的完整字段。
2. **自动回归可比**
   - 同一 job 在不同 commit、不同算法分支、不同运行之间，可以按结构化主键做 diff。
3. **不污染核心算法**
   - 核心对齐逻辑不直接写文件，不在 solver / builder 内部拼接 JSON。
4. **落盘到 jobs/project**
   - 所有日志都写入对应 `jobs/{project_id_or_job_id}` 目录，和任务生命周期天然绑定。
5. **环境变量强制全开**
   - 通过单个 env 变量控制。
   - 一旦开启，就是全量 full dump。
   - 不做任务级 summary/full 的混合模式。
6. **与现有 postprocess_trace 并存**
   - 新系统是独立审计层，不继续把旧 trace writer 扩展成大杂烩。

## 3. 范围与非目标

### 3.1 范围

本设计覆盖后处理对齐链的以下阶段：

1. `10_preparation`
2. `20_anchor_mount`
3. `30_decision_ingress_package`
4. `31_decision_ingress_adapter`
5. `40_decision`
6. `50_output_projection`
7. `60_output_dispatch`

对应真实链路为：

`preparation -> anchor_mount -> decision_ingress -> decision -> output_projection -> output_dispatch`

### 3.2 非目标

首版明确不做：

1. 前端实时查看器
2. 数据库存储或远程上传
3. 跨 job 聚合分析
4. 新事件总线架构
5. 让日志系统反向参与对齐算法决策
6. 任务级 summary/full 颗粒切换
7. 自动压缩归档或对象存储迁移

## 4. 关键设计原则

### 4.1 独立于现有 `postprocess_trace`

现有 [postprocess_trace_writer.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/services/postprocess_trace_writer.py) 保留，用作轻量阶段调试层。  
新系统单独命名为 `alignment_observability`，负责正式审计输出。

角色划分：

1. `postprocess_trace`
   - 轻量阶段快照
   - 保持现有兼容性
2. `alignment_observability`
   - 全量实体日志
   - 稳定 schema
   - 支持自动 diff

### 4.2 Stage boundary 收集，不侵入算法细胞

日志采集只允许发生在：

1. [alignment_stage_service.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py)
2. [anchor_mount/service.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/anchor_mount/service.py)

不允许直接在以下模块里写文件或拼 JSON：

1. `SeedDiscovery`
2. `ChainSolver`
3. `TemporalEnvelopeBuilder`
4. `DecisionLayer`
5. `OutputProjector`

### 4.3 `raw / derived / links / diagnostics` 严格分离

所有 entity 记录都必须满足：

1. `raw`
   - 只放当前对象原生已有字段
2. `derived`
   - 只放 collector 推导字段
3. `links`
   - 只放跨对象关联
4. `diagnostics`
   - 只放状态、原因、trace 注记

### 4.4 日志系统 fail-open，元数据 fail-honest

1. 单个日志文件写失败，不阻断字幕任务主流程
2. 但 `manifest/snapshot/run_index` 必须诚实记录失败
3. 测试环境下 schema 校验必须 fail-fast

### 4.5 单一 env 强制全开

首版只认一个环境变量：

`ANCHORFLUX_ALIGNMENT_OBSERVABILITY=1`

规则：

1. env 未开启
   - 新系统完全关闭
2. env 开启
   - 全量 snapshot + entity JSONL 全部写出
   - 不读取任务级 summary/full 配置
   - 不做阶段级裁剪

## 5. 总体架构

建议新增独立包：

`backend/app/pipelines/dual_pipeline/services/alignment_observability/`

### 5.1 文件边界

1. [contracts.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/services/alignment_observability/contracts.py)
   - 定义 `Manifest / RunMeta / StageSnapshot / EntityRecord / SessionConfig`
2. [env_config.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/services/alignment_observability/env_config.py)
   - 解析 `ANCHORFLUX_ALIGNMENT_OBSERVABILITY`
3. [serializer.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/services/alignment_observability/serializer.py)
   - 负责对象到日志记录的统一序列化
4. [writer.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/services/alignment_observability/writer.py)
   - 原子写 snapshot / JSONL / manifest / run_index
5. [session.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/services/alignment_observability/session.py)
   - 定义 `AlignmentObservabilitySession` 与 `NoOpAlignmentObservabilitySession`
6. [collectors_preparation.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/services/alignment_observability/collectors_preparation.py)
   - 采集 Stage 10
7. [collectors_anchor_mount.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/services/alignment_observability/collectors_anchor_mount.py)
   - 采集 Stage 20
8. [collectors_textflow.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/services/alignment_observability/collectors_textflow.py)
   - 采集 Stage 30/31/40/50/60

### 5.2 分层职责

1. **Schema 层**
   - 定义文件与记录长什么样
2. **Collector 层**
   - 从真实业务对象提取字段
3. **Writer 层**
   - 负责落盘与索引
4. **Integration 层**
   - 只在 stage 边界调用 session

## 6. jobs 目录结构与命名

### 6.1 顶层目录

新系统写入：

`jobs/{project_id_or_job_id}/debug/alignment_observability/`

不复用现有 `debug/postprocess/`。

### 6.2 目录结构

```text
jobs/{project}/
  debug/
    alignment_observability/
      manifest.json
      run_index.json
      chunks/
        chunk_0000/
          run.json
          stages/
            10_preparation.snapshot.json
            20_anchor_mount.snapshot.json
            30_decision_ingress.snapshot.json
            31_decision_ingress_adapter.snapshot.json
            40_decision.snapshot.json
            50_output_projection.snapshot.json
            60_output_dispatch.snapshot.json
          entities/
            10_slots.jsonl
            10_hooks.jsonl
            10_punctuation_evidences.jsonl
            10_pronunciation_hints.jsonl
            10_language_runs.jsonl
            20_anchor_candidates.jsonl
            20_local_blocks.jsonl
            20_chain_committed_blocks.jsonl
            20_items.jsonl
            20_envelopes.jsonl
            20_hook_claims.jsonl
            20_cross_chunk_locks.jsonl
            20_boundary_evidences.jsonl
            20_punctuation_facts.jsonl
            20_raw_mount_trace.jsonl
            30_anchored_token_units.jsonl
            30_punctuation_facts.jsonl
            30_boundary_evidences.jsonl
            30_cross_chunk_locks.jsonl
            31_annotated_words.jsonl
            31_speaker_turns.jsonl
            31_canonical_punctuation_facts.jsonl
            31_canonical_candidate_boundaries.jsonl
            31_time_mappings.jsonl
            40_output_traces.jsonl
            40_final_sentences.jsonl
            40_subtitle_items.jsonl
            50_projected_batches.jsonl
            50_projected_subtitles.jsonl
            60_dispatch_batches.jsonl
          graphs/
            20_anchor_mount.svg
            20_anchor_mount.html
```

### 6.3 命名规则

1. stage snapshot：
   - `{stage_no}_{stage_name}.snapshot.json`
2. entity table：
   - `{stage_no}_{entity_type}.jsonl`
3. graph artifact：
   - `{stage_no}_{artifact_name}.{ext}`

这样排序天然稳定，也和现有 `10/20/30/40/50/60` 心智一致。

## 7. Manifest / Run Index / Run Meta

### 7.1 `manifest.json`

职责：

1. 标记该任务是否启用了 alignment observability
2. 列出有哪些 chunk/run 被记录
3. 记录 schema 版本、生成时间、全局错误计数

建议结构：

```json
{
  "schema_version": "alignment_observability.v1",
  "created_at": "2026-03-29T12:34:56Z",
  "updated_at": "2026-03-29T12:35:20Z",
  "job_id": "p-20260328-202014-tr-test-en-1-248i",
  "project_id": "p-20260328-202014-tr-test-en-1-248i",
  "env": {
    "alignment_observability_enabled": true
  },
  "runs": [
    {
      "run_id": "run-20260329-123456-0001",
      "chunk_index": 20,
      "window_id": "win_0020",
      "path": "chunks/chunk_0020/run.json"
    }
  ],
  "write_error_count": 0
}
```

### 7.2 `run_index.json`

职责：

1. 作为轻量 diff 索引
2. 快速定位每个 run 的 stage/entity 文件、实体计数、关键路由、关键状态

建议字段：

1. `run_id`
2. `chunk_index`
3. `window_id`
4. `stage_names`
5. `entity_file_names`
6. `entity_counts`
7. `timeline_validity`
8. `should_fallback`
9. `text_route`
10. `edge_route`
11. `final_route`
12. `partial_write`

### 7.3 `run.json`

职责：

1. 记录该 chunk/run 的头部元信息
2. 固定后续所有 snapshot/entity 的公共上下文

建议结构：

```json
{
  "run_id": "run-20260329-123456-0001",
  "job_id": "...",
  "project_id": "...",
  "chunk_index": 20,
  "window_id": "win_0020",
  "owner_chunk_id": "...",
  "owner_chunk_index": 20,
  "source_chunk_ids": ["...", "..."],
  "source_chunk_indices": [19, 20],
  "algorithm_versions": {
    "alignment_pipeline": "timeanchored_main_chain",
    "observability_schema": "alignment_observability.v1"
  },
  "routes": {
    "text_route": "slow",
    "edge_route": "slow",
    "final_route": "slow"
  }
}
```

## 8. 通用记录契约

### 8.1 Entity 统一包络

每条 `*.jsonl` 记录都必须具备统一外层：

```json
{
  "schema_version": "alignment_observability.v1",
  "record_kind": "entity",
  "stage_no": 20,
  "stage_name": "anchor_mount",
  "entity_type": "anchor_candidate",
  "entity_id": "cand:20:000123",
  "diff_key": "win_0020:candidate:3-4:7-7:lexical",
  "run_id": "run-20260329-123456-0001",
  "job_id": "...",
  "project_id": "...",
  "chunk_index": 20,
  "window_id": "win_0020",
  "owner_chunk_id": "...",
  "owner_chunk_index": 20,
  "source_chunk_ids": ["...", "..."],
  "source_chunk_indices": [19, 20],
  "created_at": "2026-03-29T12:34:56Z",
  "raw": {},
  "derived": {},
  "links": {},
  "diagnostics": {}
}
```

### 8.2 Stage Snapshot 统一包络

每个 `*.snapshot.json` 都必须至少包含：

1. `header`
2. `counts`
3. `quality`
4. `routes`
5. `files`
6. `errors`

### 8.3 主键规则

每条关键记录必须同时具备：

1. **运行内主键**
   - `run_id`
   - `stage_name`
   - `entity_type`
   - `entity_id`
2. **跨运行 diff 主键**
   - `diff_key`

### 8.4 `raw / derived / links / diagnostics` 规则

1. `raw`
   - 只放原始对象字段
2. `derived`
   - 只放 collector 派生字段
3. `links`
   - 只放跨对象关联 ID
4. `diagnostics`
   - 只放状态、原因、trace 注记

## 9. Stage 级 schema 设计

### 9.1 Stage 10: Preparation

#### Snapshot

`10_preparation.snapshot.json`

建议字段：

1. `counts`
   - `slot_count`
   - `hook_count`
   - `punctuation_evidence_count`
   - `pronunciation_hint_count`
   - `protected_unit_count`
   - `language_run_count`
2. `quality`
   - `time_base_raw_units_count`
   - `time_base_word_units_count`
   - `fallback_text_len`
3. `window`
   - `window_text`
   - `display_text`
   - `source_language`
4. `coverage`
   - `source_chunk_ids`
   - `source_chunk_indices`
   - `coverage_core_segments`
5. `compat`
   - `time_base_quality`
   - `text_truth_quality`
   - `pronunciation_report`

#### Entity tables

`10_slots.jsonl`

Raw 字段：

1. `unit_id`
2. `token_text`
3. `normalized_text`
4. `char_start`
5. `char_end`
6. `speaker_id`
7. `turn_id`
8. `source_chunk_ids`
9. `source_chunk_indices`
10. `source_unit_ids`

Derived 字段：

1. `slot_index`
2. `char_length`
3. `covered_by_protected_unit`
4. `language_run_id`
5. `overlapping_pronunciation_hint_ids`

`10_hooks.jsonl`

Raw 字段：

1. `hook_text`
2. `start`
3. `end`
4. `confidence`
5. `source_chunk_id`
6. `source_chunk_index`
7. `token_type`

Derived 字段：

1. `hook_id`
2. `hook_index`
3. `duration_ms`
4. `midpoint_ms`
5. `normalized_text`

`10_punctuation_evidences.jsonl`

Raw 字段：

1. `mark`
2. `source_char_index`
3. `attach_side`
4. `evidence_source`

Derived 字段：

1. `mapped_left_slot_id`
2. `mapped_right_slot_id`
3. `mapped_char_token_overlap`

`10_pronunciation_hints.jsonl`

Raw 字段：

1. `token_text`
2. `reading_key`
3. `language`
4. `char_start`
5. `char_end`
6. `source`

Derived 字段：

1. `hint_index`
2. `mapped_slot_ids`

`10_language_runs.jsonl`

Raw 字段：

1. `run_text`
2. `run_language`
3. `char_start`
4. `char_end`
5. `is_protected`
6. `is_foreign_island`

Derived 字段：

1. `run_index`
2. `slot_ids`

### 9.2 Stage 20: Anchor Mount

#### Snapshot

`20_anchor_mount.snapshot.json`

建议字段：

1. `counts`
   - `candidate_count`
   - `block_count`
   - `committed_block_count`
   - `item_count`
   - `envelope_count`
   - `hook_claim_count`
   - `boundary_evidence_count`
   - `cross_chunk_lock_count`
2. `quality`
   - `alignment_score`
   - `coverage_ratio`
   - `anchored_count`
   - `inferred_count`
   - `unresolved_count`
   - `largest_unresolved_span`
   - `hook_waste_ratio`
3. `state`
   - `should_fallback`
   - `timeline_validity`
4. `diagnostics`
   - `pipeline_report_summary`
   - `raw_mount_trace_present`

#### Entity tables

`20_anchor_candidates.jsonl`

Raw 字段：

1. `unit_indices`
2. `hook_indices`
3. `anchor_kind`
4. `score`
5. `is_hard`

Derived 字段：

1. `candidate_id`
2. `unit_span`
3. `hook_span`
4. `token_texts`
5. `hook_texts`
6. `candidate_rank`
7. `selected_as_primary`
8. `reject_reason`

`20_local_blocks.jsonl`

Raw 字段：

1. `block_id`
2. `unit_indices`
3. `hook_indices`
4. `score`
5. `block_kind`
6. `anchor_kind`

Derived 字段：

1. `token_texts`
2. `hook_texts`
3. `block_order`
4. `chain_compatible_prev_ids`

`20_chain_committed_blocks.jsonl`

Raw 字段：

1. 同 `LocalAlignmentBlock`

Derived 字段：

1. `chain_index`
2. `committed`
3. `transition_score_from_prev`
4. `reseed_epoch`

`20_items.jsonl`

Raw 字段：

1. `unit_id`
2. `unit_index`
3. `token_text`
4. `display_text`
5. `normalized_text`
6. `speaker_id`
7. `turn_id`
8. `source_chunk_ids`
9. `source_chunk_indices`
10. `mount_status`
11. `anchor_kind`
12. `envelope_index`
13. `source_hook_ids`
14. `match_confidence`
15. `cross_chunk_lock_ids`
16. `alignment_block_id`

Derived 字段：

1. `slot_diff_key`
2. `source_hook_count`

`20_envelopes.jsonl`

Raw 字段：

1. `unit_id`
2. `envelope_kind`
3. `left_bound`
4. `right_bound`
5. `preferred_start`
6. `preferred_end`
7. `provisional_start`
8. `provisional_end`
9. `confidence`
10. `source_hook_ids`
11. `source_chunk_ids`
12. `source_chunk_indices`
13. `cross_chunk_lock`

Derived 字段：

1. `duration_ms`
2. `bound_width_ms`
3. `provisional_width_ms`
4. `compression_flag`

`20_hook_claims.jsonl`

Raw 字段：

1. `hook_id`
2. `owner_window_id`
3. `claim_level`
4. `claim_reason`
5. `overlap_ratio`
6. `anchor_score`
7. `finalized`

Derived 字段：

1. `claimed_unit_ids`
2. `claimed_candidate_ids`

`20_cross_chunk_locks.jsonl`

Raw 字段：

1. `lock_id`
2. `unit_ids`
3. `hook_ids`
4. `reason`
5. `source_chunk_ids`
6. `source_chunk_indices`

Derived 字段：

1. `lock_span_size`

`20_boundary_evidences.jsonl`

Raw 字段：

1. `split_idx`
2. `event_time`
3. `left_end`
4. `right_start`
5. `reason`
6. `score`
7. `hard_flag`
8. `metadata`

Derived 字段：

1. `boundary_id`
2. `left_unit_id`
3. `right_unit_id`

`20_punctuation_facts.jsonl`

Raw 字段：

1. `fact_id`
2. `left_token_index`
3. `right_token_index`
4. `attach_mode`
5. `normalized_text`
6. `punct_class`
7. `confidence`
8. `source`
9. `group_id`
10. `boundary_weight`
11. `render_default`
12. `metadata`

Derived 字段：

1. `left_unit_id`
2. `right_unit_id`

`20_raw_mount_trace.jsonl`

职责：

1. 兼容旧 `metrics["raw_mount_trace"]`
2. 首版不强行扁平化全部字段

### 9.3 Stage 30: Decision Ingress Package

#### Snapshot

`30_decision_ingress.snapshot.json`

建议字段：

1. `counts`
   - `anchored_token_unit_count`
   - `punctuation_fact_count`
   - `boundary_evidence_count`
   - `cross_chunk_lock_count`
2. `quality`
   - `quality_metrics`
   - `should_fallback`
3. `coverage`
   - `source_chunk_ids`
   - `source_chunk_indices`

#### Entity tables

1. `30_anchored_token_units.jsonl`
2. `30_punctuation_facts.jsonl`
3. `30_boundary_evidences.jsonl`
4. `30_cross_chunk_locks.jsonl`

### 9.4 Stage 31: Decision Ingress Adapter

#### Snapshot

`31_decision_ingress_adapter.snapshot.json`

建议字段：

1. `counts`
   - `annotated_word_count`
   - `canonical_boundary_count`
   - `speaker_turn_count`
   - `canonical_punctuation_fact_count`
2. `quality`
   - `alignment_score`
   - `gap_ratio`
   - `candidate_boundary_count`
3. `diagnostics`
   - `compat_report`
   - `ingress_context`

#### Entity tables

1. `31_annotated_words.jsonl`
2. `31_speaker_turns.jsonl`
3. `31_canonical_punctuation_facts.jsonl`
4. `31_canonical_candidate_boundaries.jsonl`
5. `31_time_mappings.jsonl`

### 9.5 Stage 40: Decision

#### Snapshot

`40_decision.snapshot.json`

建议字段：

1. `counts`
   - `sentence_count`
   - `output_trace_count`
   - `subtitle_item_count`
2. `quality`
   - `segmentation_report`
   - `has_subtitle_batch`
   - `fallback_used`
3. `diagnostics`
   - `split_reason_stats`
   - `split_risk_stats`

#### Entity tables

1. `40_output_traces.jsonl`
2. `40_final_sentences.jsonl`
3. `40_subtitle_items.jsonl`

### 9.6 Stage 50: Output Projection

#### Snapshot

`50_output_projection.snapshot.json`

建议字段：

1. `counts`
   - `projected_batch_count`
   - `projected_item_count`
2. `projection`
   - `owner_chunk_index`
   - `source_chunk_indices`
   - `replace_scope_chunk_ids`
3. `diagnostics`
   - `decision_metadata`
   - `projection_meta`

#### Entity tables

1. `50_projected_batches.jsonl`
2. `50_projected_subtitles.jsonl`

### 9.7 Stage 60: Output Dispatch

#### Snapshot

`60_output_dispatch.snapshot.json`

建议字段：

1. `counts`
   - `dispatch_count`
   - `output_error_count`
2. `routes`
   - `text_route`
   - `edge_route`
   - `final_route`
3. `diagnostics`
   - `dispatch_payloads`
   - `error_codes`

#### Entity tables

1. `60_dispatch_batches.jsonl`

## 10. Integration 设计

### 10.1 Session 是唯一入口

对业务层只暴露 session：

```python
class AlignmentObservabilitySession:
    def record_preparation(...)
    def record_anchor_mount(...)
    def record_decision_ingress(...)
    def record_decision(...)
    def record_output_projection(...)
    def record_output_dispatch(...)
    def finalize(...)
```

env 未开启时注入 `NoOpAlignmentObservabilitySession`。

### 10.2 接入点

#### 全局配置入口

建议接在：

1. [transcription_service.py](f:/video_to_srt_gpu/backend/app/services/transcription_service.py)
2. [orchestrator.py](f:/video_to_srt_gpu/backend/app/pipelines/orchestrator.py)

职责：

1. 读取 env
2. 构造 observability config
3. 注入 pipeline host

#### Stage 接入点

建议接在：

1. [alignment_stage_service.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py)
   - 负责 Stage `10/30/31/40/50/60`
2. [anchor_mount/service.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/anchor_mount/service.py)
   - 负责 Stage `20`

### 10.3 AnchorMount 观察 bundle

为避免侵入 `SeedDiscovery / ChainSolver / EnvelopeBuilder`，建议在 `anchor_mount/service.py` 收口：

```python
@dataclass(frozen=True)
class AnchorMountObservationBundle:
    input_view: AnchorMountInputView
    punctuation_facts: tuple[PunctuationFact, ...]
    punctuation_pair_states: tuple[PunctuationPairState, ...]
    candidates: tuple[AnchorCandidate, ...]
    masked_candidates: tuple[AnchorCandidate, ...]
    blocks: tuple[LocalAlignmentBlock, ...]
    solve_result: ChainSolveResult
    items: tuple[AnchorMountItem, ...]
    envelopes: tuple[TemporalEnvelope, ...]
    hook_claims: tuple[HookClaimRecord, ...]
    cross_chunk_locks: tuple[CrossChunkLock, ...]
    boundary_evidences: tuple[BoundaryEvidence, ...]
    metrics: dict[str, Any]
    pipeline_report: PipelineReport
```

然后只在 service 末端调用：

`session.record_anchor_mount(bundle=...)`

## 11. 环境变量与配置策略

### 11.1 环境变量

首版新增：

`ANCHORFLUX_ALIGNMENT_OBSERVABILITY=1`

### 11.2 优先级

1. env 开启
   - 强制启用 alignment observability
   - 固定全量 full dump
2. env 关闭
   - 新系统完全关闭

### 11.3 与现有 debug config 的关系

当前 [job_models.py](f:/video_to_srt_gpu/backend/app/models/job_models.py) 已有：

1. `postprocess_trace_enabled`
2. `postprocess_trace_level`
3. `anchor_mount_graph`

首版策略：

1. `alignment_observability` 不复用 `postprocess_trace_level`
2. `alignment_observability` 不挂到 `job.settings.debug`
3. `anchor_mount_graph` 图产物可纳入 `graphs/` 子目录并进入 manifest

## 12. 错误处理与原子写入

### 12.1 原子写入

沿用现有 writer 的原则：

1. 先写 `.tmp`
2. 再 `replace`
3. 单文件失败不回滚已成功文件

### 12.2 失败处理

1. 单个 entity file 写失败
   - 该 stage 标记 `partial`
2. 单个 stage snapshot 写失败
   - 该 run 标记 `partial`
3. manifest 更新失败
   - logger 记录
   - 保留已落盘文件

### 12.3 诚实元数据

不能假装“日志完整”。  
必须在：

1. `manifest.json`
2. `run_index.json`
3. 对应 `stage snapshot.errors`

里显式记录：

1. `write_error_count`
2. `failed_files`
3. `partial=true`

## 13. 测试设计

### 13.1 基础设施测试

1. `test_alignment_observability_env_force_enable.py`
2. `test_alignment_observability_manifest_contract.py`
3. `test_alignment_observability_writer_atomicity.py`
4. `test_alignment_observability_jobs_layout.py`

### 13.2 Collector 测试

1. `test_alignment_observability_preparation_collector.py`
2. `test_alignment_observability_anchor_mount_collector.py`
3. `test_alignment_observability_textflow_collector.py`

### 13.3 回归测试

1. `test_job_p20260328_193846_alignment_observability_diff_keys.py`
2. `test_job_p20260328_202014_alignment_observability_stage_completeness.py`

### 13.4 关键断言

必须覆盖：

1. 各 stage snapshot 文件存在
2. 各 entity JSONL 文件存在
3. 每条关键记录都有 `entity_id + diff_key + run_id + window_id`
4. env 开启时必定落盘，关闭时完全不落
5. 写失败时 `partial` 和错误元数据正确记录

## 14. 落地顺序

### Phase 0: 基础设施

1. 新建 `alignment_observability/` 包
2. 打通 `contracts / env_config / serializer / writer / session`
3. 实现 `manifest.json / run_index.json / run.json`
4. 实现 `NoOpAlignmentObservabilitySession`

### Phase 1: Preparation + AnchorMount

1. 接入 Stage 10
2. 接入 Stage 20
3. 先保证 `slots/hooks/candidates/blocks/items/envelopes/hook_claims` 全量可见

### Phase 2: DecisionIngress + Decision

1. 接入 Stage 30/31
2. 接入 Stage 40
3. 建立“挂载结果”和“Decision 实际看到的输入”之间的可比链

### Phase 3: OutputProjection + OutputDispatch

1. 接入 Stage 50
2. 接入 Stage 60
3. 打通 `replace scope / projected batch / dispatch payload`

### Phase 4: 回归与稳定化

1. 校验 diff_key 稳定性
2. 校验 schema 完整性
3. 校验指定坏例子的可定位性

## 15. 完成定义

本设计对应的“完成”，不是“又多写了几份 debug 文件”，而是同时满足以下条件：

1. 设置 `ANCHORFLUX_ALIGNMENT_OBSERVABILITY=1` 后，所有任务都能在对应 `jobs/{project}/debug/alignment_observability/` 下生成完整目录结构。
2. `10/20/30/31/40/50/60` 各阶段都有稳定的 `snapshot.json`。
3. `slots/hooks/candidates/blocks/items/envelopes/hook_claims/anchored_token_units/final_sentences/projected_batches/dispatch_batches` 等关键实体都有独立 JSONL。
4. 每条关键记录都具备 `entity_id + diff_key + run_id + window_id + chunk_index`。
5. 日志写失败不会打断字幕任务，但 `manifest/snapshot` 会诚实记录失败。
6. 指定坏例子可以通过这套日志直接定位错误首次发生的阶段和对象。
7. 同一 job 在两次不同运行之间，可以按 `diff_key` 做结构化 diff。

## 16. 最终结论

首版 Alignment Observability 应被实现为：

1. 一个独立于 `postprocess_trace` 的审计子系统
2. 通过单一 env 全局控制
3. 打开后完整落盘后处理对齐链的真实对象数据
4. 以 `stage snapshot + entity JSONL + manifest/index` 为核心输出形态
5. 首版优先解决“真实数据看不到、回归不可比、坏例子无法精确定位”的问题

这套系统建立后，后续不论是 timeanchored 重构、speaker 桥接修复、还是 commit scope 问题追踪，都将具备稳定、可复查、可对比的底层证据。
