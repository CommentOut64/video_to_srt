# 双流统一 Timed Finalization 与切分收口设计

**Status:** Proposed

**Date:** 2026-04-07

**Owner:** wgh / Codex

## 1. 背景

本轮问题已经不再是单点 bug，而是同一条运行时链路里同时存在多套“定稿输入真相”和多套“切分规则”：

1. `timeanchored success` 走 `Preparation -> AlignmentDecoder -> AlignmentPathAdapter -> Decision -> Output`
2. `fast_direct fallback` 走独立 final 提交路径
3. `slow_text_fallback` 直接构造 text-only final 句子并提交
4. `SemanticBuffer` 还保留一套偏标点驱动的草稿切分逻辑

这种结构直接带来了两类问题：

1. 分支之间的契约不同步，局部修复很容易在另一条链上炸掉。
2. 切分规则不统一，表现为“漏字幕缓解了，但句子仍然糊成一团”。

`F:\video_to_srt_gpu\jobs\p-20260407-114227-tr-test-en-1-0mgb` 已经证明上述问题同时存在：

1. 对齐 reject 后进入 `slow_text_fallback` 时，`_commit_slow_text_fallback_result()` 仍按旧契约调用 `_emit_output_layer(...)`，传入 `sentence_records=()` 和 `subtitle_batch=None`，触发：
   `ValueError: _emit_output_layer 需要 sentence_records；subtitle_batch 仅作为南向 compat 边界保留。`
2. 快流草稿里的长句没有经过统一补切，像 `#15`、`#22` 这类“内部可切但没有句末标点”的 case 仍然整块输出。

这说明继续补丁式加兜底只会让分支更多、契约更多、语义更乱。

## 2. 已确认的原则约束

本设计以以下原则为最高约束，任何实现不得违反：

1. **时间戳精确始终第一位。**
   无论是快流还是慢流 anchor 后的结果，都不能为了更好切分、更好 wording 或更好阅读感而牺牲时间戳真实性。
2. **speaker 检测开启时，speaker 切换是最高优先级切分信号。**
   但该优先级只作用于“是否在已有可靠时间边界处切开”，不能反向驱动伪造时间戳。
3. **所有 final fallback 都必须经过统一切分层。**
   不允许再有 bypass `Decision -> Output(sentence_records)` 的 final 提交旁路。
4. **切分规则统一，语言只做特化。**
   不再允许不同链路各自维护一套“英文特判 / fallback 特判 / buffer 特判”。
5. **不做过多兜底。**
   对于缺乏可靠 timed facts 的 slow text，不再继续追加 text-only final 兜底。

## 3. 根因归纳

### 3.1 `slow_text_fallback` 违反新的输出层契约

`backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`

`_commit_slow_text_fallback_result()` 仍然自己构造单句 `final_sentences`，然后直接调用：

1. `_emit_output_layer(...)`
2. `sentence_records=()`
3. `chunk_sentence_indices=()`
4. `subtitle_batch=None`

但当前正式输出层入口 `backend/app/pipelines/dual_pipeline/implementation.py::_emit_output_layer()` 已明确要求必须提供 `sentence_records`。因此该分支不是“结果质量差”，而是**契约层面必炸**。

### 3.2 `slow_text_fallback` 在语义上也已经错误

该分支的本质是：

1. 对齐 reject 或 `alignment_low_confidence`
2. 缺少可信 timeanchored path
3. 仍然拿 slow wording 构造 final 句子
4. 用 chunk 边界伪装成 final 时间范围

这直接违反“时间戳精确第一位”的原则。换句话说，即使把 crash 修掉，这条分支本身也不应继续保留。

### 3.3 快流草稿切分仍主要依赖标点

`backend/app/services/punctuation/semantic_buffer.py`

当前 `SemanticBuffer` 会：

1. 累积 pending text / pending words
2. 优先根据标点模型给出的 `split_end_indices` 切句
3. 只有在 `max_pending_chars / max_pending_duration / force_flush_duration` 等阈值触发时才强制 flush

因此当标点模型在慢速、少标点英文里只给出尾部一个句末标点时，内部本可切开的停顿不会被正式提升成切分边界，最终表现成：

1. 多句被粘成一条
2. 超长句直到 flush 才被整块输出

### 3.4 final 路由没有强制统一到同一切分入口

当前 runtime 里：

1. timeanchored final 的正式切句层是 `Decision`
2. fast fallback final 仍存在直接提交变体
3. slow fallback final 完全绕过统一切分

这意味着“同样是 final 结果”，有的走统一切分，有的自己造句，有的再走 compat 投影，结构上必然长期失真。

### 3.5 timeanchored 坏窗会把 slow 分支频繁推向错误 fallback

当前已确认的坏窗特征不是单纯“token 数差几位”，而是 decoder 在小错位后出现连续 `null_align` 塌陷。只要还保留 `slow_text_fallback`，系统就会倾向于用 slow wording 伪装 final，掩盖真正应该暴露的 timeanchored reject。

## 4. 设计目标

本方案的目标是：

1. 修复当前所有已确认问题，包括 crash、fallback 语义错误、快流草稿糊句。
2. 删除会继续制造混乱的 final 分支，减少运行时语义面。
3. 保证所有 final 结果都来自可靠 timed facts，而不是 text-only 兜底。
4. 保证所有 final fallback 都经过统一切分层。
5. 让 draft 与 final 至少共享一套正式切分策略定义，不再各自漂移。

## 5. 非目标

本方案不做以下事情：

1. 不承诺本轮重构后所有历史 bad case 一次性全部消失。
2. 不重写整个 decoder / lattice / Viterbi 主体。
3. 不把草稿和 final 在同一个类里硬并成一个大对象。
4. 不为了兜底保住 slow wording 而继续保留 text-only final 语义。

## 6. 方案选择

本轮明确采用以下方案：

1. **删除 `slow_text_fallback final`**
2. **所有 final 结果统一归一成 timed input，再进入统一切分层**
3. **`SemanticBuffer` 只保留缓冲职责，切分职责下沉到统一 split 内核**

放弃的方案：

1. 只补 crash，不改路由。原因：保留了错误语义。
2. 保留 slow/fast 多套 final fallback，只是让它们都适配新契约。原因：语义仍然分叉。

## 7. 目标态总览

目标态只允许两种 final 输入真相：

1. **Timeanchored Timed Truth**
   - 文本主导权来自对齐成功的 slow/selected truth
   - 时间事实来自可信 `AlignmentPath`
2. **Fast Timed Truth**
   - 文本与时间都来自 fast 词级事实
   - 仅在 timeanchored 不可接受时作为正式 final 回退

目标态正式链路为：

`timed truth -> unified final split layer -> sentence_records -> Output`

其中：

1. `timeanchored success` 通过 `AlignmentPathAdapter` 进入统一 final split layer
2. `timeanchored reject` 通过新的 `FastTimedIngressAdapter` 进入同一个 final split layer
3. 不再存在 `slow_text_fallback -> direct final commit`

## 8. Final 路由重构

### 8.1 正式路由矩阵

`AlignmentStageService` 的正式行为调整为：

1. `timeanchored success`
   - 进入 `AlignmentPathAdapter -> Decision -> Output`
2. `selected fast`
   - 不尝试 slow text final
   - 直接进入 `FastTimedIngressAdapter -> Decision -> Output`
3. `selected slow` 且 `timeanchored reject / alignment_low_confidence`
   - 不再进入 `slow_text_fallback`
   - 统一降级为 `FastTimedIngressAdapter -> Decision -> Output`
4. `fast timed facts` 也不可用
   - fail-closed
   - 显式报错，不产 final

### 8.2 为什么 slow reject 后要退到 fast timed final

因为此时系统唯一还能证明可靠的，是 fast 的词级时间事实，而不是 slow wording 的文本完整性。根据“时间戳精确第一位”的原则，宁可退到 fast timed final，也不能继续拿 slow text-only 结果冒充最终字幕。

### 8.3 `alignment_low_confidence` 语义调整

`alignment_low_confidence` 不再被视为“还能凑合产 slow final”的弱失败，而是正式 reject 信号。它的职责是：

1. 阻止不可信 `AlignmentPath` 进入 final
2. 触发统一的 `fast timed final` 回退
3. 把真正的 timeanchored 质量问题保留在可观测层，而不是用 slow text-only fallback 掩盖

## 9. 统一 Final 切分层

### 9.1 单一 final 切分入口

`Decision` 继续作为唯一 final 切句层，但输入契约要统一：

1. `AlignmentPathAdapter`
   - 把 timeanchored 结果投影成标准 `DecisionLayerInput`
2. `FastTimedIngressAdapter`
   - 把 fast 词级结果投影成同形 `DecisionLayerInput`

这样 `Decision` 不再区分“这是 anchor 成功的 final”还是“这是 fallback final”，它只消费标准化 timed evidence。

### 9.2 Final 切分优先级

在 `speaker detection enabled` 条件下，final 切分优先级固定为：

1. `speaker_change`
2. `strong_sentence_end_punctuation`
3. `stable_punctuation`
4. `reliable_pause_or_gap`
5. `length_or_hard_limit`
6. `no_split / tail_hold`

注意：

1. `speaker_change` 是最高优先级切分信号
2. 但它不是“时间戳重写权”
3. 若某个 speaker 边界本身没有可靠时间事实，只能拒绝该切点，不能强造边界

### 9.3 语言特化方式

语言特化只允许存在于同一套策略字段中，例如：

1. 最短句长度
2. 最小时长
3. pause 权重
4. 标点强弱映射
5. 受保护 token 规则

不再允许以“fallback 专用英文特判”“buffer 专用特判”“slow 专用特判”的形式散落在不同链路。

## 10. Draft 切分收口

### 10.1 `SemanticBuffer` 角色收缩

`backend/app/services/punctuation/semantic_buffer.py` 不再持有正式切分逻辑，只保留以下职责：

1. 文本/词时间缓冲
2. speaker change flush
3. pending 长度与时长控制
4. punctuation 调度结果透传
5. flush 触发与 chunk 聚合

### 10.2 草稿切分下沉

`SemanticBuffer` flush 后，不再依赖自己的 `split_end_indices -> _build_sentences()` 逻辑决定最终草稿句边界，而是把缓冲后的 timed words 交给统一 draft split 内核。

建议实现形态：

1. `SemanticBuffer` 产出 `BufferedTimedChunk`
2. `UnifiedSplitter` 消费 `BufferedTimedChunk`
3. 统一使用共享的 `SplitPolicySnapshot`

### 10.3 为什么这能解决 `#15 / #22`

因为现在的问题不是“后面又并回去了”，而是根本没有一个正式层去把：

1. speaker change
2. 内部可靠停顿
3. 长句硬上限
4. 最短句约束

共同参与裁决。只要把草稿切分权从 `SemanticBuffer` 的标点主导逻辑下沉到统一 split 内核，少标点英文里“整句糊住”的问题才会被系统性修正。

## 11. 统一切分策略快照

为避免 draft/final 再次漂移，本方案引入统一的 `SplitPolicySnapshot` 概念。它是一次编译、只读消费的规则快照，至少包含：

1. 是否启用 speaker 边界
2. speaker 边界优先级
3. 强句末标点集合与权重
4. 弱标点集合与权重
5. 最短句时长
6. 最短 token/char 数
7. soft pause / long pause
8. hard limit
9. 语言组
10. 保护 token 规则

消费方式：

1. final 路径：`Decision` 消费同一份快照
2. draft 路径：`UnifiedSplitter` 消费同一份快照

因此，draft 与 final 可以保留不同输入适配器，但不再允许维护两套独立规则宇宙。

## 12. 需要裁掉的分支与旧语义

### 12.1 必删分支

1. `AlignmentStageService._commit_slow_text_fallback_result()`
2. 所有绕过 `Decision -> Output(sentence_records)` 的 final 直接提交路径
3. `SemanticBuffer` 内部以 `split_end_indices` 为核心的正式草稿切句职责

### 12.2 保留但降级的兼容边界

1. `subtitle_batch`
   - 继续只作为南向 compat 投影保留
   - 不再作为内部真源
2. `SemanticBuffer`
   - 继续存在
   - 但只做缓冲/flush，不再自己主导句边界

## 13. 为支撑该方案必须同步修的 timeanchored 闭环

本方案不会重写 decoder，但必须补上一个最小闭环，否则 slow 会过度掉到 fast timed final：

### 13.1 坏窗 gate

在 decoder / alignment service 中正式暴露并消费以下坏窗信号：

1. `longest_null_align_span`
2. `null_align_ratio`
3. `reanchor_failed`

当这些指标达到拒收阈值时：

1. 不允许继续产 slow final
2. 统一进入 fast timed final

### 13.2 自适应候选窗

当出现连续低置信或连续 `null_align` 时，候选窗不再只围绕当前位置做固定窄窗搜索，而应围绕最近强锚点推导出的预测位置做受控放宽。目标不是“全局乱搜”，而是尽量阻止整串塌陷。

### 13.3 `local_repair` 只做锚点间插值

`local_repair` 只允许在左右都有可信锚点时做 span 插值。

如果：

1. 没有右锚点
2. 右锚点距离异常
3. 估计时间窗无法证明合法

则直接拒收该段，让路由进入 fast timed final，而不是继续伪造 slow 时间轴。

## 14. 关键不变量

### 14.1 Final 真相不变量

1. final 结果只能来自 `timeanchored timed truth` 或 `fast timed truth`
2. 不允许 text-only final
3. 不允许为了切分或 speaker 去重写可靠时间戳

### 14.2 路由不变量

1. `alignment_low_confidence` 不得再落入 slow text final
2. 所有 final fallback 都必须进入统一 final split layer
3. final 输出必须通过 `sentence_records -> Output`

### 14.3 切分不变量

1. speaker detection 开启时，`speaker_change` 是最高优先级切分信号
2. 语言只允许通过 policy 特化，不允许通过分支特化
3. draft 与 final 不允许继续维护两套互不收敛的规则

### 14.4 输出契约不变量

1. `_emit_output_layer()` 必须收到正式 `sentence_records`
2. `subtitle_batch` 不得再被当成内部真源
3. 不允许新分支再次直接构造 compat batch 绕过正式输出层

## 15. 影响范围

本次最小闭环建议限制在以下范围：

1. `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
2. `backend/app/pipelines/dual_pipeline/implementation.py`
3. `backend/app/services/textflow/alignment_path_adapter.py`
4. 新增 `backend/app/services/textflow/fast_timed_ingress_adapter.py`
5. `backend/app/services/punctuation/semantic_buffer.py`
6. `backend/app/services/segmentation/unified_splitter.py`
7. `backend/app/services/punctuation/final_splitter.py`
8. 相关 `Decision` / `Output` 输入契约与测试

如需新增共享 policy 编译器，也应局限在切分层内部，不扩散到无关模块。

## 16. 测试策略

### 16.1 路由测试

至少覆盖：

1. `selected slow + alignment_low_confidence -> fast timed final`
2. `selected fast -> fast timed final`
3. `timeanchored success -> Decision`
4. `timeanchored reject` 时不会再触发 `slow_text_fallback`
5. final 输出都能产出 `sentence_records`

### 16.2 契约测试

至少覆盖：

1. `_emit_output_layer()` 对所有正式路径都收到非空 `sentence_records`
2. `subtitle_batch` 仅为 compat 投影
3. `FastTimedIngressAdapter` 与 `AlignmentPathAdapter` 产出的 `DecisionLayerInput` 满足同形契约

### 16.3 切分测试

至少覆盖：

1. speaker 开启时，speaker 切换优先于 pause/标点
2. speaker 边界不足以证明时间时，不强造切点
3. 英文少标点长句会按 pause/hard limit 补切
4. 短句不会因为统一补切而被过度碎片化
5. fallback final 与 timeanchored final 在相同 policy 下得到一致的切分优先级

### 16.4 回归样例

至少复跑：

1. `F:\video_to_srt_gpu\jobs\p-20260407-114227-tr-test-en-1-0mgb`
2. 与 `#15 / #22` 对应的 checkpoint 样例
3. 已知 `alignment_low_confidence` 坏窗样例

重点观察：

1. 不再出现 `_emit_output_layer 需要 sentence_records`
2. `chunk_0005` 不再卡死
3. 快流草稿内部可切句不再整块糊住
4. slow reject 时 final 明确走 fast timed final，而不是 text-only final

## 17. 文档同步要求

实现时必须同步重写以下已过期叙述，禁止追加式修补：

1. `llmdoc/architecture/fast-worker.md`
2. `llmdoc/architecture/segmentation-alignment-services.md`
3. 如实现中改变 final 主链，还需同步 `llmdoc/architecture/dual-flow-finalization.md`

重写后的口径必须反映当前真实状态：

1. final fallback 已统一成 timed final
2. `slow_text_fallback` 已删除
3. `SemanticBuffer` 不再是正式切句真源

## 18. 实施顺序建议

建议按以下顺序实施：

1. 先删除 `slow_text_fallback` 路由，补上 `FastTimedIngressAdapter`
2. 让所有 final 路径统一进入 `Decision -> Output(sentence_records)`
3. 收缩 `SemanticBuffer` 职责，把草稿切分下沉到统一内核
4. 引入共享 `SplitPolicySnapshot`，锁住 draft/final 规则一致性
5. 最后补 timeanchored 坏窗 gate 与局部恢复，减少 slow 被动降级比例

## 19. 结论

这次修复的关键不是“再补一个 fallback”，而是正式承认两件事：

1. 没有可靠 timed facts 的 slow text，不应该再进入 final。
2. 切分规则必须统一，只允许语言特化，不允许分支特化。

因此，本方案的核心动作是：

1. 删除 `slow_text_fallback`
2. 把所有 final fallback 统一成 `fast timed final`
3. 把 `Decision` 固定为唯一 final 切分入口
4. 把 `SemanticBuffer` 收回为缓冲层
5. 用共享 policy 收口 draft/final 的切分规则

只有这样，当前 crash、fallback 语义错位、快流糊句和后续继续叠补丁变乱的问题，才能在同一个最小闭环里一起解决。
