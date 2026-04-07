# Timeanchored Real Observation Anti-Future-Hijack Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 timeanchored 主链的 observation 时间真源恢复为真实 time base slices，消除 `mystery -> box` / `divert -> it` 这类 future hijack。

**Architecture:** 只改 `preparation -> decoder -> local_repair -> 回归测试` 这个最小闭环。preparation 不再按 canonical token 贪心前投影 observation；decoder 在真实 slices 上做单调选择，并允许 `null-align` 胜过无证据大跳；`local_repair` 只修 synthetic/null。

**Tech Stack:** Python, pytest, timeanchored_alignment pipeline

---

## Chunk 1: 锁住失败回归

### Task 1: preparation/decoder 红灯测试

**Files:**
- Modify: `backend/tests/timeanchored_alignment/test_alignment_preparation.py`
- Modify: `backend/tests/timeanchored_alignment/test_phase3_decoder_shadow.py`

- [ ] 为 protected structure 场景改写 observation 断言，要求 slices 跟随真实 time base units，而不是 canonical token
- [ ] 新增 `mystery -> box` future hijack 回归，要求 decoder 不得把 `box` 绑到未来真实 slice
- [ ] 运行定向 pytest，确认当前实现失败

## Chunk 2: 恢复真实 observation + decoder 防劫持

### Task 2: 最小闭环实现

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/preparation/assembler.py`
- Modify: `backend/app/services/timeanchored_alignment/decoder/observation_lattice.py`
- Modify: `backend/app/services/timeanchored_alignment/decoder/viterbi_decoder.py`
- Modify: `backend/app/services/timeanchored_alignment/decoder/local_repair.py`

- [ ] preparation 直接从 `window_time_base.word_units/raw_units` 生成真实 observation slices
- [ ] decoder 保留/补强 `null-align`，让无证据大跳成本高于 `null-align`
- [ ] `local_repair` 只对 synthetic/null/estimated 做 interpolation
- [ ] 运行定向 pytest，确认红灯转绿

## Chunk 3: 全链回归

### Task 3: 相关测试与样本核验

**Files:**
- Modify: `llmdoc/changelog.md`

- [ ] 运行 `backend/tests/timeanchored_alignment -q`
- [ ] 运行 `backend/tests/textflow -q`
- [ ] 记录变更与验证结果到 `llmdoc/changelog.md`
