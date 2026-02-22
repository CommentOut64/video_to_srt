# Phase 5: 测试矩阵与验收

> Type: Implementation Plan | Status: Ready
> Phase: 5 (回归保障)
> 目标: 把本次修复从"经验修"升级为"可回归"

## 1. 测试矩阵

### 1.1 状态机单元测试

| 测试项 | 文件 | 不变量 | 优先级 |
|--------|------|--------|--------|
| 合法迁移全覆盖 | `test_task_state_machine.py` | INV-1 | P0 |
| 终态吸收性 | `test_task_state_machine.py` | INV-2 | P0 |
| state_seq 单调递增 | `test_task_state_machine.py` | INV-3 | P0 |
| 别名映射正确 | `test_task_state_machine.py` | - | P0 |
| 并发迁移安全 | `test_task_state_machine.py` | INV-3 | P1 |

### 1.2 取消生命周期集成测试

| 测试场景 | 文件 | 不变量 | 优先级 |
|----------|------|--------|--------|
| 运行中取消不落 failed | `test_cancel_lifecycle.py` | INV-4 | P0 |
| 运行中删除不回落 paused | `test_cancel_lifecycle.py` | - | P0 |
| canceling 超时收敛 | `test_cancel_lifecycle.py` | INV-7 | P0 |
| 取消返回结构化结果 | `test_cancel_lifecycle.py` | - | P1 |
| 排队中取消直接终态 | `test_cancel_lifecycle.py` | - | P1 |
| 取消后队列推进 | `test_cancel_lifecycle.py` | INV-6 | P0 |

### 1.3 暂停恢复集成测试

| 测试场景 | 文件 | 不变量 | 优先级 |
|----------|------|--------|--------|
| 暂停后 finalized_indices 不倒退 | `test_pause_resume_lifecycle.py` | INV-5 | P0 |
| 暂停后字幕快照完整 | `test_pause_resume_lifecycle.py` | - | P0 |
| 恢复后预处理不重跑 | `test_pause_resume_lifecycle.py` | - | P1 |
| 双源合并决策可追踪 | `test_resume_state_merger.py` | - | P1 |

### 1.4 任务阶段取消点矩阵

每个预处理/转录阶段均需验证取消响应：

| 阶段 | 取消响应 | 终态 | 队列推进 | 断点可恢复 |
|------|---------|------|---------|-----------|
| FFmpeg 提取 | 原子区域结束后 | canceled | 是 | N/A（预处理前） |
| 频谱分诊 | 单元边界 | canceled | 是 | 是（跳过已分诊） |
| 全局分离 | 原子区域结束后 | canceled | 是 | 是（跳过已分离） |
| 按需分离 | 单元边界 | canceled | 是 | 是 |
| LangID | 单元边界 | canceled | 是 | 是 |
| Fast 转录 | 每 chunk 边界 | canceled | 是 | 是（跳过已处理） |
| Slow 转录 | 每 chunk 边界 | canceled | 是 | 是（含上文恢复） |
| 对齐 | 每 chunk 边界 | canceled | 是 | 是（跳过已对齐） |
| 定稿 | 每 chunk 边界 | canceled | 是 | 是 |

### 1.5 前后端一致性测试

| 场景 | 验证点 | 优先级 |
|------|--------|--------|
| SSE 正常 | 状态实时同步 | P0 |
| SSE 断连 3s | 重连后 state_seq 校验 | P1 |
| SSE 延迟 5s | 旧事件被正确拒绝 | P1 |
| HTTP 兜底 | 60s 轮询修复异常状态 | P2 |
| 快速切换页面 | SSE 不泄漏 | P2 |

### 1.6 生命周期场景矩阵

| 场景 | 起始状态 | 操作 | 期望终态 | 优先级 |
|------|---------|------|---------|--------|
| 运行中取消 | processing | cancel | canceled | P0 |
| 运行中删除 | processing | delete | removed | P0 |
| 排队取消 | queued | cancel | canceled | P0 |
| 排队删除 | queued | delete | removed | P0 |
| 暂停后恢复 | paused | resume | queued->processing | P0 |
| 暂停后取消 | paused | cancel | canceled | P1 |
| 重启后恢复 | paused(重启纠偏) | 自动 | paused(等待用户) | P1 |
| 取消超时 | canceling | timeout | force_canceled | P0 |

## 2. PBT 属性测试模板

```python
# test_lifecycle_pbt.py
from hypothesis import given, strategies as st

# 可用状态列表
STATUSES = [s.value for s in TaskStatus]
NON_TERMINAL = [s for s in STATUSES if s not in TERMINAL_STATES]

@given(
    from_status=st.sampled_from(STATUSES),
    to_status=st.sampled_from(STATUSES),
)
def test_pbt_transition_closure(from_status, to_status):
    """INV-1: 任何迁移要么在白名单内成功，要么被拒绝。不存在第三种结果。"""
    guard = TaskStateGuard()
    result = guard.transition("pbt-job", from_status, to_status)
    is_valid = to_status in VALID_TRANSITIONS.get(from_status, set())
    assert result.success == is_valid

@given(
    terminal=st.sampled_from(list(TERMINAL_STATES - {TaskStatus.REMOVED})),
    target=st.sampled_from([s for s in STATUSES if s != TaskStatus.REMOVED]),
)
def test_pbt_terminal_absorption(terminal, target):
    """INV-2: 终态（除 removed）不可迁移到其他状态。"""
    guard = TaskStateGuard()
    result = guard.transition("pbt-job", terminal, target)
    assert not result.success

@given(
    operations=st.lists(
        st.tuples(
            st.sampled_from(NON_TERMINAL),
            st.sampled_from(STATUSES),
        ),
        min_size=1,
        max_size=20,
    ),
)
def test_pbt_state_seq_monotonic(operations):
    """INV-3: state_seq 严格单调递增。"""
    guard = TaskStateGuard()
    last_seq = 0
    for from_s, to_s in operations:
        result = guard.transition("pbt-job", from_s, to_s)
        if result.success:
            assert result.state_seq > last_seq
            last_seq = result.state_seq
```

## 3. 灰度发布策略

### 3.1 灰度开关

```python
# 每 Phase 独立开关
STATE_MACHINE_GUARD_ENABLED = True   # Phase 0
CANCEL_V2_ENABLED = True             # Phase 1
RESUME_MERGER_ENABLED = True         # Phase 2
RUNNER_GATE_ENABLED = True           # Phase 3
# 前端开关通过后端 API 下发
```

### 3.2 回滚策略

| Phase | 回滚方式 | 影响范围 |
|-------|---------|---------|
| Phase 0 | 关闭 `STATE_MACHINE_GUARD_ENABLED` | 状态校验失效，回退直接赋值 |
| Phase 1 | 关闭 `CANCEL_V2_ENABLED` | 取消链路回退旧行为 |
| Phase 2 | 关闭 `RESUME_MERGER_ENABLED` | 恢复逻辑回退旧路径 |
| Phase 3 | 关闭 `RUNNER_GATE_ENABLED` | 孤儿检测失效，回退旧超时放行 |

### 3.3 发布顺序

```
1. Phase 0 上线 → 观察 1 天 → 确认日志正常
2. Phase 1 上线 → 观察 2 天 → 确认取消链路收敛
3. Phase 2 上线 → 观察 1 天 → 确认恢复路径正常
4. Phase 3 上线 → 灰度观察 → 确认 GPU 调度安全
5. Phase 4 前端上线 → 确认状态同步正常
```

## 4. 完成定义（DoD）

1. 所有 P0 测试通过
2. 状态机 PBT 属性全部满足（INV-1 到 INV-7）
3. 取消后队列推进成功率 100%（测试环境）
4. 暂停/恢复后字幕与进度恢复一致性通过
5. 前端状态与后端 state_seq 一致
6. 灰度开关可独立控制每个 Phase
7. 所有合并决策有完整日志审计

## 5. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| 状态机校验误拒合法迁移 | 中 | 任务卡住 | 影子模式先行（仅日志不拦截） |
| 旧代码有绕过状态机的直接赋值 | 高 | 校验失效 | 代码扫描 + 运行时 monkey-patch 检测 |
| 前端 state_seq 校验过严 | 低 | 状态延迟更新 | seq=0 时不校验（兼容旧后端） |
| 合并器逻辑与旧恢复路径不一致 | 中 | 恢复异常 | 灰度开关 + 双路径对比日志 |
| 孤儿超时估算不准 | 中 | GPU 空闲延迟 | 动态调整超时（基于历史任务耗时） |

## 6. Codex 评估补充建议（已采纳）

1. **统一状态命名（去别名）**: 在 Phase 0 中统一 `completed -> finished`
2. **给 runtime snapshot 增加提交时间**: 用于合并冲突决策（Phase 2 中 `transcription_hint` 含时间戳）
3. **补 `pause_pending/pause_ack` 前端处理**: 在 Phase 1 的 sseChannelManager 改动中已包含
4. **补 ADR**: 在 `llmdoc/decisions/` 中记录本次架构决策（状态机选型、合并策略选择等）
