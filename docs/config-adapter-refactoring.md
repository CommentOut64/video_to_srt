# 配置适配器重构文档

## 1. 重构背景

### 问题描述

在 v3.5 重构中，引入了新的配置结构（`PreprocessingConfig`, `TranscriptionConfig`, `RefinementConfig`, `ComputeConfig`），但代码中仍然存在大量旧版配置访问（`sensevoice.preset_id`, `demucs.enabled` 等），导致新旧配置混用，出现以下问题：

1. **流水线选择错误**：`job_queue_service.py:372` 使用旧版 `sensevoice.preset_id` 判断流水线，导致新配置无法触发双流对齐
2. **配置读取不一致**：部分代码读取新版配置，部分读取旧版配置，导致行为不可预测
3. **维护困难**：新旧配置散落在各处，难以统一管理和调试

### 影响范围

| 文件 | 问题 | 影响 |
|------|------|------|
| `job_queue_service.py:372` | 读取 `sensevoice.preset_id` 决定流水线 | 致命：新配置无法走新架构 |
| `transcription_service.py:2717` | 读取 `demucs.max_escalations` | Demucs 升级逻辑可能失效 |
| `transcription_service.py:4002` | 读取 `sensevoice.preset_id` 作语言参数 | 语言检测逻辑混乱 |
| `transcription_service.py:1359-1368` | checkpoint 只保存旧版格式 | 断点续传配置丢失 |
| `solution_matrix.py:92-151` | 只读取新版配置 | 旧版配置无法转换为 SolutionConfig |

---

## 2. 解决方案：ConfigAdapter 配置适配器

### 设计原则

1. **统一访问接口**：所有配置读取都通过 `ConfigAdapter` 进行，隐藏新旧配置的差异
2. **优先新版配置**：优先读取新版配置，回退到旧版配置
3. **零侵入性**：不修改 `JobSettings` 数据模型，只修改配置访问代码
4. **向后兼容**：确保旧版配置仍然可以正常工作

### 核心映射关系

#### 转录流水线模式

```
旧版 sensevoice.enhancement -> 新版 transcription.transcription_profile
- off           -> sensevoice_only
- smart_patch   -> sv_whisper_patch
- deep_listen   -> sv_whisper_dual
```

#### Demucs 策略

```
旧版 demucs.enabled + demucs.mode -> 新版 preprocessing.demucs_strategy
- enabled=False -> off
- mode=auto     -> auto
- mode=always   -> force_on
- mode=never    -> off
```

#### LLM 任务

```
旧版 sensevoice.proofread + translate -> 新版 refinement.llm_task + llm_scope
- proofread=off, translate=off -> llm_task=off
- proofread=sparse             -> llm_task=proofread, llm_scope=sparse
- proofread=full               -> llm_task=proofread, llm_scope=global
- translate=full               -> llm_task=translate, llm_scope=global
```

---

## 3. 实施细节

### 3.1 创建 ConfigAdapter

**文件**：`backend/app/services/config_adapter.py`

**核心方法**：

| 方法 | 功能 | 返回值 |
|------|------|--------|
| `get_transcription_profile()` | 获取转录流水线模式 | `sensevoice_only` / `sv_whisper_patch` / `sv_whisper_dual` |
| `needs_dual_alignment()` | 判断是否需要双流对齐 | `bool` |
| `get_preset_id()` | 获取预设 ID | `default` / `preset1-5` / `fast` / `balanced` / `quality` |
| `get_demucs_strategy()` | 获取 Demucs 策略 | `off` / `auto` / `force_on` |
| `is_demucs_enabled()` | 判断 Demucs 是否启用 | `bool` |
| `get_patching_threshold()` | 获取 Whisper 复核阈值 | `float` (0.0-1.0) |
| `get_llm_task()` | 获取 LLM 任务类型 | `off` / `proofread` / `translate` |
| `get_llm_scope()` | 获取 LLM 介入范围 | `sparse` / `global` |
| `get_config_source()` | 判断配置来源（调试用） | `new` / `legacy` / `mixed` / `default` |

**示例代码**：

```python
from app.services.config_adapter import ConfigAdapter

# 获取转录流水线模式
profile = ConfigAdapter.get_transcription_profile(job.settings)

# 判断是否需要双流对齐
use_dual = ConfigAdapter.needs_dual_alignment(job.settings)

# 获取 Demucs 策略
demucs_strategy = ConfigAdapter.get_demucs_strategy(job.settings)
```

### 3.2 修改 job_queue_service.py

**位置**：`backend/app/services/job_queue_service.py:370-385`

**修改前**：

```python
# 根据引擎和预设选择流水线
engine = getattr(job.settings, 'engine', 'sensevoice')
preset_id = getattr(job.settings.sensevoice, 'preset_id', 'default') if hasattr(job.settings, 'sensevoice') else 'default'

# Phase 5: 对于非 default 预设，使用双流对齐流水线
use_dual_alignment = preset_id != 'default' and engine == 'sensevoice'
```

**修改后**：

```python
# 根据引擎和配置选择流水线 (使用 ConfigAdapter 统一新旧配置)
engine = getattr(job.settings, 'engine', 'sensevoice')
use_dual_alignment = ConfigAdapter.needs_dual_alignment(job.settings)
transcription_profile = ConfigAdapter.get_transcription_profile(job.settings)
preset_id = ConfigAdapter.get_preset_id(job.settings)

# 调试日志: 输出配置来源和关键参数
config_source = ConfigAdapter.get_config_source(job.settings)
logger.debug(f"配置来源: {config_source}, profile={transcription_profile}, preset={preset_id}")
```

**影响**：修复了新配置无法触发双流对齐的致命问题

### 3.3 修改 transcription_service.py

#### 修改点 1：Demucs 最大升级次数

**位置**：`backend/app/services/transcription_service.py:2717`

**修改前**：

```python
"max_escalations": job.settings.demucs.max_escalations,
```

**修改后**：

```python
"max_escalations": ConfigAdapter.get_max_escalations(job.settings),
```

#### 修改点 2：SenseVoice 语言参数

**位置**：`backend/app/services/transcription_service.py:4002`

**修改前**：

```python
language=job.settings.sensevoice.preset_id if hasattr(job.settings, 'sensevoice') else "auto"
```

**修改后**：

```python
# 注: SenseVoice 的 language 参数实际上是自动检测，这里传入 "auto" 即可
language="auto"
```

**说明**：SenseVoice 本身支持自动语言检测，不需要从配置中读取语言参数

#### 修改点 3：Checkpoint 中的 Demucs 配置

**位置**：`backend/app/services/transcription_service.py:1352-1378`

**修改前**：

```python
data["original_settings"] = {
    # ...
    "demucs": {
        "enabled": job.settings.demucs.enabled,
        "mode": job.settings.demucs.mode,
    }
}
```

**修改后**：

```python
# 使用 ConfigAdapter 统一新旧配置
demucs_strategy = ConfigAdapter.get_demucs_strategy(job.settings)
data["original_settings"] = {
    # ...
    "demucs": {
        "enabled": ConfigAdapter.is_demucs_enabled(job.settings),
        "mode": demucs_strategy,
    }
}
```

**影响**：确保断点续传时配置正确保存和恢复

### 3.4 修改 solution_matrix.py

**位置**：`backend/app/services/solution_matrix.py:92-149`

**修改前**：

```python
# 获取 transcription_profile (默认 sensevoice_only)
transcription_profile = getattr(
    job_settings.transcription, 'transcription_profile', 'sensevoice_only'
)

# 获取 refinement 配置
llm_task = getattr(job_settings.refinement, 'llm_task', 'off')
llm_scope = getattr(job_settings.refinement, 'llm_scope', 'sparse')
```

**修改后**：

```python
from app.services.config_adapter import ConfigAdapter

# 使用 ConfigAdapter 统一获取配置 (自动兼容新旧格式)
transcription_profile = ConfigAdapter.get_transcription_profile(job_settings)
preset_id = ConfigAdapter.get_preset_id(job_settings)
llm_task = ConfigAdapter.get_llm_task(job_settings)
llm_scope = ConfigAdapter.get_llm_scope(job_settings)
target_language = ConfigAdapter.get_target_language(job_settings)
confidence_threshold = ConfigAdapter.get_patching_threshold(job_settings)
```

**影响**：`SolutionConfig.from_job_settings()` 现在可以正确处理新旧两种配置格式

---

## 4. 测试验证

### 4.1 新版配置测试

**测试场景**：前端使用新版 `task_config` 提交任务

**预期行为**：
1. `ConfigAdapter.get_config_source()` 返回 `"new"`
2. 流水线选择正确（根据 `transcription_profile`）
3. Demucs 策略正确应用
4. Checkpoint 正确保存新版配置

### 4.2 旧版配置测试

**测试场景**：使用旧版 `sensevoice` + `demucs` 配置提交任务

**预期行为**：
1. `ConfigAdapter.get_config_source()` 返回 `"legacy"`
2. 旧版配置正确映射到新版逻辑
3. 流水线选择正确（根据 `sensevoice.enhancement`）
4. Checkpoint 正确保存旧版配置

### 4.3 混合配置测试

**测试场景**：同时存在新旧配置字段

**预期行为**：
1. `ConfigAdapter.get_config_source()` 返回 `"mixed"`
2. 优先使用新版配置
3. 日志中输出配置来源警告

### 4.4 调试方法

**查看配置来源**：

```python
from app.services.config_adapter import ConfigAdapter

# 获取统一格式的配置字典
config_dict = ConfigAdapter.to_unified_dict(job.settings)
print(config_dict)

# 输出示例:
# {
#     'source': 'new',
#     'transcription_profile': 'sv_whisper_patch',
#     'needs_dual_alignment': True,
#     'preset_id': 'balanced',
#     'demucs_strategy': 'auto',
#     'whisper_model': 'medium',
#     'patching_threshold': 0.6,
#     'llm_task': 'off',
#     'llm_scope': 'sparse',
#     'target_language': 'zh',
# }
```

---

## 5. 后续工作

### 5.1 阶段三：清理废弃代码（可选）

在确认新版配置稳定运行后，可以考虑：

1. **标记旧版字段为 deprecated**：在 `JobSettings` 中添加注释
2. **前端移除旧版配置支持**：统一使用新版 `task_config`
3. **清理兼容代码**：移除 `ConfigAdapter` 中的旧版回退逻辑

### 5.2 文档更新

- [ ] 更新 `llmdoc/reference/transcription-api.md`：说明新版配置格式
- [ ] 更新 `llmdoc/architecture/sensevoice-presets.md`：说明配置映射关系
- [ ] 创建 `llmdoc/guides/config-migration.md`：配置迁移指南

---

## 6. 总结

### 修改文件清单

| 文件 | 修改类型 | 行数变化 |
|------|---------|---------|
| `backend/app/services/config_adapter.py` | 新建 | +400 |
| `backend/app/services/job_queue_service.py` | 修改 | +8 -5 |
| `backend/app/services/transcription_service.py` | 修改 | +10 -7 |
| `backend/app/services/solution_matrix.py` | 修改 | +12 -15 |

### 核心收益

1. **修复致命 Bug**：新配置现在可以正确触发双流对齐流水线
2. **统一配置访问**：所有配置读取都通过 `ConfigAdapter`，消除混乱
3. **向后兼容**：旧版配置仍然可以正常工作
4. **易于调试**：`get_config_source()` 和 `to_unified_dict()` 提供调试支持
5. **可维护性提升**：配置逻辑集中在一个文件中，易于修改和扩展

### 风险评估

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| 配置映射错误 | 低 | 中 | 详细测试新旧配置场景 |
| 性能影响 | 极低 | 低 | ConfigAdapter 方法都是简单的 getattr 调用 |
| 回归问题 | 低 | 中 | 保留旧版配置支持，确保向后兼容 |

---

**文档版本**：v1.0
**创建日期**：2025-12-16
**作者**：Claude Code
**相关 Issue**：新旧配置混用导致流水线选择错误
