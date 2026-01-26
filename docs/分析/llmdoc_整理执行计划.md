# LLMDOC 整理执行计划

> **制定日期**: 2026-01-18
> **当前版本**: V3.2.0
> **规范版本**: LLMDOC Lite-Spec V2.0
> **执行者**: wgh + Claude

---

## 一、整理目标

### 1.1 核心目标

- ✅ **旧实现全部归档** - 所有 V3.0/V3.1 相关的历史文档移至 archive
- ✅ **版本统一** - 所有 Active 文档版本号更新为 V3.2.0
- ✅ **分类整合** - Agent 报告按功能模块分类组织
- ✅ **职责清晰** - 每个目录功能明确，无重复无遗漏
- ✅ **格式统一** - 所有文档遵循 LLMDOC V2.0 规范

### 1.2 量化指标

| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| Active 文档版本号一致性 | 未知 | 100% |
| Agent 报告分类率 | 0% | 100% |
| 元数据完整性 | 约50% | 100% |
| 双向链接完整性 | 约20% | 80%+ |
| 文档规范符合度 | 约60% | 95%+ |

---

## 二、整理策略

### 2.1 核心原则

1. **渐进式执行** - 分阶段，每阶段可独立验证
2. **安全优先** - 归档而非删除，保留历史追溯能力
3. **自动化辅助** - 使用脚本批量处理，减少人工错误
4. **双向验证** - 整理前后对比，确保无信息丢失

### 2.2 风险控制

**风险 1: 误归档有价值文档**
- **控制措施**: 归档前人工审查，使用分类清单
- **回滚方案**: 保留完整归档记录，可随时恢复

**风险 2: 链接失效**
- **控制措施**: 自动扫描所有文档的内部链接
- **修复方案**: 批量更新链接引用

**风险 3: 版本号错误**
- **控制措施**: 脚本自动检测，生成待审查清单
- **修复方案**: 批量替换元数据字段

---

## 三、执行计划

### Phase 0: 准备阶段 (1-2小时)

**目标**: 建立基础设施，确保整理过程可追溯

#### Step 0.1: 创建归档目录结构

```bash
mkdir -p llmdoc/archive/superseded/2026-01
mkdir -p llmdoc/archive/legacy
mkdir -p llmdoc/archive/bug-investigations/2025-12
mkdir -p llmdoc/archive/bug-investigations/2026-01
```

#### Step 0.2: 创建整理工具

**工具 1**: `/scripts/llmdoc_audit.py`
```python
# 功能：
# - 扫描所有文档元数据
# - 识别版本号不一致的文档
# - 识别缺少元数据的文档
# - 生成审计报告
```

**工具 2**: `/scripts/llmdoc_classify_agent.py`
```python
# 功能：
# - 分析 agent 报告内容
# - 根据关键词自动推荐分类
# - 生成分类清单（待人工确认）
```

**工具 3**: `/scripts/llmdoc_update_metadata.py`
```python
# 功能：
# - 批量更新文档元数据（Version/Last Updated）
# - 插入缺失的元数据字段
# - 验证元数据格式
```

#### Step 0.3: 备份当前状态

```bash
# 完整备份
cp -r llmdoc llmdoc.backup.20260118

# 创建文档清单
find llmdoc -name "*.md" > llmdoc_filelist_before.txt
```

#### Step 0.4: 审计当前状态

```bash
python scripts/llmdoc_audit.py --output docs/分析/llmdoc_audit_before.md
```

**产出**:
- 所有文档的元数据清单
- 版本号不一致列表
- 缺失元数据列表
- Agent 报告分类建议

---

### Phase 1: Architecture/Overview/Guides/Reference 整理 (2-3小时)

**目标**: 核心文档全部更新到 V3.2.0，格式统一

#### Step 1.1: 扫描并更新元数据

**对象**: `architecture/`, `overview/`, `guides/`, `reference/` 下所有文档

**操作**:
1. 使用 `llmdoc_audit.py` 生成待更新清单
2. 人工审查每篇文档:
   - 内容是否描述 V3.2.0 实现？
   - 如果是 → 更新 Version 为 V3.2.0
   - 如果不是 → 标记为待归档
3. 补充缺失的元数据字段:
   - Type
   - Status
   - Version
   - Last Updated
   - Tags

**脚本辅助**:
```bash
# 批量更新 Version 字段
python scripts/llmdoc_update_metadata.py \
  --dirs architecture,overview,guides,reference \
  --version V3.2.0 \
  --dry-run  # 先预览，确认后去掉此参数
```

**预期结果**:
- 所有 Active 文档 Version = V3.2.0
- 所有文档元数据完整

#### Step 1.2: 归档旧版本文档

**识别规则**:
- 文件名包含 `v3.0`, `v3.1`, `phase1`, `phase2` 等历史标记
- 文档内容明确描述已废弃的实现
- 文档 Status 已标记为 `Deprecated`

**归档清单**（示例）:
```
architecture/
├── whisper-migration_v3.1.0.md → archive/superseded/2026-01/
└── old-pipeline-v3.0.md → archive/legacy/

overview/
└── phase1-planning.md → archive/legacy/
```

**操作**:
```bash
# 移动文档
mv llmdoc/architecture/xxx_v3.1.0.md llmdoc/archive/superseded/2026-01/

# 更新其 Status 为 Superseded
python scripts/llmdoc_update_metadata.py \
  --file archive/superseded/2026-01/xxx_v3.1.0.md \
  --status Superseded \
  --add-field "Archive Date: 2026-01-18"
```

#### Step 1.3: 建立双向链接

**规则**:
- Architecture 文档引用对应的 Agent 报告
- Agent 报告引用对应的 Architecture 文档

**示例**:

在 `architecture/confidence-mapping-mechanism.md` 末尾添加:
```markdown
## 相关资源

* **详细调查**: `/llmdoc/agent/core-services/confidence-system/confidence_persistence_investigation_report.md`
* **代码入口**: `/backend/app/core/confidence_mapper.py`
```

---

### Phase 2: Agent 报告分类整理 (3-4小时)

**目标**: 119 个 agent 报告按功能模块分类

#### Step 2.1: 创建分类目录结构

```bash
mkdir -p llmdoc/agent/core-services/{transcription-service,model-management,confidence-system}
mkdir -p llmdoc/agent/pipeline-architecture/{async-dual-pipeline,preprocessing,workers}
mkdir -p llmdoc/agent/frontend-components/{editor,waveform,subtitle-list}
mkdir -p llmdoc/agent/audio-processing/{vad,separation,spectrum-analysis}
mkdir -p llmdoc/agent/bug-investigations/2025-12
mkdir -p llmdoc/agent/bug-investigations/2026-01
```

#### Step 2.2: 自动分类建议

```bash
python scripts/llmdoc_classify_agent.py \
  --output docs/分析/agent_classification_plan.md
```

**输出**: 每个报告的推荐分类和置信度

**示例输出**:
```
confidence_persistence_investigation_report.md
  → core-services/confidence-system/ (置信度: 95%)

async_dual_pipeline_investigation.md
  → pipeline-architecture/async-dual-pipeline/ (置信度: 98%)

subtitle_loss_investigation_v3_1_0.md
  → bug-investigations/2026-01/ (置信度: 100%, 类型: Bug已解决)
```

#### Step 2.3: 人工审查与批量移动

**审查要点**:
- 分类是否合理？
- 是否有更合适的分类？
- Bug 调查报告是否已解决？

**执行移动**:
```bash
# 移动文件
mv llmdoc/agent/confidence_persistence_investigation_report.md \
   llmdoc/agent/core-services/confidence-system/

# 批量移动（使用脚本）
python scripts/llmdoc_move_agent_reports.py \
  --plan docs/分析/agent_classification_plan.md \
  --execute
```

#### Step 2.4: 更新 Agent 报告元数据

**对所有 Agent 报告**:
```bash
python scripts/llmdoc_update_metadata.py \
  --dir agent \
  --type AgentReport \
  --ensure-fields "调查目标,Code Sections,Report,相关架构文档"
```

**特殊处理**: Bug 调查报告
```bash
# 标记为 Archived
python scripts/llmdoc_update_metadata.py \
  --dir agent/bug-investigations \
  --status Archived \
  --add-field "Resolution Date: YYYY-MM-DD"
```

---

### Phase 3: Index.md 重构 (1小时)

**目标**: 清晰的导航结构，突出 V3.2.0 状态

#### Step 3.1: 重新组织 index.md 结构

**新结构**:
```markdown
# video_to_srt_gpu 文档索引

> **当前版本**: V3.2.0
> **文档规范**: LLMDOC Lite-Spec V2.0
> **最后更新**: 2026-01-18

## 快速开始

* 新手入门: [项目概览](overview/xxx.md)
* 核心概念: [时空解耦架构](architecture/xxx.md)

## 📋 Overview (项目概览)

* V3.2.0 当前实现概览
* 核心特性和技术栈

## 🏗️ Architecture (架构设计)

* 按功能模块分类列出
* 每项标注 V3.2.0

## 📖 Guides (操作指南)

## 📚 Reference (参考资料)

## 🔍 Agent 技术档案 (按分类)

* 核心服务
* 流水线架构
* 前端组件
* 音频处理

## 📦 归档文档

* [历史版本文档索引](archive/INDEX.md)
* 仅供追溯查询，不反映当前实现
```

#### Step 3.2: 移除过时引用

- 删除所有指向归档文档的链接
- 删除历史更新记录（移至 changelog.md）

---

### Phase 4: 格式统一与验证 (1-2小时)

**目标**: 所有文档符合 LLMDOC V2.0 规范

#### Step 4.1: 批量格式检查

```bash
python scripts/check_llmdoc_format.py \
  --dir llmdoc \
  --output docs/分析/format_issues.md
```

**检查项**:
- [ ] 元数据完整性
- [ ] 层级深度（最大 H3）
- [ ] 绝对路径引用
- [ ] 列表优先（段落<3行）

#### Step 4.2: 批量修复

**常见问题批量修复**:
```bash
# 替换相对路径为绝对路径
python scripts/fix_relative_paths.py --dir llmdoc

# 统一元数据格式
python scripts/normalize_metadata.py --dir llmdoc
```

#### Step 4.3: 人工复查

**抽样检查**（每个目录抽查 20%）:
- 元数据正确性
- 内容准确性
- 链接有效性

---

### Phase 5: 创建归档索引 (30分钟)

**目标**: 归档文档可追溯

#### Step 5.1: 创建 archive/INDEX.md

```markdown
# 归档文档索引

> 此目录包含已过时或已废弃的历史文档，仅供追溯查询。

## Superseded (被替代文档)

### 2026-01

* [Whisper 迁移 V3.1.0](superseded/2026-01/whisper-migration_v3.1.0_archived.md)
  - 当前版本: [Whisper 迁移](../architecture/whisper-migration.md)
  - 替代日期: 2026-01-18

## Legacy (废弃实现)

### 音频处理

* [旧 VAD 实现](legacy/old-vad-implementation.md)
  - 废弃版本: V2.x
  - 废弃原因: 性能问题，已用 Silero VAD 替代

## Bug 调查归档

### 2026-01

* [字幕丢失问题调查](bug-investigations/2026-01/subtitle_loss_investigation_v3_1_0.md)
  - 解决日期: 2026-01-15
  - 解决方案: 智能累积算法

### 2025-12

* ...
```

---

### Phase 6: 工具集成与自动化 (1小时)

**目标**: 持续维护机制

#### Step 6.1: 集成 pre-commit hook

```bash
# .git/hooks/pre-commit
python scripts/check_llmdoc_format.py --quick
python scripts/check_version_consistency.py
```

#### Step 6.2: 设置月度审计任务

**创建**: `scripts/monthly_audit.sh`
```bash
#!/bin/bash
# 每月 1 号自动执行

python scripts/llmdoc_audit.py --full --output llmdoc/.audit-$(date +%Y%m).md
python scripts/check_outdated_docs.py --days 90
python scripts/archive_old_bugs.py --days 30
```

---

## 四、执行检查清单

### Phase 0: 准备

- [ ] 创建归档目录结构
- [ ] 开发整理工具
- [ ] 备份当前状态
- [ ] 生成初始审计报告

### Phase 1: 核心文档

- [ ] 扫描元数据完整性
- [ ] 更新 Version 为 V3.2.0
- [ ] 归档历史版本文档
- [ ] 建立双向链接

### Phase 2: Agent 分类

- [ ] 创建分类目录
- [ ] 生成自动分类建议
- [ ] 人工审查分类计划
- [ ] 执行批量移动
- [ ] 更新元数据

### Phase 3: Index 重构

- [ ] 设计新结构
- [ ] 移除过时引用
- [ ] 添加快速导航
- [ ] 创建归档索引链接

### Phase 4: 格式统一

- [ ] 批量格式检查
- [ ] 自动修复常见问题
- [ ] 人工抽样复查

### Phase 5: 归档索引

- [ ] 创建 archive/INDEX.md
- [ ] 建立历史文档链接
- [ ] 标注替代/废弃原因

### Phase 6: 自动化

- [ ] 集成 pre-commit hook
- [ ] 设置月度审计任务
- [ ] 文档化维护流程

---

## 五、验证标准

### 5.1 整理完成标准

**必须满足**:
- ✅ 所有 Active 文档 Version = V3.2.0
- ✅ Agent 报告 100% 分类
- ✅ 元数据完整率 100%
- ✅ 归档文档完整索引

**优化目标**:
- 🎯 双向链接覆盖率 80%+
- 🎯 格式规范符合度 95%+
- 🎯 文档检索效率提升 50%+

### 5.2 验证方法

**自动化验证**:
```bash
# 最终审计
python scripts/llmdoc_audit.py --verify --strict

# 链接检查
python scripts/check_all_links.py

# 格式验证
python scripts/check_llmdoc_format.py --strict
```

**人工验证**:
- [ ] 随机抽取 10 篇文档，验证内容准确性
- [ ] 模拟 LLM 检索，验证导航效率
- [ ] 检查归档文档可追溯性

---

## 六、时间估算

| 阶段 | 预计时间 | 备注 |
|------|----------|------|
| Phase 0 | 1-2小时 | 一次性投入 |
| Phase 1 | 2-3小时 | 约30-40篇文档 |
| Phase 2 | 3-4小时 | 119篇agent报告分类 |
| Phase 3 | 1小时 | index.md重构 |
| Phase 4 | 1-2小时 | 格式统一 |
| Phase 5 | 0.5小时 | 归档索引 |
| Phase 6 | 1小时 | 自动化工具 |
| **总计** | **9-13.5小时** | 可分2-3天完成 |

---

## 七、执行建议

### 7.1 分批执行

**第一批** (4-5小时):
- Phase 0 完整
- Phase 1 完整
- Phase 3 部分（基础结构）

**第二批** (4-5小时):
- Phase 2 完整（最耗时）

**第三批** (2-3小时):
- Phase 3 完成
- Phase 4 完整
- Phase 5 完整
- Phase 6 完整

### 7.2 优先级排序

**P0 (必须立即完成)**:
- Phase 0: 备份和工具准备
- Phase 1: 核心文档版本统一

**P1 (重要但可延后)**:
- Phase 2: Agent 分类
- Phase 3: Index 重构

**P2 (优化项)**:
- Phase 4: 格式统一
- Phase 5: 归档索引
- Phase 6: 自动化

---

## 八、风险应对预案

### 问题 1: 工具开发耗时超预期

**应对**: 手工操作 + 简化工具
- 使用简单的 bash 脚本代替复杂 Python 程序
- 优先处理高频操作（如元数据更新）

### 问题 2: Agent 分类决策困难

**应对**: 保守策略
- 不确定的报告暂时保留在根目录
- 标记 `TODO: classify`，后续整理

### 问题 3: 链接失效影响范围广

**应对**: 使用相对路径缓冲
- 先移动文件，暂不更新链接
- 集中一次性批量更新所有链接

---

## 九、后续维护

### 月度任务 (每月1号)

```bash
# 自动审计
bash scripts/monthly_audit.sh

# 人工审查
# 1. 检查版本号一致性
# 2. 归档超过30天的Bug报告
# 3. 更新 index.md 最近更新列表
```

### 季度任务 (每季度首月)

```bash
# 深度审计
python scripts/llmdoc_audit.py --deep --output llmdoc/.audit-Q1-2026.md

# 人工审查
# 1. 架构文档与代码一致性核对
# 2. Agent 分类合理性评估
# 3. 归档文档清理（超过12个月）
```

---

## 十、成功标准

**整理成功的标志**:

1. ✅ LLM 检索效率明显提升
   - 找到目标文档的平均时间减少 50%
   - 干扰项（过时文档）减少 80%

2. ✅ 文档维护成本降低
   - 更新文档时，明确知道改哪些文件
   - 新增功能时，快速找到文档模板

3. ✅ 历史追溯能力完整
   - 任何历史决策都能追溯到原始调查报告
   - 归档文档完整索引，查询便捷

4. ✅ 文档质量可持续
   - 自动化工具保障格式一致性
   - 定期审计发现问题及时修复

---

**计划制定完成时间**: 2026-01-18 23:55
**下一步**: 等待 wgh 审核，确认后开始执行 Phase 0
