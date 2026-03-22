# Timeanchored Alignment 测试样本策略

## 1. Fixture 分层

### 1.1 纯合成 Fixture（长期保留）
- 目标：验证契约、边界值、枚举与生命周期，不依赖真实模型输出。
- 适用：`contracts`、`time_base_builder`、`fast_worker` 的单元测试。
- 约束：
  - 不引入真实音频或敏感文本。
  - 数据规模最小化（通常 5~20 帧 logits 即可）。

### 1.2 脱敏采样 Fixture（按阶段维护）
- 目标：覆盖 `TimeBasePackage / TextTruthPackage / island-run / mixed fallback` 的真实形态。
- 适用：shadow 对比、语言 run 相关回归。
- 约束：
  - 只能保留脱敏后的统计/片段，不保留可还原用户身份的信息。
  - 样本应可复现，不依赖在线模型。

### 1.3 生命周期约定
- 长期保留：合成 fixture、契约 fixture、最小化脱敏样本。
- 临时保留：仅为 shadow 采样验证服务的中间样本；在对应阶段完成后清理或归档。

## 2. 存放与命名规则

### 2.1 CTC 样本
- 目录：`backend/tests/timeanchored_alignment/fixtures/ctc/`
- 命名：`<lang>_<scene>_<version>.json|npy`
- 示例：`zh_greeting_v1.npy`、`en_subword_overlap_v1.npy`

### 2.2 Whisper 文本样本
- 目录：`backend/tests/timeanchored_alignment/fixtures/whisper/`
- 命名：`<lang>_<style>_<version>.json`
- 示例：`ja_formal_short_v1.json`

### 2.3 脱敏真实样本
- 目录：`backend/tests/timeanchored_alignment/fixtures/sanitized/`
- 命名：`sample_<domain>_<id>_v<ver>.json`
- 约束：
  - 去除人名、地名、账号、联系方式。
  - 仅保留验证链路所需最小字段。
  - 优先裁剪为单 chunk 或单窗口。

## 3. 最小化原则
- 优先用合成样本表达行为；只有合成样本无法覆盖时才引入脱敏样本。
- 每个样本必须有唯一测试意图，避免“一个样本覆盖所有场景”的耦合。
- 不允许将完整 `ctc_logits` 长期存入仓库；仅保留可复现最小矩阵或统计摘要。
