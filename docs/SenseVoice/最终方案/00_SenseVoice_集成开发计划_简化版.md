# SenseVoice 集成开发计划（简化版）

> 个人开发、小项目快速集成方案

---

## 一、核心目标

1. **SenseVoice ONNX (INT8)** 作为主转录引擎
2. **智能熔断机制**：熔断升级优先于 Whisper 复核
3. **分句算法**：句级粒度字幕输出（非 VAD 粒度）
4. **动态权重**：进度条根据实际场景调整
5. **硬件自适应**：无 GPU 时禁用人声分离

---

## 二、解决的核心问题

| 问题 | 解决方案 |
|------|----------|
| 进度条僵化 | 动态权重计算（根据分离/复核片段数调整） |
| 字幕粒度粗 | 分句算法（标点+停顿+长度） |
| 熔断割裂 | 熔断决策器（升级优先于复核） |
| VAD 定位模糊 | 物理切分（防幻觉+显存保护） |

---

## 三、整体架构

```
音频提取 → VAD物理切分(15-30s) → 频谱预判 → 按需分离(Demucs)
    → SenseVoice-ONNX转录 → 分句算法 → 置信度评估
    → [低置信度] → 熔断决策(升级优先) → Whisper复核
    → 句级SSE推送 → 字幕生成
```

---

## 四、开发阶段（5个Phase）

### Phase 1: 基础能力（2-3天）
- SenseVoice ONNX 服务
- 分句算法
- 硬件检测

### Phase 2: 智能熔断（2-3天）
- 频谱指纹检测器
- 熔断决策器
- 动态权重计算

### Phase 3: 转录服务重构（3-4天）

- VAD 物理切分
- SenseVoice + 分句集成
- 熔断 + 复核流程
- 句级 SSE 推送

### Phase 4: 前端适配（1-2天）
- 引擎选择器
- 硬件状态显示
- 实时字幕预览

### Phase 5: 整合测试（1-2天）

- 端到端测试（3-5个典型场景）
- 性能验证
- 文档完善

---

## 五、关键技术决策

1. **SenseVoice 必须用 ONNX Runtime (INT8 量化)**
2. **无 GPU 默认禁用人声分离**
3. **熔断升级优先于 Whisper 复核**
4. **VAD 负责物理切分，分句算法负责语义切分**
5. **分句条件**：
   - 标点符号（`。？！`）必切
   - 长停顿（>0.4s）必切
   - 强制长度（5秒或30字）强制切

---

## 六、硬件适配策略

| 硬件条件 | 配置 |
|----------|------|
| 无 GPU | SenseVoice CPU + 禁用分离 |
| GPU < 4GB | SenseVoice GPU + 强BGM时用 htdemucs |
| GPU 4-8GB | SenseVoice GPU + htdemucs |
| GPU >= 8GB | SenseVoice GPU + mdx_extra |

---

## 七、文件结构

```
backend/app/services/
  ├── sensevoice_onnx_service.py    # Phase 1
  ├── sentence_splitter.py          # Phase 1
  ├── hardware_detector.py          # Phase 1
  ├── audio_circuit_breaker.py      # Phase 2
  ├── fuse_breaker.py               # Phase 2
  └── transcription_service.py      # Phase 3 (重构)

backend/app/models/
  └── sensevoice_models.py          # Phase 1

backend/app/core/
  └── config.py                     # Phase 2 (动态权重)

frontend/src/views/
  └── TaskListView.vue              # Phase 4
```

---

## 八、开发原则（个人开发）

1. **快速迭代**：每个 Phase 完成后立即测试核心功能
2. **简化测试**：只做关键路径测试，不做完整单元测试
3. **实用主义**：参数可以先硬编码，后期再抽配置
4. **渐进优化**：先跑通流程，再优化性能

---

## 九、详细开发文档

- [Phase 1: 基础能力搭建](./01_Phase1_基础能力搭建.md)
- [Phase 2: 智能熔断集成](./02_Phase2_智能熔断集成.md)
- [Phase 3: 转录服务重构](./03_Phase3_转录服务重构.md)
- [Phase 4: 前端适配](./04_Phase4_前端适配.md)
- [Phase 5: 整合测试](./05_Phase5_整合测试.md)

---

## 十、总工期估算

- **Phase 1**: 2-3天
- **Phase 2**: 2-3天
- **Phase 3**: 3-4天
- **Phase 4**: 1-2天
- **Phase 5**: 1-2天

**总计**: 9-14天（个人开发）
