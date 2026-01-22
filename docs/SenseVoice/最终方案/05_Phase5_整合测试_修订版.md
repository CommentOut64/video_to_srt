# Phase 5: 整合测试（修订版 v2.0）

> 目标：端到端测试和性能验证，包含时空解耦架构验证
>
> 工期：1-2天
>
> 版本更新：整合 [06_转录层深度优化_时空解耦架构](./06_转录层深度优化_时空解耦架构.md) 设计

---

## ⚠️ 重要修订

### v2.0 新增（时空解耦架构）

- ✅ **新增**：时空解耦验证（伪对齐、来源追踪）
- ✅ **新增**：流式输出系统验证
- ✅ **新增**：预设方案验证
- ✅ **新增**：警告高亮系统验证
- ✅ **新增**：SSE 统一 Tag 验证

### v1.0 修订

- ✅ **确认**：测试脚本使用现有服务
- ✅ **修正**：硬件检测测试使用 `hardware_service.py`
- ✅ **新增**：P0 优化项验证（时间戳、显存、文本清洗、熔断止损、极短音频）
- ✅ **新增**：P1 优化项验证（VAD软限制、SSE状态、Whisper initial_prompt）

---

## 一、测试策略

**个人开发原则**：
- 只做关键路径测试
- 不做完整单元测试
- 重点验证核心功能
- 快速迭代修复

**优化验证重点（新增）**：
- P0：时间戳正确性、显存管理、文本质量、熔断止损、极短音频
- P1：用户体验、操作反馈、VAD软限制

---

## 二、测试场景（扩展版）

### 场景1：纯净语音（无BGM）

**测试目标**：验证基础转录流程 + P0优化项

**输入**：
- 视频：纯人声采访/演讲（无背景音乐）
- 时长：5-10分钟

**预期结果**：
- [ ] VAD 正确切分
- [ ] 频谱检测判定为"无需分离"
- [ ] SenseVoice 直接转录
- [ ] 字幕为句级粒度（10-20字/句）
- [ ] 置信度 > 0.6
- [ ] 无 Whisper 复核
- [ ] 进度条平滑（无停滞）

**P0 优化验证（新增）**：
- [ ] **时间戳对齐**：字幕时间与视频画面一致（误差 < 0.5s）
- [ ] **文本清洗**：输出无 `<|zh|>`、`<|en|>` 等特殊标签
- [ ] **标点统一**：全角标点（或根据配置半角）
- [ ] **极短片段**：无 < 0.5s 的字幕条目

**P1 优化验证（新增）**：
- [ ] **VAD 软限制**：片段长度在 20-30s 范围内
- [ ] **SSE 状态**：前端显示 "正在转录..." 等操作文本

**验证命令**：
```bash
# 后端测试
python backend/test_scenario1_clean_speech.py

# 前端测试
# 1. 上传测试视频
# 2. 选择 SenseVoice 引擎
# 3. 开始转录
# 4. 观察进度和实时字幕
# 5. 检查字幕时间戳是否与视频对齐
```

---

### 场景2：轻度BGM

**测试目标**：验证按需分离机制 + 显存管理

**输入**：
- 视频：轻度背景音乐的视频（vlog/新闻）
- 时长：5-10分钟

**预期结果**：
- [ ] 频谱检测识别出部分片段需要分离
- [ ] 仅识别出的片段进行人声分离
- [ ] 其他片段直接转录
- [ ] 动态权重调整生效（分离权重 < 15%）
- [ ] 字幕质量良好

**P0 优化验证（新增）**：
- [ ] **显存释放**：Demucs 分离后显存及时释放
- [ ] **时间戳偏移**：分离后片段的时间戳正确（全局时间）

**P1 优化验证（新增）**：
- [ ] **SSE 分离事件**：前端显示 "正在对片段 X 进行 AI 降噪..."
- [ ] **SSE 完成事件**：显示 "片段 X 降噪完成（耗时 Xs）"

**关键指标**：
- 分离片段占比：10%-30%
- 总处理时间：< 2分钟（10分钟视频）
- 显存峰值：< 6GB（8GB GPU）

---

### 场景3：重度BGM

**测试目标**：验证熔断升级机制 + 止损点

**输入**：
- 视频：重度背景音乐（MV/音乐会）
- 时长：3-5分钟

**预期结果**：
- [ ] 频谱检测识别大量片段需要分离
- [ ] 人声分离启用（htdemucs 或 mdx_extra）
- [ ] 转录置信度合理
- [ ] 部分低置信度片段触发 Whisper 复核
- [ ] 熔断决策优先升级分离，后复核

**P0 优化验证（新增）**：
- [ ] **熔断止损**：每个片段最多重试 1 次（不会无限循环）
- [ ] **升级优先**：先升级 Demucs 模型，再考虑 Whisper 复核

**P1 优化验证（新增）**：
- [ ] **SSE 熔断事件**：显示 "检测到低质量转录，正在升级降噪模型..."
- [ ] **Whisper initial_prompt**：复核时使用前一句作为上下文

**关键指标**：
- 分离片段占比：> 50%
- 复核片段数：< 10%
- 单片段最大重试次数：1

---

### 场景4：低质量音频

**测试目标**：验证熔断机制 + Whisper复核

**输入**：
- 视频：噪音较大的视频（街头采访/会议录音）
- 时长：5分钟

**预期结果**：
- [ ] SenseVoice 初步转录
- [ ] 检测到 BGM/Noise 事件标签
- [ ] 熔断决策触发"升级分离"
- [ ] 重新分离后再转录
- [ ] 如仍低置信度，触发 Whisper 复核
- [ ] 最终字幕可用

**P0 优化验证（新增）**：
- [ ] **文本清洗**：噪音标签 `<|Noise|>` 被正确移除
- [ ] **重复字符**：无连续 3 个以上相同字符

**P1 优化验证（新增）**：
- [ ] **SSE 复核事件**：显示 "正在使用 Whisper 复核重新识别片段 X..."
- [ ] **initial_prompt 效果**：复核结果与前后文风格一致

---

### 场景5：连续处理（新增 - P0显存验证）

**测试目标**：验证显存释放机制

**输入**：
- 连续处理 3 个视频（无需重启后端）
- 每个视频 5 分钟

**预期结果**：
- [ ] 3 个视频均成功处理
- [ ] 无 CUDA OOM 错误
- [ ] 每个任务完成后显存释放

**P0 验证要点**：
- [ ] **Demucs 卸载**：任务完成后 `torch.cuda.empty_cache()` 被调用
- [ ] **显存恢复**：任务间显存占用无累积

**验证脚本**：
```python
"""
连续处理测试脚本
"""
import asyncio
import torch
from pathlib import Path

async def test_continuous_processing():
    videos = [
        "test_videos/video1.mp4",
        "test_videos/video2.mp4",
        "test_videos/video3.mp4"
    ]

    for i, video in enumerate(videos):
        print(f"\n=== 处理视频 {i+1}/3: {video} ===")

        # 记录处理前显存
        if torch.cuda.is_available():
            before_mem = torch.cuda.memory_allocated() / 1024**3
            print(f"处理前显存: {before_mem:.2f} GB")

        # 执行转录
        # ... 调用转录服务 ...

        # 记录处理后显存
        if torch.cuda.is_available():
            after_mem = torch.cuda.memory_allocated() / 1024**3
            print(f"处理后显存: {after_mem:.2f} GB")

            # 验证显存释放
            assert after_mem - before_mem < 0.5, "显存未正确释放!"

    print("\n=== 连续处理测试通过 ===")
```

---

### 场景6：极短音频片段（新增 - P0验证）

**测试目标**：验证极短音频防御机制

**输入**：
- 视频：包含快速对话/叹词的内容
- 期望产生 < 0.5s 的片段

**预期结果**：
- [ ] 极短片段被过滤或合并
- [ ] 输出字幕无 < 0.5s 的条目
- [ ] 过滤日志记录正确

**验证方法**：
```python
def validate_subtitle_duration(srt_path):
    """验证字幕时长"""
    import pysrt
    subs = pysrt.open(srt_path)

    short_count = 0
    for sub in subs:
        duration = (sub.end.seconds + sub.end.milliseconds/1000) - \
                   (sub.start.seconds + sub.start.milliseconds/1000)
        if duration < 0.5:
            print(f"[WARNING] 极短字幕: {sub.text} ({duration:.2f}s)")
            short_count += 1

    assert short_count == 0, f"存在 {short_count} 条极短字幕!"
    print("极短音频防御验证通过")
```

---

### 场景7：无GPU环境（可选）

**测试目标**：验证 CPU 模式

**输入**：
- 在无 GPU 的机器上运行
- 视频：任意纯人声视频

**预期结果**：
- [ ] 硬件检测正确识别无 GPU
- [ ] 人声分离自动禁用
- [ ] SenseVoice 使用 CPU 推理
- [ ] 转录成功完成（速度较慢）
- [ ] 前端显示硬件状态

---

## 三、P0/P1 优化综合验证清单（新增）

### 3.1 P0 验证清单（必须通过）

| 验证项 | 验证方法 | 状态 |
|--------|----------|------|
| 时间戳偏移修正 | 手动对比字幕与视频 | [ ] |
| 显存释放机制 | 连续处理 3 个视频无 OOM | [ ] |
| 文本清洗 | 检查输出无特殊标签 | [ ] |
| 标点统一 | 检查全角/半角一致 | [ ] |
| 熔断止损点 | 检查单片段重试 <= 1 | [ ] |
| 极短音频防御 | 检查无 < 0.5s 字幕 | [ ] |

### 3.2 P1 验证清单（重要优化）

| 验证项 | 验证方法 | 状态 |
|--------|----------|------|
| VAD 软限制 | 检查片段在 25-30s 边界切分 | [ ] |
| SSE 分离事件 | 前端显示操作文本 | [ ] |
| SSE 熔断事件 | 前端显示升级提示 | [ ] |
| SSE 复核事件 | 前端显示复核提示 | [ ] |
| Whisper initial_prompt | 复核结果风格一致 | [ ] |

---

## 四、性能验证

### 4.1 性能指标（扩展版）

| 指标 | 目标值 | 实际值 | 状态 |
|------|--------|--------|------|
| 10分钟视频处理时间 (GPU) | < 2分钟 | _____ | [ ] |
| 句级平均长度 | 10-20字 | _____ | [ ] |
| 置信度准确率 | > 85% | _____ | [ ] |
| 显存峰值 (8GB GPU) | < 6GB | _____ | [ ] |
| 进度条准确性 | 无停滞/回跳 | _____ | [ ] |
| **时间戳误差** | < 0.5s | _____ | [ ] |
| **连续处理稳定性** | 3个视频无OOM | _____ | [ ] |
| **极短字幕数量** | 0 | _____ | [ ] |
| **熔断重试次数** | <= 1/片段 | _____ | [ ] |

### 4.2 性能测试脚本（扩展版 - 包含 P0/P1 验证）

**路径**: `backend/test_performance_extended.py`

```python
"""
性能测试脚本（扩展版 - 包含 P0/P1 验证）
"""
import sys
import time
from pathlib import Path
import asyncio

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.transcription_service import TranscriptionService
from app.models.job_models import JobState, JobSettings
from app.services.hardware_service import get_hardware_detector, get_hardware_optimizer


class P0Validator:
    """P0 优化项验证器"""

    @staticmethod
    def validate_timestamps(srt_path: str, video_duration: float) -> bool:
        """验证时间戳是否在有效范围内"""
        import pysrt
        subs = pysrt.open(srt_path)

        for sub in subs:
            start_sec = sub.start.seconds + sub.start.milliseconds / 1000
            end_sec = sub.end.seconds + sub.end.milliseconds / 1000

            # 验证时间戳不超过视频时长
            if end_sec > video_duration + 1:  # 允许1秒误差
                print(f"[ERROR] 时间戳超出视频范围: {end_sec:.2f}s > {video_duration:.2f}s")
                return False

            # 验证时间戳递增
            if start_sec >= end_sec:
                print(f"[ERROR] 时间戳顺序错误: {start_sec:.2f} >= {end_sec:.2f}")
                return False

        print("[PASS] 时间戳验证通过")
        return True

    @staticmethod
    def validate_text_cleaning(srt_path: str) -> bool:
        """验证文本清洗"""
        import re

        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查特殊标签
        special_tags = re.findall(r'<\|.*?\|>', content)
        if special_tags:
            print(f"[ERROR] 发现特殊标签: {special_tags[:5]}")
            return False

        # 检查重复字符（连续3个以上）
        repeated = re.findall(r'(.)\1{2,}', content)
        if repeated:
            print(f"[WARNING] 发现重复字符: {repeated[:5]}")

        print("[PASS] 文本清洗验证通过")
        return True

    @staticmethod
    def validate_short_segments(srt_path: str, min_duration: float = 0.5) -> bool:
        """验证极短片段过滤"""
        import pysrt
        subs = pysrt.open(srt_path)

        short_count = 0
        for sub in subs:
            duration = (sub.end.seconds + sub.end.milliseconds/1000) - \
                       (sub.start.seconds + sub.start.milliseconds/1000)
            if duration < min_duration:
                short_count += 1

        if short_count > 0:
            print(f"[ERROR] 存在 {short_count} 条极短字幕 (< {min_duration}s)")
            return False

        print("[PASS] 极短片段过滤验证通过")
        return True


async def test_hardware_info():
    """测试硬件信息获取（使用现有服务）"""
    print("\n=== 硬件信息 ===")

    detector = get_hardware_detector()
    hardware_info = detector.detect()

    print(f"GPU: {hardware_info.gpu_name}")
    print(f"CUDA 可用: {hardware_info.cuda_available}")
    print(f"显存: {max(hardware_info.gpu_memory_mb or [0])/1024:.1f} GB")
    print(f"CPU: {hardware_info.cpu_name}")
    print(f"CPU 核心: {hardware_info.cpu_cores}")

    optimizer = get_hardware_optimizer()
    config = optimizer.get_optimization_config(hardware_info)

    print(f"\n优化配置:")
    print(f"  SenseVoice 设备: {config.sensevoice_device}")
    print(f"  启用 Demucs: {config.enable_demucs}")
    print(f"  Demucs 模型: {config.demucs_model}")
    print(f"  说明: {config.note}")


async def test_performance(video_path: str, validate_p0: bool = True):
    """
    性能测试（带 P0 验证）

    Args:
        video_path: 测试视频路径
        validate_p0: 是否进行 P0 优化验证
    """
    print(f"\n=== 性能测试: {video_path} ===")

    # 创建任务
    job = JobState(
        job_id=f"perf_test_{int(time.time())}",
        input_path=video_path,
        output_path=f"test_output/{Path(video_path).stem}_output.srt",
        settings=JobSettings(engine='sensevoice')
    )

    service = TranscriptionService()

    # 记录开始时间
    start_time = time.time()

    try:
        # 执行转录
        await service._process_video_sensevoice(job)

        # 记录结束时间
        end_time = time.time()
        elapsed = end_time - start_time

        # 计算指标
        print(f"\n性能指标:")
        print(f"  处理时间: {elapsed:.2f}秒")
        print(f"  任务状态: {job.status}")

        # 分析字幕
        output_path = job.output_path
        if Path(output_path).exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
                sentence_count = content.count('\n\n')
                print(f"  字幕数量: {sentence_count}")

            # P0 优化验证
            if validate_p0:
                print("\n=== P0 优化验证 ===")
                validator = P0Validator()

                # 获取视频时长（需要 ffprobe）
                video_duration = 600  # 假设10分钟，实际应通过 ffprobe 获取

                results = {
                    '时间戳验证': validator.validate_timestamps(output_path, video_duration),
                    '文本清洗': validator.validate_text_cleaning(output_path),
                    '极短片段': validator.validate_short_segments(output_path)
                }

                print("\n=== P0 验证结果 ===")
                for name, passed in results.items():
                    status = "PASS" if passed else "FAIL"
                    print(f"  {name}: {status}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys

    # 先测试硬件信息
    asyncio.run(test_hardware_info())

    if len(sys.argv) < 2:
        print("\n用法: python test_performance_extended.py <video_path>")
        print("示例: python test_performance_extended.py test_data/sample.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    asyncio.run(test_performance(video_path))
```

---

## 五、问题排查清单（扩展版）

### 5.1 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 字幕粒度太粗 | 分句参数不合理 | 调整 `pause_threshold`/`max_chars` |
| 进度条停滞 | 权重计算错误 | 检查 `calculate_dynamic_weights()` |
| 频繁误判 BGM | 阈值过低 | 调整 `music_score_threshold` |
| 显存不足 | 模型未卸载 | 检查模型串行化逻辑 |
| 转录质量差 | 未升级分离 | 检查熔断决策逻辑 |
| **时间戳错位** | 未加全局偏移 | 检查 `chunk_global_start` 加法 |
| **特殊标签残留** | 未调用清洗器 | 检查 `TextNormalizer.process()` |
| **连续OOM** | 显存未释放 | 检查 `torch.cuda.empty_cache()` |
| **无限熔断循环** | 止损点未生效 | 检查 `max_retry_count` 配置 |
| **极短字幕** | 过滤逻辑缺失 | 检查 `_filter_short_segments()` |
| **硬件检测失败** | 使用了错误的 API | 确认使用 `hardware_service.py` |
| **SSE 推送失败** | 使用了错误的方法 | 确认使用 `broadcast_sync()` |

### 5.2 P0 问题快速定位

```python
# 时间戳偏移问题定位
# 在 transcription_service.py 中检查：
for word in result.words:
    print(f"[DEBUG] 原始: {word.start:.2f}, 偏移后: {word.start + chunk_global_start:.2f}")
    word.start += chunk_global_start
    word.end += chunk_global_start

# 显存释放问题定位
# 在 model_preload_manager.py 中检查：
import torch
print(f"[DEBUG] 卸载前显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
del self._demucs_model
torch.cuda.empty_cache()
print(f"[DEBUG] 卸载后显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# 熔断止损问题定位
# 在 fuse_breaker.py 中检查：
print(f"[DEBUG] 片段 {segment_id} 重试次数: {state.retry_count}, 上限: {self.max_retry_count}")
```

### 5.3 调试技巧

1. **查看日志**：
   ```bash
   tail -f logs/transcription.log
   ```

2. **单步调试**：
   - 在关键方法设置断点
   - 检查中间结果

3. **临时输出**：
   ```python
   self.logger.info(f"[DEBUG] 片段 {i}: 置信度={confidence}, 决策={decision}")
   ```

4. **验证服务使用**：
   ```python
   # 在代码中添加验证
   from services.hardware_service import get_hardware_detector
   detector = get_hardware_detector()
   self.logger.info(f"使用硬件检测器: {detector.__class__.__name__}")
   ```

---

## 六、优化建议

### 6.1 参数调优

如果测试效果不理想，可调整以下参数：

#### 分句算法参数

```python
# sentence_splitter.py
config = {
    'pause_threshold': 0.4,   # 调整停顿阈值（0.3-0.6）
    'max_duration': 5.0,      # 调整最大时长（4-6秒）
    'max_chars': 30,          # 调整最大字数（25-35）
}
```

#### 频谱检测参数

```python
# audio_circuit_breaker.py
config = {
    'music_score_threshold': 0.35,  # 调整音乐性阈值（0.25-0.45）
    'history_threshold': 0.6,       # 调整惯性阈值（0.5-0.7）
}
```

#### 熔断决策参数

```python
# fuse_breaker.py
config = {
    'confidence_threshold': 0.6,   # 调整置信度阈值（0.5-0.7）
    'upgrade_threshold': 0.4,      # 调整升级阈值（0.3-0.5）
    'max_retry_count': 1,          # 止损点（建议保持为1）
}
```

#### 极短音频防御参数（新增）

```python
# transcription_service.py
MIN_SEGMENT_DURATION = 0.5  # 最小片段时长（0.3-0.8）
```

### 6.2 性能优化

如果处理速度不达标：

1. **减少 VAD 片段数**：
   - 增大 `target_segment_duration_s`

2. **批处理优化**：
   - 增加 `batch_size`（如果显存充足）

3. **并行处理**：
   - 多片段并行转录（需注意显存）

---

## 七、交付物

### 7.1 测试报告模板（扩展版）

```markdown
# SenseVoice 集成测试报告

## 测试环境
- GPU: [型号]
- 显存: [容量]
- 操作系统: [系统]
- 使用服务: ✅ hardware_service.py, ✅ sse_service.py

## P0 优化验证结果（必须全部通过）

| 验证项 | 状态 | 备注 |
|--------|------|------|
| 时间戳偏移修正 | [ ] PASS / [ ] FAIL | |
| 显存释放机制 | [ ] PASS / [ ] FAIL | |
| 文本清洗 | [ ] PASS / [ ] FAIL | |
| 标点统一 | [ ] PASS / [ ] FAIL | |
| 熔断止损点 | [ ] PASS / [ ] FAIL | |
| 极短音频防御 | [ ] PASS / [ ] FAIL | |

## P1 优化验证结果

| 验证项 | 状态 | 备注 |
|--------|------|------|
| VAD 软限制 | [ ] PASS / [ ] FAIL | |
| SSE 状态事件 | [ ] PASS / [ ] FAIL | |
| Whisper initial_prompt | [ ] PASS / [ ] FAIL | |

## 测试场景结果

### 场景1：纯净语音
- 状态: [ ] 通过 / [ ] 失败
- 处理时间: [时间]
- 字幕质量: [评价]
- 问题: [描述]

### 场景2：轻度BGM
- ...

### 场景3：重度BGM
- ...

### 场景4：低质量音频
- ...

### 场景5：连续处理（新增）
- 视频1: [ ] 成功 / [ ] 失败
- 视频2: [ ] 成功 / [ ] 失败
- 视频3: [ ] 成功 / [ ] 失败
- OOM 发生: [ ] 是 / [ ] 否

## 性能指标
- 10分钟视频处理时间: [时间]
- 句级平均长度: [字数]
- 显存峰值: [GB]
- 时间戳误差: [秒]
- 极短字幕数量: [个]

## 代码对齐验证
- [ ] 硬件检测使用 `hardware_service.py`
- [ ] SSE 推送使用 `sse_service.py`
- [ ] 模型预加载使用现有 `_global_lock`
- [ ] 动态权重集成到 `config.py`

## 问题列表
1. [问题描述]
2. [问题描述]

## 优化建议
1. [建议]
2. [建议]
```

### 7.2 最终检查清单（扩展版）

**功能验证**：
- [ ] 所有测试场景通过
- [ ] 性能指标达标
- [ ] 硬件适配正常
- [ ] 前端界面完整
- [ ] API 接口正常
- [ ] 错误处理健全
- [ ] 日志记录完善

**P0 优化验证**：
- [ ] 时间戳与视频对齐
- [ ] 连续处理 3 个视频不 OOM
- [ ] 输出文本无特殊标签
- [ ] 标点符号统一
- [ ] 极短片段被正确过滤
- [ ] 熔断重试不超过 1 次

**P1 优化验证**：
- [ ] VAD 在合理位置切分
- [ ] 前端显示操作状态文本
- [ ] Whisper 复核结果一致

**代码对齐验证**：
- [ ] 使用现有硬件服务，无重复代码
- [ ] 使用现有 SSE 服务，无错误 API
- [ ] 模型预加载使用现有锁机制

**交付准备**：
- [ ] 代码提交 git
- [ ] 文档更新完成

---

## 八、上线准备

### 8.1 配置检查

```python
# config.py
SENSEVOICE_DEFAULT_CONFIG = {
    'use_onnx': True,
    'quantization': 'int8',
    'confidence_threshold': 0.6,
    'enable_demucs': 'auto',  # auto/true/false
}

# P0 优化配置（新增）
P0_OPTIMIZATION_CONFIG = {
    'min_segment_duration': 0.5,  # 极短音频阈值
    'max_retry_count': 1,         # 熔断止损点
    'enable_text_cleaning': True, # 文本清洗开关
    'punctuation_style': 'fullwidth',  # 标点风格
}
```

### 8.2 文档更新

- [ ] 更新 README.md
- [ ] 更新用户手册
- [ ] 更新 API 文档
- [ ] 更新 CHANGELOG
- [ ] **记录代码复用情况（重要）**

### 8.3 依赖更新

```bash
# requirements.txt
onnxruntime-gpu>=1.16.0
# 或
onnxruntime>=1.16.0  # CPU only
```

---

## 九、后续优化（可选）

### 9.1 功能增强

1. **ONNX 模型实际推理**：
   - 当前返回模拟数据
   - 需实现真实推理逻辑

2. **Whisper 复核完善**：
   - 当前仅返回原句子
   - 需实现真实的 Whisper 转录

3. **模型自动下载**：
   - 首次运行自动下载 ONNX 模型

### 9.2 用户体验优化

1. **进度预估更准确**：
   - 基于历史数据预测处理时间

2. **字幕编辑器**：
   - 支持在线编辑字幕

3. **批量处理**：
   - 支持多视频批量转录

---

## 十、总结

完成 Phase 5（修订版）后，SenseVoice 集成项目基本完成。主要成果：

### 核心功能成果

1. SenseVoice ONNX 服务
2. 智能分句算法
3. 智能熔断机制
4. 动态权重调整
5. 硬件自适应
6. 前端界面完整
7. 端到端测试通过
8. **代码与现有系统完美对齐**

### P0 优化成果（新增）

- **时间戳偏移修正**：VAD 切分后加上全局偏移量
- **显存释放机制**：Demucs 卸载后 `torch.cuda.empty_cache()`
- **文本清洗与标点统一**：移除特殊标签，统一标点风格
- **熔断止损点**：`max_retry_count = 1` 防止无限循环
- **极短音频防御**：过滤 < 0.5s 片段

### P1 优化成果（新增）

- **VAD 软限制**：25-30s 搜索窗口寻找最佳切分点
- **SSE 状态事件**：实时显示操作文本（分离、熔断、复核）
- **Whisper initial_prompt**：复核时使用前一句作为上下文

### 关键修订点

- 复用现有 `hardware_service.py` 而非创建重复代码
- 使用正确的 SSE API（`get_sse_manager()` + `broadcast_sync()`）
- 扩展现有类而非创建新类
- 保持向后兼容

**总工期**：9-14天

**下一步**：根据实际使用反馈持续优化和迭代。
