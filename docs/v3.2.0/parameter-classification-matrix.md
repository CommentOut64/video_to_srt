# 参数分类矩阵文档

> **文档版本**: V3.2.0+dev.20260118.02
> **创建日期**: 2026-01-18
> **文档目的**: 明确区分各类参数，纠正统一模型管理系统中的参数分类混淆问题

---

## 执行摘要

本文档对 video_to_srt_gpu 项目中所有模型和流水线的参数进行了系统分类，
给出前端可暴露参数的边界与归属规范。

### 核心结论

统一模型管理系统仅覆盖模型运行参数与少量模型级阈值/开关；业务逻辑阈值与
流水线开关归入独立配置域，不进入运行参数系统。

### 关键要点

1. 频谱分诊阈值属于业务逻辑阈值，统一归入诊断配置域
2. 参数总数: 150+ 个参数分散在各个模块中
3. 分类体系: 按照 7 大类进行系统分类（运行参数、输入参数、输出参数、内部参数、阈值参数、开关参数、其他参数）
4. 多层优先级: 统一管理系统实现了 per_model > runtime > env > hardware > default 的优先级链

### 执行规范

1. 模型运行参数进入统一管理；业务逻辑参数进入独立配置域
2. 为每个参数标注应用时机（初始化时 vs 推理时）
3. VAD 预设覆盖提供预设导出能力，减少特殊分支

---

## 参数分类规则

### 第 1 类：模型运行参数（Model Runtime Parameters）

**定义**: 模型运行时需要的参数，用于控制推理过程的性能和精度。

**特征**:
- 影响模型的执行方式（设备、精度、线程数等）
- 不改变模型的输入输出格式
- 通常在模型加载或推理前设置

**是否应纳入统一管理**: ✅ 是

**示例**: `device`, `compute_type`, `cpu_threads`, `vad_filter`, `condition_on_previous_text`

---

### 第 2 类：模型输入参数（Model Input Parameters）

**定义**: 提供给模型的数据和处理参数，控制模型的输入形态和预处理方式。

**特征**:
- 每次推理时传入的数据或配置
- 直接影响模型的输入内容
- 通常是动态变化的

**是否应纳入统一管理**: ❌ 否（这些是推理时的动态参数，不应作为配置管理）

**示例**: `audio`, `language`, `initial_prompt`, `start_time`, `end_time`

---

### 第 3 类：模型输出参数（Model Output Parameters）

**定义**: 模型生成并传递的结果参数。

**特征**:
- 模型推理的返回值
- 由模型生成，不可配置
- 用于后续流水线处理

**是否应纳入统一管理**: ❌ 否（这些是模型的输出结果，不是配置参数）

**示例**: `text`, `confidence`, `segments`, `word_timestamps`, `snr`, `c50`

---

### 第 4 类：模型内部参数（Model Internal Parameters）

**定义**: 模型内部状态、缓存等运行时数据。

**特征**:
- 流水线内部传递的状态数据
- 不对外暴露配置
- 用于跨阶段的数据传递

**是否应纳入统一管理**: ❌ 否（这些是内部实现细节）

**示例**: `AudioChunk.index`, `AudioChunk.needs_separation`, `SpectrumDiagnosis.triage_layer`

---

### 第 5 类：阈值参数（Threshold Parameters）

**定义**: 用于判断、触发或分类的阈值参数，属于业务逻辑参数而非模型参数。

**特征**:
- 用于条件判断和决策
- 影响业务逻辑流程
- 通常是固定的经验值或调优值

**是否应纳入统一管理**: ⚠️ 部分可以，但需明确区分
- ✅ 模型级阈值（如 Whisper 补刀阈值）可以纳入
- ❌ 业务逻辑阈值（如频谱分诊阈值）不应纳入

**示例**:
- 应纳入: `whisper_patch_trigger_confidence`, `sv_confidence_low`
- 不应纳入: `snr_high_threshold`, `c50_good_threshold`, `music_score_threshold`

---

### 第 6 类：开关参数（Switch Parameters）

**定义**: 用于开启关闭功能的布尔值或枚举参数。

**特征**:
- 控制功能的启用/禁用
- 通常是布尔值或枚举类型
- 影响流水线的执行路径

**是否应纳入统一管理**: ⚠️ 部分可以
- ✅ 模型级开关（如 `vad_filter`）可以纳入
- ❌ 流水线级开关（如 `enable_spectral_triage`）不应纳入

**示例**:
- 应纳入: `vad_filter`, `condition_on_previous_text`
- 不应纳入: `enable_spectral_triage`, `enable_fuse_breaker`, `use_yamnet`

---

### 第 7 类：其他参数（Miscellaneous Parameters）

**定义**: 无法明确归类的参数，通常是配置、策略或模式选择参数。

**特征**:
- 不属于上述任何类别
- 通常是高层次的策略选择
- 影响整体行为模式

**是否应纳入统一管理**: ⚠️ 视情况而定

**示例**: `processing_mode`, `transcription_profile`, `quality_preset`, `separation_mode`

---

## 统一模型管理系统范围界定

### 应纳入统一管理的参数

1. **模型运行参数**（第 1 类）
   - 设备选择: `device`, `device_preference`
   - 计算精度: `compute_type`, `quantize`
   - 线程配置: `cpu_threads`, `onnx_intra_threads`, `onnx_inter_threads`
   - 内存管理: `max_vram_mb`, `reserved_vram_mb`, `max_models`
   - 模型行为: `vad_filter`, `condition_on_previous_text`, `shifts`, `overlap`

2. **部分模型级阈值参数**（第 5 类的子集）
   - Whisper 补刀阈值: `whisper_patch_trigger_confidence`
   - SenseVoice 置信度阈值: `sv_confidence_low`, `sv_confidence_high`

3. **部分模型级开关参数**（第 6 类的子集）
   - Whisper 功能开关: `vad_filter`, `condition_on_previous_text`
   - Brouhaha 功能开关: `fallback_to_wada`

### 不应纳入统一管理的参数

1. **模型输入参数**（第 2 类）- 全部不应纳入
2. **模型输出参数**（第 3 类）- 全部不应纳入
3. **模型内部参数**（第 4 类）- 全部不应纳入
4. **业务逻辑阈值**（第 5 类的子集）
   - ❌ 频谱分诊阈值: `snr_high_threshold`, `snr_low_threshold`, `c50_good_threshold`, `c50_bad_threshold`
   - ❌ 频谱特征阈值: `harmonic_ratio_music`, `spectral_centroid_music_low`, `music_score_threshold`
   - ❌ 分句算法阈值: `pause_threshold`, `max_duration`, `short_sentence_threshold`
5. **流水线级开关**（第 6 类的子集）
   - ❌ 分诊开关: `enable_spectral_triage`, `use_snr_triage`, `use_yamnet`, `use_snr_strategy`
   - ❌ 熔断开关: `enable_fuse_breaker`
   - ❌ 分句开关: `use_dynamic_pause`, `merge_short_sentences`, `delay_split_to_punctuation`

---

## 业务逻辑参数独立归属清单

以下参数明确归属独立配置域，不进入统一模型运行参数系统：

### 频谱分诊阈值（诊断配置域）

| 参数名 | 归属域 | 管理方式 | 说明 |
|--------|--------|---------|------|
| `snr_high_threshold` | diagnosis.spectrum_thresholds | 诊断阈值配置 | 业务逻辑判定阈值 |
| `snr_low_threshold` | diagnosis.spectrum_thresholds | 诊断阈值配置 | 业务逻辑判定阈值 |
| `c50_good_threshold` | diagnosis.spectrum_thresholds | 诊断阈值配置 | 业务逻辑判定阈值 |
| `c50_bad_threshold` | diagnosis.spectrum_thresholds | 诊断阈值配置 | 业务逻辑判定阈值 |
| `spectral_contrast_low` | diagnosis.spectrum_thresholds | 诊断阈值配置 | 频谱特征判定阈值 |
| `spectral_contrast_critical` | diagnosis.spectrum_thresholds | 诊断阈值配置 | 频谱特征判定阈值 |
| `spectral_flatness_high` | diagnosis.spectrum_thresholds | 诊断阈值配置 | 频谱特征判定阈值 |
| `harmonic_ratio_music` | diagnosis.spectrum_thresholds | 诊断阈值配置 | 音乐检测阈值 |
| `spectral_centroid_music_low` | diagnosis.spectrum_thresholds | 诊断阈值配置 | 音乐检测阈值 |
| `spectral_centroid_music_high` | diagnosis.spectrum_thresholds | 诊断阈值配置 | 音乐检测阈值 |
| `energy_variance_music` | diagnosis.spectrum_thresholds | 诊断阈值配置 | 音乐检测阈值 |
| `onset_strength_music` | diagnosis.spectrum_thresholds | 诊断阈值配置 | 音乐检测阈值 |
| `zcr_noise_high` | diagnosis.spectrum_thresholds | 诊断阈值配置 | 噪音检测阈值 |
| `zcr_variance_noise` | diagnosis.spectrum_thresholds | 诊断阈值配置 | 噪音检测阈值 |
| `high_freq_ratio_noise` | diagnosis.spectrum_thresholds | 诊断阈值配置 | 噪音检测阈值 |
| `spectral_flatness_noise` | diagnosis.spectrum_thresholds | 诊断阈值配置 | 噪音检测阈值 |
| `music_score_threshold` | diagnosis.spectrum_thresholds | 诊断阈值配置 | 音乐得分判定阈值 |
| `noise_score_threshold` | diagnosis.spectrum_thresholds | 诊断阈值配置 | 噪音得分判定阈值 |
| `clean_score_threshold` | diagnosis.spectrum_thresholds | 诊断阈值配置 | 纯净度判定阈值 |
| `heavy_bgm_threshold` | diagnosis.spectrum_thresholds | 诊断阈值配置 | BGM 强度判定阈值 |
| `light_bgm_threshold` | diagnosis.spectrum_thresholds | 诊断阈值配置 | BGM 强度判定阈值 |

### 频谱分诊开关（流水线配置域）

| 参数名 | 归属域 | 管理方式 | 说明 |
|--------|--------|---------|------|
| `use_yamnet` | pipeline.spectrum_switches | 流水线开关配置 | 流水线级功能开关 |
| `use_snr_strategy` | pipeline.spectrum_switches | 流水线开关配置 | 流水线级功能开关 |

**依据**:

- 文档 `model-runtime-api-reference.md:93-94` 说明："频谱分诊阈值属于分诊判定参数，不通过模型运行参数系统管理"
- 更新日志 `llmdoc/index.md:332` 记录："频谱分诊阈值不再纳入模型运行参数系统"
- 如实现仍通过运行参数系统读取这些值，应以本文档为准对齐

---

## 详细参数矩阵

### 第 1 类：模型运行参数详细清单

#### 1.1 Whisper (Faster-Whisper) 运行参数

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|--------|---------------|---------|
| `device` | whisper | 运行参数 | 推理设备选择 | auto | auto/cuda/cpu | `whisper_service.py` |
| `compute_type` | whisper | 运行参数 | 计算精度类型 | auto | float16/int8/int8_float16/auto | `whisper_service.py:36` |
| `cpu_threads` | whisper | 运行参数 | CPU 线程数 | None（继承全局） | int > 0 | 全局配置 |
| `vad_filter` | whisper | 运行参数 + 开关参数 | VAD 前处理开关（补刀场景） | False | bool | `whisper_executor.py:57,62` |
| `condition_on_previous_text` | whisper | 运行参数 + 开关参数 | 条件文本生成开关 | False | bool | `whisper_executor.py:57,62` |
| `repetition_penalty` | whisper | 运行参数 + 输入参数 | 重复惩罚系数 | None（默认） | float >= 1.0 | `whisper_executor.py:76,120` |
| `no_repeat_ngram_size` | whisper | 运行参数 + 输入参数 | N-gram 重复抑制大小 | None（默认） | int >= 2 | `whisper_executor.py:76,121` |

**注**: `vad_filter` 和 `condition_on_previous_text` 同时属于运行参数和开关参数。

---

#### 1.2 SenseVoice ONNX 运行参数

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|--------|---------------|---------|
| `device` | sensevoice | 运行参数 | 推理设备 | auto | auto/cuda/cpu | `sensevoice_onnx_service.py:27` |
| `model_type` | sensevoice | 运行参数 | 模型规格 | small | small/medium | 配置文件 |
| `quantize` | sensevoice | 运行参数 | 量化类型 | None | None/int8 | 配置文件 |

**SenseVoice VAD 预设覆盖参数**（通过运行时组控制）:

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|--------|---------------|---------|
| `sensevoice_merge_max_gap` | vad (sensevoice 预设) | 运行参数 | VAD 合并最大间隔 | 0.5 | float 秒 | `runtime_param_resolver.py:116` |
| `sensevoice_merge_max_duration` | vad (sensevoice 预设) | 运行参数 | VAD 合并最大时长 | 15.0 | float 秒 | `runtime_param_resolver.py:117` |
| `sensevoice_smart_target_duration` | vad (sensevoice 预设) | 运行参数 | 智能累积目标时长 | 5.0 | float 秒 | `runtime_param_resolver.py:118-121` |

---

#### 1.3 Demucs 人声分离运行参数

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|--------|---------------|---------|
| `device` | demucs | 运行参数 | 推理设备 | cuda | cuda/cpu | `demucs_service.py:41` |
| `shifts` | demucs | 运行参数 | 增强次数（质量参数） | 1 | 1/2/5 | `demucs_service.py:42` |
| `overlap` | demucs | 运行参数 | 分段重叠率 | 0.5 | float 0-1 | `demucs_service.py:43` |
| `segment_length` | demucs | 运行参数 | 单段处理长度 | 10 | int 秒数 | `demucs_service.py:44` |

---

#### 1.4 VAD (Silero/Pyannote) 运行参数

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|--------|---------------|---------|
| `method` | vad | 运行参数 | VAD 模型选择 | silero | silero/pyannote | `vad_service.py:22-28` |
| `hf_token` | vad | 运行参数 | HuggingFace Token | None | str | `vad_service.py` |
| `onset` | vad | 运行参数 | 语音起始阈值 | 0.5 | float 0-1 | `vad_service.py` |
| `offset` | vad | 运行参数 | 语音结束阈值 | 0.363 | float 0-1 | `vad_service.py` |
| `chunk_size` | vad | 运行参数 | 处理块大小 | 512 | int 采样数 | `vad_service.py` |
| `min_speech_duration_ms` | vad | 运行参数 | 最小语音时长 | 100 | int ms | `vad_service.py` |
| `min_silence_duration_ms` | vad | 运行参数 | 最小静音时长 | 200 | int ms | `vad_service.py` |
| `speech_pad_ms` | vad | 运行参数 | 语音边界填充 | 50 | int ms | `vad_service.py` |
| `merge_max_gap` | vad | 运行参数 | 合并最大间隔 | 0.3 | float 秒 | `vad_service.py` |
| `merge_max_duration` | vad | 运行参数 | 合并最大时长 | 10.0 | float 秒 | `vad_service.py` |
| `merge_min_fragment` | vad | 运行参数 | 合并最小片段 | 0.1 | float 秒 | `vad_service.py` |
| `smart_target_duration` | vad | 运行参数 | 智能累积目标 | 5.0 | float 秒 | `vad_service.py` |
| `smart_max_duration` | vad | 运行参数 | 智能累积上限 | 15.0 | float 秒 | `vad_service.py` |
| `smart_min_gap_to_split` | vad | 运行参数 | 智能切分最小间隔 | 0.5 | float 秒 | `vad_service.py` |

---

#### 1.5 SmartProbe 智能探针运行参数

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|--------|---------------|---------|
| `snr_threshold` | smart_probe | 运行参数 | SNR 探测阈值 | 15.0 | float dB | `smart_probe_service.py:37` |
| `max_step_chunks` | smart_probe | 运行参数 | 最大扩散步长 | 30 | int chunks | `smart_probe_service.py:38` |

---

#### 1.6 全局管理参数（Global）

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|--------|---------------|---------|
| `device_preference` | global | 运行参数 | 全局设备偏好 | auto | auto/cuda/cpu | `model_runtime_config_service.py:88` |
| `allow_download` | global | 运行参数 | 允许自动下载 | True | bool | `model_runtime_config_service.py:89` |
| `max_vram_mb` | global | 运行参数 | 显存预算上限 | 根据硬件 | int MB | `model_runtime_config_service.py:90` |
| `reserved_vram_mb` | global | 运行参数 | 预留显存 | 500 | int MB | `model_runtime_config_service.py:91` |
| `max_models` | global | 运行参数 | 最大缓存模型数 | 3 | int | `model_runtime_config_service.py:92` |
| `cpu_threads` | global | 运行参数 | 全局 CPU 线程 | 自动 | int | `model_runtime_config_service.py:93` |
| `cpu_affinity_strategy` | global | 运行参数 | CPU 亲和策略 | auto | auto/p_cores/all | `model_runtime_config_service.py:94` |
| `onnx_intra_threads` | global | 运行参数 | ONNX 线程（内） | 自动 | int | `model_runtime_config_service.py:95` |
| `onnx_inter_threads` | global | 运行参数 | ONNX 线程（间） | 自动 | int | `model_runtime_config_service.py:96` |

---

#### 1.7 单模型管理参数（Per Model）

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|--------|---------------|---------|
| `device` | per_model | 运行参数 | 模型特定设备 | None（继承全局） | auto/cuda/cpu | `model_runtime_config_service.py:102` |
| `compute_type` | per_model | 运行参数 | 模型特定精度 | None（继承全局） | float16/int8/int8_float16 | `model_runtime_config_service.py:103` |
| `cpu_threads` | per_model | 运行参数 | 模型特定线程数 | None | int | `model_runtime_config_service.py:104` |
| `keep_resident` | per_model | 运行参数 | 是否常驻显存 | False | bool | `model_runtime_config_service.py:105` |
| `evict_priority` | per_model | 运行参数 | 驱逐优先级 | normal | low/normal/high | `model_runtime_config_service.py:106` |
| `max_concurrency` | per_model | 运行参数 | 最大并发数 | 1 | int >= 1 | `model_runtime_config_service.py:107` |

---

### 第 2 类：模型输入参数详细清单

#### 2.1 Whisper 输入参数

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|--------|---------------|---------|
| `audio` | whisper | 输入参数 | 音频数据（已切片 Chunk） | 必需 | np.ndarray float32 | `whisper_executor.py:68-77` |
| `start_time` | whisper | 输入参数 | Chunk 起始时间（日志用） | 0 | float 秒 | `whisper_executor.py:71` |
| `end_time` | whisper | 输入参数 | Chunk 结束时间（日志用） | 必需 | float 秒 | `whisper_executor.py:72` |
| `language` | whisper | 输入参数 | 语言代码 | auto | zh/en/ja/ko/auto | `whisper_executor.py:74` |
| `initial_prompt` | whisper | 输入参数 | 上下文引导提示 | None | str 自由文本 | `whisper_executor.py:75` |

#### 2.2 SenseVoice ONNX 输入参数

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|--------|---------------|---------|
| `audio` | sensevoice | 输入参数 | 音频数据（原始波形） | 必需 | np.ndarray float32 16kHz | `sensevoice_onnx_service.py` |
| `language` | sensevoice | 输入参数 | 预期语言 | auto | zh/en/auto | 推理执行器 |

#### 2.3 其他模型输入参数

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|--------|---------------|---------|
| `audio` | demucs | 输入参数 | 混合音频 | 必需 | np.ndarray float32 16kHz | `demucs_service.py` |
| `segment_buffer_sec` | demucs | 输入参数 | 分离缓冲区大小 | 2.0 | float 秒 | `demucs_service.py:47` |
| `audio` | vad | 输入参数 | 原始音频 | 必需 | np.ndarray 16kHz | `vad_service.py` |
| `audio` | brouhaha | 输入参数 | 音频数据 | 必需 | np.ndarray float32 16kHz | `brouhaha_service.py:126` |
| `sr` | brouhaha | 输入参数 | 采样率 | 16000 | int Hz | `brouhaha_service.py:126` |
| `waveform` | yamnet | 输入参数 | 原始波形 | 必需 | np.ndarray float32 16kHz | `yamnet_classifier.py:72` |
| `language` | splitter | 输入参数 | 目标语言 | auto | zh/en/ja/auto | `sentence_splitter.py:44` |

---

### 第 3 类：模型输出参数详细清单

#### 3.1 Whisper 输出参数

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|---------------|---------|
| `text` | whisper | 输出参数 | 识别文本 | str | `whisper_executor.py:133` |
| `confidence` | whisper | 输出参数 | 置信度估算 | float 0-1 | `whisper_executor.py:135` |
| `language` | whisper | 输出参数 | 检测语言 | str 代码 | `whisper_executor.py:136` |
| `segments` | whisper | 输出参数 | 段级信息 | List[Dict] | `whisper_service.py` |

#### 3.2 SenseVoice ONNX 输出参数

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|---------------|---------|
| `text` | sensevoice | 输出参数 | 识别文本 | str | `sensevoice_onnx_service.py:150` |
| `word_timestamps` | sensevoice | 输出参数 | 字级时间戳 | List[Dict] {word, start, end, confidence} | `sensevoice_onnx_service.py:69` |
| `confidence` | sensevoice | 输出参数 | 句级平均置信度 | float 0-1 | `sensevoice_onnx_service.py:153` |

#### 3.3 其他模型输出参数

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|---------------|---------|
| `vocals` | demucs | 输出参数 | 人声分离结果 | np.ndarray | `demucs_service.py` |
| `accompaniment` | demucs | 输出参数 | 背景分离结果 | np.ndarray | `demucs_service.py` |
| `segments` | vad | 输出参数 | 语音活动片段 | List[{start, end}] | `vad_service.py` |
| `activities` | vad | 输出参数 | 活动标志 | List[bool] | `vad_service.py` |
| `snr` | brouhaha | 输出参数 | 信噪比 | float dB -10~50 | `brouhaha_service.py:64` |
| `c50` | brouhaha | 输出参数 | 清晰度指数 | float dB -20~30 | `brouhaha_service.py:65` |
| `vad` | brouhaha | 输出参数 | 语音活动概率 | float 0-1 | `brouhaha_service.py:66` |
| `is_valid` | brouhaha | 输出参数 | 结果有效性 | bool | `brouhaha_service.py:67` |
| `is_music` | yamnet | 输出参数 | 是否需要分离 | bool | `yamnet_classifier.py:26` |
| `confidence` | yamnet | 输出参数 | 音乐置信度 | float 0-1 | `yamnet_classifier.py:27` |
| `speech_score` | yamnet | 输出参数 | 人声得分 | float 0-1 | `yamnet_classifier.py:28` |
| `music_score` | yamnet | 输出参数 | 音乐得分 | float 0-1 | `yamnet_classifier.py:29` |
| `top_classes` | yamnet | 输出参数 | Top-N 音频类别 | List[Tuple[str, float]] | `yamnet_classifier.py:31` |
| `decision` | smart_probe | 输出参数 | 分离决策 | 'SEPARATE_ALL'/'PASS_ALL' | `smart_probe_service.py:73` |
| `cache` | smart_probe | 输出参数 | 检测缓存 | Dict[index: {snr, c50}] | `smart_probe_service.py:74` |

---

### 第 4 类：模型内部参数详细清单

#### 4.1 AudioChunk 内部字段

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 代码位置 |
|--------|--------------|---------|---------|--------|---------|
| `index` | audio_chunk | 内部参数 | chunk 序号 | auto | `chunk_engine.py:23-77` |
| `audio` | audio_chunk | 内部参数 | 音频数据 | 必需 | `chunk_engine.py` |
| `sample_rate` | audio_chunk | 内部参数 | 采样率 | 16000 | `chunk_engine.py` |
| `start_time` | audio_chunk | 内部参数 | 起始时间 | 计算 | `chunk_engine.py` |
| `end_time` | audio_chunk | 内部参数 | 结束时间 | 计算 | `chunk_engine.py` |
| `needs_separation` | audio_chunk | 内部参数 | 分离标记 | False | `chunk_engine.py:20` |
| `spectrum_diagnosis` | audio_chunk | 内部参数 | 频谱诊断结果 | None | `chunk_engine.py:22` |
| `separation_level` | audio_chunk | 内部参数 | 分离级别 | NONE | `chunk_engine.py:24` |
| `original_audio` | audio_chunk | 内部参数 | 原始音频缓存 | None | `chunk_engine.py:25`（熔断回溯用） |
| `fuse_retry_count` | audio_chunk | 内部参数 | 熔断重试次数 | 0 | `chunk_engine.py:26` |
| `last_confidence` | audio_chunk | 内部参数 | 上次置信度 | 1.0 | `chunk_engine.py:27` |

#### 4.2 SpectrumDiagnosis 内部数据

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|---------------|---------|
| `snr` | spectrum_diagnosis | 内部参数 | 频谱诊断 SNR | float | `circuit_breaker_models.py:48-66` |
| `c50` | spectrum_diagnosis | 内部参数 | 频谱诊断 C50 | float | `circuit_breaker_models.py` |
| `needs_separation` | spectrum_diagnosis | 内部参数 | 分离需求 | bool | `circuit_breaker_models.py` |
| `triage_layer` | spectrum_diagnosis | 内部参数 | 决策层级 | int 1-3 | `circuit_breaker_models.py` |

---

### 第 5 类：阈值参数详细清单

#### 5.1 Whisper 补刀触发阈值

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|--------|---------------|---------|
| `whisper_patch_trigger_confidence` | whisper | 阈值参数 | 补刀触发置信度 | 0.6 | float 0-1 | `thresholds.py:43` |
| `short_segment_duration` | whisper | 阈值参数 | 短片段时长 | 1.0 | float 秒 | `thresholds.py:46` |
| `short_segment_chars` | whisper | 阈值参数 | 短片段字符数 | 3 | int | `thresholds.py:47` |
| `single_char_force_patch` | whisper | 阈值参数 + 开关参数 | 单字符强制补刀开关 | True | bool | `thresholds.py:48` |
| `single_char_confidence_threshold` | whisper | 阈值参数 | 单字符补刀置信度 | 0.9 | float 0-1 | `thresholds.py` |
| `word_warning_confidence` | whisper | 阈值参数 | 字级警告阈值 | 0.5 | float 0-1 | `thresholds.py:56` |
| `word_critical_confidence` | whisper | 阈值参数 | 字级严重警告 | 0.3 | float 0-1 | `thresholds.py:57` |

#### 5.2 SenseVoice 置信度阈值

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|--------|---------------|---------|
| `sv_confidence_high` | sensevoice | 阈值参数 | 高置信度 | 0.85 | float 0-1 | `thresholds.py:27` |
| `sv_confidence_medium` | sensevoice | 阈值参数 | 中等置信度 | 0.6 | float 0-1 | `thresholds.py:28` |
| `sv_confidence_low` | sensevoice | 阈值参数 | 低置信度 | 0.4 | float 0-1 | `thresholds.py:29` |

#### 5.3 Whisper 仲裁阈值

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|--------|---------------|---------|
| `whisper_logprob_good` | whisper | 阈值参数 | 好的 logprob | -0.5 | float | `thresholds.py:32` |
| `arbitration_whisper_threshold` | whisper | 阈值参数 | 仲裁判决阈值 | 0.5 | float 0-1 | `threshold_config.md` |
| `garbage_sentence_confidence` | sensevoice | 阈值参数 | 垃圾句子置信度 | 0.4 | float 0-1 | `threshold_config.md` |

#### 5.4 句级置信度阈值

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|--------|---------------|---------|
| `sentence_warning_confidence` | transcription | 阈值参数 | 句级警告阈值 | 0.6 | float 0-1 | `thresholds.py:60` |
| `sentence_warning_perplexity` | transcription | 阈值参数 | 困惑度警告 | 50.0 | float | `thresholds.py:61` |

#### 5.5 频谱分诊阈值（业务逻辑阈值，独立管理）

**重要说明**: 以下参数归属诊断配置域，不通过统一模型运行参数系统管理。

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|--------|---------------|---------|
| `snr_high_threshold` | spectrum | 阈值参数（业务逻辑） | SNR 高阈值（放行） | 40.0 | float dB | `spectrum_thresholds.py:58` |
| `snr_low_threshold` | spectrum | 阈值参数（业务逻辑） | SNR 低阈值（分离） | 25.0 | float dB | `spectrum_thresholds.py:59` |
| `c50_good_threshold` | spectrum | 阈值参数（业务逻辑） | C50 良好阈值 | 13.70 | float dB | `spectrum_thresholds.py:63` |
| `c50_bad_threshold` | spectrum | 阈值参数（业务逻辑） | C50 差阈值 | -12.33 | float dB | `spectrum_thresholds.py:64` |
| `spectral_contrast_low` | spectrum | 阈值参数（业务逻辑） | 频谱对比度低 | 19.84 | float dB | `spectrum_thresholds.py:68` |
| `spectral_contrast_critical` | spectrum | 阈值参数（业务逻辑） | 频谱对比度临界 | 13.59 | float dB | `spectrum_thresholds.py:69` |
| `spectral_flatness_high` | spectrum | 阈值参数（业务逻辑） | 频谱平坦度高 | 0.29 | float | `spectrum_thresholds.py:70` |
| `harmonic_ratio_music` | spectrum | 阈值参数（业务逻辑） | 谐波比（音乐） | 0.6 | float | `spectrum_thresholds.py:22` |
| `spectral_centroid_music_low` | spectrum | 阈值参数（业务逻辑） | 质心低边界 | 1500 | int Hz | `spectrum_thresholds.py:25` |
| `spectral_centroid_music_high` | spectrum | 阈值参数（业务逻辑） | 质心高边界 | 4000 | int Hz | `spectrum_thresholds.py:26` |
| `energy_variance_music` | spectrum | 阈值参数（业务逻辑） | 能量方差（音乐） | 0.15 | float | `spectrum_thresholds.py:27` |
| `onset_strength_music` | spectrum | 阈值参数（业务逻辑） | 起始强度（音乐） | 0.5 | float | `spectrum_thresholds.py:28` |
| `zcr_noise_high` | spectrum | 阈值参数（业务逻辑） | 过零率高（噪音） | 0.15 | float | `spectrum_thresholds.py:32` |
| `zcr_variance_noise` | spectrum | 阈值参数（业务逻辑） | 过零率方差（噪音） | 0.05 | float | `spectrum_thresholds.py:33` |
| `high_freq_ratio_noise` | spectrum | 阈值参数（业务逻辑） | 高频比（噪音） | 0.4 | float | `spectrum_thresholds.py:34` |
| `spectral_flatness_noise` | spectrum | 阈值参数（业务逻辑） | 频谱平坦度（噪音） | 0.5 | float | `spectrum_thresholds.py:35` |
| `music_score_threshold` | spectrum | 阈值参数（业务逻辑） | 音乐得分阈值 | 0.35 | float 0-1 | `spectrum_thresholds.py:46` |
| `noise_score_threshold` | spectrum | 阈值参数（业务逻辑） | 噪音得分阈值 | 0.45 | float 0-1 | `spectrum_thresholds.py:47` |
| `clean_score_threshold` | spectrum | 阈值参数（业务逻辑） | 纯净度阈值 | 0.7 | float 0-1 | `spectrum_thresholds.py:48` |
| `heavy_bgm_threshold` | spectrum | 阈值参数（业务逻辑） | 强 BGM 阈值 | 0.15 | float | `spectrum_thresholds.py:51` |
| `light_bgm_threshold` | spectrum | 阈值参数（业务逻辑） | 轻 BGM 阈值 | 0.02 | float | `spectrum_thresholds.py:52` |

#### 5.6 分句算法阈值

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|--------|---------------|---------|
| `pause_multiplier` | splitter | 阈值参数 | 停顿乘数 | 2.5 | float | `sentence_splitter.py:121` |
| `min_pause_threshold` | splitter | 阈值参数 | 最小停顿阈值 | 0.5 | float 秒 | `sentence_splitter.py:122` |
| `pause_threshold` | splitter | 阈值参数 | 静态停顿阈值 | 0.7 | float 秒 | `sentence_splitter.py:124` |
| `long_pause_threshold` | splitter | 阈值参数 | 长停顿阈值 | 1.5 | float 秒 | `sentence_splitter.py:125` |
| `max_duration` | splitter | 阈值参数 | 最大句时长 | 5.0 | float 秒 | `sentence_splitter.py:126` |
| `hard_limit_duration` | splitter | 阈值参数 | 硬上限时长 | 10.0 | float 秒 | `sentence_splitter.py:131` |
| `short_sentence_threshold` | splitter | 阈值参数 | 短句字符数 | 8 | int | `sentence_splitter.py:135` |
| `min_duration_threshold` | splitter | 阈值参数 | 短句时长 | 0.5 | float 秒 | `sentence_splitter.py:136` |
| `delay_split_max_wait` | splitter | 阈值参数 | 延迟等待时长 | 2.0 | float 秒 | `sentence_splitter.py:141` |
| `max_boundary_gap` | splitter | 阈值参数 | 最大边界间隙 | 0.3 | float 秒 | `sentence_splitter.py:147` |

---

### 第 6 类：开关参数详细清单

#### 6.1 分离相关开关

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 代码位置 |
|--------|--------------|---------|---------|--------|---------|
| `enable_spectral_triage` | spectrum | 开关参数 | 启用频谱分诊 | True | `job_models.py:42` |
| `use_snr_triage` | spectrum | 开关参数 | 启用 SNR 策略 | True | `job_models.py:43` |
| `enable_fuse_breaker` | separation | 开关参数 | 启用熔断机制 | True | `job_models.py:45` |
| `separation_mode` | separation | 开关参数 | 分离模式 | on_demand | global/on_demand | `job_models.py:40` |

#### 6.2 频谱分诊功能开关（流水线级开关，独立管理）

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 代码位置 |
|--------|--------------|---------|---------|--------|---------|
| `use_yamnet` | spectrum | 开关参数 | 启用 YAMNet | True | `audio_spectrum_classifier.py:82` |
| `use_snr_strategy` | spectrum | 开关参数 | 启用 SNR 策略 | True | `audio_spectrum_classifier.py:85` |

#### 6.3 Whisper 功能开关

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 代码位置 |
|--------|--------------|---------|---------|--------|---------|
| `single_char_force_patch` | whisper | 开关参数 | 单字符强制补刀 | True | `thresholds.py:48` |
| `vad_filter` | whisper | 开关参数 + 运行参数 | VAD 前处理 | False | `whisper_executor.py:57` |
| `condition_on_previous_text` | whisper | 开关参数 + 运行参数 | 条件文本生成 | False | `whisper_executor.py:57` |

#### 6.4 分句功能开关

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 代码位置 |
|--------|--------------|---------|---------|--------|---------|
| `use_dynamic_pause` | splitter | 开关参数 | 动态停顿阈值 | True | `sentence_splitter.py:119` |
| `merge_short_sentences` | splitter | 开关参数 | 短句合并 | True | `sentence_splitter.py:134` |
| `delay_split_to_punctuation` | splitter | 开关参数 | 延迟切分 | True | `sentence_splitter.py:139` |
| `enable_hard_limit` | splitter | 开关参数 | 硬上限异常保护 | False（慢流）/True（快流） | `sentence_splitter.py:130` |
| `trim_leading_silence` | splitter | 开关参数 | 修剪句首静音 | True | `sentence_splitter.py:145` |
| `trim_trailing_silence` | splitter | 开关参数 | 修剪句尾静音 | True | `sentence_splitter.py:146` |

#### 6.5 Brouhaha 功能开关

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 代码位置 |
|--------|--------------|---------|---------|--------|---------|
| `fallback_to_wada` | brouhaha | 开关参数 | 回退到 WADA-SNR | True | `brouhaha_service.py:105` |
| `log_decisions` | brouhaha | 开关参数 | 详细日志 | False | `brouhaha_service.py:108` |
| `collect_statistics` | brouhaha | 开关参数 | 统计收集 | True | `brouhaha_service.py:109` |

---

### 第 7 类：其他参数详细清单

#### 7.1 转录模式参数

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|--------|---------------|---------|
| `processing_mode` | transcription | 其他参数 | 处理模式 | MEMORY | MEMORY/DISK | `transcription_service.py:23-31` |
| `transcription_profile` | transcription | 其他参数 | 转录预设 | balanced | sensevoice/whisper | 架构文档 |

#### 7.2 Demucs 策略参数

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|--------|---------------|---------|
| `quality_preset` | demucs | 其他参数 | 质量预设 | balanced | fast/balanced/quality | `demucs_service.py:164` |
| `mode` | demucs | 其他参数 | 分离模式 | auto | auto/always/never | `demucs_service.py` |
| `bgm_light_threshold` | demucs | 其他参数 | 轻 BGM 阈值 | 0.02 | float | `demucs_service.py:51` |
| `bgm_heavy_threshold` | demucs | 其他参数 | 强 BGM 阈值 | 0.15 | float | `demucs_service.py:52` |

#### 7.3 置信度映射参数

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|--------|---------------|---------|
| `display_confidence` | confidence | 其他参数 | 显示置信度 | 计算 | float 0.5-0.98 | `confidence_mapper.py` |
| `confidence_source` | confidence | 其他参数 | 置信度来源 | sensevoice/whisper | str | `confidence_mapper.py` |

#### 7.4 预处理配置参数

| 参数名 | 所属模型/阶段 | 参数分类 | 参数用途 | 默认值 | 期望格式和范围 | 代码位置 |
|--------|--------------|---------|---------|--------|---------------|---------|
| `fuse_max_retry` | preprocessing | 其他参数 | 最大重试次数 | 2 | int | `job_models.py:44` |
| `fuse_confidence_threshold` | preprocessing | 其他参数 | 置信度阈值 | 0.5 | float 0-1 | `job_models.py:46` |
| `fuse_auto_upgrade` | preprocessing | 其他参数 | 自动升级开关 | True | bool | `job_models.py:47` |

---

## 参数优先级与来源策略

### 运行参数解析优先级（高 → 低）

```
1. 单模型运行参数覆盖 (per_model.{id}.runtime)
2. 全局运行参数覆盖 (runtime.{group})
3. 环境变量 (MODEL_RUNTIME_*)
4. 硬件推荐 (hardware_service + cpu_optimizer)
5. 服务内置默认值
```

代码位置：`model_runtime_config_service.py:21`

### 参数来源标记（Source Tracking）

每个参数都标注来源：
- `model_override` - 单模型覆盖
- `runtime_override` - 全局运行时覆盖
- `env` - 环境变量
- `hardware` - 硬件推荐
- `default` - 内置默认值

代码位置：`model_runtime_config_service.py:29-49`，API 返回值含 `sources` 字段

---

## 统一聚合层最小设计草案（前端参数暴露）

### 目标

- 统一前端读写入口，减少跨域配置耦合
- 保持运行参数系统的优先级与来源追踪机制
- 允许业务逻辑阈值与流水线开关独立管理

### 边界与原则

- 聚合层不持久化配置，仅做聚合读取与写入路由
- 模型输入/输出/内部参数不提供写入接口
- 每个参数必须带上 `domain`、`apply_time`、`read_only`、`source` 元数据
- 写入校验由各域适配器负责，聚合层只做基础格式校验

### 组件最小集合

- ParameterRegistry: 参数元数据注册表，维护类别、范围、默认值、读写权限
- DomainAdapter: runtime、thresholds、pipeline、presets 的读写适配
- ParameterAggregatorService: 聚合查询、批量校验、路由写入

### API 分类（最小集合）

- **统一聚合**: `GET /api/params/overview`（聚合快照）, `GET /api/params/catalog`（元数据清单）
- **模型运行参数**: `GET/PATCH /api/params/runtime`, `GET/PATCH /api/params/runtime/per-model/{model_id}`
- **业务阈值**: `GET/PATCH /api/params/thresholds/spectrum`, `GET/PATCH /api/params/thresholds/splitter`, `GET/PATCH /api/params/thresholds/whisper`
- **流水线开关**: `GET/PATCH /api/params/pipeline`
- **预设与策略**: `GET /api/params/presets`, `POST /api/params/presets/apply`
- **只读状态**: `GET /api/params/status`（运行时来源与硬件建议）

---

## 关键对齐要点

### 要点 1：频谱分诊阈值归属

**规范**: 频谱分诊阈值属于业务逻辑阈值，归入 `diagnosis.spectrum_thresholds` 域管理，
不进入统一运行参数系统。

**依据**:
- 文档 `model-runtime-api-reference.md:93-94` 说明"频谱分诊阈值属于分诊判定参数，不通过模型运行参数系统管理"
- 更新日志 `llmdoc/index.md:332` 记录"频谱分诊阈值不再纳入模型运行参数系统"

**对齐提示**: 如实现仍从运行参数域读取阈值，应按本文档对齐。

### 要点 2：VAD 预设覆盖

**现状**: SenseVoice 模式下存在 `sensevoice_` 前缀参数覆盖，增加了参数继承复杂度。

**建议**:
1. 重新评估参数分组结构，是否需要为不同预设建立独立分组
2. 提供"预设切换"工具，一键应用预设的完整参数集
3. 文档中明确 SenseVoice 预设的参数继承链

### 要点 3：参数应用时机

**规范**: 每个参数必须标注应用时机（初始化时/推理时），并保证运行时读取策略一致。

**建议**:
1. 推理时参数支持热更新，初始化参数保持稳定并记录重载边界
2. 在矩阵中新增"应用时机"字段，避免前端误写导致无效更新

---

## 总结

### 参数统计

- **第 1 类（模型运行参数）**: 70+ 个参数
- **第 2 类（模型输入参数）**: 15+ 个参数
- **第 3 类（模型输出参数）**: 20+ 个参数
- **第 4 类（模型内部参数）**: 15+ 个参数
- **第 5 类（阈值参数）**: 50+ 个参数（含 21 个频谱分诊阈值，独立管理）
- **第 6 类（开关参数）**: 20+ 个参数
- **第 7 类（其他参数）**: 10+ 个参数

**总计**: 150+ 个参数

### 核心结论

1. **边界明确**: 统一模型管理系统仅覆盖模型运行参数与少量模型级阈值/开关
2. **分域管理**: 业务逻辑阈值与流水线开关独立配置域管理
3. **可聚合暴露**: 统一聚合层可在不破坏分域的前提下对前端暴露关键参数
4. **来源可追踪**: 运行参数系统具备优先级与来源标记，可直接复用

### 实施建议

1. 以本文档为准对齐实现，确保诊断阈值与流水线开关不进入运行参数系统
2. 更新 API 文档，补充聚合层接口分类与参数元数据字段
3. 在矩阵中补充"应用时机"列，避免前端无效更新
4. 简化 VAD 预设覆盖链路，提供预设导出能力

---

**文档完成日期**: 2026-01-18
**版本**: V3.2.0+dev.20260118.02

