# V3.2.0+dev.20260118.04 运行参数完整矩阵（统一模型管理）

> 目标：给出**运行参数完整矩阵 + 来源/优先级**，并明确**允许范围与默认值**。  
> 仅包含统一模型管理相关内容，不混入其他主题。

---

## 1. 参数来源与优先级

### 1.1 模型管理参数（global / per_model）
优先级（高 → 低）：
1. `per_model` 覆盖
2. `global` 覆盖
3. `.env`（仅对部分字段生效）
4. 硬件推荐（hardware）
5. 默认值（models.yaml 或代码默认）

### 1.2 运行参数（runtime）
优先级（高 → 低）：
1. `per_model.runtime` 覆盖（单模型）
2. `runtime.{group}` 全局覆盖
3. `.env`（仅对部分字段生效）
4. 代码默认值

`.env` 影响的运行参数示例：
- `SENSEVOICE_DEVICE`、`SENSEVOICE_MODEL_TYPE`
- `USE_HF_MIRROR`（影响 Whisper 下载源）

---

## 2. 全局模型管理参数（global）

- `device_preference`
  - 范围：`auto | cuda | cpu`
  - 默认：`auto`
- `allow_download`
  - 范围：`true | false`
  - 默认：`false`
- `max_vram_mb`
  - 范围：`>= 0`
  - 默认：`8000`（若有 GPU 则取显存 80%）
- `reserved_vram_mb`
  - 范围：`>= 0`
  - 默认：`500`
- `max_models`
  - 范围：`>= 1`
  - 默认：`3`
- `cpu_threads`
  - 范围：`>= 1`
  - 默认：`4`（或硬件推荐）
- `cpu_affinity_strategy`
  - 范围：`auto | half | custom | null`
  - 默认：`null`（不覆盖，保留现有策略）
- `onnx_intra_threads`
  - 范围：`>= 1`
  - 默认：`1`
- `onnx_inter_threads`
  - 范围：`>= 1`
  - 默认：`1`

---

## 3. 单模型管理覆盖（per_model）

- `device`
  - 范围：`auto | cuda | cpu`
  - 默认：来自 `models.yaml: default_device`
- `compute_type`
  - 范围：`auto | int8 | int8_float16 | float16 | float32 | fp32`
  - 默认：来自 `models.yaml: compute_type`（Whisper 会按显存自动调整）
- `cpu_threads`
  - 范围：`>= 1`
  - 默认：来自 `models.yaml: resources.cpu_threads` 或硬件推荐
- `keep_resident`
  - 范围：`true | false`
  - 默认：`false`
- `evict_priority`
  - 范围：任意整数
  - 默认：`0`
- `max_concurrency`
  - 范围：`>= 1`
  - 默认：`1`

---

## 4. 运行参数矩阵（runtime）

### 4.1 Whisper（faster-whisper）
- `language`
  - 范围：`auto` 或 `2-10` 位语言码（小写字母/数字/短横线）
  - 默认：`auto`
- `initial_prompt`
  - 范围：任意字符串
  - 默认：`null`
- `word_timestamps`
  - 范围：`true | false`
  - 默认：`false`
- `beam_size`
  - 范围：`>= 1`
  - 默认：`5`
- `vad_filter`
  - 范围：`true | false`
  - 默认：`true`
- `vad_parameters`
  - 范围：任意字典（值必须为非负数）
  - 默认：`null`
- `temperature`
  - 范围：`0.0 ~ 1.0`
  - 默认：`0.0`
- `condition_on_previous_text`
  - 范围：`true | false`
  - 默认：`true`
- `suppress_tokens`
  - 范围：`int[]`（元素 `>= 0`）
  - 默认：`null`（自动从配置提取）
- `repetition_penalty`
  - 范围：`>= 0.0`
  - 默认：`1.0`
- `no_repeat_ngram_size`
  - 范围：`>= 0`
  - 默认：`0`

### 4.2 SenseVoice（ONNX）
- `language`
  - 范围：`auto | zh | en | yue | ja | ko | nospeech`
  - 默认：`auto`
- `use_itn`
  - 范围：`true | false`
  - 默认：`true`
- `ban_emo_unk`
  - 范围：`true | false`
  - 默认：`false`
- `model_type`
  - 范围：`quantized | fp32`
  - 默认：`quantized`（受 `SENSEVOICE_MODEL_TYPE` 影响）
- `device`
  - 范围：`auto | cuda | cpu`
  - 默认：`cpu`（受 `SENSEVOICE_DEVICE` 影响）
- `batch_size`
  - 范围：`>= 1`
  - 默认：`1`
- `quantize`
  - 范围：`true | false`
  - 默认：`true`

### 4.3 Demucs
- `model_name`
  - 范围：`htdemucs | htdemucs_ft | mdx_extra | mdx_extra_q`
  - 默认：`htdemucs`
- `device`
  - 范围：`auto | cuda | cpu`
  - 默认：`cuda`（不可用时回退 CPU）
- `shifts`
  - 范围：`1 ~ 5`
  - 默认：`1`
- `overlap`
  - 范围：`0.0 ~ 1.0`
  - 默认：`0.5`
- `segment_length`
  - 范围：`>= 1`
  - 默认：`10`
- `segment_buffer_sec`
  - 范围：`>= 0.0`
  - 默认：`2.0`
- `bgm_sample_duration`
  - 范围：`>= 1.0`
  - 默认：`10.0`
- `bgm_light_threshold`
  - 范围：`0.0 ~ 1.0`
  - 默认：`0.02`
- `bgm_heavy_threshold`
  - 范围：`0.0 ~ 1.0`
  - 默认：`0.15`

### 4.4 VAD
- `method`
  - 范围：`silero | pyannote`
  - 默认：`silero`
- `hf_token`
  - 范围：任意字符串
  - 默认：`null`
- `onset`
  - 范围：`0.0 ~ 1.0`
  - 默认：`0.4`
- `offset`
  - 范围：`0.0 ~ 1.0`
  - 默认：`0.4`
- `chunk_size`
  - 范围：`>= 1`
  - 默认：`30`
- `min_speech_duration_ms`
  - 范围：`>= 0`
  - 默认：`250`
- `min_silence_duration_ms`
  - 范围：`>= 0`
  - 默认：`400`
- `speech_pad_ms`
  - 范围：`>= 0`
  - 默认：`300`
- `merge_max_gap`
  - 范围：`>= 0.0`
  - 默认：`1.0`
- `merge_max_duration`
  - 范围：`>= 0.0`
  - 默认：`12.0`
- `merge_min_fragment`
  - 范围：`>= 0.0`
  - 默认：`1.0`
- `smart_target_duration`
  - 范围：`>= 0.0`
  - 默认：`12.0`
- `smart_max_duration`
  - 范围：`>= 0.0`
  - 默认：`30.0`
- `smart_min_gap_to_split`
  - 范围：`>= 0.0`
  - 默认：`0.3`
- SenseVoice 预设覆盖（仅在 SenseVoice 模式使用）
  - `sensevoice_merge_max_gap`: `>= 0.0`，默认 `0.3`
  - `sensevoice_merge_max_duration`: `>= 0.0`，默认 `8.0`
  - `sensevoice_smart_target_duration`: `>= 0.0`，默认 `8.0`

### 4.5 频谱分诊（规则阈值 + SNR/C50）
- `use_yamnet`
  - 范围：`true | false`
  - 默认：`true`
- `use_snr_strategy`
  - 范围：`true | false`
  - 默认：`true`
- 规则阈值（用于传统频谱规则）
  - `harmonic_ratio_music`: `0.0 ~ 1.0`，默认 `0.6`
  - `spectral_centroid_music_low`: `>= 0`，默认 `1500`
  - `spectral_centroid_music_high`: `>= 0`，默认 `4000`
  - `energy_variance_music`: `>= 0`，默认 `0.25`
  - `onset_strength_music`: `>= 0`，默认 `0.3`
  - `zcr_noise_high`: `0.0 ~ 1.0`，默认 `0.15`
  - `zcr_variance_noise`: `>= 0`，默认 `0.02`
  - `high_freq_ratio_noise`: `0.0 ~ 1.0`，默认 `0.4`
  - `spectral_flatness_noise`: `0.0 ~ 1.0`，默认 `0.5`
  - `music_score_threshold`: `0.0 ~ 1.0`，默认 `0.35`
  - `noise_score_threshold`: `0.0 ~ 1.0`，默认 `0.45`
  - `clean_score_threshold`: `0.0 ~ 1.0`，默认 `0.7`
  - `heavy_bgm_threshold`: `0.0 ~ 1.0`，默认 `0.6`
  - `light_bgm_threshold`: `0.0 ~ 1.0`，默认 `0.35`
- SNR/C50 阈值（Brouhaha）
  - `snr_high_threshold`: `>= 0`，默认 `40.0`
  - `snr_low_threshold`: `>= 0`，默认 `25.0`
  - `c50_good_threshold`: 任意浮点，默认 `13.70`
  - `c50_bad_threshold`: 任意浮点，默认 `-12.33`
  - `spectral_contrast_low`: `>= 0`，默认 `19.84`
  - `spectral_contrast_critical`: `>= 0`，默认 `13.59`
  - `spectral_flatness_high`: `0.0 ~ 1.0`，默认 `0.29`

### 4.6 SmartProbe（Brouhaha 探针）
- `snr_threshold`
  - 范围：`>= 0`
  - 默认：`15.0`

### 4.7 YAMNet 判决阈值
- `acappella_threshold`: `0.0 ~ 1.0`，默认 `0.3`
- `music_max_threshold`: `0.0 ~ 1.0`，默认 `0.15`
- `music_avg_threshold`: `0.0 ~ 1.0`，默认 `0.10`
- `speech_max_threshold`: `0.0 ~ 1.0`，默认 `0.8`
- `speech_max_music_threshold`: `0.0 ~ 1.0`，默认 `0.1`
- `speech_dominant_delta`: `0.0 ~ 1.0`，默认 `0.3`
- `speech_dominant_music_max`: `0.0 ~ 1.0`，默认 `0.15`
- `probe_window_count`: `1 ~ 5`，默认 `3`
- `probe_window_duration_sec`: `0.5 ~ 2.0`，默认 `0.975`

### 4.8 标点模型（Punctuation）
- `enable_punctuation`: `true | false`，默认 `true`
- `default_language`: 任意语言码，默认 `zh`
- `fallback_priority`: `fast | slow`，默认 `fast`
- `cache_models`: `true | false`，默认 `true`
- `max_cached_models`: `>= 1`，默认 `3`
- `device`: `cpu | cuda`，默认 `cpu`
- `num_threads`: `>= 1`，默认 `4`
- `use_int8`: `true | false`，默认 `true`
- `batch_size`: `>= 1`，默认 `1`
- `max_sequence_length`: `>= 1`，默认 `512`
- `enable_arbitration`: `true | false`，默认 `true`
- `confidence_threshold`: `0.0 ~ 1.0`，默认 `0.6`
- `vad_tolerance`: `>= 0.0`，默认 `0.3`
- `min_sentence_gap`: `>= 0.0`，默认 `1.5`
- `hallucination_check`: `true | false`，默认 `true`
- `require_vad_pause`: `true | false`，默认 `true`
- `cross_validation`: `true | false`，默认 `true`
- `whisper_confidence_min`: `0.0 ~ 1.0`，默认 `0.7`
- `alignment_method`: `anchor | dtw | levenshtein`，默认 `anchor`
- `anchor_confidence_threshold`: `0.0 ~ 1.0`，默认 `0.9`
- `max_local_alignment_length`: `>= 1`，默认 `50`
- `enable_semantic_buffer`: `true | false`，默认 `true`
- `max_buffer_duration`: `>= 0.0`，默认 `15.0`
- `hard_limit_duration`: `>= 0.0`，默认 `20.0`
- `min_chunk_duration`: `>= 0.0`，默认 `1.0`
- `prefer_punctuation_split`: `true | false`，默认 `true`
- `min_subtitle_chars_zh_ja`: `>= 1`，默认 `4`
- `min_subtitle_words_en`: `>= 1`，默认 `4`
- `min_subtitle_duration_sec`: `>= 0.0`，默认 `0.8`
- `fast_delay_budget_sec`: `>= 0.0`，默认 `2.0`
- `force_split_on_sentence_end_punct`: `true | false`，默认 `true`

### 4.9 流水线默认参数（Pipeline）
- `batch_size`
  - 范围：`>= 1`
  - 默认：`16`
- `word_timestamps`
  - 范围：`true | false`
  - 默认：`false`

### 4.10 LangID
- 暂无额外运行参数（仅使用模型管理参数）

---

## 5. API 对应（与实现一致）

- `GET /api/models/runtime`
  - 返回 `global`、`runtime`（全局运行参数）、`models`（单模型运行参数）
- `GET /api/models/runtime/{model_id}`
  - 返回指定模型的管理参数与运行参数（含来源与默认值）
- `PUT /api/models/runtime`
  - 更新 `global` 与 `runtime`（部分更新）
- `PUT /api/models/runtime/{model_id}`
  - 更新单模型管理参数与 `runtime` 覆盖（部分更新）
- `POST /api/models/runtime/apply`
  - 应用管理参数（预算/下载开关）
- `GET /api/models/params`
  - 返回所有字段、范围、默认值（用于前端表单生成）

---

## 6. 配置文件结构（model_runtime_config.json）

```json
{
  "version": "1.1",
  "global": { "...": "..." },
  "runtime": {
    "whisper": { "...": "..." },
    "sensevoice": { "...": "..." },
    "demucs": { "...": "..." },
    "vad": { "...": "..." },
    "smart_probe": { "...": "..." },
    "yamnet": { "...": "..." },
    "punctuation": { "...": "..." },
    "pipeline": { "...": "..." }
  },
  "per_model": {
    "whisper-medium": {
      "device": "cuda",
      "runtime": { "...": "..." }
    }
  }
}
```
