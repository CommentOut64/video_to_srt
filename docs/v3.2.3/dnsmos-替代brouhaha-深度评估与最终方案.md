# DNSMOS 替代 Brouhaha 深度评估与最终方案（v3.2.3）

> 结论先行：当前音频预检真实行为已经是“YAMNet/规则兜底主导”，`brouhaha` 仅保留代码壳；DNSMOS 可作为主检测器接管，但必须按当前代码现状重构分支、缓存键、运行参数和 SmartProbe 才能稳定落地。

---

## 1. 评估范围与证据

本评估基于以下代码与文档的交叉核验（已实际读取）：

- 音频预检主链：
  - `backend/app/pipelines/preprocessing_pipeline.py`
  - `backend/app/pipelines/stages/spectral_triage_stage.py`
  - `backend/app/services/audio_spectrum_classifier.py`
  - `backend/app/services/smart_probe_service.py`
  - `backend/app/services/yamnet_classifier.py`
  - `backend/app/services/preprocess_cache_service.py`
- 数据结构与参数：
  - `backend/app/models/circuit_breaker_models.py`
  - `backend/app/core/spectrum_thresholds.py`
  - `backend/app/models/job_models.py`
  - `backend/app/services/model_runtime_config_service.py`
  - `backend/app/api/routes/model_runtime_routes.py`
  - `model_runtime_config.json`
  - `backend/app/config/models.yaml`
  - `pyproject.toml`
- Brouhaha 现状：
  - `backend/app/services/brouhaha_service.py`
  - `backend/tests/test_brouhaha_service.py`
  - `backend/models/pretrained/brouhaha/pytorch_model.bin`（当前为 LFS 指针）
- 待评估方案：
  - `docs/v3.2.3/dnsmos-replace-brouhaha-plan.md`
- DNSMOS 官方脚本（本地副本）：
  - `tmp/DNS-Challenge-master/DNSMOS/dnsmos_local.py`
  - `tmp/DNS-Challenge-master/DNSMOS/README.md`

---

## 2. 当前音频预检“真实结构”全分支拆解

## 2.1 Pipeline 入口分支（`PreprocessingPipeline.process`）

Stage 2（音频预检）并不是“直接跑分类器”，而是先走缓存/跳过分支：

1. 若分离缓存命中（`skip_triage=True`）：
- 直接跳过分诊。
- 目的：恢复任务时避免重复计算。

2. 若分诊缓存命中（`triage_results.json`）：
- 要求键一致：`total_chunks/use_snr_triage/threshold/use_smart_probe/smart_probe_params`。
- 命中后仅回填 `needs_separation` 和 `recommended_model`。
- 目的：参数一致时复用分诊结果。

3. 缓存未命中：
- 执行 `SpectralTriageStage.process()`。

这意味着：任何 DNSMOS 改造都必须同步改缓存键，否则会出现“旧缓存误复用”。

## 2.2 `SpectralTriageStage` 分支（阶段级）

### 2.2.1 初始化分支

- `use_snr_triage` 默认 `False`。
- `use_smart_probe` 默认 `False`。
- 若请求 `use_smart_probe=True` 且 `util.find_spec("brouhaha") is None`：
  - 强制关闭 SmartProbe 并回退标准模式。

当前环境下（`pyproject.toml` 未安装 `brouhaha` 包）该条件恒触发。

### 2.2.2 `process()` 主分支

1. `chunks` 为空：直接返回。
2. `self.use_smart_probe=True`：进入 `_process_with_smart_probe()`。
3. 否则：进入 `_process_standard()`（当前默认路径）。

### 2.2.3 `_process_standard()` 分支

逐 chunk 执行 `classifier.diagnose_chunk()`，每个 chunk 都会：

- 写回：
  - `chunk.needs_separation`
  - `chunk.recommended_model`
  - `chunk.spectrum_diagnosis`
- 记录 triage_log（当前字段含 `snr/c50/snr_level/c50_level/triage_layer`）。
- 每 5 个 chunk checkpoint 一次（可暂停/恢复）。

### 2.2.4 `_process_with_smart_probe()` 分支

先 `smart_probe.run_probe()`，返回 `decision/cache/sequence`：

1. `SEPARATE_ALL`：
- 先应用探针已测 chunk 的结果。
- 再回退到 `_process_standard()` 继续测剩余 chunk。
- 目的：避免直接“整轨分离”，先拿到更细粒度分诊标记。

2. `PASS_ALL`：
- 全部 chunk 标记为无需分离。
- 只为探针覆盖到的 chunk 写日志项。

### 2.2.5 `_finalize_triage()` 输出

- 统计 `need_separation`、模型分布。
- 若有 `job_dir`，保存 `triage_log.json`。

## 2.3 `AudioSpectrumClassifier` 分支（核心决策层）

`diagnose_chunk()` 的真实顺序：

1. `duration < 0.5s`：
- 直接 CLEAN，`need_separation=False`。
- 目的：样本不足，不做不可靠判定。

2. 若启用 `_use_snr_strategy`：
- 尝试 `_get_brouhaha()`。
- 成功则进入 SNR+C50 三层策略。
- 失败则回退 YAMNet/规则。

3. 若 YAMNet 可用：
- 进入 `_diagnose_with_yamnet()`。

4. 否则：
- 进入 `_diagnose_with_rules()`。

## 2.4 SNR+C50 三层内部判断（保留逻辑）

`_diagnose_with_snr_c50_strategy()`：

Layer 1（SNR/C50 快筛）：
- `snr_level == high` 且 `c50_level == good` -> 放行
- `snr_level == low` -> 分离
- `c50_level == bad` -> 分离
- 其他 -> Layer 2

Layer 2（频谱特征）：
- `spectral_contrast < critical` -> 分离
- `contrast < low 且 flatness > high` -> 分离
- `high SNR + warn C50 + contrast 达标` -> 放行
- 其他 -> Layer 3

Layer 3（语义兜底）：
- 优先 YAMNet
- 若 `snr < 30` 且 YAMNet 判人声，但 `speech_score < 0.95` -> 仍分离（保守）
- 无 YAMNet 时回退规则法

## 2.5 YAMNet 分支（当前主力）

`YAMNetClassifier.classify_chunk()`（多窗口探针）：

1. A Cappella 豁免：`max_acappella > threshold` -> PASS
2. BGM 检测：`max_music > thr` 或 `avg_music > thr` -> SEPARATE
3. 纯净人声：`max_speech > thr` 且 `max_music < thr` -> PASS
4. 人声主导：`avg_speech > avg_music + delta` 且 `max_music < thr` -> PASS
5. 模糊地带：`PassToSenseVoice` -> PASS（交给后续置信度链路）

## 2.6 SmartProbe 分支（当前“保留未默认启用”）

`SmartProbeService` 的判定本质只有一条：

- `is_dirty = (snr < snr_threshold)`。

执行策略：

1. 先测中心 chunk。
2. 若中心 dirty -> `SEPARATE_ALL`。
3. 否则按“斐波那契扩散 + max_step 截断后线性”测左右。
4. 任一 dirty -> `SEPARATE_ALL`，全通过 -> `PASS_ALL`。

## 2.7 关键事实与风险

1. `brouhaha` 运行链路实际不可用：
- 当前环境中 `brouhaha` 包缺失、`pyannote.audio` 为 4.0、模型文件是 LFS 指针。
- 即使保留 SNR 代码，也基本不会进入有效推理。

2. 文档与代码存在偏差：
- `llmdoc/architecture/spectral-triage-stage.md` 写了“默认开启智能探针 + L0 平坦度探测”，代码中并无该默认行为，且未看到 L0 平坦度早筛实现。

3. SmartProbe 参数暴露不足：
- 运行参数 API 仅暴露 `smart_probe.snr_threshold`，`max_step_chunks` 未暴露到 API schema。

4. 配置/字段兼容问题待处理：
- 目前缓存键、日志字段、诊断模型都围绕 `snr/c50`；DNSMOS 替换不能只改一处。

---

## 3. 对 `dnsmos-replace-brouhaha-plan.md` 的深度评估

## 3.1 方案中正确且有价值的部分

1. 方向正确：用 ONNX DNSMOS 替代 pyannote+brouhaha，能显著降低依赖风险。
2. 识别到当前 Brouhaha 实际失效，判断成立。
3. DNSMOS 输入窗口、P808 辅助输出、阈值参数化思路总体可行。
4. 提到 YAMNet 继续作为兜底层，方向合理。
5. 提到 Optuna 自动校准，而非纯手调，方向正确。

## 3.2 需要修正/补强的关键点（重点：DNSMOS 原理与 I/O）

### 3.2.1 DNSMOS I/O 解释需更精确

根据 `tmp/DNS-Challenge-master/DNSMOS/dnsmos_local.py`：

1. 主模型：
- 输入：`input_1`，shape `N x 144160`，16kHz 波形。
- 输出：`N x 3`（raw `SIG/BAK/OVRL`）。

2. P808 模型：
- 输入：`input_1`，shape `N x 900 x 120`。
- 不是“直接波形”，是 log-mel 特征。
- 官方脚本对 `audio_seg[:-160]`（去尾 10ms）做 mel，再送 P808。

3. 校准：
- raw 分数需经多项式校准（regular 与 personalized 两套系数）。
- 方案文档应明确“阈值是针对 raw 还是 calibrated”，否则调参不可复现。

### 3.2.2 运行机制细节需要补充

1. 长音频并不是任意 hop：
- 官方脚本固定 1 秒 hop，窗口 9.01 秒，取窗口均值。

2. 单声道处理要显式定义：
- 官方脚本未严格处理多声道边界；工程实现必须定义 `mono_strategy`（建议 `mean`）。

3. personalized MOS 模式：
- 使用 `pDNSMOS/sig_bak_ovr.onnx` + 另一组多项式系数。
- 是否启用应参数化，不应硬编码。

### 3.2.3 与当前仓库结构的冲突点

1. “删除 `pyannote_compat.py`”不可直接执行：
- 该文件还被 `vad_service`、`timeline/diarization_service`、`timeline/segmentation_service` 使用，不是 Brouhaha 专属。

2. `models.yaml` 示例结构不符合当前项目规范：
- 当前模型注册是 `id/kind/framework/format/source.local_path/files` 结构，不是简化 `type/path`。

3. `pyproject` 依赖删除建议需更谨慎：
- `pyannote.audio` 仍用于 VAD/说话人链路，不能因替换 Brouhaha 直接移除。

4. 配置字段重命名（`use_snr_triage -> use_dnsmos_triage`）必须做兼容别名：
- 否则旧任务 JSON/缓存/前端请求会出现断裂。

---

## 4. 最终架构设计（精简版）：DNSMOS 主判 + 极简兜底

V3.2.0+dev.20260215.01 更新：本节按“分支减法”重写，目标是删除重复检测，合并分散逻辑，让 DNSMOS 成为唯一质量主线。

## 4.1 设计原则（强约束）

1. 只保留一条主判据：音频质量相关判断统一由 DNSMOS 负责。
2. 只保留一个语义兜底：YAMNet 仅在 DNSMOS 灰区时触发。
3. 删除重复判断：DNSMOS 可判的“噪声/清晰度”不再由传统频谱规则重复判断。
4. 前置风险优先：ASR 风险判定必须在 DNSMOS 预检前执行，命中后直接切全局分离。
5. 保持接口稳定：下游仍只消费 `needs_separation/recommended_model`。
6. 保守优先：兜底不可用时不做复杂规则推理，直接走保守策略。

## 4.2 新决策树（唯一主链）

```text
Chunk
  -> [R0] ASR 风险前置守卫（chunk内多点采样平坦度）
      -> 命中: 强制 GLOBAL 分离 + 跳过 DNSMOS 预检
      -> 未命中: 进入 DNSMOS 主链
  -> [G0] 极短片段(<0.5s): PASS
  -> [G1] DNSMOS 可用?
      -> 否: 进入兼容降级链（仅用于模型缺失）
      -> 是: DNSMOS 计算 sig/bak/ovrl/p808
          -> [D1] 硬分离区: SEPARATE
          -> [D2] 硬放行区: PASS
          -> [D3] 灰区: 进入 YAMNet-Lite
               -> Music 强: SEPARATE
               -> Speech 强: PASS
               -> 其余: SEPARATE(保守)
```

说明：
- 常态路径为 `R0 + G0 + G1 + D1/D2/D3`，不再并列多套质量判定体系。
- `YAMNet-Lite` 不是主分诊器，只是 DNSMOS 灰区裁决器。

## 4.3 明确删除/合并清单（回答“分支是否过多”）

必须删除主路径依赖：

1. `SNR/C50 Layer1` 主判（Brouhaha 相关）。
2. `Layer2 contrast+flatness` 主判。
3. `_calculate_music_score/_calculate_noise_score` 的常态判定职责。
4. 旧 SmartProbe 的 `snr < threshold` 判定核心。

仅保留为“兼容应急”的内容：

1. 规则法 `_diagnose_with_rules()`：仅在 DNSMOS 与 YAMNet 同时不可用时启用。
2. 旧字段 `snr/c50`：仅为旧日志/缓存读取兼容，不参与新决策。

## 4.4 DNSMOS 判定规则（唯一质量层）

输入统一使用 calibrated 分数：`sig/bak/ovrl/p808`。

1. 硬分离（D1）：
- `ovrl <= ovrl_sep_hard` 或
- `sig <= sig_sep_hard` 或
- `p808 <= p808_sep_hard`

2. 硬放行（D2）：
- `ovrl >= ovrl_pass_hard` 且
- `sig >= sig_pass_hard` 且
- `bak >= bak_pass_hard` 且
- `p808 >= p808_pass_soft`

3. 灰区（D3）：
- 非 D1 且非 D2 的样本。

决策来源统一标记：
- `dnsmos_hard_separate`
- `dnsmos_hard_pass`
- `dnsmos_yamnet_fallback`
- `compat_fallback`

## 4.5 YAMNet-Lite（只服务 DNSMOS 灰区）

把现有 5 条规则收敛为 3 条，减少参数与行为分叉：

1. `music_score >= yamnet_music_conf_min` -> `SEPARATE`
2. `speech_score >= yamnet_speech_conf_min` 且 `music_score < yamnet_music_weak_max` -> `PASS`
3. 其余 -> `SEPARATE`（保守）

说明：
- A Cappella、SpeechDominant 等规则不再独立保留，统一并入上面三条。
- 避免“同一语义被多规则重复判定”导致行为不透明。
- 执行设备固定为 `CPUExecutionProvider`，避免因 CUDA Provider/DLL 依赖异常导致灰区兜底链路抖动。

## 4.6 SmartProbe v2（DNSMOS 驱动）

保留“中心扩散采样”框架，删除 SNR 驱动判定：

1. probe 命中 D1 比例高于 `probe_sep_ratio_min` -> `SEPARATE_ALL`
2. probe 全为 D2 且覆盖率 >= `probe_min_coverage` -> `PASS_ALL`
3. 否则 -> `ESCALATE_STANDARD`（回到逐 chunk 主链）

新增目标：
- 只决定“是否全量快判”，不单独承担质量判定。
- 工程默认启用中心扩散探针（`use_smart_probe=true`），仅在任务配置显式关闭时回退标准逐 chunk 模式。

## 4.7 ASR 风险前置守卫（新增）

守卫位置与优先级：

1. 位置固定在 `PreprocessingPipeline` 的 Stage2（DNSMOS 预检）之前。
2. 判定命中后，直接旁路 `SpectralTriageStage`，并将当次任务分离模式强制为 `global`。
3. 命中后不再信任 `on_demand` 分离缓存，优先尝试 `global` 缓存，不命中则重跑全局分离。

chunk 内采样约束（避免“到原音频随机取点”）：

1. 采样点固定在每个 chunk 内：`5% / 25% / 50% / 75% / 95%`。
2. 采样窗口固定 `1.2s`，低能量窗口剔除（RMS 门限）。
3. 每个 chunk 至少 `4` 个有效窗口才参与风险判定。

当前阈值约束（针对 `input/日语听力测试.mp4` 校准）：

1. chunk 级命中：
- `mean_flatness >= 0.33`
- `max_flatness >= 0.65`
- `high_flatness_ratio >= 0.40`
2. 视频级命中：
- `risk_chunk_count >= max(2, ceil(0.05 * analyzed_chunk_count))`
- 同时满足 `risk_chunk_ratio >= 0.05`

## 4.8 数据结构与兼容

`SpectrumDiagnosis` 统一核心字段：
- `sig/bak/ovrl/p808`
- `need_separation`
- `decision_source`
- `decision_margin`

兼容策略：

1. 新增 `use_dnsmos_triage`，继续兼容读取 `use_snr_triage`。
2. 缓存键必须新增：
- `triage_version`
- `use_dnsmos_triage`
- `dnsmos_model_hash`
- `dnsmos_threshold_profile_hash`
3. `snr/c50` 可继续写入空值或兼容占位，但不再用于判定。

---

## 5. 参数体系（精简后）

## 5.1 固定参数（不调）

1. `window_sec=9.01`
2. `hop_sec=1.0`
3. 官方多项式校准系数
4. `mono_strategy=mean`（无业务特例不改）
5. `asr_risk_guard.sample_positions=(0.05, 0.25, 0.5, 0.75, 0.95)`
6. `asr_risk_guard.sample_window_sec=1.2`

## 5.2 必要运行参数（保留）

DNSMOS 阈值：

1. `ovrl_sep_hard`
2. `sig_sep_hard`
3. `p808_sep_hard`
4. `ovrl_pass_hard`
5. `sig_pass_hard`
6. `bak_pass_hard`
7. `p808_pass_soft`

YAMNet-Lite 阈值：

1. `yamnet_music_conf_min`
2. `yamnet_speech_conf_min`
3. `yamnet_music_weak_max`

SmartProbe v2 阈值：

1. `probe_sep_ratio_min`
2. `probe_min_coverage`
3. `probe_max_step_chunks`（固定，先不纳入自动调参）

## 5.3 明确移除参数（避免重复控制）

从“新主路径”移除：

1. `spectral_contrast_low`
2. `spectral_contrast_critical`
3. `spectral_flatness_high`
4. `spectral_flatness_noise`
5. `snr_high_threshold/snr_low_threshold/c50_*`

说明：
- 这些参数可保留在兼容层，但不再影响 DNSMOS 主链决策。

---

## 6. Optuna 调参指南（精简版）

## 6.1 哪些参数必须自动调

高优先级（必须）：

1. `ovrl_sep_hard`
2. `sig_sep_hard`
3. `p808_sep_hard`
4. `ovrl_pass_hard`
5. `sig_pass_hard`
6. `bak_pass_hard`
7. `p808_pass_soft`

中优先级（建议，仅用于时延/吞吐）：

1. `probe_sep_ratio_min`
2. `probe_min_coverage`

不调（固定）：

1. `window_sec/hop_sec`
2. DNSMOS 校准系数
3. `mono_strategy`（默认 mean）
4. `yamnet_music_conf_min`
5. `yamnet_speech_conf_min`
6. `yamnet_music_weak_max`
7. `probe_max_step_chunks`
8. `asr_risk_guard` 的窗口采样结构参数（位置、窗口长度、最小窗口数）

## 6.2 标注目标（业务导向）

标签定义保持“分离收益标签”：

- `label=SEPARATE` 若 `CER_raw - CER_sep >= cer_gain_threshold`
- 否则 `label=PASS`

这样直接优化真实业务目标，而不是拟合模型分数本身。

## 6.3 目标函数（建议）

```text
objective = 12*FN + 1*FP + 0.15*AvgLatencyMs + 0.05*FallbackRate
```

说明：
- 提高 `FN` 惩罚，确保“该分离不漏分离”。
- 降低对 fallback 的惩罚，避免为追求低 fallback 牺牲精度。

## 6.4 搜索空间（精简参数集）

```python
def suggest_params(trial):
    ovrl_sep_hard = trial.suggest_float("ovrl_sep_hard", 1.0, 2.4)
    sig_sep_hard = trial.suggest_float("sig_sep_hard", 1.0, 2.6)
    p808_sep_hard = trial.suggest_float("p808_sep_hard", 1.0, 2.4)

    ovrl_pass_hard = trial.suggest_float("ovrl_pass_hard", ovrl_sep_hard + 0.2, 3.8)
    sig_pass_hard = trial.suggest_float("sig_pass_hard", sig_sep_hard + 0.2, 4.2)
    bak_pass_hard = trial.suggest_float("bak_pass_hard", 1.8, 4.0)
    p808_pass_soft = trial.suggest_float("p808_pass_soft", p808_sep_hard + 0.2, 3.8)

    probe_sep_ratio_min = trial.suggest_float("probe_sep_ratio_min", 0.20, 0.80)
    probe_min_coverage = trial.suggest_float("probe_min_coverage", 0.15, 0.60)
```

## 6.5 训练与验证流程

1. 粗搜：`200` trials（TPE + MedianPruner）
2. 精搜：`100` trials（在前 20% 参数域局部细化）
3. 验证：按场景分层 K-fold + 独立回放集
4. 产物：
- `best_params.json`
- `trial_history.csv`
- `confusion_matrix_by_scene.json`
- `latency_profile.json`

## 6.6 回灌与灰度

1. 自动回灌到 `runtime.dnsmos/runtime.smart_probe`（`yamnet_*` 与 `probe_max_step_chunks` 固定不自动写）。
2. 按流量灰度对比旧链路：
- 漏分离率
- CER 改善
- 音频预检平均耗时

---

## 7. 落地实施步骤（按精简架构）

1. 新增 `dnsmos_service.py`，提供统一 `infer(audio)->sig/bak/ovrl/p808`。
2. 改造 `AudioSpectrumClassifier.diagnose_chunk()` 为单主链：
- `DNSMOS 硬判 -> 灰区 YAMNet-Lite -> 保守默认`
3. 下线主路径中的 SNR/C50 三层与频谱规则判定。
4. 改造 `SmartProbeService` 为 DNSMOS v2 判定接口。
5. 升级 runtime schema 与缓存键版本。
6. 增加测试：
- `test_dnsmos_service.py`
- `test_dnsmos_triage_integration.py`
- `test_dnsmos_smart_probe.py`
- `test_triage_cache_compat.py`
7. 完成 Optuna 校准后，切换默认开关到 DNSMOS 主链。

### 7.1 当前分支已落地项（V3.2.4+dev.20260225.02）

1. 新增 `backend/app/services/dnsmos_service.py`，默认 CPU 推理，支持 SIG/BAK/OVRL/P808 输出。
2. `AudioSpectrumClassifier` 已切换到 DNSMOS 主链，并输出 `decision_source/decision_margin`。
3. `SmartProbeService` 已改为 DNSMOS v2 三态判定（`SEPARATE_ALL/PASS_ALL/ESCALATE_STANDARD`）。
4. 音频预检缓存键已升级：`triage_version/use_dnsmos_triage/dnsmos_model_hash/dnsmos_threshold_profile_hash`。
5. 运行参数已新增 `runtime.dnsmos`，并同步 SmartProbe v2 与 YAMNet-Lite 参数。
6. 已新增单元测试：
- `ci_tests/unit/test_dnsmos_triage_classifier.py`
- `ci_tests/unit/test_dnsmos_smart_probe.py`
- `ci_tests/unit/test_triage_cache_compat.py`
7. 已新增自动调参服务：
- `backend/app/services/dnsmos_auto_tune_service.py`
- `GET /api/models/params` 新增 `auto_tune.dnsmos` 参数分层（mandatory/recommended/fixed）
- 新增单测：`ci_tests/unit/test_dnsmos_auto_tune_service.py`、`ci_tests/unit/test_model_runtime_params_schema_dnsmos.py`
8. 新增前置守卫：`backend/app/services/asr_risk_guard_service.py` 与 `PreprocessingPipeline` Stage1.5 接入。
9. 当前行为：命中 ASR 风险后，预检直接旁路，分离模式强制 `global`（仅对当次任务生效）。
10. `ModelRuntimeConfigService._runtime_defaults()` 已与本轮自动调参结果一致：
- `runtime.dnsmos`：
  `ovrl_sep_hard=2.0960908371`、
  `sig_sep_hard=2.5541660107`、
  `p808_sep_hard=1.7823731697`、
  `ovrl_pass_hard=2.3441922434`、
  `sig_pass_hard=2.7914833248`、
  `bak_pass_hard=1.8809400946`、
  `p808_pass_soft=3.2350228539`
- `runtime.smart_probe`：
  `probe_sep_ratio_min=0.2679550189`、
  `probe_min_coverage=0.3668912432`、
  `probe_max_step_chunks=30`

### 7.2 API 字段变更清单（音频预检）

1. `GET /api/models/params`
- 新增 `auto_tune.dnsmos.mandatory`：
  `ovrl_sep_hard/sig_sep_hard/p808_sep_hard/ovrl_pass_hard/sig_pass_hard/bak_pass_hard/p808_pass_soft`
- 新增 `auto_tune.dnsmos.recommended`：
  `probe_sep_ratio_min/probe_min_coverage`
- 新增 `auto_tune.dnsmos.fixed`：
  `yamnet_music_conf_min/yamnet_speech_conf_min/yamnet_music_weak_max/probe_max_step_chunks`

2. `PUT /api/models/runtime`
- `runtime.dnsmos` 支持 7 个核心阈值更新；
- `runtime.smart_probe` 支持 `probe_sep_ratio_min/probe_min_coverage` 更新；
- `runtime.yamnet` 参数仍可手动改，但不属于自动调参主域。

---

## 8. 验收标准（精简架构版）

功能验收：

1. 常态分诊只经过 `DNSMOS + (可选)YAMNet-Lite`。
2. 旧 `SNR/C50 + contrast/flatness` 不再参与主决策。
3. DNSMOS 缺失时能降级，但降级路径不会触发复杂多分支震荡。
4. `triage_log.json` 必含 `sig/bak/ovrl/p808/decision_source/decision_margin`。

质量验收：

1. 漏分离率不高于当前线上。
2. CER 优于当前基线。
3. 平均分诊耗时满足预算。

兼容验收：

1. 旧任务配置可平滑读取（含 `use_snr_triage`）。
2. 缓存版本升级后不会误命中旧结果。

---

## 9. 最终裁定（针对“是否需要精简”）

需要精简，而且必须精简。  
最终落地应采用“DNSMOS 主判 + YAMNet 灰区兜底 + DNSMOS 探针”的三块结构，删除 DNSMOS 可覆盖的重复检测分支，统一参数入口与调参目标，避免继续维持多套并行判定体系。
