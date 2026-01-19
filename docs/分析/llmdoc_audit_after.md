# LLMDOC 审计报告

> **审计时间**: 2026-01-18 23:07:18
> **当前版本**: V3.2.0
> **扫描文档数**: 181

---

## 一、文档分布统计

### 按目录分布

| 目录 | 文档数 |
|------|--------|
| agent\bug-investigations\2025-12 | 37 |
| architecture | 30 |
| agent\core-services\transcription-service | 18 |
| reference | 16 |
| archive\duplicate-reports-removed-20260118 | 15 |
| agent\core-services\model-management | 13 |
| guides | 10 |
| agent\frontend-components\editor | 8 |
| agent\audio-processing\separation | 6 |
| agent\bug-investigations\2026-01 | 6 |
| agent\pipeline-architecture\async-dual-pipeline | 4 |
| agent\pipeline-architecture\preprocessing | 4 |
| overview | 4 |
| agent\core-services\confidence-system | 3 |
| root | 2 |
| agent\audio-processing\spectrum-analysis | 2 |
| agent\audio-processing\vad | 2 |
| agent\frontend-components\waveform | 1 |

### 按类型分布

| 类型 | 数量 |
|------|------|
| Architecture | 29 |
| Reference | 16 |
| Guide | 10 |
| Overview | 4 |
| [Architecture | Guide | Decision | Reference | AgentReport] | 1 |

### 按状态分布

| 状态 | 数量 |
|------|------|
| Active | 59 |
| [Active | Superseded | Archived | Draft] | 1 |

---

## 二、元数据完整性问题

**发现 181 个文档缺少必需元数据**

| 文档 | 缺失字段 |
|------|---------|
| changelog.md | Type, Status, Last Updated |
| LLMDOC 通用文档规范.md | Last Updated |
| agent\audio-processing\separation\audio_spectrum_classifier_investigation_report.md | Type, Status, Last Updated |
| agent\audio-processing\separation\demucs_integration_investigation.md | Type, Status, Last Updated |
| agent\audio-processing\separation\demucs_service_legacy_architecture_investigation.md | Type, Status, Last Updated |
| agent\audio-processing\separation\gpu_coordinator_status_investigation.md | Type, Status, Last Updated |
| agent\audio-processing\separation\Phase2-现有代码调查检索摘要.md | Type, Status, Last Updated |
| agent\audio-processing\separation\spectral_triage_separation_integration_report.md | Type, Status, Last Updated |
| agent\audio-processing\spectrum-analysis\parameter_classification_investigation.md | Type, Status, Last Updated |
| agent\audio-processing\spectrum-analysis\triage-separation-progress-investigation.md | Type, Status, Last Updated |
| agent\audio-processing\vad\readme-installation-verification.md | Type, Status, Last Updated |
| agent\audio-processing\vad\vad_deprecated_code_investigation_report.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\25min_video_6min_subtitles_investigation_report.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\alignment-mechanism-investigation.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\audio_processing_architecture_migration_report.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\branch_comparison_spectral_separation_fuse.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\codec-check-frequency-investigation.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\demucs_passthrough_issue_investigation.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\dual-stream-transcription-investigation.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\encoding_unicode_fix_summary.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\environment_configuration_investigation_report.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\fast_preset_sensevoice_only_investigation.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\frontend-deep-investigation.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\frontend-innovation-details.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\frontend-legacy-investigation-20251229.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\frontend_memory_leak_investigation.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\fuse_breaker_architecture_investigation_report.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\ghost-task-and-cancel-fix-plan.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\investigation_report_last_sentence_missing.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\new_pipeline_architecture_investigation_report.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\onnxruntime_gpu_cuda_compatibility_investigation.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\phase2-pause-coverage-investigation.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\phase4-design-analysis.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\pipeline-atomicity-analysis.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\pipeline_architecture_completeness_verification_report.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\progress-bar-investigation.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\safe_shutdown_system_and_browser_tab_reuse_report.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\sensevoice_ctc_decoding_investigation_report.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\sensevoice_text_cleaning_investigation_report.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\sentence_splitting_issue_investigation_report.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\silero_vad_output_characteristics_investigation.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\sse-communication-investigation.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\subtitle_loss_investigation_v3_1_0.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\subtitle_overwrite_investigation.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\timestamp_accuracy_investigation.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\user_feedback_investigation_summary.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\v373-subtitle-persistence-bug-investigation.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\video_to_srt_gpu_checkpoint_resume_report.md | Type, Status, Last Updated |
| agent\bug-investigations\2025-12\waveform-regions-disappear-investigation.md | Type, Status, Last Updated |
| agent\bug-investigations\2026-01\architecture_extensibility_and_coupling_deep_investigation.md | Type, Status, Last Updated |

---

## 三、版本号一致性问题

**发现 0 个 Active 文档版本号不一致**

✅ 所有 Active 文档版本号一致

---

## 四、格式规范问题

**发现 178 个文档存在格式问题**

| 文档 | 问题 |
|------|------|
| LLMDOC 通用文档规范.md | 标题深度H4, 长段落 |
| agent\audio-processing\separation\audio_spectrum_classifier_investigation_report.md | 标题深度H4, 长段落 |
| agent\audio-processing\separation\demucs_integration_investigation.md | 标题深度H4, 长段落 |
| agent\audio-processing\separation\demucs_service_legacy_architecture_investigation.md | 标题深度H5, 长段落 |
| agent\audio-processing\separation\gpu_coordinator_status_investigation.md | 标题深度H4 |
| agent\audio-processing\separation\Phase2-现有代码调查检索摘要.md | 长段落 |
| agent\audio-processing\separation\spectral_triage_separation_integration_report.md | 标题深度H4, 长段落 |
| agent\audio-processing\spectrum-analysis\parameter_classification_investigation.md | 标题深度H4, 长段落 |
| agent\audio-processing\spectrum-analysis\triage-separation-progress-investigation.md | 标题深度H4, 长段落 |
| agent\audio-processing\vad\readme-installation-verification.md | 标题深度H4 |
| agent\audio-processing\vad\vad_deprecated_code_investigation_report.md | 标题深度H5, 长段落 |
| agent\bug-investigations\2025-12\25min_video_6min_subtitles_investigation_report.md | 标题深度H4 |
| agent\bug-investigations\2025-12\alignment-mechanism-investigation.md | 标题深度H5, 长段落 |
| agent\bug-investigations\2025-12\audio_processing_architecture_migration_report.md | 标题深度H5, 长段落 |
| agent\bug-investigations\2025-12\branch_comparison_spectral_separation_fuse.md | 标题深度H5, 长段落 |
| agent\bug-investigations\2025-12\codec-check-frequency-investigation.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2025-12\demucs_passthrough_issue_investigation.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2025-12\dual-stream-transcription-investigation.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2025-12\encoding_unicode_fix_summary.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2025-12\environment_configuration_investigation_report.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2025-12\fast_preset_sensevoice_only_investigation.md | 标题深度H4 |
| agent\bug-investigations\2025-12\frontend-deep-investigation.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2025-12\frontend-innovation-details.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2025-12\frontend-legacy-investigation-20251229.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2025-12\frontend_memory_leak_investigation.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2025-12\fuse_breaker_architecture_investigation_report.md | 标题深度H5, 长段落 |
| agent\bug-investigations\2025-12\ghost-task-and-cancel-fix-plan.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2025-12\investigation_report_last_sentence_missing.md | 标题深度H4 |
| agent\bug-investigations\2025-12\new_pipeline_architecture_investigation_report.md | 标题深度H5, 长段落 |
| agent\bug-investigations\2025-12\onnxruntime_gpu_cuda_compatibility_investigation.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2025-12\phase2-pause-coverage-investigation.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2025-12\phase4-design-analysis.md | 长段落 |
| agent\bug-investigations\2025-12\pipeline-atomicity-analysis.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2025-12\pipeline_architecture_completeness_verification_report.md | 标题深度H5, 长段落 |
| agent\bug-investigations\2025-12\progress-bar-investigation.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2025-12\safe_shutdown_system_and_browser_tab_reuse_report.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2025-12\sensevoice_ctc_decoding_investigation_report.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2025-12\sensevoice_text_cleaning_investigation_report.md | 长段落 |
| agent\bug-investigations\2025-12\sentence_splitting_issue_investigation_report.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2025-12\silero_vad_output_characteristics_investigation.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2025-12\sse-communication-investigation.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2025-12\subtitle_loss_investigation_v3_1_0.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2025-12\subtitle_overwrite_investigation.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2025-12\timestamp_accuracy_investigation.md | 长段落 |
| agent\bug-investigations\2025-12\user_feedback_investigation_summary.md | 标题深度H5, 长段落 |
| agent\bug-investigations\2025-12\v373-subtitle-persistence-bug-investigation.md | 长段落 |
| agent\bug-investigations\2025-12\video_to_srt_gpu_checkpoint_resume_report.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2025-12\waveform-regions-disappear-investigation.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2026-01\architecture_extensibility_and_coupling_deep_investigation.md | 标题深度H4, 长段落 |
| agent\bug-investigations\2026-01\confidence_calibration_summary.md | 长段落 |

---

## 五、Agent 报告分类情况

**发现 0 个未分类的 Agent 报告**

✅ 所有 Agent 报告已分类

---

## 六、审计总结

### 问题优先级

- **P0 (必须立即修复)**: 版本号不一致 (0个)
- **P1 (重要)**: 元数据缺失 (181个)
- **P2 (优化)**: 格式问题 (178个)
- **P2 (优化)**: Agent 未分类 (0个)

### 建议行动

2. **补充元数据**:
   人工审查每个文档，补充缺失字段


---

**报告生成时间**: 2026-01-18 23:07:18