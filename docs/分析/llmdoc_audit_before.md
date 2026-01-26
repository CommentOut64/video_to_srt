# LLMDOC 审计报告

> **审计时间**: 2026-01-18 22:01:27
> **当前版本**: V3.2.0
> **扫描文档数**: 181

---

## 一、文档分布统计

### 按目录分布

| 目录 | 文档数 |
|------|--------|
| agent | 119 |
| architecture | 30 |
| reference | 16 |
| guides | 10 |
| overview | 4 |
| root | 2 |

### 按类型分布

| 类型 | 数量 |
|------|------|
| [Architecture | Guide | Decision | Reference | AgentReport] | 1 |

### 按状态分布

| 状态 | 数量 |
|------|------|
| [Active | Superseded | Archived | Draft] | 1 |

---

## 二、元数据完整性问题

**发现 181 个文档缺少必需元数据**

| 文档 | 缺失字段 |
|------|---------|
| changelog.md | Type, Status, Last Updated |
| LLMDOC 通用文档规范.md | Last Updated |
| agent\25min_video_6min_subtitles_investigation_report.md | Type, Status, Last Updated |
| agent\720p_idle_detection_and_model_management_investigation.md | Type, Status, Last Updated |
| agent\alignment-mechanism-investigation.md | Type, Status, Last Updated |
| agent\architecture-migration-analysis.md | Type, Status, Last Updated |
| agent\architecture_extensibility_and_coupling_deep_investigation.md | Type, Status, Last Updated |
| agent\asr_abstraction_interface_research.md | Type, Status, Last Updated |
| agent\audio_processing_architecture_migration_report.md | Type, Status, Last Updated |
| agent\audio_spectrum_classifier_investigation_report.md | Type, Status, Last Updated |
| agent\branch_comparison_spectral_separation_fuse.md | Type, Status, Last Updated |
| agent\cancellation-token-investigation.md | Type, Status, Last Updated |
| agent\checkpoint_resume_complete_investigation_report.md | Type, Status, Last Updated |
| agent\codec-check-frequency-investigation.md | Type, Status, Last Updated |
| agent\complete-system-architecture-investigation-report.md | Type, Status, Last Updated |
| agent\confidence-warning-backend-investigation.md | Type, Status, Last Updated |
| agent\confidence-warning-frontend-investigation.md | Type, Status, Last Updated |
| agent\confidence_calibration_investigation.md | Type, Status, Last Updated |
| agent\confidence_calibration_summary.md | Type, Status, Last Updated |
| agent\confidence_persistence_investigation_report.md | Type, Status, Last Updated |
| agent\demucs_integration_investigation.md | Type, Status, Last Updated |
| agent\demucs_passthrough_issue_investigation.md | Type, Status, Last Updated |
| agent\demucs_service_legacy_architecture_investigation.md | Type, Status, Last Updated |
| agent\dual-pipeline-comparison-report.md | Type, Status, Last Updated |
| agent\dual-stream-transcription-investigation.md | Type, Status, Last Updated |
| agent\encoding_fix_checklist.md | Type, Status, Last Updated |
| agent\encoding_issue_investigation_report.md | Type, Status, Last Updated |
| agent\encoding_unicode_fix_summary.md | Type, Status, Last Updated |
| agent\environment_configuration_investigation_report.md | Type, Status, Last Updated |
| agent\env_loading_test_report.md | Type, Status, Last Updated |
| agent\exit_system_flow_investigation_report.md | Type, Status, Last Updated |
| agent\fast_preset_sensevoice_only_investigation.md | Type, Status, Last Updated |
| agent\ffmpeg_ffprobe_usage_report.md | Type, Status, Last Updated |
| agent\frontend-deep-investigation.md | Type, Status, Last Updated |
| agent\frontend-innovation-details.md | Type, Status, Last Updated |
| agent\frontend-legacy-investigation-20251229.md | Type, Status, Last Updated |
| agent\frontend-progressbar-v35-investigation.md | Type, Status, Last Updated |
| agent\frontend-style-analysis-report.md | Type, Status, Last Updated |
| agent\frontend-ux-analysis.md | Type, Status, Last Updated |
| agent\frontend_confidence_highlighting_investigation.md | Type, Status, Last Updated |
| agent\frontend_memory_leak_investigation.md | Type, Status, Last Updated |
| agent\frontend_tasks_interface_investigation_report.md | Type, Status, Last Updated |
| agent\frontend_update_window_preparation_investigation_report.md | Type, Status, Last Updated |
| agent\fuse_breaker_architecture_investigation_report.md | Type, Status, Last Updated |
| agent\garbage_sentence_filter_location_report.md | Type, Status, Last Updated |
| agent\ghost-task-and-cancel-fix-plan.md | Type, Status, Last Updated |
| agent\gpu_coordinator_status_investigation.md | Type, Status, Last Updated |
| agent\investigation_report_last_sentence_missing.md | Type, Status, Last Updated |
| agent\legacy_architecture_investigation_report.md | Type, Status, Last Updated |
| agent\model_manager_v2_implementation_completeness_report.md | Type, Status, Last Updated |

---

## 三、版本号一致性问题

**发现 0 个 Active 文档版本号不一致**

✅ 所有 Active 文档版本号一致

---

## 四、格式规范问题

**发现 165 个文档存在格式问题**

| 文档 | 问题 |
|------|------|
| LLMDOC 通用文档规范.md | 标题深度H4, 长段落 |
| agent\25min_video_6min_subtitles_investigation_report.md | 标题深度H4 |
| agent\720p_idle_detection_and_model_management_investigation.md | 标题深度H4, 长段落 |
| agent\alignment-mechanism-investigation.md | 标题深度H5, 长段落 |
| agent\architecture-migration-analysis.md | 长段落 |
| agent\architecture_extensibility_and_coupling_deep_investigation.md | 标题深度H4, 长段落 |
| agent\asr_abstraction_interface_research.md | 标题深度H4, 长段落 |
| agent\audio_processing_architecture_migration_report.md | 标题深度H5, 长段落 |
| agent\audio_spectrum_classifier_investigation_report.md | 标题深度H4, 长段落 |
| agent\branch_comparison_spectral_separation_fuse.md | 标题深度H5, 长段落 |
| agent\cancellation-token-investigation.md | 标题深度H4, 长段落 |
| agent\checkpoint_resume_complete_investigation_report.md | 标题深度H4, 长段落 |
| agent\codec-check-frequency-investigation.md | 标题深度H4, 长段落 |
| agent\complete-system-architecture-investigation-report.md | 标题深度H4, 长段落 |
| agent\confidence-warning-backend-investigation.md | 标题深度H4, 长段落 |
| agent\confidence-warning-frontend-investigation.md | 标题深度H4, 长段落 |
| agent\confidence_calibration_investigation.md | 标题深度H5, 长段落 |
| agent\confidence_calibration_summary.md | 长段落 |
| agent\confidence_persistence_investigation_report.md | 标题深度H4, 长段落 |
| agent\demucs_integration_investigation.md | 标题深度H4, 长段落 |
| agent\demucs_passthrough_issue_investigation.md | 标题深度H4, 长段落 |
| agent\demucs_service_legacy_architecture_investigation.md | 标题深度H5, 长段落 |
| agent\dual-pipeline-comparison-report.md | 标题深度H4, 长段落 |
| agent\dual-stream-transcription-investigation.md | 标题深度H4, 长段落 |
| agent\encoding_fix_checklist.md | 标题深度H5, 长段落 |
| agent\encoding_issue_investigation_report.md | 标题深度H5, 长段落 |
| agent\encoding_unicode_fix_summary.md | 标题深度H4, 长段落 |
| agent\environment_configuration_investigation_report.md | 标题深度H4, 长段落 |
| agent\env_loading_test_report.md | 长段落 |
| agent\exit_system_flow_investigation_report.md | 标题深度H5, 长段落 |
| agent\fast_preset_sensevoice_only_investigation.md | 标题深度H4 |
| agent\ffmpeg_ffprobe_usage_report.md | 标题深度H4 |
| agent\frontend-deep-investigation.md | 标题深度H4, 长段落 |
| agent\frontend-innovation-details.md | 标题深度H4, 长段落 |
| agent\frontend-legacy-investigation-20251229.md | 标题深度H4, 长段落 |
| agent\frontend-progressbar-v35-investigation.md | 标题深度H4 |
| agent\frontend-style-analysis-report.md | 长段落 |
| agent\frontend-ux-analysis.md | 标题深度H4, 长段落 |
| agent\frontend_confidence_highlighting_investigation.md | 标题深度H4 |
| agent\frontend_memory_leak_investigation.md | 标题深度H4, 长段落 |
| agent\frontend_tasks_interface_investigation_report.md | 标题深度H4 |
| agent\frontend_update_window_preparation_investigation_report.md | 标题深度H4, 长段落 |
| agent\fuse_breaker_architecture_investigation_report.md | 标题深度H5, 长段落 |
| agent\garbage_sentence_filter_location_report.md | 标题深度H4 |
| agent\ghost-task-and-cancel-fix-plan.md | 标题深度H4, 长段落 |
| agent\gpu_coordinator_status_investigation.md | 标题深度H4 |
| agent\investigation_report_last_sentence_missing.md | 标题深度H4 |
| agent\legacy_architecture_investigation_report.md | 标题深度H4, 长段落 |
| agent\model_manager_v2_implementation_completeness_report.md | 标题深度H4, 长段落 |
| agent\multi-page-conflict-investigation.md | 标题深度H4, 长段落 |

---

## 五、Agent 报告分类情况

**发现 119 个未分类的 Agent 报告**

以下报告位于 `agent/` 根目录，需要分类：

- 25min_video_6min_subtitles_investigation_report.md
- 720p_idle_detection_and_model_management_investigation.md
- alignment-mechanism-investigation.md
- architecture-migration-analysis.md
- architecture_extensibility_and_coupling_deep_investigation.md
- asr_abstraction_interface_research.md
- audio_processing_architecture_migration_report.md
- audio_spectrum_classifier_investigation_report.md
- branch_comparison_spectral_separation_fuse.md
- cancellation-token-investigation.md
- checkpoint_resume_complete_investigation_report.md
- codec-check-frequency-investigation.md
- complete-system-architecture-investigation-report.md
- confidence-warning-backend-investigation.md
- confidence-warning-frontend-investigation.md
- confidence_calibration_investigation.md
- confidence_calibration_summary.md
- confidence_persistence_investigation_report.md
- demucs_integration_investigation.md
- demucs_passthrough_issue_investigation.md
- demucs_service_legacy_architecture_investigation.md
- dual-pipeline-comparison-report.md
- dual-stream-transcription-investigation.md
- encoding_fix_checklist.md
- encoding_issue_investigation_report.md
- encoding_unicode_fix_summary.md
- environment_configuration_investigation_report.md
- env_loading_test_report.md
- exit_system_flow_investigation_report.md
- fast_preset_sensevoice_only_investigation.md
- ffmpeg_ffprobe_usage_report.md
- frontend-deep-investigation.md
- frontend-innovation-details.md
- frontend-legacy-investigation-20251229.md
- frontend-progressbar-v35-investigation.md
- frontend-style-analysis-report.md
- frontend-ux-analysis.md
- frontend_confidence_highlighting_investigation.md
- frontend_memory_leak_investigation.md
- frontend_tasks_interface_investigation_report.md
- frontend_update_window_preparation_investigation_report.md
- fuse_breaker_architecture_investigation_report.md
- garbage_sentence_filter_location_report.md
- ghost-task-and-cancel-fix-plan.md
- gpu_coordinator_status_investigation.md
- investigation_report_last_sentence_missing.md
- legacy_architecture_investigation_report.md
- model_manager_v2_implementation_completeness_report.md
- multi-page-conflict-investigation.md
- new_pipeline_architecture_investigation_report.md
- onnxruntime_gpu_cuda_compatibility_investigation.md
- parameter_classification_investigation.md
- phase2-pause-coverage-investigation.md
- Phase2-现有代码调查检索摘要.md
- phase3-alignment-investigation.md
- phase3-frontend-progress-implementation-status.md
- phase3_implementation_verification_report.md
- phase3_transcription_service_components_status_report.md
- phase4-design-analysis.md
- phase4-prerequisite-investigation.md
- pipeline-atomicity-analysis.md
- pipeline_architecture_completeness_verification_report.md
- preprocessing-bypass-investigation.md
- preprocessing-pipeline-investigation.md
- preprocessing_pipeline_integration_investigation.md
- process_architecture_investigation_report.md
- progress-bar-investigation.md
- progress-update-investigation.md
- readme-installation-verification.md
- safe_shutdown_system_and_browser_tab_reuse_report.md
- sensevoice-ctc-deduplication-report.md
- sensevoice-onnx-model-investigation.md
- sensevoice-onnx-parameters-investigation.md
- sensevoice-raw-subtitle-save-investigation.md
- sensevoice_confidence_calculation_report.md
- sensevoice_ctc_decoder_special_tags_report.md
- sensevoice_ctc_decoding_investigation_report.md
- sensevoice_ctc_decoding_process_analysis.md
- sensevoice_only_mode_display_confidence_investigation.md
- sensevoice_phase1_code_structure_report.md
- sensevoice_text_cleaning_investigation_report.md
- sensevoice_word_confidence_investigation.md
- sentence-splitting-four-layers-architecture-investigation.md
- sentence-splitting-investigation.md
- sentence_splitting_issue_investigation_report.md
- shift_click_subtitle_move_investigation_report.md
- silero_vad_output_characteristics_investigation.md
- space_key_focus_issue_investigation_report.md
- spectral_triage_false_positive_misclassification_investigation.md
- spectral_triage_separation_integration_report.md
- sse-communication-investigation.md
- sse-progress-investigation.md
- sse_and_subtitle_streaming_investigation_report.md
- startup_browser_mechanism_report.md
- subtitle-backend-integration-investigation.md
- subtitle_editing_infrastructure_investigation.md
- subtitle_export_and_media_investigation.md
- subtitle_loss_investigation_v3_1_0.md
- subtitle_overwrite_investigation.md
- timestamp_accuracy_investigation.md
- transcription_post_process_enhancement_investigation.md
- transcription_service_investigation_report.md
- transcription_service_methods_investigation.md
- triage-separation-progress-investigation.md
- user_feedback_investigation_summary.md
- v3.2.0_model_manager_local_paths_investigation.md
- v3.2.2_hardlimit_investigation.md
- v373-subtitle-persistence-bug-investigation.md
- vad_deprecated_code_investigation_report.md
- video_file_copy_investigation_report.md
- video_to_srt_gpu_checkpoint_resume_report.md
- video_to_srt_gpu_complete_pipeline_architecture_report.md
- vue-component-complexity-analysis.md
- waveform-regions-disappear-investigation.md
- whisperx_migration_verification_report.md
- whisper_model_configuration_investigation.md
- whisper_service_investigation_report.md
- windows_python_encoding_issues_investigation.md
- 播放进度条跳回问题深入调查报告.md

---

## 六、审计总结

### 问题优先级

- **P0 (必须立即修复)**: 版本号不一致 (0个)
- **P1 (重要)**: 元数据缺失 (181个)
- **P2 (优化)**: 格式问题 (165个)
- **P2 (优化)**: Agent 未分类 (119个)

### 建议行动

2. **补充元数据**:
   人工审查每个文档，补充缺失字段

3. **Agent 报告分类**:
   ```bash
   python scripts/llmdoc_classify_agent.py \
     --output docs/分析/agent_classification_plan.md
   ```


---

**报告生成时间**: 2026-01-18 22:01:27