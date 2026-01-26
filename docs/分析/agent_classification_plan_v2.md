# Agent 报告分类计划

> **生成时间**: 2026-01-18 23:00:49
> **未分类报告数**: 104

---

## 使用说明

1. 人工审查下方的分类建议
2. 确认或修改推荐分类
3. 执行批量移动命令

```bash
# 预览（不执行）
python scripts/llmdoc_move_agent_reports.py \
  --plan docs/分析/agent_classification_plan.md \
  --dry-run

# 确认无误后执行
python scripts/llmdoc_move_agent_reports.py \
  --plan docs/分析/agent_classification_plan.md \
  --execute
```

---

## 分类统计

| 分类 | 数量 |
|------|------|
| bug-investigations/2025-12 | 37 |
| core-services/transcription-service | 18 |
| core-services/model-management | 13 |
| frontend-components/editor | 8 |
| bug-investigations/2026-01 | 6 |
| audio-processing/separation | 6 |
| pipeline-architecture/async-dual-pipeline | 4 |
| pipeline-architecture/preprocessing | 4 |
| core-services/confidence-system | 3 |
| audio-processing/spectrum-analysis | 2 |
| audio-processing/vad | 2 |
| frontend-components/waveform | 1 |

---

## 分类明细

### audio-processing/separation

共 6 个报告：

#### demucs_service_legacy_architecture_investigation.md

- **标题**: 
- **置信度**: 🔴 42%
- **匹配原因**: 文件名:demucs, demucs(80), separation(12), 分离(54)
- **目标路径**: `agent/audio-processing/separation/demucs_service_legacy_architecture_investigation.md`

#### spectral_triage_separation_integration_report.md

- **标题**: 
- **置信度**: 🔴 37%
- **匹配原因**: 文件名:separation, demucs(27), separation(54), 分离(48)
- **目标路径**: `agent/audio-processing/separation/spectral_triage_separation_integration_report.md`

#### demucs_integration_investigation.md

- **标题**: 
- **置信度**: 🔴 32%
- **匹配原因**: 文件名:demucs, demucs(25), separation(7), 分离(28)
- **目标路径**: `agent/audio-processing/separation/demucs_integration_investigation.md`

#### audio_spectrum_classifier_investigation_report.md

- **标题**: 
- **置信度**: 🔴 27%
- **匹配原因**: demucs(14), separation(6), 分离(14)
- **目标路径**: `agent/audio-processing/separation/audio_spectrum_classifier_investigation_report.md`

#### Phase2-现有代码调查检索摘要.md

- **标题**: Phase 2 现有代码调查检索摘要
- **置信度**: 🔴 21%
- **匹配原因**: demucs(9), separation(3), 分离(6)
- **目标路径**: `agent/audio-processing/separation/Phase2-现有代码调查检索摘要.md`

#### gpu_coordinator_status_investigation.md

- **标题**: 
- **置信度**: 🔴 11%
- **匹配原因**: demucs(19), 分离(7)
- **目标路径**: `agent/audio-processing/separation/gpu_coordinator_status_investigation.md`

### audio-processing/spectrum-analysis

共 2 个报告：

#### parameter_classification_investigation.md

- **标题**: video_to_srt_gpu 项目参数分类调查报告
- **置信度**: 🔴 40%
- **匹配原因**: spectrum(45), 频谱(21), yamnet(24)
- **目标路径**: `agent/audio-processing/spectrum-analysis/parameter_classification_investigation.md`

#### triage-separation-progress-investigation.md

- **标题**: 分诊检测和人声分离实现调查报告
- **置信度**: 🔴 40%
- **匹配原因**: spectrum(13), 频谱(12), yamnet(9)
- **目标路径**: `agent/audio-processing/spectrum-analysis/triage-separation-progress-investigation.md`

### audio-processing/vad

共 2 个报告：

#### vad_deprecated_code_investigation_report.md

- **标题**: 
- **置信度**: 🔴 17%
- **匹配原因**: 文件名:vad, vad(133), silero(22)
- **目标路径**: `agent/audio-processing/vad/vad_deprecated_code_investigation_report.md`

#### readme-installation-verification.md

- **标题**: 
- **置信度**: 🔴 5%
- **匹配原因**: vad(2), silero(2)
- **目标路径**: `agent/audio-processing/vad/readme-installation-verification.md`

### bug-investigations/2025-12

共 37 个报告：

#### 25min_video_6min_subtitles_investigation_report.md

- **标题**: 
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/25min_video_6min_subtitles_investigation_report.md`

#### alignment-mechanism-investigation.md

- **标题**: 对齐机制深度调查报告
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/alignment-mechanism-investigation.md`

#### audio_processing_architecture_migration_report.md

- **标题**: 
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/audio_processing_architecture_migration_report.md`

#### branch_comparison_spectral_separation_fuse.md

- **标题**: 
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/branch_comparison_spectral_separation_fuse.md`

#### codec-check-frequency-investigation.md

- **标题**: 视频编码检测频繁触发调查报告
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/codec-check-frequency-investigation.md`

#### demucs_passthrough_issue_investigation.md

- **标题**: 
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/demucs_passthrough_issue_investigation.md`

#### dual-stream-transcription-investigation.md

- **标题**: 快慢双流转录架构深度调查报告
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/dual-stream-transcription-investigation.md`

#### encoding_unicode_fix_summary.md

- **标题**: 编码问题修复总结 (V3.1.0)
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/encoding_unicode_fix_summary.md`

#### environment_configuration_investigation_report.md

- **标题**: video_to_srt_gpu 完整环境配置调查报告
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词, 问题已解决
- **类型**: Bug 调查
- **状态**: 已解决
- **目标路径**: `agent/bug-investigations/2025-12/environment_configuration_investigation_report.md`

#### fast_preset_sensevoice_only_investigation.md

- **标题**: 前端"极速转录"预设配置不生效的根本原因调查报告
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/fast_preset_sensevoice_only_investigation.md`

#### frontend-deep-investigation.md

- **标题**: 前端代码深度调查报告
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词, 问题已解决
- **类型**: Bug 调查
- **状态**: 已解决
- **目标路径**: `agent/bug-investigations/2025-12/frontend-deep-investigation.md`

#### frontend-innovation-details.md

- **标题**: 前端创新实现深度技术调查报告
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词, 问题已解决
- **类型**: Bug 调查
- **状态**: 已解决
- **目标路径**: `agent/bug-investigations/2025-12/frontend-innovation-details.md`

#### frontend-legacy-investigation-20251229.md

- **标题**: 前端旧架构残留及功能不匹配调查报告
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/frontend-legacy-investigation-20251229.md`

#### frontend_memory_leak_investigation.md

- **标题**: 前端内存泄漏和长时间使用崩溃问题调查报告
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/frontend_memory_leak_investigation.md`

#### fuse_breaker_architecture_investigation_report.md

- **标题**: 
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/fuse_breaker_architecture_investigation_report.md`

#### ghost-task-and-cancel-fix-plan.md

- **标题**: 幽灵任务与暂停/取消缺陷修复方案
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/ghost-task-and-cancel-fix-plan.md`

#### investigation_report_last_sentence_missing.md

- **标题**: 
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/investigation_report_last_sentence_missing.md`

#### new_pipeline_architecture_investigation_report.md

- **标题**: 
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/new_pipeline_architecture_investigation_report.md`

#### onnxruntime_gpu_cuda_compatibility_investigation.md

- **标题**: 
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词, 问题已解决
- **类型**: Bug 调查
- **状态**: 已解决
- **目标路径**: `agent/bug-investigations/2025-12/onnxruntime_gpu_cuda_compatibility_investigation.md`

#### phase2-pause-coverage-investigation.md

- **标题**: Phase 2: 暂停机制完整覆盖的当前代码状态调查报告
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/phase2-pause-coverage-investigation.md`

#### phase4-design-analysis.md

- **标题**: Phase 4 双流流水线组装方案合理性分析报告
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词, 问题已解决
- **类型**: Bug 调查
- **状态**: 已解决
- **目标路径**: `agent/bug-investigations/2025-12/phase4-design-analysis.md`

#### pipeline-atomicity-analysis.md

- **标题**: 流水线各阶段断点续传可行性深度分析
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/pipeline-atomicity-analysis.md`

#### pipeline_architecture_completeness_verification_report.md

- **标题**: 
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/pipeline_architecture_completeness_verification_report.md`

#### progress-bar-investigation.md

- **标题**: 进度条问题深度调查报告
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/progress-bar-investigation.md`

#### safe_shutdown_system_and_browser_tab_reuse_report.md

- **标题**: 
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/safe_shutdown_system_and_browser_tab_reuse_report.md`

#### sensevoice_ctc_decoding_investigation_report.md

- **标题**: 
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词, 问题已解决
- **类型**: Bug 调查
- **状态**: 已解决
- **目标路径**: `agent/bug-investigations/2025-12/sensevoice_ctc_decoding_investigation_report.md`

#### sensevoice_text_cleaning_investigation_report.md

- **标题**: SenseVoice 文本清理问题调查报告
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/sensevoice_text_cleaning_investigation_report.md`

#### sentence_splitting_issue_investigation_report.md

- **标题**: 
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/sentence_splitting_issue_investigation_report.md`

#### silero_vad_output_characteristics_investigation.md

- **标题**: Silero VAD 原始输出特征调查报告
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/silero_vad_output_characteristics_investigation.md`

#### sse-communication-investigation.md

- **标题**: SSE 管理和前后端通讯机制深度调查报告
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词, 问题已解决
- **类型**: Bug 调查
- **状态**: 已解决
- **目标路径**: `agent/bug-investigations/2025-12/sse-communication-investigation.md`

#### subtitle_loss_investigation_v3_1_0.md

- **标题**: V3.1.0 字幕丢失问题深度调查报告
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/subtitle_loss_investigation_v3_1_0.md`

#### subtitle_overwrite_investigation.md

- **标题**: 
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/subtitle_overwrite_investigation.md`

#### timestamp_accuracy_investigation.md

- **标题**: 时间戳准确性调查报告
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/timestamp_accuracy_investigation.md`

#### user_feedback_investigation_summary.md

- **标题**: 
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/user_feedback_investigation_summary.md`

#### v373-subtitle-persistence-bug-investigation.md

- **标题**: V3.1.0 字幕覆盖和时间戳丢失问题调查报告
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/v373-subtitle-persistence-bug-investigation.md`

#### video_to_srt_gpu_checkpoint_resume_report.md

- **标题**: Video-to-SRT GPU 项目断点保存和恢复机制调查报告
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/video_to_srt_gpu_checkpoint_resume_report.md`

#### waveform-regions-disappear-investigation.md

- **标题**: 
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2025-12/waveform-regions-disappear-investigation.md`

### bug-investigations/2026-01

共 6 个报告：

#### architecture_extensibility_and_coupling_deep_investigation.md

- **标题**: 架构可扩展性、耦合度与通用性深度调查报告
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词, 问题已解决
- **类型**: Bug 调查
- **状态**: 已解决
- **目标路径**: `agent/bug-investigations/2026-01/architecture_extensibility_and_coupling_deep_investigation.md`

#### confidence_calibration_summary.md

- **标题**: 置信度校准快速参考
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2026-01/confidence_calibration_summary.md`

#### frontend-style-analysis-report.md

- **标题**: 前端 Vue 组件样式规范违规分析报告
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2026-01/frontend-style-analysis-report.md`

#### sensevoice_word_confidence_investigation.md

- **标题**: SenseVoice 词级别置信度使用情况深度调查报告
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2026-01/sensevoice_word_confidence_investigation.md`

#### v3.2.0_model_manager_local_paths_investigation.md

- **标题**: 
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2026-01/v3.2.0_model_manager_local_paths_investigation.md`

#### video_file_copy_investigation_report.md

- **标题**: 
- **置信度**: 🟢 95%
- **匹配原因**: 包含 Bug 调查关键词
- **类型**: Bug 调查
- **状态**: 未知
- **目标路径**: `agent/bug-investigations/2026-01/video_file_copy_investigation_report.md`

### core-services/confidence-system

共 3 个报告：

#### sensevoice_confidence_calculation_report.md

- **标题**: 
- **置信度**: 🔴 15%
- **匹配原因**: 文件名:confidence, confidence(31), 置信度(54)
- **目标路径**: `agent/core-services/confidence-system/sensevoice_confidence_calculation_report.md`

#### sensevoice_only_mode_display_confidence_investigation.md

- **标题**: 
- **置信度**: 🔴 15%
- **匹配原因**: 文件名:confidence, confidence(94), 置信度(23)
- **目标路径**: `agent/core-services/confidence-system/sensevoice_only_mode_display_confidence_investigation.md`

#### frontend_confidence_highlighting_investigation.md

- **标题**: 
- **置信度**: 🔴 11%
- **匹配原因**: 文件名:confidence, confidence(4), 置信度(14)
- **目标路径**: `agent/core-services/confidence-system/frontend_confidence_highlighting_investigation.md`

### core-services/model-management

共 13 个报告：

#### model_manager_v2_implementation_completeness_report.md

- **标题**: 
- **置信度**: 🔴 25%
- **匹配原因**: 文件名:model_manager, model(92), 模型(63), manager(37)
- **目标路径**: `agent/core-services/model-management/model_manager_v2_implementation_completeness_report.md`

#### process_architecture_investigation_report.md

- **标题**: 
- **置信度**: 🔴 25%
- **匹配原因**: model(31), 模型(46), manager(15)
- **目标路径**: `agent/core-services/model-management/process_architecture_investigation_report.md`

#### sensevoice_phase1_code_structure_report.md

- **标题**: 
- **置信度**: 🔴 25%
- **匹配原因**: model(15), 模型(16), manager(10)
- **目标路径**: `agent/core-services/model-management/sensevoice_phase1_code_structure_report.md`

#### whisper_service_investigation_report.md

- **标题**: 
- **置信度**: 🔴 21%
- **匹配原因**: model(12), 模型(22), manager(8)
- **目标路径**: `agent/core-services/model-management/whisper_service_investigation_report.md`

#### 720p_idle_detection_and_model_management_investigation.md

- **标题**: 720p 空闲检测与模型管理调查报告
- **置信度**: 🔴 16%
- **匹配原因**: model(24), 模型(33)
- **目标路径**: `agent/core-services/model-management/720p_idle_detection_and_model_management_investigation.md`

#### sensevoice-onnx-model-investigation.md

- **标题**: 
- **置信度**: 🔴 11%
- **匹配原因**: model(82), 模型(38)
- **目标路径**: `agent/core-services/model-management/sensevoice-onnx-model-investigation.md`

#### multi-page-conflict-investigation.md

- **标题**: 
- **置信度**: 🔴 9%
- **匹配原因**: model(1), 模型(1), manager(8)
- **目标路径**: `agent/core-services/model-management/multi-page-conflict-investigation.md`

#### sensevoice-onnx-parameters-investigation.md

- **标题**: 
- **置信度**: 🔴 9%
- **匹配原因**: model(24), 模型(32)
- **目标路径**: `agent/core-services/model-management/sensevoice-onnx-parameters-investigation.md`

#### exit_system_flow_investigation_report.md

- **标题**: 
- **置信度**: 🔴 7%
- **匹配原因**: model(1), 模型(2), manager(1)
- **目标路径**: `agent/core-services/model-management/exit_system_flow_investigation_report.md`

#### frontend_update_window_preparation_investigation_report.md

- **标题**: 
- **置信度**: 🔴 7%
- **匹配原因**: model(3), manager(5)
- **目标路径**: `agent/core-services/model-management/frontend_update_window_preparation_investigation_report.md`

#### startup_browser_mechanism_report.md

- **标题**: 
- **置信度**: 🔴 6%
- **匹配原因**: model(2), 模型(2)
- **目标路径**: `agent/core-services/model-management/startup_browser_mechanism_report.md`

#### env_loading_test_report.md

- **标题**: .env 配置加载功能测试报告
- **置信度**: 🔴 5%
- **匹配原因**: model(13), 模型(3)
- **目标路径**: `agent/core-services/model-management/env_loading_test_report.md`

#### ffmpeg_ffprobe_usage_report.md

- **标题**: 
- **置信度**: 🔴 5%
- **匹配原因**: manager(4)
- **目标路径**: `agent/core-services/model-management/ffmpeg_ffprobe_usage_report.md`

### core-services/transcription-service

共 18 个报告：

#### transcription_service_methods_investigation.md

- **标题**: 
- **置信度**: 🔴 22%
- **匹配原因**: 文件名:transcription_service, transcription(6), 转录(4), subtitle(4)
- **目标路径**: `agent/core-services/transcription-service/transcription_service_methods_investigation.md`

#### sensevoice-raw-subtitle-save-investigation.md

- **标题**: 
- **置信度**: 🔴 20%
- **匹配原因**: transcription(22), 转录(13), subtitle(8)
- **目标路径**: `agent/core-services/transcription-service/sensevoice-raw-subtitle-save-investigation.md`

#### confidence_persistence_investigation_report.md

- **标题**: 置信度数据持久化调查报告
- **置信度**: 🔴 16%
- **匹配原因**: transcription(15), 转录(4), subtitle(47)
- **目标路径**: `agent/core-services/transcription-service/confidence_persistence_investigation_report.md`

#### phase3_transcription_service_components_status_report.md

- **标题**: 
- **置信度**: 🔴 16%
- **匹配原因**: 文件名:transcription_service, transcription(14), 转录(9), subtitle(2)
- **目标路径**: `agent/core-services/transcription-service/phase3_transcription_service_components_status_report.md`

#### transcription_service_investigation_report.md

- **标题**: 
- **置信度**: 🔴 16%
- **匹配原因**: 文件名:transcription_service, transcription(17), 转录(14)
- **目标路径**: `agent/core-services/transcription-service/transcription_service_investigation_report.md`

#### phase3_implementation_verification_report.md

- **标题**: 
- **置信度**: 🔴 15%
- **匹配原因**: transcription(3), 转录(10), subtitle(6)
- **目标路径**: `agent/core-services/transcription-service/phase3_implementation_verification_report.md`

#### video_to_srt_gpu_complete_pipeline_architecture_report.md

- **标题**: video_to_srt_gpu 完整流水线架构调查报告
- **置信度**: 🔴 15%
- **匹配原因**: transcription(14), 转录(10), subtitle(9)
- **目标路径**: `agent/core-services/transcription-service/video_to_srt_gpu_complete_pipeline_architecture_report.md`

#### confidence-warning-backend-investigation.md

- **标题**: 
- **置信度**: 🔴 15%
- **匹配原因**: transcription(12), 转录(2), subtitle(24)
- **目标路径**: `agent/core-services/transcription-service/confidence-warning-backend-investigation.md`

#### asr_abstraction_interface_research.md

- **标题**: 
- **置信度**: 🔴 13%
- **匹配原因**: transcription(6), 转录(2), subtitle(6)
- **目标路径**: `agent/core-services/transcription-service/asr_abstraction_interface_research.md`

#### confidence-warning-frontend-investigation.md

- **标题**: 前端置信度警告系统实现调查报告
- **置信度**: 🔴 13%
- **匹配原因**: transcription(6), subtitle(59)
- **目标路径**: `agent/core-services/transcription-service/confidence-warning-frontend-investigation.md`

#### phase3-alignment-investigation.md

- **标题**: 
- **置信度**: 🔴 13%
- **匹配原因**: 转录(4), subtitle(4)
- **目标路径**: `agent/core-services/transcription-service/phase3-alignment-investigation.md`

#### sse_and_subtitle_streaming_investigation_report.md

- **标题**: 
- **置信度**: 🔴 13%
- **匹配原因**: transcription(11), 转录(26)
- **目标路径**: `agent/core-services/transcription-service/sse_and_subtitle_streaming_investigation_report.md`

#### garbage_sentence_filter_location_report.md

- **标题**: 
- **置信度**: 🔴 12%
- **匹配原因**: transcription(5), 转录(2), subtitle(2)
- **目标路径**: `agent/core-services/transcription-service/garbage_sentence_filter_location_report.md`

#### sentence-splitting-investigation.md

- **标题**: 分句实现调查报告
- **置信度**: 🔴 12%
- **匹配原因**: transcription(4), 转录(3)
- **目标路径**: `agent/core-services/transcription-service/sentence-splitting-investigation.md`

#### sentence-splitting-four-layers-architecture-investigation.md

- **标题**: 
- **置信度**: 🔴 11%
- **匹配原因**: transcription(2), subtitle(15)
- **目标路径**: `agent/core-services/transcription-service/sentence-splitting-four-layers-architecture-investigation.md`

#### sse-progress-investigation.md

- **标题**: 
- **置信度**: 🔴 11%
- **匹配原因**: transcription(14), 转录(2), subtitle(35)
- **目标路径**: `agent/core-services/transcription-service/sse-progress-investigation.md`

#### shift_click_subtitle_move_investigation_report.md

- **标题**: 
- **置信度**: 🔴 6%
- **匹配原因**: subtitle(14)
- **目标路径**: `agent/core-services/transcription-service/shift_click_subtitle_move_investigation_report.md`

#### transcription_post_process_enhancement_investigation.md

- **标题**: 
- **置信度**: 🔴 4%
- **匹配原因**: transcription(12), 转录(1)
- **目标路径**: `agent/core-services/transcription-service/transcription_post_process_enhancement_investigation.md`

### frontend-components/editor

共 8 个报告：

#### frontend-ux-analysis.md

- **标题**: 前端用户体验与视觉交互调查报告
- **置信度**: 🔴 25%
- **匹配原因**: 文件名:frontend, editor(15), 编辑(14), waveform(4)
- **目标路径**: `agent/frontend-components/editor/frontend-ux-analysis.md`

#### subtitle_editing_infrastructure_investigation.md

- **标题**: 
- **置信度**: 🔴 20%
- **匹配原因**: editor(11), 编辑(26), waveform(4)
- **目标路径**: `agent/frontend-components/editor/subtitle_editing_infrastructure_investigation.md`

#### subtitle-backend-integration-investigation.md

- **标题**: 字幕切分功能后端集成调查报告
- **置信度**: 🔴 18%
- **匹配原因**: editor(4), 编辑(27), waveform(3)
- **目标路径**: `agent/frontend-components/editor/subtitle-backend-integration-investigation.md`

#### subtitle_export_and_media_investigation.md

- **标题**: 
- **置信度**: 🔴 18%
- **匹配原因**: editor(4), 编辑(8), waveform(3)
- **目标路径**: `agent/frontend-components/editor/subtitle_export_and_media_investigation.md`

#### vue-component-complexity-analysis.md

- **标题**: 
- **置信度**: 🔴 18%
- **匹配原因**: editor(34), 编辑(15), waveform(14)
- **目标路径**: `agent/frontend-components/editor/vue-component-complexity-analysis.md`

#### frontend_tasks_interface_investigation_report.md

- **标题**: 
- **置信度**: 🔴 11%
- **匹配原因**: 文件名:frontend, editor(10), 编辑(5)
- **目标路径**: `agent/frontend-components/editor/frontend_tasks_interface_investigation_report.md`

#### phase3-frontend-progress-implementation-status.md

- **标题**: 
- **置信度**: 🔴 10%
- **匹配原因**: 文件名:frontend, editor(21), 编辑(3)
- **目标路径**: `agent/frontend-components/editor/phase3-frontend-progress-implementation-status.md`

#### 播放进度条跳回问题深入调查报告.md

- **标题**: 播放视频时拖动进度条跳回原进度点问题深入调查报告
- **置信度**: 🔴 5%
- **匹配原因**: waveform(3)
- **目标路径**: `agent/frontend-components/editor/播放进度条跳回问题深入调查报告.md`

### frontend-components/waveform

共 1 个报告：

#### space_key_focus_issue_investigation_report.md

- **标题**: 
- **置信度**: 🔴 13%
- **匹配原因**: waveform(9), 波形(5)
- **目标路径**: `agent/frontend-components/waveform/space_key_focus_issue_investigation_report.md`

### pipeline-architecture/async-dual-pipeline

共 4 个报告：

#### complete-system-architecture-investigation-report.md

- **标题**: 
- **置信度**: 🔴 25%
- **匹配原因**: async(6), 异步(1), dual(8)
- **目标路径**: `agent/pipeline-architecture/async-dual-pipeline/complete-system-architecture-investigation-report.md`

#### architecture-migration-analysis.md

- **标题**: 旧架构 vs 新架构对比分析
- **置信度**: 🔴 12%
- **匹配原因**: async(5), dual(7)
- **目标路径**: `agent/pipeline-architecture/async-dual-pipeline/architecture-migration-analysis.md`

#### v3.2.2_hardlimit_investigation.md

- **标题**: 
- **置信度**: 🔴 11%
- **匹配原因**: 无
- **目标路径**: `agent/pipeline-architecture/async-dual-pipeline/v3.2.2_hardlimit_investigation.md`

#### phase4-prerequisite-investigation.md

- **标题**: 
- **置信度**: 🔴 10%
- **匹配原因**: 无
- **目标路径**: `agent/pipeline-architecture/async-dual-pipeline/phase4-prerequisite-investigation.md`

### pipeline-architecture/preprocessing

共 4 个报告：

#### preprocessing_pipeline_integration_investigation.md

- **标题**: 
- **置信度**: 🔴 30%
- **匹配原因**: 文件名:preprocessing, preprocessing(17), 预处理(2), spectral(13)
- **目标路径**: `agent/pipeline-architecture/preprocessing/preprocessing_pipeline_integration_investigation.md`

#### cancellation-token-investigation.md

- **标题**: CancellationToken 与暂停/恢复机制完整实现调查
- **置信度**: 🔴 26%
- **匹配原因**: preprocessing(20), 预处理(19), spectral(5)
- **目标路径**: `agent/pipeline-architecture/preprocessing/cancellation-token-investigation.md`

#### checkpoint_resume_complete_investigation_report.md

- **标题**: 断点恢复机制完整调查报告
- **置信度**: 🔴 26%
- **匹配原因**: preprocessing(12), 预处理(6), spectral(13)
- **目标路径**: `agent/pipeline-architecture/preprocessing/checkpoint_resume_complete_investigation_report.md`

#### spectral_triage_false_positive_misclassification_investigation.md

- **标题**: 
- **置信度**: 🔴 25%
- **匹配原因**: 文件名:spectral, preprocessing(2), spectral(19)
- **目标路径**: `agent/pipeline-architecture/preprocessing/spectral_triage_false_positive_misclassification_investigation.md`


---

## 需要人工确认的报告

以下 61 个报告需要人工确认分类：

- **720p_idle_detection_and_model_management_investigation.md**
  - 标题: 720p 空闲检测与模型管理调查报告
  - 推荐: core-services/model-management
  - 置信度: 16%

- **architecture-migration-analysis.md**
  - 标题: 旧架构 vs 新架构对比分析
  - 推荐: pipeline-architecture/async-dual-pipeline
  - 置信度: 12%

- **asr_abstraction_interface_research.md**
  - 标题: 
  - 推荐: core-services/transcription-service
  - 置信度: 13%

- **audio_spectrum_classifier_investigation_report.md**
  - 标题: 
  - 推荐: audio-processing/separation
  - 置信度: 27%

- **cancellation-token-investigation.md**
  - 标题: CancellationToken 与暂停/恢复机制完整实现调查
  - 推荐: pipeline-architecture/preprocessing
  - 置信度: 26%

- **checkpoint_resume_complete_investigation_report.md**
  - 标题: 断点恢复机制完整调查报告
  - 推荐: pipeline-architecture/preprocessing
  - 置信度: 26%

- **complete-system-architecture-investigation-report.md**
  - 标题: 
  - 推荐: pipeline-architecture/async-dual-pipeline
  - 置信度: 25%

- **confidence-warning-backend-investigation.md**
  - 标题: 
  - 推荐: core-services/transcription-service
  - 置信度: 15%

- **confidence-warning-frontend-investigation.md**
  - 标题: 前端置信度警告系统实现调查报告
  - 推荐: core-services/transcription-service
  - 置信度: 13%

- **confidence_persistence_investigation_report.md**
  - 标题: 置信度数据持久化调查报告
  - 推荐: core-services/transcription-service
  - 置信度: 16%

- **demucs_integration_investigation.md**
  - 标题: 
  - 推荐: audio-processing/separation
  - 置信度: 32%

- **demucs_service_legacy_architecture_investigation.md**
  - 标题: 
  - 推荐: audio-processing/separation
  - 置信度: 42%

- **env_loading_test_report.md**
  - 标题: .env 配置加载功能测试报告
  - 推荐: core-services/model-management
  - 置信度: 5%

- **exit_system_flow_investigation_report.md**
  - 标题: 
  - 推荐: core-services/model-management
  - 置信度: 7%

- **ffmpeg_ffprobe_usage_report.md**
  - 标题: 
  - 推荐: core-services/model-management
  - 置信度: 5%

- **frontend-ux-analysis.md**
  - 标题: 前端用户体验与视觉交互调查报告
  - 推荐: frontend-components/editor
  - 置信度: 25%

- **frontend_confidence_highlighting_investigation.md**
  - 标题: 
  - 推荐: core-services/confidence-system
  - 置信度: 11%

- **frontend_tasks_interface_investigation_report.md**
  - 标题: 
  - 推荐: frontend-components/editor
  - 置信度: 11%

- **frontend_update_window_preparation_investigation_report.md**
  - 标题: 
  - 推荐: core-services/model-management
  - 置信度: 7%

- **garbage_sentence_filter_location_report.md**
  - 标题: 
  - 推荐: core-services/transcription-service
  - 置信度: 12%

- **gpu_coordinator_status_investigation.md**
  - 标题: 
  - 推荐: audio-processing/separation
  - 置信度: 11%

- **model_manager_v2_implementation_completeness_report.md**
  - 标题: 
  - 推荐: core-services/model-management
  - 置信度: 25%

- **multi-page-conflict-investigation.md**
  - 标题: 
  - 推荐: core-services/model-management
  - 置信度: 9%

- **parameter_classification_investigation.md**
  - 标题: video_to_srt_gpu 项目参数分类调查报告
  - 推荐: audio-processing/spectrum-analysis
  - 置信度: 40%

- **Phase2-现有代码调查检索摘要.md**
  - 标题: Phase 2 现有代码调查检索摘要
  - 推荐: audio-processing/separation
  - 置信度: 21%

- **phase3-alignment-investigation.md**
  - 标题: 
  - 推荐: core-services/transcription-service
  - 置信度: 13%

- **phase3-frontend-progress-implementation-status.md**
  - 标题: 
  - 推荐: frontend-components/editor
  - 置信度: 10%

- **phase3_implementation_verification_report.md**
  - 标题: 
  - 推荐: core-services/transcription-service
  - 置信度: 15%

- **phase3_transcription_service_components_status_report.md**
  - 标题: 
  - 推荐: core-services/transcription-service
  - 置信度: 16%

- **phase4-prerequisite-investigation.md**
  - 标题: 
  - 推荐: pipeline-architecture/async-dual-pipeline
  - 置信度: 10%

- **preprocessing_pipeline_integration_investigation.md**
  - 标题: 
  - 推荐: pipeline-architecture/preprocessing
  - 置信度: 30%

- **process_architecture_investigation_report.md**
  - 标题: 
  - 推荐: core-services/model-management
  - 置信度: 25%

- **readme-installation-verification.md**
  - 标题: 
  - 推荐: audio-processing/vad
  - 置信度: 5%

- **sensevoice-onnx-model-investigation.md**
  - 标题: 
  - 推荐: core-services/model-management
  - 置信度: 11%

- **sensevoice-onnx-parameters-investigation.md**
  - 标题: 
  - 推荐: core-services/model-management
  - 置信度: 9%

- **sensevoice-raw-subtitle-save-investigation.md**
  - 标题: 
  - 推荐: core-services/transcription-service
  - 置信度: 20%

- **sensevoice_confidence_calculation_report.md**
  - 标题: 
  - 推荐: core-services/confidence-system
  - 置信度: 15%

- **sensevoice_only_mode_display_confidence_investigation.md**
  - 标题: 
  - 推荐: core-services/confidence-system
  - 置信度: 15%

- **sensevoice_phase1_code_structure_report.md**
  - 标题: 
  - 推荐: core-services/model-management
  - 置信度: 25%

- **sentence-splitting-four-layers-architecture-investigation.md**
  - 标题: 
  - 推荐: core-services/transcription-service
  - 置信度: 11%

- **sentence-splitting-investigation.md**
  - 标题: 分句实现调查报告
  - 推荐: core-services/transcription-service
  - 置信度: 12%

- **shift_click_subtitle_move_investigation_report.md**
  - 标题: 
  - 推荐: core-services/transcription-service
  - 置信度: 6%

- **space_key_focus_issue_investigation_report.md**
  - 标题: 
  - 推荐: frontend-components/waveform
  - 置信度: 13%

- **spectral_triage_false_positive_misclassification_investigation.md**
  - 标题: 
  - 推荐: pipeline-architecture/preprocessing
  - 置信度: 25%

- **spectral_triage_separation_integration_report.md**
  - 标题: 
  - 推荐: audio-processing/separation
  - 置信度: 37%

- **sse-progress-investigation.md**
  - 标题: 
  - 推荐: core-services/transcription-service
  - 置信度: 11%

- **sse_and_subtitle_streaming_investigation_report.md**
  - 标题: 
  - 推荐: core-services/transcription-service
  - 置信度: 13%

- **startup_browser_mechanism_report.md**
  - 标题: 
  - 推荐: core-services/model-management
  - 置信度: 6%

- **subtitle-backend-integration-investigation.md**
  - 标题: 字幕切分功能后端集成调查报告
  - 推荐: frontend-components/editor
  - 置信度: 18%

- **subtitle_editing_infrastructure_investigation.md**
  - 标题: 
  - 推荐: frontend-components/editor
  - 置信度: 20%

- **subtitle_export_and_media_investigation.md**
  - 标题: 
  - 推荐: frontend-components/editor
  - 置信度: 18%

- **transcription_post_process_enhancement_investigation.md**
  - 标题: 
  - 推荐: core-services/transcription-service
  - 置信度: 4%

- **transcription_service_investigation_report.md**
  - 标题: 
  - 推荐: core-services/transcription-service
  - 置信度: 16%

- **transcription_service_methods_investigation.md**
  - 标题: 
  - 推荐: core-services/transcription-service
  - 置信度: 22%

- **triage-separation-progress-investigation.md**
  - 标题: 分诊检测和人声分离实现调查报告
  - 推荐: audio-processing/spectrum-analysis
  - 置信度: 40%

- **v3.2.2_hardlimit_investigation.md**
  - 标题: 
  - 推荐: pipeline-architecture/async-dual-pipeline
  - 置信度: 11%

- **vad_deprecated_code_investigation_report.md**
  - 标题: 
  - 推荐: audio-processing/vad
  - 置信度: 17%

- **video_to_srt_gpu_complete_pipeline_architecture_report.md**
  - 标题: video_to_srt_gpu 完整流水线架构调查报告
  - 推荐: core-services/transcription-service
  - 置信度: 15%

- **vue-component-complexity-analysis.md**
  - 标题: 
  - 推荐: frontend-components/editor
  - 置信度: 18%

- **whisper_service_investigation_report.md**
  - 标题: 
  - 推荐: core-services/model-management
  - 置信度: 21%

- **播放进度条跳回问题深入调查报告.md**
  - 标题: 播放视频时拖动进度条跳回原进度点问题深入调查报告
  - 推荐: frontend-components/editor
  - 置信度: 5%


---

**计划生成时间**: 2026-01-18 23:00:49