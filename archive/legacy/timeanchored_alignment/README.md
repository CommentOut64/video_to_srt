# Timeanchored Legacy 归档目录

本目录归档 Phase 6 退役的旧 timeanchored 主链代码。

## 归档范围

- `backend/app/services/timeanchored_alignment/anchor_mount/`
- `backend/app/services/textflow/decision_ingress_adapter.py`
- `backend/app/pipelines/dual_pipeline/services/anchor_mount_graph_renderer.py`
- 对应旧链测试（包含 `test_boundary_evidence_builder.py` 等）

## 归档原因

- runtime 正式主链已切到 `PreparationBundle -> AlignmentDecoderService -> AlignmentPathAdapter -> Decision -> Output(sentence-first)`
- `AnchorMount / DecisionIngressPackage / should_fallback` 不再参与正式切分入口与正式降级路由
- `ow-<window_id>` 已仅保留为南向 compat 标识，旧链图渲染与旧 ingress DTO 不再属于活跃代码

## 说明

- 本目录只供历史参考与必要复盘，不参与默认运行时
- 若需单独研究旧实现，请以本目录镜像结构为准，不要把代码移回活跃主链
