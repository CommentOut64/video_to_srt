"""AnchorMount 输入校验与视图投影。"""

from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.contracts import AnchorMountInputView
from app.services.timeanchored_alignment.preparation.contracts import AlignmentPreparationPackage


class IngressValidator:
    """把 Preparation 包投影为 AnchorMount 只读视图。"""

    def validate(
        self,
        *,
        preparation: AlignmentPreparationPackage,
        language: str,
        policy_snapshot: object | None = None,
    ) -> AnchorMountInputView:
        token_units = tuple(preparation.slow_text.token_units)
        if not token_units:
            raise ValueError("AnchorMountAlignment 需要至少一个 prepared token unit")
        source_chunk_ids = set(preparation.source_chunk_ids)
        source_chunk_indices = set(preparation.source_chunk_indices)
        previous_end = -1
        for token_unit in token_units:
            if token_unit.char_start < previous_end:
                raise ValueError("PreparedTokenUnit 必须按 char 空间顺序排列")
            if token_unit.char_end > len(preparation.slow_text.window_text.text):
                raise ValueError("PreparedTokenUnit.char_end 越过 window_text 边界")
            if not token_unit.source_chunk_ids or not token_unit.source_chunk_indices:
                raise ValueError("PreparedTokenUnit 必须保留 source chunk provenance")
            if not set(token_unit.source_chunk_ids).issubset(source_chunk_ids):
                raise ValueError(
                    "PreparedTokenUnit.source_chunk_ids 必须属于 window source_chunk_ids"
                )
            if not set(token_unit.source_chunk_indices).issubset(source_chunk_indices):
                raise ValueError(
                    "PreparedTokenUnit.source_chunk_indices 必须属于 window source_chunk_indices"
                )
            previous_end = token_unit.char_end
        text_length = len(preparation.slow_text.window_text.text)
        for evidence in preparation.slow_text.punctuation_evidences:
            if evidence.source_char_index >= text_length:
                raise ValueError("PunctuationEvidence.source_char_index 越界")
        previous_hook_start = -1.0
        previous_hook_end = -1.0
        fast_hooks = tuple(preparation.fast_hooks)
        for hook in fast_hooks:
            if hook.start < previous_hook_start or hook.end < previous_hook_end:
                raise ValueError("FastHook 必须按时间单调排列")
            if hook.source_chunk_id not in source_chunk_ids:
                raise ValueError("FastHook.source_chunk_id 必须属于 window source_chunk_ids")
            if hook.source_chunk_index not in source_chunk_indices:
                raise ValueError("FastHook.source_chunk_index 必须属于 window source_chunk_indices")
            previous_hook_start = float(hook.start)
            previous_hook_end = float(hook.end)
        return AnchorMountInputView(
            window_id=preparation.window_id,
            owner_chunk_id=preparation.owner_chunk_id,
            owner_chunk_index=preparation.owner_chunk_index,
            source_chunk_ids=preparation.source_chunk_ids,
            source_chunk_indices=preparation.source_chunk_indices,
            language=str(language or preparation.compat.text_truth.language or "auto"),
            token_units=token_units,
            window_text=preparation.slow_text.window_text,
            punctuation_evidences=tuple(preparation.slow_text.punctuation_evidences),
            fast_hooks=fast_hooks,
            pronunciation_hints=tuple(preparation.slow_text.pronunciation_hints),
            policy_snapshot=policy_snapshot,
            coverage=preparation.coverage,
        )
