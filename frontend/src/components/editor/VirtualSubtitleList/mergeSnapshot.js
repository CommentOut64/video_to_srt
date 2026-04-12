export function resolveMergeSideSnapshot(localId, entity, mergePayload) {
  if (!entity) {
    return null
  }

  const fallbackSnapshot = {
    text: entity.text,
    startMs: entity.startMs,
    endMs: entity.endMs,
  }

  if (
    !mergePayload
    || mergePayload.localId !== localId
    || mergePayload.hasDraftOverride !== true
    || typeof mergePayload.draftText !== 'string'
  ) {
    return fallbackSnapshot
  }

  // 合并时的草稿文本需要与普通 blur 提交保持同一口径：
  // 仅在真实编辑态下折叠进结构命令，且空白草稿回退到原文本。
  const normalizedDraftText = mergePayload.draftText.trim()
  return {
    text: normalizedDraftText || entity.text,
    startMs: entity.startMs,
    endMs: entity.endMs,
  }
}
