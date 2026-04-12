export const STRUCTURAL_CONTEXT_MENU_SELECTION_KEYS = new Set([
  'split',
  'merge-prev',
  'merge-next',
])

export function resolveContextMenuCloseEditingAction({
  pendingBlurWhileContextMenuOpen = false,
  closeMeta = null,
} = {}) {
  if (!pendingBlurWhileContextMenuOpen) {
    return 'noop'
  }

  if (
    closeMeta?.reason === 'selection'
    && STRUCTURAL_CONTEXT_MENU_SELECTION_KEYS.has(closeMeta?.key)
  ) {
    return 'discard_pending_blur'
  }

  return 'commit_pending_blur'
}
