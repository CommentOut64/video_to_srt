const PROTECTED_SHORTCUT_TARGET_SELECTOR = [
  'input',
  'textarea',
  'select',
  'button',
  '[contenteditable]',
  '[role="textbox"]',
].join(', ')

function normalizeTagName(target) {
  return String(target?.tagName ?? '').trim().toUpperCase()
}

function isDirectProtectedTarget(target) {
  const tagName = normalizeTagName(target)
  if (['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON'].includes(tagName)) {
    return true
  }
  return Boolean(target?.isContentEditable)
}

function isNestedProtectedTarget(target) {
  if (typeof target?.closest !== 'function') {
    return false
  }
  return Boolean(target.closest(PROTECTED_SHORTCUT_TARGET_SELECTOR))
}

export function shouldIgnoreStructuralShortcut(event) {
  if (!event) {
    return false
  }
  if (event.isComposing || event.keyCode === 229) {
    return true
  }
  const target = event.target
  return isDirectProtectedTarget(target) || isNestedProtectedTarget(target)
}
