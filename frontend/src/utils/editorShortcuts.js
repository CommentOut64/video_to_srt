/**
 * 编辑器快捷键配置工具
 * 目标：支持“任意组合键录制 + 统一归一化 + 冲突检测 + 事件匹配”。
 */

export const EDITOR_SHORTCUT_FIELDS = Object.freeze([
  { key: 'togglePlay', label: '播放/暂停' },
  { key: 'stepBackward', label: '上一帧' },
  { key: 'stepForward', label: '下一帧' },
  { key: 'seekBackward', label: '快退 5 秒' },
  { key: 'seekForward', label: '快进 5 秒' },
  { key: 'seekToStart', label: '跳到开头' },
  { key: 'seekToEnd', label: '跳到结尾' },
  { key: 'save', label: '保存' },
  { key: 'undo', label: '撤销' },
  { key: 'redo', label: '重做' },
])

export const DEFAULT_EDITOR_SHORTCUT_CONFIG = Object.freeze({
  togglePlay: 'Space',
  stepBackward: 'ArrowLeft',
  stepForward: 'ArrowRight',
  seekBackward: 'Shift+ArrowLeft',
  seekForward: 'Shift+ArrowRight',
  seekToStart: 'Home',
  seekToEnd: 'End',
  save: 'Ctrl+KeyS',
  undo: 'Ctrl+KeyZ',
  redo: 'Ctrl+Shift+KeyZ',
})

const ACTION_LABEL_MAP = Object.freeze(
  EDITOR_SHORTCUT_FIELDS.reduce((acc, item) => {
    acc[item.key] = item.label
    return acc
  }, {})
)

const BROWSER_RESERVED_COMBOS = new Set([
  'Ctrl+KeyW',
  'Ctrl+KeyN',
  'Ctrl+KeyT',
  'Ctrl+Shift+KeyT',
  'Ctrl+KeyR',
  'Ctrl+F5',
  'F5',
  'Ctrl+KeyL',
  'Ctrl+KeyP',
  'Ctrl+KeyJ',
  'Ctrl+KeyK',
  'Alt+ArrowLeft',
  'Alt+ArrowRight',
])

const MODIFIER_ORDER = Object.freeze(['Ctrl', 'Shift', 'Alt'])
const MODIFIER_ALIAS_MAP = Object.freeze({
  ctrl: 'Ctrl',
  control: 'Ctrl',
  cmd: 'Ctrl',
  command: 'Ctrl',
  meta: 'Ctrl',
  shift: 'Shift',
  alt: 'Alt',
  option: 'Alt',
})

const KEY_ALIAS_MAP = Object.freeze({
  space: 'Space',
  spacebar: 'Space',
  left: 'ArrowLeft',
  right: 'ArrowRight',
  up: 'ArrowUp',
  down: 'ArrowDown',
  arrowleft: 'ArrowLeft',
  arrowright: 'ArrowRight',
  arrowup: 'ArrowUp',
  arrowdown: 'ArrowDown',
  home: 'Home',
  end: 'End',
  pageup: 'PageUp',
  pagedown: 'PageDown',
  enter: 'Enter',
  tab: 'Tab',
  backspace: 'Backspace',
  delete: 'Delete',
  insert: 'Insert',
  escape: 'Escape',
  esc: 'Escape',
  minus: 'Minus',
  equal: 'Equal',
  comma: 'Comma',
  period: 'Period',
  slash: 'Slash',
  backslash: 'Backslash',
  semicolon: 'Semicolon',
  quote: 'Quote',
  backquote: 'Backquote',
  bracketleft: 'BracketLeft',
  bracketright: 'BracketRight',
  '[': 'BracketLeft',
  ']': 'BracketRight',
  '-': 'Minus',
  '=': 'Equal',
  ',': 'Comma',
  '.': 'Period',
  '/': 'Slash',
  '\\': 'Backslash',
  ';': 'Semicolon',
  "'": 'Quote',
  '`': 'Backquote',
})

const DISPLAY_KEY_LABEL_MAP = Object.freeze({
  Space: 'Space',
  ArrowLeft: 'ArrowLeft',
  ArrowRight: 'ArrowRight',
  ArrowUp: 'ArrowUp',
  ArrowDown: 'ArrowDown',
  Home: 'Home',
  End: 'End',
  PageUp: 'PageUp',
  PageDown: 'PageDown',
  Enter: 'Enter',
  Tab: 'Tab',
  Backspace: 'Backspace',
  Delete: 'Delete',
  Insert: 'Insert',
  Escape: 'Escape',
  Minus: '-',
  Equal: '=',
  Comma: ',',
  Period: '.',
  Slash: '/',
  Backslash: '\\',
  Semicolon: ';',
  Quote: "'",
  Backquote: '`',
  BracketLeft: '[',
  BracketRight: ']',
})

function normalizeModifierToken(token) {
  if (typeof token !== 'string') return null
  return MODIFIER_ALIAS_MAP[token.trim().toLowerCase()] || null
}

function normalizeKeyToken(token) {
  if (typeof token !== 'string') return null
  const trimmed = token.trim()
  if (!trimmed) return null
  const lower = trimmed.toLowerCase()

  if (KEY_ALIAS_MAP[lower]) {
    return KEY_ALIAS_MAP[lower]
  }

  if (/^[a-z]$/i.test(trimmed)) {
    return `Key${trimmed.toUpperCase()}`
  }

  if (/^\d$/.test(trimmed)) {
    return `Digit${trimmed}`
  }

  if (/^key[a-z]$/i.test(trimmed)) {
    return `Key${trimmed.slice(3).toUpperCase()}`
  }

  if (/^digit\d$/i.test(trimmed)) {
    return `Digit${trimmed.slice(5)}`
  }

  if (/^f([1-9]|1[0-2])$/i.test(trimmed)) {
    return trimmed.toUpperCase()
  }

  if (/^(ArrowLeft|ArrowRight|ArrowUp|ArrowDown|Home|End|PageUp|PageDown|Enter|Tab|Backspace|Delete|Insert|Escape|Space|Minus|Equal|Comma|Period|Slash|Backslash|Semicolon|Quote|Backquote|BracketLeft|BracketRight)$/i.test(trimmed)) {
    return trimmed.charAt(0).toUpperCase() + trimmed.slice(1)
  }

  return null
}

function formatComboFromParts(parts) {
  const ordered = []
  for (const modifier of MODIFIER_ORDER) {
    if (parts[modifier]) {
      ordered.push(modifier)
    }
  }
  ordered.push(parts.key)
  return ordered.join('+')
}

function parseShortcutCombo(rawCombo) {
  if (typeof rawCombo !== 'string') {
    return null
  }
  const tokens = rawCombo.split('+').map((token) => token.trim()).filter(Boolean)
  if (tokens.length === 0) {
    return null
  }

  const parts = { Ctrl: false, Shift: false, Alt: false, key: null }
  for (const token of tokens) {
    const modifier = normalizeModifierToken(token)
    if (modifier) {
      parts[modifier] = true
      continue
    }
    const key = normalizeKeyToken(token)
    if (!key) {
      return null
    }
    if (parts.key) {
      return null
    }
    parts.key = key
  }

  if (!parts.key) {
    return null
  }
  return formatComboFromParts(parts)
}

function toDisplayKeyLabel(keyToken) {
  if (DISPLAY_KEY_LABEL_MAP[keyToken]) {
    return DISPLAY_KEY_LABEL_MAP[keyToken]
  }
  if (/^Key[A-Z]$/.test(keyToken)) {
    return keyToken.slice(3)
  }
  if (/^Digit\d$/.test(keyToken)) {
    return keyToken.slice(5)
  }
  if (/^F([1-9]|1[0-2])$/.test(keyToken)) {
    return keyToken
  }
  return keyToken
}

function getEventKeyToken(event) {
  const code = String(event?.code || '').trim()
  if (code && !/^Control(Left|Right)$/.test(code) && !/^Shift(Left|Right)$/.test(code) && !/^Alt(Left|Right)$/.test(code) && code !== 'MetaLeft' && code !== 'MetaRight') {
    const normalizedCode = normalizeKeyToken(code)
    if (normalizedCode) {
      return normalizedCode
    }
  }

  const key = String(event?.key || '').trim()
  return normalizeKeyToken(key)
}

export function buildShortcutComboFromKeyboardEvent(event) {
  const keyToken = getEventKeyToken(event)
  if (!keyToken) {
    return null
  }
  const parts = {
    Ctrl: Boolean(event?.ctrlKey || event?.metaKey),
    Shift: Boolean(event?.shiftKey),
    Alt: Boolean(event?.altKey),
    key: keyToken,
  }
  return formatComboFromParts(parts)
}

export function normalizeEditorShortcutConfig(rawConfig = {}) {
  const normalized = {}
  for (const field of EDITOR_SHORTCUT_FIELDS) {
    const fallbackCombo = parseShortcutCombo(DEFAULT_EDITOR_SHORTCUT_CONFIG[field.key])
    const currentCombo = parseShortcutCombo(rawConfig?.[field.key])
    normalized[field.key] = currentCombo || fallbackCombo
  }
  return normalized
}

export function getEditorShortcutActionLabel(actionKey) {
  return ACTION_LABEL_MAP[actionKey] || actionKey
}

export function getEditorShortcutComboLabel(combo) {
  const parsed = parseShortcutCombo(combo)
  if (!parsed) {
    return '未设置'
  }
  const tokens = parsed.split('+')
  const keyToken = tokens[tokens.length - 1]
  const modifierTokens = tokens.slice(0, -1)
  const labelTokens = [...modifierTokens, toDisplayKeyLabel(keyToken)]
  return labelTokens.join(' + ')
}

export function detectEditorShortcutConflicts(rawConfig = {}) {
  const normalized = normalizeEditorShortcutConfig(rawConfig)
  const comboToActionMap = new Map()
  const conflicts = []

  for (const field of EDITOR_SHORTCUT_FIELDS) {
    const actionKey = field.key
    const combo = normalized[actionKey]
    const existingAction = comboToActionMap.get(combo)
    if (existingAction) {
      conflicts.push({
        combo,
        firstAction: existingAction,
        secondAction: actionKey,
      })
      continue
    }
    comboToActionMap.set(combo, actionKey)
  }

  return conflicts
}

export function detectBrowserReservedShortcutConflicts(rawConfig = {}) {
  const normalized = normalizeEditorShortcutConfig(rawConfig)
  const conflicts = []
  for (const field of EDITOR_SHORTCUT_FIELDS) {
    const combo = normalized[field.key]
    if (!BROWSER_RESERVED_COMBOS.has(combo)) {
      continue
    }
    conflicts.push({
      combo,
      action: field.key,
    })
  }
  return conflicts
}

export function isEditorShortcutMatch(event, combo) {
  const normalizedCombo = parseShortcutCombo(combo)
  if (!normalizedCombo) return false
  const eventCombo = buildShortcutComboFromKeyboardEvent(event)
  if (!eventCombo) return false
  return normalizedCombo === eventCombo
}

export function resolveEditorShortcutAction(event, rawConfig = {}) {
  const normalized = normalizeEditorShortcutConfig(rawConfig)
  for (const field of EDITOR_SHORTCUT_FIELDS) {
    const actionKey = field.key
    if (isEditorShortcutMatch(event, normalized[actionKey])) {
      return actionKey
    }
  }
  return null
}
