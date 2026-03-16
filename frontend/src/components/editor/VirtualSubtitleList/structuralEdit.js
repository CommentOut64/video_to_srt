// V3.2.5+dev.20260316.03: 虚拟字幕列表结构性编辑工具

function readMergeSeparatorConfig() {
  if (typeof window === 'undefined' || !window.localStorage) {
    return null
  }

  try {
    const raw = window.localStorage.getItem('editor-merge-separator')
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

export function getMergeSeparator() {
  const config = readMergeSeparatorConfig()
  if (!config) return ' '

  const presetMap = {
    space: ' ',
    'comma-full': '，',
    'comma-half': ',',
    'period-full': '。',
    'period-half': '.',
  }

  if (config.type === 'custom') {
    return config.custom || ''
  }
  return presetMap[config.type] ?? ' '
}

export function normalizeProjectionWordsToCold(words = [], projectStore) {
  if (!Array.isArray(words) || words.length === 0 || !projectStore) {
    return []
  }

  return words
    .map((word) => {
      const start = Number(word?.start)
      const end = Number(word?.end)
      const startMs = Number.isFinite(start)
        ? Math.round(projectStore.toBaseTime(start) * 1000)
        : Number(word?.startMs)
      const endMs = Number.isFinite(end)
        ? Math.round(projectStore.toBaseTime(end) * 1000)
        : Number(word?.endMs)

      return {
        ...word,
        word: word?.word ?? word?.text ?? '',
        startMs: Number.isFinite(startMs) ? startMs : 0,
        endMs: Number.isFinite(endMs) ? endMs : Number.isFinite(startMs) ? startMs : 0,
      }
    })
    .filter((word) => Number.isFinite(word.startMs) && Number.isFinite(word.endMs))
}

export function splitSubtitleByCursor(subtitle, cursorPosition) {
  const text = String(subtitle?.text || '')
  const start = Number(subtitle?.start)
  const end = Number(subtitle?.end)
  const words = Array.isArray(subtitle?.words) ? subtitle.words : []

  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    return { success: false, error: '字幕时间无效，无法切分' }
  }

  const splitAtTextOffset = Number(cursorPosition)
  if (!Number.isInteger(splitAtTextOffset) || splitAtTextOffset <= 0 || splitAtTextOffset >= text.length) {
    return { success: false, error: '光标位置必须在文本中间' }
  }

  const leftText = text.slice(0, splitAtTextOffset)
  const rightText = text.slice(splitAtTextOffset)

  if (words.length > 0) {
    let charCount = 0
    let splitWordIndex = 0

    for (let index = 0; index < words.length; index += 1) {
      const word = words[index]
      charCount += String(word?.word ?? word?.text ?? '').length
      if (charCount >= splitAtTextOffset) {
        splitWordIndex = index + 1
        break
      }
    }

    splitWordIndex = Math.max(1, Math.min(words.length - 1, splitWordIndex))
    const leftWords = words.slice(0, splitWordIndex)
    const rightWords = words.slice(splitWordIndex)

    if (leftWords.length > 0 && rightWords.length > 0) {
      return {
        success: true,
        splitAtTextOffset,
        splitAtTime: Number(rightWords[0]?.start ?? leftWords[leftWords.length - 1]?.end ?? start),
        left: {
          start,
          end: Number(leftWords[leftWords.length - 1]?.end ?? start),
          text: leftText,
          words: leftWords,
        },
        right: {
          start: Number(rightWords[0]?.start ?? end),
          end,
          text: rightText,
          words: rightWords,
        },
      }
    }
  }

  const ratio = splitAtTextOffset / text.length
  const splitAtTime = start + (end - start) * ratio

  return {
    success: true,
    splitAtTextOffset,
    splitAtTime,
    left: {
      start,
      end: splitAtTime,
      text: leftText,
      words: [],
    },
    right: {
      start: splitAtTime,
      end,
      text: rightText,
      words: [],
    },
  }
}
