import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

function getCssRuleBlock(source, selector) {
  const escapedSelector = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const match = source.match(new RegExp(`${escapedSelector}\\s*\\{([\\s\\S]*?)\\n\\}`, 'm'))
  return match?.[1] ?? ''
}

describe('SubtitleRow style regression', () => {
  it('展示态高亮样式不应改变多行文本的布局尺寸', () => {
    const source = readFileSync(
      resolve(import.meta.dirname, './SubtitleRow.vue'),
      'utf8'
    )

    const textInputRule = getCssRuleBlock(source, '.text-input')
    const warningRule = getCssRuleBlock(source, '.text-display :deep(.word-warning)')
    const criticalRule = getCssRuleBlock(source, '.text-display :deep(.word-critical)')
    const matchRule = getCssRuleBlock(source, '.text-display :deep(.match-highlight)')

    expect(textInputRule).toContain('display: block;')
    expect(textInputRule).toContain('white-space: pre-wrap;')
    expect(textInputRule).toContain('overflow-wrap: break-word;')

    expect(warningRule).not.toContain('border-bottom:')
    expect(warningRule).not.toContain('padding:')

    expect(criticalRule).not.toContain('border-bottom:')
    expect(criticalRule).not.toContain('padding:')
    expect(criticalRule).not.toContain('font-weight:')

    expect(matchRule).not.toContain('padding:')
    expect(matchRule).not.toContain('font-weight:')
  })

  it('多行编辑自动扩展应包含 1px 取整补偿，避免编辑态偶发比展示态更矮', () => {
    const source = readFileSync(
      resolve(import.meta.dirname, './SubtitleRow.vue'),
      'utf8'
    )

    expect(source).toContain('textarea.scrollHeight + 1')
  })
})
