/**
 * 安全自动修复配置
 *
 * 只包含100%安全的可自动修复规则：
 * - 属性顺序
 * - 空格/缩进
 * - 引号风格
 * - 数字格式
 *
 * 使用方法：
 * npx stylelint src/path/to/file.vue --config .stylelintrc.safe-fix.cjs --fix
 */
module.exports = {
  extends: [
    'stylelint-config-standard',
    'stylelint-config-recommended-vue'
  ],

  plugins: [
    'stylelint-order'
  ],

  customSyntax: 'postcss-html',

  rules: {
    // ========== 只启用100%安全的自动修复规则 ==========

    // 属性顺序（可自动修复）
    'order/properties-order': [
      'position',
      'top',
      'right',
      'bottom',
      'left',
      'z-index',
      'display',
      'flex',
      'flex-direction',
      'flex-wrap',
      'justify-content',
      'align-items',
      'gap',
      'width',
      'height',
      'padding',
      'margin',
      'background',
      'border',
      'border-radius',
      'color',
      'font-size',
      'font-weight',
      'transition',
      'animation'
    ],

    // 移除零值的单位（可自动修复）
    'length-zero-no-unit': true,

    // 颜色函数使用现代格式（可自动修复）
    'color-function-notation': 'modern',

    // Alpha 值使用百分比（可自动修复）
    'alpha-value-notation': 'percentage',

    // 禁止重复的属性（可自动移除）
    'declaration-block-no-duplicate-properties': true,

    // ========== 禁用所有语义检查规则 ==========
    'color-no-hex': null,
    'color-named': null,
    'declaration-no-important': null,
    'custom-property-pattern': null,
    'selector-class-pattern': null,
    'max-nesting-depth': null,
    'block-no-empty': null,
    'no-duplicate-selectors': null
  },

  overrides: [
    {
      files: ['**/*.vue'],
      rules: {
        'selector-pseudo-class-no-unknown': [
          true,
          {
            ignorePseudoClasses: ['deep', 'global', 'slotted']
          }
        ]
      }
    }
  ]
}
