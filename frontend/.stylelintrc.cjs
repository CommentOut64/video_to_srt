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
    // ========== 规则 1：CSS 函数白名单 ==========

    // 允许的 CSS 函数
    'function-no-unknown': [
      true,
      {
        ignoreFunctions: [
          'var',           // CSS Variables
          'calc',          // CSS calc
          'rgba',          // CSS rgba
          'rgb',           // CSS rgb
          'url',           // CSS url
          'linear-gradient',
          'radial-gradient',
          'v-bind'         // Vue v-bind()
        ]
      }
    ],

    // ========== 规则 4：禁止硬编码颜色 ==========

    // 禁止 Hex 颜色
    'color-no-hex': true,

    // 禁止颜色关键字（除了 transparent 和 currentColor）
    'color-named': [
      'never',
      {
        ignore: ['inside-function']
      }
    ],

    // ========== 规则 5：限制 !important ==========

    // 警告使用 !important（不完全禁止，但需要审查）
    'declaration-no-important': [
      true,
      {
        severity: 'warning'
      }
    ],

    // ========== 代码质量规则 ==========

    // 禁止未知的伪类选择器
    'selector-pseudo-class-no-unknown': [
      true,
      {
        ignorePseudoClasses: ['deep', 'global', 'slotted']
      }
    ],

    // 禁止未知的伪元素选择器
    'selector-pseudo-element-no-unknown': [
      true,
      {
        ignorePseudoElements: ['v-deep', 'v-global', 'v-slotted']
      }
    ],

    // 属性顺序（提高可读性）
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

    // 禁止空规则
    'block-no-empty': true,

    // 禁止重复的选择器
    'no-duplicate-selectors': true,

    // 禁止重复的属性
    'declaration-block-no-duplicate-properties': true,

    // 最大嵌套深度（防止过度嵌套）
    'max-nesting-depth': [
      3,
      {
        ignore: ['pseudo-classes']
      }
    ],

    // 选择器类名模式（推荐 kebab-case）
    // 允许: kebab-case (my-class), Element Plus BEM (el-*, is-*, has-*, 包含双下划线)
    'selector-class-pattern': [
      '^([a-z][a-z0-9]*(-[a-z0-9]+)*|el-[a-z0-9_-]+|is-[a-z0-9-]+|has-[a-z0-9-]+)$',
      {
        message: 'Expected class selector to be kebab-case (except third-party libraries like Element Plus)',
        resolveNestedSelectors: true
      }
    ],

    // 允许未知的 @ 规则（支持 Tailwind）
    'at-rule-no-unknown': [
      true,
      {
        ignoreAtRules: [
          'tailwind',
          'apply',
          'layer',
          'variants',
          'responsive',
          'screen'
        ]
      }
    ],

    // 自定义属性命名规范：必须以 --af- 开头
    // 允许第三方库的 CSS 变量（如 Element Plus 的 --el-*）
    'custom-property-pattern': [
      '^(af-[a-z]([a-z0-9-]+)?|el-.*)$',
      {
        message: 'CSS 变量必须以 --af- 开头并使用 kebab-case 命名（如 --af-bg-primary），第三方库变量除外'
      }
    ],

    // 允许空的 <style> 块（迁移过程中可能出现）
    'no-empty-source': null
  },

  overrides: [
    {
      // 对 Vue 文件的特殊规则
      files: ['**/*.vue'],
      rules: {
        // Vue 文件中允许使用 :deep()
        'selector-pseudo-class-no-unknown': [
          true,
          {
            ignorePseudoClasses: ['deep', 'global', 'slotted']
          }
        ]
      }
    },
    {
      // 全局样式文件可以使用 Hex 颜色（仅限定义 CSS Variables）
      files: ['**/styles/**/*.css'],
      rules: {
        'color-no-hex': null,
        'custom-property-pattern': null
      }
    }
  ],

  // 忽略特定文件
  ignoreFiles: [
    '**/node_modules/**',
    '**/dist/**',
    '**/legacy/**'
  ]
}
