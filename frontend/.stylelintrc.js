export default {
  extends: ['stylelint-config-standard'],
  rules: {
    // 禁止硬编码颜色值
    'color-no-hex': true,
    'color-named': 'never',

    // 禁止使用 !important
    'declaration-no-important': true,

    // 允许未知的 @ 规则（支持 Tailwind 的 @tailwind, @apply 等）
    'at-rule-no-unknown': [
      true,
      {
        ignoreAtRules: [
          'tailwind',
          'apply',
          'layer',
          'variants',
          'responsive',
          'screen',
        ],
      },
    ],

    // 允许 Vue 的 :deep(), :slotted(), :global() 等伪类
    'selector-pseudo-class-no-unknown': [
      true,
      {
        ignorePseudoClasses: ['deep', 'slotted', 'global'],
      },
    ],

    // 允许 Vue 的 v-bind() 函数
    'function-no-unknown': [
      true,
      {
        ignoreFunctions: ['v-bind'],
      },
    ],

    // 允许空的 <style> 块（迁移过程中可能出现）
    'no-empty-source': null,

    // 自定义属性命名规范：必须以 --af- 开头（AnchorFlux 前缀）
    'custom-property-pattern': [
      '^af-[a-z]([a-z0-9-]+)?$',
      {
        message: 'CSS 变量必须以 --af- 开头并使用 kebab-case 命名（如 --af-bg-primary）',
      },
    ],

    // 警告级别：允许但发出警告
    'color-hex-length': 'short', // 如果使用 hex，必须用短格式
  },

  // 忽略特定文件
  ignoreFiles: [
    '**/node_modules/**',
    '**/dist/**',
    '**/legacy/**', // 忽略旧的 SCSS 文件
  ],
}
