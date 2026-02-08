export default {
  // 基础配置
  semi: false, // 不使用分号
  singleQuote: true, // 使用单引号
  trailingComma: 'es5', // ES5 兼容的尾随逗号
  tabWidth: 2, // 缩进宽度
  useTabs: false, // 使用空格而非 Tab
  printWidth: 100, // 每行最大字符数

  // Vue 特定配置
  vueIndentScriptAndStyle: false, // <script> 和 <style> 标签内不额外缩进

  // HTML 配置
  htmlWhitespaceSensitivity: 'ignore', // 忽略 HTML 空格敏感性

  // 其他配置
  endOfLine: 'lf', // 使用 LF 换行符
  arrowParens: 'always', // 箭头函数总是使用括号
  bracketSpacing: true, // 对象字面量的括号间添加空格
  quoteProps: 'as-needed', // 仅在需要时为对象属性添加引号
}
