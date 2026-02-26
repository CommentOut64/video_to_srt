/**
 * 前端 flavor 单一真值入口。
 *
 * 约束：
 * - 组件层禁止直接读取 import.meta.env 的 flavor 标志。
 * - UI 能力门控仅做体验收敛，后端接口仍是最终安全边界。
 */

const runtimeFlavor = String(import.meta.env.VITE_APP_FLAVOR || '').toLowerCase()
const runtimeLiteFlag = String(import.meta.env.VITE_LITE_MODE || '').toLowerCase() === 'true'

// 编译时常量优先，运行时变量兜底
export const FLAVOR = (__APP_FLAVOR__ || runtimeFlavor || 'full').toLowerCase()
export const IS_LITE = Boolean(__IS_LITE__) || runtimeLiteFlag || FLAVOR === 'lite'

export const CAPABILITIES = Object.freeze({
  // 转录相关（Lite 收敛）
  canTranscribe: !IS_LITE,
  canSeparateVocal: !IS_LITE,
  canSpectralTriage: !IS_LITE,
  canManageModels: !IS_LITE,
  // 编辑相关
  canImportSubtitle: true,
  canExportSubtitle: true,
  canEditSubtitle: true,
  canPreviewMedia: true,
})

export const ROUTE_VISIBILITY = Object.freeze({
  taskList: true, // Lite 也使用任务列表
  taskCreate: !IS_LITE,
  importPage: true,
  editor: true,
})

