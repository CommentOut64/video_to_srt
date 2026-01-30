/**
 * 任务阶段和状态配置常量
 * 颜色值从主题系统导入，确保与全局主题一致
 */
import { darkTheme } from '@/theme'

// 从主题系统提取颜色
const colors = {
  neutral: darkTheme.colors.text.secondary,      // #8b949e
  primary: darkTheme.colors.accent.primary,      // #58a6ff
  success: darkTheme.colors.accent.success,      // #3fb950
  warning: darkTheme.colors.accent.warning,      // #d29922
  danger: darkTheme.colors.accent.danger,        // #f85149
  purple: darkTheme.colors.accent.purple,        // #a371f7
  pink: darkTheme.colors.accent.pink,            // #db61a2
}

// 辅助函数：生成半透明背景色
const alphaBg = (rgb, alpha = 0.15) => {
  // 将 hex 转换为 rgba
  const hex = rgb.replace('#', '')
  const r = parseInt(hex.substring(0, 2), 16)
  const g = parseInt(hex.substring(2, 4), 16)
  const b = parseInt(hex.substring(4, 6), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

// 阶段配置
export const PHASE_CONFIG = {
  pending: { label: '等待中', color: colors.neutral, bgColor: alphaBg(colors.neutral) },
  extract: { label: '提取音频', color: colors.neutral, bgColor: alphaBg(colors.neutral) },
  split: { label: '分段中', color: colors.neutral, bgColor: alphaBg(colors.neutral) },
  transcribe: { label: '转录中', color: colors.primary, bgColor: alphaBg(colors.primary) },
  align: { label: '对齐中', color: colors.success, bgColor: alphaBg(colors.success) },
  translate: { label: '翻译中', color: colors.purple, bgColor: alphaBg(colors.purple) },
  proofread: { label: '校对中', color: colors.pink, bgColor: alphaBg(colors.pink) },
  srt: { label: '生成字幕', color: colors.success, bgColor: alphaBg(colors.success) },
  complete: { label: '已完成', color: colors.success, bgColor: alphaBg(colors.success) }
}

// 状态配置
export const STATUS_CONFIG = {
  created: { label: '已创建', color: colors.neutral, bgColor: alphaBg(colors.neutral) },
  queued: { label: '排队中', color: colors.neutral, bgColor: alphaBg(colors.neutral) },
  processing: { label: '处理中', color: colors.primary, bgColor: alphaBg(colors.primary) },
  pausing: { label: '正在暂停...', color: colors.warning, bgColor: alphaBg(colors.warning) },  // V3.1.0: 新增正在暂停状态
  paused: { label: '已暂停', color: colors.warning, bgColor: alphaBg(colors.warning) },
  canceling: { label: '正在取消...', color: colors.danger, bgColor: alphaBg(colors.danger) },  // V3.1.0: 新增正在取消状态
  force_canceled: { label: '已强制取消', color: colors.danger, bgColor: alphaBg(colors.danger) },  // V3.1.0: 新增强制取消状态
  finished: { label: '已完成', color: colors.success, bgColor: alphaBg(colors.success) },
  failed: { label: '失败', color: colors.danger, bgColor: alphaBg(colors.danger) },
  canceled: { label: '已取消', color: colors.neutral, bgColor: alphaBg(colors.neutral) },
  error: { label: '连接错误', color: colors.danger, bgColor: alphaBg(colors.danger) }
}

// 获取阶段显示信息
export function getPhaseInfo(phase) {
  return PHASE_CONFIG[phase] || PHASE_CONFIG.pending
}

// 获取状态显示信息
export function getStatusInfo(status) {
  return STATUS_CONFIG[status] || STATUS_CONFIG.created
}

// 格式化进度显示
export function formatProgress(percent) {
  if (typeof percent !== 'number' || isNaN(percent)) return '0.0'
  return percent.toFixed(1)
}
