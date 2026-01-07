/**
 * 任务阶段和状态配置常量
 */

// 阶段配置
export const PHASE_CONFIG = {
  pending: { label: '等待中', color: '#8b949e', bgColor: 'rgba(139, 148, 158, 0.15)' },
  extract: { label: '提取音频', color: '#8b949e', bgColor: 'rgba(139, 148, 158, 0.15)' },
  split: { label: '分段中', color: '#8b949e', bgColor: 'rgba(139, 148, 158, 0.15)' },
  transcribe: { label: '转录中', color: '#58a6ff', bgColor: 'rgba(88, 166, 255, 0.15)' },
  align: { label: '对齐中', color: '#3fb950', bgColor: 'rgba(63, 185, 80, 0.15)' },
  translate: { label: '翻译中', color: '#a371f7', bgColor: 'rgba(163, 113, 247, 0.15)' },
  proofread: { label: '校对中', color: '#f778ba', bgColor: 'rgba(247, 120, 186, 0.15)' },
  srt: { label: '生成字幕', color: '#3fb950', bgColor: 'rgba(63, 185, 80, 0.15)' },
  complete: { label: '已完成', color: '#3fb950', bgColor: 'rgba(63, 185, 80, 0.15)' }
}

// 状态配置
export const STATUS_CONFIG = {
  created: { label: '已创建', color: '#8b949e', bgColor: 'rgba(139, 148, 158, 0.15)' },
  queued: { label: '排队中', color: '#8b949e', bgColor: 'rgba(139, 148, 158, 0.15)' },
  processing: { label: '处理中', color: '#58a6ff', bgColor: 'rgba(88, 166, 255, 0.15)' },
  pausing: { label: '正在暂停...', color: '#d29922', bgColor: 'rgba(210, 153, 34, 0.15)' },  // V3.1.0: 新增正在暂停状态
  paused: { label: '已暂停', color: '#d29922', bgColor: 'rgba(210, 153, 34, 0.15)' },
  canceling: { label: '正在取消...', color: '#f85149', bgColor: 'rgba(248, 81, 73, 0.15)' },  // V3.1.0: 新增正在取消状态
  force_canceled: { label: '已强制取消', color: '#f85149', bgColor: 'rgba(248, 81, 73, 0.15)' },  // V3.1.0: 新增强制取消状态
  finished: { label: '已完成', color: '#3fb950', bgColor: 'rgba(63, 185, 80, 0.15)' },
  failed: { label: '失败', color: '#f85149', bgColor: 'rgba(248, 81, 73, 0.15)' },
  canceled: { label: '已取消', color: '#8b949e', bgColor: 'rgba(139, 148, 158, 0.15)' },
  error: { label: '连接错误', color: '#f85149', bgColor: 'rgba(248, 81, 73, 0.15)' }
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
