/**
 * 进度条配置常量
 * V3.2.0: 统一进度条设计
 */

// 进度分配（与后端保持一致）
export const PROGRESS_ALLOCATION = {
  preprocessing: { start: 0, end: 20 }, // 前处理 20%
  transcription: { start: 20, end: 75 }, // 转录 55%
  refinement: { start: 75, end: 95 }, // 精修/翻译 20%（暂未实现，自动跳过）
  export: { start: 95, end: 100 }, // 导出 5%
};

// 节点配置（阈值与进度分配对应）
export const DEFAULT_NODES = [
  { id: "preprocess", threshold: 20, label: "前处理" },
  { id: "transcribe", threshold: 75, label: "转录" },
  { id: "refine", threshold: 95, label: "精修" },
];

// V3.2.0+dev.20260122.02: 尺寸预设（lg 轨道厚度 = 4 × 圆点半径）
export const SIZE_PRESETS = {
  sm: { trackWidth: 90, trackHeight: 5, dotSize: 3 },
  md: { trackWidth: 110, trackHeight: 6, dotSize: 4 },
  lg: { trackWidth: 220, trackHeight: 12, dotSize: 6 },
};

// V3.2.0+dev.20260122.01: 深色主题颜色
export const COLORS = {
  trackBg: "#21262d", // 轨道背景（调暗以增加对比度）
  trackBorder: "rgba(48, 54, 61, 0.6)", // 轨道边框
  fastStream: "#58a6ff", // 快流（蓝）
  slowStream: "#3fb950", // 慢流（绿）
  error: "rgba(248, 81, 73, 0.4)", // 异常填充
  nodeInactive: "#6e7681", // 未激活节点
  nodeActive: "#ffffff", // 已激活节点
  nodeCutter: "#161b22", // 节点外圈（与页面背景一致）
};
