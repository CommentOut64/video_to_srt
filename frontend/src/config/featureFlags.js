// V3.2.5+dev.20260315.01: Feature Flags
export const FEATURE_FLAGS = {
  // Phase 4: 新编辑器内核开关
  USE_EDITOR_V2: true, // 默认启用新内核
}

export function isFeatureEnabled(flag) {
  return FEATURE_FLAGS[flag] === true
}
