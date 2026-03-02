import { CAPABILITIES, FLAVOR, IS_LITE, ROUTE_VISIBILITY } from '@/config/flavor'

const DEFAULT_CAPABILITY_SNAPSHOT = Object.freeze({
  profile: IS_LITE ? 'lite' : 'full',
  flavor: FLAVOR,
  isLite: IS_LITE,
  version: 'phase0.5-placeholder',
  capabilities: CAPABILITIES,
  routeVisibility: ROUTE_VISIBILITY,
  // Full/Lite 能力矩阵未来字段（本期仅留位，不启用）
  canShowEditorProgress: true,
  canUseTaskMonitor: true,
  taskSwitcherMode: 'legacy',
})

const ALLOWED_TOP_KEYS = new Set([
  'profile',
  'flavor',
  'isLite',
  'version',
  'capabilities',
  'routeVisibility',
  // 技术债字段：暂保留顶层兼容，后续再归并到 capabilities
  'canShowEditorProgress',
  'canUseTaskMonitor',
  'taskSwitcherMode',
])

function mergeSnapshot(snapshot = {}) {
  const normalizedSnapshot = (
    snapshot
    && typeof snapshot === 'object'
    && !Array.isArray(snapshot)
  ) ? snapshot : {}
  const filtered = {}
  for (const key of Object.keys(normalizedSnapshot)) {
    if (ALLOWED_TOP_KEYS.has(key)) {
      filtered[key] = normalizedSnapshot[key]
    }
  }

  return {
    ...DEFAULT_CAPABILITY_SNAPSHOT,
    ...filtered,
    capabilities: {
      ...DEFAULT_CAPABILITY_SNAPSHOT.capabilities,
      ...(filtered.capabilities || {}),
    },
    routeVisibility: {
      ...DEFAULT_CAPABILITY_SNAPSHOT.routeVisibility,
      ...(filtered.routeVisibility || {}),
    },
  }
}

export function buildDefaultCapabilitySnapshot(overrides = {}) {
  return mergeSnapshot(overrides)
}

export function selectCapabilitySnapshot(snapshot = null) {
  return mergeSnapshot(snapshot || {})
}

export function selectCapabilities(snapshot = null) {
  return selectCapabilitySnapshot(snapshot).capabilities
}

export function selectRouteVisibility(snapshot = null) {
  return selectCapabilitySnapshot(snapshot).routeVisibility
}

export function selectFlavor(snapshot = null) {
  return selectCapabilitySnapshot(snapshot).flavor
}
