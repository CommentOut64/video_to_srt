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

function mergeSnapshot(snapshot = {}) {
  return {
    ...DEFAULT_CAPABILITY_SNAPSHOT,
    ...snapshot,
    capabilities: {
      ...DEFAULT_CAPABILITY_SNAPSHOT.capabilities,
      ...(snapshot.capabilities || {}),
    },
    routeVisibility: {
      ...DEFAULT_CAPABILITY_SNAPSHOT.routeVisibility,
      ...(snapshot.routeVisibility || {}),
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
