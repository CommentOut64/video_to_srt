function safeGetVersions() {
  return {
    node: process.versions?.node || null,
    chrome: process.versions?.chrome || null,
    electron: process.versions?.electron || null,
    v8: process.versions?.v8 || null,
  };
}

function collectVendorHints(rawValue, hints) {
  const lower = String(rawValue || "").toLowerCase();
  if (!lower) {
    return;
  }
  if (lower.includes("intel")) {
    hints.add("intel");
  }
  if (lower.includes("nvidia") || lower.includes("geforce") || lower.includes("rtx") || lower.includes("gtx")) {
    hints.add("nvidia");
  }
  if (lower.includes("amd") || lower.includes("radeon")) {
    hints.add("amd");
  }
}

function inferGpuTopology(gpuInfo) {
  const hints = new Set();
  const serialized = JSON.stringify(gpuInfo || {});
  collectVendorHints(serialized, hints);
  const vendors = Array.from(hints);

  let activeGpuType = "unknown";
  if (vendors.includes("intel") && vendors.some((vendor) => vendor === "nvidia" || vendor === "amd")) {
    activeGpuType = "hybrid";
  } else if (vendors.includes("intel")) {
    activeGpuType = "igpu";
  } else if (vendors.includes("nvidia") || vendors.includes("amd")) {
    activeGpuType = "dgpu";
  }

  return {
    gpuVendors: vendors,
    activeGpuType,
  };
}

function buildBaseRuntimeInfo(policy) {
  return {
    collectedAt: new Date().toISOString(),
    pid: process.pid,
    platform: process.platform,
    arch: process.arch,
    versions: safeGetVersions(),
    policy,
  };
}

async function collectRuntimeDiagnostics(app, policy) {
  const runtimeInfo = buildBaseRuntimeInfo(policy);
  runtimeInfo.hardwareAccelerationEnabled = typeof app.isHardwareAccelerationEnabled === "function"
    ? app.isHardwareAccelerationEnabled()
    : null;

  try {
    runtimeInfo.gpuFeatureStatus = app.getGPUFeatureStatus();
  } catch (error) {
    runtimeInfo.gpuFeatureStatus = null;
    runtimeInfo.gpuFeatureStatusError = error instanceof Error ? error.message : String(error);
  }

  try {
    runtimeInfo.gpuInfo = await app.getGPUInfo("basic");
    Object.assign(runtimeInfo, inferGpuTopology(runtimeInfo.gpuInfo));
  } catch (error) {
    runtimeInfo.gpuInfo = null;
    runtimeInfo.gpuInfoError = error instanceof Error ? error.message : String(error);
    runtimeInfo.activeGpuType = "unknown";
    runtimeInfo.gpuVendors = [];
  }

  return runtimeInfo;
}

module.exports = {
  buildBaseRuntimeInfo,
  collectRuntimeDiagnostics,
};
