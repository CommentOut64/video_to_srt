const VALID_GPU_MODES = new Set(["auto", "prefer_dgpu", "prefer_igpu", "safe"]);
const LEGACY_GPU_MODE_ALIASES = {
  prefer_hardware: "prefer_dgpu",
  off: "safe",
};

function normalizeGpuMode(rawMode) {
  const normalized = String(rawMode || "").trim().toLowerCase();
  const aliased = LEGACY_GPU_MODE_ALIASES[normalized] || normalized;
  if (VALID_GPU_MODES.has(aliased)) {
    return aliased;
  }
  return "auto";
}

function normalizeMediaProfile(rawProfile) {
  const normalized = String(rawProfile || "").trim().toLowerCase();
  if (["browser_compat", "electron_native", "lite_safe"].includes(normalized)) {
    return normalized;
  }
  return "electron_native";
}

function resolveGpuPolicy(env = process.env) {
  const mediaProfile = normalizeMediaProfile(env.ANCHORFLUX_MEDIA_PROFILE);
  const requestedMode = normalizeGpuMode(env.ANCHORFLUX_GPU_MODE);
  const effectiveMode = requestedMode;
  const chromiumSwitches = [];
  const notes = [];
  let disableHardwareAcceleration = false;

  if (effectiveMode === "prefer_dgpu") {
    chromiumSwitches.push(["force_high_performance_gpu"]);
    notes.push("已请求 Chromium 优先使用高性能 GPU（best-effort）");
  } else if (effectiveMode === "prefer_igpu") {
    chromiumSwitches.push(["force_low_power_gpu"]);
    notes.push("已请求 Chromium 优先使用低功耗 GPU（best-effort）");
  } else if (effectiveMode === "safe") {
    disableHardwareAcceleration = true;
    notes.push("safe 模式已关闭硬件加速");
  }

  if (mediaProfile === "lite_safe") {
    notes.push("当前媒体 profile 为 lite_safe，优先安全预览链路");
  }

  return {
    requestedMode,
    effectiveMode,
    mediaProfile,
    disableHardwareAcceleration,
    chromiumSwitches,
    notes,
    allowSourcePlayback: mediaProfile !== "lite_safe",
  };
}

function applyGpuPolicy(app, policy) {
  const appliedSwitches = [];
  for (const switchArgs of policy.chromiumSwitches) {
    if (!Array.isArray(switchArgs) || switchArgs.length === 0) {
      continue;
    }
    const [name, value] = switchArgs;
    if (value === undefined) {
      app.commandLine.appendSwitch(name);
    } else {
      app.commandLine.appendSwitch(name, value);
    }
    appliedSwitches.push(name);
  }

  if (policy.disableHardwareAcceleration) {
    app.disableHardwareAcceleration();
  }

  return {
    ...policy,
    appliedSwitches,
  };
}

module.exports = {
  resolveGpuPolicy,
  applyGpuPolicy,
  normalizeGpuMode,
  normalizeMediaProfile,
};
