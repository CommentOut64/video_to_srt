const fs = require("node:fs");
const path = require("node:path");

function parseBooleanEnvFlag(rawValue, fallbackValue) {
  if (rawValue === undefined || rawValue === null || rawValue === "") {
    return fallbackValue;
  }

  const normalized = String(rawValue).trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(normalized)) {
    return true;
  }
  if (["0", "false", "no", "off"].includes(normalized)) {
    return false;
  }
  return fallbackValue;
}

function parsePositiveInteger(rawValue, fallbackValue) {
  const parsed = Number(rawValue);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return fallbackValue;
  }
  return Math.round(parsed);
}

function isDevBuildVersion(buildVersion) {
  const normalized = String(buildVersion || "").trim().toLowerCase();
  return normalized.includes("-dev");
}

function parseEnvFileContent(rawContent) {
  const result = {};
  for (const rawLine of String(rawContent || "").split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#") || !line.includes("=")) {
      continue;
    }
    const separatorIndex = line.indexOf("=");
    const key = line.slice(0, separatorIndex).trim();
    const value = line.slice(separatorIndex + 1).trim();
    if (key) {
      result[key] = value;
    }
  }
  return result;
}

function resolveShellEnvFileCandidates(options = {}) {
  const candidates = [];
  const cwd = options.cwd || process.cwd();
  const execPath = options.execPath || process.execPath;
  const resourcesPath = options.resourcesPath || process.resourcesPath;

  const pushCandidate = (candidate) => {
    if (!candidate) {
      return;
    }
    const normalized = path.resolve(candidate);
    if (!candidates.includes(normalized)) {
      candidates.push(normalized);
    }
  };

  pushCandidate(path.join(cwd, "..", "..", ".env"));
  pushCandidate(path.join(cwd, ".env"));

  if (execPath) {
    pushCandidate(path.join(path.dirname(execPath), "..", "..", ".env"));
    pushCandidate(path.join(path.dirname(execPath), ".env"));
  }

  if (resourcesPath) {
    pushCandidate(path.join(resourcesPath, "..", "..", ".env"));
    pushCandidate(path.join(resourcesPath, "..", "..", "..", ".env"));
  }

  return candidates;
}

function loadShellEnvFallback(options = {}) {
  const explicitCandidates = Array.isArray(options.envFileCandidates)
    ? options.envFileCandidates
    : resolveShellEnvFileCandidates(options);

  for (const candidate of explicitCandidates) {
    try {
      if (!fs.existsSync(candidate)) {
        continue;
      }
      return parseEnvFileContent(fs.readFileSync(candidate, "utf8"));
    } catch (_) {
      // Why: 运行时配置读取失败时应静默回退到环境变量，不阻断主进程启动。
    }
  }

  return {};
}

function resolveShellRuntimeConfig(options = {}) {
  const env = options.env || process.env;
  const fallbackEnv = loadShellEnvFallback(options);
  const mergedEnv = {
    ...fallbackEnv,
    ...env,
  };
  const buildVersion = String(mergedEnv.ANCHORFLUX_BUILD_VERSION || "").trim();
  const isDevBuild = isDevBuildVersion(buildVersion);

  const nativeContextMenuEnabled = parseBooleanEnvFlag(
    mergedEnv.ANCHORFLUX_ENABLE_NATIVE_CONTEXT_MENU,
    isDevBuild
  );
  const devToolsEnabled = parseBooleanEnvFlag(
    mergedEnv.ANCHORFLUX_ENABLE_DEVTOOLS,
    nativeContextMenuEnabled || isDevBuild
  );
  const rendererProfilingEnabled = parseBooleanEnvFlag(
    mergedEnv.ANCHORFLUX_ENABLE_RENDERER_PROFILING,
    devToolsEnabled
  );
  const antiThrottlingEnabled = parseBooleanEnvFlag(
    mergedEnv.ANCHORFLUX_ENABLE_ANTI_THROTTLING,
    true
  );
  const metricsLoggingEnabled = parseBooleanEnvFlag(
    mergedEnv.ANCHORFLUX_ENABLE_SHELL_METRICS,
    rendererProfilingEnabled
  );
  const openDevToolsOnLaunch = parseBooleanEnvFlag(
    mergedEnv.ANCHORFLUX_OPEN_DEVTOOLS_ON_LAUNCH,
    devToolsEnabled && isDevBuild
  );

  return {
    buildVersion,
    isDevBuild,
    debugFlags: {
      nativeContextMenuEnabled,
      devToolsEnabled,
      rendererProfilingEnabled,
      openDevToolsOnLaunch,
    },
    performanceFlags: {
      antiThrottlingEnabled,
      metricsLoggingEnabled,
      diagnosticsIntervalMs: parsePositiveInteger(
        env.ANCHORFLUX_SHELL_DIAGNOSTICS_INTERVAL_MS,
        15000
      ),
    },
  };
}

function getChromiumSwitches(runtimeConfig) {
  const switches = [];
  if (!runtimeConfig?.performanceFlags?.antiThrottlingEnabled) {
    return switches;
  }

  return [
    "disable-renderer-backgrounding",
    "disable-background-timer-throttling",
    "disable-backgrounding-occluded-windows",
  ];
}

module.exports = {
  getChromiumSwitches,
  isDevBuildVersion,
  loadShellEnvFallback,
  parseBooleanEnvFlag,
  parseEnvFileContent,
  resolveShellEnvFileCandidates,
  resolveShellRuntimeConfig,
};
