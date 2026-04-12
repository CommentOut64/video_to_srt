const fs = require("fs");
const path = require("path");

function normalizePath(rawValue) {
  if (typeof rawValue !== "string") {
    return "";
  }
  return rawValue.trim();
}

function resolveShellLogDir({
  env = process.env,
  execPath = process.execPath,
  cwd = process.cwd(),
} = {}) {
  const explicit = normalizePath(env.ANCHORFLUX_SHELL_LOG_DIR);
  if (explicit) {
    return path.resolve(explicit);
  }

  if (typeof execPath === "string" && execPath) {
    const shellDir = path.dirname(execPath);
    const coreDir = path.dirname(shellDir);
    const appRoot = path.dirname(coreDir);
    const shellName = path.basename(shellDir).toLowerCase();
    const coreName = path.basename(coreDir).toLowerCase();
    if (shellName === "shell" && coreName === "core") {
      return path.join(appRoot, "logs");
    }
  }

  return path.join(cwd, "logs");
}

function pad2(value) {
  return String(value).padStart(2, "0");
}

function formatUtcTimestamp(date) {
  return [
    date.getUTCFullYear(),
    pad2(date.getUTCMonth() + 1),
    pad2(date.getUTCDate()),
  ].join("") + `-${pad2(date.getUTCHours())}${pad2(date.getUTCMinutes())}${pad2(date.getUTCSeconds())}`;
}

function buildTimestampedLogPath(logDir, prefix, now = new Date()) {
  return path.join(logDir, `${prefix}-${formatUtcTimestamp(now)}.log`);
}

function ensureLogDir(logDir) {
  fs.mkdirSync(logDir, { recursive: true });
}

function formatError(error) {
  if (!error) {
    return "";
  }
  if (error instanceof Error) {
    return error.stack || error.message || String(error);
  }
  return String(error);
}

function safeAppendLog(logPath, level, message, error = null) {
  const timestamp = new Date().toISOString();
  const errorText = formatError(error);
  const line = errorText
    ? `${timestamp} [${level}] ${message}\n${errorText}\n`
    : `${timestamp} [${level}] ${message}\n`;
  fs.appendFileSync(logPath, line, { encoding: "utf-8" });
}

module.exports = {
  resolveShellLogDir,
  buildTimestampedLogPath,
  ensureLogDir,
  safeAppendLog,
};
