const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("path");

const {
  resolveShellLogDir,
  buildTimestampedLogPath,
} = require("./shell_logging");

test("resolveShellLogDir 优先使用 ANCHORFLUX_SHELL_LOG_DIR", () => {
  const logDir = resolveShellLogDir({
    env: { ANCHORFLUX_SHELL_LOG_DIR: "D:\\AnchorFlux\\logs" },
    execPath: "D:\\AnchorFlux\\core\\shell\\AnchorFluxShell.exe",
  });
  assert.equal(logDir, "D:\\AnchorFlux\\logs");
});

test("resolveShellLogDir 从 shell 路径回溯到应用根目录", () => {
  const logDir = resolveShellLogDir({
    env: {},
    execPath: "D:\\AnchorFlux\\core\\shell\\AnchorFluxShell.exe",
  });
  assert.equal(logDir, "D:\\AnchorFlux\\logs");
});

test("buildTimestampedLogPath 生成带时间戳的日志文件名", () => {
  const fixedDate = new Date("2026-03-18T11:45:22.000Z");
  const logPath = buildTimestampedLogPath("D:\\AnchorFlux\\logs", "electron-main", fixedDate);
  assert.equal(
    logPath,
    path.join("D:\\AnchorFlux\\logs", "electron-main-20260318-114522.log")
  );
});
