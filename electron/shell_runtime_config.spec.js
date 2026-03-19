const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const {
  resolveShellRuntimeConfig,
} = require("./shell_runtime_config");

test("dev 版本默认开启原生右键、DevTools 与反节流", () => {
  const runtimeConfig = resolveShellRuntimeConfig({
    env: {
      ANCHORFLUX_BUILD_VERSION: "3.2.5-dev5",
    },
  });

  assert.equal(runtimeConfig.debugFlags.nativeContextMenuEnabled, true);
  assert.equal(runtimeConfig.debugFlags.devToolsEnabled, true);
  assert.equal(runtimeConfig.debugFlags.rendererProfilingEnabled, true);
  assert.equal(runtimeConfig.performanceFlags.antiThrottlingEnabled, true);
});

test("显式环境变量关闭时应覆盖 dev 默认值", () => {
  const runtimeConfig = resolveShellRuntimeConfig({
    env: {
      ANCHORFLUX_BUILD_VERSION: "3.2.5-dev5",
      ANCHORFLUX_ENABLE_NATIVE_CONTEXT_MENU: "0",
      ANCHORFLUX_ENABLE_DEVTOOLS: "false",
      ANCHORFLUX_ENABLE_RENDERER_PROFILING: "no",
      ANCHORFLUX_ENABLE_ANTI_THROTTLING: "off",
    },
  });

  assert.equal(runtimeConfig.debugFlags.nativeContextMenuEnabled, false);
  assert.equal(runtimeConfig.debugFlags.devToolsEnabled, false);
  assert.equal(runtimeConfig.debugFlags.rendererProfilingEnabled, false);
  assert.equal(runtimeConfig.performanceFlags.antiThrottlingEnabled, false);
});

test("当进程环境缺失时，应能从打包根目录 .env 恢复 dev 调试开关", () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "anchorflux-shell-"));
  const shellDir = path.join(tempRoot, "AnchorFlux", "core", "shell");
  fs.mkdirSync(shellDir, { recursive: true });
  fs.writeFileSync(
    path.join(tempRoot, "AnchorFlux", ".env"),
    [
      "ANCHORFLUX_BUILD_VERSION=v3.2.5-dev6",
      "ANCHORFLUX_ENABLE_DEVTOOLS=1",
      "ANCHORFLUX_ENABLE_NATIVE_CONTEXT_MENU=1",
    ].join("\n"),
    "utf8"
  );

  const runtimeConfig = resolveShellRuntimeConfig({
    env: {},
    cwd: shellDir,
  });

  assert.equal(runtimeConfig.isDevBuild, true);
  assert.equal(runtimeConfig.debugFlags.devToolsEnabled, true);
  assert.equal(runtimeConfig.debugFlags.nativeContextMenuEnabled, true);
});
