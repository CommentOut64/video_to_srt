const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

test("electron-builder 必须打包 shell_runtime_config.js，避免主进程启动时模块缺失", () => {
  const builderConfig = fs.readFileSync(
    path.join(__dirname, "electron-builder.yml"),
    "utf8"
  );

  assert.match(
    builderConfig,
    /^\s*-\s*shell_runtime_config\.js\s*$/m
  );
});
