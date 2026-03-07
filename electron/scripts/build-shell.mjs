import { spawnSync } from "node:child_process";

const isPack = process.argv.includes("--pack");
const args = [
  "exec",
  "electron-builder",
  "--win",
  "--x64",
  "--config",
  "electron-builder.yml",
];

if (!isPack) {
  args.push("--dir");
}

const result = spawnSync("npm", args, {
  stdio: "inherit",
  shell: process.platform === "win32",
});

if (result.status !== 0) {
  process.exit(result.status ?? 1);
}
