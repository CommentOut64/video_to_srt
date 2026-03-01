# wgh 的 Electron 打包体系实施规格（Lite 优先，Full 可扩展）

## 1. 简要摘要
1. 采用你已确认的统一拓扑：`AnchorFlux.exe(Go Stub) -> launcher -> backend + Electron Shell`，Lite 和 Full 共用同一启动链路。
2. Lite 首版走“离线开箱即用（预打包全部运行时）”，同时输出 `安装器EXE` 和 `便携ZIP`。
3. Full 预留两种构建形态并共用同一套脚本：`full-offline`（预打包）与 `full-hybrid`（首启补依赖）。
4. 打包工具锁定 `electron-builder`，其职责是构建 Shell 运行时；最终整包安装器由统一总装脚本生成（NSIS），以保留 Stub 主入口。

## 2. 决策冻结（本方案不再留空）
1. 启动拓扑：Lite/Full 同拓扑，不拆两套入口。
2. 产物形态：Lite 必须同时产出安装器和便携包；Full 同标准执行。
3. Electron 工具链：锁定 `electron-builder`。
4. 运行时策略：
   1. `lite-offline`：预打包完整 Lite 运行时，首启不联网。
   2. `full-offline`：预打包完整 Full 运行时，首启不联网。
   3. `full-hybrid`：预打包最小 Python 运行时，首启 `uv sync` 补齐 Full 依赖。
5. 平台范围：Windows 10/11 x64（当前阶段不做 macOS/Linux）。

## 3. 当前基线与差异点（以真实仓库为准）
1. 仓库当前主入口是 `AnchorFlux.exe + stub/main.go + launcher/main.py`，不是 `bootloader.py`。
2. `backend/app/main.py` 启动后会无条件 `open_browser_if_needed()`，这与 Electron 壳模式冲突。
3. 现有 release/update 脚本存在历史残留（仍引用 `bootloader.py`、`requirements.txt`），需改为 launcher/stub 体系。
4. Full/Lite 运行时开关已存在（`ANCHORFLUX_FLAVOR`、`VITE_APP_FLAVOR/VITE_LITE_MODE`），可直接作为 profile 输入。
5. 你要求按“假设前端状态管理重构已完成”设计，本方案仅依赖能力选择器契约，不绑定旧 store 细节。

## 4. 目标目录与文件改造（决策完成版）
1. 新增 `electron/`：
   1. `electron/main.js`
   2. `electron/preload.js`
   3. `electron/loading.html`
   4. `electron/package.json`
   5. `electron/electron-builder.yml`
   6. `electron/profiles/lite-offline.env`
   7. `electron/profiles/full-offline.env`
   8. `electron/profiles/full-hybrid.env`
   9. `electron/scripts/build-shell.mjs`
2. 新增 `scripts/packaging/`：
   1. `scripts/packaging/build-profile.ps1`
   2. `scripts/packaging/assemble-runtime.ps1`
   3. `scripts/packaging/profile-manifests/lite-offline.json`
   4. `scripts/packaging/profile-manifests/full-offline.json`
   5. `scripts/packaging/profile-manifests/full-hybrid.json`
   6. `scripts/packaging/nsis/anchorflux-installer.nsi`
3. 新增 `packaging/requirements/`：
   1. `packaging/requirements/lite-runtime.lock.txt`
   2. `packaging/requirements/full-runtime.lock.txt`（给 offline）
4. 修改 `launcher/main.py`：
   1. 增加 Shell 启动流程（健康检查后拉起 Electron）。
   2. 增加 `offline/hybrid` 运行时策略分支。
   3. 增加 Shell 缺失时回退浏览器策略。
5. 新增 `launcher/shell_manager.py`（统一封装 Shell 启动、轮询、超时与退出联动）。
6. 修改 `backend/app/main.py` 与 `backend/app/api/routes/system_routes.py`：
   1. 新增后端就绪接口。
   2. 在 `ANCHORFLUX_UI_MODE=electron` 时禁止自动开浏览器。
7. 修改 `frontend/package.json`：
   1. 新增 `build:lite` 与 `build:full`。
8. 文档改造：
   1. 新增 `llmdoc/architecture/electron-packaging-full-lite.md`。
   2. 更新 `llmdoc/index.md`。
   3. 更新 `llmdoc/changelog.md`。
   4. 修正文档中 `bootloader` 口径到 `stub+launcher` 现状。

## 5. 公共接口/类型/命令变更（必须落地）
1. 新增环境变量：
   1. `ANCHORFLUX_UI_MODE=browser|electron|none`（默认 `browser`）。
   2. `ANCHORFLUX_RUNTIME_POLICY=offline|hybrid`（默认 `offline`）。
   3. `ANCHORFLUX_SHELL_PATH`（可选，覆盖默认 Shell 可执行文件路径）。
2. 新增后端接口：
   1. `GET /api/system/ready`
   2. 响应结构固定：`{ success, ready, flavor, ui_mode, version, timestamp }`
3. 构建命令接口（统一入口）：
   1. `pwsh scripts/packaging/build-profile.ps1 -Profile lite-offline -Artifacts all`
   2. `pwsh scripts/packaging/build-profile.ps1 -Profile full-offline -Artifacts all`
   3. `pwsh scripts/packaging/build-profile.ps1 -Profile full-hybrid -Artifacts all`
4. 前端构建命令：
   1. `npm run build:lite` -> `VITE_APP_FLAVOR=lite VITE_LITE_MODE=true`
   2. `npm run build:full` -> `VITE_APP_FLAVOR=full VITE_LITE_MODE=false`

## 6. 构建与打包流水线（Lite 先行）
1. Lite-offline（第一优先）：
   1. 前端按 Lite profile 构建。
   2. 构建 Electron Shell（win-unpacked）。
   3. 组装运行时：Stub、launcher、backend、frontend/dist、Shell、`.venv`(lite lock 预装)、tools。
   4. 产出便携包 ZIP。
   5. 产出 NSIS 安装器 EXE。
2. Full-offline（第二阶段）：
   1. 复用同脚本，仅切换 profile 和 full lock。
   2. 输出同样两种产物。
3. Full-hybrid（第三阶段）：
   1. 仅预置最小 `.venv` 与 `uv`，不打重依赖。
   2. 首启执行 `uv sync` 补齐 full 依赖。
   3. 输出同样两种产物。

## 7. 测试用例与验收场景
1. 单元测试：
   1. launcher 的 `ui_mode/runtime_policy` 决策分支。
   2. profile manifest 解析与文件装配规则。
   3. Shell 就绪轮询和超时逻辑。
2. 集成测试（Lite-offline，必须通过）：
   1. 断网首启可进 UI，不触发依赖安装。
   2. 导入字幕 -> 编辑 -> 导出闭环可用。
   3. 双击 `AnchorFlux.exe` 不再自动开浏览器，只开 Electron。
   4. Shell 缺失时自动回退浏览器并给出可见日志。
3. 集成测试（Full-offline）：
   1. 转录相关路由可用，模型链路可启动。
   2. Electron 壳模式下任务创建/监控正常。
4. 集成测试（Full-hybrid）：
   1. 首启触发 `uv sync`，完成后进入主界面。
   2. 二次启动不重复全量安装。
5. 打包产物检查：
   1. Lite 包中不得包含 `torch/onnxruntime/faster-whisper/pyannote/demucs`。
   2. Full-offline 包必须包含完整重依赖。
   3. Full-hybrid 包允许不含重依赖，但必须含 `uv`、`pyproject.toml`、`uv.lock`。
6. 时长约束：
   1. 单测与构建校验拆分执行，单项不超过 60s 的自动化检查先行，重流程单独任务跑。

## 8. 风险与回滚
1. 风险：旧文档和旧脚本仍按 `bootloader.py` 思路。
   1. 处置：先完成文档口径统一，再替换构建脚本默认入口。
2. 风险：Lite 依赖裁剪后存在隐式导入导致启动失败。
   1. 处置：在 CI 加“Lite 启动冒烟 + import 守卫”并锁死基线。
3. 风险：Shell 启动失败导致用户无界面。
   1. 处置：launcher 强制回退浏览器模式，并输出明确日志。
4. 回滚策略：
   1. 保留 `ANCHORFLUX_UI_MODE=browser` 作为一键降级开关。
   2. 新打包链与旧链并行一段时间，先灰度 Lite 包。

## 9. 明确假设与默认值
1. 假设 `llmdoc/architecture/frontend-state-management-refactor-plan.md` 的目标契约可用，当前未完全落地不阻塞打包基建。
2. 默认后端端口仍为 `8000`，前端生产仍由后端静态托管。
3. 默认 Electron 仅负责窗口壳，不承载业务计算。
4. 默认主版本保持 `3.2.4`，开发迭代采用 `+dev.YYYYMMDD.NN`。
5. 默认先交付 `lite-offline`，`full-offline` 与 `full-hybrid` 按同一脚本体系扩展，不再重做架构。

