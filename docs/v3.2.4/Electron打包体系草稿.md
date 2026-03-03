# wgh 的 Electron 打包体系落地手册（Lite 开箱即用）

> 状态：可执行实施版
> 版本：V3.2.4+dev.20260303.03
> 适用范围：Windows 10/11 x64

## 1. 目标与交付标准

### 1.1 目标

1. 产出可直接分发的 **Lite Electron 安装包** 与 **Lite 便携包**
2. Lite 包必须 **开箱即用**，首启不联网、不安装依赖
3. 构建体系可平滑扩展为 `full-offline` 与 `full-hybrid`
4. 支持“修改代码后一键打包”和“自动生成更新包”

### 1.2 本期必须交付的产物

1. `AnchorFlux-Lite-Setup.exe`（安装包）
2. `AnchorFlux-Lite-Portable.zip`（便携包）
3. `anchorflux-lite-offline-update.zip`（更新包）
4. `latest-lite.json`（更新清单）

### 1.3 启动拓扑（冻结）

`AnchorFlux.exe(Go Stub) -> launcher -> backend + Electron Shell`

- 不再引入 bootloader 分支
- Lite/Full 共用同一启动架构，仅通过 profile 切换行为

## 2. 当前仓库基线与差距

### 2.1 已具备

1. 主入口已是 `stub/main.go + launcher/main.py`
2. Full/Lite 依赖拆分已完成（`pyproject.toml` 的 `optional-dependencies.full`）
3. 媒体 profile 分流已完成（`browser_compat` / `electron_native`）
4. 同音路由已 project-only，可在 Lite 中可用
5. 现有在线更新机制可复用（`update_signal.json` + launcher/updater）

### 2.2 待补齐（阻塞项）

1. 后端仍会启动后自动打开浏览器，需支持 `electron` UI 模式
2. 缺少 Shell 启动就绪探针 `GET /api/system/ready`
3. launcher 缺少 shell 管理模块（等待后端就绪、拉起 Electron、失败回退）
4. 仓库尚无 `electron/` 工程和 `scripts/packaging/` 一键打包脚本

## 3. 最终目录结构（实施后）

```text
electron/
  main.js
  preload.js
  loading.html
  package.json
  electron-builder.yml
  scripts/
    build-shell.mjs

scripts/packaging/
  build-profile.ps1
  assemble-runtime.ps1
  make-update-manifest.ps1
  profile-manifests/
    lite-offline.json
    full-offline.json
    full-hybrid.json
  nsis/
    anchorflux-installer.nsi

dist/releases/
  lite-offline/<version>/
    installer/
    portable/
    update/
```

## 4. Phase A：先改后端与启动器（必须先做）

### A-1 增加运行模式环境变量

**操作目标**

1. 增加 `ANCHORFLUX_UI_MODE=browser|electron|none`
2. 增加 `ANCHORFLUX_RUNTIME_POLICY=offline|hybrid`
3. launcher 能读取并传递给 backend

**修改点**

1. `launcher/config.py`
2. `launcher/main.py`

**操作步骤**

1. 在 `LauncherConfig` 增加字段：
   - `ui_mode: str = "browser"`
   - `runtime_policy: str = "offline"`
   - `shell_path: Optional[Path] = None`
2. 在 `load_env_config()` 的结果中读取上述变量并归一化
3. `start_backend()` 注入环境变量：
   - `ANCHORFLUX_UI_MODE`
   - `ANCHORFLUX_RUNTIME_POLICY`

**完成判定**

1. 启动日志能打印 `ui_mode/runtime_policy`
2. backend 进程环境变量可读取到对应值

---

### A-2 增加后端就绪探针与浏览器抑制

**修改点**

1. `backend/app/api/routes/system_routes.py`
2. `backend/app/main.py`

**操作步骤**

1. 新增接口 `GET /api/system/ready`，返回：
   - `success`
   - `ready`
   - `flavor`
   - `ui_mode`
   - `version`
   - `timestamp`
2. 在 `main.py` 启动阶段读取 `ANCHORFLUX_UI_MODE`
3. 仅当 `ui_mode == "browser"` 时执行 `open_browser_if_needed()`
4. `ui_mode == "electron"` 时打印日志“跳过自动打开浏览器”

**验收命令**

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/api/system/ready
```

**期望**

1. 返回 JSON 且 `ready=true`
2. `ANCHORFLUX_UI_MODE=electron` 时不自动弹浏览器

---

### A-3 增加 launcher Shell 管理模块

**新增文件**

1. `launcher/shell_manager.py`

**职责**

1. 轮询后端 `GET /api/system/ready`
2. 拉起 Electron 可执行文件
3. 失败时回退打开浏览器

**操作步骤**

1. `wait_backend_ready(base_url, timeout_sec=60)`  
   - 每 1 秒探测一次
   - 超时返回 false，不抛异常导致主流程崩溃
2. `launch_electron(shell_path)`  
   - shell_path 优先级：
     1. `ANCHORFLUX_SHELL_PATH`
     2. 默认 `core/shell/AnchorFluxShell.exe`
3. `open_browser_fallback(url)`  
   - 仅作为降级分支
4. 在 `launcher/main.py` 里接入：
   - backend 启动成功后，`ui_mode == electron` 时执行就绪等待与 shell 拉起

**验收标准**

1. shell 存在：进入 Electron 窗口
2. shell 缺失：自动回退浏览器并有明确日志

## 5. Phase B：搭建 Electron 壳工程

### B-1 初始化工程

**创建目录**

```powershell
New-Item -ItemType Directory -Force electron
New-Item -ItemType Directory -Force electron\scripts
```

**初始化依赖（在 `electron/` 下）**

```powershell
npm init -y
npm install electron electron-builder wait-on --save-dev
```

---

### B-2 编写 `electron/main.js`

**最小职责**

1. 创建窗口（隐藏菜单、最小尺寸限制）
2. 先加载 `loading.html`
3. 等待后端 ready 后加载 `http://127.0.0.1:8000`
4. 关闭窗口时退出 app

**关键参数建议**

1. `width: 1600, height: 960`
2. `show: false`，ready 后再 `show()`
3. `webPreferences.contextIsolation=true`
4. `nodeIntegration=false`

---

### B-3 编写 `electron/loading.html`

**要求**

1. 显示“后端启动中”
2. 显示超时错误提示
3. 不承担业务逻辑，只作启动过渡页

---

### B-4 编写 `electron/electron-builder.yml`

**建议配置**

1. `appId: com.anchorflux.shell`
2. `productName: AnchorFluxShell`
3. `asar: true`
4. `win.target: nsis`（shell 自身构建）
5. 输出目录固定到：`electron/dist-shell`

---

### B-5 增加 Shell 构建命令

**`electron/package.json` 脚本建议**

1. `dev:shell`: 本地调试 main.js
2. `build:shell`: `electron-builder --win --x64 --dir`
3. `pack:shell`: `electron-builder --win --x64`

**验收命令**

```powershell
npm --prefix electron run build:shell
```

## 6. Python 运行时离线打包策略

> 本章是 Phase C 打包流水线的前置知识。定义后端依赖如何在构建机上生成、瘦身、捆绑进安装包，实现用户侧"开箱即用、首启不联网"。

### 6.1 设计目标

1. 安装包内自带完整 Python 解释器 + 第三方依赖，用户机器**无需安装 Python**
2. `.venv` 可在任意目录运行，不依赖构建机路径
3. 构建过程完全基于 `uv`，从 `uv.lock` 锁文件复现，确保位对位一致
4. 支持离线修复和联网兜底两种恢复路径

### 6.2 核心原理：uv 的自包含 .venv

`uv venv --python 3.10` 使用 [python-build-standalone](https://github.com/indygreg/python-build-standalone) 创建虚拟环境。与标准 `python -m venv` 的关键区别：

| | 标准 venv | uv venv |
|---|---|---|
| 解释器 | 符号链接/轻量副本，依赖系统 Python | python-build-standalone 完整副本，自包含 |
| 标准库 | 共享系统 Python | 打包在 .venv 内 |
| 可搬迁 | 不行（pyvenv.cfg 路径断裂） | 可以（设计为可重定位） |
| 分发 | 需额外嵌入解释器 | 直接打包 .venv 目录即可 |

**结论**：只要用 `uv` 创建 `.venv`，就已经自带嵌入式 Python，无需额外操作。

### 6.3 三层离线保障体系

```
Layer 1 (首选)：预构建 .venv 直接捆绑 → 开箱即用
    ↓ .venv 损坏或版本升级时
Layer 2 (本地修复)：内置 Wheel 缓存 + uv pip install --no-index → 离线重建
    ↓ 缓存也丢失时
Layer 3 (兜底恢复)：tools/uv.exe + pyproject.toml + uv.lock → 联网重建
```

#### Layer 1：预构建 .venv 捆绑（核心）

构建机上执行，产出直接放入安装包。用户拿到即可运行，无需任何安装步骤。

```powershell
# Step 1: 在构建机临时目录创建干净 venv
#   uv 自动下载 python-build-standalone 3.10，解释器完整复制进 .venv/Scripts/
tools\uv.exe venv .\_build\staging\.venv --python 3.10

# Step 2: 从锁文件精确同步依赖（--frozen 禁止更新锁文件）
#   Lite:
tools\uv.exe sync --frozen --no-dev --project . --python .\_build\staging\.venv\Scripts\python.exe
#   Full:
tools\uv.exe sync --frozen --no-dev --extra full --project . --python .\_build\staging\.venv\Scripts\python.exe

# Step 3: 瘦身清理（见 6.4）

# Step 4: 将 .venv 目录整体放入安装包 staging 区
```

**可重定位验证**（在一台没有 Python 的干净机器上执行）：

```powershell
.\.venv\Scripts\python.exe -c "import sys; print(sys.prefix, sys.executable)"
.\.venv\Scripts\python.exe -c "import fastapi; print('ok')"
```

#### Layer 2：内置 Wheel 缓存（离线修复用）

不是首次安装需要的，而是为**离线环境修复**和**离线增量更新**服务。

```powershell
# 导出精确依赖清单
tools\uv.exe export --frozen --no-dev --format requirements-txt -o .\_build\requirements.txt
# Full 版追加：--extra full

# 下载所有 Wheel 到本地目录
tools\uv.exe pip wheel -r .\_build\requirements.txt --wheel-dir .\_build\wheels\
```

修复时的使用方式：

```powershell
# 用户侧：.venv 损坏时，从内置缓存离线重建
tools\uv.exe pip install --python .venv\Scripts\python.exe --no-index --find-links _vendor\wheels\ -r _vendor\requirements.txt
```

> **已知限制**：`uv sync --find-links` 在某些版本下仍可能尝试访问 PyPI（[astral-sh/uv#15519](https://github.com/astral-sh/uv/issues/15519)）。Layer 2 的修复命令因此使用 `uv pip install` 而非 `uv sync`，绕过此问题。

#### Layer 3：联网兜底

最后手段，需要网络。产物中保留 `pyproject.toml` + `uv.lock` + `tools/uv.exe`，用户可手动执行：

```powershell
tools\uv.exe sync --frozen
```

### 6.4 .venv 瘦身规则

构建时对 `.venv` 执行清理，减小安装包体积。由 `assemble-runtime.ps1` 中的清理函数实现。

**清理操作**：

1. **删除 `__pycache__`**：运行时自动重建，节省 10-20%
2. **精简 `.dist-info`**：仅保留 `METADATA`、`RECORD`、`INSTALLER` 三个文件
3. **删除包内测试目录**：`**/tests/`、`**/test/` 递归删除
4. **Lite 强制排除 Full 依赖**（防御性，正常 sync 不会安装这些）：
   - `torch/`、`torchaudio/`、`demucs/`、`faster_whisper/`
   - `pyannote/`、`speechbrain/`、`onnxruntime_gpu*/`
   - `librosa/`、`scipy/`、`silero_vad/`
5. **可选：预编译 .pyc**（用空间换首启速度）：
   `.venv\Scripts\python.exe -m compileall -q .venv\Lib\site-packages`

**体积预估**：

| Profile | sync 后原始 | 瘦身后 | 压缩后 (.zip) |
|---------|------------|--------|--------------|
| Lite | ~300MB | ~200MB | ~70-90MB |
| Full | ~5GB | ~4GB | ~1.5-2GB |

### 6.5 打包产物中的运行时目录结构

```text
AnchorFlux/
├── AnchorFlux.exe              # Go Stub（入口）
├── launcher/                    # Python 启动器
├── backend/                     # 后端代码
├── frontend/dist/               # 前端构建产物
├── core/shell/                  # Electron Shell
│   ├── AnchorFluxShell.exe
│   └── ...
├── .venv/                       # [Layer 1] 预构建 Python 环境
│   ├── Scripts/
│   │   ├── python.exe           #   python-build-standalone 解释器（~40MB）
│   │   └── python310.dll
│   └── Lib/
│       └── site-packages/       #   已安装的第三方依赖
├── _vendor/                     # [Layer 2] 离线恢复资源
│   ├── wheels/                  #   Wheel 缓存（.whl 文件）
│   └── requirements.txt         #   精确依赖清单
├── tools/
│   ├── uv.exe                   # [Layer 3] 内置 uv
│   └── ffmpeg.exe
├── pyproject.toml               # [Layer 3] 项目定义
└── uv.lock                      # [Layer 3] 依赖锁文件
```

### 6.6 Launcher 恢复能力增强

`launcher/uv_manager.py` 的 `sync_dependencies()` 需增强为三层恢复策略：

1. 正常启动：`uv sync --check` 返回 0 → 跳过，直接启动后端
2. 需要同步时优先走 Layer 2：`uv pip install --no-index --find-links _vendor/wheels/`
3. Layer 2 失败再走 Layer 3：`uv sync --frozen`（联网）
4. 全部失败：打印明确错误信息，不崩溃

### 6.7 Full-Hybrid 模式特殊处理

`full-hybrid` 产物内仅包含 Lite 基线 `.venv`（~200MB），首启时由 launcher 增量安装 Full 依赖：

```
首启流程:
1. Go Stub 启动
2. launcher 检测 flavor=full + runtime_policy=hybrid
3. launcher 执行 uv sync --frozen --extra full --find-links _vendor/wheels/
   - 本地 Wheel 缓存命中的包直接安装（大部分 Lite 基线包）
   - Full 增量包（torch 等）从网络下载
4. 后续启动走 uv sync --check 快速验证
```

优势：安装包体积从 ~2GB 降到 ~300MB，代价是首启需联网约 5-10 分钟。

---

## 7. Phase C：建立 Profile 驱动的一键打包流水线

### C-1 新增 profile 清单

**文件**

1. `scripts/packaging/profile-manifests/lite-offline.json`
2. `scripts/packaging/profile-manifests/full-offline.json`
3. `scripts/packaging/profile-manifests/full-hybrid.json`

**`lite-offline.json` 完整字段**

```json
{
  "profile": "lite-offline",
  "flavor": "lite",
  "uiMode": "electron",
  "runtimePolicy": "offline",
  "python": {
    "version": "3.10",
    "syncArgs": ["--frozen", "--no-dev"],
    "excludePackages": [
      "torch", "torchaudio", "demucs", "faster-whisper",
      "onnxruntime-gpu", "pyannote-audio", "speechbrain",
      "librosa", "scipy", "silero-vad", "punctuators"
    ],
    "includeWheelCache": true,
    "includeUvBinary": true,
    "precompilePyc": true
  },
  "frontendBuildEnv": {
    "VITE_APP_FLAVOR": "lite",
    "VITE_LITE_MODE": "true"
  }
}
```

**`full-offline.json` 差异项**

```json
{
  "profile": "full-offline",
  "flavor": "full",
  "python": {
    "syncArgs": ["--frozen", "--no-dev", "--extra", "full"],
    "excludePackages": []
  }
}
```

**`full-hybrid.json` 差异项**

```json
{
  "profile": "full-hybrid",
  "flavor": "full",
  "runtimePolicy": "hybrid",
  "python": {
    "syncArgs": ["--frozen", "--no-dev"],
    "excludePackages": [],
    "includeWheelCache": false
  }
}
```

---

### C-2 实现 `build-profile.ps1`（统一入口）

**输入参数**

1. `-Profile lite-offline|full-offline|full-hybrid`
2. `-Artifacts portable|installer|update|all`
3. `-Version 3.2.4+dev.20260303.XX`

**固定流程**

1. 读取 profile manifest
2. 构建前端（写入 flavor 环境变量）
3. 构建 electron shell（win-unpacked）
4. 构建 stub（`go build`）
5. 构建 Python 运行时（详见步骤 5 展开）
6. 调用 `assemble-runtime.ps1` 组装目录
7. 产出 portable zip
8. 产出 installer exe（NSIS）
9. 产出 update zip + latest json

**步骤 5 展开：构建 Python 运行时**

```powershell
function Build-PythonRuntime {
    param(
        [string]$Profile,
        [string]$StagingDir,
        [string]$ProjectRoot
    )

    $manifest = Get-Content "$ProjectRoot\scripts\packaging\profile-manifests\$Profile.json" | ConvertFrom-Json
    $pyConf = $manifest.python
    $venvDir = "$StagingDir\.venv"
    $vendorDir = "$StagingDir\_vendor"
    $uvExe = "$ProjectRoot\tools\uv.exe"

    # 5a: 创建干净 venv（自带 python-build-standalone 解释器）
    Write-Host "[runtime] 创建虚拟环境 (python $($pyConf.version))..."
    & $uvExe venv $venvDir --python $pyConf.version

    # 5b: 从锁文件同步依赖
    Write-Host "[runtime] 同步依赖 (profile=$Profile)..."
    $syncArgs = @("sync") + $pyConf.syncArgs + @("--project", $ProjectRoot, "--python", "$venvDir\Scripts\python.exe")
    & $uvExe @syncArgs

    # 5c: 瘦身清理
    Write-Host "[runtime] 清理虚拟环境..."
    # 删除 __pycache__
    Get-ChildItem $venvDir -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
    # 精简 .dist-info
    Get-ChildItem "$venvDir\Lib\site-packages\*.dist-info" -Directory | ForEach-Object {
        Get-ChildItem $_ -File | Where-Object { $_.Name -notin @("METADATA","RECORD","INSTALLER") } | Remove-Item -Force
    }
    # 删除包内测试目录
    Get-ChildItem "$venvDir\Lib\site-packages" -Recurse -Directory | Where-Object { $_.Name -in @("tests","test") } | Remove-Item -Recurse -Force
    # 防御性排除 Full 依赖
    foreach ($pkg in $pyConf.excludePackages) {
        Get-ChildItem "$venvDir\Lib\site-packages" -Directory -Filter $pkg | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }
    # 可选：预编译 .pyc
    if ($pyConf.precompilePyc) {
        & "$venvDir\Scripts\python.exe" -m compileall -q "$venvDir\Lib\site-packages"
    }

    # 5d: 生成 Wheel 缓存（Layer 2）
    if ($pyConf.includeWheelCache) {
        Write-Host "[runtime] 生成离线 Wheel 缓存..."
        New-Item -ItemType Directory -Force "$vendorDir\wheels" | Out-Null
        & $uvExe export --frozen --no-dev --format requirements-txt -o "$vendorDir\requirements.txt"
        & $uvExe pip wheel -r "$vendorDir\requirements.txt" --wheel-dir "$vendorDir\wheels"
    }

    # 5e: 复制 uv.exe 和项目元文件（Layer 3）
    if ($pyConf.includeUvBinary) {
        New-Item -ItemType Directory -Force "$StagingDir\tools" | Out-Null
        Copy-Item "$ProjectRoot\tools\uv.exe" "$StagingDir\tools\uv.exe"
    }
    Copy-Item "$ProjectRoot\pyproject.toml" "$StagingDir\pyproject.toml"
    Copy-Item "$ProjectRoot\uv.lock" "$StagingDir\uv.lock"

    Write-Host "[runtime] Python 运行时准备完成"
}
```

---

### C-3 实现 `assemble-runtime.ps1`

**输入**

1. profile
2. version
3. staging 目录

**组装规则（Lite 必须包含）**

1. `AnchorFlux.exe`（Go stub）
2. `launcher/`
3. `backend/`
4. `frontend/dist/`
5. `core/shell/AnchorFluxShell.exe`（以及壳运行时目录）
6. `.venv/`（自包含 Python 解释器 + Lite 依赖，见第 6 章）
7. `_vendor/`（Wheel 缓存 + requirements.txt，见 6.3 Layer 2）
8. `tools/uv.exe`（见 6.3 Layer 3）
9. `pyproject.toml`, `uv.lock`（见 6.3 Layer 3）

**Lite 禁止打入**

1. torch/torchaudio
2. onnxruntime-gpu
3. demucs
4. faster-whisper
5. pyannote.audio/speechbrain

## 8. Phase D：Lite 离线包实操（按顺序执行）

### D-1 清理旧产物

```powershell
Remove-Item -Recurse -Force dist\releases\lite-offline -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force dist\staging\lite-offline -ErrorAction SilentlyContinue
```

### D-2 执行一键打包

```powershell
pwsh scripts/packaging/build-profile.ps1 -Profile lite-offline -Artifacts all -Version 3.2.4+dev.20260303.02
```

### D-3 断网冒烟验证（必须做）

1. 断网
2. 安装 `AnchorFlux-Lite-Setup.exe`
3. 双击启动
4. 验证不触发依赖下载
5. 验证可打开编辑器、导入字幕、导出字幕

### D-4 产物检查

```powershell
Get-ChildItem dist\releases\lite-offline\3.2.4+dev.20260303.02 -Recurse
```

应至少包含：

1. `installer/AnchorFlux-Lite-Setup.exe`
2. `portable/AnchorFlux-Lite-Portable.zip`
3. `update/anchorflux-lite-offline-update.zip`
4. `update/latest-lite.json`

## 9. Phase E：为 Full 与自动更新预留扩展位

### E-1 Full 扩展策略

1. `full-offline`：切换 profile，`python.syncArgs` 追加 `--extra full`，其余流程不变
2. `full-hybrid`：包内仅放 Lite 基线 `.venv`，首启由 launcher 增量安装（详见 6.7）
3. 不新增新的打包脚本，只新增 profile manifest

### E-2 自动更新包生成

**新增脚本**

1. `scripts/packaging/make-update-manifest.ps1`

**输出格式**

`latest-lite.json` 必含字段：

1. `version`
2. `profile`
3. `sha256`
4. `size`
5. `download_url`
6. `release_date`

**对接方式**

1. 继续使用现有 `update_signal.json` 协议
2. 后端更新接口返回的 `download_url` 指向对应 profile 的 update zip

## 10. CI 落地步骤（建议立即接入）

### 10.1 新增工作流

1. `.github/workflows/electron_packaging.yml`

### 10.2 Lite Job（首期必须）

1. 安装 Node + Python + Go + NSIS
2. 执行 `build-profile.ps1 -Profile lite-offline -Artifacts all`
3. 执行离线冒烟脚本（至少验证可启动与 `/api/system/ready`）
4. 上传 installer/portable/update 三类产物

### 10.3 后续扩展

1. 增加 matrix：`full-offline`, `full-hybrid`
2. 对 full 仅夜间构建，避免 PR 耗时过长

## 11. 验收清单（发布门禁）

### 11.1 功能门禁

1. 双击 `AnchorFlux.exe` 默认打开 Electron，不自动开浏览器
2. Shell 缺失可回退浏览器
3. Lite 断网首启可用
4. 同音、字幕编辑、导入导出可用

### 11.2 依赖门禁

1. Lite 包不含 Full 推理重依赖
2. `.venv` 自包含验证：在无 Python 的干净机器上 `.venv\Scripts\python.exe -c "import fastapi"` 成功
3. `uv sync` 与 `uv sync --extra full` 均可成功

### 11.3 构建门禁

1. 一条命令可完成打包
2. 必须同时产出 `installer + portable + update`
3. `latest-lite.json` 与 update zip 的哈希一致

## 12. 回滚与应急

1. 紧急降级：设置 `ANCHORFLUX_UI_MODE=browser`
2. Shell 崩溃：launcher 自动回退浏览器
3. 打包链故障：可临时回退到旧分发方式，但保留 profile manifest 与脚本入口不变

## 13. 实施顺序建议（严格按序）

1. 先做 Phase A（后端/launcher 改造）
2. 再做 Phase B（Electron 壳）
3. 再做 Phase C（打包脚本，依赖第 6 章的运行时打包策略）
4. 然后做 Phase D（Lite 离线实操）
5. 最后做 Phase E（Full 与更新扩展）

> 注意：不要在 Electron 工程未落地前直接开始安装包联调，否则会在启动链路上反复返工
> 注意：Phase C 开始前建议先在本地手动验证第 6 章的 .venv 可重定位性（见 6.3 Layer 1 验证命令）

