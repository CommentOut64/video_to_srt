param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("lite-offline", "full-offline", "full-hybrid")]
    [string]$Profile,
    [Parameter(Mandatory = $false)]
    [ValidateSet("portable", "installer", "update", "all")]
    [string]$Artifacts = "all",
    [Parameter(Mandatory = $true)]
    [string]$Version
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-ProjectRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Action
    )
    Write-Host "[step] $Label"
    & $Action
}

function Get-UvExecutable {
    param([string]$ProjectRoot)
    $bundled = Join-Path $ProjectRoot "tools\uv.exe"
    if (Test-Path $bundled) {
        return $bundled
    }
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($null -ne $uvCmd) {
        return $uvCmd.Source
    }
    throw "未找到 uv 可执行文件（tools\\uv.exe 或 PATH 中的 uv）"
}

function Test-UvPipWheelSupport {
    param([string]$UvExe)
    $output = & $UvExe pip --help 2>&1
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    return (($output | Out-String) -match "(?m)^\s*wheel\s")
}

function Get-ExtraArgsFromSyncArgs {
    param([object[]]$SyncArgs)
    $result = @()
    for ($i = 0; $i -lt $SyncArgs.Count; $i++) {
        if ($SyncArgs[$i] -eq "--extra" -and $i + 1 -lt $SyncArgs.Count) {
            $result += @("--extra", [string]$SyncArgs[$i + 1])
            $i++
        }
    }
    return $result
}

function Build-Frontend {
    param(
        [string]$ProjectRoot,
        [pscustomobject]$Manifest
    )
    $savedEnv = @{}
    foreach ($entry in $Manifest.frontendBuildEnv.PSObject.Properties) {
        $savedEnv[$entry.Name] = [Environment]::GetEnvironmentVariable($entry.Name, "Process")
        [Environment]::SetEnvironmentVariable($entry.Name, [string]$entry.Value, "Process")
    }
    try {
        & npm --prefix (Join-Path $ProjectRoot "frontend") run build
        if ($LASTEXITCODE -ne 0) {
            throw "前端构建失败"
        }
    }
    finally {
        foreach ($entry in $savedEnv.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
        }
    }
}

function Build-ElectronShell {
    param([string]$ProjectRoot)
    Prepare-ElectronBuildResources -ProjectRoot $ProjectRoot
    $electronRoot = Join-Path $ProjectRoot "electron"
    Push-Location $electronRoot
    try {
        & npm install | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "Electron 依赖安装失败"
        }
        & npm run build:shell | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "Electron Shell 构建失败"
        }
    }
    finally {
        Pop-Location
    }
    $shellDir = Join-Path $ProjectRoot "electron\dist-shell\win-unpacked"
    if (-not (Test-Path $shellDir)) {
        throw "未找到 Shell 产物目录: $shellDir"
    }
    return [string]$shellDir
}

function Prepare-ElectronBuildResources {
    param([string]$ProjectRoot)
    $electronBuildDir = Join-Path $ProjectRoot "electron\build"
    New-Item -ItemType Directory -Force -Path $electronBuildDir | Out-Null

    $iconCandidates = @(
        (Join-Path $ProjectRoot "frontend\dist\favicon.ico"),
        (Join-Path $ProjectRoot "frontend\public\favicon.ico")
    )
    $iconSource = $iconCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $iconSource) {
        throw "未找到 Electron 图标源文件（尝试路径：frontend\\dist\\favicon.ico / frontend\\public\\favicon.ico）"
    }

    $iconTarget = Join-Path $electronBuildDir "icon.ico"

    $iconBuildScript = @"
from PIL import Image
src = r'''$iconSource'''
dst = r'''$iconTarget'''
image = Image.open(src).convert("RGBA")
if image.width != 256 or image.height != 256:
    image = image.resize((256, 256), Image.LANCZOS)
image.save(
    dst,
    format="ICO",
    sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
)
"@
    # 防御性处理：某些终端/编码组合下，管道给 python - 会注入 BOM，触发 SyntaxError(U+FEFF)。
    # 改为写入 UTF-8 无 BOM 的临时脚本文件再执行。
    $iconBuildScript = $iconBuildScript.TrimStart([char]0xFEFF)
    $tempIconScript = Join-Path $env:TEMP ("anchorflux-icon-build-" + [guid]::NewGuid().ToString("N") + ".py")
    [System.IO.File]::WriteAllText(
        $tempIconScript,
        $iconBuildScript,
        (New-Object System.Text.UTF8Encoding($false))
    )
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    try {
        if ($null -ne $pythonCmd) {
            & $pythonCmd.Source $tempIconScript
        }
        else {
            $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
            if ($null -eq $pyLauncher) {
                throw "未找到 python 或 py 启动器，无法生成 Electron 图标"
            }
            & $pyLauncher.Source -3 $tempIconScript
        }
    }
    finally {
        Remove-Item -Path $tempIconScript -Force -ErrorAction SilentlyContinue
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $iconTarget)) {
        throw "生成 Electron 图标失败: $iconTarget"
    }
}

function Build-GoStub {
    param(
        [string]$ProjectRoot,
        [string]$OutputExePath,
        [switch]$HideConsole
    )
    New-Item -ItemType Directory -Force -Path (Split-Path $OutputExePath -Parent) | Out-Null
    Push-Location (Join-Path $ProjectRoot "stub")
    try {
        $isWindowsPlatform = ($env:OS -eq "Windows_NT")
        if ($HideConsole -and $isWindowsPlatform) {
            & go build -ldflags "-H=windowsgui" -o $OutputExePath .
        }
        else {
            & go build -o $OutputExePath .
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Go Stub 构建失败"
        }
    }
    finally {
        Pop-Location
    }
}

function Build-PythonRuntime {
    param(
        [string]$ProjectRoot,
        [pscustomobject]$Manifest,
        [string]$RuntimeDir
    )
    $uvExe = Get-UvExecutable -ProjectRoot $ProjectRoot
    Remove-Item -Recurse -Force -Path $RuntimeDir -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

    $pythonConfig = $Manifest.python
    $venvDir = Join-Path $RuntimeDir ".venv"
    $pythonExe = Join-Path $venvDir "Scripts\python.exe"
    $syncArgs = @([string[]]$pythonConfig.syncArgs)

    $savedUvProjectEnv = [Environment]::GetEnvironmentVariable("UV_PROJECT_ENVIRONMENT", "Process")
    try {
        # 强制 uv 指向打包产物的 .venv，避免受开发机 .env 中 UV_PROJECT_ENVIRONMENT 干扰。
        [Environment]::SetEnvironmentVariable("UV_PROJECT_ENVIRONMENT", $venvDir, "Process")

        & $uvExe venv $venvDir --python ([string]$pythonConfig.version)
        if ($LASTEXITCODE -ne 0) {
            throw "创建 .venv 失败"
        }

        $syncCommand = @("sync") + $syncArgs + @("--no-install-project", "--project", $ProjectRoot, "--python", $pythonExe)
        & $uvExe @syncCommand
        if ($LASTEXITCODE -ne 0) {
            throw "uv sync 失败"
        }
    }
    finally {
        [Environment]::SetEnvironmentVariable("UV_PROJECT_ENVIRONMENT", $savedUvProjectEnv, "Process")
    }

    Get-ChildItem -Path $venvDir -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path (Join-Path $venvDir "Lib\site-packages") -Recurse -Directory -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -in @("tests", "test")
    } | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    if ([bool]$pythonConfig.precompilePyc) {
        & $pythonExe -m compileall -q (Join-Path $venvDir "Lib\site-packages")
    }

    $extraArgs = Get-ExtraArgsFromSyncArgs -SyncArgs $syncArgs
    if ([bool]$pythonConfig.includeWheelCache) {
        $vendorDir = Join-Path $RuntimeDir "_vendor"
        $wheelsDir = Join-Path $vendorDir "wheels"
        New-Item -ItemType Directory -Force -Path $wheelsDir | Out-Null

        $requirementsPath = Join-Path $vendorDir "requirements.txt"
        $exportArgs = @("export", "--frozen", "--no-dev", "--no-emit-project") + $extraArgs + @("--format", "requirements-txt", "-o", $requirementsPath)
        & $uvExe @exportArgs
        if ($LASTEXITCODE -ne 0) {
            throw "导出 requirements 失败"
        }

        if (Test-UvPipWheelSupport -UvExe $uvExe) {
            $wheelArgs = @("pip", "wheel", "-r", $requirementsPath, "--wheel-dir", $wheelsDir)
            & $uvExe @wheelArgs
            if ($LASTEXITCODE -ne 0) {
                throw "构建 wheel 缓存失败"
            }
        }
        else {
            $notice = Join-Path $vendorDir "WHEEL_CACHE_DISABLED.txt"
            "当前 uv 版本不支持 'uv pip wheel'，已跳过 wheel 缓存生成。" | Set-Content -Path $notice -Encoding UTF8
            Write-Warning "当前 uv 版本不支持 pip wheel，已跳过 _vendor/wheels 生成。"
        }
    }

    if ([bool]$pythonConfig.includeUvBinary) {
        $runtimeToolsDir = Join-Path $RuntimeDir "tools"
        New-Item -ItemType Directory -Force -Path $runtimeToolsDir | Out-Null
        if (Test-Path (Join-Path $ProjectRoot "tools\uv.exe")) {
            Copy-Item -Path (Join-Path $ProjectRoot "tools\uv.exe") -Destination (Join-Path $runtimeToolsDir "uv.exe") -Force
        }
        else {
            Copy-Item -Path $uvExe -Destination (Join-Path $runtimeToolsDir "uv.exe") -Force
        }
    }

    foreach ($toolName in @("ffmpeg.exe", "ffprobe.exe")) {
        $toolPath = Join-Path $ProjectRoot "tools\$toolName"
        if (Test-Path $toolPath) {
            New-Item -ItemType Directory -Force -Path (Join-Path $RuntimeDir "tools") | Out-Null
            Copy-Item -Path $toolPath -Destination (Join-Path $RuntimeDir "tools\$toolName") -Force
        }
    }

    Copy-Item -Path (Join-Path $ProjectRoot "pyproject.toml") -Destination (Join-Path $RuntimeDir "pyproject.toml") -Force
    Copy-Item -Path (Join-Path $ProjectRoot "uv.lock") -Destination (Join-Path $RuntimeDir "uv.lock") -Force
}

function Get-Channel {
    param([string]$Flavor)
    if ($Flavor -eq "lite") { return "lite" }
    return "full"
}

function Get-PortableFileName {
    param([string]$ProfileName, [string]$Flavor)
    if ($Flavor -eq "lite") {
        return "AnchorFlux-Lite-Portable.zip"
    }
    if ($ProfileName -eq "full-hybrid") {
        return "AnchorFlux-Full-Hybrid-Portable.zip"
    }
    return "AnchorFlux-Full-Portable.zip"
}

function Get-InstallerFileName {
    param([string]$ProfileName, [string]$Flavor)
    if ($Flavor -eq "lite") {
        return "AnchorFlux-Lite-Setup.exe"
    }
    if ($ProfileName -eq "full-hybrid") {
        return "AnchorFlux-Full-Hybrid-Setup.exe"
    }
    return "AnchorFlux-Full-Setup.exe"
}

function Get-UpdateZipFileName {
    param([string]$ProfileName, [string]$Flavor)
    if ($Flavor -eq "lite") {
        return "anchorflux-lite-offline-update.zip"
    }
    if ($ProfileName -eq "full-hybrid") {
        return "anchorflux-full-hybrid-update.zip"
    }
    return "anchorflux-full-offline-update.zip"
}

$projectRoot = Get-ProjectRoot
$manifestPath = Join-Path $PSScriptRoot "profile-manifests\$Profile.json"
if (-not (Test-Path $manifestPath)) {
    throw "未找到 profile 清单: $manifestPath"
}
$manifest = Get-Content $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$flavor = [string]$manifest.flavor
if ([string]::IsNullOrWhiteSpace($flavor)) {
    $flavor = "full"
}
$channel = Get-Channel -Flavor $flavor
$uiMode = if (-not [string]::IsNullOrWhiteSpace([string]$manifest.uiMode)) {
    [string]$manifest.uiMode
}
else {
    "browser"
}
$runtimePolicy = if (-not [string]::IsNullOrWhiteSpace([string]$manifest.runtimePolicy)) {
    [string]$manifest.runtimePolicy
}
else {
    "offline"
}

$stagingRoot = Join-Path $projectRoot "dist\staging\$Profile\$Version"
$runtimeDir = Join-Path $stagingRoot "runtime"
$buildDir = Join-Path $stagingRoot "build"
$releaseRoot = Join-Path $projectRoot "dist\releases\$Profile\$Version"
$portableDir = Join-Path $releaseRoot "portable"
$installerDir = Join-Path $releaseRoot "installer"
$updateDir = Join-Path $releaseRoot "update"

Remove-Item -Recurse -Force -Path $stagingRoot -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $stagingRoot, $buildDir, $portableDir, $installerDir, $updateDir | Out-Null

Invoke-Checked -Label "构建前端" -Action {
    Build-Frontend -ProjectRoot $projectRoot -Manifest $manifest
}

$shellDir = $null
Invoke-Checked -Label "构建 Electron Shell (win-unpacked)" -Action {
    $script:shellDir = Build-ElectronShell -ProjectRoot $projectRoot
}

$stubExePath = Join-Path $buildDir "AnchorFlux.exe"
Invoke-Checked -Label "构建 Go Stub" -Action {
    Build-GoStub -ProjectRoot $projectRoot -OutputExePath $stubExePath -HideConsole:($flavor -eq "lite")
}

Invoke-Checked -Label "构建 Python 运行时" -Action {
    Build-PythonRuntime -ProjectRoot $projectRoot -Manifest $manifest -RuntimeDir $runtimeDir
}

$frontendDistDir = Join-Path $projectRoot "frontend\dist"
Invoke-Checked -Label "组装运行时目录" -Action {
    & (Join-Path $PSScriptRoot "assemble-runtime.ps1") `
        -ProjectRoot $projectRoot `
        -Profile $Profile `
        -Flavor $flavor `
        -Version $Version `
        -StagingRoot $stagingRoot `
        -RuntimeDir $runtimeDir `
        -ShellDir $shellDir `
        -FrontendDistDir $frontendDistDir `
        -StubExePath $stubExePath `
        -UiMode $uiMode `
        -RuntimePolicy $runtimePolicy `
        -ExcludePackages @([string[]]$manifest.python.excludePackages)
}

$appRoot = Join-Path $stagingRoot "AnchorFlux"
if (-not (Test-Path $appRoot)) {
    throw "运行时目录不存在: $appRoot"
}

$shouldBuildPortable = $Artifacts -in @("portable", "all")
$shouldBuildInstaller = $Artifacts -in @("installer", "all")
$shouldBuildUpdate = $Artifacts -in @("update", "all")

if ($shouldBuildPortable) {
    $portableFileName = Get-PortableFileName -ProfileName $Profile -Flavor $flavor
    $portableZip = Join-Path $portableDir $portableFileName
    Remove-Item -Path $portableZip -Force -ErrorAction SilentlyContinue
    Compress-Archive -Path $appRoot -DestinationPath $portableZip -Force
    Write-Host "[artifact] portable: $portableZip"
}

if ($shouldBuildInstaller) {
    $installerFileName = Get-InstallerFileName -ProfileName $Profile -Flavor $flavor
    $installerPath = Join-Path $installerDir $installerFileName
    $nsisScript = Join-Path $PSScriptRoot "nsis\anchorflux-installer.nsi"
    $makensis = Get-Command makensis -ErrorAction SilentlyContinue
    if ($null -eq $makensis) {
        throw "未检测到 makensis，请先安装 NSIS 并加入 PATH"
    }
    & $makensis.Source "/INPUTCHARSET" "UTF8" "/DSOURCE_DIR=$appRoot" "/DOUT_FILE=$installerPath" "/DAPP_PROFILE=$Profile" "/DAPP_VERSION=$Version" $nsisScript
    if ($LASTEXITCODE -ne 0) {
        throw "NSIS 安装包构建失败"
    }
    Write-Host "[artifact] installer: $installerPath"
}

if ($shouldBuildUpdate) {
    $updateZipName = Get-UpdateZipFileName -ProfileName $Profile -Flavor $flavor
    $updateZipPath = Join-Path $updateDir $updateZipName
    Remove-Item -Path $updateZipPath -Force -ErrorAction SilentlyContinue
    Compress-Archive -Path $appRoot -DestinationPath $updateZipPath -Force

    $manifestName = if ($channel -eq "lite") { "latest-lite.json" } else { "latest-full.json" }
    $manifestOutputPath = Join-Path $updateDir $manifestName
    $downloadBase = [string]$env:ANCHORFLUX_UPDATE_DOWNLOAD_BASE
    $downloadUrl = ""
    if (-not [string]::IsNullOrWhiteSpace($downloadBase)) {
        $downloadUrl = ($downloadBase.TrimEnd("/") + "/" + $updateZipName).Replace("\", "/")
    }
    & (Join-Path $PSScriptRoot "make-update-manifest.ps1") `
        -UpdateZipPath $updateZipPath `
        -ManifestPath $manifestOutputPath `
        -Version $Version `
        -Profile $Profile `
        -Channel $channel `
        -DownloadUrl $downloadUrl

    Write-Host "[artifact] update: $updateZipPath"
    Write-Host "[artifact] manifest: $manifestOutputPath"
}

Write-Host "[done] 打包完成: profile=$Profile version=$Version artifacts=$Artifacts"
