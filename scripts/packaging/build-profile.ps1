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

function Resolve-EmbeddedPythonVersion {
    param([pscustomobject]$PythonConfig)
    $hasEmbeddedVersion = $false
    if ($null -ne $PythonConfig) {
        $hasEmbeddedVersion = $PythonConfig.PSObject.Properties.Name -contains "embeddedVersion"
    }
    if ($hasEmbeddedVersion -and -not [string]::IsNullOrWhiteSpace([string]$PythonConfig.embeddedVersion)) {
        return [string]$PythonConfig.embeddedVersion
    }
    $rawVersion = [string]$PythonConfig.version
    if ($rawVersion -match "^\d+\.\d+\.\d+$") {
        return $rawVersion
    }
    if ($rawVersion -match "^\d+\.\d+$") {
        if ($rawVersion -eq "3.10") { return "3.10.11" }
        if ($rawVersion -eq "3.11") { return "3.11.9" }
    }
    throw "无法解析嵌入式 Python 版本，请在 profile manifest 中设置 python.embeddedVersion"
}

function Get-EmbeddedPythonDownloadUrl {
    param([string]$Version)
    return "https://www.python.org/ftp/python/$Version/python-$Version-embed-amd64.zip"
}

function Download-EmbeddedPython {
    param(
        [string]$Version,
        [string]$RuntimeDir
    )
    $pythonRoot = Join-Path $RuntimeDir "tools\python"
    New-Item -ItemType Directory -Force -Path $pythonRoot | Out-Null

    $zipPath = Join-Path $RuntimeDir ("python-embed-" + $Version + ".zip")
    $downloadUrl = Get-EmbeddedPythonDownloadUrl -Version $Version

    Write-Host "[step] 下载嵌入式 Python: $downloadUrl"
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $pythonRoot -Force
    Remove-Item -Path $zipPath -Force -ErrorAction SilentlyContinue

    $pythonExe = Join-Path $pythonRoot "python.exe"
    if (-not (Test-Path $pythonExe)) {
        throw "嵌入式 Python 解压失败: $pythonExe"
    }
    return $pythonRoot
}

function Enable-EmbeddedPythonSitePackages {
    param([string]$PythonRoot)
    $pthFile = Get-ChildItem -Path $PythonRoot -Filter "python*._pth" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $pthFile) {
        throw "未找到嵌入式 Python ._pth 文件，无法启用 site-packages"
    }
    # V3.2.4+dev.20260306.01: 移除 BOM 并修正路径
    $content = [System.IO.File]::ReadAllText($pthFile.FullName, [System.Text.Encoding]::UTF8)
    $content = $content.TrimStart([char]0xFEFF)
    $lines = $content -split "`r?`n"

    $updated = @()
    $hasSitePackages = $false
    $hasImportSite = $false
    $hasDot = $false
    foreach ($line in $lines) {
        $trimmed = $line.Trim()
        if ($trimmed -eq "Lib\\site-packages") {
            $hasSitePackages = $true
        }
        if ($trimmed -eq ".") {
            $hasDot = $true
        }
        if ($trimmed -match "^\s*#?\s*import\s+site\s*$") {
            $updated += "import site"
            $hasImportSite = $true
            continue
        }
        $updated += $line
    }
    if (-not $hasDot) {
        $updated += "."
    }
    if (-not $hasSitePackages) {
        $updated += "Lib\\site-packages"
    }
    if (-not $hasImportSite) {
        $updated += "import site"
    }
    [System.IO.File]::WriteAllText($pthFile.FullName, ($updated -join "`r`n"), (New-Object System.Text.UTF8Encoding($false)))
}

function Get-ExtrasFromSyncArgs {
    param([object[]]$SyncArgs)
    $result = @()
    for ($i = 0; $i -lt $SyncArgs.Count; $i++) {
        if ($SyncArgs[$i] -eq "--extra" -and $i + 1 -lt $SyncArgs.Count) {
            $result += [string]$SyncArgs[$i + 1]
            $i++
        }
    }
    return $result
}

function Install-EmbeddedPythonDependencies {
    param(
        [string]$ProjectRoot,
        [string]$PythonRoot,
        [pscustomobject]$Manifest
    )
    $pythonExe = Join-Path $PythonRoot "python.exe"
    if (-not (Test-Path $pythonExe)) {
        throw "未找到嵌入式 Python: $pythonExe"
    }
    $sitePackages = Join-Path $PythonRoot "Lib\\site-packages"
    New-Item -ItemType Directory -Force -Path $sitePackages | Out-Null

    $pythonConfig = $Manifest.python
    $syncArgs = @([string[]]$pythonConfig.syncArgs)
    $extras = @(Get-ExtrasFromSyncArgs -SyncArgs $syncArgs)

    $packageSpec = "."
    if ($extras.Count -gt 0) {
        $packageSpec = ".[" + ($extras -join ",") + "]"
    }

    $pipEnv = @{
        "PIP_DISABLE_PIP_VERSION_CHECK" = "1"
        "PIP_NO_INPUT" = "1"
        "PYTHONIOENCODING" = "utf-8"
        "PYTHONUTF8" = "1"
    }

    $indexUrl = [string]$env:ANCHORFLUX_PIP_INDEX_URL
    if ([string]::IsNullOrWhiteSpace($indexUrl)) {
        $indexUrl = [string]$env:PIP_INDEX_URL
    }
    if ([string]::IsNullOrWhiteSpace($indexUrl)) {
        $indexUrl = "https://pypi.tuna.tsinghua.edu.cn/simple"
    }
    $pipEnv["PIP_INDEX_URL"] = $indexUrl

    $extraIndexUrl = [string]$env:ANCHORFLUX_PIP_EXTRA_INDEX_URL
    if ([string]::IsNullOrWhiteSpace($extraIndexUrl) -and ($extras -contains "full")) {
        $extraIndexUrl = "https://download.pytorch.org/whl/cu128"
    }
    if (-not [string]::IsNullOrWhiteSpace($extraIndexUrl)) {
        $pipEnv["PIP_EXTRA_INDEX_URL"] = $extraIndexUrl
    }

    $envBackup = @{}
    foreach ($entry in $pipEnv.GetEnumerator()) {
        $envBackup[$entry.Key] = [Environment]::GetEnvironmentVariable($entry.Key, "Process")
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
    }

    try {
        Write-Host "[step] 初始化嵌入式 Python pip"
        & $pythonExe -m ensurepip --upgrade
        if ($LASTEXITCODE -ne 0) {
            $getPipUrl = "https://bootstrap.pypa.io/get-pip.py"
            $getPipPath = Join-Path $env:TEMP ("anchorflux-get-pip-" + [guid]::NewGuid().ToString("N") + ".py")
            Write-Host "[step] ensurepip 不可用，改用 get-pip.py"
            Invoke-WebRequest -Uri $getPipUrl -OutFile $getPipPath
            & $pythonExe $getPipPath
            Remove-Item -Path $getPipPath -Force -ErrorAction SilentlyContinue
        }
        if ($LASTEXITCODE -ne 0) {
            throw "pip 初始化失败"
        }
        & $pythonExe -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) {
            throw "pip 升级失败"
        }

        Write-Host "[step] 安装构建后端(hatchling)"
        & $pythonExe -m pip install --no-warn-script-location --no-cache-dir hatchling
        if ($LASTEXITCODE -ne 0) {
            throw "hatchling 安装失败"
        }

        Write-Host "[step] 安装依赖到嵌入式 Python: $packageSpec"
        Push-Location $ProjectRoot
        try {
            & $pythonExe -m pip install --no-warn-script-location --no-cache-dir --no-build-isolation $packageSpec
            if ($LASTEXITCODE -ne 0) {
                throw "pip 安装依赖失败"
            }
        }
        finally {
            Pop-Location
        }

        & $pythonExe -m pip uninstall -y hatchling | Out-Host
    }
    finally {
        foreach ($entry in $envBackup.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
        }
    }
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

    # V3.2.4+dev.20260306.01: 使用项目 Python 而非系统 Python
    $projectPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    try {
        if (Test-Path $projectPython) {
            & $projectPython $tempIconScript
        }
        else {
            $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
            if ($null -ne $pythonCmd) {
                & $pythonCmd.Source $tempIconScript
            }
            else {
                $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
                if ($null -eq $pyLauncher) {
                    throw "未找到 python，无法生成 Electron 图标"
                }
                & $pyLauncher.Source -3 $tempIconScript
            }
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
    # V3.2.4+dev.20260306.02: 生产打包不再使用 uv，改为嵌入式 Python + pip 安装依赖
    Remove-Item -Recurse -Force -Path $RuntimeDir -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

    $pythonConfig = $Manifest.python
    $embeddedVersion = Resolve-EmbeddedPythonVersion -PythonConfig $pythonConfig
    $pythonRoot = Download-EmbeddedPython -Version $embeddedVersion -RuntimeDir $RuntimeDir
    Enable-EmbeddedPythonSitePackages -PythonRoot $pythonRoot
    Install-EmbeddedPythonDependencies -ProjectRoot $ProjectRoot -PythonRoot $pythonRoot -Manifest $Manifest

    $sitePackagesPath = Join-Path $pythonRoot "Lib\site-packages"
    if (Test-Path $sitePackagesPath) {
        Get-ChildItem -Path $sitePackagesPath -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Path $sitePackagesPath -Recurse -Directory -ErrorAction SilentlyContinue | Where-Object {
            $_.Name -in @("tests", "test")
        } | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }

    if ([bool]$pythonConfig.precompilePyc -and (Test-Path $sitePackagesPath)) {
        & (Join-Path $pythonRoot "python.exe") -m compileall -q $sitePackagesPath
    }

    foreach ($toolName in @("ffmpeg.exe", "ffprobe.exe")) {
        $toolPath = Join-Path $ProjectRoot "tools\$toolName"
        if (Test-Path $toolPath) {
            New-Item -ItemType Directory -Force -Path (Join-Path $RuntimeDir "tools") | Out-Null
            Copy-Item -Path $toolPath -Destination (Join-Path $RuntimeDir "tools\$toolName") -Force
        }
    }

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
