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

function Copy-Tree {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,
        [Parameter(Mandatory = $true)]
        [string]$Destination
    )

    if (-not (Test-Path $Source)) {
        throw "复制源目录不存在: $Source"
    }

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item -Path (Join-Path $Source "*") -Destination $Destination -Recurse -Force
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

function Get-FileSha256OrEmpty {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return ""
    }
    return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-Sha256FromText {
    param([string]$Text)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
        $hashBytes = $sha.ComputeHash($bytes)
        return ([System.BitConverter]::ToString($hashBytes)).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
}

function Get-DependencyInputDigests {
    param([string]$ProjectRoot)

    # 设计取舍：
    # - pyproject.toml 定义直接依赖与 extras，必须纳入指纹。
    # - uv.lock 代表已解析依赖集合；即便当前打包由 pip 执行，也要纳入失效输入，避免锁文件更新后复用旧缓存。
    $trackedFiles = @(
        "pyproject.toml",
        "uv.lock"
    )
    $digests = [ordered]@{}
    foreach ($relativePath in $trackedFiles) {
        $absolutePath = Join-Path $ProjectRoot $relativePath
        $digests[$relativePath] = Get-FileSha256OrEmpty -Path $absolutePath
    }
    return $digests
}

function Get-PipEnvironmentOverrides {
    param(
        [string]$ProjectRoot,
        [string[]]$Extras
    )
    $pipEnv = [ordered]@{
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
    if ([string]::IsNullOrWhiteSpace($extraIndexUrl) -and ($Extras -contains "full")) {
        $extraIndexUrl = "https://download.pytorch.org/whl/cu128"
    }
    if (-not [string]::IsNullOrWhiteSpace($extraIndexUrl)) {
        $pipEnv["PIP_EXTRA_INDEX_URL"] = $extraIndexUrl
    }

    $pipCacheDir = Join-Path $ProjectRoot "dist\cache\pip"
    New-Item -ItemType Directory -Force -Path $pipCacheDir | Out-Null
    $pipEnv["PIP_CACHE_DIR"] = $pipCacheDir

    return $pipEnv
}

function Get-DependencyFingerprint {
    param(
        [string]$ProjectRoot,
        [string]$EmbeddedVersion,
        [string[]]$Extras,
        [System.Collections.IDictionary]$PipEnvOverrides,
        [System.Collections.IDictionary]$DependencyInputDigests
    )

    $extrasSorted = @($Extras | Sort-Object -Unique)
    $payload = [ordered]@{
        schema = "runtime-cache-v2"
        embedded_version = $EmbeddedVersion
        extras = $extrasSorted
        dependency_inputs = $DependencyInputDigests
        pyproject_sha256 = [string]$DependencyInputDigests["pyproject.toml"]
        uv_lock_sha256 = [string]$DependencyInputDigests["uv.lock"]
        pip_index_url = [string]$PipEnvOverrides["PIP_INDEX_URL"]
        pip_extra_index_url = [string]$PipEnvOverrides["PIP_EXTRA_INDEX_URL"]
    }
    $payloadJson = $payload | ConvertTo-Json -Depth 8 -Compress
    return Get-Sha256FromText -Text $payloadJson
}

function Get-EmbeddedPythonCacheZipPath {
    param(
        [string]$ProjectRoot,
        [string]$Version
    )
    $cacheDir = Join-Path $ProjectRoot "dist\cache\python-embed"
    New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
    return (Join-Path $cacheDir ("python-" + $Version + "-embed-amd64.zip"))
}

function Download-EmbeddedPythonZipToCache {
    param(
        [string]$ProjectRoot,
        [string]$Version
    )
    $cachedZipPath = Get-EmbeddedPythonCacheZipPath -ProjectRoot $ProjectRoot -Version $Version
    if (Test-Path $cachedZipPath) {
        Write-Host "[cache] 命中嵌入式 Python 包缓存: $cachedZipPath"
        return $cachedZipPath
    }

    $downloadUrl = Get-EmbeddedPythonDownloadUrl -Version $Version
    $tmpZipPath = $cachedZipPath + ".tmp"
    Remove-Item -Path $tmpZipPath -Force -ErrorAction SilentlyContinue
    Write-Host "[step] 下载嵌入式 Python: $downloadUrl"
    Invoke-WebRequest -Uri $downloadUrl -OutFile $tmpZipPath
    Move-Item -Path $tmpZipPath -Destination $cachedZipPath -Force
    return $cachedZipPath
}

function Download-EmbeddedPython {
    param(
        [string]$ProjectRoot,
        [string]$Version,
        [string]$RuntimeDir
    )
    $pythonRoot = Join-Path $RuntimeDir "tools\python"
    New-Item -ItemType Directory -Force -Path $pythonRoot | Out-Null

    $cachedZipPath = Download-EmbeddedPythonZipToCache -ProjectRoot $ProjectRoot -Version $Version
    try {
        Expand-Archive -Path $cachedZipPath -DestinationPath $pythonRoot -Force
    }
    catch {
        Write-Host "[warn] 嵌入式 Python 缓存损坏，删除后重新下载: $cachedZipPath"
        Remove-Item -Path $cachedZipPath -Force -ErrorAction SilentlyContinue
        $cachedZipPath = Download-EmbeddedPythonZipToCache -ProjectRoot $ProjectRoot -Version $Version
        Expand-Archive -Path $cachedZipPath -DestinationPath $pythonRoot -Force
    }

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

function Get-ExtrasFromPythonConfig {
    param([pscustomobject]$PythonConfig)

    $hasExtras = $PythonConfig.PSObject.Properties.Name -contains "extras"
    if (-not $hasExtras) {
        throw "profile manifest 缺少 python.extras 字段，请更新清单语义"
    }

    $extrasValue = $PythonConfig.extras
    if ($null -eq $extrasValue) {
        return @()
    }
    if ($extrasValue -isnot [System.Collections.IEnumerable] -or $extrasValue -is [string]) {
        throw "profile manifest 的 python.extras 必须是数组"
    }

    $extras = @()
    foreach ($extra in $extrasValue) {
        $normalized = [string]$extra
        if ([string]::IsNullOrWhiteSpace($normalized)) {
            continue
        }
        $extras += $normalized.Trim()
    }
    return $extras
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
    $extras = @(Get-ExtrasFromPythonConfig -PythonConfig $pythonConfig)

    $pipEnv = Get-PipEnvironmentOverrides -ProjectRoot $ProjectRoot -Extras $extras

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
        & $pythonExe -m pip install --no-warn-script-location hatchling
        if ($LASTEXITCODE -ne 0) {
            throw "hatchling 安装失败"
        }

        # 设计取舍：
        # - 基线依赖（Lite/Full 共用）先安装，确保两种 profile 命中同一套 pip 缓存。
        # - Full 再按 extras 做增量安装，避免每次都按 Full 全量下载。
        Push-Location $ProjectRoot
        try {
            Write-Host "[step] 安装 Lite/Full 共用基线依赖: ."
            & $pythonExe -m pip install --no-warn-script-location --no-build-isolation .
            if ($LASTEXITCODE -ne 0) {
                throw "pip 安装基线依赖失败"
            }

            if ($extras.Count -gt 0) {
                $extraPackageSpec = ".[" + ($extras -join ",") + "]"
                Write-Host "[step] 安装 Full 增量依赖: $extraPackageSpec"
                & $pythonExe -m pip install --no-warn-script-location --no-build-isolation $extraPackageSpec
                if ($LASTEXITCODE -ne 0) {
                    throw "pip 安装增量依赖失败"
                }
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
    $extras = @(Get-ExtrasFromPythonConfig -PythonConfig $pythonConfig)
    $embeddedVersion = Resolve-EmbeddedPythonVersion -PythonConfig $pythonConfig
    $pipEnvOverrides = Get-PipEnvironmentOverrides -ProjectRoot $ProjectRoot -Extras $extras
    $dependencyInputDigests = Get-DependencyInputDigests -ProjectRoot $ProjectRoot
    $dependencyFingerprint = Get-DependencyFingerprint `
        -ProjectRoot $ProjectRoot `
        -EmbeddedVersion $embeddedVersion `
        -Extras $extras `
        -PipEnvOverrides $pipEnvOverrides `
        -DependencyInputDigests $dependencyInputDigests

    $runtimeCacheRoot = Join-Path $ProjectRoot "dist\cache\python-runtime"
    $runtimeCacheDir = Join-Path $runtimeCacheRoot $dependencyFingerprint
    if (Test-Path $runtimeCacheDir) {
        Write-Host "[cache] 命中 Python 运行时缓存: $runtimeCacheDir"
        Copy-Tree -Source $runtimeCacheDir -Destination $RuntimeDir
        return
    }
    Write-Host "[cache] 未命中 Python 运行时缓存，执行全量构建: $runtimeCacheDir"

    $pythonRoot = Download-EmbeddedPython -ProjectRoot $ProjectRoot -Version $embeddedVersion -RuntimeDir $RuntimeDir
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

    New-Item -ItemType Directory -Force -Path $runtimeCacheRoot | Out-Null
    $tmpRuntimeCacheDir = Join-Path $runtimeCacheRoot ("tmp-" + [guid]::NewGuid().ToString("N"))
    Copy-Tree -Source $RuntimeDir -Destination $tmpRuntimeCacheDir
    if (Test-Path $runtimeCacheDir) {
        Remove-Item -Recurse -Force -Path $runtimeCacheDir -ErrorAction SilentlyContinue
    }
    Move-Item -Path $tmpRuntimeCacheDir -Destination $runtimeCacheDir -Force
    $runtimeCacheMetadata = [ordered]@{
        fingerprint = $dependencyFingerprint
        generated_at = (Get-Date).ToUniversalTime().ToString("o")
        embedded_version = $embeddedVersion
        extras = @($extras | Sort-Object -Unique)
        dependency_inputs = $dependencyInputDigests
        pyproject_sha256 = [string]$dependencyInputDigests["pyproject.toml"]
        uv_lock_sha256 = [string]$dependencyInputDigests["uv.lock"]
        pip_index_url = [string]$pipEnvOverrides["PIP_INDEX_URL"]
        pip_extra_index_url = [string]$pipEnvOverrides["PIP_EXTRA_INDEX_URL"]
    }
    $runtimeCacheMetadata | ConvertTo-Json -Depth 8 | Set-Content -Path (Join-Path $runtimeCacheDir "runtime-cache-meta.json") -Encoding UTF8
    Write-Host "[cache] 已写入 Python 运行时缓存: $runtimeCacheDir"

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
