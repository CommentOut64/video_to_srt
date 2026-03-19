param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,
    [Parameter(Mandatory = $true)]
    [string]$Profile,
    [Parameter(Mandatory = $false)]
    [string]$Flavor = "full",
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [Parameter(Mandatory = $true)]
    [string]$StagingRoot,
    [Parameter(Mandatory = $true)]
    [string]$RuntimeDir,
    [Parameter(Mandatory = $true)]
    [string]$ShellDir,
    [Parameter(Mandatory = $true)]
    [string]$FrontendDistDir,
    [Parameter(Mandatory = $true)]
    [string]$StubExePath,
    [Parameter(Mandatory = $false)]
    [string]$UiMode = "browser",
    [Parameter(Mandatory = $false)]
    [string]$RuntimePolicy = "offline",
    [Parameter(Mandatory = $false)]
    [string[]]$ExcludePackages = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Copy-Tree {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,
        [Parameter(Mandatory = $true)]
        [string]$Destination
    )
    if (-not (Test-Path $Source)) {
        throw "复制源不存在: $Source"
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item -Path (Join-Path $Source "*") -Destination $Destination -Recurse -Force
}

function Remove-DirectoryIfExists {
    param([string]$PathToRemove)
    if (Test-Path $PathToRemove) {
        Remove-Item -Path $PathToRemove -Recurse -Force
    }
}

function Remove-LiteExcludedPackages {
    param(
        [string]$SitePackagesPath,
        [string[]]$Names
    )
    if (-not (Test-Path $SitePackagesPath)) {
        return
    }
    foreach ($name in $Names) {
        $candidates = @(
            $name,
            $name.Replace("-", "_"),
            $name.Replace("-", "")
        ) | Select-Object -Unique
        foreach ($candidate in $candidates) {
            Get-ChildItem -Path $SitePackagesPath -Filter "$candidate*" -Force -ErrorAction SilentlyContinue | ForEach-Object {
                Remove-Item -Path $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

function Write-PackagedEnvFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetPath,
        [Parameter(Mandatory = $true)]
        [string]$VersionValue,
        [Parameter(Mandatory = $true)]
        [string]$ProfileValue,
        [Parameter(Mandatory = $true)]
        [string]$FlavorValue,
        [Parameter(Mandatory = $true)]
        [string]$UiModeValue,
        [Parameter(Mandatory = $true)]
        [string]$RuntimePolicyValue
    )

    $isLiteFlavor = $FlavorValue.ToLowerInvariant() -eq "lite"
    $liteFlag = if ($isLiteFlavor) { "true" } else { "false" }

    $lines = @(
        "# 打包产物运行配置（由打包脚本生成）",
        "DEV_MODE=false",
        "ANCHORFLUX_BUILD_VERSION=$VersionValue",
        "ANCHORFLUX_PROFILE=$ProfileValue",
        "ANCHORFLUX_FLAVOR=$FlavorValue",
        "ANCHORFLUX_LITE=$liteFlag",
        "ANCHORFLUX_UI_MODE=$UiModeValue",
        "ANCHORFLUX_RUNTIME_POLICY=$RuntimePolicyValue",
        "USE_HF_MIRROR=true"
    )
    $content = $lines -join [Environment]::NewLine
    Set-Content -Path $TargetPath -Value $content -Encoding UTF8
}

$appRoot = Join-Path $StagingRoot "AnchorFlux"
Remove-DirectoryIfExists -PathToRemove $appRoot
New-Item -ItemType Directory -Force -Path $appRoot | Out-Null

Write-Host "[assemble] 组装运行时目录: $appRoot"

# 1) 入口与核心代码
Copy-Item -Path $StubExePath -Destination (Join-Path $appRoot "AnchorFlux.exe") -Force
Copy-Tree -Source (Join-Path $ProjectRoot "launcher") -Destination (Join-Path $appRoot "launcher")
Copy-Tree -Source (Join-Path $ProjectRoot "backend") -Destination (Join-Path $appRoot "backend")

# Lite 产物不携带 backend/models（该目录用于 Full 重依赖模型）。
if ($Flavor.ToLowerInvariant() -eq "lite") {
    Remove-DirectoryIfExists -PathToRemove (Join-Path $appRoot "backend\models")
}

# 2) 前端产物
$frontendTarget = Join-Path $appRoot "frontend\dist"
New-Item -ItemType Directory -Force -Path $frontendTarget | Out-Null
Copy-Item -Path (Join-Path $FrontendDistDir "*") -Destination $frontendTarget -Recurse -Force

# 3) Electron Shell
$shellTarget = Join-Path $appRoot "core\shell"
Copy-Tree -Source $ShellDir -Destination $shellTarget

# 4) 运行时（嵌入式 Python + 工具）
$toolsSource = Join-Path $RuntimeDir "tools"
if (Test-Path $toolsSource) {
    Copy-Tree -Source $toolsSource -Destination (Join-Path $appRoot "tools")
}

# 5) 常用配置文件
foreach ($optionalFile in @("user_config.json", "model_runtime_config.json")) {
    $sourcePath = Join-Path $ProjectRoot $optionalFile
    if (Test-Path $sourcePath) {
        Copy-Item -Path $sourcePath -Destination (Join-Path $appRoot $optionalFile) -Force
    }
}
Write-PackagedEnvFile `
    -TargetPath (Join-Path $appRoot ".env") `
    -VersionValue $Version `
    -ProfileValue $Profile `
    -FlavorValue $Flavor `
    -UiModeValue $UiMode `
    -RuntimePolicyValue $RuntimePolicy

# 6) 清理缓存与无关目录
Get-ChildItem -Path $appRoot -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item -Path $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
}
Get-ChildItem -Path $appRoot -Recurse -Directory -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -in @("tests", "test", ".pytest_cache")
} | ForEach-Object {
    Remove-Item -Path $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
}

$sitePackagesPath = Join-Path $appRoot "tools\python\Lib\site-packages"
if ($ExcludePackages.Count -gt 0) {
    Remove-LiteExcludedPackages -SitePackagesPath $sitePackagesPath -Names $ExcludePackages
}

Write-Host "[assemble] 运行时目录组装完成: $appRoot"
