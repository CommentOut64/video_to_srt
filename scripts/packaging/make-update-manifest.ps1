param(
    [Parameter(Mandatory = $true)]
    [string]$UpdateZipPath,
    [Parameter(Mandatory = $true)]
    [string]$ManifestPath,
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [Parameter(Mandatory = $true)]
    [string]$Profile,
    [Parameter(Mandatory = $true)]
    [ValidateSet("lite", "full")]
    [string]$Channel,
    [Parameter(Mandatory = $false)]
    [string]$DownloadUrl = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path $UpdateZipPath)) {
    throw "更新包不存在: $UpdateZipPath"
}

$zipFile = Get-Item $UpdateZipPath
$hash = (Get-FileHash -Path $UpdateZipPath -Algorithm SHA256).Hash.ToLowerInvariant()
$resolvedDownloadUrl = $DownloadUrl
if ([string]::IsNullOrWhiteSpace($resolvedDownloadUrl)) {
    $resolvedDownloadUrl = $zipFile.Name
}

$payload = [ordered]@{
    version = $Version
    profile = $Profile
    channel = $Channel
    sha256 = $hash
    size = [int64]$zipFile.Length
    download_url = $resolvedDownloadUrl
    release_date = (Get-Date).ToUniversalTime().ToString("o")
}

$manifestDir = Split-Path -Path $ManifestPath -Parent
New-Item -ItemType Directory -Force -Path $manifestDir | Out-Null
$payload | ConvertTo-Json -Depth 5 | Set-Content -Path $ManifestPath -Encoding UTF8

Write-Host "[manifest] 已生成更新清单: $ManifestPath"
