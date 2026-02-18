param(
    [string]$ProjectDir
)

$ErrorActionPreference = "Stop"

if (-not $ProjectDir) {
    $ProjectDir = & wsl.exe wslpath -w '~/cloudflare-agents/examples/windows-agent' 2>$null
    if (-not $ProjectDir) {
        Write-Error "Cannot determine project directory. Pass -ProjectDir <path>."
        exit 1
    }
    $ProjectDir = $ProjectDir.Trim()
}

Write-Host "==> Source directory: $ProjectDir" -ForegroundColor Cyan

# Use a Windows-local temp directory to avoid UNC path / monorepo workspace issues
$WorkDir = Join-Path $env:TEMP "windows-agent-build"

if (-not (Test-Path $WorkDir)) {
    New-Item -ItemType Directory -Path $WorkDir | Out-Null
}

# Copy project files (package.json + electron/) to Windows temp dir
Write-Host "==> Syncing files to $WorkDir ..." -ForegroundColor Cyan
Copy-Item (Join-Path $ProjectDir "package.json") -Destination $WorkDir -Force
$ElectronDest = Join-Path $WorkDir "electron"
if (-not (Test-Path $ElectronDest)) {
    New-Item -ItemType Directory -Path $ElectronDest | Out-Null
}
Copy-Item (Join-Path (Join-Path $ProjectDir "electron") "*") -Destination $ElectronDest -Recurse -Force

Push-Location $WorkDir
try {
    Write-Host "==> npm install (in $WorkDir) ..." -ForegroundColor Cyan
    & npm install
    if ($LASTEXITCODE -ne 0) { throw "npm install failed" }

    Write-Host "==> Starting Electron app ..." -ForegroundColor Green
    & npm run dev
} finally {
    Pop-Location
}
