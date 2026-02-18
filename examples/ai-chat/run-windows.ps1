param(
    [string]$ProjectDir
)

$ErrorActionPreference = "Stop"

if (-not $ProjectDir) {
    $ProjectDir = & wsl.exe wslpath -w '~/cloudflare-agents/examples/ai-chat' 2>$null
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

# Generate a minimal Electron-only package.json (the full ai-chat one has
# monorepo file: references that cannot resolve outside the workspace).
Write-Host "==> Syncing files to $WorkDir ..." -ForegroundColor Cyan
$miniPkg = @'
{
  "name": "windows-agent",
  "private": true,
  "type": "module",
  "version": "0.0.0",
  "scripts": {
    "dev": "electron electron/main.ts",
    "start": "electron electron/main.ts"
  },
  "devDependencies": {
    "electron": "^40.0.0",
    "tsx": "^4.19.0"
  }
}
'@
# Write without BOM (PowerShell 5.1 -Encoding UTF8 adds BOM which breaks tsx)
$pkgPath = Join-Path $WorkDir "package.json"
[System.IO.File]::WriteAllText($pkgPath, $miniPkg, [System.Text.UTF8Encoding]::new($false))
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
    $env:NODE_OPTIONS = "--import tsx"
    & npm run dev
} finally {
    Pop-Location
}
