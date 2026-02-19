param(
    [string]$ProjectDir,
    [string]$JustBashTarball,
    [Parameter(ValueFromRemainingArguments)]
    [string[]]$AgentArgs
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

$WorkDir = Join-Path $env:TEMP "windows-agent-standalone"

if (-not (Test-Path $WorkDir)) {
    New-Item -ItemType Directory -Path $WorkDir | Out-Null
}

# Copy TypeScript sources
Write-Host "==> Syncing files to $WorkDir ..." -ForegroundColor Cyan
$ElectronDest = Join-Path $WorkDir "electron"

if (-not (Test-Path $ElectronDest)) {
    New-Item -ItemType Directory -Path $ElectronDest | Out-Null
}
Copy-Item (Join-Path $ProjectDir "electron\*") -Destination $ElectronDest -Recurse -Force

# Copy src/shared directory (agent-loop, types, aliases)
$SharedDest = Join-Path $WorkDir "src\shared"
if (-not (Test-Path $SharedDest)) {
    New-Item -ItemType Directory -Path $SharedDest -Force | Out-Null
}
Copy-Item (Join-Path $ProjectDir "src\shared\*") -Destination $SharedDest -Recurse -Force

# Determine just-bash dependency value
$justBashDep = '"^2.10.0"'
if ($JustBashTarball -and (Test-Path $JustBashTarball)) {
    # Copy tarball to WorkDir to avoid UNC path / backslash escaping issues in JSON
    $localTarball = Join-Path $WorkDir (Split-Path $JustBashTarball -Leaf)
    Copy-Item $JustBashTarball -Destination $localTarball -Force
    $justBashDep = """file:$(Split-Path $JustBashTarball -Leaf)"""
    Write-Host "==> Using local just-bash tarball: $localTarball" -ForegroundColor Yellow
}

# Generate minimal package.json
$miniPkg = @"
{
  "name": "windows-agent-standalone",
  "private": true,
  "type": "module",
  "dependencies": {
    "ai": "^6.0.86",
    "@ai-sdk/openai-compatible": "^2.0.24",
    "@ai-sdk/google": "^3.0.29",
    "zod": "^4.3.6",
    "just-bash": $justBashDep,
    "tsx": "^4.19.0"
  }
}
"@
# Write without BOM (PowerShell 5.1 -Encoding UTF8 adds BOM which breaks tsx)
$pkgPath = Join-Path $WorkDir "package.json"
[System.IO.File]::WriteAllText($pkgPath, $miniPkg, [System.Text.UTF8Encoding]::new($false))

# Load .env.standalone from source directory if env vars not already set
$envFile = Join-Path $ProjectDir ".env.standalone"
if ((Test-Path $envFile) -and -not $env:LLM_API_KEY) {
    Write-Host "==> Loading $envFile" -ForegroundColor Cyan
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+?)\s*=\s*(.+?)\s*$') {
            [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
        }
    }
}

# Write PID so the WSL wrapper script can taskkill the whole process tree
$pidFile = Join-Path $env:TEMP "windows-agent-standalone.pid"
[System.IO.File]::WriteAllText($pidFile, "$PID")

Push-Location $WorkDir
try {
    Write-Host "==> npm install (in $WorkDir) ..." -ForegroundColor Cyan
    & npm install
    if ($LASTEXITCODE -ne 0) { throw "npm install failed" }

    Write-Host "==> Running standalone agent ..." -ForegroundColor Green
    & npx tsx electron/standalone.ts @AgentArgs
} finally {
    Pop-Location
    Remove-Item $pidFile -ErrorAction SilentlyContinue
}
