#!/usr/bin/env bash
# One-click: build deps → deploy ai-chat → launch Electron on Windows.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================"
echo "  Windows Agent — Build & Deploy & Run"
echo "========================================"

# ---- 1. Build just-bash (ai-chat depends on its dist/) ----
echo ""
echo "==> [1/4] Building just-bash ..."
npm run build --prefix "$REPO_ROOT/packages/just-bash"

# ---- 2. Build workspace packages (agents, @cloudflare/ai-chat, etc.) ----
echo ""
echo "==> [2/4] Building workspace packages ..."
npm run build --prefix "$REPO_ROOT"

# ---- 3. Deploy ai-chat to Cloudflare ----
echo ""
echo "==> [3/4] Deploying ai-chat to Cloudflare ..."
npm run deploy --prefix "$REPO_ROOT/examples/ai-chat"

# ---- 4. Launch Electron on Windows ----
echo ""
echo "==> [4/4] Launching Electron on Windows ..."
WIN_DIR="$(wslpath -w "$SCRIPT_DIR")"
WIN_PS1="$(wslpath -w "$SCRIPT_DIR/run-windows.ps1")"

# Resolve Windows %TEMP% as a WSL path for PID file cleanup
WIN_TEMP="$(cmd.exe /c "echo %TEMP%" 2>/dev/null | tr -d '\r')"
WIN_PID_FILE="$(wslpath "$WIN_TEMP")/windows-agent-build.pid"

powershell.exe -ExecutionPolicy Bypass -File "$WIN_PS1" -ProjectDir "$WIN_DIR" &
PS_PID=$!

cleanup() {
    echo ""
    echo "Stopping Electron..."
    # Kill the Windows process tree via taskkill (WSL kill alone won't reach children)
    if [ -f "$WIN_PID_FILE" ]; then
        WIN_PID=$(cat "$WIN_PID_FILE")
        taskkill.exe /PID "$WIN_PID" /T /F 2>/dev/null
        rm -f "$WIN_PID_FILE"
    fi
    kill $PS_PID 2>/dev/null
    wait $PS_PID 2>/dev/null
}
trap cleanup EXIT INT TERM

wait $PS_PID
