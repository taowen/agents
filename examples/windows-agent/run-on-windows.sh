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

powershell.exe -ExecutionPolicy Bypass -File "$WIN_PS1" -ProjectDir "$WIN_DIR"
