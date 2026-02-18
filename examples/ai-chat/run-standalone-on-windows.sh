#!/usr/bin/env bash
# Build just-bash, pack it, then launch the standalone Windows agent via PowerShell.
# Usage: bash examples/ai-chat/run-standalone-on-windows.sh "Take a screenshot"
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================"
echo "  Windows Agent — Standalone Mode"
echo "========================================"

# ---- Load .env.standalone if present ----
ENV_FILE="$SCRIPT_DIR/.env.standalone"
if [ -f "$ENV_FILE" ]; then
    echo ""
    echo "==> Loading $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
fi

# ---- 1. Build just-bash ----
echo ""
echo "==> [1/3] Building just-bash ..."
npm run build --prefix "$REPO_ROOT/packages/just-bash"

# ---- 2. Pack just-bash tarball for Windows-side install ----
echo ""
echo "==> [2/3] Packing just-bash ..."
TARBALL=$(cd "$REPO_ROOT/packages/just-bash" && npm pack --pack-destination "$SCRIPT_DIR" 2>/dev/null | tail -1)
TARBALL_PATH="$SCRIPT_DIR/$TARBALL"
WIN_TARBALL="$(wslpath -w "$TARBALL_PATH")"
echo "    Tarball: $WIN_TARBALL"

# ---- 3. Launch standalone agent on Windows ----
echo ""
echo "==> [3/3] Launching standalone agent on Windows ..."
WIN_DIR="$(wslpath -w "$SCRIPT_DIR")"
WIN_PS1="$(wslpath -w "$SCRIPT_DIR/run-standalone.ps1")"

# Convert DEBUG_DIR to Windows path if set
DEBUG_ARGS=()
if [ -n "$DEBUG_DIR" ]; then
  mkdir -p "$DEBUG_DIR"
  WIN_DEBUG_DIR="$(wslpath -w "$DEBUG_DIR")"
  DEBUG_ARGS+=("-DebugDir" "$WIN_DEBUG_DIR")
fi

powershell.exe -ExecutionPolicy Bypass -File "$WIN_PS1" \
  -ProjectDir "$WIN_DIR" \
  -JustBashTarball "$WIN_TARBALL" \
  "${DEBUG_ARGS[@]}" \
  "$@"

# Clean up tarball
rm -f "$TARBALL_PATH"
