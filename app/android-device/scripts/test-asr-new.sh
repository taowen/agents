#!/usr/bin/env bash
#
# ASR Profiling Test — builds, installs, runs batch transcription with
# per-operation breakdown to identify bottlenecks.
#
# Usage:
#   cd app/android-device && bash scripts/test-asr-new.sh
#

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
PKG="ai.connct_screen.rn"
ACTION="${PKG}.VOICE_DEBUG"
MODEL_DIR="/data/local/tmp/qwen3-asr-0.6b"
WAV_PATH="/data/local/tmp/jfk.wav"
EXPECTED_PHRASE="ask not what your country can do for you"

TIMEOUT_LOAD=60
TIMEOUT_WAV=120

# ── Helpers ─────────────────────────────────────────────────────────────────
PASS=0
FAIL=0
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ANDROID_DIR="$(cd "$SCRIPT_DIR/../android" && pwd)"

pass() { PASS=$((PASS + 1)); echo "[✓] $1"; }
fail() { FAIL=$((FAIL + 1)); echo "[✗] $1"; }
info() { echo "    $1"; }

dump_log() {
    if [ -n "${LOGCAT_FILE:-}" ] && [ -f "$LOGCAT_FILE" ]; then
        echo ""
        echo "--- logcat dump ---"
        cat "$LOGCAT_FILE"
        echo "--- end logcat ---"
    fi
}

cleanup() { rm -f "${LOGCAT_FILE:-}"; }
trap cleanup EXIT

# ── Preflight ───────────────────────────────────────────────────────────────
echo "=== ASR Profiling Test ==="
echo ""

if ! command -v adb &>/dev/null; then echo "ERROR: adb not found"; exit 1; fi

DEVICE_COUNT=$(adb devices | grep -cw 'device' || true)
if [ "$DEVICE_COUNT" -lt 1 ]; then echo "ERROR: No device connected"; exit 1; fi
pass "Device connected"

if ! adb shell "ls ${MODEL_DIR}/" &>/dev/null; then
    echo "ERROR: Push model first: adb push qwen3-asr-0.6b ${MODEL_DIR}"; exit 1
fi

if ! adb shell "ls ${WAV_PATH}" &>/dev/null; then
    echo "ERROR: Push WAV first: adb push jfk.wav ${WAV_PATH}"; exit 1
fi

echo ""

# ── Build & Install ─────────────────────────────────────────────────────────
echo "--- Build & Install ---"

if (cd "$ANDROID_DIR" && ./gradlew assembleDebug) >/dev/null 2>&1; then
    pass "Build succeeded"
else
    fail "Build failed"; exit 1
fi

APK="$ANDROID_DIR/app/build/outputs/apk/debug/app-debug.apk"
if [ ! -f "$APK" ]; then fail "APK not found"; exit 1; fi

if adb install -r "$APK" >/dev/null 2>&1; then
    pass "Install succeeded"
else
    fail "Install failed"; exit 1
fi

adb shell am start -n "${PKG}/.MainActivity" >/dev/null 2>&1
sleep 2
echo ""

# ── Load Model ──────────────────────────────────────────────────────────────
echo "--- Load Model ---"

adb logcat -c
adb shell "am broadcast -a ${ACTION} -p ${PKG} --es cmd load_model --es path ${MODEL_DIR}" >/dev/null 2>&1

ELAPSED=0
LOAD_OK=false
while [ $ELAPSED -lt $TIMEOUT_LOAD ]; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    if adb logcat -d -s VoiceDebug 2>/dev/null | grep -q "load_model result="; then
        LOAD_OK=true; break
    fi
done

LOGCAT_FILE=$(mktemp)
adb logcat -d >"$LOGCAT_FILE" 2>/dev/null

if [ "$LOAD_OK" = false ]; then fail "Model loading timed out"; dump_log; exit 1; fi

if grep -q "load_model result=true" "$LOGCAT_FILE"; then
    pass "Model loaded"
else
    fail "Model load failed"; dump_log; exit 1
fi
echo ""

# ── Batch Transcription ────────────────────────────────────────────────────
echo "--- Batch Transcription (jfk.wav) ---"

adb logcat -c
adb shell "am broadcast -a ${ACTION} -p ${PKG} --es cmd test_wav --es path ${WAV_PATH}" >/dev/null 2>&1

ELAPSED=0
WAV_DONE=false
while [ $ELAPSED -lt $TIMEOUT_WAV ]; do
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if adb logcat -d -s QwenASR_JNI 2>/dev/null | grep -q "nativeTestWav: done"; then
        WAV_DONE=true; break
    fi
done

rm -f "$LOGCAT_FILE"
adb logcat -d >"$LOGCAT_FILE" 2>/dev/null

if [ "$WAV_DONE" = false ]; then fail "Batch timed out (${TIMEOUT_WAV}s)"; dump_log; exit 1; fi

pass "Batch inference completed"

# Check correctness
RESULT_LINE=$(grep "nativeTestWav: result = " "$LOGCAT_FILE" || true)
if [ -z "$RESULT_LINE" ]; then
    fail "No result found"; dump_log
else
    RESULT_LOWER=$(echo "$RESULT_LINE" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z ]//g')
    if echo "$RESULT_LOWER" | grep -q "$EXPECTED_PHRASE"; then
        RESULT_TEXT=$(echo "$RESULT_LINE" | sed 's/.*nativeTestWav: result = //')
        pass "Correct: \"${RESULT_TEXT}\""
    else
        RESULT_TEXT=$(echo "$RESULT_LINE" | sed 's/.*nativeTestWav: result = //')
        fail "Mismatch: \"${RESULT_TEXT}\""
    fi
fi

echo ""

# ── Performance Report ──────────────────────────────────────────────────────
echo "=== Performance Report ==="
echo ""

# Phase-level timing
BATCH_MEL_MS=$(grep -oP 'Mel: \d+ frames \(\K\d+' "$LOGCAT_FILE" || echo "")
BATCH_ENC_MS=$(grep -oP 'Encoder: \d+ tokens \(\K\d+' "$LOGCAT_FILE" || echo "")
BATCH_ENC_TOKENS=$(grep -oP 'Encoder: \K\d+' "$LOGCAT_FILE" || echo "")
BATCH_PRE_MS=$(grep -oP 'Prefill: \d+ tokens \(\K\d+' "$LOGCAT_FILE" || echo "")
BATCH_PRE_TOKENS=$(grep -oP 'Prefill: \K\d+' "$LOGCAT_FILE" || echo "")
BATCH_DEC_MS=$(grep -oP 'Decode: \d+ tokens \(\K\d+' "$LOGCAT_FILE" || echo "")
BATCH_DEC_TOKENS=$(grep -oP 'Decode: \K\d+' "$LOGCAT_FILE" || echo "")
BATCH_DEC_PER=$(grep -oP 'Decode: \d+ tokens \(\d+ ms, \K[0-9.]+' "$LOGCAT_FILE" || echo "")

if [ -n "$BATCH_ENC_MS" ] && [ -n "$BATCH_PRE_MS" ] && [ -n "$BATCH_DEC_MS" ]; then
    BATCH_TOTAL_MS=$((${BATCH_MEL_MS:-0} + BATCH_ENC_MS + BATCH_PRE_MS + BATCH_DEC_MS))
    printf "Mel:      %5s ms\n" "${BATCH_MEL_MS:-?}"
    printf "Encoder:  %5s ms  (%s tokens)\n" "$BATCH_ENC_MS" "${BATCH_ENC_TOKENS:-?}"
    printf "Prefill:  %5s ms  (%s tokens)\n" "$BATCH_PRE_MS" "${BATCH_PRE_TOKENS:-?}"
    printf "Decode:   %5s ms  (%s tokens, %s ms/tok)\n" "$BATCH_DEC_MS" "${BATCH_DEC_TOKENS:-?}" "${BATCH_DEC_PER:-?}"
    printf "Total:    %5s ms\n" "$BATCH_TOTAL_MS"
else
    echo "(could not parse phase timing from logcat)"
fi

echo ""

# Per-operation breakdown
ENC_BREAKDOWN=$(grep -oP 'Encoder breakdown: \K.*' "$LOGCAT_FILE" || echo "")
PRE_BREAKDOWN=$(grep -oP 'Prefill breakdown: \K.*' "$LOGCAT_FILE" || echo "")
DEC_BREAKDOWN=$(grep -oP 'Decode breakdown: \K.*' "$LOGCAT_FILE" || echo "")

if [ -n "$ENC_BREAKDOWN" ]; then
    echo "Encoder breakdown:  $ENC_BREAKDOWN"
fi
if [ -n "$PRE_BREAKDOWN" ]; then
    echo "Prefill breakdown:  $PRE_BREAKDOWN"
fi
if [ -n "$DEC_BREAKDOWN" ]; then
    echo "Decode breakdown:   $DEC_BREAKDOWN"
fi

echo ""

# ── Summary ─────────────────────────────────────────────────────────────────
if [ "$FAIL" -eq 0 ]; then
    echo "=== PASSED (${PASS} checks) ==="
    exit 0
else
    echo "=== ${FAIL} FAILED, ${PASS} passed ==="
    exit 1
fi
