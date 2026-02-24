#!/usr/bin/env bash
#
# ASR Profiling Test — builds, installs, runs batch + streaming transcription
# with per-operation breakdown to identify bottlenecks.
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
TIMEOUT_STREAM=120

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

# ── Batch Performance Report ──────────────────────────────────────────────
echo "=== Batch Performance ==="
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

# ── Streaming Transcription ────────────────────────────────────────────────
echo "--- Streaming Transcription (jfk.wav) ---"

adb logcat -c
adb shell "am broadcast -a ${ACTION} -p ${PKG} --es cmd test_wav_stream --es path ${WAV_PATH}" >/dev/null 2>&1

ELAPSED=0
STREAM_DONE=false
while [ $ELAPSED -lt $TIMEOUT_STREAM ]; do
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if adb logcat -d -s QwenASR_JNI 2>/dev/null | grep -q "nativeTestWavStream: done"; then
        STREAM_DONE=true; break
    fi
done

rm -f "$LOGCAT_FILE"
adb logcat -d >"$LOGCAT_FILE" 2>/dev/null

if [ "$STREAM_DONE" = false ]; then fail "Stream timed out (${TIMEOUT_STREAM}s)"; dump_log; exit 1; fi

pass "Stream inference completed"

# Check correctness
STREAM_RESULT_LINE=$(grep "nativeTestWavStream: result = " "$LOGCAT_FILE" || true)
if [ -z "$STREAM_RESULT_LINE" ]; then
    fail "No stream result found"; dump_log
else
    STREAM_RESULT_LOWER=$(echo "$STREAM_RESULT_LINE" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z ]//g')
    if echo "$STREAM_RESULT_LOWER" | grep -q "$EXPECTED_PHRASE"; then
        STREAM_RESULT_TEXT=$(echo "$STREAM_RESULT_LINE" | sed 's/.*nativeTestWavStream: result = //')
        pass "Correct: \"${STREAM_RESULT_TEXT}\""
    else
        STREAM_RESULT_TEXT=$(echo "$STREAM_RESULT_LINE" | sed 's/.*nativeTestWavStream: result = //')
        fail "Mismatch: \"${STREAM_RESULT_TEXT}\""
    fi
fi

echo ""

# ── Streaming Performance Report ──────────────────────────────────────────
echo "=== Streaming Performance ==="
echo ""

# Extract per-chunk timings from verbose log lines:
#   "  Encoder: N tokens from 0.0-X.X s (cached windows=W, partial=P.P s, MMM ms)"
#   "  Stem cache: H/T chunks cached, R recomputed"
#   "  Prefill: N tokens (P prefix, reused R) (MMM ms)"
#   "  Decode: N tokens (MMM ms, D.D ms/token...)"
STREAM_LOG=$(grep -E 'QwenASR' "$LOGCAT_FILE" || true)

# Per-chunk table
CHUNK_NUM=0
echo "Chunk | Encoder ms | Stem cached | Prefill ms | Decode ms | Decode toks"
echo "------+------------+-------------+------------+-----------+------------"

# Parse encoder lines (one per chunk)
ENC_LINES=$(echo "$STREAM_LOG" | grep -oP 'Encoder: \d+ tokens from .* \d+ ms\)' || true)
STEM_LINES=$(echo "$STREAM_LOG" | grep -oP 'Stem cache: \d+/\d+ chunks cached, \d+ recomputed' || true)
PRE_LINES=$(echo "$STREAM_LOG" | grep -oP 'Prefill: \d+ tokens \(\d+ prefix, reused \d+\) \(\d+ ms\)' || true)
DEC_LINES=$(echo "$STREAM_LOG" | grep -oP 'Decode: \d+ tokens \(\d+ ms' || true)

# Convert to arrays
mapfile -t ENC_ARR <<< "$ENC_LINES"
mapfile -t STEM_ARR <<< "$STEM_LINES"
mapfile -t PRE_ARR <<< "$PRE_LINES"
mapfile -t DEC_ARR <<< "$DEC_LINES"

STEM_IDX=0
for i in "${!ENC_ARR[@]}"; do
    [ -z "${ENC_ARR[$i]}" ] && continue
    CHUNK_NUM=$((i + 1))

    ENC_MS=$(echo "${ENC_ARR[$i]}" | grep -oP '\d+(?= ms\))' || echo "?")

    # Stem cache info (only present for chunks with partial windows)
    STEM_INFO="-"
    if [ $STEM_IDX -lt ${#STEM_ARR[@]} ] && [ -n "${STEM_ARR[$STEM_IDX]:-}" ]; then
        # Check if this stem line comes before the next encoder line
        STEM_INFO=$(echo "${STEM_ARR[$STEM_IDX]}" | grep -oP '\d+/\d+' || echo "-")
        STEM_IDX=$((STEM_IDX + 1))
    fi

    PRE_MS="?"
    if [ -n "${PRE_ARR[$i]:-}" ]; then
        PRE_MS=$(echo "${PRE_ARR[$i]}" | grep -oP '\d+(?= ms\))' || echo "?")
    fi

    DEC_MS="?"
    DEC_TOKS="?"
    if [ -n "${DEC_ARR[$i]:-}" ]; then
        DEC_MS=$(echo "${DEC_ARR[$i]}" | grep -oP '\(\K\d+' || echo "?")
        DEC_TOKS=$(echo "${DEC_ARR[$i]}" | grep -oP 'Decode: \K\d+' || echo "?")
    fi

    printf "  %2d  | %10s | %11s | %10s | %9s | %s\n" \
        "$CHUNK_NUM" "$ENC_MS" "$STEM_INFO" "$PRE_MS" "$DEC_MS" "$DEC_TOKS"
done

echo ""

# Prefill reuse summary
REUSE_LINE=$(echo "$STREAM_LOG" | grep -oP 'Prefill reuse: \K.*' || true)
if [ -n "$REUSE_LINE" ]; then
    echo "Prefill reuse: $REUSE_LINE"
fi

# Total wall time estimate: sum of all per-chunk times
TOTAL_ENC_MS=0
TOTAL_PRE_MS=0
TOTAL_DEC_MS=0
for i in "${!ENC_ARR[@]}"; do
    [ -z "${ENC_ARR[$i]}" ] && continue
    ms=$(echo "${ENC_ARR[$i]}" | grep -oP '\d+(?= ms\))' || echo "0")
    TOTAL_ENC_MS=$((TOTAL_ENC_MS + ms))
done
for i in "${!PRE_ARR[@]}"; do
    [ -z "${PRE_ARR[$i]}" ] && continue
    ms=$(echo "${PRE_ARR[$i]}" | grep -oP '\d+(?= ms\))' || echo "0")
    TOTAL_PRE_MS=$((TOTAL_PRE_MS + ms))
done
for i in "${!DEC_ARR[@]}"; do
    [ -z "${DEC_ARR[$i]}" ] && continue
    ms=$(echo "${DEC_ARR[$i]}" | grep -oP '\(\K\d+' || echo "0")
    TOTAL_DEC_MS=$((TOTAL_DEC_MS + ms))
done
TOTAL_STREAM_MS=$((TOTAL_ENC_MS + TOTAL_PRE_MS + TOTAL_DEC_MS))

echo ""
printf "Encoder total:  %5d ms\n" "$TOTAL_ENC_MS"
printf "Prefill total:  %5d ms\n" "$TOTAL_PRE_MS"
printf "Decode total:   %5d ms\n" "$TOTAL_DEC_MS"
printf "Wall time est:  %5d ms\n" "$TOTAL_STREAM_MS"

echo ""

# ── Summary ─────────────────────────────────────────────────────────────────
if [ "$FAIL" -eq 0 ]; then
    echo "=== PASSED (${PASS} checks) ==="
    exit 0
else
    echo "=== ${FAIL} FAILED, ${PASS} passed ==="
    exit 1
fi
