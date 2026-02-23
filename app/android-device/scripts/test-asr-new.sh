#!/usr/bin/env bash
#
# ASR Baseline Integration Test for Android Device
#
# Builds the APK, installs it, loads the model, runs batch + stream
# transcription with jfk.wav, checks correctness, and prints a
# detailed performance report (no pass/fail thresholds — baseline is
# too slow for the old thresholds, so we just record numbers).
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

# Timeout for async operations (seconds)
TIMEOUT_LOAD=60
TIMEOUT_WAV=600   # baseline stream ~115s, leave wide margin

# ── Helpers ─────────────────────────────────────────────────────────────────
PASS=0
FAIL=0
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ANDROID_DIR="$(cd "$SCRIPT_DIR/../android" && pwd)"

pass() { PASS=$((PASS + 1)); echo "[✓] $1"; }
fail() { FAIL=$((FAIL + 1)); echo "[✗] $1"; }
info() { echo "    $1"; }

# Dump captured logcat on failure for debugging
dump_log() {
    if [ -n "${LOGCAT_FILE:-}" ] && [ -f "$LOGCAT_FILE" ]; then
        echo ""
        echo "--- logcat dump ---"
        cat "$LOGCAT_FILE"
        echo "--- end logcat ---"
    fi
}

cleanup() {
    rm -f "${LOGCAT_FILE:-}"
}
trap cleanup EXIT

# ── Test 1: Preflight checks ─────────────────────────────────────────────
echo "=== ASR Baseline Integration Test ==="
echo ""
echo "--- Preflight ---"

# Check adb
if ! command -v adb &>/dev/null; then
    echo "ERROR: adb not found in PATH"
    exit 1
fi

# Check device connected
DEVICE_COUNT=$(adb devices | grep -cw 'device' || true)
if [ "$DEVICE_COUNT" -lt 1 ]; then
    echo "ERROR: No Android device connected (adb devices shows no device)"
    exit 1
fi
pass "Device connected"

# Check model directory
if ! adb shell "ls ${MODEL_DIR}/" &>/dev/null; then
    fail "Model not found at ${MODEL_DIR}"
    echo "ERROR: Push the model first: adb push qwen3-asr-0.6b ${MODEL_DIR}"
    exit 1
fi
pass "Model found at ${MODEL_DIR}"

# Check test WAV
if ! adb shell "ls ${WAV_PATH}" &>/dev/null; then
    fail "Test WAV not found at ${WAV_PATH}"
    echo "ERROR: Push test audio first: adb push jfk.wav ${WAV_PATH}"
    exit 1
fi
pass "Test WAV found at ${WAV_PATH}"

echo ""

# ── Test 2: Build & Install ──────────────────────────────────────────────
echo "--- Build & Install ---"

if (cd "$ANDROID_DIR" && ./gradlew assembleDebug) >/dev/null 2>&1; then
    pass "Build succeeded"
else
    fail "Build failed"
    echo "ERROR: ./gradlew assembleDebug failed. Run manually to see errors."
    exit 1
fi

APK="$ANDROID_DIR/app/build/outputs/apk/debug/app-debug.apk"
if [ ! -f "$APK" ]; then
    fail "APK not found at expected path"
    exit 1
fi

if adb install -r "$APK" >/dev/null 2>&1; then
    pass "Install succeeded"
else
    fail "Install failed"
    exit 1
fi

# Start the app (broadcast receiver needs the app process alive)
adb shell am start -n "${PKG}/.MainActivity" >/dev/null 2>&1
sleep 2
pass "App started"

echo ""

# ── Test 3: Load Model ───────────────────────────────────────────────────
echo "--- Load Model ---"

adb logcat -c
adb shell "am broadcast -a ${ACTION} -p ${PKG} --es cmd load_model --es path ${MODEL_DIR}" >/dev/null 2>&1

ELAPSED=0
LOAD_OK=false
while [ $ELAPSED -lt $TIMEOUT_LOAD ]; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    if adb logcat -d -s VoiceDebug 2>/dev/null | grep -q "load_model result="; then
        LOAD_OK=true
        break
    fi
done

LOGCAT_FILE=$(mktemp)
adb logcat -d -s QwenASR_JNI QwenASR VoiceDebug >"$LOGCAT_FILE" 2>/dev/null

if [ "$LOAD_OK" = false ]; then
    fail "Model loading timed out (${TIMEOUT_LOAD}s)"
    dump_log
    exit 1
fi

if grep -q "load_model result=true" "$LOGCAT_FILE"; then
    pass "load_model result=true"
else
    fail "load_model result was not true"
    dump_log
fi

if grep -q "Model loaded successfully" "$LOGCAT_FILE"; then
    pass "Model loaded successfully"
else
    fail "\"Model loaded successfully\" not found in logcat"
    dump_log
fi

echo ""

# ── Test 4: Batch Transcription (jfk.wav) ────────────────────────────────
echo "--- Batch Transcription (jfk.wav) ---"

adb logcat -c
adb shell "am broadcast -a ${ACTION} -p ${PKG} --es cmd test_wav --es path ${WAV_PATH}" >/dev/null 2>&1

ELAPSED=0
WAV_DONE=false
while [ $ELAPSED -lt $TIMEOUT_WAV ]; do
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if adb logcat -d -s QwenASR_JNI 2>/dev/null | grep -q "nativeTestWav: done"; then
        WAV_DONE=true
        break
    fi
done

rm -f "$LOGCAT_FILE"
adb logcat -d -s QwenASR_JNI QwenASR VoiceDebug >"$LOGCAT_FILE" 2>/dev/null

if [ "$WAV_DONE" = false ]; then
    fail "Batch transcription timed out (${TIMEOUT_WAV}s)"
    dump_log
    exit 1
fi

if grep -q "nativeTestWav: done" "$LOGCAT_FILE"; then
    pass "Batch inference completed without crash"
else
    fail "nativeTestWav: done not found"
    dump_log
fi

# Check correctness
RESULT_LINE=$(grep "nativeTestWav: result = " "$LOGCAT_FILE" || true)
if [ -z "$RESULT_LINE" ]; then
    fail "No batch transcription result found in logcat"
    dump_log
else
    RESULT_LOWER=$(echo "$RESULT_LINE" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z ]//g')
    if echo "$RESULT_LOWER" | grep -q "$EXPECTED_PHRASE"; then
        RESULT_TEXT=$(echo "$RESULT_LINE" | sed 's/.*nativeTestWav: result = //')
        pass "Batch transcription correct: \"${RESULT_TEXT}\""
    else
        RESULT_TEXT=$(echo "$RESULT_LINE" | sed 's/.*nativeTestWav: result = //')
        fail "Batch transcription mismatch"
        info "Expected phrase: \"${EXPECTED_PHRASE}\""
        info "Got: \"${RESULT_TEXT}\""
    fi
fi

# Extract batch performance numbers
BATCH_MEL_MS=$(grep -oP 'Mel: \d+ frames \(\K\d+' "$LOGCAT_FILE" || echo "")
BATCH_ENC_MS=$(grep -oP 'Encoder: \d+ tokens \(\K\d+' "$LOGCAT_FILE" || echo "")
BATCH_ENC_TOKENS=$(grep -oP 'Encoder: \K\d+' "$LOGCAT_FILE" || echo "")
BATCH_PRE_MS=$(grep -oP 'Prefill: \d+ tokens \(\K\d+' "$LOGCAT_FILE" || echo "")
BATCH_PRE_TOKENS=$(grep -oP 'Prefill: \K\d+' "$LOGCAT_FILE" || echo "")
BATCH_DEC_MS=$(grep -oP 'Decode: \d+ tokens \(\K\d+' "$LOGCAT_FILE" || echo "")
BATCH_DEC_TOKENS=$(grep -oP 'Decode: \K\d+' "$LOGCAT_FILE" || echo "")
BATCH_DEC_PER=$(grep -oP 'Decode: \d+ tokens \(\d+ ms, \K[0-9.]+' "$LOGCAT_FILE" || echo "")

echo ""

# ── Test 5: Stream Transcription (jfk.wav) ───────────────────────────────
echo "--- Stream Transcription (jfk.wav) ---"

adb logcat -c
adb shell "am broadcast -a ${ACTION} -p ${PKG} --es cmd test_wav_stream --es path ${WAV_PATH}" >/dev/null 2>&1

ELAPSED=0
STREAM_DONE=false
while [ $ELAPSED -lt $TIMEOUT_WAV ]; do
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if adb logcat -d -s QwenASR_JNI 2>/dev/null | grep -q "nativeTestWavStream: done"; then
        STREAM_DONE=true
        break
    fi
done

rm -f "$LOGCAT_FILE"
adb logcat -d -s QwenASR_JNI QwenASR VoiceDebug >"$LOGCAT_FILE" 2>/dev/null

if [ "$STREAM_DONE" = false ]; then
    fail "Stream transcription timed out (${TIMEOUT_WAV}s)"
    dump_log
    exit 1
fi

if grep -q "nativeTestWavStream: done" "$LOGCAT_FILE"; then
    pass "Stream inference completed without crash"
else
    fail "nativeTestWavStream: done not found"
    dump_log
fi

# Check correctness
STREAM_RESULT_LINE=$(grep "nativeTestWavStream: result = " "$LOGCAT_FILE" || true)
if [ -z "$STREAM_RESULT_LINE" ]; then
    fail "No stream transcription result found in logcat"
    dump_log
else
    STREAM_LOWER=$(echo "$STREAM_RESULT_LINE" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z ]//g')
    if echo "$STREAM_LOWER" | grep -q "$EXPECTED_PHRASE"; then
        STREAM_TEXT=$(echo "$STREAM_RESULT_LINE" | sed 's/.*nativeTestWavStream: result = //')
        pass "Stream transcription correct: \"${STREAM_TEXT}\""
    else
        STREAM_TEXT=$(echo "$STREAM_RESULT_LINE" | sed 's/.*nativeTestWavStream: result = //')
        fail "Stream transcription mismatch"
        info "Expected phrase: \"${EXPECTED_PHRASE}\""
        info "Got: \"${STREAM_TEXT}\""
    fi
fi

# Extract stream per-chunk performance from QwenASR tag
# Encoder lines: "Encoder: <N> tokens from 0.0-<T> s (..., <MS> ms)"
# Prefill lines: "Prefill: <N> tokens (<PFX> prefix, reused <R>) (<MS> ms)"
# Decode lines:  "Decode: <N> tokens (<MS> ms, <PER> ms/token)"
# Prefill reuse: "Prefill reuse: <N>/<Total> tokens (<PCT>%)"

STREAM_CHUNK_COUNT=0
declare -a STREAM_ENC_MS_ARR=()
declare -a STREAM_PRE_MS_ARR=()
declare -a STREAM_DEC_MS_ARR=()
declare -a STREAM_RANGE_ARR=()
declare -a STREAM_ENC_TOK_ARR=()

while IFS= read -r line; do
    STREAM_CHUNK_COUNT=$((STREAM_CHUNK_COUNT + 1))
    enc_ms=$(echo "$line" | grep -oP '\d+ ms\)' | grep -oP '\d+' | tail -1 || echo "?")
    enc_tok=$(echo "$line" | grep -oP 'Encoder: \K\d+' || echo "?")
    range=$(echo "$line" | grep -oP 'from \K[0-9.]+-[0-9.]+ s' || echo "?")
    STREAM_ENC_MS_ARR+=("$enc_ms")
    STREAM_ENC_TOK_ARR+=("$enc_tok")
    STREAM_RANGE_ARR+=("$range")
done < <(grep "Encoder:.*tokens from" "$LOGCAT_FILE" || true)

while IFS= read -r line; do
    pre_ms=$(echo "$line" | grep -oP '\(\d+ ms\)' | grep -oP '\d+' || echo "?")
    STREAM_PRE_MS_ARR+=("$pre_ms")
done < <(grep "Prefill:.*tokens.*ms)" "$LOGCAT_FILE" | grep -v "reuse:" || true)

while IFS= read -r line; do
    dec_ms=$(echo "$line" | grep -oP 'Decode: \d+ tokens \(\K\d+' || echo "?")
    STREAM_DEC_MS_ARR+=("$dec_ms")
done < <(grep "Decode:.*tokens.*ms/token" "$LOGCAT_FILE" || true)

STREAM_PREFILL_REUSE=$(grep -oP 'Prefill reuse: \K.*' "$LOGCAT_FILE" || echo "")

# Compute stream wall time from first and last logcat timestamps
STREAM_FIRST_TS=$(grep -m1 "Encoder:.*tokens from" "$LOGCAT_FILE" | grep -oP '^\S+ \K[0-9:.]+' || echo "")
STREAM_LAST_TS=$(grep "nativeTestWavStream: done" "$LOGCAT_FILE" | grep -oP '^\S+ \K[0-9:.]+' || echo "")

stream_wall_ms=""
if [ -n "$STREAM_FIRST_TS" ] && [ -n "$STREAM_LAST_TS" ]; then
    # Parse HH:MM:SS.mmm timestamps to milliseconds
    ts_to_ms() {
        local ts="$1"
        local h m s ms_part
        h=$(echo "$ts" | cut -d: -f1 | sed 's/^0*//' )
        m=$(echo "$ts" | cut -d: -f2 | sed 's/^0*//' )
        s=$(echo "$ts" | cut -d: -f3 | cut -d. -f1 | sed 's/^0*//')
        ms_part=$(echo "$ts" | grep -oP '\.\K\d+' || echo "0")
        h=${h:-0}; m=${m:-0}; s=${s:-0}
        echo $(( h * 3600000 + m * 60000 + s * 1000 + ${ms_part:-0} ))
    }
    first_ms=$(ts_to_ms "$STREAM_FIRST_TS")
    last_ms=$(ts_to_ms "$STREAM_LAST_TS")
    if [ "$last_ms" -ge "$first_ms" ]; then
        stream_wall_ms=$((last_ms - first_ms))
    fi
fi

echo ""

# ── Test 6: Repeat Inference ─────────────────────────────────────────────
echo "--- Repeat Inference ---"

adb logcat -c
adb shell "am broadcast -a ${ACTION} -p ${PKG} --es cmd test_wav --es path ${WAV_PATH}" >/dev/null 2>&1

ELAPSED=0
WAV2_DONE=false
while [ $ELAPSED -lt $TIMEOUT_WAV ]; do
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if adb logcat -d -s QwenASR_JNI 2>/dev/null | grep -q "nativeTestWav: done"; then
        WAV2_DONE=true
        break
    fi
done

rm -f "$LOGCAT_FILE"
adb logcat -d -s QwenASR_JNI QwenASR VoiceDebug >"$LOGCAT_FILE" 2>/dev/null

if [ "$WAV2_DONE" = false ]; then
    fail "Repeat inference timed out (${TIMEOUT_WAV}s)"
    dump_log
else
    if grep -q "nativeTestWav: done" "$LOGCAT_FILE"; then
        pass "Repeat inference completed without crash"
    else
        fail "nativeTestWav: done not found on repeat"
        dump_log
    fi

    REPEAT_RESULT_LINE=$(grep "nativeTestWav: result = " "$LOGCAT_FILE" || true)
    if [ -z "$REPEAT_RESULT_LINE" ]; then
        fail "No repeat transcription result found"
        dump_log
    else
        REPEAT_LOWER=$(echo "$REPEAT_RESULT_LINE" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z ]//g')
        if echo "$REPEAT_LOWER" | grep -q "$EXPECTED_PHRASE"; then
            pass "Repeat inference result correct (KV cache reset works)"
        else
            REPEAT_TEXT=$(echo "$REPEAT_RESULT_LINE" | sed 's/.*nativeTestWav: result = //')
            fail "Repeat inference result mismatch"
            info "Expected phrase: \"${EXPECTED_PHRASE}\""
            info "Got: \"${REPEAT_TEXT}\""
        fi
    fi
fi

echo ""

# ── Test 7: Unload → Reload → Inference ──────────────────────────────────
echo "--- Unload / Reload / Inference ---"

# Free model
adb logcat -c
adb shell "am broadcast -a ${ACTION} -p ${PKG} --es cmd free_model" >/dev/null 2>&1

ELAPSED=0
FREE_OK=false
while [ $ELAPSED -lt $TIMEOUT_LOAD ]; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    if adb logcat -d -s QwenASR_JNI 2>/dev/null | grep -q "Model freed"; then
        FREE_OK=true
        break
    fi
done

if [ "$FREE_OK" = true ]; then
    pass "Model freed"
else
    fail "free_model timed out"
    dump_log
fi

# Reload model
adb logcat -c
adb shell "am broadcast -a ${ACTION} -p ${PKG} --es cmd load_model --es path ${MODEL_DIR}" >/dev/null 2>&1

ELAPSED=0
RELOAD_OK=false
while [ $ELAPSED -lt $TIMEOUT_LOAD ]; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    if adb logcat -d -s VoiceDebug 2>/dev/null | grep -q "load_model result=true"; then
        RELOAD_OK=true
        break
    fi
done

if [ "$RELOAD_OK" = true ]; then
    pass "Model reloaded"
else
    fail "Reload timed out"
    rm -f "$LOGCAT_FILE"
    adb logcat -d -s QwenASR_JNI QwenASR VoiceDebug >"$LOGCAT_FILE" 2>/dev/null
    dump_log
fi

# Inference after reload
adb logcat -c
adb shell "am broadcast -a ${ACTION} -p ${PKG} --es cmd test_wav --es path ${WAV_PATH}" >/dev/null 2>&1

ELAPSED=0
WAV3_DONE=false
while [ $ELAPSED -lt $TIMEOUT_WAV ]; do
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if adb logcat -d -s QwenASR_JNI 2>/dev/null | grep -q "nativeTestWav: done"; then
        WAV3_DONE=true
        break
    fi
done

rm -f "$LOGCAT_FILE"
adb logcat -d -s QwenASR_JNI QwenASR VoiceDebug >"$LOGCAT_FILE" 2>/dev/null

if [ "$WAV3_DONE" = false ]; then
    fail "Post-reload inference timed out"
    dump_log
else
    if grep -q "nativeTestWav: done" "$LOGCAT_FILE"; then
        pass "Post-reload inference completed"
    else
        fail "nativeTestWav: done not found after reload"
        dump_log
    fi

    RELOAD_RESULT_LINE=$(grep "nativeTestWav: result = " "$LOGCAT_FILE" || true)
    if [ -z "$RELOAD_RESULT_LINE" ]; then
        fail "No transcription result after reload"
        dump_log
    else
        RELOAD_LOWER=$(echo "$RELOAD_RESULT_LINE" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z ]//g')
        if echo "$RELOAD_LOWER" | grep -q "$EXPECTED_PHRASE"; then
            pass "Post-reload transcription correct"
        else
            RELOAD_TEXT=$(echo "$RELOAD_RESULT_LINE" | sed 's/.*nativeTestWav: result = //')
            fail "Post-reload transcription mismatch"
            info "Expected phrase: \"${EXPECTED_PHRASE}\""
            info "Got: \"${RELOAD_TEXT}\""
        fi
    fi
fi

echo ""

# ── Performance Report ────────────────────────────────────────────────────
echo "=== Performance Report ==="
echo ""

# Batch report
echo "Batch (jfk.wav, 11.0s audio):"
if [ -n "$BATCH_ENC_MS" ] && [ -n "$BATCH_PRE_MS" ] && [ -n "$BATCH_DEC_MS" ]; then
    BATCH_TOTAL_MS=$((${BATCH_MEL_MS:-0} + BATCH_ENC_MS + BATCH_PRE_MS + BATCH_DEC_MS))
    printf "  Mel:      %s ms\n" "${BATCH_MEL_MS:-?}"
    printf "  Encoder:  %s ms  (%s tokens)\n" "$BATCH_ENC_MS" "${BATCH_ENC_TOKENS:-?}"
    printf "  Prefill:  %s ms  (%s tokens)\n" "$BATCH_PRE_MS" "${BATCH_PRE_TOKENS:-?}"
    if [ -n "$BATCH_DEC_PER" ]; then
        printf "  Decode:   %s ms   (%s tokens, %s ms/tok)\n" "$BATCH_DEC_MS" "${BATCH_DEC_TOKENS:-?}" "$BATCH_DEC_PER"
    else
        printf "  Decode:   %s ms   (%s tokens)\n" "$BATCH_DEC_MS" "${BATCH_DEC_TOKENS:-?}"
    fi
    printf "  Total:    %s ms\n" "$BATCH_TOTAL_MS"
else
    echo "  (could not parse batch timing from logcat)"
fi

echo ""

# Stream report
echo "Stream (jfk.wav, 11.0s audio):"
if [ "$STREAM_CHUNK_COUNT" -gt 0 ]; then
    for ((i=0; i<STREAM_CHUNK_COUNT; i++)); do
        chunk_num=$((i + 1))
        enc="${STREAM_ENC_MS_ARR[$i]:-?}"
        pre="${STREAM_PRE_MS_ARR[$i]:-?}"
        dec="${STREAM_DEC_MS_ARR[$i]:-?}"
        range="${STREAM_RANGE_ARR[$i]:-?}"
        enc_tok="${STREAM_ENC_TOK_ARR[$i]:-?}"
        printf "  Chunk %2d: enc %5sms  pre %5sms  dec %5sms  (%s, %s enc_tok)\n" \
            "$chunk_num" "$enc" "$pre" "$dec" "$range" "$enc_tok"
    done
    if [ -n "$stream_wall_ms" ]; then
        printf "  Stream wall:  %s ms\n" "$stream_wall_ms"
    fi
    if [ -n "$STREAM_PREFILL_REUSE" ]; then
        printf "  Prefill reuse: %s\n" "$STREAM_PREFILL_REUSE"
    fi
else
    echo "  (could not parse stream timing from logcat)"
fi

echo ""

# ── Summary ───────────────────────────────────────────────────────────────
if [ "$FAIL" -eq 0 ]; then
    echo "=== ALL TESTS PASSED (${PASS} checks) ==="
    exit 0
else
    echo "=== ${FAIL} FAILED, ${PASS} passed ==="
    exit 1
fi
