#!/usr/bin/env bash
#
# ASR Integration Test for Android Device
#
# Builds the APK, installs it, loads the model, runs WAV transcription,
# and checks correctness + performance against regression thresholds.
#
# Usage:
#   cd app/android-device && bash scripts/test-asr.sh
#   # or: npm run test:asr

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
PKG="ai.connct_screen.rn"
ACTION="${PKG}.VOICE_DEBUG"
MODEL_DIR="/data/local/tmp/qwen3-asr-0.6b"
WAV_PATH="/data/local/tmp/test.wav"
EXPECTED_PHRASE="ask not what your country can do for you"

# Performance thresholds (ms)
THRESH_ENCODER=5000
THRESH_PREFILL=5000
THRESH_DECODE=5000
THRESH_TOTAL=15000

# Timeout for async operations (seconds)
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

assert() {
    local desc="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        pass "$desc"
    else
        fail "$desc"
    fi
}

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

# ── Preflight checks ───────────────────────────────────────────────────────
echo "=== ASR Integration Test ==="
echo ""

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

# ── Test 1: Build & Install ────────────────────────────────────────────────
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

# Start the app (needed so the broadcast receiver's context works)
adb shell am start -n "${PKG}/.MainActivity" >/dev/null 2>&1
sleep 2
pass "App started"

echo ""

# ── Test 2: Model Loading ──────────────────────────────────────────────────
echo "--- Model Loading ---"

# Clear logcat, then send load_model
adb logcat -c
adb shell "am broadcast -a ${ACTION} -p ${PKG} --es cmd load_model --es path ${MODEL_DIR}" >/dev/null 2>&1

# Poll for completion
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

# Check load_model result=true
if grep -q "load_model result=true" "$LOGCAT_FILE"; then
    pass "load_model result=true"
else
    fail "load_model result was not true"
    dump_log
fi

# Check "Model loaded successfully"
if grep -q "Model loaded successfully" "$LOGCAT_FILE"; then
    pass "Model loaded successfully"
else
    fail "\"Model loaded successfully\" not found in logcat"
    dump_log
fi

echo ""

# ── Test 3: WAV Transcription ──────────────────────────────────────────────
echo "--- WAV Transcription ---"

# Clear logcat, then send test_wav
adb logcat -c
adb shell "am broadcast -a ${ACTION} -p ${PKG} --es cmd test_wav --es path ${WAV_PATH}" >/dev/null 2>&1

# Poll for "nativeTestWav: done"
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

# Capture full logcat
rm -f "$LOGCAT_FILE"
adb logcat -d -s QwenASR_JNI QwenASR VoiceDebug >"$LOGCAT_FILE" 2>/dev/null

if [ "$WAV_DONE" = false ]; then
    fail "WAV transcription timed out (${TIMEOUT_WAV}s)"
    dump_log
    exit 1
fi

# Check "nativeTestWav: done"
if grep -q "nativeTestWav: done" "$LOGCAT_FILE"; then
    pass "Inference completed without crash"
else
    fail "nativeTestWav: done not found"
    dump_log
fi

# Extract transcription result (case-insensitive phrase match)
RESULT_LINE=$(grep "nativeTestWav: result = " "$LOGCAT_FILE" || true)
if [ -z "$RESULT_LINE" ]; then
    fail "No transcription result found in logcat"
    dump_log
else
    # Lowercase and strip punctuation for comparison
    RESULT_LOWER=$(echo "$RESULT_LINE" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z ]//g')
    if echo "$RESULT_LOWER" | grep -q "$EXPECTED_PHRASE"; then
        # Show a snippet of the actual result
        RESULT_TEXT=$(echo "$RESULT_LINE" | sed 's/.*nativeTestWav: result = //')
        pass "Transcription correct: \"${RESULT_TEXT}\""
    else
        RESULT_TEXT=$(echo "$RESULT_LINE" | sed 's/.*nativeTestWav: result = //')
        fail "Transcription mismatch"
        info "Expected phrase: \"${EXPECTED_PHRASE}\""
        info "Got: \"${RESULT_TEXT}\""
    fi
fi

echo ""

# ── Test 4: Performance ────────────────────────────────────────────────────
echo "--- Performance ---"

# Parse timing from QwenASR tag (stderr redirect output)
MEL_MS=$(grep -oP 'Mel: \d+ frames \(\K\d+' "$LOGCAT_FILE" || echo "")
ENC_MS=$(grep -oP 'Encoder: \d+ tokens \(\K\d+' "$LOGCAT_FILE" || echo "")
PRE_MS=$(grep -oP 'Prefill: \d+ tokens \(\K\d+' "$LOGCAT_FILE" || echo "")
DEC_MS=$(grep -oP 'Decode: \d+ tokens \(\K\d+' "$LOGCAT_FILE" || echo "")

if [ -z "$ENC_MS" ] || [ -z "$PRE_MS" ] || [ -z "$DEC_MS" ]; then
    fail "Could not parse timing from logcat"
    dump_log
else
    TOTAL_MS=$((ENC_MS + PRE_MS + DEC_MS))

    echo ""
    echo "  Performance:"
    printf "  ┌───────────┬──────────┬───────────┐\n"
    printf "  │ %-9s │ %-8s │ %-9s │\n" "Phase" "Time" "Threshold"
    printf "  ├───────────┼──────────┼───────────┤\n"
    printf "  │ %-9s │ %5sms  │ %-9s │\n" "Mel" "${MEL_MS:-?}" "-"
    printf "  │ %-9s │ %5sms  │ %-9s │\n" "Encoder" "$ENC_MS" "< ${THRESH_ENCODER}ms"
    printf "  │ %-9s │ %5sms  │ %-9s │\n" "Prefill" "$PRE_MS" "< ${THRESH_PREFILL}ms"
    printf "  │ %-9s │ %5sms  │ %-9s │\n" "Decode" "$DEC_MS" "< ${THRESH_DECODE}ms"
    printf "  │ %-9s │ %5sms  │ %-9s │\n" "TOTAL" "$TOTAL_MS" "< ${THRESH_TOTAL}ms"
    printf "  └───────────┴──────────┴───────────┘\n"
    echo ""

    PERF_OK=true

    if [ "$ENC_MS" -ge "$THRESH_ENCODER" ]; then
        fail "Encoder too slow: ${ENC_MS}ms >= ${THRESH_ENCODER}ms"
        PERF_OK=false
    fi
    if [ "$PRE_MS" -ge "$THRESH_PREFILL" ]; then
        fail "Prefill too slow: ${PRE_MS}ms >= ${THRESH_PREFILL}ms"
        PERF_OK=false
    fi
    if [ "$DEC_MS" -ge "$THRESH_DECODE" ]; then
        fail "Decode too slow: ${DEC_MS}ms >= ${THRESH_DECODE}ms"
        PERF_OK=false
    fi
    if [ "$TOTAL_MS" -ge "$THRESH_TOTAL" ]; then
        fail "Total too slow: ${TOTAL_MS}ms >= ${THRESH_TOTAL}ms"
        PERF_OK=false
    fi

    if [ "$PERF_OK" = true ]; then
        pass "Performance within thresholds"
    fi
fi

# ── Test 5: Streaming Transcription ─────────────────────────────────────────
echo "--- Streaming Transcription ---"

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
    fail "Streaming transcription timed out (${TIMEOUT_WAV}s)"
    dump_log
else
    if grep -q "nativeTestWavStream: done" "$LOGCAT_FILE"; then
        pass "Streaming inference completed without crash"
    else
        fail "nativeTestWavStream: done not found"
        dump_log
    fi

    STREAM_RESULT_LINE=$(grep "nativeTestWavStream: result = " "$LOGCAT_FILE" || true)
    if [ -z "$STREAM_RESULT_LINE" ]; then
        fail "No streaming transcription result found in logcat"
        dump_log
    else
        STREAM_LOWER=$(echo "$STREAM_RESULT_LINE" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z ]//g')
        if echo "$STREAM_LOWER" | grep -q "$EXPECTED_PHRASE"; then
            STREAM_TEXT=$(echo "$STREAM_RESULT_LINE" | sed 's/.*nativeTestWavStream: result = //')
            pass "Streaming transcription correct: \"${STREAM_TEXT}\""
        else
            STREAM_TEXT=$(echo "$STREAM_RESULT_LINE" | sed 's/.*nativeTestWavStream: result = //')
            fail "Streaming transcription mismatch"
            info "Expected phrase: \"${EXPECTED_PHRASE}\""
            info "Got: \"${STREAM_TEXT}\""
        fi
    fi
fi

echo ""

# ── Test 6: Repeat Inference (batch x2) ────────────────────────────────────
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
            pass "Repeat inference result correct"
        else
            REPEAT_TEXT=$(echo "$REPEAT_RESULT_LINE" | sed 's/.*nativeTestWav: result = //')
            fail "Repeat inference result mismatch"
            info "Expected phrase: \"${EXPECTED_PHRASE}\""
            info "Got: \"${REPEAT_TEXT}\""
        fi
    fi
fi

echo ""

# ── Test 7: Unload → Reload → Inference ───────────────────────────────────
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

# ── Test 8: Error — test_wav without model ─────────────────────────────────
echo "--- Error: No Model ---"

# Free model first
adb logcat -c
adb shell "am broadcast -a ${ACTION} -p ${PKG} --es cmd free_model" >/dev/null 2>&1
sleep 2

# Try test_wav without model loaded
adb logcat -c
adb shell "am broadcast -a ${ACTION} -p ${PKG} --es cmd test_wav --es path ${WAV_PATH}" >/dev/null 2>&1
sleep 3

rm -f "$LOGCAT_FILE"
adb logcat -d -s QwenASR_JNI QwenASR VoiceDebug >"$LOGCAT_FILE" 2>/dev/null

if grep -q "model not loaded" "$LOGCAT_FILE"; then
    pass "Correctly reported 'model not loaded'"
else
    fail "'model not loaded' message not found"
    dump_log
fi

# Check process is still alive
APP_PID=$(adb shell pidof "$PKG" 2>/dev/null || true)
if [ -n "$APP_PID" ]; then
    pass "Process still alive after no-model call (pid=$APP_PID)"
else
    fail "Process crashed after no-model test_wav call"
fi

echo ""

# ── Test 9: Error — nonexistent WAV path ───────────────────────────────────
echo "--- Error: Bad WAV Path ---"

# Reload model (Test 8 freed it)
adb logcat -c
adb shell "am broadcast -a ${ACTION} -p ${PKG} --es cmd load_model --es path ${MODEL_DIR}" >/dev/null 2>&1

ELAPSED=0
RELOAD9_OK=false
while [ $ELAPSED -lt $TIMEOUT_LOAD ]; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    if adb logcat -d -s VoiceDebug 2>/dev/null | grep -q "load_model result=true"; then
        RELOAD9_OK=true
        break
    fi
done

if [ "$RELOAD9_OK" = false ]; then
    fail "Model reload for Test 9 timed out"
    rm -f "$LOGCAT_FILE"
    adb logcat -d -s QwenASR_JNI QwenASR VoiceDebug >"$LOGCAT_FILE" 2>/dev/null
    dump_log
fi

# Send test_wav with nonexistent path
adb logcat -c
adb shell "am broadcast -a ${ACTION} -p ${PKG} --es cmd test_wav --es path /data/local/tmp/nonexistent.wav" >/dev/null 2>&1
sleep 3

rm -f "$LOGCAT_FILE"
adb logcat -d -s QwenASR_JNI QwenASR VoiceDebug >"$LOGCAT_FILE" 2>/dev/null

if grep -q "WAV file not found" "$LOGCAT_FILE"; then
    pass "Correctly reported 'WAV file not found'"
else
    fail "'WAV file not found' message not found"
    dump_log
fi

APP_PID=$(adb shell pidof "$PKG" 2>/dev/null || true)
if [ -n "$APP_PID" ]; then
    pass "Process still alive after bad-path call (pid=$APP_PID)"
else
    fail "Process crashed after bad-path test_wav call"
fi

echo ""

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "=== ALL TESTS PASSED (${PASS} checks) ==="
    exit 0
else
    echo "=== ${FAIL} FAILED, ${PASS} passed ==="
    exit 1
fi
