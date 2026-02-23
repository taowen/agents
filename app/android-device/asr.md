# ASR Offline Testing (adb broadcast backdoor)

## Commands

Three commands via `am broadcast`, results in logcat tags `VoiceDebug` / `QwenASR_JNI`.

```bash
# 1. Check status (model loaded? VoiceService alive?)
adb shell "am broadcast -a ai.connct_screen.rn.VOICE_DEBUG -p ai.connct_screen.rn --es cmd status"

# 2. Load model (default path or custom)
adb shell "am broadcast -a ai.connct_screen.rn.VOICE_DEBUG -p ai.connct_screen.rn --es cmd load_model --es path /data/local/tmp/qwen3-asr-0.6b"

# 3. Test WAV file (model must be loaded first)
adb shell "am broadcast -a ai.connct_screen.rn.VOICE_DEBUG -p ai.connct_screen.rn --es cmd test_wav --es path /data/local/tmp/test.wav"

# View results
adb logcat -d -s QwenASR_JNI VoiceDebug | tail -30
```

## Setup

```bash
# Push model (~1.8GB)
adb push /home/taowen/qwen-asr/qwen3-asr-0.6b/ /data/local/tmp/qwen3-asr-0.6b/

# Push test WAV
adb push /home/taowen/qwen-asr/samples/jfk.wav /data/local/tmp/test.wav
```

## Gotchas

- **Default model path doesn't work**: `status` reports `model_dir=/data/user/0/ai.connct_screen.rn/files/qwen3-asr-0.6b` but that directory is empty unless the app has downloaded the model itself. Must use `--es path /data/local/tmp/qwen3-asr-0.6b` explicitly.
- **Model load is fast**: ~0.6s on device (Snapdragon).
- **Transcription was slow (fixed)**: jfk.wav (11s audio) originally took ~217s. Root causes: (1) `qwen_set_threads()` was never called so inference ran single-threaded despite `nThreads=4` being passed; (2) `test_wav` used streaming mode (`qwen_transcribe_stream_live`) which re-prefills O(N²) per chunk. Fixed by calling `qwen_set_threads()` after load and switching `test_wav` to batch mode (`qwen_transcribe_audio`).
- **Token callback works**: `onNativeToken` fires during inference but tokens only appear via the `VoiceService.handleToken` path (main thread Handler). In `test_wav` mode without a running `VoiceService` instance, tokens are logged by JNI but silently dropped in Java since `sInstance == null`.
- **Thread blocking**: `nativeTestWav` blocks the calling thread. `VoiceDebugReceiver.doTestWav` runs it on a `new Thread()`, so the broadcast returns immediately and results appear in logcat later.
- **No concurrent safety**: If `test_wav` is called while another test or live ASR is running, behavior is undefined (shared `g_ctx`). Always wait for completion before issuing another test.

## Full test session log

```
# Build & install
./gradlew assembleDebug   # BUILD SUCCESSFUL in 17s
adb install -r app/build/outputs/apk/debug/app-debug.apk

# Start app (needed for the broadcast receiver to be active)
adb shell am start -n ai.connct_screen.rn/.MainActivity

# Status check
I VoiceDebug: cmd=status
I VoiceDebug: model_ready=false
I VoiceDebug: model_dir=/data/user/0/ai.connct_screen.rn/files/qwen3-asr-0.6b
I VoiceDebug: voice_service=false

# Load model (default path fails)
I VoiceDebug: Loading model from: /data/user/0/ai.connct_screen.rn/files/qwen3-asr-0.6b
E QwenASR_JNI: nativeLoadModel: qwen_load failed
I VoiceDebug: load_model result=false

# Load model (explicit path works)
I VoiceDebug: Loading model from: /data/local/tmp/qwen3-asr-0.6b
I QwenASR_JNI: Model loaded successfully
I VoiceDebug: load_model result=true

# Test WAV
I QwenASR_JNI: nativeTestWav: loading /data/local/tmp/test.wav
I QwenASR_JNI: nativeTestWav: loaded 176000 samples (11.00 sec)
I QwenASR_JNI: nativeTestWav: starting transcription...
# ... ~3m37s later ...
I QwenASR_JNI: nativeTestWav: result = And so, my fellow Americans, ask not what your country can do for you; ask what you can do for your country.
I QwenASR_JNI: nativeTestWav: done
I VoiceDebug: test_wav completed
```

## Performance diagnosis (2026-02-23)

jfk.wav (11s) took 217s (~20x realtime). Three issues identified:

### 1. Thread count not set (severe)

`nativeLoadModel` received `nThreads` parameter but never called `qwen_set_threads()`.
The thread pool defaulted to 1, so all matmul/attention ran single-threaded.

**Fix**: Added `qwen_set_threads(nThreads > 0 ? nThreads : 4)` after `qwen_load()`.

### 2. Streaming mode O(N²) for WAV test (medium)

`nativeTestWav` used `qwen_transcribe_stream_live()` which processes 2s chunks,
re-prefilling all encoder tokens each chunk (no cross-chunk KV cache).
For 11s audio: 6 chunks with increasing prefill lengths = O(N²).

**Fix**: Switched `nativeTestWav` to `qwen_transcribe_audio()` (single encoder + prefill + decode).

### 3. No quantization (noted, not fixed)

Model is 1.8GB BF16 safetensors. Encoder weights loaded as FP32, decoder stays BF16.
Prefill matmul uses naive C loops (no BLAS), only matvec has NEON optimization.
INT8/INT4 quantization is a potential future optimization.
