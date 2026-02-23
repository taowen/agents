# ASR Offline Testing (adb broadcast backdoor)

## Commands

Six commands via `am broadcast`, results in logcat tags `VoiceDebug` / `QwenASR_JNI`.

```bash
# 1. Check status (model loaded? VoiceService alive?)
adb shell "am broadcast -a ai.connct_screen.rn.VOICE_DEBUG -p ai.connct_screen.rn --es cmd status"

# 2. Load model (default path or custom)
adb shell "am broadcast -a ai.connct_screen.rn.VOICE_DEBUG -p ai.connct_screen.rn --es cmd load_model --es path /data/local/tmp/qwen3-asr-0.6b"

# 3. Test WAV file — batch mode (model must be loaded first)
adb shell "am broadcast -a ai.connct_screen.rn.VOICE_DEBUG -p ai.connct_screen.rn --es cmd test_wav --es path /data/local/tmp/test.wav"

# 4. Test WAV file — streaming mode (same production codepath as live mic)
adb shell "am broadcast -a ai.connct_screen.rn.VOICE_DEBUG -p ai.connct_screen.rn --es cmd test_wav_stream --es path /data/local/tmp/test.wav"

# 5. Free model (unload from memory)
adb shell "am broadcast -a ai.connct_screen.rn.VOICE_DEBUG -p ai.connct_screen.rn --es cmd free_model"

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

### 3. No quantization (fixed — see Q8_0 section below)

Model is 1.8GB BF16 safetensors. Encoder weights loaded as FP32, decoder stays BF16.
Prefill matmul uses naive C loops (no BLAS), only matvec has NEON optimization.
Fixed by Q8_0 quantization — see below.

## Q8_0 quantization (2026-02-23)

All large weight matrices converted from FP32/BF16 to Q8_0 (32 int8 values + 1 float scale per block = 1.125 bytes/weight). Quantization happens at model load time (BF16 safetensors -> Q8_0 in memory). Token embeddings remain BF16 (used for argmax), conv stem weights remain FP32 (small).

### Results (jfk.wav, 11s audio, 4 threads, Snapdragon)

| Phase | Before | After | Speedup |
|-------|--------|-------|---------|
| Mel | 19ms | 19ms | - |
| Encoder | 2666ms | 2152ms | 1.24x |
| Prefill | 2541ms | 2541ms | 1.11x |
| Decode | 1654ms | 1217ms | 1.36x |
| **Total** | **7131ms** | **5910ms** | **1.21x** |

### Analysis: why speedup is below 2x estimate

The plan predicted ~2x overall speedup. Actual result is 1.21x. The main reasons:

1. **GEMM kernel dequantizes to FP32 on-the-fly**. The Q8_0 GEMM (used by encoder and prefill when seq_len > 1) dequantizes each Q8_0 block to 8 float32x4 vectors, then uses the same `vfmaq_laneq_f32` FMA pattern as the original FP32 GEMM. This means the compute is still FP32 — we only save on DRAM bandwidth, not on ALU throughput. The dequantization overhead partially offsets the bandwidth gain.

2. **Matvec kernel uses vdotq_s32 (true INT8 compute)**. The Q8_0 matvec (used by decode, seq_len=1) quantizes the input vector to Q8_0 once, then uses `vdotq_s32` for INT8 x INT8 dot products directly. This is the "real" quantization speedup — both bandwidth AND compute savings. The 1.36x decode speedup reflects this, closer to the theoretical 2x for BF16->Q8 bandwidth reduction.

3. **Encoder bottleneck is attention, not linear**. The encoder has windowed bidirectional attention that is compute-bound (softmax, attention score computation), not memory-bound. Quantizing the linear layers helps but doesn't touch the attention computation.

4. **Prefill is dominated by KV cache writes**. The prefill processes ~143 tokens through 28 decoder layers. Much of the time is spent writing to the FP32 KV cache and computing attention over the growing sequence, not in the weight matmuls.

### What was implemented

Files created:
- `qwen_asr_quant.h` / `qwen_asr_quant.c` — Q8_0 block type, quantize (NEON-optimized), dequantize

Files modified:
- `qwen_asr_kernels_impl.h` — Q8_0 matvec dispatch macros (NEON / generic)
- `qwen_asr_kernels_neon.c` — `qwen_q8_matvec_fused_neon` with `vdotq_s32` (ARMv8.2 dotprod) + fallback path
- `qwen_asr_kernels_generic.c` — `qwen_q8_matvec_fused_generic` scalar reference
- `qwen_asr_kernels.h` / `qwen_asr_kernels.c` — Q8_0 GEMM (batched, threaded), matvec (threaded), fused QKV matvec, `qwen_linear_q8`, `qwen_linear_nobias_q8`
- `qwen_asr.h` — struct fields `float *` / `uint16_t *` -> `block_q8_0 *` for all weight matrices
- `qwen_asr_encoder.c` — BF16->Q8_0 at load, Q8 linear in forward
- `qwen_asr_decoder.c` — BF16->Q8_0 at load (including gate+up interleaved fusion), Q8 linear in forward
- `qwen_asr.c` — free Q8_0 weights (decoder weights were previously mmap'd BF16, now malloc'd Q8_0)
- `CMakeLists.txt` — added `qwen_asr_quant.c`

### Potential next optimizations

1. ~~**INT8 GEMM** (high impact).~~ Done — see Round 2 below.

2. **KV cache FP16** (medium impact). KV cache is FP32. Storing as FP16 would halve attention memory bandwidth. Matters more for longer sequences.

3. **Q4_K_M for decoder** (medium impact). After validating Q8_0 quality, the decoder (language model) could use 4-bit quantization for ~2x further bandwidth reduction. Encoder should stay Q8_0 (audio features are more quantization-sensitive).

4. ~~**GEMM buffer pre-allocation** (low impact).~~ Done — see Round 2 below.

## INT8 GEMM + workspace pre-allocation (2026-02-23)

Round 2 optimization: replaced FP32 FMA-based Q8_0 GEMM with true INT8×INT8 dot product GEMM using `vdotq_s32`, and eliminated per-GEMM malloc/free overhead with a persistent workspace.

### Problem

The Q8_0 GEMM kernel (`q8_gemm_worker`) was dequantizing each Q8_0 block (32 int8 weights) to 8 × float32x4 vectors (~30 NEON instructions), then doing FP32 FMA via `vfmaq_laneq_f32`. This meant the GEMM path (encoder + prefill, seq_len > 1) never got the compute benefit of INT8 — only bandwidth savings from smaller weights. Additionally, each GEMM call malloc'd/free'd transposed input and output buffers (~650 malloc/free per inference).

### Solution

1. **`quantize_f32_rows_transpose_q8`**: New function in `qwen_asr_quant.c` that quantizes the input activation matrix X[M,K] to Q8_0 in a transposed-block layout `X_q8t[n_blocks, M_pad]`, so the GEMM inner loop can access all M tokens' Q8_0 blocks contiguously for a given K block index.

2. **INT8 GEMM kernel**: Replaced `q8_gemm_worker` to use `vdotq_s32` for INT8×INT8 dot products. For each Q8_0 block: 2 `vdotq_s32` calls per token (32 elements = 16+16), then `vpaddq_s32` horizontal reduction across 4 tokens, scale by `w_scale * x_scale`. Three fallback tiers: `__ARM_FEATURE_DOTPROD` (vdotq_s32), NEON-only (vmovl_s8 + vmlal_s16), scalar.

3. **Static workspace**: Module-level `gemm_ws` struct with `x_q8t` and `yt` buffers that grow lazily and never shrink. `qwen_gemm_workspace_free()` called from `qwen_free()`.

4. **Fused QKV GEMM**: `qwen_linear_q8_qkv_batched` quantizes the input once and reuses it for Q, K, V projections. Saves 2 redundant quantizations per attention layer (encoder: 4 layers, decoder prefill: 28 layers).

### Results (jfk.wav, 11s audio, 4 threads, Snapdragon)

| Phase | Q8_0 FP32-FMA | INT8 GEMM | Speedup |
|-------|---------------|-----------|---------|
| Mel | 19ms | 19ms | - |
| Encoder | 2152ms | 1750ms | 1.23x |
| Prefill | 2541ms | 1451ms | 1.75x |
| Decode | 1217ms | 1261ms | ~1.0x |
| **Total** | **5910ms** | **4462ms** | **1.32x** |

All 23 tests pass (correctness, repeat inference, unload/reload, error handling).

### Analysis

- **Prefill got the biggest win (1.75x)**: Decoder prefill is dominated by Q8_0 linear projections (QKV, output proj, gate+up, down — 6 GEMMs per layer × 28 layers = 168 GEMMs). The INT8 dot product path is significantly faster than dequantize-to-FP32 + FMA. The fused QKV also saves 2 quantize passes per layer.

- **Encoder improved modestly (1.23x)**: The encoder has only 4 transformer layers (vs 28 in decoder), so the linear layers are a smaller fraction of total time. Bidirectional attention (windowed, 143 tokens) is still compute-bound and untouched by this optimization.

- **Decode unchanged (~1.0x)**: Expected — decode uses seq_len=1 which already takes the matvec path (`q8_matvec_fused_neon` with `vdotq_s32`). The GEMM changes don't affect matvec.

- **Cumulative speedup from baseline**: Total time went from 7131ms (FP32/BF16) → 5910ms (Q8_0 FP32-FMA) → 4462ms (INT8 GEMM). That's **1.60x** overall from the original.

### Files modified

| File | Change |
|------|--------|
| `qwen_asr_quant.h` | Added `quantize_f32_rows_transpose_q8` declaration |
| `qwen_asr_quant.c` | Implemented `quantize_f32_rows_transpose_q8` (NEON-optimized) |
| `qwen_asr_kernels.h` | Added `qwen_linear_q8_qkv_batched` and `qwen_gemm_workspace_free` declarations |
| `qwen_asr_kernels.c` | Replaced `q8_gemm_task_t`/`q8_gemm_worker`/`q8_gemm_batched` with INT8 GEMM; added `gemm_ws` workspace; added `q8_gemm_batched_with_q8t` and `qwen_linear_q8_qkv_batched` |
| `qwen_asr_encoder.c` | Replaced 3× `qwen_linear_q8` Q/K/V with 1× `qwen_linear_q8_qkv_batched` |
| `qwen_asr_decoder.c` | Replaced 3× `qwen_linear_nobias_q8` prefill Q/K/V with 1× `qwen_linear_q8_qkv_batched` |
| `qwen_asr.c` | Added `qwen_gemm_workspace_free()` call in `qwen_free()` |

### Potential next optimizations

1. **KV cache FP16** (medium impact). KV cache is still FP32. Halving it would reduce attention memory bandwidth.

2. **Q4_K_M for decoder** (medium impact). Decoder weights could use 4-bit quantization for ~2x further bandwidth reduction on the 168 GEMM calls per inference.

3. **Tiled GEMM with better L2 reuse** (medium impact). Current INT8 GEMM is K-outer, N-middle, M-inner. A tiled approach (blocking N into cache-friendly chunks) could improve L2 hit rate for larger N dimensions (e.g. gate+up: N=6144).

4. **Attention optimization** (medium impact for encoder). Encoder bidirectional attention is now a larger fraction of the remaining time. Flash-attention style or tiled attention could help.

## End-to-end test suite (2026-02-23)

Automated test script: `scripts/test-asr.sh` (run via `cd app/android-device && bash scripts/test-asr.sh`).

Builds the APK, installs, and runs 9 test scenarios (23 assertions total):

| Test | Scenario | What it covers |
|------|----------|----------------|
| 1 | Build & Install | APK compiles, installs on device |
| 2 | Model Loading | `load_model` broadcast → `nativeLoadModel` → logcat confirms success |
| 3 | WAV Transcription (batch) | `test_wav` → `qwen_transcribe_audio` — correctness check against expected phrase |
| 4 | Performance | Parses Mel/Encoder/Prefill/Decode timing from logcat, fails if above thresholds |
| 5 | Streaming Transcription | `test_wav_stream` → `qwen_transcribe_stream_live` — same codepath as live mic, verifies encoder window cache / prefix rollback / token fix logic |
| 6 | Repeat Inference (batch x2) | Runs `test_wav` again without reloading — catches KV cache not being reset properly between runs |
| 7 | Unload → Reload → Inference | `free_model` → `load_model` → `test_wav` — catches memory leaks, global state residue, post-reload crashes |
| 8 | Error: no model | `free_model` then `test_wav` — asserts `model not loaded` message and no crash (JNI null pointer guard) |
| 9 | Error: bad WAV path | `test_wav` with `/data/local/tmp/nonexistent.wav` — asserts `WAV file not found` and no crash |

### Key implementation details

- **`nativeTestWavStream` (JNI)**: Loads a WAV file, pushes all samples into a `qwen_live_audio_t`, signals EOF, then calls `qwen_transcribe_stream_live`. This exercises the exact same code path as production live-mic ASR but without needing `AudioRecord`.
- **`free_model` command**: Calls `nativeFreeModel()` synchronously on the broadcast receiver thread (fast, no need for a background thread).
- **Test 8 guards against JNI null-pointer crash**: The `nativeTestWav` and `nativeTestWavStream` functions check `g_ctx != nullptr` before proceeding. The Java-side `VoiceDebugReceiver.doTestWav` also checks `File.exists()` before calling JNI, so the "WAV file not found" message comes from `VoiceDebugReceiver` (Java layer), not from native code.
- **Polling pattern**: Tests poll `adb logcat -d` every 1-2s for a sentinel string (e.g. `nativeTestWav: done`), with configurable timeouts (60s for load, 120s for inference).
