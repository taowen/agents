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

1. ~~**Tiled GEMM with better L2 reuse**~~ Done — see Round 3 below.

2. **KV cache FP16** (medium impact). KV cache is still FP32. Halving it would reduce attention memory bandwidth.

3. **Q4_K_M for decoder** (medium impact). Decoder weights could use 4-bit quantization for ~2x further bandwidth reduction on the 168 GEMM calls per inference.

4. **Attention optimization** (medium impact for encoder). Encoder bidirectional attention is now a larger fraction of the remaining time. Flash-attention style or tiled attention could help.

## NEON activations + Q8_0 LM Head + GEMM N-tiling (2026-02-23)

Round 3 optimization: three independent improvements targeting different bottlenecks.

### Problem

Three bottlenecks identified from Round 2 profiling (4462ms total):

1. **LM Head argmax reads 311MB BF16/token**: `qwen_argmax_matvec_bf16` scans 151936 × 1024 × 2 bytes = 311MB per decode token. ~30 tokens → ~9.3GB DRAM. Q8_0 would reduce to 175MB (44% savings) with `vdotq_s32` for faster compute.

2. **Scalar GELU (encoder) and SiLU (decoder)**: `qwen_gelu` calls `tanhf()` per element (9.2M calls across 18 encoder layers). `swiglu_worker` calls `expf()` per element (13.3M calls across 28 decoder layers). ARM scalar tanhf/expf costs ~20-50 cycles each; NEON polynomial approximation: ~5-8 cycles per 4 elements.

3. **GEMM Yt working set exceeds L1**: `q8_gemm_worker` loop structure K-outer → N-middle → M-inner. For encoder fc1 (N=3584, 4 threads): each thread processes ~896 rows. Yt working set = 896 × 144 × 4 bytes = 516KB, far exceeding L1D (32KB). Every `vld1q_f32`/`vst1q_f32` on Yt hits L2 (~10 cycles) instead of L1 (~4 cycles).

### Solution

**1. NEON GELU and SiLU vectorization** (`qwen_asr_kernels.c`)

- Added `neon_expf()`: 7th-order minimax polynomial for `2^f` on [-0.5, 0.5], combined with integer exponent scaling via `vshlq_n_s32`. Max error ~1e-5 vs standard `expf()`. Uses `vrndnq_f32` for banker's rounding.
- Added `neon_tanhf()` via `tanh(x) = 1 - 2/(1 + exp(2x))`, reusing `neon_expf`.
- Vectorized `qwen_gelu()`: processes 4 elements per iteration with `neon_tanhf`, scalar tail for remainder.
- Vectorized SiLU in `swiglu_worker()` for both non-alias (prefill) and in-place (decode seq=1) paths. Uses `vld2q_f32` to deinterleave gate/up pairs, `neon_expf` for sigmoid, `vdivq_f32` for the division. In-place write is safe because `out[j]` writes position j while reading positions 2j and 2j+1 (2j >= j for j >= 1 when scanning forward).

**2. Q8_0 LM Head argmax** (7 files)

- Added `tok_embeddings_q8` field to `qwen_decoder_t`. Quantized from BF16 at model load time (~175MB Q8_0 vs 311MB BF16).
- New `qwen_argmax_matvec_q8()`: quantizes input x to Q8_0 once, then computes INT8 dot products against all 151936 vocab rows. Multi-threaded (splits vocab rows across threads).
- NEON kernel (`qwen_argmax_q8_range_neon`): 2-row processing, block-unrolled by 2, both `__ARM_FEATURE_DOTPROD` and widening-multiply fallback paths (same structure as existing `qwen_q8_matvec_fused_neon`).
- Embedding lookup (`tok_embed_bf16_to_f32`) unchanged — still uses mmap'd BF16 (reads only 1 row = 1024 values per token, negligible).

**3. GEMM N-tiling** (`qwen_asr_kernels.c`)

- Added N-tile outer loop to `q8_gemm_worker`. Tile size Nc = 32768 / (M_pad × 4). For M_pad=144: Nc=56.
- New loop order: `for n_tile → for kb → for n in tile → for m`. Yt[Nc, M_pad] = 56 × 144 × 4 = 32KB → fits L1D.
- x_col per kb (5KB) and W blocks per tile per kb (2KB) also fit L1D. Total ~39KB per tile.
- Trade-off: x_col is re-read from L2 for each N-tile, but x_col total is small (~140KB) and stays warm in L2.
- Applied to all three code paths (dotprod, NEON-no-dotprod, scalar).

### Results (jfk.wav, 11s audio, 4 threads, Snapdragon)

| Phase | Round 2 | Round 3 | Speedup |
|-------|---------|---------|---------|
| Mel | 19ms | 19ms | - |
| Encoder | 1750ms | 1385ms | 1.26x |
| Prefill | 1451ms | 1477ms | ~1.0x |
| Decode | 1261ms | 1061ms | 1.19x |
| **Total** | **4462ms** | **3923ms** | **1.14x** |

All 23 tests pass (correctness, streaming, repeat inference, unload/reload, error handling).

Cumulative from baseline: 7131ms → 3923ms = **1.82x**.

### Analysis

- **Encoder improved 1.26x**: Combination of NEON GELU (eliminated 9.2M scalar `tanhf` calls) and N-tiled GEMM (Yt now stays in L1D). The encoder has 18 layers of GELU-activated FFN (3584 dim), so both optimizations contribute meaningfully.

- **Prefill unchanged (~1.0x)**: Surprising — expected improvement from NEON SiLU + N-tiled GEMM. Possible reasons: (1) prefill SiLU time was smaller than estimated (SwiGLU MLP is gate×up after activation, the linear projections dominate); (2) N-tiling helps less when the N dimension is smaller (decoder intermediate=3072 vs encoder ffn=3584), and the decoder has fewer tokens per prefill; (3) measurement variance between runs may mask a small improvement.

- **Decode improved 1.19x**: Q8_0 argmax replacing BF16 argmax. Each token now reads 175MB instead of 311MB for the LM head scan. The 1.19x is below the theoretical 1.78x (bandwidth ratio) likely because: (1) argmax is only part of decode time — the 28 decoder layers' matvec + attention are unchanged; (2) Q8_0 dot product has slightly more overhead per element than BF16 FMA (block scale handling).

### Files modified

| File | Change |
|------|--------|
| `qwen_asr_kernels.c` | NEON `neon_expf`/`neon_tanhf` helpers; vectorized `qwen_gelu` and `swiglu_worker`; N-tiled `q8_gemm_worker`; threaded `qwen_argmax_matvec_q8` |
| `qwen_asr.h` | Added `tok_embeddings_q8` field to `qwen_decoder_t` |
| `qwen_asr_decoder.c` | Quantize tok_embeddings BF16→Q8_0 at load; replaced `qwen_argmax_matvec_bf16` with `qwen_argmax_matvec_q8` |
| `qwen_asr_kernels.h` | Declared `qwen_argmax_matvec_q8` |
| `qwen_asr_kernels_impl.h` | Added `qwen_argmax_q8_range_{neon,generic}` declarations and dispatch macros |
| `qwen_asr_kernels_neon.c` | `qwen_argmax_q8_range_neon` with dotprod + fallback |
| `qwen_asr_kernels_generic.c` | `qwen_argmax_q8_range_generic` scalar reference |
| `qwen_asr.c` | Free `tok_embeddings_q8` in `qwen_free` |

### Potential next optimizations

1. **KV cache FP16** (medium impact). KV cache is still FP32. Halving it would reduce attention memory bandwidth, especially for decode phase (28 layers × growing sequence).

2. **Q4_K_M for decoder** (medium impact). Decoder weights could use 4-bit quantization for ~2x further bandwidth reduction on the 168 GEMM calls per inference.

3. **Attention optimization** (medium impact for encoder). Encoder bidirectional attention is now the largest remaining bottleneck. Flash-attention style tiled computation could reduce memory traffic.

4. **Prefill investigation**: Prefill didn't improve despite NEON SiLU + N-tiling. Profiling individual layers would reveal whether the bottleneck is attention (KV cache writes, softmax) rather than MLP.

## NEON Norm/Add + stack argmax (2026-02-23)

Round 4 optimization: added NEON SIMD paths for all normalization and element-wise add operations that were falling back to scalar on ARM, plus eliminated per-token malloc/free in argmax.

### Problem

Code review found three categories of scalar fallback on ARM and one unnecessary allocation:

1. **`qwen_rms_norm` / `qwen_rms_norm_per_head` scalar on ARM**: Had AVX512/AVX2 SIMD paths but the `#else` branch was pure scalar loops. Called ~15,500 times (RMSNorm) and ~248K times (per-head) per inference. Two-pass loops (sum_sq + scale_multiply) over hidden=896/1024 or head_dim=128 elements.

2. **`qwen_layer_norm` scalar on ARM**: Same pattern — AVX512/AVX2 paths existed, `#else` was scalar. Called ~5,148 times (encoder only, 18 layers × 2 norms × 143 tokens). Three-pass loops (mean + variance + normalize) over hidden=896.

3. **`qwen_add_inplace` no SIMD at all**: `for (int i = 0; i < n; i++) a[i] += b[i];` — no architecture-specific optimization. Called for every residual connection: ~15.2M float adds per inference.

4. **`qwen_argmax_matvec_q8` per-token malloc/free**: Allocated ~1152 bytes via `malloc`/`free` on every decode token (~30 calls). Fixed buffer size, safe for stack allocation.

### Solution

**1. NEON `qwen_rms_norm`** — Inserted `#elif defined(__ARM_NEON)` between `__AVX2__` and `#else` for both phases:
- sum_sq: Two `float32x4_t` accumulators with `vfmaq_f32` (fused multiply-add), `vaddvq_f32` horizontal reduction. 8-wide unroll per iteration.
- scale_multiply: `vdupq_n_f32(rms_inv)` broadcast, `vmulq_f32(vmulq_f32(x, w), scale)`. 8-wide unroll.

**2. NEON `qwen_rms_norm_per_head`** — Same pattern as above, operating on `vec`/`head_dim` (128 elements, exactly 16 iterations of 8-wide). In-place write back.

**3. NEON `qwen_layer_norm`** — Three NEON phases:
- mean: `vaddq_f32` accumulation + `vaddvq_f32` reduction, two accumulators.
- variance: `vsubq_f32(x, mean)` + `vfmaq_f32` squared-diff accumulation, two accumulators.
- normalize: `vsubq_f32` + `vmulq_f32` × 2 (inv_std, weight) + `vaddq_f32` (bias). 8-wide unroll.

**4. NEON `qwen_add_inplace`** — 8-wide `vaddq_f32` + `vst1q_f32` with scalar tail, wrapped in `#ifdef __ARM_NEON` / `#else`.

**5. Stack-allocated argmax buffer** — Replaced `malloc`/`free` of `x_q8` with `block_q8_0 x_q8[64]` (fixed-size stack array, ~2.3KB, sufficient for hidden=1024 which needs 32 blocks).

### Results (jfk.wav, 11s audio, 4 threads, Snapdragon)

| Phase | Round 3 | Round 4 | Speedup |
|-------|---------|---------|---------|
| Mel | 19ms | 19ms | - |
| Encoder | 1385ms | 1161ms | 1.19x |
| Prefill | 1477ms | 1275ms | 1.16x |
| Decode | 1061ms | 1041ms | 1.02x |
| **Total** | **3923ms** | **3477ms** | **1.13x** |

All 23 tests pass (correctness, streaming, repeat inference, unload/reload, error handling).

Cumulative from baseline: 7131ms → 3477ms = **2.05x**.

### Analysis

- **Encoder improved 1.19x**: NEON LayerNorm (5,148 calls × hidden=896) was the primary contributor. The encoder uses LayerNorm (3-pass: mean + variance + normalize) which had more scalar overhead per call than RMSNorm (2-pass). NEON add_inplace also contributed (~36 residual adds × 143 × 896 elements).

- **Prefill improved 1.16x**: NEON RMSNorm (~8,680 calls × hidden=1024) and NEON RMSNorm_per_head (~208K calls × head_dim=128) both contributed. The per-head normalization has extremely high call count because it runs per-head per-token per-layer (28 layers × 2 × 155 tokens × 24 heads). Even though each call is small (128 elements), the aggregate NEON vs scalar difference adds up.

- **Decode improved marginally (1.02x)**: Expected — decode runs only ~30 tokens, so norm calls (~1,680 RMSNorm + ~40K per-head) are far fewer. The stack allocation for argmax saved trivial overhead. Decode is dominated by matvec DRAM bandwidth, not normalization compute.

- **Actual savings (~446ms) exceeded the conservative estimate (~150-240ms)**: The plan underestimated the per-head norm impact. With ~248K per-head calls, even small per-call savings (scalar ~128 cycles → NEON ~20 cycles) compound significantly at 2GHz: ~248K × 108 cycles saved = ~27M cycles = ~13ms just from per-head norm. The larger-than-expected encoder improvement suggests LayerNorm's 3-pass structure benefited more from NEON than the 2-pass RMSNorm estimate predicted.

### Files modified

| File | Change |
|------|--------|
| `qwen_asr_kernels.c` | NEON paths for `qwen_rms_norm`, `qwen_rms_norm_per_head`, `qwen_layer_norm`, `qwen_add_inplace`; stack alloc in `qwen_argmax_matvec_q8` |

### Cumulative optimization history

| Round | Optimization | Total | Speedup vs prev | Cumulative |
|-------|-------------|-------|-----------------|------------|
| Baseline | FP32/BF16 | 7131ms | - | 1.00x |
| Round 1 | Q8_0 weight quantization | 5910ms | 1.21x | 1.21x |
| Round 2 | INT8 GEMM + workspace pre-alloc | 4462ms | 1.32x | 1.60x |
| Round 3 | NEON activations + Q8_0 LM Head + N-tiling | 3923ms | 1.14x | 1.82x |
| Round 4 | NEON Norm/Add + stack argmax | 3477ms | 1.13x | 2.05x |
| Round 5 | NEON RoPE + FP16 KV Cache + 2-pass attention | 3331ms | 1.04x | 2.14x |

### Potential next optimizations

1. **Q4_K_M for decoder** (medium impact). Decoder weights could use 4-bit quantization for ~2x further bandwidth reduction on the 168 GEMM calls per inference.

2. **Attention optimization** (medium impact for encoder). Encoder bidirectional attention is now the largest remaining bottleneck. Flash-attention style tiled computation could reduce memory traffic.

3. **Prefill profiling**: Per-layer breakdown would clarify whether GEMM or attention dominates the remaining ~1186ms prefill time.

## NEON RoPE + FP16 KV Cache + 2-pass attention (2026-02-23)

Round 5 optimization: three changes targeting RoPE vectorization, KV cache bandwidth reduction, and attention softmax vectorization.

### Problem

After 4 rounds of optimization, GEMM/matvec (INT8 vdotq_s32), norm/activation (NEON) were all optimized. Remaining bottlenecks:

1. **`qwen_apply_rope_neox` scalar on ARM**: AVX512/AVX2 paths existed, but `#else` fallback was pure scalar. Called for every head in every attention layer. Encoder: ~46M scalar FMA ops, Prefill: ~33M ops.

2. **KV Cache FP32**: `float *kv_cache_k/v` at 4 bytes/element. Decode attention reads 28 layers × ~185 positions × 1024 × 4 bytes × 2 (K+V) = ~42MB/token, ×30 tokens = ~1.26GB. FP16 would halve this.

3. **Scalar `expf()` in attention**: Online softmax called scalar `expf()` ~37.6M times (encoder) + ~5.3M times (prefill). ARM scalar `expf` costs ~30-50 cycles; `neon_expf` does 4 values in ~8 cycles.

### Solution

**1. NEON RoPE** (`qwen_asr_kernels.c`)

Inserted `#elif defined(__ARM_NEON)` path in `qwen_apply_rope_neox` between AVX2 and scalar fallback:
- `vsubq_f32(vmulq_f32(x1, cos), vmulq_f32(x2, sin))` for first half
- `vfmaq_f32(vmulq_f32(x2, cos2), x1, sin2)` for second half
- 4-wide processing, scalar tail loop (unnecessary since half=32/64 always divides by 4)

**2. FP16 KV Cache** (4 files)

- `qwen_asr.h`: Changed `float *kv_cache_k/v` → `uint16_t *kv_cache_k/v`
- `qwen_asr_decoder.c`: `kv_cache_init`/`kv_cache_grow` use `sizeof(uint16_t)`, `kv_cache_k_at`/`kv_cache_v_at` return `uint16_t *`, KV writes use `qwen_f32_to_f16()`
- `qwen_asr_kernels.h`: `qwen_causal_attention` signature changed to `const uint16_t *K_fp16/V_fp16`, added `qwen_f32_to_f16`/`qwen_f16_to_f32` declarations
- `qwen_asr_kernels.c`: New NEON-accelerated conversion functions, plus mixed-precision helpers:
  - `qwen_dot_f32_f16()`: dot(FP32 Q, FP16 K) with on-the-fly `vcvt_f32_f16` conversion pipelined with `vfmaq_f32`
  - `qwen_vec_axpy_f16_inplace()`: `dst += alpha * FP16_src`
  - `qwen_vec_scale_add_f16()`: `dst = dst * correction + FP16_src`

Note: encoder bidirectional attention is unchanged (operates on FP32 Q/K/V tensors directly, no KV cache).

**3. 2-pass attention with NEON expf** (`qwen_asr_kernels.c`)

Replaced online softmax (sequential `expf` with data-dependent max tracking) with 3-pass algorithm:
- **Pass 1**: Compute all scores via dot products into stack buffer `float scores[ATTN_MAX_KEYS]` (max 2048 = 8KB), find max score
- **Pass 2**: NEON batch `neon_expf` 4-wide: `scores[j] = neon_expf(scores[j] - max_score)`, accumulate sum
- **Pass 3**: Weighted V sum: `o_row += scores[j] * inv_sum * V_row[j]`

Applied to both `qwen_bidirectional_attention_heads` (encoder, FP32 K/V) and `qwen_causal_attention_heads` (decoder, FP16 K/V).

Trade-off: V is now read in a separate pass instead of being fused with the softmax loop. But V working set is small (encoder: ~104 × 64 × 4 = 26KB, decoder: ~185 × 128 × 2 = 47KB FP16) and stays in L2 cache, so the re-read cost is minimal.

### Results (jfk.wav, 11s audio, 4 threads, Snapdragon)

| Phase | Round 4 | Round 5 | Speedup |
|-------|---------|---------|---------|
| Mel | 19ms | 19ms | - |
| Encoder | 1161ms | 1134ms | 1.02x |
| Prefill | 1275ms | 1186ms | 1.07x |
| Decode | 1041ms | 1011ms | 1.03x |
| **Total** | **3477ms** | **3331ms** | **1.04x** |

All 23 tests pass (correctness, streaming, repeat inference, unload/reload, error handling).

Cumulative from baseline: 7131ms → 3331ms = **2.14x**.

### Analysis

The overall 1.04x speedup (146ms saved) is below the conservative estimate of ~250-480ms. Reasons:

- **Encoder: only 1.02x (27ms saved)**. The 2-pass attention + NEON expf was expected to save ~110-180ms on encoder. The small actual improvement suggests that the encoder bottleneck has shifted away from attention `expf` to other operations — likely the dot products in Pass 1 and the V accumulation in Pass 3 now dominate. The windowed attention inner loop was already partially latency-hidden by the NEON dot product computation, so removing the sequential expf dependency gave less benefit than predicted. NEON RoPE also contributed minimally (~27ms shared with attention improvement).

- **Prefill: 1.07x (89ms saved)**. FP16 KV cache writes + NEON RoPE + 2-pass attention combined for a modest improvement. The FP16 write bandwidth savings during prefill are real but small (writing is a small fraction of total prefill time). The 2-pass attention helps prefill less than encoder because prefill has fewer total `expf` calls (~5.3M vs ~37.6M).

- **Decode: 1.03x (30ms saved)**. FP16 KV cache reduced read bandwidth from ~42MB/token to ~21MB/token. The expected ~60-100ms saving was offset by the overhead of the mixed-precision dot product (`qwen_dot_f32_f16` with on-the-fly `vcvt_f32_f16` is slightly slower per-element than a pure FP32 dot). The net effect is positive but smaller than the raw bandwidth calculation suggested.

- **Why the estimates were off**: The plan predicted savings based on cycle counts of individual operations (scalar expf, DRAM bandwidth) without accounting for: (1) instruction-level parallelism already hiding some latency; (2) the overhead of FP16↔FP32 conversion instructions in the mixed-precision path; (3) the 2-pass algorithm's extra V re-read cost; (4) the fact that at this optimization stage, the easy wins are gone and remaining bottlenecks are more distributed across many operations.

### Files modified

| File | Change |
|------|--------|
| `qwen_asr_kernels.c` | NEON RoPE; FP16 conversion functions + mixed-precision dot/axpy/scale_add; 2-pass attention for both bidirectional and causal; `ATTN_MAX_KEYS=2048` |
| `qwen_asr_kernels.h` | `qwen_f32_to_f16`/`qwen_f16_to_f32` declarations; `qwen_causal_attention` signature changed to FP16 K/V |
| `qwen_asr.h` | `kv_cache_k/v` type `float *` → `uint16_t *` |
| `qwen_asr_decoder.c` | Cache init/grow/at use `uint16_t`; KV writes use `qwen_f32_to_f16`; attention calls pass FP16 pointers |

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
