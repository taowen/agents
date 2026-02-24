# ASR New: Stream Mode Baseline

## Overview

Fresh start from the original unoptimized `qwen-asr` codebase, with Q8_0 quantization applied at load time. Source code lives in two places:
- **Host (x86)**: `/home/taowen/qwen-asr/` — for local testing, no NEON
- **Android (ARM64)**: `android/app/src/main/jni/qwen-asr/` — actual device build, has NEON DOTPROD

The Android JNI copy is the one that gets built into the APK. Changes must be applied to the JNI copy to take effect on device.

## Stream Mode: How It Works

1. **Chunked encoding**: Audio split into fixed-size windows (`--enc-window-sec`, default 8s). Each window → Whisper encoder → audio embeddings.
2. **Prefix rollback**: Decoder "rolls back" common prefix tokens from previous chunk for continuity across chunk boundaries.
3. **Incremental decoding**: Each stream step generates up to `stream_max_new_tokens` (default 32) tokens.
4. **Encoder cache**: Default on. Previously encoded frames cached to skip redundant computation on overlapping regions.

## On-Device Testing

### Setup

```bash
# Push model + test audio (one-time)
adb push /home/taowen/qwen-asr/qwen3-asr-0.6b/ /data/local/tmp/qwen3-asr-0.6b/
adb push /home/taowen/qwen-asr/samples/jfk.wav /data/local/tmp/jfk.wav
```

### Automated test

```bash
cd app/android-device && bash scripts/test-asr-new.sh
```

Tests: preflight → build & install → load model → batch (jfk.wav) → batch performance → streaming (jfk.wav) → per-chunk streaming performance with stem cache hits.

### Manual commands

```bash
PKG="ai.connct_screen.rn"
adb shell "am broadcast -a ${PKG}.VOICE_DEBUG -p ${PKG} --es cmd load_model --es path /data/local/tmp/qwen3-asr-0.6b"
adb shell "am broadcast -a ${PKG}.VOICE_DEBUG -p ${PKG} --es cmd test_wav --es path /data/local/tmp/jfk.wav"
adb shell "am broadcast -a ${PKG}.VOICE_DEBUG -p ${PKG} --es cmd test_wav_stream --es path /data/local/tmp/jfk.wav"
adb shell "am broadcast -a ${PKG}.VOICE_DEBUG -p ${PKG} --es cmd free_model"
adb logcat -d | grep -E 'QwenASR|QwenTTS|VoiceDebug' | tail -30
```

### Gotchas

- **stderr redirect tag**: TTS JNI (`qwen_tts_jni.c`) and ASR JNI (`qwen_asr_jni.cpp`) both redirect stderr to logcat. Whichever runs first grabs the fd. If TTS loads first, ASR stderr output appears under `QwenTTS` tag. Use unfiltered `adb logcat -d` or grep both tags.
- **OOM on long audio**: test.wav (17.6s) causes SIGKILL — FP32 encoder intermediate buffers too large. Use jfk.wav (11s).
- **Must use explicit model path**: `--es path /data/local/tmp/qwen3-asr-0.6b` (default app-internal path is empty).
- **Process restart loses model**: If app is killed (OOM, swipe), must reload. Check PID in logcat.

## Profiling Infrastructure

Per-operation profiling added to encoder, decoder prefill, and decoder forward. Controlled by `qwen_verbose >= 3` (set in JNI `nativeTestWav`/`nativeTestWavStream`).

### What's measured

**Encoder** (`qwen_asr_encoder.c`):
- `conv` — Conv2D stem (3 layers + reshape + project + sinusoidal PE)
- `attn_proj` — QKV + output projection (Q8_0 GEMM)
- `attn` — bidirectional windowed attention
- `ffn_proj` — fc1 + fc2 (Q8_0 GEMM)
- `ffn_act` — GELU activation
- `norm` — LayerNorm
- `proj` — final projection (proj1 GELU + proj2)

**Decoder prefill** (`qwen_asr_decoder.c`):
- `qkv` — Q/K/V projection (Q8_0 GEMM, 3 separate)
- `attn` — causal GQA attention
- `out_proj` — output projection (Q8_0 GEMM)
- `mlp` — gate_up fused + SwiGLU + down (Q8_0 GEMM)
- `norm_rope` — RMSNorm (input + per-head Q/K + post-attn) + RoPE

**Decoder forward** (per-token, accumulated in `ctx->prof_dec_*`, printed from `qwen_asr.c`):
- `qkv` — norm + QKV projection + per-head norm + RoPE
- `attn` — causal attention over full KV cache
- `mlp` — out_proj + post-attn norm + gate_up + SwiGLU + down
- `argmax` — final norm + argmax matvec (Q8_0 tok_embeddings)

## Current Performance: Conv2D Q8_0 GEMM (jfk.wav, 11s, Snapdragon)

### Batch mode — phase timing

| Phase | Time | Tokens |
|-------|------|--------|
| Mel | 18ms | — |
| Encoder | 1609ms | 143 |
| Prefill | 858ms | 158 |
| Decode | 666ms (22.2 ms/tok) | 30 |
| **Total** | **3151ms** | — |

### Batch mode — per-operation breakdown

**Encoder (1609ms):**

| Operation | Time | % |
|-----------|------|---|
| conv (Conv2D stem) | **903ms** | **56.1%** |
| attn_proj (QKV+O GEMM) | 134ms | 8.3% |
| attn (bidir attention) | 102ms | 6.3% |
| ffn_proj (fc1+fc2 GEMM) | 241ms | 15.0% |
| ffn_act (GELU) | 214ms | 13.3% |
| norm (LayerNorm) | 3ms | 0.2% |
| proj (final projection) | 10ms | 0.6% |

**Prefill (858ms):**

| Operation | Time | % |
|-----------|------|---|
| qkv (Q/K/V GEMM) | 188ms | 21.9% |
| attn (causal attention) | 88ms | 10.3% |
| out_proj (O GEMM) | 93ms | 10.8% |
| mlp (gate_up+SwiGLU+down) | **433ms** | **50.5%** |
| norm_rope | 11ms | 1.3% |

**Decode (666ms, 30 tokens):**

| Operation | Time | % | Per token |
|-----------|------|---|-----------|
| qkv (norm+proj+rope) | 108ms | 16.2% | 3.6ms |
| attn (causal over KV) | 101ms | 15.2% | 3.4ms |
| mlp (O+norm+gate_up+down) | **349ms** | **52.4%** | 11.6ms |
| argmax (norm+matvec) | 103ms | 15.5% | 3.4ms |

### Key findings

1. **Conv2D Q8_0 GEMM resolved the bottleneck**: 20857ms → 903ms (**23x speedup**). The im2col + Q8_0 GEMM approach routes Conv2D layers 2&3 (K=4320, multiple of 32) through the existing multi-threaded INT8 GEMM with NEON `vdotq_s32`. Layer 1 (K=9) stays FP32 and is negligible.

2. **Encoder is now balanced**: Conv2D is 56% of encoder time (903ms), transformer layers are 44% (~700ms). No single operation dominates overwhelmingly.

3. **Total inference 3.15s for 11s audio** — real-time factor 0.29x. Faster than real-time on a mobile SoC.

4. **Decoder/prefill unchanged**: Prefill ~858ms, decode ~666ms — consistent with previous measurements. The Conv2D optimization didn't affect these phases.

5. **Correctness preserved**: Full JFK transcription matches expected output exactly.

## Optimization History

### Round 1: Q8_0 Conv2D GEMM (20857ms → 903ms, 23x)

**What**: Route Conv2D layers 2&3 through existing Q8_0 INT8 GEMM infrastructure instead of naive scalar FP32 GEMM.

**How**:
- `im2col_transposed` outputs `[spatial_out, patch_size]` — the layout `qwen_linear_q8` expects as `X[M, K]`
- `qwen_linear_q8` internally: quantize input to Q8_0, multi-threaded INT8 GEMM with NEON `vdotq_s32` + N-tiling, bias handling, transpose back
- Pre-quantize conv2/conv3 weights (480×4320 each) to Q8_0 at load time. K=4320 = 135×32, block boundaries align with row boundaries
- Layer 1 (K=9, not multiple of 32) stays FP32 — only ~150ms, not worth optimizing

**Files changed**: `qwen_asr.h` (struct fields), `qwen_asr_encoder.c` (quantize at load + call q8 conv), `qwen_asr_kernels.h/.c` (im2col_transposed + qwen_conv2d_q8), `qwen_asr.c` (free)

**Key insight**: The Conv2D GEMM has the same structure as a linear layer — `Y = X @ W^T + bias` where X is the im2col matrix and W is the reshaped conv weight. By transposing im2col output to match the Q8_0 GEMM's expected input layout, we reuse the entire optimized GEMM pipeline (quantization, tiling, threading, NEON) with zero new SIMD code.

## Remaining Optimization Opportunities

### Encoder conv (903ms, 56% of encoder)

Still the largest single operation. Potential further optimizations:
- **Winograd F(2,3)**: For 3×3 convolutions, reduces multiplications by ~2.25x. Would need custom Winograd kernels.
- **im2col overhead**: The transposed im2col itself takes time for the large spatial dimensions. Could be NEON-optimized.
- **Output transpose**: The `[spatial_out, c_out] → [c_out, spatial_out]` transpose after GEMM could be avoided by restructuring downstream code to accept row-major output.

### Encoder GELU (214ms, 13% of encoder)

`qwen_gelu` is scalar `tanhf()`. Could use NEON polynomial approximation or lookup table.

### Decoder MLP (349ms, 52% of decode)

Already Q8_0 quantized. Could try Q4_K_M for decoder weights but diminishing returns given decode total is only 666ms.

## Performance Comparison

| Version | Encoder | Prefill | Decode | Total |
|---------|---------|---------|--------|-------|
| FP32 baseline | 30128ms | 15826ms | 1025ms | 47000ms |
| Q8_0 baseline | 21505ms | 1054ms | 635ms | 23210ms |
| **Q8_0 + Conv2D Q8 GEMM** | **1609ms** | **858ms** | **666ms** | **3151ms** |
| Optimized 5-round (old, from asr.md) | 1134ms | 1186ms | 1011ms | 3331ms |

The Conv2D Q8_0 GEMM optimization reduced total inference from 23.2s to 3.15s (**7.4x overall speedup**). The new codebase now matches the old heavily-optimized version's performance, with cleaner code and a more maintainable architecture.

## Round 2: Pre-quantized .qmodel (near-instant model loading)

### What

Offline pre-quantization of all ASR weights into a single flat binary (`.qmodel`) that the device mmap's directly — eliminates runtime quantization (BF16→Q8_0 + gate/up fusion) at model load time.

### Binary format

128-byte header (magic `0x384D5141` / "AQM8", version 1, all config dimensions as uint32), followed by all weights laid out sequentially: encoder conv stem → encoder layers × N → encoder post → decoder tok_embeddings (bf16 + q8_0) → decoder layers × 28 → final norm. Q8_0 block = 36 bytes (4-byte float scale + 32 int8 qs). 4-byte aligned throughout, no padding needed.

### File sizes

| File | Size |
|------|------|
| model.safetensors (original, BF16) | 1.8 GB |
| **model.qmodel (pre-quantized)** | **1.1 GB** |

Larger than the initial 950 MB estimate because `tok_embeddings` is stored twice: bf16 (311 MB, for embedding lookup) + Q8_0 (175 MB, for argmax matvec).

### Model load time

**Before (safetensors):** mmap safetensors → quantize ~200 weight tensors BF16→Q8_0 → fuse gate+up per decoder layer. Several seconds of CPU-heavy work.

**After (qmodel):** `open()` + `fstat()` + `mmap(PROT_READ, MAP_PRIVATE)` + walk cursor. **~1 ms.**

Logcat confirms all timestamps identical:
```
22:36:01.094 Found pre-quantized model: /data/local/tmp/qwen3-asr-0.6b/model.qmodel
22:36:01.095 Loading pre-quantized .qmodel: Qwen3-ASR-0.6B
22:36:01.095 qmodel loaded: 1192212864 / 1192212864 bytes used (1137.0 MB)
22:36:01.095 Model loaded (qmodel).
```

### Inference performance (unchanged)

| Phase | safetensors path | qmodel path |
|-------|-----------------|-------------|
| Mel | 18ms | 17ms |
| Encoder | 1609ms | 1662ms |
| Prefill | 858ms | 1058ms |
| Decode | 666ms | 646ms |
| Total | 3151ms | 3383ms |

Within normal variance. Inference performance identical — same quantized weights, just pre-computed.

### Files changed

| File | Change |
|------|--------|
| `qwen_asr.h` | Added `qmodel_mmap`, `qmodel_mmap_size` to `qwen_ctx_t` |
| `qwen_asr.c` | Added `qwen_load_qmodel()`, updated `qwen_load()` + `qwen_free()` |
| `ModelManager.java` | `model.safetensors` → `model.qmodel` in download list |

### Gotchas

1. **qmodel and safetensors coexist** — if both are present, qmodel is preferred. Delete `model.qmodel` to fall back.
2. **All weight pointers into mmap** — `qwen_free()` must NOT `free()` individual weight pointers when loaded from qmodel. Only `munmap()` the whole region.

## Round 3: Source file split & dead code removal

### What

Refactored qwen-asr source: split two oversized files (`qwen_asr.c` 2511 lines, `qwen_asr_kernels.c` 2936 lines) by logical domain, and removed ~2300 lines of dead code. Target is Android ARM64 only, so x86 AVX kernels, generic scalar fallbacks, BF16/F32 weight paths, and CLI-only functions were all dead.

### Result

| File | Before | After |
|------|--------|-------|
| `qwen_asr.c` | 2511 | 638 |
| `qwen_asr_transcribe.c` | — | 658 |
| `qwen_asr_stream.c` | — | 1117 |
| `qwen_asr_kernels.c` | 2936 | 1162 |
| `qwen_asr_kernels_ops.c` | — | 906 |
| `qwen_asr_kernels_neon.c` | 1182 | 759 |
| `qwen_asr_kernels_generic.c` | 323 | DELETED |
| `qwen_asr_kernels_avx.c` | 587 | DELETED |

No file exceeds ~1162 lines. ~2300 lines of dead code removed. Build passes, all 6 test checks pass, transcription matches reference exactly.

### Post-refactor performance (jfk.wav, 11s)

| Phase | Time | Tokens |
|-------|------|--------|
| Mel | 26ms | — |
| Encoder | 1144ms | 143 |
| Prefill | 1137ms | 158 |
| Decode | 741ms (24.7 ms/tok) | 30 |
| **Total** | **3048ms** | — |

Within normal variance of previous baseline (3151ms). No regression.

### What was removed

- **x86 SIMD**: `qwen_asr_kernels_avx.c` (587 lines), all `#ifdef __AVX2__`/`__AVX512F__` blocks in norms/RoPE
- **Generic scalar fallbacks**: `qwen_asr_kernels_generic.c` (323 lines)
- **BF16/F32 weight paths**: `bf16_gemm_batched`, `f32_gemm_batched`, `bf16_matvec_*`, `f32_matvec_*`, all BF16/F32 linear/QKV/argmax dispatch — superseded by Q8_0/Q4_K
- **CLI-only functions**: `qwen_transcribe` (WAV file), `qwen_transcribe_stdin`, `qwen_transcribe_stream` (non-live), `qwen_set_prompt`, `qwen_set_force_language`, language selection infrastructure
- **Dead activations**: `qwen_silu` (unused), `qwen_mul_inplace`, `qwen_scale`, `qwen_copy`

### What was split

- `qwen_asr.c` → `qwen_asr.c` (load/free/config) + `qwen_asr_transcribe.c` (batch) + `qwen_asr_stream.c` (streaming)
- `qwen_asr_kernels.c` → `qwen_asr_kernels.c` (thread pool + Q8/Q4K GEMM/matvec + conv2d) + `qwen_asr_kernels_ops.c` (norm, GELU, SwiGLU, softmax, attention, FP16, RoPE)

### New files

| File | Role |
|------|------|
| `qwen_asr_internal.h` | Shared declarations + prompt constants for split qwen_asr*.c files |
| `qwen_asr_transcribe.c` | Batch transcription: silence compaction, segment splitting, `qwen_transcribe_audio` |
| `qwen_asr_stream.c` | Streaming: encoder cache, rollback, `stream_impl`, `qwen_transcribe_stream_live` |
| `qwen_asr_kernels_ops.c` | Norms, activations, attention, position embeddings, FP16 conversion |

### Architecture decisions

- **`parallel_for` renamed to `qwen_parallel_for`** and made non-static (was static in `qwen_asr_kernels.c`, now callable from `qwen_asr_kernels_ops.c` via `qwen_asr_kernels_impl.h`).
- **`qwen_get_n_threads()` added** — ops file needs thread count for conditional parallelization but can't access the static `tp` struct directly.
- **Prompt constants moved to `qwen_asr_internal.h`** — needed by both transcribe and stream files. Defined as `static const int[]` in the header (each TU gets its own copy, fine for small arrays).
- **`compact_silence` and `transcribe_segment` made non-static** — shared between transcribe (defines them) and stream (may reference them via internal header).

### Gotchas

1. **`transpose_back` must be kept** even though `transpose_pad` is dead. Q8 GEMM uses `transpose_back` to convert transposed output `Yt[N, M_pad]` back to row-major `Y[M, N]`. Deleting it breaks the Q8 GEMM path.
2. **`neon_expf`/`neon_tanhf` are only used in ops** (GELU, SwiGLU, attention softmax). They moved cleanly to `qwen_asr_kernels_ops.c` with no duplication issues.
3. **Prompt constants are shared across 3 files** (`qwen_asr.c`, `qwen_asr_transcribe.c`, `qwen_asr_stream.c`). Initially left in `qwen_asr.c` which caused build errors. Moved to `qwen_asr_internal.h`.

## Round 4: Streaming Conv2D stem cache + cold-start truncation (7.2s → 7.0s)

### What

Three optimizations targeting streaming mode overhead:

1. **Conv2D stem cache for partial windows**: The encoder's Conv2D stem processes each ~1s mel chunk independently (no cross-chunk interaction). Only the transformer layers use bidirectional attention. By caching Conv2D stem outputs for unchanged mel chunks across streaming steps, redundant computation is skipped.

2. **Stem cache reuse for complete window encoding**: When a partial window (0-6s) becomes a complete window (0-8s), the stem cache entries from partial processing are reused. At chunk 4 (the 8s boundary), 5 out of 8 mel chunks are served from cache.

3. **Cold-start decode truncation**: The first 2 "unfixed" chunks generate tokens that are never emitted. Limiting to 5 tokens (enough for language detection + `<asr_text>`) saves ~300ms.

### Key insight: mel normalization determinism

`qwen_asr_audio.c` uses `(val + 4.0f) / 4.0f` — a fixed linear transform. The only `global_max` dependency is the clamping floor (`min_val = global_max - 8`). Locking `global_max` from the first partial chunk makes mel values deterministic for existing frames, enabling exact Conv2D stem caching.

### Encoder split: stem_chunk + transformer

`qwen_encoder_forward` was split into two functions:

- `qwen_encoder_stem_chunk()` — processes one mel chunk through Conv2D stem → reshape → project → sinusoidal PE. Returns `[n_tokens, d_model]`.
- `qwen_encoder_transformer()` — runs transformer layers + final projection on concatenated stem outputs. Returns `[total_tokens, output_dim]`.

The original `qwen_encoder_forward` now delegates to these. Batch mode behavior is identical.

### Streaming performance (jfk.wav, 11s)

| Chunk | Encoder ms | Stem cached | Prefill ms | Decode ms | Total |
|-------|-----------|-------------|-----------|----------|-------|
| 1 (0-2s) | 254 | 0/2 | 293 | 98 | 645 |
| 2 (0-4s) | 365 | 1/4 | 422 | 117 | 904 |
| 3 (0-6s) | 419 | 3/6 | 597 | 359 | 1375 |
| 4 (0-8s) | **504** | **5/8** | 767 | 444 | **1715** |
| 5 (8-10s) | 215 | 0/2 | 267 | 592 | 1074 |
| 6 (8-11s) | 255 | 1/3 | 361 | 632 | 1248 |

| Metric | Value |
|--------|-------|
| Encoder total | 2012 ms |
| Prefill total | 2707 ms |
| Decode total | 2242 ms |
| **Wall time** | **6961 ms** |
| Prefill KV reuse | 41.0% |

### Improvement vs no stem cache

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Chunk 4 encoder | 815ms | 504ms | **-311ms** (38%) |
| Chunk 4 stem hits | 0/2 | 5/8 | 5 chunks reused |
| Chunk 4 total | 2018ms | 1715ms | **under 2s threshold** |
| Encoder total | 2260ms | 2012ms | -248ms |
| Wall time | 7166ms | 6961ms | -205ms |

### Files changed

| File | Change |
|------|--------|
| `qwen_asr_audio.h` | Added `preset_global_max` parameter to `qwen_mel_spectrogram` |
| `qwen_asr_audio.c` | Implemented preset global_max: lock on first call, reuse on subsequent |
| `qwen_asr.h` | Added `qwen_encoder_stem_chunk`, `qwen_encoder_transformer`, `qwen_encoder_stem_tokens` |
| `qwen_asr_encoder.c` | Split `qwen_encoder_forward` into stem_chunk + transformer |
| `qwen_asr_stream.c` | Added `stream_encode_stem_cached` helper, stem cache for both partial and complete windows, cold-start decode truncation |
| `qwen_asr_transcribe.c` | Updated `qwen_mel_spectrogram` caller (pass NULL) |

### Batch performance (unchanged)

| Phase | Time | Tokens |
|-------|------|--------|
| Mel | 12ms | — |
| Encoder | 1045ms | 143 |
| Prefill | 1093ms | 158 |
| Decode | 622ms (20.7 ms/tok) | 30 |
| **Total** | **2772ms** | — |

## Remaining Optimization Opportunities

### Prefill (2707ms, 39% of streaming wall time)

Largest component. KV reuse is only 41% — 59% of prefill tokens are recomputed each chunk. Potential:
- **Encoder output caching across chunks**: the cached-window encoder outputs don't change between chunks, but their embeddings are still prefilled each time. Could cache decoder KV state for the encoder portion.
- **Sparse prefill**: skip attention computation for tokens that haven't changed.

### Decode (2242ms, 32% of streaming wall time)

Grows with chunk index as more tokens are generated. Decoder MLP is 52% of per-token cost.
- **Q4_K decoder weights**: diminishing returns given decode is already ~20ms/token.
- **Speculative decoding**: use a smaller model for draft tokens.

### Encoder GELU (est. ~200ms)

Scalar `tanhf()`. NEON polynomial approximation could cut this significantly.

## Performance Comparison

| Version | Batch Total | Stream Wall |
|---------|-------------|-------------|
| FP32 baseline | 47000ms | — |
| Q8_0 baseline | 23210ms | — |
| Q8_0 + Conv2D Q8 GEMM | 3151ms | — |
| + qmodel (Round 2) | 3383ms | — |
| + source split (Round 3) | 3048ms | — |
| **+ stem cache + cold-start (Round 4)** | **2772ms** | **6961ms** |
