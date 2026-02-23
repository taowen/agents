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

Tests: preflight → build & install → load model → batch (jfk.wav) → performance report with per-operation breakdown.

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
