# ASR New: Stream Mode Baseline

## Overview

Fresh start from the original unoptimized `qwen-asr` codebase (`/home/taowen/qwen-asr/`), replacing the previous 5-round optimized version (backed up at `jni/qwen-asr-old/`). Focus: **stream mode** performance optimization on Android.

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

Tests: preflight → build & install → load model → batch (jfk.wav) → stream (jfk.wav) → repeat inference → unload/reload. Checks correctness and prints a performance report (no pass/fail thresholds).

### Manual commands

```bash
PKG="ai.connct_screen.rn"
adb shell "am broadcast -a ${PKG}.VOICE_DEBUG -p ${PKG} --es cmd load_model --es path /data/local/tmp/qwen3-asr-0.6b"
adb shell "am broadcast -a ${PKG}.VOICE_DEBUG -p ${PKG} --es cmd test_wav --es path /data/local/tmp/jfk.wav"
adb shell "am broadcast -a ${PKG}.VOICE_DEBUG -p ${PKG} --es cmd test_wav_stream --es path /data/local/tmp/jfk.wav"
adb shell "am broadcast -a ${PKG}.VOICE_DEBUG -p ${PKG} --es cmd free_model"
adb logcat -d -s QwenASR_JNI VoiceDebug QwenASR | tail -30
```

### Gotchas

- **OOM on long audio**: test.wav (17.6s) causes SIGKILL — FP32 encoder intermediate buffers too large. Use jfk.wav (11s). Quantization should fix this.
- **Must use explicit model path**: `--es path /data/local/tmp/qwen3-asr-0.6b` (default app-internal path is empty).
- **Process restart loses model**: If app is killed (OOM, swipe), must reload. Check PID in logcat.

## Baseline Performance (jfk.wav, 11s, 4 threads, Snapdragon)

### Batch mode

| Phase | Time |
|-------|------|
| Mel | 16ms |
| Encoder | 30128ms |
| Prefill | 15826ms |
| Decode | 1025ms (30 tokens, 34.2 ms/token) |
| **Total** | **~47.0s** |

### Stream mode (6 chunks, 2s each)

Total wall time: ~115s.

| Chunk | Audio Range | Encoder | Prefill | Decode |
|-------|-------------|---------|---------|--------|
| 1 | 0-2s | 4512ms (26 tok) | 2836ms (41 tok) | 356ms (11 tok) |
| 2 | 0-4s | 9418ms (52 tok) | 5881ms (67 tok) | 407ms (13 tok) |
| 3 | 0-6s | 19024ms (78 tok) | 8914ms (93 tok) | 551ms (16 tok) |
| 4 | 0-8s | 24934ms (104 tok) | 11289ms (119 tok) | 707ms (21 tok) |
| 5 | 0-10s | 6399ms (130 tok) | 3072ms (145 tok, reused 113) | 1021ms (29 tok) |
| 6 | 0-11s | 9839ms (143 tok) | 4383ms (158 tok, reused 113) | 1036ms (30 tok) |

Key observations:
- **Encoder dominates**: chunks 1-4 re-encode from 0s each time. Chunk 4 (0-8s) = 25s encoder alone.
- After 8s window fills (chunks 5-6), encoder cache kicks in — chunk 5 drops to 6.4s.
- Prefill reuse: 41% (253/617 tokens).
- Decode is fast: ~30-35 ms/token consistently.

### Comparison with optimized version (from asr.md)

| Metric | Optimized (5 rounds) | Baseline (new) | Ratio |
|--------|---------------------|----------------|-------|
| Batch total (jfk.wav 11s) | 3331ms | 47000ms | 14.1x slower |
| Encoder | ~1134ms | 30128ms | 26.6x slower |
| Prefill | ~1186ms | 15826ms | 13.3x slower |
| Decode | ~1011ms (33.7 ms/tok) | 1025ms (34.2 ms/tok) | ~1.0x |

Decode is the same (matvec-bound, already has NEON matvec). Encoder and prefill are dramatically slower due to FP32 GEMM without INT8 quantization.

## Optimization Roadmap

1. **Q8_0 quantization + INT8 GEMM** — Encoder and prefill dominated by FP32 GEMM. Q8_0 with `vdotq_s32` gave 1.60x before. Target: encoder 30s → ~5-10s.
2. **NEON activation functions** — GELU (encoder), SiLU (decoder). Replace scalar `tanhf`/`expf` with NEON polynomial approximations.
3. **NEON norm/add** — RMSNorm, LayerNorm, add_inplace. Many small calls, aggregate ~1.13x.
4. **Memory optimization** — Reduce peak memory for longer audio. Targets: encoder intermediate buffers, FP16 KV cache.
5. **Attention optimization** — 2-pass attention with NEON expf, NEON RoPE.

Validation per round:
- Host: `asr_regression.py --stream-check-only` and `--stream-cache-check-only`
- Device: `bash scripts/test-asr-new.sh`
