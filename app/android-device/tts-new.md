# TTS Engine Notes (Qwen3-TTS 0.6B, ARM A77)

## Architecture

**Dual-Track Chunked Streaming**: accumulate `chunk_size` codec frames during AR generation, decode each chunk with `left_context` overlap, trim context audio, deliver via callback. Codec decoder is stateless.

Source: upstream `Qwen3-TTS-C/c/`, backed up to `jni/qwen-tts-old/`.

## Build & deploy

```bash
cd app/android-device/android && ./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n ai.connct_screen.rn/.MainActivity
```

## Model config

```
Talker 28 layers, hidden=1024, heads=16/8, head_dim=128
Sub-talker 5 layers, hidden=1024, heads=16/8, head_dim=128
Codec 8 layers, hidden=512, codebook_dim=512, decoder_dim=1536
Speakers: 9, Languages: 12
```

## Quantization: INT8 is the only viable option

All talker + sub-talker weight matrices use INT8 (8-bit symmetric per-row quantization). Pre-quantized weights cached in `.qcache` file.

### Q4_K rejected — twice

**Round 1 (custom Q4_K)**: Token count 29→34, audible quality degradation.

**Round 2 (standard ggml Q4_K_M, 2026-02-24)**: Replaced custom Q4_K with standard ggml implementation (256-element super-blocks, 6-bit scales/mins). New files: `qwen_tts_ggml_quants.h/c`. Results:

| Metric | BF16 | Q4_K custom | Q4_K ggml | INT8 |
|--------|------|-------------|-----------|------|
| Tokens ("Hello world") | 29 | 34 | **18** | **26** |
| Audio length | 2.32s | 2.72s | **1.44s** | 2.08s |
| Tokens ("你好今天天气") | 24 | — | **38** | **25** |
| Audio length | 1.92s | — | **3.04s** | 2.00s |
| Talker ms/token | 281.5 | 46.9 | **74.2** | **36.2** |
| Model load (first) | 467ms | 2,004ms | **50,362ms** | 1,052ms |
| Model load (cached) | — | 758ms | **595ms** | — |

**Q4_K ggml is worse than the custom Q4_K in every way:**
1. **Token count distortion is severe and inconsistent**: EN gets too few tokens (18 vs 29), CN gets too many (38 vs 24). The sampling distribution is broken.
2. **Talker is 2x slower than INT8** (74 vs 36 ms/token): Q4_K vec_dot has higher per-element overhead (6-bit scale unpacking, 256-element super-blocks) that dominates at hidden=1024. The bandwidth savings don't compensate.
3. **First quantization takes 50s** (vs INT8's 1s): `quantize_row_q4_K_ref` is expensive — `make_qkx2_quants` does iterative grid search per sub-block.
4. **Long Chinese also degraded**: 178 tokens / 75.7 ms/token / 46.8s total (vs INT8+NEON: 202 tokens / 42.9 ms/token / 40.5s).

**Conclusion: 4-bit quantization is fundamentally unsuitable for this 0.6B model.** The model has hidden=1024 with low weight redundancy. INT8 (8-bit symmetric per-row) is the right trade-off: near-BF16 quality, faster than Q4_K, negligible quantization cost.

## Benchmark: "Hello world" batch (INT8, best result)

| Metric | BF16 | INT8-only |
|--------|------|-----------|
| Codec tokens | 29 | **26** |
| Talker ms/token | 281.5 | **36.2** |
| Talker total | 8,163ms | **941ms** |
| Codec decode | 5,009ms | 4,490ms |
| **Total** | **13,695ms** | **5,667ms** |

INT8: 7.8x talker speedup vs BF16, token count close to BF16, near-baseline audio quality.

## NEON vocoder optimization (applied)

NEON k=7 output-centric fast path (8 outputs × 7 FMA), NEON k-loop scatter for transposed conv.

| Metric | Before NEON | After NEON | Speedup |
|--------|------------|-----------|---------|
| Codec decode ("Hello world") | 4,490ms | 3,524ms | 1.27x |
| Codec ms/token (long CN) | 244 | 156 | 1.56x |
| Total ("Hello world") | 5,667ms | **4,826ms** | 1.17x |

## Streaming benchmarks (INT8 + NEON vocoder, "Hello world")

| Mode | TTFA | Total | Overhead |
|------|------|-------|----------|
| Batch | =total | **4,826ms** | 1.0x |
| Stream chunk=5 | **1,524ms** | 14,643ms | 3.03x |
| Stream chunk=10 | **3,302ms** | 14,030ms | 2.91x |

## Vocoder bottleneck breakdown (26 tokens, "Hello world")

| Component | Time (ms) | % of vocoder |
|-----------|-----------|-------------|
| Conv1 (k=1) | ~1,080 | 35% |
| TransConv | ~1,066 | 34% |
| Conv7 (k=7) | ~900 | 29% |
| SnakeBeta + add | ~50 | 2% |

Vocoder is 71-81% of total time. Memory-bandwidth limited (activation buffers >> L2 512KB).

**Cache tiling was tried and reverted** — 2.6x regression due to L1 output thrashing and redundant SnakeBeta recomputation.

## Test commands

```bash
# Batch
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn \
  --es cmd speak --es text 'Hello world' --es path /data/local/tmp/qwen-tts-model"

# Stream
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn \
  --es cmd speak_stream --es text 'Hello world' --es path /data/local/tmp/qwen-tts-model \
  --ei chunk_size 10"

# Delete qcache (force re-quantization)
adb shell "run-as ai.connct_screen.rn rm -f cache/model.qcache"
```

## Next steps

1. ~~**Validate Q4_K ggml re-implementation**~~ — FAILED: token count distortion (18 vs 29), 2x slower than INT8. **Revert to INT8.**
2. **Optimize TransConv** (~1,066ms, 34%) — GEMM-per-tap approach instead of scatter
3. **Optimize k=1 conv** (~1,080ms, 35%) — profile compute vs memory bound
4. **INT8 vocoder conv weights** — halve weight memory bandwidth
5. **Tune chunk_size** for streaming
6. **Test speaker/language params**: `--es speaker serena --es language chinese`
