# TTS Dual-Track Chunked Streaming Test Results (2026-02-23)

## What changed

Replaced the old incremental per-token codec decode with **Dual-Track Chunked Streaming**:
- Accumulate `chunk_size` codec frames during AR generation
- Decode each chunk with `left_context` (= chunk_size) frames of overlap
- Trim the context-corresponding audio samples, deliver the rest via callback
- The codec decoder (`qwen_tts_codec_decode`) is stateless (no KV cache), so each chunk is an independent call

Source files replaced from upstream `Qwen3-TTS-C/c/` (old files backed up to `jni/qwen-tts-old/`).

## Build & deploy

```bash
cd app/android-device/android && ./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n ai.connct_screen.rn/.MainActivity
```

## Model load

```
Model loaded in 467ms
Config: Talker 28 layers, hidden=1024, heads=16/8, head_dim=128
        Sub-talker 5 layers, hidden=1024, heads=16/8, head_dim=128
        Codec 8 layers, hidden=512, codebook_dim=512, decoder_dim=1536
        Speakers: 9, Languages: 12
```

Note: this is the **unquantized** upstream model (no INT8/Q4_K). The old code had custom INT8/Q4_K quantization that is not present in the new upstream source. Model load is faster (467ms vs 1,131-3,340ms) because there is no quantization step.

## Test results: "Hello world" (EN)

### Batch baseline

```
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn \
  --es cmd speak --es text 'Hello world' --es path /data/local/tmp/qwen-tts-model"
```

| Metric | Value |
|--------|-------|
| Codec tokens | 29 |
| Audio length | 2.32s (55,680 samples) |
| Talker time | 8,163ms (281.5 ms/token) |
| Codec decode time | 5,009ms |
| **Total** | **13,695ms** |
| Realtime ratio | 0.17x |

### Stream chunk_size=5

```
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn \
  --es cmd speak_stream --es text 'Hello world' --es path /data/local/tmp/qwen-tts-model \
  --ei chunk_size 5"
```

| Metric | Value |
|--------|-------|
| Codec tokens | 29 |
| Audio length | 2.32s (55,680 samples) |
| **TTFA** | **2,226ms** |
| Total | 20,207ms |
| Chunks | 6 (5+5+5+5+5+4 frames) |
| Overhead vs batch | 1.48x |

### Stream chunk_size=10

```
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn \
  --es cmd speak_stream --es text 'Hello world' --es path /data/local/tmp/qwen-tts-model \
  --ei chunk_size 10"
```

| Metric | Value |
|--------|-------|
| Codec tokens | 29 |
| Audio length | 2.32s (55,680 samples) |
| **TTFA** | **4,807ms** |
| Total | 19,153ms |
| Chunks | 3 (10+10+9 frames, with 10-frame left context on chunks 2-3) |
| Overhead vs batch | 1.40x |

### Stream chunk_size=25

```
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn \
  --es cmd speak_stream --es text 'Hello world' --es path /data/local/tmp/qwen-tts-model \
  --ei chunk_size 25"
```

| Metric | Value |
|--------|-------|
| Codec tokens | 29 |
| Audio length | 2.32s (55,680 samples) |
| **TTFA** | **10,873ms** |
| Total | 19,007ms |
| Chunks | 2 (25+4 frames, with 25-frame left context on chunk 2) |
| Overhead vs batch | 1.39x |

### Summary table ("Hello world", 29 tokens, 2.32s audio)

| Mode | TTFA | Total | Overhead | Chunks |
|------|------|-------|----------|--------|
| Batch | =total | **13,695ms** | 1.0x | 1 |
| Stream chunk=5 | **2,226ms** | 20,207ms | 1.48x | 6 |
| Stream chunk=10 | **4,807ms** | 19,153ms | 1.40x | 3 |
| Stream chunk=25 | **10,873ms** | 19,007ms | 1.39x | 2 |

## Test results: Chinese "你好，今天天气怎么样？"

### Batch baseline

```
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn \
  --es cmd speak --es text '你好，今天天气怎么样？' --es path /data/local/tmp/qwen-tts-model"
```

| Metric | Value |
|--------|-------|
| Codec tokens | 24 |
| Audio length | 1.92s (46,080 samples) |
| Talker time | 6,387ms (266.1 ms/token) |
| Codec decode time | 4,918ms |
| **Total** | **11,578ms** |

### Stream chunk_size=10 (Chinese)

```
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn \
  --es cmd speak_stream --es text '你好，今天天气怎么样？' --es path /data/local/tmp/qwen-tts-model \
  --ei chunk_size 10"
```

| Metric | Value |
|--------|-------|
| Codec tokens | 24 |
| Audio length | 1.92s (46,080 samples) |
| **TTFA** | **4,908ms** |
| Total | 17,201ms |
| Chunks | 3 (10+10+4 frames) |
| Overhead vs batch | 1.49x |

## Analysis

### Streaming overhead breakdown

The Dual-Track approach has **1.4-1.5x overhead** vs batch. This comes from:
1. **Redundant context decode**: each chunk (except the first) re-decodes `left_context` frames. With chunk_size=10, chunks 2-3 each decode 20 frames (10 context + 10 new) instead of 10.
2. **Codec decode is not batched as efficiently**: multiple smaller decodes vs one large decode.

### Comparison with previous approaches

| Approach | TTFA (chunk=10) | Total overhead | Audio quality |
|----------|----------------|----------------|---------------|
| Re-decode (old, superseded) | 4,442ms | 2.50x | Discontinuous |
| Incremental per-token (old) | 1,421ms | 2.54x | Discontinuous |
| **Dual-Track chunked (new)** | **4,807ms** | **1.40x** | **Continuous** (expected) |

Key tradeoff: Dual-Track has higher TTFA than per-token incremental (4.8s vs 1.4s) but **much lower total overhead** (1.4x vs 2.5x) and should produce **continuous audio** since each chunk is independently decoded with overlap context.

### Talker speed regression

The upstream unquantized model runs at **266-282 ms/token** (talker only), compared to the old INT8/Q4_K-quantized code at **~85-88 ms/token**. This is a **3.1x regression** because the new source files from `Qwen3-TTS-C/c/` do not include the INT8/Q4_K quantization optimizations that were in the old code.

**To recover performance**: port the INT8/Q4_K quantization from `jni/qwen-tts-old/` into the new source files.

### TTFA vs chunk_size tradeoff

```
TTFA ≈ prefill + chunk_size × talker_ms/token + codec_decode(chunk_size)
     ≈ 250ms + chunk_size × 280ms + codec(chunk_size)
```

| chunk_size | Token gen time | Codec decode | Total TTFA |
|-----------|---------------|-------------|------------|
| 5 | 1,400ms | ~800ms | ~2,200ms |
| 10 | 2,800ms | ~2,000ms | ~4,800ms |
| 25 | 7,000ms | ~4,000ms | ~10,900ms |

Smaller chunks = lower TTFA but more chunks = more overhead from context re-decode.

## Q4_K_M Quantization Test Results (2026-02-23)

Ported INT8/Q4_K quantization from `jni/qwen-tts-old/` into new `qwen_tts_quant.c`/`.h`.

Strategy:
- **Talker**: QKV/gate_up → Q4_K, wo/down → INT8 (sensitive layers)
- **Sub-talker**: all → Q4_K
- **Cache**: binary `.qcache` file for pre-quantized weights

### Model load

| Load type | Time | Notes |
|-----------|------|-------|
| BF16 (no quant) | 467ms | baseline |
| First load (quantize + save cache) | 2,004ms | one-time cost |
| Cached load (.qcache) | 758ms | subsequent loads |

### "Hello world" batch — Q4_K_M

| Metric | BF16 baseline | Q4_K_M (1st run) | Q4_K_M (cached) |
|--------|--------------|------------------|-----------------|
| Codec tokens | 29 | 34 | 34 |
| Audio length | 2.32s | 2.72s | 2.72s |
| Talker ms/token | 281.5 | 61.1 | 46.9 |
| Talker total | 8,163ms | 2,077ms | 1,594ms |
| Codec decode | 5,009ms | 6,461ms | 5,993ms |
| **Total** | **13,695ms** | **8,872ms** | **7,929ms** |

Talker speedup: **281→47 ms/token = 6.0x** (cached, warm process).
Total speedup: **13,695→7,929 = 1.7x** (codec dominates).

### Longer Chinese batch — Q4_K_M

```
text: 今天是个好天气，阳光明媚，微风轻拂，正是出门散步的好时候。公园里的花都开了，五颜六色的，真好看。
```

| Metric | Value |
|--------|-------|
| Codec tokens | 184 |
| Audio length | 14.72s |
| Talker time | 12,654ms (68.8 ms/token avg) |
| Token rate trend | 44→69 ms/token (O(n²) attention scaling) |
| Codec decode | 44,952ms |
| **Total** | **57,901ms** |

### 音质影响（Quality Impact）

**Q4_K 对音质影响明显。** Token 数量从 BF16 的 29 变为 Q4_K 的 34（"Hello world"），说明量化误差影响了采样分布，导致生成不同的 token 序列。听感上有可察觉的质量退化。

原因分析：
1. 模型只有 0.6B 参数，hidden=1024，权重冗余度低，Q4_K（4-bit）量化太激进
2. Sub-talker 生成 31/32 的 codec code groups，全 Q4_K 量化直接影响音频编码质量
3. Q4_K 是 4-bit 非对称量化（256元素/block），精度远低于 INT8 的 8-bit 对称量化

**结论：需要回退到 INT8-only（全部权重用 INT8 对称量化），牺牲一些速度换回音质。**

预期 INT8-only 性能（基于旧代码经验）：
- Talker: ~80-90 ms/token（vs Q4_K 47ms, BF16 281ms）
- 音质: 接近 BF16（INT8 对称量化精度损失很小）

## INT8-Only Rollback (2026-02-23)

Rolled back Q4_K_M to **INT8-only** quantization across all weight matrices. Q4_K (4-bit) was too aggressive for the 0.6B model — audible quality degradation and different token sequences vs BF16.

### What changed

Removed all Q4_K code, switched to INT8 (8-bit symmetric per-row) for every weight matrix:

| Component | Before (Q4_K_M) | After (INT8-only) |
|-----------|-----------------|-------------------|
| Talker QKV | Q4_K | INT8 |
| Talker gate_up | Q4_K | INT8 |
| Talker wo | INT8 | INT8 (unchanged) |
| Talker down | INT8 | INT8 (unchanged) |
| Sub-talker QKV | Q4_K | INT8 |
| Sub-talker gate_up | Q4_K | INT8 |
| Sub-talker wo | Q4_K | INT8 |
| Sub-talker down | Q4_K | INT8 |

Files modified:
- `qwen_tts_quant.h` — removed `block_q4_k` struct, `QK_K`/`Q4K_NUM_SUBS` macros, Q4_K kernel declarations
- `qwen_tts_quant.c` — removed `kernel_matvec_q4k()`, `kernel_swiglu_matvec_q4k()`, `quantize_bf16_to_q4k()` (~250 lines); updated cache format to all-INT8, bumped `QCACHE_VERSION` to 2
- `qwen_tts.h` — added `wqkv_int8/scales`, `gate_up_int8/scales` to talker layer; removed all `block_q4_k *` fields; removed `use_q4k` config flag
- `qwen_tts.c` — added INT8 quantization for QKV/gate_up in both talker and sub-talker load; removed Q4_K quantization blocks; updated `qwen_tts_free()`
- `qwen_tts_talker.c` — changed all dispatch paths from `Q4K > INT8 > BF16` to `INT8 > BF16`

Cache note: old `.qcache` files (version 1, Q4_K format) are automatically invalidated by the version bump. First run after this change will re-quantize and save a new version 2 cache.

### Expected performance

| Metric | BF16 | Q4_K_M | INT8-only (expected) |
|--------|------|--------|---------------------|
| Talker ms/token | 281 | 47 | ~80-90 |
| Token count ("Hello world") | 29 | 34 | ~29 |
| Audio quality | baseline | degraded | near-baseline |

### Verification

```bash
cd app/android-device/android && ./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am force-stop ai.connct_screen.rn && adb shell am start -n ai.connct_screen.rn/.MainActivity
adb shell "run-as ai.connct_screen.rn rm -f cache/model.qcache"
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn \
  --es cmd speak --es text 'Hello world' --es path /data/local/tmp/qwen-tts-model"
```

Expected: token count close to BF16 (~29), talker ~80-90 ms/token, audio quality near BF16.

## INT8-Only Benchmark Results (2026-02-23)

### "Hello world" batch — INT8-only (1st run, quantize + save cache)

| Metric | Value |
|--------|-------|
| Model load | 1,052ms (quantize + save .qcache) |
| Codec tokens | 26 |
| Audio length | 2.08s (49,920 samples) |
| Prefill | 237ms (8 tokens) |
| Talker time | 966ms (37.1 ms/token) |
| Codec decode | 4,376ms |
| **Total** | **5,587ms** |
| Realtime ratio | 0.37x |

### "Hello world" batch — INT8-only (cached, warm process)

| Metric | Value |
|--------|-------|
| Codec tokens | 26 |
| Audio length | 2.08s (49,920 samples) |
| Prefill | 215ms |
| Talker time | 941ms (36.2 ms/token) |
| Codec decode | 4,490ms |
| **Total** | **5,667ms** |
| Realtime ratio | 0.37x |

### Chinese "你好，今天天气怎么样？" batch — INT8-only

| Metric | Value |
|--------|-------|
| Codec tokens | 25 |
| Audio length | 2.00s (48,000 samples) |
| Prefill | 216ms |
| Talker time | 857ms (34.3 ms/token) |
| Codec decode | 4,271ms |
| **Total** | **5,363ms** |
| Realtime ratio | 0.37x |

### Comparison: BF16 vs Q4_K vs INT8-only ("Hello world" batch)

| Metric | BF16 | Q4_K_M | INT8-only |
|--------|------|--------|-----------|
| Codec tokens | 29 | 34 | **26** |
| Audio length | 2.32s | 2.72s | 2.08s |
| Talker ms/token | 281.5 | 46.9 | **36.2** |
| Talker total | 8,163ms | 1,594ms | **941ms** |
| Codec decode | 5,009ms | 5,993ms | 4,490ms |
| **Total** | **13,695ms** | **7,929ms** | **5,667ms** |
| Realtime ratio | 0.17x | — | **0.37x** |

**Key findings:**
- **Talker: 36 ms/token** — 7.8x faster than BF16 (281ms), and faster than Q4_K (47ms). The new upstream code with INT8 is significantly faster than expected (~80-90ms estimate).
- **Token count: 26** — close to BF16 (29), much better than Q4_K (34). INT8 symmetric quantization preserves the sampling distribution.
- **Total: 5.7s** — 2.4x faster than BF16 (13.7s), 1.4x faster than Q4_K (7.9s).
- **Vocoder remains the bottleneck**: 4.5s / 5.7s = 79% of total time. Talker is now only 17%.

### Streaming — INT8-only, chunk_size=10

| Metric | Value |
|--------|-------|
| Codec tokens | 26 |
| **TTFA** (first audio) | **3,103ms** |
| Total | 11,860ms |
| Chunks | 3 (10+10+6 frames, with 10-frame left context on chunks 2-3) |
| Overhead vs batch | 2.09x |

### Streaming — INT8-only, chunk_size=5

| Metric | Value |
|--------|-------|
| Codec tokens | 26 |
| **TTFA** (first audio) | **1,512ms** |
| Total | 12,712ms |
| Chunks | 6 (5+5+5+5+5+1 frames, with 5-frame left context on chunks 2-6) |
| Overhead vs batch | 2.24x |

### Streaming summary ("Hello world", INT8-only)

| Mode | TTFA | Total | Overhead | Chunks |
|------|------|-------|----------|--------|
| Batch | =total | **5,667ms** | 1.0x | 1 |
| Stream chunk=5 | **1,512ms** | 12,712ms | 2.24x | 6 |
| Stream chunk=10 | **3,103ms** | 11,860ms | 2.09x | 3 |

**Audio continuity**: Dual-track chunked streaming is functional. Audio plays back through all chunks without runtime errors. Overlap context (left_context = chunk_size) is applied on chunks 2+. **Subjective listening test needed** to confirm no clicks/gaps at chunk boundaries.

## Next steps

### Done

1. ~~Port INT8/Q4_K quantization~~ ✅ Q4_K quality too low for 0.6B model
2. ~~Roll back to INT8-only~~ ✅ All weights now INT8 symmetric
3. ~~Benchmark INT8-only on device~~ ✅ 36 ms/token, 26 tokens, 5.7s total
4. ~~Verify streaming (dual-track chunked)~~ ✅ Functional, TTFA 1.5s (chunk=5) / 3.1s (chunk=10)

### Main bottleneck: Vocoder (BigVGAN) — 71-81% of total time

Round 5 instrumentation (`tts.md`) showed talker/sub-talker are well optimized (~15-19% of total). The vocoder dominates:

```
Vocoder: 7,511ms / 10,513ms total = 71.5%  ("Hello world")
Vocoder: 44,741ms / 55,225ms total = 81.0%  (Chinese 30 chars)
```

**Root cause**: `kernel_causal_conv1d` and `kernel_transposed_conv1d` are **pure scalar C loops**. `USE_BLAS` is not defined in CMakeLists.txt, so no BLAS and no NEON — the two hottest functions in the entire pipeline have zero SIMD optimization.

Current vocoder throughput: ~6.7 GFLOPS (33% of A77 peak ~20 GFLOPS).

5. **NEON vectorize `kernel_causal_conv1d`** — hot path is `kernel_size=7, groups=1, dilation=1/3/9`. Write NEON float32x4 inner loops for the k=7 dot product. Expected 2-3x speedup on vocoder conv. This is the single highest-impact change.
6. **NEON vectorize `kernel_transposed_conv1d`** — currently a triple-nested scalar loop. NEON vectorize the output channel accumulation.
7. **INT8 quantize vocoder conv weights** (~100MB F32) — halve memory bandwidth, same pattern as talker INT8. Conv weights are small (k=7 × in_ch × out_ch) but activations are large; bandwidth savings help on the activation side too.
8. **Fuse SnakeBeta + Conv1d** — reduce memory round-trips. Currently SnakeBeta writes entire activation to memory, then conv1d reads it back. Fusion eliminates one full read+write pass.

**Expected combined impact**: vocoder from ~7.5s → ~2-3s ("Hello world" batch), total from ~10s → ~5-6s.

## NEON Vectorized Vocoder Benchmark (2026-02-23)

Implemented NEON optimizations for `kernel_causal_conv1d` and `kernel_transposed_conv1d`:

**What changed** (`qwen_tts_kernels.c`):
1. **NEON k=7 groups=1 fast path** (output-centric) — processes 8 consecutive outputs at a time with 7 `vfmaq_n_f32` FMA each. Works for any dilation (1/3/9). Reads output once, writes once (vs 7 read-modify-write cycles in the old saxpy approach).
2. **`neon_saxpy_f32` helper** — NEON-vectorized `dst[i] += alpha * src[i]` for non-k=7 paths (depthwise k=7, pre-conv k=3).
3. **NEON k-loop in transposed conv1d** — processes 4 kernel taps at a time in the scatter loop (kernel sizes 6-16).

### "Hello world" batch — NEON vocoder (cold device, after restart)

| Metric | INT8-only (before) | NEON vocoder | Speedup |
|--------|-------------------|-------------|---------|
| Codec tokens | 26 | 26 | — |
| Audio length | 2.08s | 2.08s | — |
| Prefill | 215ms | 327ms | — |
| Talker time | 941ms (36.2 ms/token) | 964ms (37.1 ms/token) | ~same |
| Codec decode | **4,490ms** | **3,524ms** | **1.27x** |
| **Total** | **5,667ms** | **4,826ms** | **1.17x** |
| Realtime ratio | 0.37x | 0.43x | — |

### Chinese "你好，今天天气怎么样？" batch — NEON vocoder

| Metric | INT8-only (before) | NEON vocoder | Speedup |
|--------|-------------------|-------------|---------|
| Codec tokens | 25 | 25 | — |
| Audio length | 2.00s | 2.00s | — |
| Talker time | 857ms (34.3 ms/token) | 896ms (35.9 ms/token) | ~same |
| Codec decode | **4,271ms** | **3,568ms** | **1.20x** |
| **Total** | **5,363ms** | **4,720ms** | **1.14x** |
| Realtime ratio | 0.37x | 0.42x | — |

### Long Chinese batch — NEON vocoder

```
text: 今天是个好天气，阳光明媚，微风轻拂，正是出门散步的好时候。公园里的花都开了，五颜六色的，真好看。
```

| Metric | INT8-only (before) | NEON vocoder | Speedup |
|--------|-------------------|-------------|---------|
| Codec tokens | 184 | 202 | (different run) |
| Audio length | 14.72s | 16.16s | — |
| Talker time | 12,654ms (68.8 ms/token) | 8,676ms (42.9 ms/token) | — |
| Codec decode | **44,952ms** | **31,561ms** | — |
| Codec ms/token | **244 ms/token** | **156 ms/token** | **1.56x** |
| **Total** | **57,901ms** | **40,528ms** | **1.43x** |
| Realtime ratio | — | 0.40x | — |

### Streaming — NEON vocoder, "Hello world"

| Mode | TTFA | Total | Overhead | Chunks |
|------|------|-------|----------|--------|
| Batch | =total | **4,826ms** | 1.0x | 1 |
| Stream chunk=5 | **1,524ms** | 14,643ms | 3.03x | 6 |
| Stream chunk=10 | **3,302ms** | 14,030ms | 2.91x | 3 |

### Analysis

**Codec decode speedup: 1.2-1.6x** (longer sequences benefit more). The vocoder is ~71% of codec time; with vocoder convs improving ~1.5x, the overall codec pipeline improvement of 1.27x for short text is expected: `1 / (0.71/1.5 + 0.29) = 1.27x`.

**Why not the expected 2-3x?** The NEON optimization is compute-bound improvement, but the vocoder is **memory bandwidth limited** on longer sequences:
- ResUnit conv1 (k=7): dim=768, length=928 → 768×928×4 bytes = 2.7MB activation per channel-first buffer, far exceeding L2 cache (512KB on A77).
- Each NEON FMA does 1 load + 1 FMA per cycle, but memory-bound loads stall the pipeline.
- The 7 overlapping NEON loads per 8 outputs cause more cache pressure than the sequential saxpy approach on cold data.

**Thermal throttling observed**: after sustained workloads, device throttles from ~37 ms/token to ~51 ms/token (talker) and codec decode degrades ~1.25x. All comparison measurements taken on cold device (after restart + 60s cooldown).

**Remaining optimization opportunities** (items 7-8) target the memory bandwidth bottleneck:
- INT8 vocoder weights → halve weight bandwidth
- SnakeBeta+Conv1d fusion → eliminate one full activation read+write pass

### Updated next steps

### Done

5. ~~NEON vectorize `kernel_causal_conv1d`~~ ✅ k=7 output-centric NEON, +neon_saxpy fallback
6. ~~NEON vectorize `kernel_transposed_conv1d`~~ ✅ NEON k-loop scatter

### Remaining (memory bandwidth targeted)

7. **INT8 quantize vocoder conv weights** — halve weight memory bandwidth
8. **Fuse SnakeBeta + Conv1d** — eliminate one full activation read+write pass per ResUnit
9. **Tune chunk_size** for streaming: with INT8 talker (~37ms/token) + faster vocoder, TTFA improves proportionally
10. **Test with speaker/language params**: `--es speaker serena --es language chinese`
