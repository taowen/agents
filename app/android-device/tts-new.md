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

## Next steps

1. **Port INT8/Q4_K quantization** from `jni/qwen-tts-old/` to recover 3x talker speedup
2. **Verify audio continuity** by listening — the overlap/trim should produce seamless audio between chunks
3. **Tune chunk_size**: with quantized talker (~88ms/token), chunk_size=10 would give TTFA ≈ 250 + 880 + ~2000 ≈ **3.1s**
4. **Test with speaker/language params**: `--es speaker serena --es language chinese`
