# ASR New: Qwen3-ASR-0.6B on Android

Source: `android/app/src/main/jni/qwen-asr/`. All weights Q8_0 quantized, pre-quantized `.qmodel` for instant loading (~1ms via mmap).

## Testing

```bash
# One-time setup: push model + test audio
adb push /home/taowen/qwen-asr/qwen3-asr-0.6b/ /data/local/tmp/qwen3-asr-0.6b/
adb push /home/taowen/qwen-asr/samples/jfk.wav /data/local/tmp/jfk.wav

# Automated: build → install → batch test → streaming test → performance report
cd app/android-device && bash scripts/test-asr-new.sh

# Manual
PKG="ai.connct_screen.rn"
adb shell "am broadcast -a ${PKG}.VOICE_DEBUG -p ${PKG} --es cmd load_model --es path /data/local/tmp/qwen3-asr-0.6b"
adb shell "am broadcast -a ${PKG}.VOICE_DEBUG -p ${PKG} --es cmd test_wav --es path /data/local/tmp/jfk.wav"
adb shell "am broadcast -a ${PKG}.VOICE_DEBUG -p ${PKG} --es cmd test_wav_stream --es path /data/local/tmp/jfk.wav"
adb shell "am broadcast -a ${PKG}.VOICE_DEBUG -p ${PKG} --es cmd free_model"
adb logcat -d | grep -E 'QwenASR|QwenTTS|VoiceDebug' | tail -30
```

### Gotchas

- **stderr redirect tag**: TTS/ASR JNI both redirect stderr to logcat. If TTS loads first, ASR output appears under `QwenTTS` tag.
- **OOM on long audio**: test.wav (17.6s) causes SIGKILL. Use jfk.wav (11s).
- **Screen-off throttling**: CPU governor may throttle when screen is off, inflating benchmarks ~25%.
- **qmodel vs safetensors**: if both present, qmodel is preferred. Delete `model.qmodel` to fall back.

## Stream Mode Architecture

Audio → 2s chunks. Encoder window = 8s (bidirectional attention). Key mechanisms:

1. **Encoder window cache**: completed 8s windows are cached; only the partial tail is re-encoded each chunk.
2. **Conv2D stem cache**: the encoder's Conv2D stem (3 layers, ~56% of encoder time) processes each ~1s mel chunk independently. Cached stem outputs are reused across chunks — only the last mel chunk (affected by edge padding) and new chunks are recomputed. Mel normalization uses a locked `global_max` for deterministic caching.
3. **Cold-start skip**: first 2 "unfixed" chunks are skipped entirely (their decode output is never emitted). First real chunk processes 6s of audio.
4. **Prefix rollback**: decoder rolls back last K tokens for boundary continuity.
5. **KV cache reuse**: prefill embeddings are compared row-by-row; matching prefix is skipped via KV cache truncation.

Key split: `qwen_encoder_stem_chunk()` (per-chunk Conv2D) + `qwen_encoder_transformer()` (bidirectional attention). `stream_encode_stem_cached()` manages the cache for both partial and complete window encoding.

## Current Performance (jfk.wav, 11s, Snapdragon)

### Batch

| Phase | Time | Tokens |
|-------|------|--------|
| Mel | 12ms | — |
| Encoder | 974ms | 143 |
| Prefill | 1022ms | 158 |
| Decode | 618ms (20.6 ms/tok) | 30 |
| **Total** | **2626ms** | — |

### Streaming (4 chunks, cold-start skipped)

| Chunk | Audio | Encoder | Stem cached | Prefill | Decode | Total |
|-------|-------|---------|-------------|---------|--------|-------|
| 1 | 0-6s | 613 | 0/6 | 660 | 348 | 1621 |
| 2 | 0-8s | 515 | 5/8 | 743 | 430 | 1688 |
| 3 | 8-10s | 252 | 0/2 | 260 | 588 | 1100 |
| 4 | 8-11s | 292 | 1/3 | 352 | 617 | 1261 |

**Totals**: Encoder 1672ms (29%) / Prefill 2015ms (36%) / Decode 1983ms (35%) / **Wall 5670ms**. KV reuse 46%.

### Per-operation breakdown (batch, for profiling reference)

**Encoder**: conv stem 56% · ffn GEMM 15% · GELU 13% · attn GEMM 8% · attention 6%
**Prefill**: MLP 50% · QKV GEMM 22% · attention 10% · out GEMM 11%
**Decode**: MLP 52% · QKV 16% · attention 15% · argmax 16% — all ~20ms/token

## Optimization History

| Round | What | Batch | Stream | Key change |
|-------|------|-------|--------|------------|
| 0 | Q8_0 baseline | 23210ms | — | BF16→Q8_0 quantization at load time |
| 1 | Conv2D Q8 GEMM | 3151ms | — | im2col + Q8_0 GEMM for conv layers 2&3 (23x encoder speedup) |
| 2 | Pre-quantized .qmodel | ~3200ms | — | mmap pre-quantized weights, ~1ms model load |
| 3 | Source split + cleanup | ~3050ms | — | Split oversized files, removed ~2300 lines dead code |
| 4 | Stem cache + cold-start skip | 2626ms | **5670ms** | Conv2D stem cache, cold-start chunk skip, encoder split |

## Next Optimization Opportunities

### 1. Prefill at window boundary (chunk 2: 743ms)

The biggest single prefill. Root cause: at the 8s boundary, the complete window is re-encoded with bidirectional attention over 8s, producing completely new encoder output. All KV cache after the template head (~15 tokens) is invalidated.

**Option A — skip complete window re-encoding**: use the last partial window's encoder output as the cached window. The partial at 6s was encoded with bidirectional attention over 6s — slightly less context but saves ~500ms (encoder) + ~300ms (prefill KV reuse improves). Trade-off: cached window quality slightly lower for subsequent chunks.

**Option B — fixed-length encoder output padding**: pad encoder output to constant length so positions of suffix/text-prefix tokens don't shift between chunks. Would improve KV reuse but requires model behavior testing.

### 2. Encoder GELU (~200ms across streaming, 13% of encoder)

`qwen_gelu` uses scalar `tanhf()`. NEON polynomial approximation (e.g. rational Padé) could cut this by 50-70%. Straightforward implementation, no quality impact.

### 3. Decoder MLP (~50% of per-token cost)

Already Q8_0. Q4_K quantization would reduce MLP GEMM bandwidth ~2x but needs quality validation on ASR output. Affects both prefill MLP and per-token decode.

### 4. Chunk 1 encoder (613ms, no stem cache)

Cold-start skip means no stem cache is built for the first real chunk. Running encoder-only (no prefill/decode) during cold-start would cost ~600ms but only save ~200ms at chunk 1. Net loss with current audio length. Might help for longer audio where window reuse is more frequent.
