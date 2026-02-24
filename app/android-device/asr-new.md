# ASR New: Qwen3-ASR-0.6B on Android

Source: `android/app/src/main/jni/qwen-asr/`. Encoder Q8_0, decoder Q4_K. Pre-quantized `.qmodel` for instant loading (~1ms via mmap).

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
- **Screen-off throttling**: CPU governor may throttle when screen is off, inflating benchmarks ~25%.
- **qmodel vs safetensors**: if both present, qmodel is preferred. Delete `model.qmodel` to fall back.
- **Chinese test audio**: generate with `edge-tts --voice zh-CN-YunxiNeural --text "..." --write-media /tmp/chinese_test.mp3`, convert to 16kHz mono WAV, trim to ≤11s. Push to `/data/local/tmp/chinese_test.wav`. Useful for testing text dedup at chunk boundaries (CJK tokenization differs more across chunks).

## Stream Mode Architecture

Audio → 2s chunks. Encoder window = 8s (bidirectional attention). Key mechanisms:

1. **Encoder window cache**: completed 8s windows are cached; only the partial tail is re-encoded each chunk.
2. **Conv2D stem cache**: the encoder's Conv2D stem (3 layers, ~56% of encoder time) processes each ~1s mel chunk independently. Cached stem outputs are reused across chunks — only the last mel chunk (affected by edge padding) and new chunks are recomputed. Mel normalization uses a locked `global_max` for deterministic caching.
3. **Cold-start skip**: first 2 "unfixed" chunks are skipped entirely (their decode output is never emitted). First real chunk processes 6s of audio.
4. **Prefix rollback**: decoder rolls back last K tokens for boundary continuity.
5. **KV cache reuse**: prefill embeddings are compared row-by-row; matching prefix is skipped via KV cache truncation.
6. **Text-level dedup**: token-level overlap detection fails when the same text is tokenized differently across chunks (different audio context → different encoder output → different token IDs). A text-level fallback searches for the longest suffix of already-emitted `result` as a substring anywhere in the pending new tokens' decoded text, skipping both the overlap and any boundary artifact tokens (e.g., a misplaced comma before the repeated portion).

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
| 5 | Streaming text dedup | — | **5670ms** | Text-level overlap detection fixes cross-chunk repetition |

## Streaming Text Dedup (Round 5)

### Problem

When `past_text_conditioning=0` (default), each chunk decodes independently. At chunk boundaries, the same text can be tokenized differently because the audio context changed (e.g., 6s vs 8s window). The token-level overlap detection (`memcmp` on token IDs) misses these overlaps, causing repeated text in output.

Example (Chinese, 10s audio): chunk N emits "...今天我们要讨论的话题", chunk N+1 decodes "，我们要讨论的话题是：人工智能..." — the token-level LCP diverges early because token IDs differ, so "我们要讨论的话题" gets emitted twice.

### Root cause

The divergence has two components:
1. **Different tokenization**: same text gets different token IDs when audio context changes (bidirectional attention sees different windows).
2. **Boundary artifact tokens**: the new chunk may start with "wrong" tokens (e.g., a comma "，" that belongs to the previous segment). These precede the actual overlapping text.

### Fix: suffix-substring search

After the token-level overlap check, decode pending tokens to text (`new_text`). Search for the longest **suffix** of already-emitted `result` that appears as a **substring** anywhere in `new_text`. If found at offset P, skip P + match_len bytes from the start of new_text (dropping both boundary artifacts and the repeated portion).

Key details:
- Min match: 6 bytes (~2 CJK characters) to avoid false positives
- Max suffix search: 256 bytes (sufficient for typical boundary overlaps)
- Token skip: walk emit_start forward until cumulative decoded bytes ≥ text_skip
- Location: `qwen_asr_stream.c`, between token-level overlap check and emit loop

### Why suffix-of-result, not prefix-of-new_text

The new chunk may produce boundary artifact tokens before the overlap. Searching for a prefix of new_text in result fails because new_text starts with "，" (comma, 3 bytes) which isn't at the end of result. Searching for result's suffix in new_text finds the match at offset 3, correctly skipping the artifact.

### Bug fix: emitted_total log

`n_stable_text_tokens` (candidate window size) was logged as `emitted_total` instead of `n_emitted_text_tokens` (actual tokens sent to user). Fixed to show the correct count.

## Next Optimization Opportunities

### 1. Prefill at window boundary (chunk 2: 743ms)

The biggest single prefill. Root cause: at the 8s boundary, the complete window is re-encoded with bidirectional attention over 8s, producing completely new encoder output. All KV cache after the template head (~15 tokens) is invalidated.

**Option A — skip complete window re-encoding**: use the last partial window's encoder output as the cached window. The partial at 6s was encoded with bidirectional attention over 6s — slightly less context but saves ~500ms (encoder) + ~300ms (prefill KV reuse improves). Trade-off: cached window quality slightly lower for subsequent chunks.

**Option B — fixed-length encoder output padding**: pad encoder output to constant length so positions of suffix/text-prefix tokens don't shift between chunks. Would improve KV reuse but requires model behavior testing.

### 2. Encoder GELU (~200ms across streaming, 13% of encoder)

`qwen_gelu` uses scalar `tanhf()`. NEON polynomial approximation (e.g. rational Padé) could cut this by 50-70%. Straightforward implementation, no quality impact.

### 3. ~~Encoder Q4_K~~ (tested, rejected)

Tested: encoder Q4_K with padded d_model (896→1024). Batch encoder **+17% slower** (974→1139ms), streaming encoder flat (+30ms). Root cause: encoder uses batched GEMM (seq=100-143 tokens) where compute dominates over bandwidth. Q4_K's complex dequant (4-bit unpack + sub-group corrections) is slower than Q8_0's simple INT8×INT8 SDOT. Plus 14% padding overhead for 0.6B (896→1024). Q4_K only helps bandwidth-bound paths (decoder matvec).

### 4. Chunk 1 encoder (613ms, no stem cache)

Cold-start skip means no stem cache is built for the first real chunk. Running encoder-only (no prefill/decode) during cold-start would cost ~600ms but only save ~200ms at chunk 1. Net loss with current audio length. Might help for longer audio where window reuse is more frequent.
