# ASR New: Qwen3-ASR-0.6B on Android

Source: `android/app/src/main/jni/qwen-asr/`. Encoder Q8_0, decoder Q4_K. Pre-quantized `.qmodel` for instant loading (~1ms via mmap).

## Model Download

Model weights are downloaded at runtime from ModelScope or HF mirror. Native code auto-quantizes BF16→Q8_0/Q4_K on first load and saves `.qcache` for instant subsequent loads.

```bash
# Download from ModelScope (default, best for China)
adb shell "am broadcast -a ai.connct_screen.rn.VOICE_DEBUG -p ai.connct_screen.rn --es cmd download_model"

# Download from HF mirror
adb shell "am broadcast -a ai.connct_screen.rn.VOICE_DEBUG -p ai.connct_screen.rn --es cmd download_model --es source HF_MIRROR"

# Check download/model status
adb shell "am broadcast -a ai.connct_screen.rn.VOICE_DEBUG -p ai.connct_screen.rn --es cmd status"

# Monitor download progress
adb logcat -s VoiceDebug ModelManager
```

Files downloaded (~1.88 GB total):
- `model.safetensors` (1.876 GB) — raw BF16 weights
- `vocab.json` (2.78 MB) — tokenizer vocabulary
- `merges.txt` (1.67 MB) — BPE merges

Stored in app internal storage: `context.getFilesDir()/qwen3-asr-0.6b/`. First `load_model` after download takes extra time for quantization; subsequent loads use mmap'd `.qcache` (~1ms).

## Testing

```bash
# Alt: push model files manually for development
adb push /home/taowen/qwen-asr/qwen3-asr-0.6b/ /data/local/tmp/qwen3-asr-0.6b/
adb push /home/taowen/qwen-asr/samples/jfk.wav /data/local/tmp/jfk.wav

# Generate complex Chinese test audio (19.6s, tests long-audio streaming)
uvx edge-tts --voice zh-CN-YunxiNeural --text "二零二五年三月，人工智能技术取得了重大突破。研究人员在Nature杂志上发表了一篇论文，指出大语言模型的参数量已经突破了一万亿。这项技术不仅改变了搜索引擎的工作方式，还深刻影响了医疗诊断和自动驾驶等领域。" --write-media /tmp/complex_chinese.mp3
ffmpeg -i /tmp/complex_chinese.mp3 -ar 16000 -ac 1 -sample_fmt s16 /tmp/complex_chinese.wav -y
adb push /tmp/complex_chinese.wav /data/local/tmp/complex_chinese.wav

# Automated: build → install → batch test → streaming test → performance report
cd app/android-device && bash scripts/test-asr-new.sh

# Manual
PKG="ai.connct_screen.rn"
adb shell "am broadcast -a ${PKG}.VOICE_DEBUG -p ${PKG} --es cmd load_model --es path /data/local/tmp/qwen3-asr-0.6b"
adb shell "am broadcast -a ${PKG}.VOICE_DEBUG -p ${PKG} --es cmd test_wav --es path /data/local/tmp/jfk.wav"
adb shell "am broadcast -a ${PKG}.VOICE_DEBUG -p ${PKG} --es cmd test_wav_stream --es path /data/local/tmp/jfk.wav"
adb shell "am broadcast -a ${PKG}.VOICE_DEBUG -p ${PKG} --es cmd test_wav_stream --es path /data/local/tmp/complex_chinese.wav"
adb shell "am broadcast -a ${PKG}.VOICE_DEBUG -p ${PKG} --es cmd free_model"
adb logcat -d | grep -E 'QwenASR|QwenTTS|VoiceDebug' | tail -30
```

### Gotchas

- **stderr redirect tag**: TTS/ASR JNI both redirect stderr to logcat. If TTS loads first, ASR output appears under `QwenTTS` tag.
- **Screen-off throttling**: CPU governor may throttle when screen is off, inflating benchmarks ~25%.
- **qmodel vs safetensors**: if both present, qmodel is preferred. Delete `model.qmodel` to fall back.
- **Chinese test audio**: generate with `uvx edge-tts --voice zh-CN-YunxiNeural --text "..." --write-media /tmp/chinese_test.mp3`, convert to 16kHz mono WAV with `ffmpeg -i /tmp/chinese_test.mp3 -ar 16000 -ac 1 -sample_fmt s16 /tmp/chinese_test.wav`. Push to `/data/local/tmp/`. No length limit — streaming handles long audio via past text conditioning.

## Stream Mode Architecture

Audio → 2s chunks. Encoder window = 8s (bidirectional attention). `past_text_conditioning=1` (default). Key mechanisms:

1. **Encoder window cache**: completed 8s windows are cached; only the partial tail is re-encoded each chunk.
2. **Conv2D stem cache**: the encoder's Conv2D stem (3 layers, ~56% of encoder time) processes each ~1s mel chunk independently. Cached stem outputs are reused across chunks — only the last mel chunk (affected by edge padding) and new chunks are recomputed. Mel normalization uses a locked `global_max` for deterministic caching.
3. **Cold-start skip**: first 2 "unfixed" chunks are skipped entirely (their decode output is never emitted). First real chunk processes 6s of audio.
4. **Past text conditioning**: previously decoded tokens (minus rollback) are fed as prefix to the next chunk's decoder. This means `max_new=32` only needs to cover *incremental* text (~6-8 tokens per 2s chunk), not the full transcription from scratch. Critical for audio >11s.
5. **Prefix rollback**: decoder rolls back last K tokens for boundary continuity.
6. **KV cache reuse**: prefill embeddings are compared row-by-row; matching prefix is skipped via KV cache truncation.
7. **Text-level dedup**: token-level overlap detection fails when the same text is tokenized differently across chunks (different audio context → different encoder output → different token IDs). A text-level fallback searches for the longest suffix of already-emitted `result` as a substring anywhere in the pending new tokens' decoded text, skipping both the overlap and any boundary artifact tokens (e.g., a misplaced comma before the repeated portion).

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

### Streaming (jfk.wav, 4 chunks, cold-start skipped, prefix=on)

| Chunk | Audio | Encoder | Stem cached | Prefill | Decode | Total |
|-------|-------|---------|-------------|---------|--------|-------|
| 1 | 0-6s | 583 | 0/6 | 639 | 348 | 1570 |
| 2 | 0-8s | 470 | 5/8 | 825 | 223 | 1518 |
| 3 | 0-10s | 232 | 0/2 | 365 | 287 | 884 |
| 4 | 0-11s | 252 | 1/3 | 488 | 156 | 896 |

**Totals**: Encoder 1537ms (32%) / Prefill 2317ms (48%) / Decode 1014ms (21%) / **Wall 4868ms**. KV reuse 42%.

### Streaming (complex_chinese.wav, 19.6s, 9 chunks, prefix=on)

| Chunk | Audio | Encoder | Stem cached | Prefill (prefix) | Decode | Total |
|-------|-------|---------|-------------|-------------------|--------|-------|
| 1 | 0-6s | 600 | 0/6 | 670 (0) | 552 | 1822 |
| 2 | 0-8s | 485 | 5/8 | 918 (17) | 235 | 1638 |
| 3 | 0-10s | 200 | 0/2 | 441 (23) | 281 | 922 |
| 4 | 0-12s | 392 | 1/4 | 704 (31) | 201 | 1297 |
| 5 | 0-14s | 429 | 3/6 | 945 (35) | 247 | 1621 |
| 6 | 0-16s | 576 | 5/8 | 1148 (41) | 225 | 1949 |
| 7 | 0-18s | 215 | 0/2 | 677 (46) | 252 | 1144 |
| 8 | 0-19.6s | 354 | 1/4 | 898 (52) | 165 | 1417 |

**Totals**: Encoder 3251ms (27%) / Prefill 6401ms (53%) / Decode 2158ms (18%) / **Wall 11810ms**. KV reuse 52.5%.

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
| 6 | Past text conditioning | — | **4868ms** | `past_text_conditioning=1`: decoder continues from previous output, fixes content loss on >11s audio, decode tokens/chunk drops from 23-32 to 7-17 |

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

## Past Text Conditioning (Round 6)

### Problem

With `past_text_conditioning=0` and `max_new=32`, each chunk decodes the full transcription from scratch. For audio ≤11s (~30 text tokens), 32 tokens suffice. For longer audio (19.6s, ~59 text tokens), the decoder hits the `max_new` cap before reaching new content. Result: `emitted_total` stalls across chunks, no new text is emitted, eventually recovery reset triggers and drops the middle of the transcription.

Example (19.6s Chinese): chunks 4-8 all produced `emitted_total=24` (identical to chunk 3). After recovery reset, only the tail "影响了医疗诊断和自动驾驶等领域。" was recovered. The middle ~40% of the text was lost.

### Fix

Set `past_text_conditioning=1` (in `qwen_asr.c`). The decoder receives previously generated tokens (minus rollback) as prefix input. With prefix feeding:
- `max_new=32` only needs to cover incremental text per chunk (~6-8 tokens for 2s of audio)
- No audio length limit — the prefix grows naturally with the transcription
- Decode time per chunk drops significantly (7-17 tokens vs 23-32 tokens)

### Impact

| Metric | prefix=off | prefix=on |
|--------|-----------|-----------|
| jfk.wav (11s) stream | 5670ms | **4868ms** (−14%) |
| 19.6s Chinese stream | content loss | **11810ms** (correct) |
| Decode tokens/chunk | 23-32 | 7-17 |
| Prefill tokens/chunk | grows with encoder only | grows with encoder + prefix |

Trade-off: prefill sequence is longer (encoder + prefix tokens), so prefill time per chunk increases. But total wall time decreases because decode time drops more than prefill grows.

## Streaming Slowdown Analysis (A/B test, chunk=1s)

### Problem

Streaming mode gets progressively slower. Per-chunk latency increases with audio length.

### Root cause

Prefill cost grows because **every chunk invalidates KV cache from the encoder output position onward**. Two sources:

1. **Encoder output changes every chunk** (main cause, ~66% of growth): the encoder uses bidirectional attention within each 8s window. As the partial window grows (1s→2s→...→8s), all positions' outputs change. The row-by-row KV comparison diverges at the first changed encoder token, forcing re-prefill of everything after it.
2. **Prefix tokens grow** (secondary, ~34%): `past_text_conditioning` appends previously decoded text (0→55 tokens over 19.6s). These tokens sit after the encoder output, so they're always part of the invalidated region.

Note: the encoder uses **local attention** (`cu_seqlens` windowed). Windows don't attend to each other, so completed windows' encoder outputs are deterministic and their KV cache entries are correctly reused. Only the partial tail window causes invalidation.

### A/B test: prefix ON vs OFF (complex_chinese.wav, 19.6s)

Profiling output written to file (`/data/data/ai.connct_screen.rn/cache/asr_bench.txt`) for reliability. Read with `adb shell "run-as ai.connct_screen.rn cat cache/asr_bench.txt"`.

Comparable range: chunks 9-13 (window 1 cached, reused=113 tokens, no recovery reset in either run).

| Chunk | Audio | Prefix ON ||| Prefix OFF |||
|-------|-------|-----------|-----------|-----------|-----------|-----------|-----------|
| | | delta | total | **prefill ms** | delta | total | **prefill ms** |
| 9 | 0-9s | 44 (25 prefix) | 157 | **358** | 19 | 132 | **160** |
| 10 | 0-10s | 59 (27 prefix) | 172 | **396** | 32 | 145 | **319** |
| 11 | 0-11s | 78 (33 prefix) | 191 | **606** | 45 | 158 | **317** |
| 12 | 0-12s | 95 (37 prefix) | 208 | **732** | 58 | 171 | **446** |
| 13 | 0-13s | 108 (37 prefix) | 221 | **865** | 71 | 184 | **552** |

Key observation: **prefix OFF still grows 160→552ms (3.5x)** because encoder output itself grows ~13 tokens/chunk.

### But prefix is a net win on total wall time

| Metric | Prefix ON | Prefix OFF |
|--------|-----------|------------|
| Total prefill | 12,376 ms | 8,313 ms |
| Total decode | **2,543 ms** | **10,815 ms** |
| Total compute | **~14.9 s** | **~19.1 s** |
| Final tokens | 55 | 54 (with recovery reset + content loss) |

Without prefix, each chunk decodes full text from scratch, hitting `max_new=32` cap. Prefix saves ~8.3s decode but costs ~4.1s extra prefill = **net 4.2s faster**.

### Official implementation comparison

The [official Qwen3-ASR streaming](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py) re-feeds ALL accumulated audio every chunk with no encoder caching and no cross-chunk KV reuse. On GPU this doesn't matter — sequence lengths of 100-300 tokens leave GPU massively underutilized, so the O(n²) attention growth is invisible. On CPU every FLOP maps to wall time.

## Next Optimization Opportunities

### 1. Freeze partial encoder output within a window

**The biggest win.** Currently the partial window encoder output changes every chunk (bidirectional attention recomputes all positions when new audio arrives). This invalidates decoder KV cache for the encoder region + suffix + prefix.

Approach: after encoding a partial window, cache the per-position encoder output. When the partial window grows by 1s, only encode the new positions and append to cached output. This requires switching the encoder transformer from bidirectional to **causal attention** for partial windows (or simply concatenating independently-encoded chunks).

Impact estimate: eliminates ~66% of prefill growth. Within a window, KV cache would only need to re-prefill the new encoder tokens (~13/chunk) + new prefix tokens (~3-5/chunk), not the full encoder output + suffix + prefix.

Risk: encoder quality degrades without bidirectional context. Needs quality testing on CER benchmarks.

### 2. Skip complete window re-encoding at boundary

At 8s boundaries the partial window is replaced by a complete-window re-encoding (bidirectional over full 8s). This produces different encoder output, invalidating all KV cache.

Instead: keep the last partial window's encoder output as the cached window. Saves encoder time (~400ms) and preserves KV cache (~800ms prefill savings at chunk 16).

Risk: slightly lower encoder quality for that window. Low risk since partial-7s vs complete-8s is marginal.

### 3. Encoder GELU (~200ms across streaming, 13% of encoder)

`qwen_gelu` uses scalar `tanhf()`. NEON polynomial approximation (e.g. rational Padé) could cut this by 50-70%. Straightforward implementation, no quality impact.

### 4. ~~Encoder Q4_K~~ (tested, rejected)

Tested: encoder Q4_K with padded d_model (896→1024). Batch encoder **+17% slower** (974→1139ms), streaming encoder flat (+30ms). Root cause: encoder uses batched GEMM (seq=100-143 tokens) where compute dominates over bandwidth. Q4_K's complex dequant (4-bit unpack + sub-group corrections) is slower than Q8_0's simple INT8×INT8 SDOT. Plus 14% padding overhead for 0.6B (896→1024). Q4_K only helps bandwidth-bound paths (decoder matvec).

### 5. Chunk 1 encoder (613ms, no stem cache)

Cold-start skip means no stem cache is built for the first real chunk. Running encoder-only (no prefill/decode) during cold-start would cost ~600ms but only save ~200ms at chunk 1. Net loss with current audio length. Might help for longer audio where window reuse is more frequent.
