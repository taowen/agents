# TTS Offline Testing (adb broadcast backdoor)

## Commands

Six commands via `am broadcast`, results in logcat tags `TtsDebug` / `QwenTTS_JNI` / `BpeTokenizer`.

```bash
# 1. Check status (model downloaded? loaded? tokenizer ready?)
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn --es cmd status"

# 2. Download model (~1.4GB from R2)
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn --es cmd download"

# 3. Load model into memory
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn --es cmd load"

# 4. Tokenize only (print token IDs to logcat, no audio)
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn --es cmd tokenize --es text '你好世界'"

# 5. Full TTS: tokenize + generate + play
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn --es cmd speak --es text 'Hello world'"

# 5b. Speak with speaker/language
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn --es cmd speak --es text '你好' --es speaker serena --es language chinese"

# 6. Free model memory
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn --es cmd free"

# View results
adb logcat -s TtsDebug:V QwenTTS_JNI:V BpeTokenizer:V
```

## Architecture

TTS pipeline has 4 stages, each testable independently:

```
download  →  load  →  tokenize  →  generate + play
(R2→disk)   (disk→RAM)  (BPE)     (model inference → AudioTrack)
```

- **Download**: `TtsModelManager` fetches 6 files from R2 with HTTP Range resume support. Files land in `context.getFilesDir()/qwen-tts-model/`.
- **Load**: `HermesRuntime.nativeTtsLoadModel(modelDir)` → C JNI → loads safetensors into memory.
- **Tokenize**: `BpeTokenizer.load(modelDir)` reads `vocab.json` + `merges.txt`, then `tokenizeForTts(text)` produces comma-separated token IDs wrapped in Qwen2 chat template: `[im_start, assistant, \n, ...text..., im_end, \n, im_start, assistant, \n]`.
- **Generate**: `HermesRuntime.nativeTtsGenerate(tokenIds, speaker, language)` → returns `short[]` PCM at 24kHz mono 16-bit.
- **Play**: `AudioTrack` in `MODE_STATIC`, blocks until playback completes.

## Model files

Downloaded to `getFilesDir()/qwen-tts-model/` or manually pushed to `/data/local/tmp/qwen-tts-model/`:

| File | Actual Size |
|------|-------------|
| `config.json` | ~5KB |
| `vocab.json` | ~2.8MB |
| `merges.txt` | ~1.7MB |
| `model.safetensors` | 1,811,626,576 (~1.8GB) |
| `speech_tokenizer/config.json` | ~2KB |
| `speech_tokenizer/model.safetensors` | 682,293,092 (~682MB) |

**Total**: ~2.5GB

## Differences from ASR debug

| | ASR (`VoiceDebugReceiver`) | TTS (`TtsDebugReceiver`) |
|---|---|---|
| Action | `VOICE_DEBUG` | `TTS_DEBUG` |
| Logcat tag | `VoiceDebug` | `TtsDebug` |
| Model source | `adb push` to `/data/local/tmp/` | Auto-download from R2 via `TtsModelManager` |
| Input | WAV file path | Text string (`--es text`) |
| Steps | `load_model` → `test_wav` | `download` → `load` → `tokenize` / `speak` → `free` |
| Model load | Via `VoiceService.nativeLoadModel` | Via `HermesRuntime.nativeTtsLoadModel` |

## Gotchas

- **App must be started first**: The broadcast receiver is only active after the app process is alive. Run `adb shell am start -n ai.connct_screen.rn/.MainActivity` before sending broadcasts.
- **Download is resumable**: If interrupted, re-running `download` resumes via HTTP Range. Already-completed files are skipped.
- **`speak` auto-loads**: The `speak` command will auto-load the model if not already loaded, but will NOT auto-download. Run `download` first.
- **Tokenizer is cached**: `BpeTokenizer` is loaded once on first `tokenize` or `speak` and reused. `free` clears it along with the native model.
- **Thread blocking**: `download`, `load`, and `speak` all run on `new Thread()`, so the broadcast returns immediately. Watch logcat for results.
- **No concurrent safety**: Don't issue `speak` while another `speak` is still generating. The native code has shared state.
- **AudioTrack MODE_STATIC**: The entire PCM buffer is written before playback starts. For very long texts this uses significant memory (`samples * 2` bytes).

## Known issues

### R2 download returns 401
The R2 bucket `ai-chat-public` does not have public access enabled. The URL `https://pub-f464632870e64014a498bb2860410020.r2.dev/qwen3-tts-0.6b/` returns HTTP 401. **Workaround**: manually push model files to the device:

```bash
adb push /path/to/qwen-tts-model/ /data/local/tmp/qwen-tts-model/
```

Then use `--es path /data/local/tmp/qwen-tts-model` with all debug commands.

**Fix**: Enable public access on the `ai-chat-public` bucket in the Cloudflare Dashboard.

### FILE_SIZES in TtsModelManager were wrong (FIXED)
`model.safetensors` was listed as 1.2GB but is actually 1.8GB. `speech_tokenizer/model.safetensors` was listed as 200MB but is actually 682MB. This caused incorrect download progress percentages. Fixed to use actual sizes.

## Optimization status

### Baseline (pre-optimization, 2026-02-23)

Tested on device `03141fff` (OnePlus, arm64-v8a), compiled with `-O2`:

| Test | Audio length | Generation time | Ratio |
|------|-------------|-----------------|-------|
| "Hello world" (EN) | 2.32s | 20,644ms | ~8.9x real-time |
| "你好世界" (ZH) | 0.96s | ~11,400ms | ~11.9x real-time |

Model load: 636ms. Tokenizer load: 395ms.

### After optimization (NEON + -O3 -ffast-math, 2026-02-23)

| Test | Audio length | Gen time (before) | Gen time (after) | Speedup |
|------|-------------|-------------------|------------------|---------|
| "Hello world" (EN) | 2.32s | 20,644ms | **16,843ms** | **18% faster** |
| "你好世界" (ZH) | 0.96s | ~11,400ms | **8,707ms** | **24% faster** |
| Model load | - | 636ms | **424ms** | **33% faster** |

Still ~7.3x real-time for English, ~9.1x for Chinese. The bottleneck is the Transformer forward pass (matvec is memory-bandwidth-bound, not compute-bound). Further speedup requires:
- INT8 quantization (halves memory traffic → potential 1.5-2x)
- OpenMP parallelization of `kernel_matvec_bf16` across rows
- Smaller model or speculative decoding

### After optimization round 2 (INT8 + OpenMP + QKV fusion + prefetch + NEON, 2026-02-23)

| Test | Audio length | Gen time (before) | Gen time (after) | Speedup |
|------|-------------|-------------------|------------------|---------|
| "Hello world" (EN) | 2.16s | 16,843ms | **12,190ms** | **28% faster** |
| "你好世界" (ZH) | 1.20s | 8,707ms | **6,561ms** | **25% faster** |
| Model load | - | 424ms | **504-1157ms** | Slower (INT8 quantization at load) |

Now ~5.6x real-time for English, ~5.5x for Chinese. From original baseline: **41% faster** (EN), **42% faster** (ZH).

**Key findings from A/B testing:**
- **INT8 quantization is the only significant speedup** (~26% of the 28% total). Halving weight bandwidth is the key to speeding up memory-bound matvec.
- **OpenMP parallelism gives ~0%** for BF16 matvec — the memory bus is already saturated by 1 thread. With INT8 (halved bandwidth), 2 threads saturate the bus; 4 and 8 threads are worse due to contention/barrier overhead.
- **QKV fusion, prefetch, NEON softmax/rope/clamp/layernorm**: combined ~1-2% — negligible for this workload.
- INT8 changes model output slightly (2.16s vs 2.32s audio for same text). Subjective quality evaluation needed.
- Model load time increases by ~0.1-0.7s due to per-row INT8 quantization at load time.

Thread count sweep (INT8 enabled, "Hello world"):
| Threads | Time | Notes |
|---------|------|-------|
| 8 (default) | 14,776ms | Memory contention + barrier overhead |
| 4 | 12,405ms | Better, but still some contention |
| 2 | 12,303ms | Optimal — saturates LPDDR bandwidth |
| 1 (no INT8) | 16,572ms | BF16 baseline — single thread saturates BF16 bandwidth |

### Changes applied

| Change | File(s) | Measured impact |
|--------|---------|-----------------|
| NEON vectorize rms_norm, add, mul, dot, sum_sq, bf16_to_f32 | `qwen_tts_kernels.c`, `qwen_tts_talker.c`, `qwen_tts_codec.c` | Part of ~20% overall speedup |
| NEON polynomial sin for snake_beta (5th-order Taylor) | `qwen_tts_kernels.c` | Eliminates per-element sinf() in vocoder |
| Eliminate hot-loop malloc in text_projection/embed_text_token | `qwen_tts.c`, `qwen_tts.h` | Reduced GC pressure |
| Compiler -O3 -ffast-math (was -O2) | `CMakeLists.txt` | Auto-vectorization + fast math |
| INT8 weight quantization with ARM SDOT | `qwen_tts_kernels.c`, `qwen_tts.c`, `qwen_tts.h`, `qwen_tts_talker.c` | **~26% speedup** — halves matvec bandwidth |
| OpenMP parallel matvec (2 threads) | `qwen_tts_kernels.c` | ~2% with INT8 (saturates halved bandwidth) |
| QKV fused projection (3 matvec → 1) | `qwen_tts.h`, `qwen_tts.c`, `qwen_tts_talker.c` | ~1% — reduces function call overhead |
| Memory prefetch in matvec inner loop | `qwen_tts_kernels.c` | ~0% — hardware prefetcher already effective |
| NEON softmax, rope, clamp, layer_norm | `qwen_tts_kernels.c` | ~0% — not on hot path |
| `-march=armv8.2-a+dotprod` for SDOT | `CMakeLists.txt` | Enables vdotq_s32 for INT8 |
| ASR sources moved to jni/qwen-asr/ (was external /home/taowen/qwen-asr) | `CMakeLists.txt`, `build.gradle` | Build portability |
| speak() accepts speaker/language params | `hermes_runtime.cpp`, `HermesRuntime.java`, `host-api.ts` | Feature completeness |
| FILE_SIZES corrected (1.8GB + 682MB) | `TtsModelManager.java` | Accurate download progress |

### After optimization round 3 (INT8 kernel tuning + fused SwiGLU + INT4 experiment, 2026-02-23)

| Test | Audio length | Gen time (before) | Gen time (after) | Speedup |
|------|-------------|-------------------|------------------|---------|
| "Hello world" (EN) | 2.16s | 12,190ms | **12,028ms** | **~1.3% faster** |
| "你好世界" (ZH) | 1.20s | 6,561ms | **7,204ms** | Output length changed (1.20s audio) |

Minimal improvement. Confirmed by repeated runs (12,030ms second run). This is consistent with the Round 2 finding that **memory bandwidth is the sole bottleneck** — once INT8 halves the bandwidth, further compute-side optimizations yield diminishing returns.

**Changes applied (Phases A-C):**

| Change | File(s) | Measured impact |
|--------|---------|-----------------|
| 4-accumulator SDOT in `kernel_matvec_int8` (was 1 acc) | `qwen_tts_kernels.c` | ~0% — A77 OoO already hides single-acc latency |
| NEON-vectorized x quantization (was scalar) | `qwen_tts_kernels.c` | ~0% — quantization is <1% of total time |
| Pre-quantized x interface (`kernel_quantize_x_int8` + `kernel_matvec_int8_pq`) | `qwen_tts_kernels.c`, `qwen_tts_kernels.h` | ~0% — eliminates redundant quantization but quantization cost is negligible |
| Fused `kernel_swiglu_matvec_int8` (gate+up+SiLU in one call) | `qwen_tts_kernels.c`, `qwen_tts_kernels.h`, `qwen_tts_talker.c` | ~0% — function call overhead is negligible |
| OpenMP threshold raised from `rows >= 64` to `rows >= 512` | `qwen_tts_kernels.c` | ~0% — reduces barrier overhead for small matrices |

**Phase D (INT4 per-group quantization) — FAILED, disabled:**

| Test | Gen time (INT8) | Gen time (INT4) | Result |
|------|----------------|-----------------|--------|
| "Hello world" (EN) | 12,190ms | **28,903ms** | **2.37x regression** |

INT4 is disabled by default (`use_int4 = 0`). Root cause: NEON INT4 nibble unpacking overhead exceeds the bandwidth savings. Each group of 32 weights requires:
1. `vld1q_u8` (load 16 packed bytes)
2. `vshlq_n_u8` + `vshrq_n_s8` (extract low/high nibbles)
3. `vzipq_s8` (interleave to natural order)
4. `vdotq_s32` (2x for 32 elements)
5. `vaddvq_s32` + float multiply by group_scale (per-group reduction)

The per-group float accumulation (32 groups per 1024-column row) prevents the long SDOT chains that make INT8 efficient. INT8 does one `vaddvq_s32` per row; INT4 does 32. This makes INT4 compute-bound rather than memory-bound, defeating the purpose.

Viable INT4 alternatives (not attempted): GPTQ/AWQ calibrated quantization, per-row INT4 (larger error), or a custom packing format that avoids per-group reductions.

### After optimization round 4 (Q4_K_M super-block quantization, 2026-02-23)

Replaced per-group INT4 (which was 2.37x slower than INT8) with Q4_K super-block quantization inspired by llama.cpp/GGML. Q4_K_M strategy: QKV + gate_up use Q4_K, wo + down keep INT8 (sensitive layers).

| Test | Audio length | Gen time (INT8) | Gen time (Q4_K_M) | Speedup |
|------|-------------|-----------------|-------------------|---------|
| "Hello world" (EN) | 1.68s | 12,190ms | **10,280ms** | **~16% faster** |
| Chinese 25 chars | 6.48s | N/A | **44,683ms** | ~6.9x real-time |
| Model load | - | 504-1157ms | **3,340ms** | Slower (Q4_K quantization at load) |

From original baseline: **50% faster** (EN, vs 20,644ms).

**Q4_K format (152 bytes / 256 elements, 4.75 bits/element):**
- Super-block: `float d` (scale) + `float dmin` (min offset)
- 8 sub-groups of 32: `uint8 scales[8]` + `uint8 mins[8]`
- 128 bytes packed unsigned int4 nibbles [0,15]
- Dequant: `weight ≈ d × scales[g] × q − dmin × mins[g]`

**Key kernel optimization:** Integer sub-scale multiply (`vmulq_n_s32`, 1 cycle lane-wise) replaces per-group `vaddvq_s32` (3-5 cycle cross-lane reduction). Only 1 `vaddvq_s32` per 256-element super-block vs 8 per 256 elements in the old INT4, or 4 per 1024-column row vs 32 in old INT4.

**Why only 16% faster (not predicted 30%):**
1. Audio output length differs due to quantization change (1.68s vs 2.16s INT8) — fewer tokens generated, confounds comparison
2. Total time includes codec decode (constant regardless of quantization) — pure talker matvec improvement is larger
3. wo + down projections still use INT8 (Q4_K_M keeps these as sensitive layers)
4. Sub-talker runs per-token with its own matvec costs

**Model load time increased** from ~0.5-1.2s to 3.3s due to the more complex Q4_K quantization algorithm (3-phase: min/max → two-level scale → pack).

**Changes applied:**

| Change | File(s) | Notes |
|--------|---------|-------|
| `block_q4_k` struct (152B/256 elements) | `qwen_tts_kernels.h` | float d/dmin + uint8 scales/mins + packed nibbles |
| `kernel_matvec_q4k` with NEON SDOT + integer sub-scales | `qwen_tts_kernels.c` | Precomputes bsums, vmulq_n_s32 per sub-group, 1 vaddvq_s32 per block |
| `kernel_swiglu_matvec_q4k` fused gate+up | `qwen_tts_kernels.c` | Calls matvec_q4k twice then SiLU×up |
| `quantize_bf16_to_q4k` (3-phase algorithm) | `qwen_tts.c` | min/max → two-level scale → unsigned int4 pack |
| Q4_K_M dispatch: QKV+gate_up=Q4_K, wo+down=INT8 | `qwen_tts_talker.c`, `qwen_tts.c` | Sensitive layers keep higher precision |
| Removed old INT4 per-group code | all 5 files | `use_int4`/`int4_group_size` → `use_q4k` |

**Simplifications vs GGML:** d/dmin use float32 (GGML uses float16), scales/mins use plain uint8 (GGML packs 6-bit into 12 bytes). Adds ~8 bytes/block vs GGML but avoids complex bit unpacking.

### Testing gotchas discovered (OPLUS ROM)

**1. Duplicate broadcast delivery (proxy broadcasts):**
OPLUS ROM's `BroadcastQueue` "proxies" broadcasts to background apps with significant delay. If a new broadcast is sent before the proxied one is delivered, both arrive simultaneously, causing concurrent TTS on shared state (hangs/corruption). **Workaround**: ensure app is in foreground (`am start` first), wait 60s after `force-stop` before testing.

**2. Background cpuset throttling (6x slowdown):**
When the screen turns off or the app loses focus, OPLUS moves the process from `top-app` to `background` cpuset, restricting it to small cores (~614MHz vs 2.4GHz big cores). This causes **74s generation time** (6x slower). Both optimized and baseline code are equally affected. **Workaround**: keep screen unlocked and app in foreground. Verify with `cat /proc/<pid>/cgroup` — should show `cpuset:/top-app`.

### After optimization round 5 (sub-talker Q4_K + weight cache + codec INT8 + instrumentation, 2026-02-23)

Added time decomposition instrumentation, sub-talker full Q4_K (wo+down), pre-quantized weight cache, and codec transformer INT8 quantization.

**Timing decomposition revealed the real bottleneck:**

| Component | "Hello world" | % | Chinese 30 chars | % |
|-----------|--------------|---|------------------|---|
| Talker (pure) | 498ms | 4.7% | 2,410ms | 4.4% |
| Sub-talker | 1,461ms | 13.9% | 5,971ms | 10.8% |
| Codec (total) | 8,040ms | 76.5% | 46,296ms | 83.8% |
| — transformer | 39ms | 0.4% | 190ms | 0.3% |
| — upsample | 467ms | 4.4% | 1,324ms | 2.4% |
| — **vocoder** | **7,511ms** | **71.5%** | **44,741ms** | **81.0%** |
| **Total** | **10,513ms** | | **55,225ms** | |
| Audio length | 1.84s | | 7.60s | |

**The vocoder (BigVGAN) is 71-81% of total time.** Previous rounds optimized talker/sub-talker (now only 14-19% of total). The codec transformer (target of Phase 4 INT8) is only 0.3-0.4% of total — negligible.

**Generation results:**

| Test | Audio length | Gen time (R4 Q4_K_M) | Gen time (R5) | Notes |
|------|-------------|---------------------|---------------|-------|
| "Hello world" (EN) | 1.84s | 10,280ms | **10,513ms** | Within noise (~same) |
| Chinese 30 chars | 7.60s | N/A | **55,225ms** | ~7.3x real-time |
| Model load (first) | - | 3,340ms | **2,471ms** | 26% faster (codec INT8 quant is fast) |
| Model load (cache) | - | 3,340ms | **1,131ms** | **66% faster** (qcache hit) |

**Generation speed per-token: 85-88ms/token** (talker + sub-talker combined, roughly same as R4).

Sub-talker Q4_K (Phase 2) reduced sub-talker time by ~14% (estimated from bandwidth analysis), but since sub-talker is only 14% of total, the overall improvement is ~2% — invisible in noise.

**Model load with quantized cache (Phase 3) works well:**
- First load: 2,471ms (quantize + save cache)
- Subsequent loads: **1,131ms** (mmap cache + copy + load norms/embeddings)
- Cache file: 368MB at `/data/data/ai.connct_screen.rn/cache/model.qcache`
- Cache validates against safetensors file size to detect model changes

**Key insight for future optimization:**

The next target must be the **vocoder (BigVGAN)**, which accounts for 71-81% of total TTS time. The vocoder processes:
- 4 blocks × (SnakeBeta + TransposedConv1d + 3 × ResUnit(k=7, dilations 1/3/9))
- Upsample ratios: 8×5×4×3 = 480× from codec domain to audio domain
- At codec rate 12.5 Hz → 24kHz audio, this means processing very long sequences at each stage
- The inner ResUnit dilated convolutions (k=7, d=1/3/9) dominate compute

Potential vocoder optimizations:
1. **INT8 quantize vocoder conv weights** (currently F32, ~100MB for the 4 blocks)
2. **OpenMP parallelize transposed conv1d** (embarrassingly parallel across output channels)
3. **Fuse SnakeBeta+Conv1d** operations to reduce memory round-trips
4. **Streaming vocoder**: process codec tokens incrementally instead of waiting for all tokens

**Changes applied:**

| Change | File(s) | Notes |
|--------|---------|-------|
| `perf_subtalker_ms` timing instrumentation | `qwen_tts.h`, `qwen_tts.c` | Phase 1: time decomposition (talker/sub-talker/codec) |
| Sub-talker wo/down Q4_K quantization | `qwen_tts.h`, `qwen_tts.c`, `qwen_tts_talker.c` | Phase 2: full Q4_K for sub-talker (lower precision OK) |
| Pre-quantized weight cache (`model.qcache`) | `qwen_tts.h`, `qwen_tts.c` | Phase 3: mmap-based cache, 66% faster reload |
| Codec transformer INT8 quantization | `qwen_tts.h`, `qwen_tts.c`, `qwen_tts_codec.c` | Phase 4: fused QKV/gate_up INT8 per-token matvec |
| `quantize_f32_to_int8` helper | `qwen_tts.c` | F32 source INT8 quantization for codec weights |
| stderr→logcat redirect in JNI | `qwen_tts_jni.c` | Pipe redirect for C fprintf(stderr) visibility |
| `qwen_tts_cache_dir_override` global | `qwen_tts.h`, `qwen_tts.c` | Allows JNI to set writable cache dir before load |

**Android SELinux gotcha:** App process cannot write to `/data/local/tmp/` even if the directory is `chmod 777`. SELinux context prevents cross-domain file creation. Solution: use app's own cache dir (`/data/data/<pkg>/cache/`) via `qwen_tts_cache_dir_override`.

## Full test session

```bash
# Build & install
cd app/android-device/android && ./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk

# Start app
adb shell am start -n ai.connct_screen.rn/.MainActivity

# Terminal 1: watch logs (use '*:S' suffix on OPLUS ROMs to suppress noise)
adb logcat -s TtsDebug:V QwenTTS_JNI:V BpeTokenizer:V '*:S'

# Terminal 2: run commands
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn --es cmd status"
# I TtsDebug: model_downloaded=false
# I TtsDebug: model_loaded=false
# I TtsDebug: tokenizer_ready=false
# I TtsDebug: model_dir=/data/user/0/ai.connct_screen.rn/files/qwen-tts-model

# Load from manual push path
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn --es cmd load --es path /data/local/tmp/qwen-tts-model"
# I TtsDebug: Loading model from /data/local/tmp/qwen-tts-model...
# I QwenTTS_JNI: Loading TTS model from: /data/local/tmp/qwen-tts-model
# I QwenTTS_JNI: TTS model loaded successfully
# I TtsDebug: load_model result=true in 636ms

# Tokenize test
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn --es cmd tokenize --es text 'Hello world' --es path /data/local/tmp/qwen-tts-model"
# I BpeTokenizer: Loaded vocab: 151643 tokens
# I BpeTokenizer: Loaded merges: 151387 rules
# I TtsDebug: tokenize count=11 in 0ms
# I TtsDebug: tokenize ids=151644,77091,198,9707,220,14615,151645,198,151644,77091,198

# Full TTS speak
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn --es cmd speak --es text 'Hello world' --es path /data/local/tmp/qwen-tts-model"
# I TtsDebug: speak: tokenized 11 tokens in 1ms
# I TtsDebug: speak: generating audio (speaker=null, language=null)...
# I QwenTTS_JNI: TTS generate: step 20 / 4096
# I QwenTTS_JNI: TTS generated 55680 samples (2.32 seconds)
# I TtsDebug: speak: generated 55680 samples (2.32s) in 20644ms
# I TtsDebug: speak: playing audio...
# I TtsDebug: speak: playback complete

# Free model
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn --es cmd free"
# I TtsDebug: Model freed, tokenizer cleared
```
