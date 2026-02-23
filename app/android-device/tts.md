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
