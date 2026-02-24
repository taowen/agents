# TTS Engine Notes (Qwen3-TTS 0.6B, ARM A77)

## Architecture

**Dual-Track Chunked Streaming**: accumulate `chunk_size` codec frames during AR generation, decode each chunk with batch vocoder on background thread, deliver via callback. Codec decoder is stateless.

Source: upstream `Qwen3-TTS-C/c/`, backed up to `jni/qwen-tts-old/`.

## Source layout (refactored 2026-02-24)

```
jni/
├── CMakeLists.txt              # Top-level: hermesruntime, qwenasr_jni, qwentts (shared JNI lib)
├── qwen_tts_jni.c              # JNI wrapper (moved out of qwen-tts/)
└── qwen-tts/
    ├── CMakeLists.txt           # Own build: qwen_tts_static STATIC library
    ├── qwen_tts.h               # Public API + types (~480 lines, was 648)
    ├── qwen_tts_quant.h         # Q8_0 block struct (36 bytes: float scale + int8_t[32])
    ├── qwen_tts_quant.c         # Q8_0 quantization (f32→q8, bf16→q8), NEON-accelerated
    ├── qwen_tts_internal.h      # Internal cross-module declarations (talker, codec, stream)
    ├── qwen_tts_kernels.h       # Kernel function declarations
    ├── qwen_tts_kernels.c       # Norms, activations, element-wise ops (~285 lines, was 1816)
    ├── qwen_tts_kernels_neon.c  # NEON Q8_0 matvec (SDOT), SwiGLU, F32 matvec/matmul
    ├── qwen_tts_kernels_ops.c   # Conv1d, TransConv1d, RoPE, M-RoPE, SnakeBeta, softmax, sampling
    ├── qwen_tts.c               # Main API: config, weight loading, generate()
    ├── qwen_tts_talker.c        # Talker transformer forward pass
    ├── qwen_tts_codec.c         # Codec decoder (tokens → waveform)
    ├── qwen_tts_safetensors.c   # SafeTensors file loader
    └── qwen_tts_audio.c         # Audio utilities
```

Key changes from monolithic structure:
- **JNI decoupled**: `qwen_tts_jni.c` moved to parent `jni/`, links against `qwen_tts_static`. Engine can be built/tested standalone.
- **Kernels split 3-way**: `qwen_tts_kernels.c` (norms/activations), `_neon.c` (NEON matvec/matmul), `_ops.c` (conv/rope/sampling).
- **Headers split**: `qwen_tts_quant.h` (block type), `qwen_tts_internal.h` (cross-module internals), `qwen_tts.h` (public API only).
- **Own CMakeLists.txt**: platform-aware ARM flags, conditional OpenMP, static library target. Mirrors `qwen-asr/` pattern.

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

## Quantization: Q8_0 unified (2026-02-24)

Talker + sub-talker weight matrices use **Q8_0** (32-element blocks, `float scale` + `int8_t[32]` = 36 bytes). Same format as qwen-asr. Replaced the previous 3-format system (per-row INT8, Q4_K super-blocks, BF16 matvec).

**Codec transformer uses F32** (not Q8_0). Q8_0 on the codec transformer (hidden=512, 8 layers) caused audible hoarseness/raspy artifacts. The small hidden dimension makes it more sensitive to quantization error. F32 codec decode is ~2x slower than Q8_0 but audio quality is correct. Codec transformer F32 adds ~16MB RAM (8 layers × 4 matrices × ~512KB each).

**What was unified:**
- Layer structs: ~12 weight pointer fields → 4 (`wqkv_q8`, `wo_q8`, `gate_up_q8`, `down_q8`)
- Kernels: 10 matvec/swiglu/matmul variants → 2 (`kernel_matvec_q8`, `kernel_swiglu_matvec_q8`)
- Qcache: single Q8_0 format (version 3, magic "QQC3")
- Forward pass: removed 3-way if/else dispatch

**Files changed:** `qwen_tts_quant.h` (block_q8_0 def), `qwen_tts_quant.c` (new, quantization functions from qwen-asr), `qwen_tts.h` (simplified structs), `qwen_tts.c` (weight loading, qcache), `qwen_tts_talker.c` (forward pass), `qwen_tts_codec.c` (codec transformer), `qwen_tts_kernels.h` + `qwen_tts_kernels_neon.c` (Q8_0 matvec with SDOT).

**Bug fix:** qcache round-trip for optional `input_proj_q8` (NULL → zeros → non-NULL on load → wrong forward path). Fixed by writing 0 bytes for NULL pointers.

### Q4_K rejected (historical)

4-bit quantization fundamentally unsuitable for this 0.6B model (hidden=1024, low weight redundancy). Token count distortion, 2x slower than INT8. See git history for details.

## Benchmark: "Hello world" batch (Q8_0)

| Metric | Value |
|--------|-------|
| Codec tokens | 25 |
| Talker | 433ms (64.8 ms/token) |
| Sub-talker | 1,187ms |
| Codec decode | 3,243ms |
| **Total** | **4,992ms** |
| Audio | 2.00s |
| Realtime factor | 0.40x |

## Benchmark: long CN batch (Q8_0)

Text: "今天的天气真不错，阳光明媚，微风轻拂，我们一起出去散步吧，外面的花都开了，空气中弥漫着花香"

| Metric | Value |
|--------|-------|
| Codec tokens | 157 |
| Talker + Sub-talker | 7,856ms (50.0 ms/token) |
| Talker | 2,289ms (29.1%) |
| Sub-talker | 5,566ms (70.9%) |
| Codec decode | 19,921ms |
| **Total** | **27,859ms** |
| Audio | 12.56s |
| Realtime factor | 0.45x |

## Streaming: pipelined batch vocoder (2026-02-24)

Replaced per-token incremental vocoder with **batch vocoder on background thread**. While vocoder decodes chunk N, talker generates chunk N+1 tokens concurrently.

**Architecture:**
```
Main thread:    [talker×5] → launch vocoder → [talker×5] → join+deliver → launch → ...
Vocoder thread:              [batch decode 5]              [batch decode 5]
```

**Streaming results (chunk_size=5, USAGE_MEDIA AudioTrack, 4s buffer):**

Same long CN text streamed: 153 tokens, 12.24s audio, TTFA=2,076ms.

**Known issue: vocoder too slow for smooth playback.** Each 5-token chunk produces 400ms of audio but takes ~1,200ms to decode. AudioTrack underruns between chunks, causing audible stuttering ("结结巴巴"). Batch mode plays smoothly because all audio is decoded before playback starts.

**AudioTrack config:** Changed `USAGE_ASSISTANT` → `USAGE_MEDIA` (assistant audio was inaudible on some devices, possibly routed to earpiece). Buffer size = 4s (192KB) to avoid `track.write()` blocking.

## Vocoder bottleneck breakdown (25 tokens batch)

| Component | Time (ms) | % of total |
|-----------|-----------|-----------|
| Vocoder (codec decode) | 3,243ms | 65.0% |
| Sub-talker | 1,187ms | 23.8% |
| Talker | 433ms | 8.7% |

Vocoder is the dominant cost. Memory-bandwidth limited (FP32 activation buffers >> L2 512KB).

**Vocoder internal pipeline (157 tokens batch):**
- 4 decoder blocks: dim 1536→768→384→192→96
- Upsampling rates: 8×5×4×3 = 480× (plus 2×2 pre-upsample = 1920× total)
- Block 2 (384→192, len=16000): 192×16000 = 12.3MB activations (24× L2)
- Block 3 (192→96, len=48000): 96×48000 = 18.4MB activations (36× L2)

## Test commands

```bash
# Batch EN
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn \
  --es cmd speak --es text 'Hello world' --es path /data/local/tmp/qwen-tts-model"

# Batch long CN
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn \
  --es cmd speak --es text '今天的天气真不错，阳光明媚，微风轻拂，我们一起出去散步吧，外面的花都开了，空气中弥漫着花香' \
  --es path /data/local/tmp/qwen-tts-model"

# Stream (pipelined batch vocoder, chunk_size=5)
adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn \
  --es cmd speak_stream --es text 'Hello world' --es path /data/local/tmp/qwen-tts-model \
  --ei chunk_size 5"

# Delete qcache (force re-quantization, needed after format change)
adb shell "run-as ai.connct_screen.rn rm -f cache/model.qcache"
```

## Codec transformer: Q8_0 → F32 (2026-02-24)

Q8_0 quantization on the codec transformer caused audible hoarseness. Codec transformer has hidden=512 (vs talker hidden=1024), making it more sensitive to quantization noise. Switching to F32 matvec for all 4 weight matrices (wqkv, wo, gate_up, down) per layer restored clean audio.

**Performance impact (batch, vivian, same text):**

| Metric | Q8_0 codec | F32 codec |
|--------|-----------|-----------|
| Codec decode (157 tokens) | 19,921ms | ~39,200ms |
| Total (157 tokens) | 27,859ms | ~50,200ms |
| Realtime factor | 0.45x | ~0.25x |

Codec transformer F32 is ~2x slower but vocoder (conv/transconv, already F32) dominates total codec time. The transformer is only ~30% of codec decode time.

## Codec transformer FP16 weights (2026-02-24)

Codec transformer weight matrices stored as `__fp16*`, computed via `kernel_matvec_f16w` (load FP16 weights → vcvt_f32_f16 → F32 FMA accumulation). Q8_0 kept as fallback. Vocoder stays F32.

**Why not FP16 vocoder:** FP16 vocoder was implemented and tested but was **1.8x slower** than F32 on the target device (Snapdragon, Cortex-A77). The `vfmaq_f16` instruction appears to have lower throughput than `vfmaq_f32` on this core. FP16 vocoder code was reverted.

**Files changed:** `qwen_tts.h` (FP16 fields in codec transformer layer struct), `qwen_tts_kernels.h` + `qwen_tts_kernels_neon.c` (`kernel_matvec_f16w`), `qwen_tts_kernels.c` (FP16 conversion utilities), `qwen_tts.c` (FP16 weight loading + LOAD_F16_CHECK macro), `qwen_tts_codec.c` (FP16 dispatch in codec transformer forward), `CMakeLists.txt` (`+fp16` flag).

**Benchmark: 73 tokens, 5.84s audio (FP16 transformer + F32 vocoder):**

| Stage | Time (ms) | % |
|-------|-----------|---|
| Talker + Sub-talker | 3,207 | 26% |
| Codec transformer | 156 | 1% |
| Upsample | 763 | 6% |
| **Vocoder** | **8,093** | **66%** |
| **Total** | **12,320** | |
| Realtime factor | 0.47x | |

Vocoder breakdown:
```
snake=46.9  transconv=1302.6  conv7=5594.0  conv1=916.3  resadd=25.2
```

conv7 (causal conv1d k=7 in ResUnits) is 69% of vocoder time. Memory-bandwidth limited: each saxpy pass over output buffer for all (oc × ic × k) combinations = TB-scale memory traffic.

## Next: OpenCL GPU vocoder

**Problem:** vocoder BigVGAN (decoder_dim=1536) is too heavy for mobile CPU. conv7 does `dim² × 7 × length` FMAs per ResUnit × 12 ResUnits. Single-core ARM bandwidth can't keep up.

**Solution:** Offload vocoder to GPU via OpenCL compute.

### Why OpenCL over Vulkan

| | OpenCL | Vulkan Compute |
|--|--------|---------------|
| API complexity | Simple C API | Extremely verbose |
| Code size | ~500-800 lines | ~1500-2000 lines |
| Kernel language | C99-like | GLSL compute |
| Android official support | No (but vendors ship drivers) | Yes |
| Adreno support | Good, `libOpenCL.so` present | Good |
| Performance | Near Vulkan | Slightly higher ceiling |
| Debugging | Easy | Hard |

ncnn, MNN, TNN all use OpenCL for mobile GPU inference. Proven approach.

### Architecture

```
CPU: Talker → Sub-talker → Codec transformer (FP16) → Upsample
                                                        ↓
GPU: vocoder pre-conv → 4 blocks × (SnakeBeta + TransConv + 3×ResUnit) → final conv
                                                        ↓
CPU: clEnqueueReadBuffer → PCM playback
```

Only vocoder on GPU. Weights uploaded once at init. All vocoder activations stay in GPU memory (no round-trip per layer).

### Expected speedup

conv1d (groups=1) = GEMM per kernel tap. Block 0: [768,768] @ [768,2336] × 7 taps.

- CPU (Cortex-A77 F32 NEON): 915ms
- GPU (Adreno, ~1 TFLOPS FP32): ~6ms theoretical, ~30-90ms practical (launch overhead, memory copy)
- **Expected vocoder: 8,093ms → 300-800ms**
- **Expected total: 12.3s → 4-5s for 5.84s audio (~1x realtime)**

### Implementation plan

**New files:**
```
qwen-tts/
├── qwen_tts_gpu.h        # GPU context interface (init/free/upload/vocoder_decode)
├── qwen_tts_gpu_cl.c     # OpenCL runtime: dlopen, device init, kernel compile, buffer mgmt
└── kernels/
    ├── conv1d_gemm.cl     # Causal conv1d as GEMM-per-tap
    ├── transconv.cl       # Transposed conv1d
    ├── snake_beta.cl      # SnakeBeta activation (polynomial sine)
    └── elementwise.cl     # add, clamp
```

**Modified files:**
- `CMakeLists.txt` — add new sources, `-ldl` for dlopen
- `qwen_tts.h` — add `qwen_tts_gpu_ctx_t *gpu` to main context
- `qwen_tts.c` — GPU init at model load, upload vocoder weights
- `qwen_tts_codec.c` — if GPU available, call `qwen_tts_gpu_vocoder_decode()` instead of CPU vocoder

**OpenCL loading (runtime, no link-time dependency):**
```c
void *libcl = dlopen("libOpenCL.so", RTLD_LAZY);
if (!libcl) { /* fallback to CPU vocoder */ }
clGetPlatformIDs = dlsym(libcl, "clGetPlatformIDs");
// ... load all CL functions
```

**Kernel strategy for conv1d_gemm.cl:**
- Reshape conv1d (groups=1, k=7) as 7 GEMM calls: `out += W_k @ input_shifted_k`
- Each GEMM: `[out_ch, length] += [out_ch, in_ch] @ [in_ch, length]`
- Tile GEMM for local memory: 16×16 tiles, coalesced global reads
- Fuse bias add into first GEMM call

**Buffer management:**
- Pre-allocate GPU buffers for max vocoder size at init
- Ping-pong pattern: 2 activation buffers (like CPU path)
- Weight buffers: one per conv layer, uploaded once

### Fallback

If `dlopen("libOpenCL.so")` fails or device has no GPU, silently fall back to CPU F32 vocoder. Zero impact on devices without OpenCL.

## Alternative: NPU graph splitting

MNN-style approach: split model into static-shape subgraphs for NPU and dynamic-shape parts for CPU. NPUs (Hexagon DSP on Qualcomm, APU on MediaTek) excel at fixed-shape tensor ops but cannot handle dynamic shapes like attention with growing KV cache.

**Graph splitting strategy:**
- **NPU-eligible**: Linear (matmul with fixed weights), Conv1d, TransConv1d — static shapes, bulk of compute
- **CPU-only**: Attention (QKV projection is NPU, but softmax + KV cache concat is CPU), LayerNorm, sampling
- Compile two shape variants per subgraph: `chunk_size=128` for prefill, `chunk_size=1` for autoregressive decode

**TTS model analysis:**

| Component | NPU eligible ops | CPU-only ops | NPU % (by FLOP) |
|-----------|-----------------|--------------|------------------|
| Talker (28 layers) | QKV/O/FFN matmul | Attention, RoPE, norm | ~85% |
| Sub-talker (5 layers) | QKV/O/FFN matmul | Attention, M-RoPE, norm | ~85% |
| Codec transformer (8 layers) | QKV/O/FFN matmul | Attention, norm | ~85% |
| **Vocoder (streaming, 1 token)** | **All conv/transconv** | **SnakeBeta, residual add** | **~95%** |

**Vocoder streaming has 100% fixed shapes** — ideal for NPU. With chunk_size=1 (one codec token at a time), every buffer size is deterministic:

```
Input:  [1024, 4]     (codebook_dim × pre_upsample)
Block 0: [1536, 4] → transconv → [768, 32]
Block 1: [768, 32] → transconv → [384, 160]
Block 2: [384, 160] → transconv → [192, 640]
Block 3: [192, 640] → transconv → [96, 1920]
Output: [1, 1920]    (mono audio samples)
```

No dynamic shapes anywhere in the vocoder — no sequence length variation, no KV cache growth. Every Conv1d, TransConv1d, and ResUnit operates on fixed dimensions. This means the entire vocoder graph can be compiled as a single NPU subgraph with no CPU fallback needed.

**Comparison with OpenCL:**

| | OpenCL GPU | NPU |
|--|-----------|-----|
| Implementation effort | Write kernels from scratch | Use vendor SDK (SNPE/QNN) |
| Portability | Most Android devices | Qualcomm/MediaTek only |
| Kernel optimization | Manual tuning needed | Vendor-optimized |
| Power efficiency | GPU power hungry | NPU very efficient |
| Latency | Low (direct dispatch) | Higher (graph compilation) |

NPU is better for vocoder (100% static shapes, vendor-optimized), but OpenCL is more portable. Could implement OpenCL first with NPU as a faster backend on supported devices.

## Next steps

1. ~~**Q8_0 unification**~~ — DONE.
2. ~~**Pipelined streaming**~~ — DONE.
3. ~~**Non-blocking audio callback**~~ — DONE.
4. ~~**Codec transformer F32**~~ — DONE. Q8_0 caused hoarseness.
5. ~~**Codec transformer FP16 weights**~~ — DONE. 2x codec decode speedup.
6. ~~**Vocoder FP16**~~ — REJECTED. 1.8x slower than F32 on target device.
7. **OpenCL GPU vocoder** — offload vocoder to GPU. Target: vocoder 300-800ms (currently 8,093ms).
8. **NPU vocoder** — alternative to OpenCL; vocoder streaming path is 100% fixed shapes, ideal for NPU (SNPE/QNN).
9. **Core pinning** — bind talker to big cores for lower token latency
