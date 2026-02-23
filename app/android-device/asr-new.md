# ASR New: Stream Mode Baseline

## Overview

This is a fresh start from the original unoptimized `qwen-asr` codebase (`/home/taowen/qwen-asr/`), replacing the previous 5-round optimized version (backed up at `jni/qwen-asr-old/`). The focus is exclusively on **stream mode** correctness and optimization.

## Stream Mode: How It Works

Stream mode (`--stream`) processes audio incrementally in overlapping chunks rather than waiting for the entire audio to finish:

1. **Chunked encoding**: Audio is split into fixed-size windows (`--enc-window-sec`, default ~30s). Each window is fed through the Whisper-style encoder to produce audio embeddings.
2. **Prefix rollback**: The decoder generates tokens for each chunk, then "rolls back" common prefix tokens from the previous chunk to maintain continuity. This ensures smooth token transitions across chunk boundaries.
3. **Incremental decoding**: Each stream step generates up to `stream_max_new_tokens` (default 32) tokens, allowing partial results to be emitted progressively.
4. **Encoder cache**: Controlled by `QWEN_STREAM_NO_ENC_CACHE=1` env var. When enabled (default), previously encoded frames are cached to avoid redundant computation on overlapping regions.

### Two stream entry points

| Function | Use case |
|----------|----------|
| `qwen_transcribe_stream()` | File-based: audio is fully loaded, then streamed through the pipeline |
| `qwen_transcribe_stream_live()` | Stdin/live: reads audio chunks from a `qwen_live_audio_t` source incrementally |

## Directory Structure

```
jni/qwen-asr/          # New baseline code (unoptimized)
jni/qwen-asr-old/      # Backup of 5-round optimized code
```

Source files:
- `qwen_asr.c/h` - Core ASR context, model loading, high-level API
- `qwen_asr_encoder.c` - Whisper encoder forward pass
- `qwen_asr_decoder.c` - Autoregressive decoder with KV cache
- `qwen_asr_kernels.c/h` - BLAS-backed matrix ops, thread pool
- `qwen_asr_kernels_{generic,neon,avx}.c` - Platform-specific kernel implementations
- `qwen_asr_audio.c/h` - Audio loading, mel spectrogram
- `qwen_asr_tokenizer.c/h` - BPE tokenizer
- `qwen_asr_safetensors.c/h` - Model weight loading
- `main.c` - CLI entry point
- `asr_regression.py` - Regression test harness

## Testing

### Prerequisites

- OpenBLAS dev headers (for `cblas.h`). If not system-installed, extract from the deb package:
  ```bash
  apt-get download libopenblas-pthread-dev
  dpkg-deb -x libopenblas-pthread-dev_*.deb /tmp/openblas-pthread-extract
  ```

### Build (x86 Linux)

```bash
cd app/android-device/android/app/src/main/jni/qwen-asr

# With system OpenBLAS:
make blas

# With locally extracted OpenBLAS:
make qwen_asr \
  'CFLAGS=-Wall -Wextra -O3 -march=native -ffast-math -DUSE_BLAS -DUSE_OPENBLAS -I/tmp/openblas-pthread-extract/usr/include/x86_64-linux-gnu/openblas-pthread' \
  'LDFLAGS=-lm -lpthread -L/tmp/openblas-pthread-extract/usr/lib/x86_64-linux-gnu/openblas-pthread -lopenblas'
```

Set `LD_LIBRARY_PATH` if using local OpenBLAS:
```bash
export LD_LIBRARY_PATH=/tmp/openblas-pthread-extract/usr/lib/x86_64-linux-gnu/openblas-pthread:$LD_LIBRARY_PATH
```

### Manual test

```bash
./qwen_asr -d /home/taowen/qwen-asr/qwen3-asr-0.6b -i samples/jfk.wav --stream --silent
```

Expected output: `And so, my fellow Americans, ask not what your country can do for you; ask what you can do for your country.`

### Automated regression tests

**Stream correctness** (pipes jfk.wav via stdin in stream mode, compares to reference):
```bash
./asr_regression.py --binary ./qwen_asr \
  --model-dir /home/taowen/qwen-asr/qwen3-asr-0.6b \
  --stream-check-only
```

**Encoder cache equivalence** (verifies cache on/off produce identical output on 10s and 45s samples):
```bash
./asr_regression.py --binary ./qwen_asr \
  --model-dir /home/taowen/qwen-asr/qwen3-asr-0.6b \
  --stream-cache-model-dir /home/taowen/qwen-asr/qwen3-asr-0.6b \
  --stream-cache-check-only
```

## Optimization Roadmap

Starting from this clean baseline, planned optimizations (in order):

1. **Profile baseline** - Identify hotspots on ARM (encoder vs decoder, which kernels dominate)
2. **INT8 GEMM** - Quantize weight matrices for matmul (biggest win from previous rounds)
3. **NEON activation functions** - SIMD-ized GELU, SiLU, softmax
4. **Encoder cache optimization** - Reduce redundant computation in overlapping stream windows
5. **Memory layout** - Optimize tensor memory layout for cache-friendliness on ARM

Each optimization should be validated against the stream regression tests before proceeding to the next.
