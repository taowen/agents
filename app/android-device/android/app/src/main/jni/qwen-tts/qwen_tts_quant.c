/*
 * qwen_tts_quant.c - INT8 Weight Quantization for Qwen3-TTS
 *
 * Contains:
 *   - INT8 matvec kernels (NEON SDOT + scalar fallback)
 *   - Fused SwiGLU variant for INT8
 *   - BF16 -> INT8 quantization function
 *   - Weight cache save/load (binary .qcache format)
 */

#include "qwen_tts_quant.h"
#include "qwen_tts.h"
#include "qwen_tts_kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

extern int qwen_tts_verbose;

/* ========================================================================
 * INT8 matvec with on-the-fly x quantization
 *
 * A_int8[r,c] ~ round(A_bf16[r,c] / scale[r] * 127)
 * out[r] = scale[r] * sum(A_int8[r,c] * x_int8[c]) * x_scale
 * ======================================================================== */

void kernel_matvec_int8(float *out, const int8_t *A_int8, const float *scales,
                         const float *x, int rows, int cols) {
    /* Quantize x vector to int8 with a single global scale */
    static int8_t *x_int8 = NULL;
    static float x_scale = 0.0f;
    static int x_int8_cap = 0;
    if (cols > x_int8_cap) {
        free(x_int8);
        x_int8 = (int8_t *)malloc(((cols + 15) & ~15) * sizeof(int8_t));
        x_int8_cap = cols;
    }

    /* Find max(|x|) */
    float x_absmax = 0.0f;
#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t vabsmax = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < cols; i += 4)
        vabsmax = vmaxq_f32(vabsmax, vabsq_f32(vld1q_f32(x + i)));
    x_absmax = vmaxvq_f32(vabsmax);
    for (; i < cols; i++) {
        float a = x[i] > 0 ? x[i] : -x[i];
        if (a > x_absmax) x_absmax = a;
    }
#else
    for (int i = 0; i < cols; i++) {
        float a = x[i] > 0 ? x[i] : -x[i];
        if (a > x_absmax) x_absmax = a;
    }
#endif

    x_scale = x_absmax / 127.0f;
    float inv_x_scale = (x_absmax > 0.0f) ? 127.0f / x_absmax : 0.0f;

    /* Quantize x to int8 */
#if defined(__ARM_NEON) || defined(__aarch64__)
    {
        float32x4_t vscale = vdupq_n_f32(inv_x_scale);
        int c = 0;
        for (; c + 7 < cols; c += 8) {
            int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + c), vscale));
            int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + c + 4), vscale));
            int16x4_t s0 = vqmovn_s32(i0);
            int16x4_t s1 = vqmovn_s32(i1);
            int8x8_t  b  = vqmovn_s16(vcombine_s16(s0, s1));
            vst1_s8(x_int8 + c, b);
        }
        for (; c < cols; c++) {
            float v = x[c] * inv_x_scale;
            int iv = (int)(v + (v > 0 ? 0.5f : -0.5f));
            if (iv > 127) iv = 127;
            if (iv < -128) iv = -128;
            x_int8[c] = (int8_t)iv;
        }
    }
#else
    for (int c = 0; c < cols; c++) {
        float v = x[c] * inv_x_scale;
        int iv = (int)(v + (v > 0 ? 0.5f : -0.5f));
        if (iv > 127) iv = 127;
        if (iv < -128) iv = -128;
        x_int8[c] = (int8_t)iv;
    }
#endif
    /* Pad remainder to 16-byte boundary with zeros */
    int cols_padded = (cols + 15) & ~15;
    for (int c = cols; c < cols_padded; c++) x_int8[c] = 0;

#if (defined(__ARM_NEON) || defined(__aarch64__)) && defined(__ARM_FEATURE_DOTPROD)
    /* ARM SDOT path: vdotq_s32 processes 16 int8s -> 4 int32 accumulators */
#ifdef USE_OPENMP
    #pragma omp parallel for schedule(static) num_threads(2) if(rows >= 512)
#endif
    for (int r = 0; r < rows; r++) {
        const int8_t *row = A_int8 + (size_t)r * cols;
        int32x4_t iacc0 = vdupq_n_s32(0);
        int32x4_t iacc1 = vdupq_n_s32(0);
        int32x4_t iacc2 = vdupq_n_s32(0);
        int32x4_t iacc3 = vdupq_n_s32(0);
        int c = 0;
        /* Main loop: 64 bytes at a time with 4 accumulators to hide SDOT latency */
        for (; c + 63 < cols; c += 64) {
            iacc0 = vdotq_s32(iacc0, vld1q_s8(row + c),      vld1q_s8(x_int8 + c));
            iacc1 = vdotq_s32(iacc1, vld1q_s8(row + c + 16), vld1q_s8(x_int8 + c + 16));
            iacc2 = vdotq_s32(iacc2, vld1q_s8(row + c + 32), vld1q_s8(x_int8 + c + 32));
            iacc3 = vdotq_s32(iacc3, vld1q_s8(row + c + 48), vld1q_s8(x_int8 + c + 48));
        }
        /* Tail: 16 bytes at a time */
        for (; c + 15 < cols; c += 16) {
            iacc0 = vdotq_s32(iacc0, vld1q_s8(row + c), vld1q_s8(x_int8 + c));
        }
        int32_t isum = vaddvq_s32(iacc0) + vaddvq_s32(iacc1) + vaddvq_s32(iacc2) + vaddvq_s32(iacc3);
        /* Handle remaining 0-15 elements */
        for (; c < cols; c++) {
            isum += (int32_t)row[c] * (int32_t)x_int8[c];
        }
        /* Dequantize: out = weight_scale * x_scale * integer_dot */
        out[r] = scales[r] * x_scale * (float)isum;
    }
#else
    /* Scalar fallback */
#ifdef USE_OPENMP
    #pragma omp parallel for schedule(static) num_threads(2) if(rows >= 512)
#endif
    for (int r = 0; r < rows; r++) {
        const int8_t *row = A_int8 + (size_t)r * cols;
        int32_t isum = 0;
        for (int c = 0; c < cols; c++) {
            isum += (int32_t)row[c] * (int32_t)x_int8[c];
        }
        out[r] = scales[r] * x_scale * (float)isum;
    }
#endif
}

/* ========================================================================
 * Quantize float vector x to int8 (standalone, for reuse)
 * ======================================================================== */

void kernel_quantize_x_int8(const float *x, int cols, int8_t *x_int8_out, float *x_scale_out) {
    /* Find max(|x|) */
    float x_absmax = 0.0f;
#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t vabsmax = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < cols; i += 4)
        vabsmax = vmaxq_f32(vabsmax, vabsq_f32(vld1q_f32(x + i)));
    x_absmax = vmaxvq_f32(vabsmax);
    for (; i < cols; i++) {
        float a = x[i] > 0 ? x[i] : -x[i];
        if (a > x_absmax) x_absmax = a;
    }
#else
    for (int i = 0; i < cols; i++) {
        float a = x[i] > 0 ? x[i] : -x[i];
        if (a > x_absmax) x_absmax = a;
    }
#endif

    *x_scale_out = x_absmax / 127.0f;
    float inv_x_scale = (x_absmax > 0.0f) ? 127.0f / x_absmax : 0.0f;

    /* Quantize x to int8 */
#if defined(__ARM_NEON) || defined(__aarch64__)
    {
        float32x4_t vscale = vdupq_n_f32(inv_x_scale);
        int c = 0;
        for (; c + 7 < cols; c += 8) {
            int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + c), vscale));
            int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + c + 4), vscale));
            int16x4_t s0 = vqmovn_s32(i0);
            int16x4_t s1 = vqmovn_s32(i1);
            int8x8_t  b  = vqmovn_s16(vcombine_s16(s0, s1));
            vst1_s8(x_int8_out + c, b);
        }
        for (; c < cols; c++) {
            float v = x[c] * inv_x_scale;
            int iv = (int)(v + (v > 0 ? 0.5f : -0.5f));
            if (iv > 127) iv = 127;
            if (iv < -128) iv = -128;
            x_int8_out[c] = (int8_t)iv;
        }
    }
#else
    for (int c = 0; c < cols; c++) {
        float v = x[c] * inv_x_scale;
        int iv = (int)(v + (v > 0 ? 0.5f : -0.5f));
        if (iv > 127) iv = 127;
        if (iv < -128) iv = -128;
        x_int8_out[c] = (int8_t)iv;
    }
#endif
    /* Pad remainder to 16-byte boundary with zeros */
    int cols_padded = (cols + 15) & ~15;
    for (int c = cols; c < cols_padded; c++) x_int8_out[c] = 0;
}

/* ========================================================================
 * INT8 matvec with pre-quantized x
 * ======================================================================== */

void kernel_matvec_int8_pq(float *out, const int8_t *A_int8, const float *scales,
                            const int8_t *x_int8, float x_scale, int rows, int cols) {
#if (defined(__ARM_NEON) || defined(__aarch64__)) && defined(__ARM_FEATURE_DOTPROD)
#ifdef USE_OPENMP
    #pragma omp parallel for schedule(static) num_threads(2) if(rows >= 512)
#endif
    for (int r = 0; r < rows; r++) {
        const int8_t *row = A_int8 + (size_t)r * cols;
        int32x4_t iacc0 = vdupq_n_s32(0);
        int32x4_t iacc1 = vdupq_n_s32(0);
        int32x4_t iacc2 = vdupq_n_s32(0);
        int32x4_t iacc3 = vdupq_n_s32(0);
        int c = 0;
        for (; c + 63 < cols; c += 64) {
            iacc0 = vdotq_s32(iacc0, vld1q_s8(row + c),      vld1q_s8(x_int8 + c));
            iacc1 = vdotq_s32(iacc1, vld1q_s8(row + c + 16), vld1q_s8(x_int8 + c + 16));
            iacc2 = vdotq_s32(iacc2, vld1q_s8(row + c + 32), vld1q_s8(x_int8 + c + 32));
            iacc3 = vdotq_s32(iacc3, vld1q_s8(row + c + 48), vld1q_s8(x_int8 + c + 48));
        }
        for (; c + 15 < cols; c += 16) {
            iacc0 = vdotq_s32(iacc0, vld1q_s8(row + c), vld1q_s8(x_int8 + c));
        }
        int32_t isum = vaddvq_s32(iacc0) + vaddvq_s32(iacc1) + vaddvq_s32(iacc2) + vaddvq_s32(iacc3);
        for (; c < cols; c++) {
            isum += (int32_t)row[c] * (int32_t)x_int8[c];
        }
        out[r] = scales[r] * x_scale * (float)isum;
    }
#else
#ifdef USE_OPENMP
    #pragma omp parallel for schedule(static) num_threads(2) if(rows >= 512)
#endif
    for (int r = 0; r < rows; r++) {
        const int8_t *row = A_int8 + (size_t)r * cols;
        int32_t isum = 0;
        for (int c = 0; c < cols; c++) {
            isum += (int32_t)row[c] * (int32_t)x_int8[c];
        }
        out[r] = scales[r] * x_scale * (float)isum;
    }
#endif
}

/* ========================================================================
 * Fused SwiGLU with INT8 weights
 * ======================================================================== */

void kernel_swiglu_matvec_int8(float *out, const int8_t *gate_up_int8,
                                const float *scales, const float *x,
                                int intermediate, int hidden) {
    static float *up_scratch = NULL;
    static size_t up_scratch_cap = 0;
    static int8_t *x_q8 = NULL;
    static int x_q8_cap = 0;

    if ((size_t)intermediate > up_scratch_cap) {
        free(up_scratch);
        up_scratch = (float *)malloc((size_t)intermediate * sizeof(float));
        up_scratch_cap = (size_t)intermediate;
    }
    if (hidden > x_q8_cap) {
        free(x_q8);
        x_q8 = (int8_t *)malloc(((hidden + 15) & ~15) * sizeof(int8_t));
        x_q8_cap = hidden;
    }

    /* Quantize x once, reuse for both gate and up */
    float x_s;
    kernel_quantize_x_int8(x, hidden, x_q8, &x_s);

    /* Gate (first half) */
    kernel_matvec_int8_pq(out, gate_up_int8, scales, x_q8, x_s, intermediate, hidden);
    /* Up (second half) */
    kernel_matvec_int8_pq(up_scratch, gate_up_int8 + (size_t)intermediate * hidden,
                          scales + intermediate, x_q8, x_s, intermediate, hidden);

    /* SiLU(gate) * up */
    for (int i = 0; i < intermediate; i++) {
        float g = out[i];
        out[i] = (g / (1.0f + expf(-g))) * up_scratch[i];
    }
}

/* ========================================================================
 * BF16 -> INT8 per-row symmetric quantization
 * ======================================================================== */

void quantize_bf16_to_int8(const uint16_t *bf16, int rows, int cols,
                             int8_t **out_int8, float **out_scales) {
    *out_int8 = (int8_t *)malloc((size_t)rows * cols * sizeof(int8_t));
    *out_scales = (float *)malloc((size_t)rows * sizeof(float));
    if (!*out_int8 || !*out_scales) {
        free(*out_int8); free(*out_scales);
        *out_int8 = NULL; *out_scales = NULL;
        return;
    }
    for (int r = 0; r < rows; r++) {
        const uint16_t *row = bf16 + (size_t)r * cols;
        /* Find max absolute value in row */
        float absmax = 0.0f;
        for (int c = 0; c < cols; c++) {
            uint32_t bits = ((uint32_t)row[c]) << 16;
            float val;
            __builtin_memcpy(&val, &bits, sizeof(float));
            float a = val > 0 ? val : -val;
            if (a > absmax) absmax = a;
        }
        float scale = absmax / 127.0f;
        (*out_scales)[r] = scale;
        float inv_scale = (absmax > 0.0f) ? 127.0f / absmax : 0.0f;
        int8_t *dst = *out_int8 + (size_t)r * cols;
        for (int c = 0; c < cols; c++) {
            uint32_t bits = ((uint32_t)row[c]) << 16;
            float val;
            __builtin_memcpy(&val, &bits, sizeof(float));
            float v = val * inv_scale;
            int iv = (int)(v + (v > 0 ? 0.5f : -0.5f));
            if (iv > 127) iv = 127;
            if (iv < -128) iv = -128;
            dst[c] = (int8_t)iv;
        }
    }
}

/* ========================================================================
 * Pre-quantized Weight Cache
 *
 * Binary cache file format (.qcache):
 *   header (qcache_header_t)
 *   for each talker layer:
 *     wqkv_int8 + wqkv_scales | gate_up_int8 + gate_up_scales |
 *     wo_int8 + wo_scales | down_int8 + down_scales
 *   for each subtalker layer:
 *     wqkv_int8 + wqkv_scales | gate_up_int8 + gate_up_scales |
 *     wo_int8 + wo_scales | down_int8 + down_scales
 * ======================================================================== */

#ifndef __EMSCRIPTEN__
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#define QCACHE_MAGIC   0x31435151   /* "QQC1" */
#define QCACHE_VERSION 2

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t source_size;         /* original safetensors total file size for validation */
    uint32_t n_talker_layers;
    uint32_t n_subtalker_layers;
    /* Talker per-layer sizes */
    uint32_t tk_wqkv_int8_bytes;
    uint32_t tk_wqkv_scales_bytes;
    uint32_t tk_gate_up_int8_bytes;
    uint32_t tk_gate_up_scales_bytes;
    uint32_t tk_wo_int8_bytes;
    uint32_t tk_wo_scales_bytes;
    uint32_t tk_down_int8_bytes;
    uint32_t tk_down_scales_bytes;
    /* Subtalker per-layer sizes */
    uint32_t st_wqkv_int8_bytes;
    uint32_t st_wqkv_scales_bytes;
    uint32_t st_gate_up_int8_bytes;
    uint32_t st_gate_up_scales_bytes;
    uint32_t st_wo_int8_bytes;
    uint32_t st_wo_scales_bytes;
    uint32_t st_down_int8_bytes;
    uint32_t st_down_scales_bytes;
    uint32_t reserved[4];
} qcache_header_t;

#ifndef __EMSCRIPTEN__

static uint64_t get_safetensors_size(const char *model_dir) {
    char path[1024];
    uint64_t total = 0;
    struct stat st;

    snprintf(path, sizeof(path), "%s/model.safetensors", model_dir);
    if (stat(path, &st) == 0) {
        total += (uint64_t)st.st_size;
    }
    for (int i = 1; i <= 10; i++) {
        snprintf(path, sizeof(path), "%s/model-%05d-of-00002.safetensors", model_dir, i);
        if (stat(path, &st) == 0) {
            total += (uint64_t)st.st_size;
        }
        snprintf(path, sizeof(path), "%s/model-%05d-of-00003.safetensors", model_dir, i);
        if (stat(path, &st) == 0) {
            total += (uint64_t)st.st_size;
        }
    }
    return total;
}

int save_quantized_cache(struct qwen_tts_ctx *ctx) {
    qwen_tts_config_t *cfg = &ctx->config;
    char path[1024];

    snprintf(path, sizeof(path), "%s/model.qcache", ctx->cache_dir);

    /* Compute per-layer sizes - Talker */
    int tk_qkv_rows = cfg->talker_heads * cfg->talker_head_dim +
                       2 * cfg->talker_kv_heads * cfg->talker_head_dim;
    uint32_t tk_wqkv_int8_bytes = (uint32_t)((size_t)tk_qkv_rows * cfg->talker_hidden);
    uint32_t tk_wqkv_scales_bytes = (uint32_t)(tk_qkv_rows * sizeof(float));

    int tk_gu_rows = 2 * cfg->talker_intermediate;
    uint32_t tk_gate_up_int8_bytes = (uint32_t)((size_t)tk_gu_rows * cfg->talker_hidden);
    uint32_t tk_gate_up_scales_bytes = (uint32_t)(tk_gu_rows * sizeof(float));

    int tk_wo_rows = cfg->talker_hidden;
    int tk_wo_cols = cfg->talker_heads * cfg->talker_head_dim;
    uint32_t tk_wo_int8_bytes = (uint32_t)((size_t)tk_wo_rows * tk_wo_cols);
    uint32_t tk_wo_scales_bytes = (uint32_t)(tk_wo_rows * sizeof(float));

    int tk_down_rows = cfg->talker_hidden;
    int tk_down_cols = cfg->talker_intermediate;
    uint32_t tk_down_int8_bytes = (uint32_t)((size_t)tk_down_rows * tk_down_cols);
    uint32_t tk_down_scales_bytes = (uint32_t)(tk_down_rows * sizeof(float));

    /* Compute per-layer sizes - Subtalker */
    int st_qkv_rows = cfg->subtalker_heads * cfg->subtalker_head_dim +
                       2 * cfg->subtalker_kv_heads * cfg->subtalker_head_dim;
    uint32_t st_wqkv_int8_bytes = (uint32_t)((size_t)st_qkv_rows * cfg->subtalker_hidden);
    uint32_t st_wqkv_scales_bytes = (uint32_t)(st_qkv_rows * sizeof(float));

    int st_gu_rows = 2 * cfg->subtalker_intermediate;
    uint32_t st_gate_up_int8_bytes = (uint32_t)((size_t)st_gu_rows * cfg->subtalker_hidden);
    uint32_t st_gate_up_scales_bytes = (uint32_t)(st_gu_rows * sizeof(float));

    int st_wo_rows = cfg->subtalker_hidden;
    int st_wo_cols = cfg->subtalker_heads * cfg->subtalker_head_dim;
    uint32_t st_wo_int8_bytes = (uint32_t)((size_t)st_wo_rows * st_wo_cols);
    uint32_t st_wo_scales_bytes = (uint32_t)(st_wo_rows * sizeof(float));

    int st_down_rows = cfg->subtalker_hidden;
    int st_down_cols = cfg->subtalker_intermediate;
    uint32_t st_down_int8_bytes = (uint32_t)((size_t)st_down_rows * st_down_cols);
    uint32_t st_down_scales_bytes = (uint32_t)(st_down_rows * sizeof(float));

    /* Build header */
    qcache_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = QCACHE_MAGIC;
    hdr.version = QCACHE_VERSION;
    hdr.source_size = get_safetensors_size(ctx->model_dir);
    hdr.n_talker_layers = (uint32_t)cfg->talker_layers;
    hdr.n_subtalker_layers = (uint32_t)cfg->subtalker_layers;
    hdr.tk_wqkv_int8_bytes = tk_wqkv_int8_bytes;
    hdr.tk_wqkv_scales_bytes = tk_wqkv_scales_bytes;
    hdr.tk_gate_up_int8_bytes = tk_gate_up_int8_bytes;
    hdr.tk_gate_up_scales_bytes = tk_gate_up_scales_bytes;
    hdr.tk_wo_int8_bytes = tk_wo_int8_bytes;
    hdr.tk_wo_scales_bytes = tk_wo_scales_bytes;
    hdr.tk_down_int8_bytes = tk_down_int8_bytes;
    hdr.tk_down_scales_bytes = tk_down_scales_bytes;
    hdr.st_wqkv_int8_bytes = st_wqkv_int8_bytes;
    hdr.st_wqkv_scales_bytes = st_wqkv_scales_bytes;
    hdr.st_gate_up_int8_bytes = st_gate_up_int8_bytes;
    hdr.st_gate_up_scales_bytes = st_gate_up_scales_bytes;
    hdr.st_wo_int8_bytes = st_wo_int8_bytes;
    hdr.st_wo_scales_bytes = st_wo_scales_bytes;
    hdr.st_down_int8_bytes = st_down_int8_bytes;
    hdr.st_down_scales_bytes = st_down_scales_bytes;

    FILE *f = fopen(path, "wb");
    if (!f) {
        if (qwen_tts_verbose >= 1)
            fprintf(stderr, "Warning: cannot create qcache at %s\n", path);
        return -1;
    }

    fwrite(&hdr, sizeof(hdr), 1, f);

    #define WRITE_OR_ZERO(ptr, n_bytes) do { \
        if (ptr) fwrite(ptr, 1, n_bytes, f); \
        else { void *z = calloc(1, n_bytes); fwrite(z, 1, n_bytes, f); free(z); } \
    } while(0)

    /* Write talker layers */
    for (int i = 0; i < cfg->talker_layers; i++) {
        qwen_tts_talker_layer_t *l = &ctx->talker.layers[i];
        WRITE_OR_ZERO(l->wqkv_int8, tk_wqkv_int8_bytes);
        WRITE_OR_ZERO(l->wqkv_scales, tk_wqkv_scales_bytes);
        WRITE_OR_ZERO(l->gate_up_int8, tk_gate_up_int8_bytes);
        WRITE_OR_ZERO(l->gate_up_scales, tk_gate_up_scales_bytes);
        WRITE_OR_ZERO(l->wo_int8, tk_wo_int8_bytes);
        WRITE_OR_ZERO(l->wo_scales, tk_wo_scales_bytes);
        WRITE_OR_ZERO(l->down_int8, tk_down_int8_bytes);
        WRITE_OR_ZERO(l->down_scales, tk_down_scales_bytes);
    }

    /* Write subtalker layers */
    for (int i = 0; i < cfg->subtalker_layers; i++) {
        qwen_tts_subtalker_layer_t *l = &ctx->subtalker.layers[i];
        WRITE_OR_ZERO(l->wqkv_int8, st_wqkv_int8_bytes);
        WRITE_OR_ZERO(l->wqkv_scales, st_wqkv_scales_bytes);
        WRITE_OR_ZERO(l->gate_up_int8, st_gate_up_int8_bytes);
        WRITE_OR_ZERO(l->gate_up_scales, st_gate_up_scales_bytes);
        WRITE_OR_ZERO(l->wo_int8, st_wo_int8_bytes);
        WRITE_OR_ZERO(l->wo_scales, st_wo_scales_bytes);
        WRITE_OR_ZERO(l->down_int8, st_down_int8_bytes);
        WRITE_OR_ZERO(l->down_scales, st_down_scales_bytes);
    }

    #undef WRITE_OR_ZERO

    fclose(f);
    if (qwen_tts_verbose >= 1)
        fprintf(stderr, "Saved quantized cache to %s\n", path);
    return 0;
}

int load_quantized_cache(struct qwen_tts_ctx *ctx) {
    qwen_tts_config_t *cfg = &ctx->config;
    char path[1024];
    snprintf(path, sizeof(path), "%s/model.qcache", ctx->cache_dir);

    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return -1; }
    size_t file_size = (size_t)st.st_size;
    if (file_size < sizeof(qcache_header_t)) { close(fd); return -1; }

    void *mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) return -1;

    const qcache_header_t *hdr = (const qcache_header_t *)mapped;

    /* Validate header */
    if (hdr->magic != QCACHE_MAGIC || hdr->version != QCACHE_VERSION) {
        munmap(mapped, file_size);
        return -1;
    }
    if ((int)hdr->n_talker_layers != cfg->talker_layers ||
        (int)hdr->n_subtalker_layers != cfg->subtalker_layers) {
        munmap(mapped, file_size);
        return -1;
    }

    /* Validate source file size */
    uint64_t expected_src = get_safetensors_size(ctx->model_dir);
    if (hdr->source_size != expected_src) {
        if (qwen_tts_verbose >= 1)
            fprintf(stderr, "qcache: source size mismatch (cache=%llu, actual=%llu), re-quantizing\n",
                    (unsigned long long)hdr->source_size, (unsigned long long)expected_src);
        munmap(mapped, file_size);
        return -1;
    }

    /* Validate total file size */
    size_t tk_per_layer = (size_t)hdr->tk_wqkv_int8_bytes + hdr->tk_wqkv_scales_bytes +
                          hdr->tk_gate_up_int8_bytes + hdr->tk_gate_up_scales_bytes +
                          hdr->tk_wo_int8_bytes + hdr->tk_wo_scales_bytes +
                          hdr->tk_down_int8_bytes + hdr->tk_down_scales_bytes;
    size_t st_per_layer = (size_t)hdr->st_wqkv_int8_bytes + hdr->st_wqkv_scales_bytes +
                          hdr->st_gate_up_int8_bytes + hdr->st_gate_up_scales_bytes +
                          hdr->st_wo_int8_bytes + hdr->st_wo_scales_bytes +
                          hdr->st_down_int8_bytes + hdr->st_down_scales_bytes;
    size_t expected_size = sizeof(qcache_header_t) +
                           tk_per_layer * hdr->n_talker_layers +
                           st_per_layer * hdr->n_subtalker_layers;
    if (file_size < expected_size) {
        munmap(mapped, file_size);
        return -1;
    }

    /* Copy weights from mmap into malloc'd buffers */
    const uint8_t *ptr = (const uint8_t *)mapped + sizeof(qcache_header_t);

    #define CACHE_COPY(dst, type, n_bytes) do { \
        if ((n_bytes) > 0) { \
            dst = (type)malloc(n_bytes); \
            if (dst) memcpy(dst, ptr, n_bytes); \
            ptr += (n_bytes); \
        } \
    } while(0)

    for (int i = 0; i < cfg->talker_layers; i++) {
        qwen_tts_talker_layer_t *l = &ctx->talker.layers[i];
        CACHE_COPY(l->wqkv_int8, int8_t *, hdr->tk_wqkv_int8_bytes);
        CACHE_COPY(l->wqkv_scales, float *, hdr->tk_wqkv_scales_bytes);
        CACHE_COPY(l->gate_up_int8, int8_t *, hdr->tk_gate_up_int8_bytes);
        CACHE_COPY(l->gate_up_scales, float *, hdr->tk_gate_up_scales_bytes);
        CACHE_COPY(l->wo_int8, int8_t *, hdr->tk_wo_int8_bytes);
        CACHE_COPY(l->wo_scales, float *, hdr->tk_wo_scales_bytes);
        CACHE_COPY(l->down_int8, int8_t *, hdr->tk_down_int8_bytes);
        CACHE_COPY(l->down_scales, float *, hdr->tk_down_scales_bytes);
    }

    for (int i = 0; i < cfg->subtalker_layers; i++) {
        qwen_tts_subtalker_layer_t *l = &ctx->subtalker.layers[i];
        CACHE_COPY(l->wqkv_int8, int8_t *, hdr->st_wqkv_int8_bytes);
        CACHE_COPY(l->wqkv_scales, float *, hdr->st_wqkv_scales_bytes);
        CACHE_COPY(l->gate_up_int8, int8_t *, hdr->st_gate_up_int8_bytes);
        CACHE_COPY(l->gate_up_scales, float *, hdr->st_gate_up_scales_bytes);
        CACHE_COPY(l->wo_int8, int8_t *, hdr->st_wo_int8_bytes);
        CACHE_COPY(l->wo_scales, float *, hdr->st_wo_scales_bytes);
        CACHE_COPY(l->down_int8, int8_t *, hdr->st_down_int8_bytes);
        CACHE_COPY(l->down_scales, float *, hdr->st_down_scales_bytes);
    }

    #undef CACHE_COPY

    munmap(mapped, file_size);

    if (qwen_tts_verbose >= 1)
        fprintf(stderr, "Loaded quantized cache from %s\n", path);
    return 0;
}

#else /* __EMSCRIPTEN__ */

int save_quantized_cache(struct qwen_tts_ctx *ctx) { (void)ctx; return -1; }
int load_quantized_cache(struct qwen_tts_ctx *ctx) { (void)ctx; return -1; }

#endif /* __EMSCRIPTEN__ */
