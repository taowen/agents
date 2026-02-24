/*
 * qwen_tts_kernels.c - Math kernel implementations for Qwen3-TTS
 *
 * Uses BLAS (Accelerate on macOS, OpenBLAS on Linux) for matrix ops.
 * Falls back to scalar implementations when BLAS is not available.
 */

#include "qwen_tts.h"
#include "qwen_tts_kernels.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

/* ======================================================================== */
/* SAXPY broadcast helper: dst[i] += alpha * src[i]                          */
/* ======================================================================== */

static inline void saxpy_broadcast(
    float * __restrict__ dst, float alpha,
    const float * __restrict__ src, int n) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t va = vdupq_n_f32(alpha);
    int i = 0;
    for (; i + 15 < n; i += 16) {
        vst1q_f32(dst+i,    vfmaq_f32(vld1q_f32(dst+i),    va, vld1q_f32(src+i)));
        vst1q_f32(dst+i+4,  vfmaq_f32(vld1q_f32(dst+i+4),  va, vld1q_f32(src+i+4)));
        vst1q_f32(dst+i+8,  vfmaq_f32(vld1q_f32(dst+i+8),  va, vld1q_f32(src+i+8)));
        vst1q_f32(dst+i+12, vfmaq_f32(vld1q_f32(dst+i+12), va, vld1q_f32(src+i+12)));
    }
    for (; i + 3 < n; i += 4)
        vst1q_f32(dst+i, vfmaq_f32(vld1q_f32(dst+i), va, vld1q_f32(src+i)));
    for (; i < n; i++) dst[i] += alpha * src[i];
#else
    for (int i = 0; i < n; i++) dst[i] += alpha * src[i];
#endif
}

/* ======================================================================== */
/* RMSNorm                                                                   */
/* ======================================================================== */

void kernel_rms_norm(float *out, const float *x, const float *weight, int dim, float eps) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t vss = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < dim; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        vss = vfmaq_f32(vss, vx, vx);
    }
    float ss = vaddvq_f32(vss);
    for (; i < dim; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)dim + eps);
    float32x4_t vinv = vdupq_n_f32(inv);
    i = 0;
    for (; i + 3 < dim; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vw = vld1q_f32(weight + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(vx, vinv), vw));
    }
    for (; i < dim; i++) out[i] = x[i] * inv * weight[i];
#else
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)dim + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * inv * weight[i];
#endif
}

void kernel_rms_norm_inplace(float *x, const float *weight, int dim, float eps) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t vss = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < dim; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        vss = vfmaq_f32(vss, vx, vx);
    }
    float ss = vaddvq_f32(vss);
    for (; i < dim; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)dim + eps);
    float32x4_t vinv = vdupq_n_f32(inv);
    i = 0;
    for (; i + 3 < dim; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vw = vld1q_f32(weight + i);
        vst1q_f32(x + i, vmulq_f32(vmulq_f32(vx, vinv), vw));
    }
    for (; i < dim; i++) x[i] = x[i] * inv * weight[i];
#else
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)dim + eps);
    for (int i = 0; i < dim; i++) x[i] = x[i] * inv * weight[i];
#endif
}

/* ======================================================================== */
/* LayerNorm                                                                 */
/* ======================================================================== */

void kernel_layer_norm(float *out, const float *x, const float *weight, const float *bias, int dim, float eps) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    /* NEON: compute mean */
    float32x4_t vsum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < dim; i += 4)
        vsum = vaddq_f32(vsum, vld1q_f32(x + i));
    float mean = vaddvq_f32(vsum);
    for (; i < dim; i++) mean += x[i];
    mean /= (float)dim;

    /* NEON: compute variance */
    float32x4_t vmean = vdupq_n_f32(mean);
    float32x4_t vvar = vdupq_n_f32(0.0f);
    i = 0;
    for (; i + 3 < dim; i += 4) {
        float32x4_t vd = vsubq_f32(vld1q_f32(x + i), vmean);
        vvar = vfmaq_f32(vvar, vd, vd);
    }
    float var = vaddvq_f32(vvar);
    for (; i < dim; i++) { float d = x[i] - mean; var += d * d; }
    var /= (float)dim;
    float inv = 1.0f / sqrtf(var + eps);

    /* NEON: normalize */
    float32x4_t vinv = vdupq_n_f32(inv);
    i = 0;
    if (weight && bias) {
        for (; i + 3 < dim; i += 4) {
            float32x4_t vn = vmulq_f32(vsubq_f32(vld1q_f32(x + i), vmean), vinv);
            vn = vfmaq_f32(vld1q_f32(bias + i), vn, vld1q_f32(weight + i));
            vst1q_f32(out + i, vn);
        }
        for (; i < dim; i++)
            out[i] = (x[i] - mean) * inv * weight[i] + bias[i];
    } else if (weight) {
        for (; i + 3 < dim; i += 4) {
            float32x4_t vn = vmulq_f32(vsubq_f32(vld1q_f32(x + i), vmean), vinv);
            vst1q_f32(out + i, vmulq_f32(vn, vld1q_f32(weight + i)));
        }
        for (; i < dim; i++)
            out[i] = (x[i] - mean) * inv * weight[i];
    } else {
        for (; i + 3 < dim; i += 4)
            vst1q_f32(out + i, vmulq_f32(vsubq_f32(vld1q_f32(x + i), vmean), vinv));
        for (; i < dim; i++)
            out[i] = (x[i] - mean) * inv;
        if (bias) for (int j = 0; j < dim; j++) out[j] += bias[j];
    }
#else
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= (float)dim;
    float var = 0.0f;
    for (int i = 0; i < dim; i++) { float d = x[i] - mean; var += d * d; }
    var /= (float)dim;
    float inv = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < dim; i++) {
        out[i] = (x[i] - mean) * inv;
        if (weight) out[i] *= weight[i];
        if (bias) out[i] += bias[i];
    }
#endif
}

/* ======================================================================== */
/* Matrix-Vector Multiply                                                    */
/* ======================================================================== */

#if defined(USE_BLAS) && !defined(__ARM_NEON) && !defined(__aarch64__)
/* Persistent scratch buffer for BF16→F32 conversion (non-NEON BLAS path) */
static float *_bf16_scratch = NULL;
static size_t _bf16_scratch_cap = 0;

static float *bf16_scratch_get(size_t n) {
    if (n > _bf16_scratch_cap) {
        free(_bf16_scratch);
        _bf16_scratch = (float *)malloc(n * sizeof(float));
        _bf16_scratch_cap = n;
    }
    return _bf16_scratch;
}
#endif

/* Persistent scratch for packed conv1d weights: [kernel, out_channels, in_channels]. */
#ifdef USE_BLAS
static float *_conv1d_wpack = NULL;
static size_t _conv1d_wpack_cap = 0;

static float *conv1d_wpack_get(size_t n) {
    if (n > _conv1d_wpack_cap) {
        float *tmp = (float *)realloc(_conv1d_wpack, n * sizeof(float));
        if (!tmp) return NULL;
        _conv1d_wpack = tmp;
        _conv1d_wpack_cap = n;
    }
    return _conv1d_wpack;
}
#endif

void kernel_matvec_bf16(float *out, const uint16_t *A_bf16, const float *x, int rows, int cols) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    /* Fused BF16→F32 + dot product using NEON — no intermediate buffer needed */
#ifdef USE_OPENMP
    #pragma omp parallel for schedule(static) num_threads(2) if(rows >= 512)
#endif
    for (int r = 0; r < rows; r++) {
        const uint16_t *row = A_bf16 + (size_t)r * cols;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        /* Prefetch x vector into L1 on first row of each thread's work */
        if (r == 0 || (r == 1 && rows >= 64)) {
            for (int p = 0; p < cols; p += 16)
                __builtin_prefetch(x + p, 0, 3);
        }
        int c = 0;
        for (; c + 7 < cols; c += 8) {
            /* Prefetch weight data 256 bytes ahead */
            __builtin_prefetch(row + c + 128, 0, 0);
            /* Load 8 BF16 values */
            uint16x8_t bf = vld1q_u16(row + c);
            /* Split into two halves and shift to F32 */
            uint32x4_t lo = vshll_n_u16(vget_low_u16(bf), 16);
            uint32x4_t hi = vshll_n_u16(vget_high_u16(bf), 16);
            float32x4_t f0 = vreinterpretq_f32_u32(lo);
            float32x4_t f1 = vreinterpretq_f32_u32(hi);
            /* Load 8 F32 x values */
            float32x4_t x0 = vld1q_f32(x + c);
            float32x4_t x1 = vld1q_f32(x + c + 4);
            /* Fused multiply-accumulate */
            acc0 = vfmaq_f32(acc0, f0, x0);
            acc1 = vfmaq_f32(acc1, f1, x1);
        }
        /* Reduce accumulators */
        acc0 = vaddq_f32(acc0, acc1);
        float sum = vaddvq_f32(acc0);
        /* Handle remaining elements */
        for (; c < cols; c++) {
            uint32_t bits = ((uint32_t)row[c]) << 16;
            float val;
            memcpy(&val, &bits, sizeof(float));
            sum += val * x[c];
        }
        out[r] = sum;
    }
#elif defined(USE_BLAS)
    /* Convert BF16 matrix to F32 using persistent scratch, then use BLAS sgemv */
    size_t total = (size_t)rows * cols;
    float *A_f32 = bf16_scratch_get(total);
    for (size_t i = 0; i < total; i++) {
        uint32_t bits = ((uint32_t)A_bf16[i]) << 16;
        memcpy(&A_f32[i], &bits, sizeof(float));
    }
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.0f, A_f32, cols, x, 1, 0.0f, out, 1);
#else
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const uint16_t *row = A_bf16 + (size_t)r * cols;
        for (int c = 0; c < cols; c++) {
            sum += bf16_to_f32(row[c]) * x[c];
        }
        out[r] = sum;
    }
#endif
}

void kernel_matvec_f32(float *out, const float *A, const float *x, int rows, int cols) {
#ifdef USE_BLAS
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.0f, A, cols, x, 1, 0.0f, out, 1);
#else
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const float *row = A + (size_t)r * cols;
        for (int c = 0; c < cols; c++) sum += row[c] * x[c];
        out[r] = sum;
    }
#endif
}

void kernel_matvec_int8(float *out, const int8_t *A_int8, const float *scales,
                         const float *x, int rows, int cols) {
    /*
     * INT8 matvec with per-row symmetric quantization.
     * A_int8[r,c] ≈ round(A_bf16[r,c] / scale[r] * 127)
     * out[r] = scale[r] * sum(A_int8[r,c] * x_int8[c]) where x_int8 is quantized on-the-fly.
     *
     * We quantize x to int8 once, then use integer dot products for the inner loop.
     * On ARM with SDOT: vdotq_s32 does 16 int8 multiplies → 4 int32 accumulations.
     */

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
    /* ARM SDOT path: vdotq_s32 processes 16 int8s → 4 int32 accumulators */
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

/* ======================================================================== */
/* Q4_K super-block matvec                                                   */
/* ======================================================================== */

void kernel_matvec_q4k(float *out, const block_q4_k *blocks,
                        const float *x, int rows, int cols) {
    /*
     * Q4_K super-block quantization matvec.
     * Each block covers 256 elements (8 sub-groups of 32).
     * Uses integer sub-scales (vmulq_n_s32) to avoid per-group vaddvq_s32.
     * Only 1 vaddvq_s32 per super-block instead of 8.
     *
     * Dequantization: weight ≈ d * scales[g] * q - dmin * mins[g]
     * where q ∈ [0, 15] (unsigned).
     */
    int blocks_per_row = cols / QK_K;

    /* Quantize x to int8 */
    static int8_t *x_int8 = NULL;
    static int x_int8_cap = 0;
    if (cols > x_int8_cap) {
        free(x_int8);
        x_int8 = (int8_t *)malloc(((cols + 15) & ~15) * sizeof(int8_t));
        x_int8_cap = cols;
    }
    float x_scale;
    kernel_quantize_x_int8(x, cols, x_int8, &x_scale);

    /* Precompute bsums: per-sub-group sum of x_int8 (shared across all rows) */
    int total_subs = cols / 32;
    static int32_t *bsums = NULL;
    static int bsums_cap = 0;
    if (total_subs > bsums_cap) {
        free(bsums);
        bsums = (int32_t *)malloc(total_subs * sizeof(int32_t));
        bsums_cap = total_subs;
    }

#if (defined(__ARM_NEON) || defined(__aarch64__)) && defined(__ARM_FEATURE_DOTPROD)
    /* NEON bsums: sum 32 int8 values per sub-group using SDOT with all-ones vector */
    {
        int8x16_t ones = vdupq_n_s8(1);
        for (int s = 0; s < total_subs; s++) {
            const int8_t *xg = x_int8 + s * 32;
            int32x4_t sum4 = vdupq_n_s32(0);
            sum4 = vdotq_s32(sum4, vld1q_s8(xg), ones);
            sum4 = vdotq_s32(sum4, vld1q_s8(xg + 16), ones);
            bsums[s] = vaddvq_s32(sum4);
        }
    }
#else
    for (int s = 0; s < total_subs; s++) {
        int32_t sum = 0;
        const int8_t *xg = x_int8 + s * 32;
        for (int i = 0; i < 32; i++) sum += (int32_t)xg[i];
        bsums[s] = sum;
    }
#endif

#if (defined(__ARM_NEON) || defined(__aarch64__)) && defined(__ARM_FEATURE_DOTPROD)
    /* NEON + SDOT path */
#ifdef USE_OPENMP
    #pragma omp parallel for schedule(static) num_threads(2) if(rows >= 512)
#endif
    for (int r = 0; r < rows; r++) {
        float row_sum = 0.0f;

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_k *blk = &blocks[(size_t)r * blocks_per_row + b];
            const int8_t *xq = x_int8 + b * QK_K;

            int32x4_t acc = vdupq_n_s32(0);
            int32_t min_acc = 0;

            for (int g = 0; g < Q4K_NUM_SUBS; g++) {
                /* Unpack unsigned nibbles */
                uint8x16_t packed = vld1q_u8(blk->qs + g * 16);
                int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0x0F)));
                int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(packed, 4));

                /* Interleave to get elements in order */
                int8x16x2_t z = vzipq_s8(lo, hi);

                /* SDOT: dot product of q4 weights with x_int8 */
                int32x4_t dot = vdupq_n_s32(0);
                dot = vdotq_s32(dot, z.val[0], vld1q_s8(xq + g * 32));
                dot = vdotq_s32(dot, z.val[1], vld1q_s8(xq + g * 32 + 16));

                /* Integer sub-scale multiply (avoids vaddvq_s32 per group) */
                dot = vmulq_n_s32(dot, (int32_t)blk->scales[g]);
                acc = vaddq_s32(acc, dot);

                /* Min correction (integer multiply-add) */
                min_acc += (int32_t)blk->mins[g] * bsums[b * Q4K_NUM_SUBS + g];
            }

            /* Only 1 vaddvq_s32 per super-block */
            row_sum += blk->d * (float)vaddvq_s32(acc) - blk->dmin * (float)min_acc;
        }
        out[r] = row_sum * x_scale;
    }
#else
    /* Scalar fallback */
#ifdef USE_OPENMP
    #pragma omp parallel for schedule(static) num_threads(2) if(rows >= 512)
#endif
    for (int r = 0; r < rows; r++) {
        float row_sum = 0.0f;

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_k *blk = &blocks[(size_t)r * blocks_per_row + b];
            const int8_t *xq = x_int8 + b * QK_K;

            int32_t scale_acc = 0;
            int32_t min_acc = 0;

            for (int g = 0; g < Q4K_NUM_SUBS; g++) {
                int32_t dot = 0;
                for (int i = 0; i < 16; i++) {
                    uint8_t packed = blk->qs[g * 16 + i];
                    int8_t lo = (int8_t)(packed & 0x0F);
                    int8_t hi = (int8_t)(packed >> 4);
                    dot += (int32_t)lo * (int32_t)xq[g * 32 + i * 2];
                    dot += (int32_t)hi * (int32_t)xq[g * 32 + i * 2 + 1];
                }
                scale_acc += dot * (int32_t)blk->scales[g];
                min_acc += (int32_t)blk->mins[g] * bsums[b * Q4K_NUM_SUBS + g];
            }

            row_sum += blk->d * (float)scale_acc - blk->dmin * (float)min_acc;
        }
        out[r] = row_sum * x_scale;
    }
#endif
}

void kernel_swiglu_matvec_q4k(float *out, const block_q4_k *gate_up_blocks,
                                const float *x, int intermediate, int hidden) {
    static float *up_scratch_q4k = NULL;
    static size_t up_scratch_q4k_cap = 0;
    if ((size_t)intermediate > up_scratch_q4k_cap) {
        free(up_scratch_q4k);
        up_scratch_q4k = (float *)malloc((size_t)intermediate * sizeof(float));
        up_scratch_q4k_cap = (size_t)intermediate;
    }

    int blocks_per_row = hidden / QK_K;

    /* Gate (first `intermediate` rows) */
    kernel_matvec_q4k(out, gate_up_blocks, x, intermediate, hidden);
    /* Up (next `intermediate` rows) */
    kernel_matvec_q4k(up_scratch_q4k,
                       gate_up_blocks + (size_t)intermediate * blocks_per_row,
                       x, intermediate, hidden);

    /* SiLU(gate) * up */
    for (int i = 0; i < intermediate; i++) {
        float g = out[i];
        out[i] = (g / (1.0f + expf(-g))) * up_scratch_q4k[i];
    }
}

/* ======================================================================== */
/* Matrix-Matrix Multiply                                                    */
/* ======================================================================== */

void kernel_matmul_f32(float *C, const float *A, const float *B, int M, int N, int K) {
    /* C[M,N] = A[M,K] @ B[N,K]^T  →  C = A * B^T */
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K,
                1.0f, A, K, B, K, 0.0f, C, N);
#else
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[m * K + k] * B[n * K + k];
            C[m * N + n] = sum;
        }
    }
#endif
}

void kernel_matmul_bf16(float *C, const float *A, const uint16_t *B_bf16, int M, int N, int K) {
    /* C[M,N] = A[M,K] @ B[N,K]^T  where B is BF16 */
    /* Convert B to F32 first for BLAS */
#ifdef USE_BLAS
    float *B_f32 = (float *)malloc((size_t)N * K * sizeof(float));
    for (size_t i = 0; i < (size_t)N * K; i++) {
        uint32_t bits = ((uint32_t)B_bf16[i]) << 16;
        memcpy(&B_f32[i], &bits, sizeof(float));
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K,
                1.0f, A, K, B_f32, K, 0.0f, C, N);
    free(B_f32);
#else
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[m * K + k] * bf16_to_f32(B_bf16[n * K + k]);
            C[m * N + n] = sum;
        }
    }
#endif
}

/* ======================================================================== */
/* SwiGLU                                                                    */
/* ======================================================================== */

void kernel_swiglu_matvec_bf16(float *out, const uint16_t *gate_up_fused_bf16,
                                const float *x, int intermediate, int hidden) {
    /* gate_up_fused is [2*intermediate, hidden], first half is gate, second is up */
    static float *up_scratch = NULL;
    static size_t up_scratch_cap = 0;
    if ((size_t)intermediate > up_scratch_cap) {
        float *tmp = (float *)realloc(up_scratch, (size_t)intermediate * sizeof(float));
        if (!tmp) return;
        up_scratch = tmp;
        up_scratch_cap = (size_t)intermediate;
    }

    kernel_matvec_bf16(out, gate_up_fused_bf16, x, intermediate, hidden);
    kernel_matvec_bf16(up_scratch, gate_up_fused_bf16 + (size_t)intermediate * hidden,
                       x, intermediate, hidden);

    /* Apply SiLU(gate) * up — expf relies on -ffast-math for auto-vectorization */
    for (int i = 0; i < intermediate; i++) {
        float g = out[i];
        out[i] = (g / (1.0f + expf(-g))) * up_scratch[i];
    }
}

void kernel_swiglu_matvec_int8(float *out, const int8_t *gate_up_int8,
                                const float *scales, const float *x,
                                int intermediate, int hidden) {
    /* Fused SwiGLU with INT8 weights:
     * 1. Quantize x once
     * 2. gate = gate_up_int8[0:intermediate] @ x_int8
     * 3. up   = gate_up_int8[intermediate:] @ x_int8
     * 4. out = SiLU(gate) * up
     */
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

/* ======================================================================== */
/* Activation functions                                                      */
/* ======================================================================== */

void kernel_silu_inplace(float *x, int n) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    /* NEON fast sigmoid approximation: sigma(x) ≈ clamp(x * 0.25 + 0.5, 0, 1)
     * SiLU(x) = x * sigma(x) ≈ x * clamp(x * 0.25 + 0.5, 0, 1)
     * This 3rd-order rational approximation is more accurate:
     *   sigma(x) ≈ 0.5 + 0.5 * x / (1 + |x|)  (max err ~0.05 at |x|=2)
     * But for TTS quality we use the exact path via -ffast-math expf vectorization.
     */
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
#else
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
#endif
}

void kernel_gelu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    }
}

void kernel_snake_beta(float *out, const float *x, const float *alpha,
                       const float *beta, int channels, int length) {
    /* SnakeBeta: out = x + inv_beta * sin^2(alpha * x)
     * alpha/beta are preprocessed at load time:
     *   alpha = exp(alpha_log)
     *   beta = 1 / (exp(beta_log) + eps)
     */
#if defined(ACCELERATE_NEW_LAPACK)
    /* Vectorized path using Accelerate vvsinf + vDSP (4-8x faster than scalar sinf) */
    int L = length;
    vDSP_Length vL = (vDSP_Length)length;
#ifdef USE_OPENMP
#pragma omp parallel
    {
        float *tmp = (float *)malloc((size_t)length * sizeof(float));
#pragma omp for schedule(static)
        for (int c = 0; c < channels; c++) {
            if (!tmp) continue;
            float a = alpha[c];
            float inv_b = beta[c];
            const float *xc = x + (size_t)c * length;
            float *oc = out + (size_t)c * length;
            vDSP_vsmul(xc, 1, &a, tmp, 1, vL);     /* tmp = alpha * x */
            vvsinf(tmp, tmp, &L);                    /* tmp = sin(alpha * x) */
            vDSP_vsq(tmp, 1, tmp, 1, vL);           /* tmp = sin^2(alpha * x) */
            vDSP_vsma(tmp, 1, &inv_b, xc, 1, oc, 1, vL); /* out = inv_b * tmp + x */
        }
        free(tmp);
    }
#else
    float *tmp = (float *)malloc((size_t)length * sizeof(float));
    if (tmp) {
        for (int c = 0; c < channels; c++) {
            float a = alpha[c];
            float inv_b = beta[c];
            const float *xc = x + (size_t)c * length;
            float *oc = out + (size_t)c * length;
            vDSP_vsmul(xc, 1, &a, tmp, 1, vL);
            vvsinf(tmp, tmp, &L);
            vDSP_vsq(tmp, 1, tmp, 1, vL);
            vDSP_vsma(tmp, 1, &inv_b, xc, 1, oc, 1, vL);
        }
        free(tmp);
    }
#endif
    return;
#endif
    /* NEON / scalar path */
#if defined(__ARM_NEON) || defined(__aarch64__)
    /*
     * NEON fast sine approximation for SnakeBeta.
     * Uses a 5th-order polynomial: sin(x) ≈ x - x³/6 + x⁵/120
     * after range reduction to [-π, π].
     * Precision is sufficient for audio activation (max error ~0.00016).
     */
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int c = 0; c < channels; c++) {
        float a_val = alpha[c];
        float inv_b_val = beta[c];
        float32x4_t va = vdupq_n_f32(a_val);
        float32x4_t vb = vdupq_n_f32(inv_b_val);
        /* Constants for range reduction and polynomial */
        float32x4_t v_inv_twopi = vdupq_n_f32(0.15915494309189533f);  /* 1/(2π) */
        float32x4_t v_twopi    = vdupq_n_f32(6.283185307179586f);     /* 2π */
        float32x4_t v_pi       = vdupq_n_f32(3.141592653589793f);     /* π */
        float32x4_t v_neg_pi   = vdupq_n_f32(-3.141592653589793f);
        float32x4_t v_c3       = vdupq_n_f32(-1.0f / 6.0f);          /* -1/3! */
        float32x4_t v_c5       = vdupq_n_f32(1.0f / 120.0f);         /* 1/5! */
        float32x4_t v_half     = vdupq_n_f32(0.5f);
        const float *xc = x + (size_t)c * length;
        float *oc = out + (size_t)c * length;
        int t = 0;
        for (; t + 3 < length; t += 4) {
            float32x4_t vx = vld1q_f32(xc + t);
            float32x4_t ax = vmulq_f32(vx, va);
            /* Range reduce to [-π, π]: ax = ax - round(ax/(2π)) * 2π */
            float32x4_t n = vfmaq_f32(v_half, ax, v_inv_twopi);
            /* floor via convert to int and back */
            int32x4_t ni = vcvtq_s32_f32(n);
            /* Adjust for negative: if ax < 0 and fractional, ni is wrong */
            float32x4_t nf = vcvtq_f32_s32(ni);
            /* Correct: compare nf > n, subtract 1 if so */
            uint32x4_t mask = vcgtq_f32(nf, n);
            nf = vsubq_f32(nf, vreinterpretq_f32_u32(vandq_u32(mask, vreinterpretq_u32_f32(vdupq_n_f32(1.0f)))));
            ax = vfmsq_f32(ax, nf, v_twopi);
            /* Clamp to [-π, π] for safety */
            ax = vmaxq_f32(ax, v_neg_pi);
            ax = vminq_f32(ax, v_pi);
            /* sin(ax) ≈ ax * (1 + ax² * (-1/6 + ax² * 1/120)) */
            float32x4_t ax2 = vmulq_f32(ax, ax);
            float32x4_t poly = vfmaq_f32(v_c3, ax2, v_c5);
            poly = vfmaq_f32(vdupq_n_f32(1.0f), ax2, poly);
            float32x4_t s = vmulq_f32(ax, poly);
            /* out = x + inv_b * s² */
            float32x4_t s2 = vmulq_f32(s, s);
            float32x4_t result = vfmaq_f32(vx, vb, s2);
            vst1q_f32(oc + t, result);
        }
        for (; t < length; t++) {
            int idx = c * length + t;
            float s = sinf(x[idx] * a_val);
            out[idx] = x[idx] + inv_b_val * s * s;
        }
    }
#else
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int c = 0; c < channels; c++) {
        float a = alpha[c];
        float inv_b = beta[c];
        for (int t = 0; t < length; t++) {
            int idx = c * length + t;
            float s = sinf(x[idx] * a);
            out[idx] = x[idx] + inv_b * s * s;
        }
    }
#endif
}

/* ======================================================================== */
/* Element-wise operations                                                   */
/* ======================================================================== */

void kernel_add(float *out, const float *a, const float *b, int n) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    int i = 0;
    for (; i + 3 < n; i += 4)
        vst1q_f32(out + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (; i < n; i++) out[i] = a[i] + b[i];
#else
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
#endif
}

void kernel_add_inplace(float *a, const float *b, int n) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    int i = 0;
    for (; i + 3 < n; i += 4)
        vst1q_f32(a + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (; i < n; i++) a[i] += b[i];
#else
    for (int i = 0; i < n; i++) a[i] += b[i];
#endif
}

void kernel_mul_inplace(float *a, const float *b, int n) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    int i = 0;
    for (; i + 3 < n; i += 4)
        vst1q_f32(a + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (; i < n; i++) a[i] *= b[i];
#else
    for (int i = 0; i < n; i++) a[i] *= b[i];
#endif
}

void kernel_scale_inplace(float *x, float scale, int n) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t vs = vdupq_n_f32(scale);
    int i = 0;
    for (; i + 3 < n; i += 4)
        vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), vs));
    for (; i < n; i++) x[i] *= scale;
#else
    for (int i = 0; i < n; i++) x[i] *= scale;
#endif
}

void kernel_zero(float *x, int n) {
    memset(x, 0, n * sizeof(float));
}

void kernel_clamp(float *x, int n, float min_val, float max_val) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t vmin = vdupq_n_f32(min_val);
    float32x4_t vmax = vdupq_n_f32(max_val);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        v = vmaxq_f32(v, vmin);
        v = vminq_f32(v, vmax);
        vst1q_f32(x + i, v);
    }
    for (; i < n; i++) {
        if (x[i] < min_val) x[i] = min_val;
        if (x[i] > max_val) x[i] = max_val;
    }
#else
    for (int i = 0; i < n; i++) {
        if (x[i] < min_val) x[i] = min_val;
        if (x[i] > max_val) x[i] = max_val;
    }
#endif
}

float kernel_dot(const float *a, const float *b, int n) {
#ifdef USE_BLAS
    return cblas_sdot(n, a, 1, b, 1);
#elif defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    acc0 = vaddq_f32(acc0, acc1);
    float sum = vaddvq_f32(acc0);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
#endif
}

float kernel_sum_sq(const float *x, int n) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc = vfmaq_f32(acc, v, v);
    }
    float sum = vaddvq_f32(acc);
    for (; i < n; i++) sum += x[i] * x[i];
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    return sum;
#endif
}

void kernel_bf16_to_f32(float *out, const uint16_t *in, int n) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    int i = 0;
    for (; i + 7 < n; i += 8) {
        uint16x8_t bf = vld1q_u16(in + i);
        uint32x4_t lo = vshll_n_u16(vget_low_u16(bf), 16);
        uint32x4_t hi = vshll_n_u16(vget_high_u16(bf), 16);
        vst1q_f32(out + i,     vreinterpretq_f32_u32(lo));
        vst1q_f32(out + i + 4, vreinterpretq_f32_u32(hi));
    }
    for (; i < n; i++) {
        uint32_t bits = ((uint32_t)in[i]) << 16;
        memcpy(&out[i], &bits, sizeof(float));
    }
#else
    for (int i = 0; i < n; i++) {
        uint32_t bits = ((uint32_t)in[i]) << 16;
        memcpy(&out[i], &bits, sizeof(float));
    }
#endif
}

/* ======================================================================== */
/* Softmax                                                                   */
/* ======================================================================== */

void kernel_softmax(float *x, int n) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    /* NEON: find max */
    float32x4_t vmax = vdupq_n_f32(-FLT_MAX);
    int i = 0;
    for (; i + 3 < n; i += 4)
        vmax = vmaxq_f32(vmax, vld1q_f32(x + i));
    float max_val = vmaxvq_f32(vmax);
    for (; i < n; i++) if (x[i] > max_val) max_val = x[i];

    /* exp and sum (exp can't be NEON-ized, rely on -ffast-math vectorization) */
    float sum = 0.0f;
    for (i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }

    /* NEON: normalize */
    float inv_sum = 1.0f / sum;
    float32x4_t vinv = vdupq_n_f32(inv_sum);
    i = 0;
    for (; i + 3 < n; i += 4)
        vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), vinv));
    for (; i < n; i++) x[i] *= inv_sum;
#else
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv_sum;
#endif
}

/* ======================================================================== */
/* Sampling                                                                  */
/* ======================================================================== */

static float rand_uniform(float *state) {
    /* Simple xorshift-based RNG */
    uint32_t s;
    memcpy(&s, state, sizeof(uint32_t));
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    memcpy(state, &s, sizeof(uint32_t));
    return (float)(s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

void kernel_apply_repetition_penalty(float *logits, const int *token_ids,
                                     int n_tokens, int vocab_size, float penalty) {
    if (penalty == 1.0f) return;
    for (int i = 0; i < n_tokens; i++) {
        int t = token_ids[i];
        if (t >= 0 && t < vocab_size) {
            if (logits[t] > 0) logits[t] /= penalty;
            else logits[t] *= penalty;
        }
    }
}

int kernel_sample_top_k(const float *logits, int vocab_size, int top_k,
                        float top_p, float temperature, float *rng_state) {
    if (temperature <= 0.0f) temperature = 1e-5f;

    /*
     * Fast path for the common case in this project:
     *   - top_k enabled (e.g., 50)
     *   - top_p disabled (top_p >= 1.0)
     *
     * Instead of softmax over the full vocab, keep only top_k logits and
     * sample from that set.
     */
    if (top_p >= 1.0f && top_k > 0 && top_k < vocab_size) {
        int k = top_k;
        int top_idx_stack[256];
        float top_val_stack[256];
        int *top_idx = top_idx_stack;
        float *top_val = top_val_stack;

        if (k > 256) {
            top_idx = (int *)malloc((size_t)k * sizeof(int));
            top_val = (float *)malloc((size_t)k * sizeof(float));
        }

        for (int j = 0; j < k; j++) {
            top_idx[j] = -1;
            top_val[j] = -FLT_MAX;
        }

        /* Maintain descending top-k list via insertion. */
        for (int i = 0; i < vocab_size; i++) {
            float v = logits[i] / temperature;
            if (v <= top_val[k - 1]) continue;

            int p = k - 1;
            while (p > 0 && v > top_val[p - 1]) {
                top_val[p] = top_val[p - 1];
                top_idx[p] = top_idx[p - 1];
                p--;
            }
            top_val[p] = v;
            top_idx[p] = i;
        }

        /* Softmax over top-k only. */
        float max_val = top_val[0];
        float sum = 0.0f;
        for (int j = 0; j < k; j++) {
            if (top_idx[j] < 0) {
                top_val[j] = 0.0f;
                continue;
            }
            float p = expf(top_val[j] - max_val);
            top_val[j] = p;
            sum += p;
        }

        int sampled = 0;
        if (sum > 0.0f) {
            float r = rand_uniform(rng_state) * sum;
            float cumsum = 0.0f;
            for (int j = 0; j < k; j++) {
                cumsum += top_val[j];
                if (cumsum >= r) {
                    sampled = top_idx[j] >= 0 ? top_idx[j] : 0;
                    break;
                }
            }
        } else {
            sampled = top_idx[0] >= 0 ? top_idx[0] : 0;
        }

        if (k > 256) {
            free(top_idx);
            free(top_val);
        }
        return sampled;
    }

    /* Apply temperature */
    float *probs = (float *)malloc(vocab_size * sizeof(float));
    for (int i = 0; i < vocab_size; i++) probs[i] = logits[i] / temperature;

    /* Softmax */
    kernel_softmax(probs, vocab_size);

    /* Top-K: find top_k largest and zero out the rest */
    if (top_k > 0 && top_k < vocab_size) {
        /* Find the kth-largest probability using partial sort */
        float *sorted = (float *)malloc(vocab_size * sizeof(float));
        memcpy(sorted, probs, vocab_size * sizeof(float));
        /* Simple selection of kth element (partial qsort) */
        for (int i = 0; i < top_k; i++) {
            int max_idx = i;
            for (int j = i + 1; j < vocab_size; j++) {
                if (sorted[j] > sorted[max_idx]) max_idx = j;
            }
            float tmp = sorted[i]; sorted[i] = sorted[max_idx]; sorted[max_idx] = tmp;
        }
        float threshold = sorted[top_k - 1];
        free(sorted);
        for (int i = 0; i < vocab_size; i++) {
            if (probs[i] < threshold) probs[i] = 0.0f;
        }
    }

    /* Top-P (nucleus): sort by probability, accumulate until sum >= top_p */
    if (top_p < 1.0f) {
        /* Create index + prob pairs and sort descending */
        int *indices = (int *)malloc(vocab_size * sizeof(int));
        for (int i = 0; i < vocab_size; i++) indices[i] = i;
        /* Simple insertion sort for top elements */
        for (int i = 1; i < vocab_size; i++) {
            int idx = indices[i];
            float val = probs[idx];
            int j = i - 1;
            while (j >= 0 && probs[indices[j]] < val) {
                indices[j + 1] = indices[j];
                j--;
            }
            indices[j + 1] = idx;
        }
        float cumsum = 0.0f;
        int cutoff = vocab_size;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += probs[indices[i]];
            if (cumsum >= top_p) { cutoff = i + 1; break; }
        }
        for (int i = cutoff; i < vocab_size; i++) probs[indices[i]] = 0.0f;
        free(indices);
    }

    /* Renormalize */
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) sum += probs[i];
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int i = 0; i < vocab_size; i++) probs[i] *= inv;
    }

    /* Sample from the distribution */
    float r = rand_uniform(rng_state);
    float cumsum = 0.0f;
    int sampled = 0;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (cumsum >= r) { sampled = i; break; }
    }

    free(probs);
    return sampled;
}

/* ======================================================================== */
/* RoPE (Rotary Position Embedding)                                          */
/* ======================================================================== */

void kernel_rope_apply(float *q, float *k, const float *cos, const float *sin,
                       int num_heads, int head_dim) {
    /* Standard RoPE: rotate pairs of dimensions
     * q/k shape: [num_heads * head_dim] (flattened)
     * cos/sin shape: [head_dim] (shared across heads)
     */
    int half = head_dim / 2;
#if defined(__ARM_NEON) || defined(__aarch64__)
    for (int h = 0; h < num_heads; h++) {
        float *qh = q + h * head_dim;
        int i = 0;
        for (; i + 3 < half; i += 4) {
            float32x4_t vc = vld1q_f32(cos + i);
            float32x4_t vs = vld1q_f32(sin + i);
            float32x4_t vc2 = vld1q_f32(cos + i + half);
            float32x4_t vs2 = vld1q_f32(sin + i + half);
            float32x4_t vq0 = vld1q_f32(qh + i);
            float32x4_t vq1 = vld1q_f32(qh + i + half);
            /* qh[i] = q0*cos - q1*sin; qh[i+half] = q1*cos2 + q0*sin2 */
            vst1q_f32(qh + i,        vfmsq_f32(vmulq_f32(vq0, vc), vq1, vs));
            vst1q_f32(qh + i + half, vfmaq_f32(vmulq_f32(vq1, vc2), vq0, vs2));
        }
        for (; i < half; i++) {
            float q0 = qh[i], q1 = qh[i + half];
            qh[i]        = q0 * cos[i] - q1 * sin[i];
            qh[i + half] = q1 * cos[i + half] + q0 * sin[i + half];
        }
        if (k) {
            float *kh = k + h * head_dim;
            i = 0;
            for (; i + 3 < half; i += 4) {
                float32x4_t vc = vld1q_f32(cos + i);
                float32x4_t vs = vld1q_f32(sin + i);
                float32x4_t vc2 = vld1q_f32(cos + i + half);
                float32x4_t vs2 = vld1q_f32(sin + i + half);
                float32x4_t vk0 = vld1q_f32(kh + i);
                float32x4_t vk1 = vld1q_f32(kh + i + half);
                vst1q_f32(kh + i,        vfmsq_f32(vmulq_f32(vk0, vc), vk1, vs));
                vst1q_f32(kh + i + half, vfmaq_f32(vmulq_f32(vk1, vc2), vk0, vs2));
            }
            for (; i < half; i++) {
                float k0 = kh[i], k1 = kh[i + half];
                kh[i]        = k0 * cos[i] - k1 * sin[i];
                kh[i + half] = k1 * cos[i + half] + k0 * sin[i + half];
            }
        }
    }
#else
    for (int h = 0; h < num_heads; h++) {
        float *qh = q + h * head_dim;
        for (int i = 0; i < half; i++) {
            float q0 = qh[i], q1 = qh[i + half];
            qh[i]        = q0 * cos[i] - q1 * sin[i];
            qh[i + half] = q1 * cos[i] + q0 * sin[i];
        }
        if (k) {
            float *kh = k + h * head_dim;
            for (int i = 0; i < half; i++) {
                float k0 = kh[i], k1 = kh[i + half];
                kh[i]        = k0 * cos[i] - k1 * sin[i];
                kh[i + half] = k1 * cos[i] + k0 * sin[i];
            }
        }
    }
#endif
}

void kernel_mrope_apply(float *q, float *k, const float *cos, const float *sin,
                        int num_heads, int head_dim, const int *mrope_section) {
    /* M-RoPE: multimodal rotary position embedding
     * cos/sin shape: [3, head_dim] - 3 position streams  
     * mrope_section: [s0, s1, s2] - how many dims per section (doubled for rotate_half)
     *
     * The Python code does:
     *   mrope_section = mrope_section * 2  (doubled)
     *   cos = cat([m[i%3] for i, m in enumerate(cos.split(mrope_section, dim=-1))])
     *   sin = cat([m[i%3] for i, m in enumerate(sin.split(mrope_section, dim=-1))])
     *
     * So for sections [s0, s1, s2], doubled to [s0, s1, s2, s0, s1, s2]:
     *   dims [0..s0-1]:              use cos/sin from stream 0
     *   dims [s0..s0+s1-1]:          use cos/sin from stream 1
     *   dims [s0+s1..s0+s1+s2-1]:    use cos/sin from stream 2
     *   dims [s0+s1+s2..]:           repeat pattern (for the rotate_half second half)
     *
     * For TTS with all-text, all 3 position streams are the same,
     * so this reduces to standard RoPE. But we implement the full version.
     */

    /* mrope_section doubled */
    int sec[6];
    sec[0] = mrope_section[0]; sec[1] = mrope_section[1]; sec[2] = mrope_section[2];
    sec[3] = mrope_section[0]; sec[4] = mrope_section[1]; sec[5] = mrope_section[2];

    /* Build the interleaved cos/sin arrays for this position
     * cos_interleaved[d] picks from the appropriate stream based on which section d falls in
     * cos input: [3, head_dim/2] (half dim since the full cos is cat(freqs, freqs))
     * After interleaving: [head_dim]
     */
    float cos_merged[512], sin_merged[512]; /* max head_dim */
    int d = 0;
    for (int chunk = 0; chunk < 6; chunk++) {
        int stream = chunk % 3;
        /* In the Python code, cos shape is [3, seq, head_dim], and after cat of freqs,freqs it's doubled.
         * The split happens on the last dim of size head_dim (already doubled).
         * cos[stream] covers the full head_dim.
         * We split head_dim into sections sec[0..5] and pick stream accordingly.
         */
        for (int i = 0; i < sec[chunk] && d < head_dim; i++, d++) {
            cos_merged[d] = cos[stream * head_dim + d];
            sin_merged[d] = sin[stream * head_dim + d];
        }
    }

    /* Apply standard rotate_half RoPE with the merged cos/sin */
    int half = head_dim / 2;
    for (int h = 0; h < num_heads; h++) {
        float *qh = q + h * head_dim;
        for (int i = 0; i < half; i++) {
            float q0 = qh[i], q1 = qh[i + half];
            qh[i]        = q0 * cos_merged[i] - q1 * sin_merged[i];
            qh[i + half] = q1 * cos_merged[i + half] + q0 * sin_merged[i + half];
        }
        if (k) {
            float *kh = k + h * head_dim;
            for (int i = 0; i < half; i++) {
                float k0 = kh[i], k1 = kh[i + half];
                kh[i]        = k0 * cos_merged[i] - k1 * sin_merged[i];
                kh[i + half] = k1 * cos_merged[i + half] + k0 * sin_merged[i + half];
            }
        }
    }
}

/* ======================================================================== */
/* Convolution operations                                                    */
/* ======================================================================== */

void kernel_causal_conv1d(float *out, const float *input, const float *weight,
                          const float *bias, int in_channels, int out_channels,
                          int kernel_size, int length, int dilation, int groups) {
    /* Causal Conv1d: left-padded by (kernel_size-1)*dilation
     * Input: [in_channels, length]
     * Weight: [out_channels, in_channels/groups, kernel_size]
     * Output: [out_channels, length]
     */
    int eff_kernel = (kernel_size - 1) * dilation + 1;
    int pad = eff_kernel - 1; /* causal: all padding on left */
    int ch_per_group = in_channels / groups;
    int out_per_group = out_channels / groups;

#ifdef USE_BLAS
    /*
     * BLAS fast path for groups=1 and kernel>1.
     * Pack weights into [k, out, in] contiguous blocks and run one GEMM per tap.
     * This is especially beneficial for large vocoder k=7 convolutions.
     */
    if (groups == 1 && kernel_size > 1) {
        size_t pack_elems = (size_t)kernel_size * out_channels * in_channels;
        float *wpack = conv1d_wpack_get(pack_elems);
        if (wpack) {
            for (int k = 0; k < kernel_size; k++) {
                float *wk = wpack + (size_t)k * out_channels * in_channels;
                for (int oc = 0; oc < out_channels; oc++) {
                    const float *src = weight + (size_t)oc * in_channels * kernel_size + k;
                    float *dst = wk + (size_t)oc * in_channels;
                    for (int ic = 0; ic < in_channels; ic++) {
                        dst[ic] = src[(size_t)ic * kernel_size];
                    }
                }
            }

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int oc = 0; oc < out_channels; oc++) {
                float *out_ch = out + (size_t)oc * length;
                float b = bias ? bias[oc] : 0.0f;
                for (int t = 0; t < length; t++) out_ch[t] = b;
            }

            for (int k = 0; k < kernel_size; k++) {
                int shift = pad - k * dilation;
                int out_start = shift;
                int in_start = 0;
                if (out_start < 0) {
                    in_start = -out_start;
                    out_start = 0;
                }
                if (out_start >= length || in_start >= length) continue;

                int n = length - out_start;
                int n_in = length - in_start;
                if (n > n_in) n = n_in;
                if (n <= 0) continue;

                const float *wk = wpack + (size_t)k * out_channels * in_channels;
                const float *in_blk = input + in_start;
                float *out_blk = out + out_start;
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            out_channels, n, in_channels,
                            1.0f, wk, in_channels,
                            in_blk, length,
                            1.0f, out_blk, length);
            }
            return;
        }
    }
#endif

    /* Fast path for pointwise conv: k=1, dilation=1 (very common in vocoder). */
    if (kernel_size == 1 && dilation == 1) {
        if (groups == in_channels && in_channels == out_channels) {
            /* Depthwise pointwise: one scalar multiply per sample. */
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int c = 0; c < in_channels; c++) {
                float w = weight[c];
                float b = bias ? bias[c] : 0.0f;
                const float *in_ch = input + c * length;
                float *out_ch = out + c * length;
                for (int t = 0; t < length; t++) {
                    out_ch[t] = in_ch[t] * w + b;
                }
            }
            return;
        }

#ifdef USE_BLAS
        if (groups == 1) {
            /*
             * Pointwise conv with groups=1 is matrix multiply:
             *   out[out_channels, length] = weight[out_channels, in_channels] * input[in_channels, length]
             */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        out_channels, length, in_channels,
                        1.0f, weight, in_channels,
                        input, length,
                        0.0f, out, length);

            if (bias) {
                for (int oc = 0; oc < out_channels; oc++) {
                    float b = bias[oc];
                    float *out_ch = out + (size_t)oc * length;
                    for (int t = 0; t < length; t++) out_ch[t] += b;
                }
            }
            return;
        }
#endif

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int oc = 0; oc < out_channels; oc++) {
            float *out_ch = out + (size_t)oc * length;
            float b = bias ? bias[oc] : 0.0f;
            for (int t = 0; t < length; t++) out_ch[t] = b;

            int g = oc / out_per_group;
            int ic_base = g * ch_per_group;
            const float *w_row = weight + (size_t)oc * ch_per_group;
            for (int ic = 0; ic < ch_per_group; ic++) {
                float w = w_row[ic];
                const float *in_ch = input + (size_t)(ic_base + ic) * length;
                saxpy_broadcast(out_ch, w, in_ch, length);
            }
        }
        return;
    }

    /*
     * Fast path for common causal conv (dilation=1):
     * avoid inner-boundary checks for the steady-state region.
     */
    if (dilation == 1) {
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int oc = 0; oc < out_channels; oc++) {
            int g = oc / out_per_group;
            int ic_base = g * ch_per_group;
            float *out_ch = out + (size_t)oc * length;
            float b = bias ? bias[oc] : 0.0f;
            for (int t = 0; t < length; t++) out_ch[t] = b;

            for (int ic = 0; ic < ch_per_group; ic++) {
                const float *w = weight + ((size_t)oc * ch_per_group + ic) * kernel_size;
                const float *in_ch = input + (size_t)(ic_base + ic) * length;

                for (int k = 0; k < kernel_size; k++) {
                    float wk = w[k];
                    int out_start = pad - k;
                    if (out_start >= length) continue;
                    const float *src = in_ch;
                    float *dst = out_ch + out_start;
                    int n = length - out_start;
#ifdef USE_BLAS
                    cblas_saxpy(n, wk, src, 1, dst, 1);
#else
                    saxpy_broadcast(dst, wk, src, n);
#endif
                }
            }
        }
        return;
    }

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int oc = 0; oc < out_channels; oc++) {
        int g = oc / out_per_group;
        int ic_base = g * ch_per_group;
        float *out_ch = out + (size_t)oc * length;
        float b = bias ? bias[oc] : 0.0f;
        for (int t = 0; t < length; t++) out_ch[t] = b;

        for (int ic = 0; ic < ch_per_group; ic++) {
            const float *w = weight + ((size_t)oc * ch_per_group + ic) * kernel_size;
            const float *in_ch = input + (size_t)(ic_base + ic) * length;

            for (int k = 0; k < kernel_size; k++) {
                float wk = w[k];
                int shift = pad - k * dilation;
                int out_start = shift;
                int in_start = 0;
                if (out_start < 0) {
                    in_start = -out_start;
                    out_start = 0;
                }
                if (out_start >= length || in_start >= length) continue;

                int n = length - out_start;
                int n_in = length - in_start;
                if (n > n_in) n = n_in;
                if (n <= 0) continue;
#ifdef USE_BLAS
                cblas_saxpy(n, wk, in_ch + in_start, 1, out_ch + out_start, 1);
#else
                saxpy_broadcast(out_ch + out_start, wk, in_ch + in_start, n);
#endif
            }
        }
    }
}

void kernel_transposed_conv1d(float *out, const float *input, const float *weight,
                              const float *bias, int in_channels, int out_channels,
                              int kernel_size, int stride, int length, int *out_length) {
    /* Transposed Conv1d (for upsampling)
     * Input: [in_channels, length]
     * Weight: [in_channels, out_channels, kernel_size]
     * Output: [out_channels, out_len]
     *
     * out_len = (length - 1) * stride + kernel_size
     * Then trim right_pad = kernel_size - stride from the right (causal convention)
     */
    int raw_out_len = (length - 1) * stride + kernel_size;
    int right_pad = kernel_size - stride;
    int final_len = raw_out_len - right_pad;
    if (final_len < 0) final_len = 0;

#ifdef USE_BLAS
    /*
     * GEMM fast path: one sgemm per kernel tap instead of millions of tiny saxpy calls.
     * For each tap k: temp[out, len] = W_k^T @ input, then scatter to strided output.
     */
    {
        size_t wk_size = (size_t)out_channels * in_channels;
        size_t temp_size = (size_t)out_channels * length;
        float *wk_packed = (float *)malloc(wk_size * sizeof(float));
        float *temp = (float *)malloc(temp_size * sizeof(float));
        if (wk_packed && temp) {
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int oc = 0; oc < out_channels; oc++) {
                float *out_ch = out + (size_t)oc * final_len;
                float b = bias ? bias[oc] : 0.0f;
                for (int t = 0; t < final_len; t++) out_ch[t] = b;
            }

            for (int k = 0; k < kernel_size; k++) {
                for (int oc = 0; oc < out_channels; oc++) {
                    for (int ic = 0; ic < in_channels; ic++) {
                        wk_packed[(size_t)oc * in_channels + ic] =
                            weight[(size_t)ic * out_channels * kernel_size + (size_t)oc * kernel_size + k];
                    }
                }

                int n = (final_len - 1 - k) / stride + 1;
                if (n <= 0) continue;
                if (n > length) n = length;

                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            out_channels, n, in_channels,
                            1.0f, wk_packed, in_channels,
                            input, length,
                            0.0f, temp, n);

#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int oc = 0; oc < out_channels; oc++) {
                    const float *tp = temp + (size_t)oc * n;
                    float *op = out + (size_t)oc * final_len + k;
                    for (int t = 0; t < n; t++) {
                        op[t * stride] += tp[t];
                    }
                }
            }
            free(wk_packed);
            free(temp);
            if (out_length) *out_length = final_len;
            return;
        }
        free(wk_packed);
        free(temp);
    }
#endif

    /* GEMM-per-tap fallback (mirrors BLAS path algorithm, using saxpy_broadcast) */
    {
        size_t wk_size = (size_t)out_channels * in_channels;
        size_t temp_size = (size_t)out_channels * length;
        float *wk_packed = (float *)malloc(wk_size * sizeof(float));
        float *temp = (float *)malloc(temp_size * sizeof(float));
        if (wk_packed && temp) {
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int oc = 0; oc < out_channels; oc++) {
                float *out_ch = out + (size_t)oc * final_len;
                float b = bias ? bias[oc] : 0.0f;
                for (int t = 0; t < final_len; t++) out_ch[t] = b;
            }

            for (int k = 0; k < kernel_size; k++) {
                /* Pack weights for this tap: wk_packed[oc, ic] */
                for (int oc = 0; oc < out_channels; oc++) {
                    for (int ic = 0; ic < in_channels; ic++) {
                        wk_packed[(size_t)oc * in_channels + ic] =
                            weight[(size_t)ic * out_channels * kernel_size + (size_t)oc * kernel_size + k];
                    }
                }

                /* Compute valid output range for this tap */
                int n = (final_len - 1 - k) / stride + 1;
                if (n <= 0) continue;
                if (n > length) n = length;

                /* Manual GEMM via saxpy: temp[oc, t] = sum_ic wk[oc,ic] * input[ic,t] */
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int oc = 0; oc < out_channels; oc++) {
                    float *tp = temp + (size_t)oc * length;
                    memset(tp, 0, (size_t)n * sizeof(float));
                    const float *wk_row = wk_packed + (size_t)oc * in_channels;
                    for (int ic = 0; ic < in_channels; ic++) {
                        const float *in_ch = input + (size_t)ic * length;
                        saxpy_broadcast(tp, wk_row[ic], in_ch, n);
                    }
                }

                /* Scatter to strided output */
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int oc = 0; oc < out_channels; oc++) {
                    const float *tp = temp + (size_t)oc * length;
                    float *op = out + (size_t)oc * final_len + k;
                    for (int t = 0; t < n; t++) {
                        op[t * stride] += tp[t];
                    }
                }
            }
            free(wk_packed);
            free(temp);
            if (out_length) *out_length = final_len;
            return;
        }
        free(wk_packed);
        free(temp);
    }

    /* Ultimate scalar fallback (malloc failed) */
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int oc = 0; oc < out_channels; oc++) {
        float *out_ch = out + (size_t)oc * final_len;
        float b = bias ? bias[oc] : 0.0f;
        for (int t = 0; t < final_len; t++) out_ch[t] = b;

        for (int ic = 0; ic < in_channels; ic++) {
            const float *in_ch = input + (size_t)ic * length;
            const float *w = weight + (size_t)ic * out_channels * kernel_size + (size_t)oc * kernel_size;
            for (int t = 0; t < length; t++) {
                float val = in_ch[t];
                int base = t * stride;
                for (int k = 0; k < kernel_size; k++) {
                    int ot = base + k;
                    if (ot < final_len) out_ch[ot] += val * w[k];
                }
            }
        }
    }

    if (out_length) *out_length = final_len;
}

/* ======================================================================== */
/* Platform dispatch (no-op for now)                                         */
/* ======================================================================== */

void kernel_init(void) {
#ifdef USE_OPENMP
    omp_set_num_threads(4);
#endif
}
