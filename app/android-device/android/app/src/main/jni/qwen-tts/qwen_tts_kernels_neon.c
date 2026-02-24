/*
 * qwen_tts_kernels_neon.c - NEON-intensive matrix/vector kernel implementations
 *
 * Split from qwen_tts_kernels.c. Contains:
 *   - BF16/F32/INT8/Q4K matvec operations
 *   - Matmul operations (F32 and BF16)
 *   - Fused SwiGLU matvec variants (BF16, INT8, Q4K)
 *   - Dot product, sum of squares
 *   - BF16 scratch buffer management
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
/* BF16 scratch buffer (non-NEON BLAS path)                                  */
/* ======================================================================== */

#if defined(USE_BLAS) && !defined(__ARM_NEON) && !defined(__aarch64__)
/* Persistent scratch buffer for BF16->F32 conversion (non-NEON BLAS path) */
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

/* ======================================================================== */
/* Matrix-Vector Multiply                                                    */
/* ======================================================================== */

void kernel_matvec_bf16(float *out, const uint16_t *A_bf16, const float *x, int rows, int cols) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    /* Fused BF16->F32 + dot product using NEON -- no intermediate buffer needed */
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
     * A_int8[r,c] ~ round(A_bf16[r,c] / scale[r] * 127)
     * out[r] = scale[r] * sum(A_int8[r,c] * x_int8[c]) where x_int8 is quantized on-the-fly.
     *
     * We quantize x to int8 once, then use integer dot products for the inner loop.
     * On ARM with SDOT: vdotq_s32 does 16 int8 multiplies -> 4 int32 accumulations.
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
     * Dequantization: weight ~ d * scales[g] * q - dmin * mins[g]
     * where q in [0, 15] (unsigned).
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
    /* C[M,N] = A[M,K] @ B[N,K]^T  ->  C = A * B^T */
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

    /* Apply SiLU(gate) * up -- expf relies on -ffast-math for auto-vectorization */
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
/* Dot product / Sum of squares                                              */
/* ======================================================================== */

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
