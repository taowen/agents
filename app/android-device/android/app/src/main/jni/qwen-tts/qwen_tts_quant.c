/*
 * qwen_tts_quant.c - INT8/Q4_K Weight Quantization for Qwen3-TTS
 *
 * Contains:
 *   - INT8 matvec kernels (NEON SDOT + scalar fallback)
 *   - Q4_K super-block matvec kernel
 *   - Fused SwiGLU variants for INT8 and Q4_K
 *   - BF16 -> INT8/Q4_K quantization functions
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
 * Q4_K super-block matvec
 * ======================================================================== */

void kernel_matvec_q4k(float *out, const block_q4_k *blocks,
                        const float *x, int rows, int cols) {
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

    /* Precompute bsums: per-sub-group sum of x_int8 */
    int total_subs = cols / 32;
    static int32_t *bsums = NULL;
    static int bsums_cap = 0;
    if (total_subs > bsums_cap) {
        free(bsums);
        bsums = (int32_t *)malloc(total_subs * sizeof(int32_t));
        bsums_cap = total_subs;
    }

#if (defined(__ARM_NEON) || defined(__aarch64__)) && defined(__ARM_FEATURE_DOTPROD)
    /* NEON bsums: sum 32 int8 values per sub-group using SDOT with all-ones */
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

/* ========================================================================
 * Fused SwiGLU with Q4_K weights
 * ======================================================================== */

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
 * BF16 -> Q4_K super-block quantization
 *
 * Three-phase quantization:
 *   1. Per sub-group min/max
 *   2. Two-level scale quantization (super-block d/dmin, sub-group scales/mins)
 *   3. Quantize weights to unsigned int4 [0,15] and pack
 * ======================================================================== */

void quantize_bf16_to_q4k(const uint16_t *bf16, int rows, int cols,
                            block_q4_k **out_blocks) {
    /* cols must be divisible by QK_K=256 */
    if (cols % QK_K != 0) {
        *out_blocks = NULL;
        return;
    }

    int blocks_per_row = cols / QK_K;
    size_t total_blocks = (size_t)rows * blocks_per_row;
    *out_blocks = (block_q4_k *)malloc(total_blocks * sizeof(block_q4_k));
    if (!*out_blocks) return;

    /* Temporary buffer for dequantized f32 values (one super-block) */
    float tmp[QK_K];

    for (int r = 0; r < rows; r++) {
        const uint16_t *row = bf16 + (size_t)r * cols;

        for (int b = 0; b < blocks_per_row; b++) {
            block_q4_k *blk = *out_blocks + (size_t)r * blocks_per_row + b;
            int col_start = b * QK_K;

            /* Convert BF16 block to F32 */
            for (int i = 0; i < QK_K; i++) {
                uint32_t bits = ((uint32_t)row[col_start + i]) << 16;
                __builtin_memcpy(&tmp[i], &bits, sizeof(float));
            }

            /* Phase 1: Per sub-group min/max */
            float per_group_scale[Q4K_NUM_SUBS];
            float per_group_min[Q4K_NUM_SUBS];

            for (int g = 0; g < Q4K_NUM_SUBS; g++) {
                float gmin = tmp[g * 32];
                float gmax = tmp[g * 32];
                for (int i = 1; i < 32; i++) {
                    float v = tmp[g * 32 + i];
                    if (v < gmin) gmin = v;
                    if (v > gmax) gmax = v;
                }
                float range = gmax - gmin;
                per_group_scale[g] = range / 15.0f;
                per_group_min[g] = -gmin;
                if (per_group_min[g] < 0.0f) per_group_min[g] = 0.0f;
            }

            /* Phase 2: Two-level scale quantization */
            float max_scale = 0.0f;
            float max_min = 0.0f;
            for (int g = 0; g < Q4K_NUM_SUBS; g++) {
                if (per_group_scale[g] > max_scale) max_scale = per_group_scale[g];
                if (per_group_min[g] > max_min) max_min = per_group_min[g];
            }

            float d = max_scale / 255.0f;
            float dmin = (max_min > 0.0f) ? max_min / 255.0f : 0.0f;
            blk->d = d;
            blk->dmin = dmin;

            float inv_d = (d > 0.0f) ? 1.0f / d : 0.0f;
            float inv_dmin = (dmin > 0.0f) ? 1.0f / dmin : 0.0f;

            for (int g = 0; g < Q4K_NUM_SUBS; g++) {
                float sv = per_group_scale[g] * inv_d;
                int si = (int)(sv + 0.5f);
                if (si > 255) si = 255;
                if (si < 0) si = 0;
                blk->scales[g] = (uint8_t)si;

                float mv = per_group_min[g] * inv_dmin;
                int mi = (int)(mv + 0.5f);
                if (mi > 255) mi = 255;
                if (mi < 0) mi = 0;
                blk->mins[g] = (uint8_t)mi;
            }

            /* Phase 3: Quantize weights -> unsigned int4 [0, 15] and pack */
            for (int g = 0; g < Q4K_NUM_SUBS; g++) {
                float eff_scale = d * (float)blk->scales[g];
                float eff_min = dmin * (float)blk->mins[g];
                float inv_eff_scale = (eff_scale > 0.0f) ? 1.0f / eff_scale : 0.0f;

                for (int i = 0; i < 16; i++) {
                    float v0 = tmp[g * 32 + i * 2];
                    float v1 = tmp[g * 32 + i * 2 + 1];

                    int q0, q1;
                    if (eff_scale > 0.0f) {
                        float fq0 = (v0 + eff_min) * inv_eff_scale;
                        float fq1 = (v1 + eff_min) * inv_eff_scale;
                        q0 = (int)(fq0 + 0.5f);
                        q1 = (int)(fq1 + 0.5f);
                    } else {
                        q0 = 0;
                        q1 = 0;
                    }
                    if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
                    if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;

                    /* Pack: low nibble = even index, high nibble = odd index */
                    blk->qs[g * 16 + i] = (uint8_t)(q0 | (q1 << 4));
                }
            }
        }
    }
}

/* ========================================================================
 * Pre-quantized Weight Cache
 *
 * Binary cache file format (.qcache):
 *   header (qcache_header_t)
 *   for each talker layer:
 *     wqkv_q4k | gate_up_q4k | wo_int8 + wo_scales | down_int8 + down_scales
 *   for each subtalker layer:
 *     wqkv_q4k | gate_up_q4k | wo_q4k | down_q4k
 * ======================================================================== */

#ifndef __EMSCRIPTEN__
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#define QCACHE_MAGIC   0x31435151   /* "QQC1" */
#define QCACHE_VERSION 1

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t source_size;         /* original safetensors total file size for validation */
    uint32_t n_talker_layers;
    uint32_t n_subtalker_layers;
    /* Talker per-layer sizes */
    uint32_t tk_wqkv_q4k_bytes;
    uint32_t tk_gate_up_q4k_bytes;
    uint32_t tk_wo_int8_bytes;
    uint32_t tk_wo_scales_bytes;
    uint32_t tk_down_int8_bytes;
    uint32_t tk_down_scales_bytes;
    /* Subtalker per-layer sizes */
    uint32_t st_wqkv_q4k_bytes;
    uint32_t st_gate_up_q4k_bytes;
    uint32_t st_wo_q4k_bytes;
    uint32_t st_down_q4k_bytes;
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

    /* Compute per-layer sizes */
    int tk_qkv_rows = cfg->talker_heads * cfg->talker_head_dim +
                       2 * cfg->talker_kv_heads * cfg->talker_head_dim;
    int tk_qkv_bpr = cfg->talker_hidden / QK_K;
    uint32_t tk_wqkv_q4k_bytes = (uint32_t)((size_t)tk_qkv_rows * tk_qkv_bpr * sizeof(block_q4_k));

    int tk_gu_rows = 2 * cfg->talker_intermediate;
    int tk_gu_bpr = cfg->talker_hidden / QK_K;
    uint32_t tk_gate_up_q4k_bytes = (uint32_t)((size_t)tk_gu_rows * tk_gu_bpr * sizeof(block_q4_k));

    int tk_wo_rows = cfg->talker_hidden;
    int tk_wo_cols = cfg->talker_heads * cfg->talker_head_dim;
    uint32_t tk_wo_int8_bytes = (uint32_t)((size_t)tk_wo_rows * tk_wo_cols);
    uint32_t tk_wo_scales_bytes = (uint32_t)(tk_wo_rows * sizeof(float));

    int tk_down_rows = cfg->talker_hidden;
    int tk_down_cols = cfg->talker_intermediate;
    uint32_t tk_down_int8_bytes = (uint32_t)((size_t)tk_down_rows * tk_down_cols);
    uint32_t tk_down_scales_bytes = (uint32_t)(tk_down_rows * sizeof(float));

    /* Subtalker layer Q4_K sizes */
    int st_qkv_rows = cfg->subtalker_heads * cfg->subtalker_head_dim +
                       2 * cfg->subtalker_kv_heads * cfg->subtalker_head_dim;
    int st_qkv_bpr = cfg->subtalker_hidden / QK_K;
    uint32_t st_wqkv_q4k_bytes = (uint32_t)((size_t)st_qkv_rows * st_qkv_bpr * sizeof(block_q4_k));

    int st_gu_rows = 2 * cfg->subtalker_intermediate;
    int st_gu_bpr = cfg->subtalker_hidden / QK_K;
    uint32_t st_gate_up_q4k_bytes = (uint32_t)((size_t)st_gu_rows * st_gu_bpr * sizeof(block_q4_k));

    int st_wo_rows = cfg->subtalker_hidden;
    int st_wo_cols = cfg->subtalker_heads * cfg->subtalker_head_dim;
    int st_wo_bpr = st_wo_cols / QK_K;
    uint32_t st_wo_q4k_bytes = (uint32_t)((size_t)st_wo_rows * st_wo_bpr * sizeof(block_q4_k));

    int st_down_rows = cfg->subtalker_hidden;
    int st_down_cols = cfg->subtalker_intermediate;
    int st_down_bpr = st_down_cols / QK_K;
    uint32_t st_down_q4k_bytes = (uint32_t)((size_t)st_down_rows * st_down_bpr * sizeof(block_q4_k));

    /* Build header */
    qcache_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = QCACHE_MAGIC;
    hdr.version = QCACHE_VERSION;
    hdr.source_size = get_safetensors_size(ctx->model_dir);
    hdr.n_talker_layers = (uint32_t)cfg->talker_layers;
    hdr.n_subtalker_layers = (uint32_t)cfg->subtalker_layers;
    hdr.tk_wqkv_q4k_bytes = tk_wqkv_q4k_bytes;
    hdr.tk_gate_up_q4k_bytes = tk_gate_up_q4k_bytes;
    hdr.tk_wo_int8_bytes = tk_wo_int8_bytes;
    hdr.tk_wo_scales_bytes = tk_wo_scales_bytes;
    hdr.tk_down_int8_bytes = tk_down_int8_bytes;
    hdr.tk_down_scales_bytes = tk_down_scales_bytes;
    hdr.st_wqkv_q4k_bytes = st_wqkv_q4k_bytes;
    hdr.st_gate_up_q4k_bytes = st_gate_up_q4k_bytes;
    hdr.st_wo_q4k_bytes = st_wo_q4k_bytes;
    hdr.st_down_q4k_bytes = st_down_q4k_bytes;

    FILE *f = fopen(path, "wb");
    if (!f) {
        if (qwen_tts_verbose >= 1)
            fprintf(stderr, "Warning: cannot create qcache at %s\n", path);
        return -1;
    }

    fwrite(&hdr, sizeof(hdr), 1, f);

    /* Write talker layers */
    for (int i = 0; i < cfg->talker_layers; i++) {
        qwen_tts_talker_layer_t *l = &ctx->talker.layers[i];
        if (l->wqkv_q4k)   fwrite(l->wqkv_q4k, 1, tk_wqkv_q4k_bytes, f);
        else { void *z = calloc(1, tk_wqkv_q4k_bytes); fwrite(z, 1, tk_wqkv_q4k_bytes, f); free(z); }
        if (l->gate_up_q4k) fwrite(l->gate_up_q4k, 1, tk_gate_up_q4k_bytes, f);
        else { void *z = calloc(1, tk_gate_up_q4k_bytes); fwrite(z, 1, tk_gate_up_q4k_bytes, f); free(z); }
        if (l->wo_int8)     fwrite(l->wo_int8, 1, tk_wo_int8_bytes, f);
        else { void *z = calloc(1, tk_wo_int8_bytes); fwrite(z, 1, tk_wo_int8_bytes, f); free(z); }
        if (l->wo_scales)   fwrite(l->wo_scales, 1, tk_wo_scales_bytes, f);
        else { void *z = calloc(1, tk_wo_scales_bytes); fwrite(z, 1, tk_wo_scales_bytes, f); free(z); }
        if (l->down_int8)   fwrite(l->down_int8, 1, tk_down_int8_bytes, f);
        else { void *z = calloc(1, tk_down_int8_bytes); fwrite(z, 1, tk_down_int8_bytes, f); free(z); }
        if (l->down_scales) fwrite(l->down_scales, 1, tk_down_scales_bytes, f);
        else { void *z = calloc(1, tk_down_scales_bytes); fwrite(z, 1, tk_down_scales_bytes, f); free(z); }
    }

    /* Write subtalker layers */
    for (int i = 0; i < cfg->subtalker_layers; i++) {
        qwen_tts_subtalker_layer_t *l = &ctx->subtalker.layers[i];
        if (l->wqkv_q4k)    fwrite(l->wqkv_q4k, 1, st_wqkv_q4k_bytes, f);
        else { void *z = calloc(1, st_wqkv_q4k_bytes); fwrite(z, 1, st_wqkv_q4k_bytes, f); free(z); }
        if (l->gate_up_q4k) fwrite(l->gate_up_q4k, 1, st_gate_up_q4k_bytes, f);
        else { void *z = calloc(1, st_gate_up_q4k_bytes); fwrite(z, 1, st_gate_up_q4k_bytes, f); free(z); }
        if (l->wo_q4k)      fwrite(l->wo_q4k, 1, st_wo_q4k_bytes, f);
        else { void *z = calloc(1, st_wo_q4k_bytes); fwrite(z, 1, st_wo_q4k_bytes, f); free(z); }
        if (l->down_q4k)    fwrite(l->down_q4k, 1, st_down_q4k_bytes, f);
        else { void *z = calloc(1, st_down_q4k_bytes); fwrite(z, 1, st_down_q4k_bytes, f); free(z); }
    }

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
    size_t tk_per_layer = (size_t)hdr->tk_wqkv_q4k_bytes + hdr->tk_gate_up_q4k_bytes +
                          hdr->tk_wo_int8_bytes + hdr->tk_wo_scales_bytes +
                          hdr->tk_down_int8_bytes + hdr->tk_down_scales_bytes;
    size_t st_per_layer = (size_t)hdr->st_wqkv_q4k_bytes + hdr->st_gate_up_q4k_bytes +
                          hdr->st_wo_q4k_bytes + hdr->st_down_q4k_bytes;
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
        CACHE_COPY(l->wqkv_q4k, block_q4_k *, hdr->tk_wqkv_q4k_bytes);
        CACHE_COPY(l->gate_up_q4k, block_q4_k *, hdr->tk_gate_up_q4k_bytes);
        CACHE_COPY(l->wo_int8, int8_t *, hdr->tk_wo_int8_bytes);
        CACHE_COPY(l->wo_scales, float *, hdr->tk_wo_scales_bytes);
        CACHE_COPY(l->down_int8, int8_t *, hdr->tk_down_int8_bytes);
        CACHE_COPY(l->down_scales, float *, hdr->tk_down_scales_bytes);
    }

    for (int i = 0; i < cfg->subtalker_layers; i++) {
        qwen_tts_subtalker_layer_t *l = &ctx->subtalker.layers[i];
        CACHE_COPY(l->wqkv_q4k, block_q4_k *, hdr->st_wqkv_q4k_bytes);
        CACHE_COPY(l->gate_up_q4k, block_q4_k *, hdr->st_gate_up_q4k_bytes);
        CACHE_COPY(l->wo_q4k, block_q4_k *, hdr->st_wo_q4k_bytes);
        CACHE_COPY(l->down_q4k, block_q4_k *, hdr->st_down_q4k_bytes);
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
