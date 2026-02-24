/*
 * qwen_tts_kernels_neon.c - NEON-intensive matrix/vector kernel implementations
 *
 * Contains:
 *   - Q8_0 matvec and SwiGLU matvec (primary compute path)
 *   - F32 matvec and matmul (codec conv projections)
 *   - Dot product, sum of squares
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
/* Q8_0 Quantize Input                                                       */
/* ======================================================================== */

void kernel_quantize_x_q8(const float *x, int n, block_q8_0 *dst) {
    quantize_f32_to_q8_0(x, dst, n);
}

/* ======================================================================== */
/* Q8_0 Matrix-Vector Multiply                                               */
/* ======================================================================== */

void kernel_matvec_q8(float *out, const block_q8_0 *W_q8, const block_q8_0 *x_q8,
                       int rows, int n_blocks) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    int o = 0;

    /* Process 2 output rows at a time */
#ifdef USE_OPENMP
    #pragma omp parallel for schedule(static) num_threads(2) if(rows >= 512)
    for (o = 0; o < rows; o++) {
        const block_q8_0 *w0 = W_q8 + (size_t)o * n_blocks;
        float s0 = 0.0f;

        for (int b = 0; b < n_blocks; b++) {
            float xs = x_q8[b].scale;
            float ws0 = w0[b].scale;
            int8x16_t xv_lo = vld1q_s8(x_q8[b].qs);
            int8x16_t xv_hi = vld1q_s8(x_q8[b].qs + 16);
            int8x16_t wv0_lo = vld1q_s8(w0[b].qs);
            int8x16_t wv0_hi = vld1q_s8(w0[b].qs + 16);

#ifdef __ARM_FEATURE_DOTPROD
            int32x4_t d0 = vdotq_s32(vdupq_n_s32(0), wv0_lo, xv_lo);
            d0 = vdotq_s32(d0, wv0_hi, xv_hi);
#else
            int32x4_t d0 = vdupq_n_s32(0);
            for (int j = 0; j < 32; j += 8) {
                int8x8_t xq = vld1_s8(x_q8[b].qs + j);
                int16x8_t xq16 = vmovl_s8(xq);
                int16x8_t wq16 = vmovl_s8(vld1_s8(w0[b].qs + j));
                d0 = vmlal_s16(d0, vget_low_s16(wq16), vget_low_s16(xq16));
                d0 = vmlal_s16(d0, vget_high_s16(wq16), vget_high_s16(xq16));
            }
#endif
            s0 += ws0 * xs * (float)vaddvq_s32(d0);
        }
        out[o] = s0;
    }
#else
    /* Non-OpenMP path: 2-rows-at-a-time */
    for (; o + 1 < rows; o += 2) {
        const block_q8_0 *w0 = W_q8 + (size_t)o * n_blocks;
        const block_q8_0 *w1 = W_q8 + (size_t)(o + 1) * n_blocks;
        float s0 = 0.0f;
        float s1 = 0.0f;

        int b = 0;
        for (; b + 1 < n_blocks; b += 2) {
            /* Block b */
            float xs0 = x_q8[b].scale;
            int8x16_t xv0_lo = vld1q_s8(x_q8[b].qs);
            int8x16_t xv0_hi = vld1q_s8(x_q8[b].qs + 16);

            float ws0_0 = w0[b].scale;
            int8x16_t wv0_0_lo = vld1q_s8(w0[b].qs);
            int8x16_t wv0_0_hi = vld1q_s8(w0[b].qs + 16);

            float ws1_0 = w1[b].scale;
            int8x16_t wv1_0_lo = vld1q_s8(w1[b].qs);
            int8x16_t wv1_0_hi = vld1q_s8(w1[b].qs + 16);

#ifdef __ARM_FEATURE_DOTPROD
            int32x4_t d0_0 = vdotq_s32(vdupq_n_s32(0), wv0_0_lo, xv0_lo);
            d0_0 = vdotq_s32(d0_0, wv0_0_hi, xv0_hi);
            int32x4_t d1_0 = vdotq_s32(vdupq_n_s32(0), wv1_0_lo, xv0_lo);
            d1_0 = vdotq_s32(d1_0, wv1_0_hi, xv0_hi);
#else
            int32x4_t d0_0 = vdupq_n_s32(0);
            int32x4_t d1_0 = vdupq_n_s32(0);
            int16x8_t xp0, wp0, wp1;
            xp0 = vmovl_s8(vget_low_s8(xv0_lo));
            wp0 = vmovl_s8(vget_low_s8(wv0_0_lo));
            wp1 = vmovl_s8(vget_low_s8(wv1_0_lo));
            d0_0 = vmlal_s16(d0_0, vget_low_s16(wp0), vget_low_s16(xp0));
            d0_0 = vmlal_s16(d0_0, vget_high_s16(wp0), vget_high_s16(xp0));
            d1_0 = vmlal_s16(d1_0, vget_low_s16(wp1), vget_low_s16(xp0));
            d1_0 = vmlal_s16(d1_0, vget_high_s16(wp1), vget_high_s16(xp0));
            xp0 = vmovl_s8(vget_high_s8(xv0_lo));
            wp0 = vmovl_s8(vget_high_s8(wv0_0_lo));
            wp1 = vmovl_s8(vget_high_s8(wv1_0_lo));
            d0_0 = vmlal_s16(d0_0, vget_low_s16(wp0), vget_low_s16(xp0));
            d0_0 = vmlal_s16(d0_0, vget_high_s16(wp0), vget_high_s16(xp0));
            d1_0 = vmlal_s16(d1_0, vget_low_s16(wp1), vget_low_s16(xp0));
            d1_0 = vmlal_s16(d1_0, vget_high_s16(wp1), vget_high_s16(xp0));
            xp0 = vmovl_s8(vget_low_s8(xv0_hi));
            wp0 = vmovl_s8(vget_low_s8(wv0_0_hi));
            wp1 = vmovl_s8(vget_low_s8(wv1_0_hi));
            d0_0 = vmlal_s16(d0_0, vget_low_s16(wp0), vget_low_s16(xp0));
            d0_0 = vmlal_s16(d0_0, vget_high_s16(wp0), vget_high_s16(xp0));
            d1_0 = vmlal_s16(d1_0, vget_low_s16(wp1), vget_low_s16(xp0));
            d1_0 = vmlal_s16(d1_0, vget_high_s16(wp1), vget_high_s16(xp0));
            xp0 = vmovl_s8(vget_high_s8(xv0_hi));
            wp0 = vmovl_s8(vget_high_s8(wv0_0_hi));
            wp1 = vmovl_s8(vget_high_s8(wv1_0_hi));
            d0_0 = vmlal_s16(d0_0, vget_low_s16(wp0), vget_low_s16(xp0));
            d0_0 = vmlal_s16(d0_0, vget_high_s16(wp0), vget_high_s16(xp0));
            d1_0 = vmlal_s16(d1_0, vget_low_s16(wp1), vget_low_s16(xp0));
            d1_0 = vmlal_s16(d1_0, vget_high_s16(wp1), vget_high_s16(xp0));
#endif
            s0 += ws0_0 * xs0 * (float)vaddvq_s32(d0_0);
            s1 += ws1_0 * xs0 * (float)vaddvq_s32(d1_0);

            /* Block b+1 */
            float xs1 = x_q8[b + 1].scale;
            int8x16_t xv1_lo = vld1q_s8(x_q8[b + 1].qs);
            int8x16_t xv1_hi = vld1q_s8(x_q8[b + 1].qs + 16);

            float ws0_1 = w0[b + 1].scale;
            int8x16_t wv0_1_lo = vld1q_s8(w0[b + 1].qs);
            int8x16_t wv0_1_hi = vld1q_s8(w0[b + 1].qs + 16);

            float ws1_1 = w1[b + 1].scale;
            int8x16_t wv1_1_lo = vld1q_s8(w1[b + 1].qs);
            int8x16_t wv1_1_hi = vld1q_s8(w1[b + 1].qs + 16);

#ifdef __ARM_FEATURE_DOTPROD
            int32x4_t d0_1 = vdotq_s32(vdupq_n_s32(0), wv0_1_lo, xv1_lo);
            d0_1 = vdotq_s32(d0_1, wv0_1_hi, xv1_hi);
            int32x4_t d1_1 = vdotq_s32(vdupq_n_s32(0), wv1_1_lo, xv1_lo);
            d1_1 = vdotq_s32(d1_1, wv1_1_hi, xv1_hi);
#else
            int32x4_t d0_1 = vdupq_n_s32(0);
            int32x4_t d1_1 = vdupq_n_s32(0);
            xp0 = vmovl_s8(vget_low_s8(xv1_lo));
            wp0 = vmovl_s8(vget_low_s8(wv0_1_lo));
            wp1 = vmovl_s8(vget_low_s8(wv1_1_lo));
            d0_1 = vmlal_s16(d0_1, vget_low_s16(wp0), vget_low_s16(xp0));
            d0_1 = vmlal_s16(d0_1, vget_high_s16(wp0), vget_high_s16(xp0));
            d1_1 = vmlal_s16(d1_1, vget_low_s16(wp1), vget_low_s16(xp0));
            d1_1 = vmlal_s16(d1_1, vget_high_s16(wp1), vget_high_s16(xp0));
            xp0 = vmovl_s8(vget_high_s8(xv1_lo));
            wp0 = vmovl_s8(vget_high_s8(wv0_1_lo));
            wp1 = vmovl_s8(vget_high_s8(wv1_1_lo));
            d0_1 = vmlal_s16(d0_1, vget_low_s16(wp0), vget_low_s16(xp0));
            d0_1 = vmlal_s16(d0_1, vget_high_s16(wp0), vget_high_s16(xp0));
            d1_1 = vmlal_s16(d1_1, vget_low_s16(wp1), vget_low_s16(xp0));
            d1_1 = vmlal_s16(d1_1, vget_high_s16(wp1), vget_high_s16(xp0));
            xp0 = vmovl_s8(vget_low_s8(xv1_hi));
            wp0 = vmovl_s8(vget_low_s8(wv0_1_hi));
            wp1 = vmovl_s8(vget_low_s8(wv1_1_hi));
            d0_1 = vmlal_s16(d0_1, vget_low_s16(wp0), vget_low_s16(xp0));
            d0_1 = vmlal_s16(d0_1, vget_high_s16(wp0), vget_high_s16(xp0));
            d1_1 = vmlal_s16(d1_1, vget_low_s16(wp1), vget_low_s16(xp0));
            d1_1 = vmlal_s16(d1_1, vget_high_s16(wp1), vget_high_s16(xp0));
            xp0 = vmovl_s8(vget_high_s8(xv1_hi));
            wp0 = vmovl_s8(vget_high_s8(wv0_1_hi));
            wp1 = vmovl_s8(vget_high_s8(wv1_1_hi));
            d0_1 = vmlal_s16(d0_1, vget_low_s16(wp0), vget_low_s16(xp0));
            d0_1 = vmlal_s16(d0_1, vget_high_s16(wp0), vget_high_s16(xp0));
            d1_1 = vmlal_s16(d1_1, vget_low_s16(wp1), vget_low_s16(xp0));
            d1_1 = vmlal_s16(d1_1, vget_high_s16(wp1), vget_high_s16(xp0));
#endif
            s0 += ws0_1 * xs1 * (float)vaddvq_s32(d0_1);
            s1 += ws1_1 * xs1 * (float)vaddvq_s32(d1_1);
        }

        /* Handle remaining block */
        for (; b < n_blocks; b++) {
            float xs = x_q8[b].scale;
            int8x16_t xv_lo = vld1q_s8(x_q8[b].qs);
            int8x16_t xv_hi = vld1q_s8(x_q8[b].qs + 16);

            float ws0 = w0[b].scale;
            int8x16_t wv0_lo = vld1q_s8(w0[b].qs);
            int8x16_t wv0_hi = vld1q_s8(w0[b].qs + 16);

            float ws1 = w1[b].scale;
            int8x16_t wv1_lo = vld1q_s8(w1[b].qs);
            int8x16_t wv1_hi = vld1q_s8(w1[b].qs + 16);

#ifdef __ARM_FEATURE_DOTPROD
            int32x4_t d0 = vdotq_s32(vdupq_n_s32(0), wv0_lo, xv_lo);
            d0 = vdotq_s32(d0, wv0_hi, xv_hi);
            int32x4_t d1 = vdotq_s32(vdupq_n_s32(0), wv1_lo, xv_lo);
            d1 = vdotq_s32(d1, wv1_hi, xv_hi);
#else
            int32x4_t d0 = vdupq_n_s32(0);
            int32x4_t d1 = vdupq_n_s32(0);
            for (int j = 0; j < 32; j += 8) {
                int8x8_t xq = vld1_s8(x_q8[b].qs + j);
                int16x8_t xq16 = vmovl_s8(xq);
                int16x8_t wq0_16 = vmovl_s8(vld1_s8(w0[b].qs + j));
                int16x8_t wq1_16 = vmovl_s8(vld1_s8(w1[b].qs + j));
                d0 = vmlal_s16(d0, vget_low_s16(wq0_16), vget_low_s16(xq16));
                d0 = vmlal_s16(d0, vget_high_s16(wq0_16), vget_high_s16(xq16));
                d1 = vmlal_s16(d1, vget_low_s16(wq1_16), vget_low_s16(xq16));
                d1 = vmlal_s16(d1, vget_high_s16(wq1_16), vget_high_s16(xq16));
            }
#endif
            s0 += ws0 * xs * (float)vaddvq_s32(d0);
            s1 += ws1 * xs * (float)vaddvq_s32(d1);
        }

        out[o] = s0;
        out[o + 1] = s1;
    }

    /* Handle remaining odd row */
    for (; o < rows; o++) {
        const block_q8_0 *w0 = W_q8 + (size_t)o * n_blocks;
        float sum = 0.0f;

        for (int b = 0; b < n_blocks; b++) {
            float xs = x_q8[b].scale;
            float ws = w0[b].scale;
            int8x16_t xv_lo = vld1q_s8(x_q8[b].qs);
            int8x16_t xv_hi = vld1q_s8(x_q8[b].qs + 16);
            int8x16_t wv_lo = vld1q_s8(w0[b].qs);
            int8x16_t wv_hi = vld1q_s8(w0[b].qs + 16);

#ifdef __ARM_FEATURE_DOTPROD
            int32x4_t d = vdotq_s32(vdupq_n_s32(0), wv_lo, xv_lo);
            d = vdotq_s32(d, wv_hi, xv_hi);
#else
            int32x4_t d = vdupq_n_s32(0);
            for (int j = 0; j < 32; j += 8) {
                int8x8_t xq = vld1_s8(x_q8[b].qs + j);
                int16x8_t xq16 = vmovl_s8(xq);
                int16x8_t wq16 = vmovl_s8(vld1_s8(w0[b].qs + j));
                d = vmlal_s16(d, vget_low_s16(wq16), vget_low_s16(xq16));
                d = vmlal_s16(d, vget_high_s16(wq16), vget_high_s16(xq16));
            }
#endif
            sum += ws * xs * (float)vaddvq_s32(d);
        }

        out[o] = sum;
    }
#endif  /* USE_OPENMP else */
#else
    /* Scalar fallback */
    for (int r = 0; r < rows; r++) {
        const block_q8_0 *wr = W_q8 + (size_t)r * n_blocks;
        float sum = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            float ws = wr[b].scale;
            float xs = x_q8[b].scale;
            int32_t isum = 0;
            for (int j = 0; j < QK8_0; j++) {
                isum += (int32_t)wr[b].qs[j] * (int32_t)x_q8[b].qs[j];
            }
            sum += ws * xs * (float)isum;
        }
        out[r] = sum;
    }
#endif
}

/* ======================================================================== */
/* Q8_0 Fused SwiGLU Matvec                                                 */
/* ======================================================================== */

void kernel_swiglu_matvec_q8(float *out, const block_q8_0 *gate_up_q8,
                               const block_q8_0 *x_q8, int intermediate, int n_blocks) {
    static float *up_scratch = NULL;
    static size_t up_scratch_cap = 0;
    if ((size_t)intermediate > up_scratch_cap) {
        free(up_scratch);
        up_scratch = (float *)malloc((size_t)intermediate * sizeof(float));
        up_scratch_cap = (size_t)intermediate;
    }

    /* Gate (first `intermediate` rows) */
    kernel_matvec_q8(out, gate_up_q8, x_q8, intermediate, n_blocks);
    /* Up (next `intermediate` rows) */
    kernel_matvec_q8(up_scratch,
                      gate_up_q8 + (size_t)intermediate * n_blocks,
                      x_q8, intermediate, n_blocks);

    /* SiLU(gate) * up */
    for (int i = 0; i < intermediate; i++) {
        float g = out[i];
        out[i] = (g / (1.0f + expf(-g))) * up_scratch[i];
    }
}

/* ======================================================================== */
/* F32 Matrix-Vector Multiply (codec conv projections)                       */
/* ======================================================================== */

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

/* ======================================================================== */
/* F32 Matrix-Matrix Multiply                                                */
/* ======================================================================== */

void kernel_matmul_f32(float *C, const float *A, const float *B, int M, int N, int K) {
    /* C[M,N] = A[M,K] @ B[N,K]^T */
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

/* ======================================================================== */
/* FP16 weights × F32 input → F32 output (codec transformer)                */
/* ======================================================================== */

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

void kernel_matvec_f16w(float *out, const __fp16 *W_f16, const float *x, int rows, int cols) {
    /* Each row: load 8 FP16 weights, convert to F32, FMA with F32 input, F32 accumulation.
     * 2-wide row unroll for better ILP. */
    int r = 0;
    for (; r + 1 < rows; r += 2) {
        const __fp16 *w0 = W_f16 + (size_t)r * cols;
        const __fp16 *w1 = W_f16 + (size_t)(r + 1) * cols;
        float32x4_t acc0a = vdupq_n_f32(0.0f);
        float32x4_t acc0b = vdupq_n_f32(0.0f);
        float32x4_t acc1a = vdupq_n_f32(0.0f);
        float32x4_t acc1b = vdupq_n_f32(0.0f);
        int c = 0;
        for (; c + 7 < cols; c += 8) {
            float32x4_t xlo = vld1q_f32(x + c);
            float32x4_t xhi = vld1q_f32(x + c + 4);

            float16x8_t hw0 = vld1q_f16(w0 + c);
            float32x4_t w0lo = vcvt_f32_f16(vget_low_f16(hw0));
            float32x4_t w0hi = vcvt_f32_f16(vget_high_f16(hw0));
            acc0a = vfmaq_f32(acc0a, w0lo, xlo);
            acc0b = vfmaq_f32(acc0b, w0hi, xhi);

            float16x8_t hw1 = vld1q_f16(w1 + c);
            float32x4_t w1lo = vcvt_f32_f16(vget_low_f16(hw1));
            float32x4_t w1hi = vcvt_f32_f16(vget_high_f16(hw1));
            acc1a = vfmaq_f32(acc1a, w1lo, xlo);
            acc1b = vfmaq_f32(acc1b, w1hi, xhi);
        }
        float s0 = vaddvq_f32(vaddq_f32(acc0a, acc0b));
        float s1 = vaddvq_f32(vaddq_f32(acc1a, acc1b));
        for (; c < cols; c++) {
            float xc = x[c];
            s0 += (float)w0[c] * xc;
            s1 += (float)w1[c] * xc;
        }
        out[r] = s0;
        out[r + 1] = s1;
    }
    /* Handle remaining odd row */
    for (; r < rows; r++) {
        const __fp16 *w0 = W_f16 + (size_t)r * cols;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        int c = 0;
        for (; c + 7 < cols; c += 8) {
            float16x8_t hw = vld1q_f16(w0 + c);
            acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(hw)), vld1q_f32(x + c));
            acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(hw)), vld1q_f32(x + c + 4));
        }
        float s = vaddvq_f32(vaddq_f32(acc0, acc1));
        for (; c < cols; c++) s += (float)w0[c] * x[c];
        out[r] = s;
    }
}

#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

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
