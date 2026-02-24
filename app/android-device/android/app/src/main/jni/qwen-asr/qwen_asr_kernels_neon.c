/*
 * qwen_asr_kernels_neon.c - ARM NEON hot kernels
 */

#include "qwen_asr_kernels_impl.h"

#ifdef __ARM_NEON

#include <arm_neon.h>
#include <string.h>
#include <stdlib.h>

void qwen_q8_matvec_fused_neon(float *y, const block_q8_0 *x_q8,
                                const block_q8_0 *W_q8, const float *bias,
                                int n_blocks, int out_dim) {
    int o = 0;

    /* Process 2 output rows at a time */
    for (; o + 1 < out_dim; o += 2) {
        const block_q8_0 *w0 = W_q8 + (size_t)o * n_blocks;
        const block_q8_0 *w1 = W_q8 + (size_t)(o + 1) * n_blocks;
        float s0 = bias ? bias[o] : 0.0f;
        float s1 = bias ? bias[o + 1] : 0.0f;

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
            /* Fallback without dotprod: widen to int16 and multiply */
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

        y[o] = s0;
        y[o + 1] = s1;
    }

    /* Handle remaining odd row */
    for (; o < out_dim; o++) {
        const block_q8_0 *w0 = W_q8 + (size_t)o * n_blocks;
        float sum = bias ? bias[o] : 0.0f;

        for (int b = 0; b < n_blocks; b++) {
            float xs = x_q8[b].scale;
            int8x16_t xv_lo = vld1q_s8(x_q8[b].qs);
            int8x16_t xv_hi = vld1q_s8(x_q8[b].qs + 16);
            float ws = w0[b].scale;
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
        y[o] = sum;
    }
}

float qwen_dot_f32_neon(const float *a, const float *b, int n) {
    int i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        acc0 = vfmaq_f32(acc0, a0, b0);
        acc1 = vfmaq_f32(acc1, a1, b1);
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

void qwen_vec_scale_inplace_neon(float *dst, float scale, int n) {
    int i = 0;
    float32x4_t s = vdupq_n_f32(scale);
    for (; i + 8 <= n; i += 8) {
        float32x4_t d0 = vld1q_f32(dst + i);
        float32x4_t d1 = vld1q_f32(dst + i + 4);
        vst1q_f32(dst + i, vfmaq_f32(vdupq_n_f32(0.0f), d0, s));
        vst1q_f32(dst + i + 4, vfmaq_f32(vdupq_n_f32(0.0f), d1, s));
    }
    for (; i < n; i++) dst[i] *= scale;
}

void qwen_vec_axpy_inplace_neon(float *dst, const float *src, float alpha, int n) {
    int i = 0;
    float32x4_t a = vdupq_n_f32(alpha);
    for (; i + 8 <= n; i += 8) {
        float32x4_t d0 = vld1q_f32(dst + i);
        float32x4_t s0 = vld1q_f32(src + i);
        float32x4_t d1 = vld1q_f32(dst + i + 4);
        float32x4_t s1 = vld1q_f32(src + i + 4);
        vst1q_f32(dst + i, vfmaq_f32(d0, s0, a));
        vst1q_f32(dst + i + 4, vfmaq_f32(d1, s1, a));
    }
    for (; i < n; i++) dst[i] += alpha * src[i];
}

void qwen_vec_scale_add_neon(float *dst, const float *src, float correction, int n) {
    int i = 0;
    float32x4_t c = vdupq_n_f32(correction);
    for (; i + 8 <= n; i += 8) {
        float32x4_t d0 = vld1q_f32(dst + i);
        float32x4_t s0 = vld1q_f32(src + i);
        float32x4_t d1 = vld1q_f32(dst + i + 4);
        float32x4_t s1 = vld1q_f32(src + i + 4);
        vst1q_f32(dst + i, vfmaq_f32(s0, d0, c));
        vst1q_f32(dst + i + 4, vfmaq_f32(s1, d1, c));
    }
    for (; i < n; i++) dst[i] = dst[i] * correction + src[i];
}

/* ========================================================================
 * Q4_K Super-Block MatVec (NEON + SDOT)
 * ======================================================================== */

/* Quantize x to int8 for Q4_K matvec (NEON-accelerated) */
static void q4k_quantize_x_int8(const float *x, int cols,
                                   int8_t *x_int8, float *x_scale_out) {
    float x_absmax = 0.0f;
    float32x4_t vabsmax = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < cols; i += 4)
        vabsmax = vmaxq_f32(vabsmax, vabsq_f32(vld1q_f32(x + i)));
    x_absmax = vmaxvq_f32(vabsmax);
    for (; i < cols; i++) {
        float a = x[i] > 0 ? x[i] : -x[i];
        if (a > x_absmax) x_absmax = a;
    }

    *x_scale_out = x_absmax / 127.0f;
    float inv_x_scale = (x_absmax > 0.0f) ? 127.0f / x_absmax : 0.0f;

    float32x4_t vscale = vdupq_n_f32(inv_x_scale);
    int c = 0;
    for (; c + 7 < cols; c += 8) {
        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + c), vscale));
        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + c + 4), vscale));
        int16x4_t s0 = vqmovn_s32(i0);
        int16x4_t s1 = vqmovn_s32(i1);
        int8x8_t b = vqmovn_s16(vcombine_s16(s0, s1));
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

void qwen_q4k_matvec_fused_neon(float *out, const block_q4_k *blocks,
                                  const float *x, int rows, int cols) {
    int blocks_per_row = cols / QK_K;

    /* Quantize x to int8 (reusable across rows) */
    static int8_t *x_int8 = NULL;
    static int x_int8_cap = 0;
    if (cols > x_int8_cap) {
        free(x_int8);
        x_int8 = (int8_t *)malloc(((cols + 15) & ~15) * sizeof(int8_t));
        x_int8_cap = cols;
    }
    float x_scale;
    q4k_quantize_x_int8(x, cols, x_int8, &x_scale);

    /* Precompute bsums: per-sub-group sum of x_int8 */
    int total_subs = cols / 32;
    static int32_t *bsums = NULL;
    static int bsums_cap = 0;
    if (total_subs > bsums_cap) {
        free(bsums);
        bsums = (int32_t *)malloc(total_subs * sizeof(int32_t));
        bsums_cap = total_subs;
    }

#ifdef __ARM_FEATURE_DOTPROD
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

#ifdef __ARM_FEATURE_DOTPROD
    /* NEON + SDOT path */
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

                /* Min correction */
                min_acc += (int32_t)blk->mins[g] * bsums[b * Q4K_NUM_SUBS + g];
            }

            /* Only 1 vaddvq_s32 per super-block */
            row_sum += blk->d * (float)vaddvq_s32(acc) - blk->dmin * (float)min_acc;
        }
        out[r] = row_sum * x_scale;
    }
#else
    /* NEON fallback without SDOT: use widening multiply-accumulate */
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

void qwen_q4k_matvec_preq_neon(float *out, const block_q4_k *blocks,
                                 const int8_t *x_int8, float x_scale,
                                 const int32_t *bsums,
                                 int rows, int cols) {
    int blocks_per_row = cols / QK_K;

#ifdef __ARM_FEATURE_DOTPROD
    for (int r = 0; r < rows; r++) {
        float row_sum = 0.0f;

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_k *blk = &blocks[(size_t)r * blocks_per_row + b];
            const int8_t *xq = x_int8 + b * QK_K;

            int32x4_t acc = vdupq_n_s32(0);
            int32_t min_acc = 0;

            for (int g = 0; g < Q4K_NUM_SUBS; g++) {
                uint8x16_t packed = vld1q_u8(blk->qs + g * 16);
                int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0x0F)));
                int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(packed, 4));
                int8x16x2_t z = vzipq_s8(lo, hi);

                int32x4_t dot = vdupq_n_s32(0);
                dot = vdotq_s32(dot, z.val[0], vld1q_s8(xq + g * 32));
                dot = vdotq_s32(dot, z.val[1], vld1q_s8(xq + g * 32 + 16));
                dot = vmulq_n_s32(dot, (int32_t)blk->scales[g]);
                acc = vaddq_s32(acc, dot);

                min_acc += (int32_t)blk->mins[g] * bsums[b * Q4K_NUM_SUBS + g];
            }

            row_sum += blk->d * (float)vaddvq_s32(acc) - blk->dmin * (float)min_acc;
        }
        out[r] = row_sum * x_scale;
    }
#else
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
                    uint8_t pk = blk->qs[g * 16 + i];
                    int8_t lo_v = (int8_t)(pk & 0x0F);
                    int8_t hi_v = (int8_t)(pk >> 4);
                    dot += (int32_t)lo_v * (int32_t)xq[g * 32 + i * 2];
                    dot += (int32_t)hi_v * (int32_t)xq[g * 32 + i * 2 + 1];
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

void qwen_q4k_argmax_range_neon(const block_q4_k *blocks,
                                  const float *x, int cols,
                                  int start, int end,
                                  int *best_out, float *best_val_out) {
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
    q4k_quantize_x_int8(x, cols, x_int8, &x_scale);

    int total_subs = cols / 32;
    static int32_t *bsums = NULL;
    static int bsums_cap = 0;
    if (total_subs > bsums_cap) {
        free(bsums);
        bsums = (int32_t *)malloc(total_subs * sizeof(int32_t));
        bsums_cap = total_subs;
    }

#ifdef __ARM_FEATURE_DOTPROD
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

    int best = start;
    float best_val = -1e30f;

#ifdef __ARM_FEATURE_DOTPROD
    for (int r = start; r < end; r++) {
        float row_sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_k *blk = &blocks[(size_t)r * blocks_per_row + b];
            const int8_t *xq = x_int8 + b * QK_K;

            int32x4_t acc = vdupq_n_s32(0);
            int32_t min_acc = 0;

            for (int g = 0; g < Q4K_NUM_SUBS; g++) {
                uint8x16_t packed = vld1q_u8(blk->qs + g * 16);
                int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0x0F)));
                int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(packed, 4));
                int8x16x2_t z = vzipq_s8(lo, hi);

                int32x4_t dot = vdupq_n_s32(0);
                dot = vdotq_s32(dot, z.val[0], vld1q_s8(xq + g * 32));
                dot = vdotq_s32(dot, z.val[1], vld1q_s8(xq + g * 32 + 16));
                dot = vmulq_n_s32(dot, (int32_t)blk->scales[g]);
                acc = vaddq_s32(acc, dot);
                min_acc += (int32_t)blk->mins[g] * bsums[b * Q4K_NUM_SUBS + g];
            }
            row_sum += blk->d * (float)vaddvq_s32(acc) - blk->dmin * (float)min_acc;
        }
        float val = row_sum * x_scale;
        if (val > best_val) { best_val = val; best = r; }
    }
#else
    for (int r = start; r < end; r++) {
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
        float val = row_sum * x_scale;
        if (val > best_val) { best_val = val; best = r; }
    }
#endif

    *best_out = best;
    *best_val_out = best_val;
}

/* Q4_K batched GEMM inner: process a chunk of output rows for all M tokens.
 * Uses 4-token unrolling to amortize weight nibble unpacking. */
void qwen_q4k_gemm_chunk_neon(
    float *Y, int Y_stride,
    const block_q4_k *W_q4k, int blocks_per_row,
    const int8_t *x_int8, int K,
    const float *x_scales,
    const int32_t *bsums, int total_subs,
    int M, int r_start, int r_end
) {
    int n_rows = r_end - r_start;
    if (n_rows <= 0) return;

    const block_q4_k *W_chunk = W_q4k + (size_t)r_start * blocks_per_row;

#ifdef __ARM_FEATURE_DOTPROD
    uint8x16_t mask_0f = vdupq_n_u8(0x0F);

    for (int r = 0; r < n_rows; r++) {
        const block_q4_k *row_blk = W_chunk + (size_t)r * blocks_per_row;
        int out_idx = r_start + r;

        /* 4-wide token loop */
        int m = 0;
        for (; m + 3 < M; m += 4) {
            const int8_t *xi0 = x_int8 + (size_t)(m+0) * K;
            const int8_t *xi1 = x_int8 + (size_t)(m+1) * K;
            const int8_t *xi2 = x_int8 + (size_t)(m+2) * K;
            const int8_t *xi3 = x_int8 + (size_t)(m+3) * K;
            const int32_t *bs0 = bsums + (size_t)(m+0) * total_subs;
            const int32_t *bs1 = bsums + (size_t)(m+1) * total_subs;
            const int32_t *bs2 = bsums + (size_t)(m+2) * total_subs;
            const int32_t *bs3 = bsums + (size_t)(m+3) * total_subs;

            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

            for (int b = 0; b < blocks_per_row; b++) {
                const block_q4_k *blk = &row_blk[b];
                int boff = b * QK_K;
                int bsoff = b * Q4K_NUM_SUBS;

                int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
                int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
                int32_t ma0 = 0, ma1 = 0, ma2 = 0, ma3 = 0;

                for (int g = 0; g < Q4K_NUM_SUBS; g++) {
                    uint8x16_t packed = vld1q_u8(blk->qs + g * 16);
                    int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(packed, mask_0f));
                    int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(packed, 4));
                    int8x16x2_t z = vzipq_s8(lo, hi);
                    int32_t sc = (int32_t)blk->scales[g];
                    int32_t mn = (int32_t)blk->mins[g];
                    int goff = boff + g * 32;

                    int8x16_t x0lo = vld1q_s8(xi0 + goff);
                    int8x16_t x0hi = vld1q_s8(xi0 + goff + 16);
                    int32x4_t d0 = vdotq_s32(vdupq_n_s32(0), z.val[0], x0lo);
                    d0 = vdotq_s32(d0, z.val[1], x0hi);
                    a0 = vaddq_s32(a0, vmulq_n_s32(d0, sc));
                    ma0 += mn * bs0[bsoff + g];

                    int8x16_t x1lo = vld1q_s8(xi1 + goff);
                    int8x16_t x1hi = vld1q_s8(xi1 + goff + 16);
                    int32x4_t d1 = vdotq_s32(vdupq_n_s32(0), z.val[0], x1lo);
                    d1 = vdotq_s32(d1, z.val[1], x1hi);
                    a1 = vaddq_s32(a1, vmulq_n_s32(d1, sc));
                    ma1 += mn * bs1[bsoff + g];

                    int8x16_t x2lo = vld1q_s8(xi2 + goff);
                    int8x16_t x2hi = vld1q_s8(xi2 + goff + 16);
                    int32x4_t d2 = vdotq_s32(vdupq_n_s32(0), z.val[0], x2lo);
                    d2 = vdotq_s32(d2, z.val[1], x2hi);
                    a2 = vaddq_s32(a2, vmulq_n_s32(d2, sc));
                    ma2 += mn * bs2[bsoff + g];

                    int8x16_t x3lo = vld1q_s8(xi3 + goff);
                    int8x16_t x3hi = vld1q_s8(xi3 + goff + 16);
                    int32x4_t d3 = vdotq_s32(vdupq_n_s32(0), z.val[0], x3lo);
                    d3 = vdotq_s32(d3, z.val[1], x3hi);
                    a3 = vaddq_s32(a3, vmulq_n_s32(d3, sc));
                    ma3 += mn * bs3[bsoff + g];
                }

                float d = blk->d, dm = blk->dmin;
                s0 += d * (float)vaddvq_s32(a0) - dm * (float)ma0;
                s1 += d * (float)vaddvq_s32(a1) - dm * (float)ma1;
                s2 += d * (float)vaddvq_s32(a2) - dm * (float)ma2;
                s3 += d * (float)vaddvq_s32(a3) - dm * (float)ma3;
            }

            Y[(size_t)(m+0) * Y_stride + out_idx] = s0 * x_scales[m+0];
            Y[(size_t)(m+1) * Y_stride + out_idx] = s1 * x_scales[m+1];
            Y[(size_t)(m+2) * Y_stride + out_idx] = s2 * x_scales[m+2];
            Y[(size_t)(m+3) * Y_stride + out_idx] = s3 * x_scales[m+3];
        }

        /* 2-wide remainder */
        for (; m + 1 < M; m += 2) {
            const int8_t *xi0 = x_int8 + (size_t)m * K;
            const int8_t *xi1 = x_int8 + (size_t)(m+1) * K;
            const int32_t *bs0 = bsums + (size_t)m * total_subs;
            const int32_t *bs1 = bsums + (size_t)(m+1) * total_subs;
            float s0 = 0.0f, s1 = 0.0f;

            for (int b = 0; b < blocks_per_row; b++) {
                const block_q4_k *blk = &row_blk[b];
                int boff = b * QK_K;
                int bsoff = b * Q4K_NUM_SUBS;
                int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
                int32_t ma0 = 0, ma1 = 0;
                for (int g = 0; g < Q4K_NUM_SUBS; g++) {
                    uint8x16_t packed = vld1q_u8(blk->qs + g * 16);
                    int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(packed, mask_0f));
                    int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(packed, 4));
                    int8x16x2_t z = vzipq_s8(lo, hi);
                    int32_t sc = (int32_t)blk->scales[g];
                    int32_t mn = (int32_t)blk->mins[g];
                    int goff = boff + g * 32;
                    int32x4_t d0 = vdotq_s32(vdupq_n_s32(0), z.val[0], vld1q_s8(xi0 + goff));
                    d0 = vdotq_s32(d0, z.val[1], vld1q_s8(xi0 + goff + 16));
                    a0 = vaddq_s32(a0, vmulq_n_s32(d0, sc));
                    ma0 += mn * bs0[bsoff + g];
                    int32x4_t d1 = vdotq_s32(vdupq_n_s32(0), z.val[0], vld1q_s8(xi1 + goff));
                    d1 = vdotq_s32(d1, z.val[1], vld1q_s8(xi1 + goff + 16));
                    a1 = vaddq_s32(a1, vmulq_n_s32(d1, sc));
                    ma1 += mn * bs1[bsoff + g];
                }
                float d = blk->d, dm = blk->dmin;
                s0 += d * (float)vaddvq_s32(a0) - dm * (float)ma0;
                s1 += d * (float)vaddvq_s32(a1) - dm * (float)ma1;
            }
            Y[(size_t)m * Y_stride + out_idx] = s0 * x_scales[m];
            Y[(size_t)(m+1) * Y_stride + out_idx] = s1 * x_scales[m+1];
        }

        /* Single remainder */
        for (; m < M; m++) {
            qwen_q4k_matvec_preq_neon(
                Y + (size_t)m * Y_stride + out_idx,
                row_blk,
                x_int8 + (size_t)m * K,
                x_scales[m],
                bsums + (size_t)m * total_subs,
                1, K);
        }
    }
#else
    /* Non-SDOT fallback: use preq kernel per token */
    for (int m = 0; m < M; m++) {
        qwen_q4k_matvec_preq_neon(
            Y + (size_t)m * Y_stride + r_start,
            W_chunk,
            x_int8 + (size_t)m * K,
            x_scales[m],
            bsums + (size_t)m * total_subs,
            n_rows, K);
    }
#endif
}

#endif /* __ARM_NEON */
