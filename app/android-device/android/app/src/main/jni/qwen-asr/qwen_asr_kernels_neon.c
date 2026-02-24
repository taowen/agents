/*
 * qwen_asr_kernels_neon.c - ARM NEON hot kernels
 */

#include "qwen_asr_kernels_impl.h"

#ifdef __ARM_NEON

#include <arm_neon.h>
#include <string.h>

void qwen_bf16_matvec_fused_neon(float *y, const float *x, const uint16_t *W_bf16,
                                 const float *bias, int in_dim, int out_dim) {
    int o = 0;

    /* Process 2 output rows at a time, 32 elements/iter, 8 accumulators */
    for (; o + 1 < out_dim; o += 2) {
        const uint16_t *w0 = W_bf16 + (size_t)o * in_dim;
        const uint16_t *w1 = W_bf16 + (size_t)(o + 1) * in_dim;
        float s0 = bias ? bias[o] : 0.0f;
        float s1 = bias ? bias[o + 1] : 0.0f;

        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
        float32x4_t b0 = vdupq_n_f32(0.0f), b1 = vdupq_n_f32(0.0f);
        float32x4_t b2 = vdupq_n_f32(0.0f), b3 = vdupq_n_f32(0.0f);
        int k = 0;

        for (; k + 32 <= in_dim; k += 32) {
            float32x4_t x0 = vld1q_f32(x + k);
            float32x4_t x1 = vld1q_f32(x + k + 4);
            float32x4_t x2 = vld1q_f32(x + k + 8);
            float32x4_t x3 = vld1q_f32(x + k + 12);
            float32x4_t x4 = vld1q_f32(x + k + 16);
            float32x4_t x5 = vld1q_f32(x + k + 20);
            float32x4_t x6 = vld1q_f32(x + k + 24);
            float32x4_t x7 = vld1q_f32(x + k + 28);

            uint16x8_t r0a = vld1q_u16(w0 + k);
            uint16x8_t r0b = vld1q_u16(w0 + k + 8);
            uint16x8_t r0c = vld1q_u16(w0 + k + 16);
            uint16x8_t r0d = vld1q_u16(w0 + k + 24);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0a), 16)), x0);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0a), 16)), x1);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0b), 16)), x2);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0b), 16)), x3);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0c), 16)), x4);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0c), 16)), x5);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0d), 16)), x6);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0d), 16)), x7);

            uint16x8_t r1a = vld1q_u16(w1 + k);
            uint16x8_t r1b = vld1q_u16(w1 + k + 8);
            uint16x8_t r1c = vld1q_u16(w1 + k + 16);
            uint16x8_t r1d = vld1q_u16(w1 + k + 24);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1a), 16)), x0);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1a), 16)), x1);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1b), 16)), x2);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1b), 16)), x3);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1c), 16)), x4);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1c), 16)), x5);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1d), 16)), x6);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1d), 16)), x7);
        }
        for (; k + 8 <= in_dim; k += 8) {
            float32x4_t x0 = vld1q_f32(x + k);
            float32x4_t x1 = vld1q_f32(x + k + 4);
            uint16x8_t r0 = vld1q_u16(w0 + k);
            uint16x8_t r1 = vld1q_u16(w1 + k);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0), 16)), x0);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0), 16)), x1);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1), 16)), x0);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1), 16)), x1);
        }
        s0 += vaddvq_f32(vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3)));
        s1 += vaddvq_f32(vaddq_f32(vaddq_f32(b0, b2), vaddq_f32(b1, b3)));

        for (; k < in_dim; k++) {
            uint32_t bits0 = ((uint32_t)w0[k]) << 16;
            uint32_t bits1 = ((uint32_t)w1[k]) << 16;
            float wv0, wv1;
            memcpy(&wv0, &bits0, sizeof(float));
            memcpy(&wv1, &bits1, sizeof(float));
            s0 += wv0 * x[k];
            s1 += wv1 * x[k];
        }
        y[o] = s0;
        y[o + 1] = s1;
    }

    /* Handle remaining odd row */
    for (; o < out_dim; o++) {
        const uint16_t *w_row = W_bf16 + (size_t)o * in_dim;
        float sum = bias ? bias[o] : 0.0f;
        int k = 0;

        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        for (; k + 8 <= in_dim; k += 8) {
            uint16x8_t bf = vld1q_u16(w_row + k);
            acc0 = vfmaq_f32(acc0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16)),
                             vld1q_f32(x + k));
            acc1 = vfmaq_f32(acc1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16)),
                             vld1q_f32(x + k + 4));
        }
        sum += vaddvq_f32(vaddq_f32(acc0, acc1));

        for (; k < in_dim; k++) {
            uint32_t f32_bits = ((uint32_t)w_row[k]) << 16;
            float w_val;
            memcpy(&w_val, &f32_bits, sizeof(float));
            sum += w_val * x[k];
        }
        y[o] = sum;
    }
}

void qwen_argmax_bf16_range_neon(const float *x, const uint16_t *W_bf16,
                                 int in_dim, int start, int end,
                                 int *best_out, float *best_val_out) {
    int best = start;
    float best_val = -1e30f;
    int o = start;

    /* Process 2 rows at a time, 32 elements/iter, 8 accumulators per row */
    for (; o + 1 < end; o += 2) {
        const uint16_t *w0 = W_bf16 + (size_t)o * in_dim;
        const uint16_t *w1 = W_bf16 + (size_t)(o + 1) * in_dim;
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
        float32x4_t b0 = vdupq_n_f32(0.0f), b1 = vdupq_n_f32(0.0f);
        float32x4_t b2 = vdupq_n_f32(0.0f), b3 = vdupq_n_f32(0.0f);
        int k = 0;

        for (; k + 32 <= in_dim; k += 32) {
            float32x4_t x0 = vld1q_f32(x + k);
            float32x4_t x1 = vld1q_f32(x + k + 4);
            float32x4_t x2 = vld1q_f32(x + k + 8);
            float32x4_t x3 = vld1q_f32(x + k + 12);
            float32x4_t x4 = vld1q_f32(x + k + 16);
            float32x4_t x5 = vld1q_f32(x + k + 20);
            float32x4_t x6 = vld1q_f32(x + k + 24);
            float32x4_t x7 = vld1q_f32(x + k + 28);

            uint16x8_t r0a = vld1q_u16(w0 + k);
            uint16x8_t r0b = vld1q_u16(w0 + k + 8);
            uint16x8_t r0c = vld1q_u16(w0 + k + 16);
            uint16x8_t r0d = vld1q_u16(w0 + k + 24);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0a), 16)), x0);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0a), 16)), x1);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0b), 16)), x2);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0b), 16)), x3);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0c), 16)), x4);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0c), 16)), x5);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0d), 16)), x6);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0d), 16)), x7);

            uint16x8_t r1a = vld1q_u16(w1 + k);
            uint16x8_t r1b = vld1q_u16(w1 + k + 8);
            uint16x8_t r1c = vld1q_u16(w1 + k + 16);
            uint16x8_t r1d = vld1q_u16(w1 + k + 24);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1a), 16)), x0);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1a), 16)), x1);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1b), 16)), x2);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1b), 16)), x3);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1c), 16)), x4);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1c), 16)), x5);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1d), 16)), x6);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1d), 16)), x7);
        }

        float s0 = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3)));
        float s1 = vaddvq_f32(vaddq_f32(vaddq_f32(b0, b2), vaddq_f32(b1, b3)));

        for (; k < in_dim; k++) {
            uint32_t bits0 = ((uint32_t)w0[k]) << 16;
            uint32_t bits1 = ((uint32_t)w1[k]) << 16;
            float wv0, wv1;
            memcpy(&wv0, &bits0, sizeof(float));
            memcpy(&wv1, &bits1, sizeof(float));
            s0 += wv0 * x[k];
            s1 += wv1 * x[k];
        }

        if (s0 > best_val) { best_val = s0; best = o; }
        if (s1 > best_val) { best_val = s1; best = o + 1; }
    }

    for (; o < end; o++) {
        const uint16_t *w_row = W_bf16 + (size_t)o * in_dim;
        float sum = 0.0f;
        int k = 0;

        float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
        for (; k + 8 <= in_dim; k += 8) {
            uint16x8_t bf = vld1q_u16(w_row + k);
            acc0 = vfmaq_f32(acc0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16)),
                             vld1q_f32(x + k));
            acc1 = vfmaq_f32(acc1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16)),
                             vld1q_f32(x + k + 4));
        }
        sum += vaddvq_f32(vaddq_f32(acc0, acc1));

        for (; k < in_dim; k++) {
            uint32_t f32_bits = ((uint32_t)w_row[k]) << 16;
            float w_val;
            memcpy(&w_val, &f32_bits, sizeof(float));
            sum += w_val * x[k];
        }
        if (sum > best_val) { best_val = sum; best = o; }
    }

    *best_out = best;
    *best_val_out = best_val;
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

void qwen_argmax_q8_range_neon(const block_q8_0 *x_q8,
                                const block_q8_0 *W_q8,
                                int n_blocks, int start, int end,
                                int *best_out, float *best_val_out) {
    int best = start;
    float best_val = -1e30f;
    int o = start;

    /* Process 2 rows at a time */
    for (; o + 1 < end; o += 2) {
        const block_q8_0 *w0 = W_q8 + (size_t)o * n_blocks;
        const block_q8_0 *w1 = W_q8 + (size_t)(o + 1) * n_blocks;
        float s0 = 0.0f, s1 = 0.0f;

        int b = 0;
        for (; b + 1 < n_blocks; b += 2) {
            /* Block b */
            float xs0 = x_q8[b].scale;
            int8x16_t xv0_lo = vld1q_s8(x_q8[b].qs);
            int8x16_t xv0_hi = vld1q_s8(x_q8[b].qs + 16);

#ifdef __ARM_FEATURE_DOTPROD
            int32x4_t d0_0 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w0[b].qs), xv0_lo);
            d0_0 = vdotq_s32(d0_0, vld1q_s8(w0[b].qs + 16), xv0_hi);
            int32x4_t d1_0 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w1[b].qs), xv0_lo);
            d1_0 = vdotq_s32(d1_0, vld1q_s8(w1[b].qs + 16), xv0_hi);
#else
            int32x4_t d0_0 = vdupq_n_s32(0), d1_0 = vdupq_n_s32(0);
            for (int j = 0; j < 32; j += 8) {
                int8x8_t xq = vld1_s8(x_q8[b].qs + j);
                int16x8_t xq16 = vmovl_s8(xq);
                int16x8_t wq0_16 = vmovl_s8(vld1_s8(w0[b].qs + j));
                int16x8_t wq1_16 = vmovl_s8(vld1_s8(w1[b].qs + j));
                d0_0 = vmlal_s16(d0_0, vget_low_s16(wq0_16), vget_low_s16(xq16));
                d0_0 = vmlal_s16(d0_0, vget_high_s16(wq0_16), vget_high_s16(xq16));
                d1_0 = vmlal_s16(d1_0, vget_low_s16(wq1_16), vget_low_s16(xq16));
                d1_0 = vmlal_s16(d1_0, vget_high_s16(wq1_16), vget_high_s16(xq16));
            }
#endif
            s0 += w0[b].scale * xs0 * (float)vaddvq_s32(d0_0);
            s1 += w1[b].scale * xs0 * (float)vaddvq_s32(d1_0);

            /* Block b+1 */
            float xs1 = x_q8[b + 1].scale;
            int8x16_t xv1_lo = vld1q_s8(x_q8[b + 1].qs);
            int8x16_t xv1_hi = vld1q_s8(x_q8[b + 1].qs + 16);

#ifdef __ARM_FEATURE_DOTPROD
            int32x4_t d0_1 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w0[b + 1].qs), xv1_lo);
            d0_1 = vdotq_s32(d0_1, vld1q_s8(w0[b + 1].qs + 16), xv1_hi);
            int32x4_t d1_1 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w1[b + 1].qs), xv1_lo);
            d1_1 = vdotq_s32(d1_1, vld1q_s8(w1[b + 1].qs + 16), xv1_hi);
#else
            int32x4_t d0_1 = vdupq_n_s32(0), d1_1 = vdupq_n_s32(0);
            for (int j = 0; j < 32; j += 8) {
                int8x8_t xq = vld1_s8(x_q8[b + 1].qs + j);
                int16x8_t xq16 = vmovl_s8(xq);
                int16x8_t wq0_16 = vmovl_s8(vld1_s8(w0[b + 1].qs + j));
                int16x8_t wq1_16 = vmovl_s8(vld1_s8(w1[b + 1].qs + j));
                d0_1 = vmlal_s16(d0_1, vget_low_s16(wq0_16), vget_low_s16(xq16));
                d0_1 = vmlal_s16(d0_1, vget_high_s16(wq0_16), vget_high_s16(xq16));
                d1_1 = vmlal_s16(d1_1, vget_low_s16(wq1_16), vget_low_s16(xq16));
                d1_1 = vmlal_s16(d1_1, vget_high_s16(wq1_16), vget_high_s16(xq16));
            }
#endif
            s0 += w0[b + 1].scale * xs1 * (float)vaddvq_s32(d0_1);
            s1 += w1[b + 1].scale * xs1 * (float)vaddvq_s32(d1_1);
        }

        /* Handle remaining block */
        for (; b < n_blocks; b++) {
            float xs = x_q8[b].scale;
            int8x16_t xv_lo = vld1q_s8(x_q8[b].qs);
            int8x16_t xv_hi = vld1q_s8(x_q8[b].qs + 16);

#ifdef __ARM_FEATURE_DOTPROD
            int32x4_t d0 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w0[b].qs), xv_lo);
            d0 = vdotq_s32(d0, vld1q_s8(w0[b].qs + 16), xv_hi);
            int32x4_t d1 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w1[b].qs), xv_lo);
            d1 = vdotq_s32(d1, vld1q_s8(w1[b].qs + 16), xv_hi);
#else
            int32x4_t d0 = vdupq_n_s32(0), d1 = vdupq_n_s32(0);
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
            s0 += w0[b].scale * xs * (float)vaddvq_s32(d0);
            s1 += w1[b].scale * xs * (float)vaddvq_s32(d1);
        }

        if (s0 > best_val) { best_val = s0; best = o; }
        if (s1 > best_val) { best_val = s1; best = o + 1; }
    }

    /* Handle remaining odd row */
    for (; o < end; o++) {
        const block_q8_0 *w0 = W_q8 + (size_t)o * n_blocks;
        float sum = 0.0f;

        for (int b = 0; b < n_blocks; b++) {
            float xs = x_q8[b].scale;
            int8x16_t xv_lo = vld1q_s8(x_q8[b].qs);
            int8x16_t xv_hi = vld1q_s8(x_q8[b].qs + 16);

#ifdef __ARM_FEATURE_DOTPROD
            int32x4_t d = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w0[b].qs), xv_lo);
            d = vdotq_s32(d, vld1q_s8(w0[b].qs + 16), xv_hi);
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
            sum += w0[b].scale * xs * (float)vaddvq_s32(d);
        }
        if (sum > best_val) { best_val = sum; best = o; }
    }

    *best_out = best;
    *best_val_out = best_val;
}

/* ========================================================================
 * Q4_K × Q8_K dot product (NEON)
 * Ported from llama.cpp ggml_vec_dot_q4_K_q8_K (__ARM_NEON block)
 * ======================================================================== */

/* Helper: load 2×16 uint8 values */
#ifndef ggml_vld1q_u8_x2
static inline uint8x16x2_t ggml_vld1q_u8_x2(const uint8_t *p) {
    uint8x16x2_t r;
    r.val[0] = vld1q_u8(p);
    r.val[1] = vld1q_u8(p + 16);
    return r;
}
#endif

#ifndef ggml_vld1q_s8_x2
static inline int8x16x2_t ggml_vld1q_s8_x2(const int8_t *p) {
    int8x16x2_t r;
    r.val[0] = vld1q_s8(p);
    r.val[1] = vld1q_s8(p + 16);
    return r;
}
#endif

static float q4k_q8k_dot_neon(const block_q4_K *x, const block_q8_K *y, int n_blocks) {
    const uint8x16_t m4b = vdupq_n_u8(0xf);
    float sumf = 0.0f;

    for (int i = 0; i < n_blocks; i++) {
        const uint8_t *q4 = x[i].qs;
        const int8_t  *q8 = y[i].qs;
        const uint8_t *scales = x[i].scales;

        const float d = fp16_to_fp32(x[i].d);
        const float m = fp16_to_fp32(x[i].dmin);
        const float d8 = y[i].d;

        int32x4_t sumv0 = vdupq_n_s32(0);
        int32x4_t sumv1 = vdupq_n_s32(0);
        float sum_ms = 0.0f;

        for (int j = 0; j < QK_K / 64; j++) {
            const uint8x16x2_t q4bits = ggml_vld1q_u8_x2(q4); q4 += 32;
            const int8x16x2_t q8bytes = ggml_vld1q_s8_x2(q8); q8 += 32;

            const uint8x16_t q4_0 = vandq_u8(q4bits.val[0], m4b);
            const uint8x16_t q4_1 = vandq_u8(q4bits.val[1], m4b);
            const uint8x16_t q4_2 = vshrq_n_u8(q4bits.val[0], 4);
            const uint8x16_t q4_3 = vshrq_n_u8(q4bits.val[1], 4);

            const int8x16_t q4s0 = vreinterpretq_s8_u8(q4_0);
            const int8x16_t q4s1 = vreinterpretq_s8_u8(q4_1);
            const int8x16_t q4s2 = vreinterpretq_s8_u8(q4_2);
            const int8x16_t q4s3 = vreinterpretq_s8_u8(q4_3);

#if defined(__ARM_FEATURE_DOTPROD)
            sumv0 = vdotq_s32(sumv0, q4s0, q8bytes.val[0]);
            sumv1 = vdotq_s32(sumv1, q4s1, q8bytes.val[1]);
            /* Second half of 64 elements (high nibbles) */
            {
                const int8x16x2_t q8b2 = ggml_vld1q_s8_x2(q8); q8 += 32;
                sumv0 = vdotq_s32(sumv0, q4s2, q8b2.val[0]);
                sumv1 = vdotq_s32(sumv1, q4s3, q8b2.val[1]);
            }
#else
            {
                const int16x8_t p0 = vmull_s8(vget_low_s8(q4s0), vget_low_s8(q8bytes.val[0]));
                const int16x8_t p1 = vmull_high_s8(q4s0, q8bytes.val[0]);
                sumv0 = vpadalq_s16(sumv0, p0);
                sumv0 = vpadalq_s16(sumv0, p1);
                const int16x8_t p2 = vmull_s8(vget_low_s8(q4s1), vget_low_s8(q8bytes.val[1]));
                const int16x8_t p3 = vmull_high_s8(q4s1, q8bytes.val[1]);
                sumv1 = vpadalq_s16(sumv1, p2);
                sumv1 = vpadalq_s16(sumv1, p3);
            }
            {
                const int8x16x2_t q8b2 = ggml_vld1q_s8_x2(q8); q8 += 32;
                const int16x8_t p0 = vmull_s8(vget_low_s8(q4s2), vget_low_s8(q8b2.val[0]));
                const int16x8_t p1 = vmull_high_s8(q4s2, q8b2.val[0]);
                sumv0 = vpadalq_s16(sumv0, p0);
                sumv0 = vpadalq_s16(sumv0, p1);
                const int16x8_t p2 = vmull_s8(vget_low_s8(q4s3), vget_low_s8(q8b2.val[1]));
                const int16x8_t p3 = vmull_high_s8(q4s3, q8b2.val[1]);
                sumv1 = vpadalq_s16(sumv1, p2);
                sumv1 = vpadalq_s16(sumv1, p3);
            }
#endif

            sum_ms += (float)(scales[2 * j + 0] & 0x3f) * (float)y[i].bsums[2 * j + 0]
                    + (float)(scales[2 * j + 1] & 0x3f) * (float)y[i].bsums[2 * j + 1]
                    + (float)(scales[2 * j + 2] & 0x3f) * (float)y[i].bsums[2 * j + 2]
                    + (float)(scales[2 * j + 3] & 0x3f) * (float)y[i].bsums[2 * j + 3];
        }

        sumf += d * d8 * (float)(vaddvq_s32(sumv0) + vaddvq_s32(sumv1)) - m * d8 * sum_ms;
    }
    return sumf;
}

void qwen_q4k_q8k_matvec_neon(float *y, const block_q8_K *x_q8k,
                                const block_q4_K *W_q4k, const float *bias,
                                int n_blocks_k, int out_dim) {
    int o = 0;

    /* Process 2 output rows at a time */
    for (; o + 1 < out_dim; o += 2) {
        const block_q4_K *w0 = W_q4k + (size_t)o * n_blocks_k;
        const block_q4_K *w1 = W_q4k + (size_t)(o + 1) * n_blocks_k;
        float s0 = bias ? bias[o] : 0.0f;
        float s1 = bias ? bias[o + 1] : 0.0f;
        s0 += q4k_q8k_dot_neon(w0, x_q8k, n_blocks_k);
        s1 += q4k_q8k_dot_neon(w1, x_q8k, n_blocks_k);
        y[o] = s0;
        y[o + 1] = s1;
    }

    /* Handle remaining odd row */
    for (; o < out_dim; o++) {
        const block_q4_K *w0 = W_q4k + (size_t)o * n_blocks_k;
        float sum = bias ? bias[o] : 0.0f;
        sum += q4k_q8k_dot_neon(w0, x_q8k, n_blocks_k);
        y[o] = sum;
    }
}

void qwen_argmax_q4k_range_neon(const block_q8_K *x_q8k,
                                 const block_q4_K *W_q4k,
                                 int n_blocks_k, int start, int end,
                                 int *best_out, float *best_val_out) {
    int best = start;
    float best_val = -1e30f;

    for (int o = start; o < end; o++) {
        const block_q4_K *w0 = W_q4k + (size_t)o * n_blocks_k;
        float sum = q4k_q8k_dot_neon(w0, x_q8k, n_blocks_k);
        if (sum > best_val) {
            best_val = sum;
            best = o;
        }
    }

    *best_out = best;
    *best_val_out = best_val;
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

#endif /* __ARM_NEON */
