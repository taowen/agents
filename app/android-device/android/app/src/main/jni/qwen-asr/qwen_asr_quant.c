/*
 * qwen_asr_quant.c - Q8_0 quantization implementation
 */

#include "qwen_asr_quant.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

void quantize_f32_to_q8_0(const float *src, block_q8_0 *dst, int n) {
    int n_blocks = n / QK8_0;

    for (int b = 0; b < n_blocks; b++) {
        const float *sp = src + b * QK8_0;
        block_q8_0 *dp = &dst[b];

        /* Find absolute maximum */
        float amax = 0.0f;
#ifdef __ARM_NEON
        float32x4_t vmax = vdupq_n_f32(0.0f);
        for (int j = 0; j < QK8_0; j += 4) {
            float32x4_t v = vld1q_f32(sp + j);
            vmax = vmaxq_f32(vmax, vabsq_f32(v));
        }
        amax = vmaxvq_f32(vmax);
#else
        for (int j = 0; j < QK8_0; j++) {
            float av = fabsf(sp[j]);
            if (av > amax) amax = av;
        }
#endif

        float scale = amax / 127.0f;
        dp->scale = scale;

        if (scale == 0.0f) {
            memset(dp->qs, 0, QK8_0);
            continue;
        }

        float inv_scale = 127.0f / amax;
#ifdef __ARM_NEON
        float32x4_t vs = vdupq_n_f32(inv_scale);
        for (int j = 0; j < QK8_0; j += 8) {
            float32x4_t v0 = vmulq_f32(vld1q_f32(sp + j), vs);
            float32x4_t v1 = vmulq_f32(vld1q_f32(sp + j + 4), vs);
            int32x4_t i0 = vcvtnq_s32_f32(v0);
            int32x4_t i1 = vcvtnq_s32_f32(v1);
            int16x4_t s0 = vqmovn_s32(i0);
            int16x4_t s1 = vqmovn_s32(i1);
            int16x8_t s01 = vcombine_s16(s0, s1);
            int8x8_t q = vqmovn_s16(s01);
            vst1_s8(dp->qs + j, q);
        }
#else
        for (int j = 0; j < QK8_0; j++) {
            float v = sp[j] * inv_scale;
            int iv = (int)roundf(v);
            if (iv < -128) iv = -128;
            if (iv > 127) iv = 127;
            dp->qs[j] = (int8_t)iv;
        }
#endif
    }
}

void quantize_bf16_to_q8_0(const uint16_t *src, block_q8_0 *dst, int n) {
    int n_blocks = n / QK8_0;

    for (int b = 0; b < n_blocks; b++) {
        const uint16_t *sp = src + b * QK8_0;
        block_q8_0 *dp = &dst[b];

        /* Convert BF16 block to float and find absmax */
        float tmp[QK8_0];
        float amax = 0.0f;

#ifdef __ARM_NEON
        float32x4_t vmax = vdupq_n_f32(0.0f);
        for (int j = 0; j < QK8_0; j += 8) {
            uint16x8_t raw = vld1q_u16(sp + j);
            float32x4_t f0 = vreinterpretq_f32_u32(
                vshlq_n_u32(vmovl_u16(vget_low_u16(raw)), 16));
            float32x4_t f1 = vreinterpretq_f32_u32(
                vshlq_n_u32(vmovl_u16(vget_high_u16(raw)), 16));
            vst1q_f32(tmp + j, f0);
            vst1q_f32(tmp + j + 4, f1);
            vmax = vmaxq_f32(vmax, vabsq_f32(f0));
            vmax = vmaxq_f32(vmax, vabsq_f32(f1));
        }
        amax = vmaxvq_f32(vmax);
#else
        for (int j = 0; j < QK8_0; j++) {
            uint32_t bits = ((uint32_t)sp[j]) << 16;
            memcpy(&tmp[j], &bits, sizeof(float));
            float av = fabsf(tmp[j]);
            if (av > amax) amax = av;
        }
#endif

        float scale = amax / 127.0f;
        dp->scale = scale;

        if (scale == 0.0f) {
            memset(dp->qs, 0, QK8_0);
            continue;
        }

        float inv_scale = 127.0f / amax;
#ifdef __ARM_NEON
        float32x4_t vs = vdupq_n_f32(inv_scale);
        for (int j = 0; j < QK8_0; j += 8) {
            float32x4_t v0 = vmulq_f32(vld1q_f32(tmp + j), vs);
            float32x4_t v1 = vmulq_f32(vld1q_f32(tmp + j + 4), vs);
            int32x4_t i0 = vcvtnq_s32_f32(v0);
            int32x4_t i1 = vcvtnq_s32_f32(v1);
            int16x4_t s0 = vqmovn_s32(i0);
            int16x4_t s1 = vqmovn_s32(i1);
            int16x8_t s01 = vcombine_s16(s0, s1);
            int8x8_t q = vqmovn_s16(s01);
            vst1_s8(dp->qs + j, q);
        }
#else
        for (int j = 0; j < QK8_0; j++) {
            float v = tmp[j] * inv_scale;
            int iv = (int)roundf(v);
            if (iv < -128) iv = -128;
            if (iv > 127) iv = 127;
            dp->qs[j] = (int8_t)iv;
        }
#endif
    }
}

/* ========================================================================
 * Q4_K Super-Block Quantization
 *
 * Two-level quantization: super-block scale/min (float) + sub-group
 * integer scales/mins (uint8).
 * Per super-block (256 elements, 8 sub-groups of 32):
 *   weight ≈ d * scales[g] * q - dmin * mins[g]  where q ∈ [0, 15]
 * ======================================================================== */

void quantize_bf16_to_q4k(const uint16_t *bf16, int rows, int cols,
                            block_q4_k **out_blocks) {
    if (cols % QK_K != 0) {
        *out_blocks = NULL;
        return;
    }

    int blocks_per_row = cols / QK_K;
    size_t total_blocks = (size_t)rows * blocks_per_row;
    *out_blocks = (block_q4_k *)malloc(total_blocks * sizeof(block_q4_k));
    if (!*out_blocks) return;

    float tmp[QK_K];

    for (int r = 0; r < rows; r++) {
        const uint16_t *row = bf16 + (size_t)r * cols;

        for (int b = 0; b < blocks_per_row; b++) {
            block_q4_k *blk = *out_blocks + (size_t)r * blocks_per_row + b;
            int col_start = b * QK_K;

            /* Convert BF16 block to F32 */
            for (int i = 0; i < QK_K; i++) {
                uint32_t bits = ((uint32_t)row[col_start + i]) << 16;
                memcpy(&tmp[i], &bits, sizeof(float));
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

            /* Phase 3: Quantize weights → unsigned int4 [0, 15] and pack */
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

void quantize_f32_rows_transpose_q8(
    const float *X,
    block_q8_0 *X_q8t,
    int M, int K, int M_pad
) {
    int n_blocks = K / QK8_0;

    /* Zero-fill padding rows (m >= M) for all blocks */
    if (M_pad > M) {
        for (int b = 0; b < n_blocks; b++) {
            for (int m = M; m < M_pad; m++) {
                block_q8_0 *dp = &X_q8t[(size_t)b * M_pad + m];
                dp->scale = 0.0f;
                memset(dp->qs, 0, QK8_0);
            }
        }
    }

    /* Quantize each row m, block b */
    for (int m = 0; m < M; m++) {
        const float *row = X + (size_t)m * K;
        for (int b = 0; b < n_blocks; b++) {
            const float *sp = row + b * QK8_0;
            block_q8_0 *dp = &X_q8t[(size_t)b * M_pad + m];

            /* Find absolute maximum */
            float amax = 0.0f;
#ifdef __ARM_NEON
            float32x4_t vmax = vdupq_n_f32(0.0f);
            for (int j = 0; j < QK8_0; j += 4) {
                float32x4_t v = vld1q_f32(sp + j);
                vmax = vmaxq_f32(vmax, vabsq_f32(v));
            }
            amax = vmaxvq_f32(vmax);
#else
            for (int j = 0; j < QK8_0; j++) {
                float av = fabsf(sp[j]);
                if (av > amax) amax = av;
            }
#endif

            float scale = amax / 127.0f;
            dp->scale = scale;

            if (scale == 0.0f) {
                memset(dp->qs, 0, QK8_0);
                continue;
            }

            float inv_scale = 127.0f / amax;
#ifdef __ARM_NEON
            float32x4_t vs = vdupq_n_f32(inv_scale);
            for (int j = 0; j < QK8_0; j += 8) {
                float32x4_t v0 = vmulq_f32(vld1q_f32(sp + j), vs);
                float32x4_t v1 = vmulq_f32(vld1q_f32(sp + j + 4), vs);
                int32x4_t i0 = vcvtnq_s32_f32(v0);
                int32x4_t i1 = vcvtnq_s32_f32(v1);
                int16x4_t s0 = vqmovn_s32(i0);
                int16x4_t s1 = vqmovn_s32(i1);
                int16x8_t s01 = vcombine_s16(s0, s1);
                int8x8_t q = vqmovn_s16(s01);
                vst1_s8(dp->qs + j, q);
            }
#else
            for (int j = 0; j < QK8_0; j++) {
                float v = sp[j] * inv_scale;
                int iv = (int)roundf(v);
                if (iv < -128) iv = -128;
                if (iv > 127) iv = 127;
                dp->qs[j] = (int8_t)iv;
            }
#endif
        }
    }
}

void dequantize_q8_0_to_f32(const block_q8_0 *src, float *dst, int n) {
    int n_blocks = n / QK8_0;

    for (int b = 0; b < n_blocks; b++) {
        const block_q8_0 *sp = &src[b];
        float *dp = dst + b * QK8_0;
        float scale = sp->scale;

#ifdef __ARM_NEON
        float32x4_t vs = vdupq_n_f32(scale);
        for (int j = 0; j < QK8_0; j += 8) {
            int8x8_t qi = vld1_s8(sp->qs + j);
            int16x8_t qi16 = vmovl_s8(qi);
            float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qi16))), vs);
            float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qi16))), vs);
            vst1q_f32(dp + j, f0);
            vst1q_f32(dp + j + 4, f1);
        }
#else
        for (int j = 0; j < QK8_0; j++) {
            dp[j] = scale * (float)sp->qs[j];
        }
#endif
    }
}
