/*
 * qwen_asr_quant.c - Q8_0 quantization implementation
 */

#include "qwen_asr_quant.h"
#include <math.h>
#include <string.h>

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

/* ========================================================================
 * Q8_K Quantization (256-element super-blocks for runtime activation quant)
 * Reference: llama.cpp quantize_row_q8_K_ref
 * ======================================================================== */

void quantize_f32_to_q8_K(const float *src, block_q8_K *dst, int n) {
    int nb = n / QK_K;

    for (int i = 0; i < nb; i++) {
        const float *sp = src + (size_t)i * QK_K;
        block_q8_K *dp = &dst[i];

        /* Find absolute maximum */
        float max = 0.0f;
#ifdef __ARM_NEON
        float32x4_t vmax = vdupq_n_f32(0.0f);
        for (int j = 0; j < QK_K; j += 4) {
            float32x4_t v = vld1q_f32(sp + j);
            vmax = vmaxq_f32(vmax, vabsq_f32(v));
        }
        max = vmaxvq_f32(vmax);
#else
        for (int j = 0; j < QK_K; j++) {
            float amax = fabsf(sp[j]);
            if (amax > max) max = amax;
        }
#endif

        if (max == 0.0f) {
            dp->d = 0.0f;
            memset(dp->qs, 0, QK_K);
            memset(dp->bsums, 0, sizeof(dp->bsums));
            continue;
        }

        dp->d = max / 127.0f;
        float id = 1.0f / dp->d;

        for (int j = 0; j < QK_K / 16; j++) {
            int sum = 0;
            for (int l = 0; l < 16; l++) {
                float v = sp[16 * j + l] * id;
                int iv = (int)roundf(v);
                if (iv < -128) iv = -128;
                if (iv > 127) iv = 127;
                dp->qs[16 * j + l] = (int8_t)iv;
                sum += dp->qs[16 * j + l];
            }
            dp->bsums[j] = (int16_t)sum;
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
