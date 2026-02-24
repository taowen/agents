/*
 * qwen_tts_kernels.c - Core math kernel implementations for Qwen3-TTS
 *
 * Contains normalization, activation, and element-wise operations.
 * NEON-intensive matvec/matmul ops are in qwen_tts_kernels_neon.c.
 * Conv, RoPE, sampling ops are in qwen_tts_kernels_ops.c.
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
/* Activation functions                                                      */
/* ======================================================================== */

void kernel_silu_inplace(float *x, int n) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    /* NEON fast sigmoid approximation: sigma(x) ~ clamp(x * 0.25 + 0.5, 0, 1)
     * SiLU(x) = x * sigma(x) ~ x * clamp(x * 0.25 + 0.5, 0, 1)
     * This 3rd-order rational approximation is more accurate:
     *   sigma(x) ~ 0.5 + 0.5 * x / (1 + |x|)  (max err ~0.05 at |x|=2)
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
/* Platform dispatch (no-op for now)                                         */
/* ======================================================================== */

void kernel_init(void) {
#ifdef USE_OPENMP
    omp_set_num_threads(4);
#endif
}
