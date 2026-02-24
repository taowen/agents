/*
 * qwen_asr_kernels_ops.c - High-level math operations for Qwen3-ASR inference
 * Normalization, activations, attention, position embeddings, FP16 conversion.
 */

#include "qwen_asr_kernels.h"
#include "qwen_asr_kernels_impl.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================
 * Normalization
 * ======================================================================== */

void qwen_layer_norm(float *out, const float *x, const float *weight, const float *bias,
                     int seq_len, int hidden, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

        /* Compute mean */
#ifdef __ARM_NEON
        float32x4_t sumv0 = vdupq_n_f32(0.0f);
        float32x4_t sumv1 = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 8 <= hidden; i += 8) {
            sumv0 = vaddq_f32(sumv0, vld1q_f32(x_row + i));
            sumv1 = vaddq_f32(sumv1, vld1q_f32(x_row + i + 4));
        }
        float mean = vaddvq_f32(vaddq_f32(sumv0, sumv1));
        for (; i < hidden; i++) mean += x_row[i];
#else
        float mean = 0.0f;
        for (int i = 0; i < hidden; i++) mean += x_row[i];
#endif
        mean /= hidden;

        /* Compute variance */
#ifdef __ARM_NEON
        float32x4_t meanv = vdupq_n_f32(mean);
        float32x4_t accv0 = vdupq_n_f32(0.0f);
        float32x4_t accv1 = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j + 8 <= hidden; j += 8) {
            float32x4_t d0 = vsubq_f32(vld1q_f32(x_row + j), meanv);
            float32x4_t d1 = vsubq_f32(vld1q_f32(x_row + j + 4), meanv);
            accv0 = vfmaq_f32(accv0, d0, d0);
            accv1 = vfmaq_f32(accv1, d1, d1);
        }
        float var = vaddvq_f32(vaddq_f32(accv0, accv1));
        for (; j < hidden; j++) {
            float d = x_row[j] - mean;
            var += d * d;
        }
#else
        float var = 0.0f;
        for (int i = 0; i < hidden; i++) {
            float d = x_row[i] - mean;
            var += d * d;
        }
#endif
        var /= hidden;

        float inv_std = 1.0f / sqrtf(var + eps);
#ifdef __ARM_NEON
        float32x4_t meanv2 = vdupq_n_f32(mean);
        float32x4_t invv = vdupq_n_f32(inv_std);
        int k = 0;
        for (; k + 8 <= hidden; k += 8) {
            float32x4_t vx0 = vsubq_f32(vld1q_f32(x_row + k), meanv2);
            float32x4_t vw0 = vld1q_f32(weight + k);
            float32x4_t vb0 = vld1q_f32(bias + k);
            float32x4_t vx1 = vsubq_f32(vld1q_f32(x_row + k + 4), meanv2);
            float32x4_t vw1 = vld1q_f32(weight + k + 4);
            float32x4_t vb1 = vld1q_f32(bias + k + 4);
            vst1q_f32(out_row + k, vaddq_f32(vmulq_f32(vmulq_f32(vx0, invv), vw0), vb0));
            vst1q_f32(out_row + k + 4, vaddq_f32(vmulq_f32(vmulq_f32(vx1, invv), vw1), vb1));
        }
        for (; k < hidden; k++) {
            out_row[k] = (x_row[k] - mean) * inv_std * weight[k] + bias[k];
        }
#else
        for (int i = 0; i < hidden; i++) {
            out_row[i] = (x_row[i] - mean) * inv_std * weight[i] + bias[i];
        }
#endif
    }
}

void qwen_rms_norm(float *out, const float *x, const float *weight,
                   int seq_len, int hidden, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

#ifdef __ARM_NEON
        float32x4_t accv0 = vdupq_n_f32(0.0f);
        float32x4_t accv1 = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 8 <= hidden; i += 8) {
            float32x4_t v0 = vld1q_f32(x_row + i);
            float32x4_t v1 = vld1q_f32(x_row + i + 4);
            accv0 = vfmaq_f32(accv0, v0, v0);
            accv1 = vfmaq_f32(accv1, v1, v1);
        }
        float sum_sq = vaddvq_f32(vaddq_f32(accv0, accv1));
        for (; i < hidden; i++) sum_sq += x_row[i] * x_row[i];
#else
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden; i++) {
            sum_sq += x_row[i] * x_row[i];
        }
#endif
        float rms_inv = 1.0f / sqrtf(sum_sq / hidden + eps);

#ifdef __ARM_NEON
        float32x4_t scalev = vdupq_n_f32(rms_inv);
        int j = 0;
        for (; j + 8 <= hidden; j += 8) {
            float32x4_t vx0 = vld1q_f32(x_row + j);
            float32x4_t vw0 = vld1q_f32(weight + j);
            float32x4_t vx1 = vld1q_f32(x_row + j + 4);
            float32x4_t vw1 = vld1q_f32(weight + j + 4);
            vst1q_f32(out_row + j, vmulq_f32(vmulq_f32(vx0, vw0), scalev));
            vst1q_f32(out_row + j + 4, vmulq_f32(vmulq_f32(vx1, vw1), scalev));
        }
        for (; j < hidden; j++) out_row[j] = x_row[j] * rms_inv * weight[j];
#else
        for (int i = 0; i < hidden; i++) {
            out_row[i] = x_row[i] * rms_inv * weight[i];
        }
#endif
    }
}

void qwen_rms_norm_per_head(float *x, const float *weight,
                             int seq_len, int n_heads, int head_dim, float eps) {
    /* x is [seq, n_heads * head_dim] - normalize each [head_dim] segment */
    int hidden = n_heads * head_dim;
    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < n_heads; h++) {
            float *vec = x + s * hidden + h * head_dim;

#ifdef __ARM_NEON
            float32x4_t accv0 = vdupq_n_f32(0.0f);
            float32x4_t accv1 = vdupq_n_f32(0.0f);
            int d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                float32x4_t v0 = vld1q_f32(vec + d);
                float32x4_t v1 = vld1q_f32(vec + d + 4);
                accv0 = vfmaq_f32(accv0, v0, v0);
                accv1 = vfmaq_f32(accv1, v1, v1);
            }
            float sum_sq = vaddvq_f32(vaddq_f32(accv0, accv1));
            for (; d < head_dim; d++) sum_sq += vec[d] * vec[d];
#else
            float sum_sq = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum_sq += vec[d] * vec[d];
            }
#endif
            float rms_inv = 1.0f / sqrtf(sum_sq / head_dim + eps);

#ifdef __ARM_NEON
            float32x4_t scalev = vdupq_n_f32(rms_inv);
            int j = 0;
            for (; j + 8 <= head_dim; j += 8) {
                float32x4_t v0 = vld1q_f32(vec + j);
                float32x4_t w0 = vld1q_f32(weight + j);
                float32x4_t v1 = vld1q_f32(vec + j + 4);
                float32x4_t w1 = vld1q_f32(weight + j + 4);
                vst1q_f32(vec + j, vmulq_f32(vmulq_f32(v0, w0), scalev));
                vst1q_f32(vec + j + 4, vmulq_f32(vmulq_f32(v1, w1), scalev));
            }
            for (; j < head_dim; j++) vec[j] = vec[j] * rms_inv * weight[j];
#else
            for (int d = 0; d < head_dim; d++) {
                vec[d] = vec[d] * rms_inv * weight[d];
            }
#endif
        }
    }
}

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

#ifdef __ARM_NEON
/* NEON fast expf: 7th-order polynomial, max error ~1e-5 */
static inline float32x4_t neon_expf(float32x4_t x) {
    /* Clamp to [-88, 88] to avoid overflow */
    x = vmaxq_f32(x, vdupq_n_f32(-88.0f));
    x = vminq_f32(x, vdupq_n_f32(88.0f));
    /* exp(x) = 2^(x / ln2) = 2^(n + f) where n = round(x/ln2), f = fractional */
    float32x4_t log2e = vdupq_n_f32(1.44269504089f);
    float32x4_t t = vmulq_f32(x, log2e);
    float32x4_t n = vrndnq_f32(t);  /* round to nearest int */
    float32x4_t f = vsubq_f32(t, n);  /* fractional part in [-0.5, 0.5] */
    /* 2^f ≈ polynomial in f (minimax on [-0.5, 0.5]) */
    float32x4_t p = vdupq_n_f32(1.535336188e-4f);
    p = vfmaq_f32(vdupq_n_f32(1.339887440e-3f), p, f);
    p = vfmaq_f32(vdupq_n_f32(9.618437357e-3f), p, f);
    p = vfmaq_f32(vdupq_n_f32(5.550332471e-2f), p, f);
    p = vfmaq_f32(vdupq_n_f32(2.402264791e-1f), p, f);
    p = vfmaq_f32(vdupq_n_f32(6.931472028e-1f), p, f);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, f);
    /* Scale by 2^n: reinterpret n as exponent bits */
    int32x4_t ni = vcvtq_s32_f32(n);
    int32x4_t exp_bits = vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(exp_bits));
}

static inline float32x4_t neon_tanhf(float32x4_t x) {
    /* tanh(x) = 1 - 2 / (1 + exp(2x)) */
    float32x4_t two_x = vaddq_f32(x, x);
    float32x4_t e2x = neon_expf(two_x);
    float32x4_t one = vdupq_n_f32(1.0f);
    return vsubq_f32(one, vdivq_f32(vdupq_n_f32(2.0f), vaddq_f32(one, e2x)));
}
#endif /* __ARM_NEON */

void qwen_gelu(float *x, int n) {
#ifdef __ARM_NEON
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t coeff = vdupq_n_f32(0.7978845608028654f);
    const float32x4_t c3 = vdupq_n_f32(0.044715f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    int i = 0;
    /* Process 8 floats at a time (2 NEON vectors) to reduce loop overhead */
    for (; i + 7 < n; i += 8) {
        float32x4_t v0 = vld1q_f32(x + i);
        float32x4_t v1 = vld1q_f32(x + i + 4);
        float32x4_t v3_0 = vmulq_f32(vmulq_f32(v0, v0), v0);
        float32x4_t v3_1 = vmulq_f32(vmulq_f32(v1, v1), v1);
        float32x4_t inner0 = vmulq_f32(coeff, vfmaq_f32(v0, c3, v3_0));
        float32x4_t inner1 = vmulq_f32(coeff, vfmaq_f32(v1, c3, v3_1));
        float32x4_t t0 = neon_tanhf(inner0);
        float32x4_t t1 = neon_tanhf(inner1);
        vst1q_f32(x + i, vmulq_f32(half, vmulq_f32(v0, vaddq_f32(one, t0))));
        vst1q_f32(x + i + 4, vmulq_f32(half, vmulq_f32(v1, vaddq_f32(one, t1))));
    }
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t v3 = vmulq_f32(vmulq_f32(v, v), v);
        float32x4_t inner = vmulq_f32(coeff, vfmaq_f32(v, c3, v3));
        float32x4_t t = neon_tanhf(inner);
        vst1q_f32(x + i, vmulq_f32(half, vmulq_f32(v, vaddq_f32(one, t))));
    }
    for (; i < n; i++) {
        float val = x[i];
        float x3 = val * val * val;
        float inner = 0.7978845608028654f * (val + 0.044715f * x3);
        x[i] = 0.5f * val * (1.0f + tanhf(inner));
    }
#else
    for (int i = 0; i < n; i++) {
        float val = x[i];
        float x3 = val * val * val;
        float inner = 0.7978845608028654f * (val + 0.044715f * x3);
        x[i] = 0.5f * val * (1.0f + tanhf(inner));
    }
#endif
}

typedef struct {
    float *out;
    const float *gate_up;
    int seq_len;
    int intermediate;
} swiglu_task_t;

static void swiglu_worker(int tid, int n_threads, void *arg) {
    swiglu_task_t *t = (swiglu_task_t *)arg;
    int chunk = (t->seq_len + n_threads - 1) / n_threads;
    int s0 = tid * chunk;
    int s1 = s0 + chunk;
    if (s1 > t->seq_len) s1 = t->seq_len;
    if (s0 >= s1) return;

    int inter = t->intermediate;
    int alias_inplace = (t->out == t->gate_up);
    for (int s = s0; s < s1; s++) {
        const float *gu = t->gate_up + (size_t)s * 2 * inter;
        float *o = t->out + (size_t)s * inter;
        if (!alias_inplace) {
#ifdef __ARM_NEON
            /* NEON SiLU: process 4 gate-up pairs at a time */
            int j = 0;
            float32x4_t one = vdupq_n_f32(1.0f);
            for (; j + 3 < inter; j += 4) {
                float32x4x2_t gu4 = vld2q_f32(gu + 2 * j);
                float32x4_t g = gu4.val[0];
                float32x4_t u = gu4.val[1];
                float32x4_t neg_g = vnegq_f32(g);
                float32x4_t silu = vdivq_f32(g, vaddq_f32(one, neon_expf(neg_g)));
                vst1q_f32(o + j, vmulq_f32(silu, u));
            }
            for (; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                g = g / (1.0f + expf(-g));
                o[j] = g * u;
            }
#else
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                g = g / (1.0f + expf(-g)); /* SiLU */
                o[j] = g * u;
            }
#endif
        } else {
            /* In-place mode (decode seq=1): gate_up is interleaved [g0,u0,g1,u1,...],
             * out writes to front half. out[j] reads from 2j,2j+1 (2j >= j for j>=1),
             * so forward order is safe. */
#ifdef __ARM_NEON
            int j = 0;
            float32x4_t one = vdupq_n_f32(1.0f);
            for (; j + 3 < inter; j += 4) {
                float32x4x2_t gu4 = vld2q_f32(gu + 2 * j);
                float32x4_t g = gu4.val[0];
                float32x4_t u = gu4.val[1];
                float32x4_t neg_g = vnegq_f32(g);
                float32x4_t silu = vdivq_f32(g, vaddq_f32(one, neon_expf(neg_g)));
                vst1q_f32(o + j, vmulq_f32(silu, u));
            }
            for (; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                g = g / (1.0f + expf(-g));
                o[j] = g * u;
            }
#else
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                g = g / (1.0f + expf(-g)); /* SiLU */
                o[j] = g * u;
            }
#endif
        }
    }
}

void qwen_swiglu_multiply(float *out, const float *gate_up, int seq_len, int intermediate) {
    swiglu_task_t task = {
        .out = out,
        .gate_up = gate_up,
        .seq_len = seq_len,
        .intermediate = intermediate
    };

    if (qwen_get_n_threads() > 1 && seq_len >= 2 && intermediate >= 256) {
        qwen_parallel_for(swiglu_worker, &task);
    } else {
        swiglu_worker(0, 1, &task);
    }
}

void qwen_softmax(float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;
        float max_val = row[0];
        for (int c = 1; c < cols; c++) {
            if (row[c] > max_val) max_val = row[c];
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - max_val);
            sum += row[c];
        }
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) {
            row[c] *= inv_sum;
        }
    }
}

/* ========================================================================
 * FP16 Conversion
 * ======================================================================== */

/* Scalar FP32→FP16 fallback using IEEE 754 bit manipulation */
static inline uint16_t f32_to_f16_scalar(float v) {
    union { float f; uint32_t u; } bits;
    bits.f = v;
    uint32_t w = bits.u;
    uint32_t sign = (w >> 16) & 0x8000;
    int exp = (int)((w >> 23) & 0xff) - 127 + 15;
    uint32_t frac = (w >> 13) & 0x3ff;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | frac);
}

static inline float f16_to_f32_scalar(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1f;
    uint32_t frac = (uint32_t)h & 0x3ff;
    if (exp == 0) {
        if (frac == 0) { union { uint32_t u; float f; } r; r.u = sign; return r.f; }
        /* denorm */
        exp = 1;
        while (!(frac & 0x400)) { frac <<= 1; exp--; }
        frac &= 0x3ff;
        exp = (127 - 15 + exp);
    } else if (exp == 31) {
        exp = 255;
    } else {
        exp = exp - 15 + 127;
    }
    union { uint32_t u; float f; } r;
    r.u = sign | (exp << 23) | (frac << 13);
    return r.f;
}

void qwen_f32_to_f16(uint16_t *dst, const float *src, int n) {
#ifdef __ARM_NEON
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t lo = vld1q_f32(src + i);
        float32x4_t hi = vld1q_f32(src + i + 4);
        float16x4_t lo16 = vcvt_f16_f32(lo);
        float16x4_t hi16 = vcvt_f16_f32(hi);
        vst1_u16(dst + i, vreinterpret_u16_f16(lo16));
        vst1_u16(dst + i + 4, vreinterpret_u16_f16(hi16));
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        float16x4_t v16 = vcvt_f16_f32(v);
        vst1_u16(dst + i, vreinterpret_u16_f16(v16));
    }
    for (; i < n; i++) {
        dst[i] = f32_to_f16_scalar(src[i]);
    }
#else
    for (int i = 0; i < n; i++) {
        dst[i] = f32_to_f16_scalar(src[i]);
    }
#endif
}

void qwen_f16_to_f32(float *dst, const uint16_t *src, int n) {
#ifdef __ARM_NEON
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x4_t lo16 = vreinterpret_f16_u16(vld1_u16(src + i));
        float16x4_t hi16 = vreinterpret_f16_u16(vld1_u16(src + i + 4));
        vst1q_f32(dst + i, vcvt_f32_f16(lo16));
        vst1q_f32(dst + i + 4, vcvt_f32_f16(hi16));
    }
    for (; i + 4 <= n; i += 4) {
        float16x4_t v16 = vreinterpret_f16_u16(vld1_u16(src + i));
        vst1q_f32(dst + i, vcvt_f32_f16(v16));
    }
    for (; i < n; i++) {
        dst[i] = f16_to_f32_scalar(src[i]);
    }
#else
    for (int i = 0; i < n; i++) {
        dst[i] = f16_to_f32_scalar(src[i]);
    }
#endif
}

/* Mixed-precision dot product: dot(fp32_a, fp16_b) with on-the-fly conversion */
static inline float qwen_dot_f32_f16(const float *a, const uint16_t *b_fp16, int n) {
#ifdef __ARM_NEON
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int d = 0;
    for (; d + 8 <= n; d += 8) {
        float32x4_t a0 = vld1q_f32(a + d);
        float32x4_t a1 = vld1q_f32(a + d + 4);
        float16x4_t b16_0 = vreinterpret_f16_u16(vld1_u16(b_fp16 + d));
        float16x4_t b16_1 = vreinterpret_f16_u16(vld1_u16(b_fp16 + d + 4));
        float32x4_t b0 = vcvt_f32_f16(b16_0);
        float32x4_t b1 = vcvt_f32_f16(b16_1);
        acc0 = vfmaq_f32(acc0, a0, b0);
        acc1 = vfmaq_f32(acc1, a1, b1);
    }
    for (; d + 4 <= n; d += 4) {
        float32x4_t a0 = vld1q_f32(a + d);
        float16x4_t b16 = vreinterpret_f16_u16(vld1_u16(b_fp16 + d));
        acc0 = vfmaq_f32(acc0, a0, vcvt_f32_f16(b16));
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; d < n; d++) {
        sum += a[d] * f16_to_f32_scalar(b_fp16[d]);
    }
    return sum;
#else
    float sum = 0.0f;
    for (int d = 0; d < n; d++) {
        sum += a[d] * f16_to_f32_scalar(b_fp16[d]);
    }
    return sum;
#endif
}

/* dst += alpha * src_fp16 (with on-the-fly FP16→FP32 conversion) */
static inline void qwen_vec_axpy_f16_inplace(float *dst, const uint16_t *src_fp16,
                                               float alpha, int n) {
#ifdef __ARM_NEON
    float32x4_t va = vdupq_n_f32(alpha);
    int d = 0;
    for (; d + 8 <= n; d += 8) {
        float32x4_t d0 = vld1q_f32(dst + d);
        float32x4_t d1 = vld1q_f32(dst + d + 4);
        float16x4_t s16_0 = vreinterpret_f16_u16(vld1_u16(src_fp16 + d));
        float16x4_t s16_1 = vreinterpret_f16_u16(vld1_u16(src_fp16 + d + 4));
        float32x4_t s0 = vcvt_f32_f16(s16_0);
        float32x4_t s1 = vcvt_f32_f16(s16_1);
        vst1q_f32(dst + d, vfmaq_f32(d0, s0, va));
        vst1q_f32(dst + d + 4, vfmaq_f32(d1, s1, va));
    }
    for (; d + 4 <= n; d += 4) {
        float32x4_t d0 = vld1q_f32(dst + d);
        float16x4_t s16 = vreinterpret_f16_u16(vld1_u16(src_fp16 + d));
        vst1q_f32(dst + d, vfmaq_f32(d0, vcvt_f32_f16(s16), va));
    }
    for (; d < n; d++) {
        dst[d] += alpha * f16_to_f32_scalar(src_fp16[d]);
    }
#else
    for (int d = 0; d < n; d++) {
        dst[d] += alpha * f16_to_f32_scalar(src_fp16[d]);
    }
#endif
}

/* dst = dst * correction + src_fp16 (scale-add with FP16 source) */
static inline void qwen_vec_scale_add_f16(float *dst, const uint16_t *src_fp16,
                                            float correction, int n) {
#ifdef __ARM_NEON
    float32x4_t vc = vdupq_n_f32(correction);
    int d = 0;
    for (; d + 8 <= n; d += 8) {
        float32x4_t d0 = vld1q_f32(dst + d);
        float32x4_t d1 = vld1q_f32(dst + d + 4);
        float16x4_t s16_0 = vreinterpret_f16_u16(vld1_u16(src_fp16 + d));
        float16x4_t s16_1 = vreinterpret_f16_u16(vld1_u16(src_fp16 + d + 4));
        float32x4_t s0 = vcvt_f32_f16(s16_0);
        float32x4_t s1 = vcvt_f32_f16(s16_1);
        vst1q_f32(dst + d, vfmaq_f32(s0, d0, vc));
        vst1q_f32(dst + d + 4, vfmaq_f32(s1, d1, vc));
    }
    for (; d + 4 <= n; d += 4) {
        float32x4_t d0 = vld1q_f32(dst + d);
        float16x4_t s16 = vreinterpret_f16_u16(vld1_u16(src_fp16 + d));
        vst1q_f32(dst + d, vfmaq_f32(vcvt_f32_f16(s16), d0, vc));
    }
    for (; d < n; d++) {
        dst[d] = dst[d] * correction + f16_to_f32_scalar(src_fp16[d]);
    }
#else
    for (int d = 0; d < n; d++) {
        dst[d] = dst[d] * correction + f16_to_f32_scalar(src_fp16[d]);
    }
#endif
}

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

static inline float qwen_dot_f32(const float *a, const float *b, int n) {
    return qwen_dot_f32_impl(a, b, n);
}

/* dst = dst * scale */
static inline void qwen_vec_scale_inplace(float *dst, float scale, int n) {
    qwen_vec_scale_inplace_impl(dst, scale, n);
}

/* dst += alpha * src */
static inline void qwen_vec_axpy_inplace(float *dst, const float *src, float alpha, int n) {
    qwen_vec_axpy_inplace_impl(dst, src, alpha, n);
}

/* dst = dst * correction + src */
static inline void qwen_vec_scale_add(float *dst, const float *src, float correction, int n) {
    qwen_vec_scale_add_impl(dst, src, correction, n);
}

/* Max scores buffer size for 2-pass attention (stack allocated) */
#define ATTN_MAX_KEYS 2048

static void qwen_bidirectional_attention_heads(float *out, const float *Q, const float *K,
                                                const float *V, int n_heads, int head_dim,
                                                float scale, const int *window_starts,
                                                int n_windows, int head_start, int head_end) {
    int hidden = n_heads * head_dim;

    for (int h = head_start; h < head_end; h++) {
        for (int w = 0; w < n_windows; w++) {
            int ws = window_starts[w];
            int we = window_starts[w + 1];

            for (int i = ws; i < we; i++) {
                const float *q_row = Q + i * hidden + h * head_dim;
                float *o_row = out + i * hidden + h * head_dim;
                int n_keys = we - ws;
                float scores[ATTN_MAX_KEYS];

                /* Pass 1: compute all scores and find max */
                float max_score = -1e30f;
                for (int j = 0; j < n_keys; j++) {
                    const float *k_row = K + (ws + j) * hidden + h * head_dim;
                    scores[j] = qwen_dot_f32(q_row, k_row, head_dim) * scale;
                    if (scores[j] > max_score) max_score = scores[j];
                }

                /* Pass 2: NEON batch exp and accumulate weights */
                float sum_exp = 0.0f;
#ifdef __ARM_NEON
                {
                    float32x4_t vmax = vdupq_n_f32(max_score);
                    float32x4_t vsum = vdupq_n_f32(0.0f);
                    int j = 0;
                    for (; j + 4 <= n_keys; j += 4) {
                        float32x4_t s = vld1q_f32(scores + j);
                        float32x4_t e = neon_expf(vsubq_f32(s, vmax));
                        vst1q_f32(scores + j, e);
                        vsum = vaddq_f32(vsum, e);
                    }
                    sum_exp = vaddvq_f32(vsum);
                    for (; j < n_keys; j++) {
                        scores[j] = expf(scores[j] - max_score);
                        sum_exp += scores[j];
                    }
                }
#else
                for (int j = 0; j < n_keys; j++) {
                    scores[j] = expf(scores[j] - max_score);
                    sum_exp += scores[j];
                }
#endif

                /* Pass 3: weighted V sum */
                float inv_sum = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;
                for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;
                for (int j = 0; j < n_keys; j++) {
                    const float *v_row = V + (ws + j) * hidden + h * head_dim;
                    qwen_vec_axpy_inplace(o_row, v_row, scores[j] * inv_sum, head_dim);
                }
            }
        }
    }
}

typedef struct {
    float *out;
    const float *Q;
    const float *K;
    const float *V;
    int n_heads;
    int head_dim;
    float scale;
    const int *window_starts;
    int n_windows;
} bidir_attn_task_t;

static void bidir_attn_worker(int tid, int n_threads, void *arg) {
    bidir_attn_task_t *t = (bidir_attn_task_t *)arg;
    int chunk = (t->n_heads + n_threads - 1) / n_threads;
    int h0 = tid * chunk;
    int h1 = h0 + chunk;
    if (h1 > t->n_heads) h1 = t->n_heads;
    if (h0 >= h1) return;

    qwen_bidirectional_attention_heads(t->out, t->Q, t->K, t->V,
                                        t->n_heads, t->head_dim, t->scale,
                                        t->window_starts, t->n_windows, h0, h1);
}

void qwen_bidirectional_attention(float *out, const float *Q, const float *K,
                                   const float *V, int seq __attribute__((unused)),
                                   int n_heads, int head_dim, float scale,
                                   const int *window_starts, int n_windows) {
    if (qwen_get_n_threads() > 1 && n_heads >= 2) {
        bidir_attn_task_t task = {
            .out = out, .Q = Q, .K = K, .V = V,
            .n_heads = n_heads, .head_dim = head_dim, .scale = scale,
            .window_starts = window_starts, .n_windows = n_windows
        };
        qwen_parallel_for(bidir_attn_worker, &task);
        return;
    }

    qwen_bidirectional_attention_heads(out, Q, K, V, n_heads, head_dim, scale,
                                        window_starts, n_windows, 0, n_heads);
}

static void qwen_causal_attention_heads(float *out, const float *Q,
                                        const uint16_t *K_fp16,
                                        const uint16_t *V_fp16,
                                        int seq_q, int seq_k,
                                        int n_heads, int n_kv_heads, int head_dim,
                                        float scale, int q_offset,
                                        int head_start, int head_end) {
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    for (int h = head_start; h < head_end; h++) {
        int kv_h = h / heads_per_kv;

        for (int i = 0; i < seq_q; i++) {
            const float *q_row = Q + i * q_hidden + h * head_dim;
            float *o_row = out + i * q_hidden + h * head_dim;
            int global_pos = q_offset + i;
            int k_end = global_pos + 1;
            if (k_end > seq_k) k_end = seq_k;

            float scores[ATTN_MAX_KEYS];

            /* Pass 1: compute all scores and find max */
            float max_score = -1e30f;
            for (int j = 0; j < k_end; j++) {
                const uint16_t *k_row = K_fp16 + j * kv_hidden + kv_h * head_dim;
                scores[j] = qwen_dot_f32_f16(q_row, k_row, head_dim) * scale;
                if (scores[j] > max_score) max_score = scores[j];
            }

            /* Pass 2: NEON batch exp and accumulate weights */
            float sum_exp = 0.0f;
#ifdef __ARM_NEON
            {
                float32x4_t vmax = vdupq_n_f32(max_score);
                float32x4_t vsum = vdupq_n_f32(0.0f);
                int j = 0;
                for (; j + 4 <= k_end; j += 4) {
                    float32x4_t s = vld1q_f32(scores + j);
                    float32x4_t e = neon_expf(vsubq_f32(s, vmax));
                    vst1q_f32(scores + j, e);
                    vsum = vaddq_f32(vsum, e);
                }
                sum_exp = vaddvq_f32(vsum);
                for (; j < k_end; j++) {
                    scores[j] = expf(scores[j] - max_score);
                    sum_exp += scores[j];
                }
            }
#else
            for (int j = 0; j < k_end; j++) {
                scores[j] = expf(scores[j] - max_score);
                sum_exp += scores[j];
            }
#endif

            /* Pass 3: weighted V sum */
            float inv_sum = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;
            for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;
            for (int j = 0; j < k_end; j++) {
                const uint16_t *v_row = V_fp16 + j * kv_hidden + kv_h * head_dim;
                qwen_vec_axpy_f16_inplace(o_row, v_row, scores[j] * inv_sum, head_dim);
            }
        }
    }
}

typedef struct {
    float *out;
    const float *Q;
    const uint16_t *K_fp16;
    const uint16_t *V_fp16;
    int seq_q, seq_k;
    int n_heads, n_kv_heads;
    int head_dim;
    float scale;
    int q_offset;
} causal_attn_task_t;

static void causal_attn_worker(int tid, int n_threads, void *arg) {
    causal_attn_task_t *t = (causal_attn_task_t *)arg;
    int chunk = (t->n_heads + n_threads - 1) / n_threads;
    int h0 = tid * chunk;
    int h1 = h0 + chunk;
    if (h1 > t->n_heads) h1 = t->n_heads;
    if (h0 >= h1) return;

    qwen_causal_attention_heads(t->out, t->Q, t->K_fp16, t->V_fp16,
                                t->seq_q, t->seq_k, t->n_heads, t->n_kv_heads,
                                t->head_dim, t->scale, t->q_offset, h0, h1);
}

void qwen_causal_attention(float *out, const float *Q,
                            const uint16_t *K_fp16, const uint16_t *V_fp16,
                            int seq_q, int seq_k, int n_heads, int n_kv_heads,
                            int head_dim, float scale, int q_offset) {
    if (qwen_get_n_threads() > 1 && n_heads >= 2 && (seq_q >= 2 || seq_k >= 128)) {
        causal_attn_task_t task = {
            .out = out, .Q = Q, .K_fp16 = K_fp16, .V_fp16 = V_fp16,
            .seq_q = seq_q, .seq_k = seq_k,
            .n_heads = n_heads, .n_kv_heads = n_kv_heads,
            .head_dim = head_dim, .scale = scale, .q_offset = q_offset
        };
        qwen_parallel_for(causal_attn_worker, &task);
        return;
    }

    qwen_causal_attention_heads(out, Q, K_fp16, V_fp16,
                                seq_q, seq_k, n_heads, n_kv_heads,
                                head_dim, scale, q_offset, 0, n_heads);
}

/* ========================================================================
 * Position Embeddings
 * ======================================================================== */

void qwen_sinusoidal_pe(float *pe, int n_pos, int d_model) {
    int half = d_model / 2;
    float log_timescale = logf(10000.0f) / (float)(half - 1);

    for (int p = 0; p < n_pos; p++) {
        float *row = pe + p * d_model;
        for (int d = 0; d < half; d++) {
            float inv_timescale = expf(-(float)d * log_timescale);
            float angle = (float)p * inv_timescale;
            row[d] = sinf(angle);          /* first half: sin */
            row[half + d] = cosf(angle);   /* second half: cos */
        }
    }
}

void qwen_compute_rope_neox(float *cos_out, float *sin_out, const int *positions,
                              int seq, int head_dim, float theta) {
    int half = head_dim / 2;

    for (int s = 0; s < seq; s++) {
        float pos = (float)positions[s];
        for (int d = 0; d < half; d++) {
            float freq = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
            float angle = pos * freq;
            float c = cosf(angle);
            float sn = sinf(angle);
            /* Duplicate for full head_dim */
            cos_out[s * head_dim + d] = c;
            cos_out[s * head_dim + half + d] = c;
            sin_out[s * head_dim + d] = sn;
            sin_out[s * head_dim + half + d] = sn;
        }
    }
}

void qwen_apply_rope_neox(float *x, const float *cos_vals, const float *sin_vals,
                            int seq, int n_heads, int head_dim) {
    /*
     * NeoX split-half style:
     *   x1 = x[..., :half], x2 = x[..., half:]
     *   rotated = cat(-x2, x1)
     *   result = x * cos + rotated * sin
     */
    int half = head_dim / 2;
    int hidden = n_heads * head_dim;

    for (int s = 0; s < seq; s++) {
        const float *c = cos_vals + s * head_dim;
        const float *sn = sin_vals + s * head_dim;

        for (int h = 0; h < n_heads; h++) {
            float *vec = x + s * hidden + h * head_dim;

#ifdef __ARM_NEON
            int d = 0;
            for (; d + 4 <= half; d += 4) {
                float32x4_t x1 = vld1q_f32(vec + d);
                float32x4_t x2 = vld1q_f32(vec + half + d);
                float32x4_t cc = vld1q_f32(c + d);
                float32x4_t ss = vld1q_f32(sn + d);
                float32x4_t cc2 = vld1q_f32(c + half + d);
                float32x4_t ss2 = vld1q_f32(sn + half + d);
                /* new1 = x1 * cos - x2 * sin */
                vst1q_f32(vec + d, vsubq_f32(vmulq_f32(x1, cc), vmulq_f32(x2, ss)));
                /* new2 = x2 * cos2 + x1 * sin2 */
                vst1q_f32(vec + half + d, vfmaq_f32(vmulq_f32(x2, cc2), x1, ss2));
            }
            for (; d < half; d++) {
                float x1 = vec[d];
                float x2 = vec[half + d];
                vec[d]        = x1 * c[d]        + (-x2) * sn[d];
                vec[half + d] = x2 * c[half + d] + x1 * sn[half + d];
            }
#else
            for (int d = 0; d < half; d++) {
                float x1 = vec[d];           /* first half */
                float x2 = vec[half + d];    /* second half */
                vec[d]        = x1 * c[d]        + (-x2) * sn[d];
                vec[half + d] = x2 * c[half + d] + x1 * sn[half + d];
            }
#endif
        }
    }
}
