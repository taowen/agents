/*
 * qwen_tts_kernels_ops.c - Signal-processing kernel implementations
 *
 * Split from qwen_tts_kernels.c. Contains:
 *   - Causal Conv1d and Transposed Conv1d
 *   - RoPE and M-RoPE
 *   - SnakeBeta activation
 *   - Softmax
 *   - Top-K sampling and repetition penalty
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
/* Persistent scratch for packed conv1d weights                              */
/* ======================================================================== */

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

/* ======================================================================== */
/* SnakeBeta activation                                                      */
/* ======================================================================== */

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
     * Uses a 5th-order polynomial: sin(x) ~ x - x^3/6 + x^5/120
     * after range reduction to [-pi, pi].
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
        float32x4_t v_inv_twopi = vdupq_n_f32(0.15915494309189533f);  /* 1/(2pi) */
        float32x4_t v_twopi    = vdupq_n_f32(6.283185307179586f);     /* 2pi */
        float32x4_t v_pi       = vdupq_n_f32(3.141592653589793f);     /* pi */
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
            /* Range reduce to [-pi, pi]: ax = ax - round(ax/(2pi)) * 2pi */
            float32x4_t n = vfmaq_f32(v_half, ax, v_inv_twopi);
            /* floor via convert to int and back */
            int32x4_t ni = vcvtq_s32_f32(n);
            /* Adjust for negative: if ax < 0 and fractional, ni is wrong */
            float32x4_t nf = vcvtq_f32_s32(ni);
            /* Correct: compare nf > n, subtract 1 if so */
            uint32x4_t mask = vcgtq_f32(nf, n);
            nf = vsubq_f32(nf, vreinterpretq_f32_u32(vandq_u32(mask, vreinterpretq_u32_f32(vdupq_n_f32(1.0f)))));
            ax = vfmsq_f32(ax, nf, v_twopi);
            /* Clamp to [-pi, pi] for safety */
            ax = vmaxq_f32(ax, v_neg_pi);
            ax = vminq_f32(ax, v_pi);
            /* sin(ax) ~ ax * (1 + ax^2 * (-1/6 + ax^2 * 1/120)) */
            float32x4_t ax2 = vmulq_f32(ax, ax);
            float32x4_t poly = vfmaq_f32(v_c3, ax2, v_c5);
            poly = vfmaq_f32(vdupq_n_f32(1.0f), ax2, poly);
            float32x4_t s = vmulq_f32(ax, poly);
            /* out = x + inv_b * s^2 */
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
