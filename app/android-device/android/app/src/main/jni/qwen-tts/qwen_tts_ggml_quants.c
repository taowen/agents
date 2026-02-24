/*
 * qwen_tts_ggml_quants.c - Q4_K / Q8_K quantization (extracted from llama.cpp/ggml)
 *
 * Contains:
 *   - make_qkx2_quants()      sub-block quantization helper
 *   - get_scale_min_k4()      unpack 6-bit scales
 *   - quantize_row_q4_K_ref() F32 -> Q4_K
 *   - quantize_row_q8_K()     F32 -> Q8_K (scalar; NEON path optional)
 *   - dequantize_row_q4_K()   Q4_K -> F32
 *   - vec_dot_q4_K_q8_K()     NEON SDOT + scalar dot product
 *
 * ARM NEON (dotprod) + scalar fallback only.  No AVX/SSE.
 * Adapted from ggml-quants.c and arch/arm/quants.c for C99.
 */

#include "qwen_tts_ggml_quants.h"

#include <math.h>
#include <string.h>
#include <assert.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

/* ========================================================================
 * Helpers
 * ======================================================================== */

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

/* ========================================================================
 * make_qkx2_quants - sub-block quantization with scale + min
 *
 * Finds optimal scale and min such that:
 *   x[i] ~= scale * L[i] - min
 * where L[i] is in [0, nmax].
 * ======================================================================== */

static float make_qkx2_quants(int n, int nmax, const float *x, const float *weights,
        uint8_t *L, float *the_min, uint8_t *Laux,
        float rmin, float rdelta, int nstep, int use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax / (max - min);
    float scale = 1 / iscale;
    float best_error = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * (x[i] - min));
        L[i] = (uint8_t)MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_error += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * (x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = (uint8_t)l;
            float w = weights[i];
            sum_l  += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                cur_error += w * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

/* ========================================================================
 * get_scale_min_k4 - unpack 6-bit scales/mins from packed format
 * ======================================================================== */

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

/* ========================================================================
 * quantize_row_q4_K_ref - F32 -> Q4_K (reference scalar implementation)
 * ======================================================================== */

void quantize_row_q4_K_ref(const float *x, block_q4_K *y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = (int)(k / QK_K);

    uint8_t L[QK_K];
    uint8_t Laux[32];
    float   weights[32];
    float   mins[QK_K / 32];
    float   scales[QK_K / 32];

    for (int i = 0; i < nb; i++) {
        float max_scale = 0;
        float max_min = 0;
        for (int j = 0; j < QK_K / 32; ++j) {
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32 * j + l] * x[32 * j + l];
            float av_x = sqrtf(sum_x2 / 32);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32 * j + l]);
            scales[j] = make_qkx2_quants(32, 15, x + 32 * j, weights, L + 32 * j,
                                          &mins[j], Laux, -1.f, 0.1f, 20, 0);
            float scale = scales[j];
            if (scale > max_scale) max_scale = scale;
            float m = mins[j];
            if (m > max_min) max_min = m;
        }

        float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f / max_min   : 0.f;
        for (int j = 0; j < QK_K / 32; ++j) {
            uint8_t ls = (uint8_t)MIN(63, nearest_int(inv_scale * scales[j]));
            uint8_t lm = (uint8_t)MIN(63, nearest_int(inv_min * mins[j]));
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j + 4] = lm;
            } else {
                y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j - 4] |= ((ls >> 4) << 6);
                y[i].scales[j - 0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d    = GGML_FP32_TO_FP16(max_scale / 63.f);
        y[i].dmin = GGML_FP32_TO_FP16(max_min / 63.f);

        uint8_t sc, m;
        for (int j = 0; j < QK_K / 32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32 * j + ii] + dm) / d);
                l = MAX(0, MIN(15, l));
                L[32 * j + ii] = (uint8_t)l;
            }
        }

        uint8_t *q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l)
                q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }

        x += QK_K;
    }
}

/* ========================================================================
 * quantize_row_q8_K - F32 -> Q8_K (scalar reference)
 *
 * The NEON path in llama.cpp just calls the ref version for q8_K on ARM.
 * ======================================================================== */

void quantize_row_q8_K(const float *x, block_q8_K *y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0;
        float max_val = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax;
                max_val = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            memset(y[i].bsums, 0, QK_K / 16 * sizeof(int16_t));
            x += QK_K;
            continue;
        }
        const float iscale = -127.f / max_val;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale * x[j]);
            y[i].qs[j] = (int8_t)MIN(127, v);
        }
        for (int j = 0; j < QK_K / 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j * 16 + ii];
            }
            y[i].bsums[j] = (int16_t)sum;
        }
        y[i].d = 1.f / iscale;
        x += QK_K;
    }
}

/* ========================================================================
 * dequantize_row_q4_K - Q4_K -> F32
 * ======================================================================== */

void dequantize_row_q4_K(const block_q4_K *x, float *y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = (int)(k / QK_K);

    for (int i = 0; i < nb; i++) {
        const uint8_t *q = x[i].qs;
        const float d   = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4)  - m2;
            q += 32;
            is += 2;
        }
    }
}

/* ========================================================================
 * vec_dot_q4_K_q8_K - dot product of Q4_K weights and Q8_K activations
 *
 * ARM NEON (dotprod) path + scalar fallback.
 * ======================================================================== */

void vec_dot_q4_K_q8_K(int n, float *s, const block_q4_K *x, const block_q8_K *y) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#if (defined(__ARM_NEON) || defined(__aarch64__)) && defined(__ARM_FEATURE_DOTPROD)
    const uint8x16_t m4b = vdupq_n_u8(0xf);
    const int32x4_t mzero = vdupq_n_s32(0);

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));

        memcpy(utmp, x[i].scales, 12);

        uint32x2_t mins8 = vdup_n_u32(0);
        mins8 = vset_lane_u32(utmp[1] & kmask1, mins8, 0);
        mins8 = vset_lane_u32(((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4), mins8, 1);

        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] &= kmask1;

        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
                                         vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        sumf -= dmin * (float)vaddvq_s32(prod);

        const uint8_t *scales = (const uint8_t *)utmp;

        const uint8_t *q4 = x[i].qs;
        const int8_t  *q8 = y[i].qs;

        int32_t sumi1 = 0;
        int32_t sumi2 = 0;

        for (int j = 0; j < QK_K / 64; ++j) {
            /* Load 32 bytes of q4 nibbles */
            const uint8x16_t q4bits_0 = vld1q_u8(q4);
            const uint8x16_t q4bits_1 = vld1q_u8(q4 + 16);
            q4 += 32;

            /* Low nibbles */
            const int8x16_t q4lo_0 = vreinterpretq_s8_u8(vandq_u8(q4bits_0, m4b));
            const int8x16_t q4lo_1 = vreinterpretq_s8_u8(vandq_u8(q4bits_1, m4b));
            const int8x16_t q8lo_0 = vld1q_s8(q8);
            const int8x16_t q8lo_1 = vld1q_s8(q8 + 16);
            q8 += 32;

            const int32x4_t p1 = vdotq_s32(vdotq_s32(mzero, q4lo_0, q8lo_0), q4lo_1, q8lo_1);
            sumi1 += vaddvq_s32(p1) * scales[2 * j + 0];

            /* High nibbles */
            const int8x16_t q4hi_0 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits_0, 4));
            const int8x16_t q4hi_1 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits_1, 4));
            const int8x16_t q8hi_0 = vld1q_s8(q8);
            const int8x16_t q8hi_1 = vld1q_s8(q8 + 16);
            q8 += 32;

            const int32x4_t p2 = vdotq_s32(vdotq_s32(mzero, q4hi_0, q8hi_0), q4hi_1, q8hi_1);
            sumi2 += vaddvq_s32(p2) * scales[2 * j + 1];
        }

        sumf += d * (sumi1 + sumi2);
    }

    *s = sumf;

#else
    /* Scalar fallback */
    const uint8_t *scale_ptr = (const uint8_t *)&utmp[0];
    const uint8_t *mins_ptr  = (const uint8_t *)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums[8];
    int32_t aux32[8];
    memset(sums, 0, 8 * sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *q4 = x[i].qs;
        const int8_t  *q8 = y[i].qs;
        memset(aux32, 0, 8 * sizeof(int32_t));
        int8_t *a = aux8;
        for (int j = 0; j < QK_K / 64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
            a += 32;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K / 16; ++j) sumi += y[i].bsums[j] * mins_ptr[j / 2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K / 32; ++j) {
            int32_t scale = scale_ptr[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}
