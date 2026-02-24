/*
 * qwen_asr_kernels_generic.c - architecture-generic hot kernels
 */

#include "qwen_asr_kernels_impl.h"

#include <string.h>

void qwen_bf16_matvec_fused_generic(float *y, const float *x, const uint16_t *W_bf16,
                                    const float *bias, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        const uint16_t *w_row = W_bf16 + (size_t)o * in_dim;
        float sum = bias ? bias[o] : 0.0f;
        for (int k = 0; k < in_dim; k++) {
            uint32_t f32_bits = ((uint32_t)w_row[k]) << 16;
            float w_val;
            memcpy(&w_val, &f32_bits, sizeof(float));
            sum += w_val * x[k];
        }
        y[o] = sum;
    }
}

void qwen_argmax_bf16_range_generic(const float *x, const uint16_t *W_bf16,
                                    int in_dim, int start, int end,
                                    int *best_out, float *best_val_out) {
    int best = start;
    float best_val = -1e30f;

    for (int o = start; o < end; o++) {
        const uint16_t *w_row = W_bf16 + (size_t)o * in_dim;
        float sum = 0.0f;
        for (int k = 0; k < in_dim; k++) {
            uint32_t f32_bits = ((uint32_t)w_row[k]) << 16;
            float w_val;
            memcpy(&w_val, &f32_bits, sizeof(float));
            sum += w_val * x[k];
        }
        if (sum > best_val) {
            best_val = sum;
            best = o;
        }
    }

    *best_out = best;
    *best_val_out = best_val;
}

void qwen_q8_matvec_fused_generic(float *y, const block_q8_0 *x_q8,
                                   const block_q8_0 *W_q8, const float *bias,
                                   int n_blocks, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        const block_q8_0 *w_row = W_q8 + (size_t)o * n_blocks;
        float sum = bias ? bias[o] : 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            float ws = w_row[b].scale;
            float xs = x_q8[b].scale;
            int32_t dot = 0;
            for (int j = 0; j < QK8_0; j++) {
                dot += (int32_t)w_row[b].qs[j] * (int32_t)x_q8[b].qs[j];
            }
            sum += ws * xs * (float)dot;
        }
        y[o] = sum;
    }
}

void qwen_argmax_q8_range_generic(const block_q8_0 *x_q8,
                                   const block_q8_0 *W_q8,
                                   int n_blocks, int start, int end,
                                   int *best_out, float *best_val_out) {
    int best = start;
    float best_val = -1e30f;

    for (int o = start; o < end; o++) {
        const block_q8_0 *w_row = W_q8 + (size_t)o * n_blocks;
        float sum = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            float ws = w_row[b].scale;
            float xs = x_q8[b].scale;
            int32_t dot = 0;
            for (int j = 0; j < QK8_0; j++) {
                dot += (int32_t)w_row[b].qs[j] * (int32_t)x_q8[b].qs[j];
            }
            sum += ws * xs * (float)dot;
        }
        if (sum > best_val) {
            best_val = sum;
            best = o;
        }
    }

    *best_out = best;
    *best_val_out = best_val;
}

/* ========================================================================
 * Q4_K × Q8_K dot product (scalar fallback)
 * Reference: llama.cpp ggml_vec_dot_q4_K_q8_K_generic
 * ======================================================================== */

static float q4k_q8k_dot_scalar(const block_q4_K *x, const block_q8_K *y, int n_blocks) {
    float sumf = 0.0f;
    for (int i = 0; i < n_blocks; i++) {
        const uint8_t *q4 = x[i].qs;
        const int8_t  *q8 = y[i].qs;
        const uint8_t *scales = x[i].scales;

        const float d = fp16_to_fp32(x[i].d);
        const float m = fp16_to_fp32(x[i].dmin);
        const float d8 = y[i].d;

        float summs = 0.0f;
        int64_t sumqx = 0;
        for (int j = 0; j < QK_K / 64; j++) {
            const uint8_t s0 = scales[2 * j + 0] & 0x3f;
            const uint8_t s1 = scales[2 * j + 1] & 0x3f;
            const uint8_t s2 = scales[2 * j + 2] & 0x3f;
            const uint8_t s3 = scales[2 * j + 3] & 0x3f;
            (void)s0; (void)s1; (void)s2; (void)s3; /* scales used for min correction only */
            for (int l = 0; l < 32; l++) {
                sumqx += (int16_t)q8[l +  0] * (q4[l] & 0xF);
                sumqx += (int16_t)q8[l + 32] * (q4[l] >> 4);
            }
            q4 += 32;
            q8 += 64;
            summs += s0 * y[i].bsums[2 * j + 0]
                   + s1 * y[i].bsums[2 * j + 1]
                   + s2 * y[i].bsums[2 * j + 2]
                   + s3 * y[i].bsums[2 * j + 3];
        }
        sumf += d8 * (d * sumqx - m * summs);
    }
    return sumf;
}

void qwen_q4k_q8k_matvec_generic(float *y, const block_q8_K *x_q8k,
                                   const block_q4_K *W_q4k, const float *bias,
                                   int n_blocks_k, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        const block_q4_K *w_row = W_q4k + (size_t)o * n_blocks_k;
        float sum = bias ? bias[o] : 0.0f;
        sum += q4k_q8k_dot_scalar(w_row, x_q8k, n_blocks_k);
        y[o] = sum;
    }
}

void qwen_argmax_q4k_range_generic(const block_q8_K *x_q8k,
                                    const block_q4_K *W_q4k,
                                    int n_blocks_k, int start, int end,
                                    int *best_out, float *best_val_out) {
    int best = start;
    float best_val = -1e30f;

    for (int o = start; o < end; o++) {
        const block_q4_K *w_row = W_q4k + (size_t)o * n_blocks_k;
        float sum = q4k_q8k_dot_scalar(w_row, x_q8k, n_blocks_k);
        if (sum > best_val) {
            best_val = sum;
            best = o;
        }
    }

    *best_out = best;
    *best_val_out = best_val;
}

float qwen_dot_f32_generic(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

void qwen_vec_scale_inplace_generic(float *dst, float scale, int n) {
    for (int i = 0; i < n; i++) dst[i] *= scale;
}

void qwen_vec_axpy_inplace_generic(float *dst, const float *src, float alpha, int n) {
    for (int i = 0; i < n; i++) dst[i] += alpha * src[i];
}

void qwen_vec_scale_add_generic(float *dst, const float *src, float correction, int n) {
    for (int i = 0; i < n; i++) dst[i] = dst[i] * correction + src[i];
}
