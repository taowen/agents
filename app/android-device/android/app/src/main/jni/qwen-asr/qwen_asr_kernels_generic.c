/*
 * qwen_asr_kernels_generic.c - architecture-generic hot kernels
 */

#include "qwen_asr_kernels_impl.h"

#include <string.h>
#include <stdlib.h>

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

void qwen_f32_matvec_fused_generic(float *y, const float *x, const float *W,
                                   const float *bias, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        const float *w_row = W + (size_t)o * in_dim;
        float sum = bias ? bias[o] : 0.0f;
        for (int k = 0; k < in_dim; k++) {
            sum += w_row[k] * x[k];
        }
        y[o] = sum;
    }
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

/* ========================================================================
 * Q4_K Super-Block MatVec (generic scalar)
 * ======================================================================== */

void qwen_q4k_matvec_fused_generic(float *out, const block_q4_k *blocks,
                                     const float *x, int rows, int cols) {
    int blocks_per_row = cols / QK_K;

    /* Quantize x to int8 */
    int cols_aligned = (cols + 15) & ~15;
    int8_t *x_int8 = (int8_t *)malloc(cols_aligned);
    float x_absmax = 0.0f;
    for (int i = 0; i < cols; i++) {
        float a = x[i] > 0 ? x[i] : -x[i];
        if (a > x_absmax) x_absmax = a;
    }
    float x_scale = x_absmax / 127.0f;
    float inv_x_scale = (x_absmax > 0.0f) ? 127.0f / x_absmax : 0.0f;
    for (int i = 0; i < cols; i++) {
        float v = x[i] * inv_x_scale;
        int iv = (int)(v + (v > 0 ? 0.5f : -0.5f));
        if (iv > 127) iv = 127;
        if (iv < -128) iv = -128;
        x_int8[i] = (int8_t)iv;
    }

    /* Precompute bsums */
    int total_subs = cols / 32;
    int32_t *bsums = (int32_t *)malloc(total_subs * sizeof(int32_t));
    for (int s = 0; s < total_subs; s++) {
        int32_t sum = 0;
        const int8_t *xg = x_int8 + s * 32;
        for (int i = 0; i < 32; i++) sum += (int32_t)xg[i];
        bsums[s] = sum;
    }

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

    free(x_int8);
    free(bsums);
}

void qwen_q4k_matvec_preq_generic(float *out, const block_q4_k *blocks,
                                    const int8_t *x_int8, float x_scale,
                                    const int32_t *bsums,
                                    int rows, int cols) {
    int blocks_per_row = cols / QK_K;

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
}

void qwen_q4k_gemm_chunk_generic(
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

    /* Per-token matvec fallback */
    for (int m = 0; m < M; m++) {
        qwen_q4k_matvec_preq_generic(
            Y + (size_t)m * Y_stride + r_start,
            W_chunk,
            x_int8 + (size_t)m * K,
            x_scales[m],
            bsums + (size_t)m * total_subs,
            n_rows, K);
    }
}

void qwen_q4k_argmax_range_generic(const block_q4_k *blocks,
                                     const float *x, int cols,
                                     int start, int end,
                                     int *best_out, float *best_val_out) {
    int blocks_per_row = cols / QK_K;

    /* Quantize x to int8 */
    int cols_aligned = (cols + 15) & ~15;
    int8_t *x_int8 = (int8_t *)malloc(cols_aligned);
    float x_absmax = 0.0f;
    for (int i = 0; i < cols; i++) {
        float a = x[i] > 0 ? x[i] : -x[i];
        if (a > x_absmax) x_absmax = a;
    }
    float x_scale = x_absmax / 127.0f;
    float inv_x_scale = (x_absmax > 0.0f) ? 127.0f / x_absmax : 0.0f;
    for (int i = 0; i < cols; i++) {
        float v = x[i] * inv_x_scale;
        int iv = (int)(v + (v > 0 ? 0.5f : -0.5f));
        if (iv > 127) iv = 127;
        if (iv < -128) iv = -128;
        x_int8[i] = (int8_t)iv;
    }

    int total_subs = cols / 32;
    int32_t *bsums = (int32_t *)malloc(total_subs * sizeof(int32_t));
    for (int s = 0; s < total_subs; s++) {
        int32_t sum = 0;
        const int8_t *xg = x_int8 + s * 32;
        for (int i = 0; i < 32; i++) sum += (int32_t)xg[i];
        bsums[s] = sum;
    }

    int best = start;
    float best_val = -1e30f;

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

    *best_out = best;
    *best_val_out = best_val;
    free(x_int8);
    free(bsums);
}
