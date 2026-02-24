/*
 * qwen_asr_kernels_impl.h - internal architecture dispatch for hot kernels
 */

#ifndef QWEN_ASR_KERNELS_IMPL_H
#define QWEN_ASR_KERNELS_IMPL_H

#include <stdint.h>
#include <stddef.h>
#include "qwen_asr_quant.h"

/* Q4_K matvec: single-row dot product, called from threaded dispatch.
 * Quantizes x to int8 internally, uses SDOT for NEON path. */
void qwen_q4k_matvec_fused_generic(float *out, const block_q4_k *blocks,
                                     const float *x, int rows, int cols);
/* Q4_K matvec with pre-quantized int8 input (no per-call x quantization). */
void qwen_q4k_matvec_preq_generic(float *out, const block_q4_k *blocks,
                                    const int8_t *x_int8, float x_scale,
                                    const int32_t *bsums,
                                    int rows, int cols);
void qwen_q4k_gemm_chunk_generic(
    float *Y, int Y_stride,
    const block_q4_k *W_q4k, int blocks_per_row,
    const int8_t *x_int8, int K,
    const float *x_scales,
    const int32_t *bsums, int total_subs,
    int M, int r_start, int r_end);
void qwen_q4k_argmax_range_generic(const block_q4_k *blocks,
                                     const float *x, int cols,
                                     int start, int end,
                                     int *best_out, float *best_val_out);

void qwen_bf16_matvec_fused_generic(float *y, const float *x, const uint16_t *W_bf16,
                                    const float *bias, int in_dim, int out_dim);
void qwen_f32_matvec_fused_generic(float *y, const float *x, const float *W,
                                   const float *bias, int in_dim, int out_dim);
void qwen_q8_matvec_fused_generic(float *y, const block_q8_0 *x_q8,
                                   const block_q8_0 *W_q8, const float *bias,
                                   int n_blocks, int out_dim);
void qwen_argmax_bf16_range_generic(const float *x, const uint16_t *W_bf16,
                                    int in_dim, int start, int end,
                                    int *best_out, float *best_val_out);
void qwen_argmax_q8_range_generic(const block_q8_0 *x_q8,
                                   const block_q8_0 *W_q8,
                                   int n_blocks, int start, int end,
                                   int *best_out, float *best_val_out);
float qwen_dot_f32_generic(const float *a, const float *b, int n);
void qwen_vec_scale_inplace_generic(float *dst, float scale, int n);
void qwen_vec_axpy_inplace_generic(float *dst, const float *src, float alpha, int n);
void qwen_vec_scale_add_generic(float *dst, const float *src, float correction, int n);

#ifdef __ARM_NEON
void qwen_q4k_matvec_fused_neon(float *out, const block_q4_k *blocks,
                                  const float *x, int rows, int cols);
void qwen_q4k_matvec_preq_neon(float *out, const block_q4_k *blocks,
                                 const int8_t *x_int8, float x_scale,
                                 const int32_t *bsums,
                                 int rows, int cols);
void qwen_q4k_gemm_chunk_neon(
    float *Y, int Y_stride,
    const block_q4_k *W_q4k, int blocks_per_row,
    const int8_t *x_int8, int K,
    const float *x_scales,
    const int32_t *bsums, int total_subs,
    int M, int r_start, int r_end);
void qwen_q4k_argmax_range_neon(const block_q4_k *blocks,
                                  const float *x, int cols,
                                  int start, int end,
                                  int *best_out, float *best_val_out);
void qwen_bf16_matvec_fused_neon(float *y, const float *x, const uint16_t *W_bf16,
                                 const float *bias, int in_dim, int out_dim);
void qwen_f32_matvec_fused_neon(float *y, const float *x, const float *W,
                                const float *bias, int in_dim, int out_dim);
void qwen_q8_matvec_fused_neon(float *y, const block_q8_0 *x_q8,
                                const block_q8_0 *W_q8, const float *bias,
                                int n_blocks, int out_dim);
void qwen_argmax_bf16_range_neon(const float *x, const uint16_t *W_bf16,
                                 int in_dim, int start, int end,
                                 int *best_out, float *best_val_out);
void qwen_argmax_q8_range_neon(const block_q8_0 *x_q8,
                                const block_q8_0 *W_q8,
                                int n_blocks, int start, int end,
                                int *best_out, float *best_val_out);
float qwen_dot_f32_neon(const float *a, const float *b, int n);
void qwen_vec_scale_inplace_neon(float *dst, float scale, int n);
void qwen_vec_axpy_inplace_neon(float *dst, const float *src, float alpha, int n);
void qwen_vec_scale_add_neon(float *dst, const float *src, float correction, int n);

#define qwen_bf16_matvec_fused_impl qwen_bf16_matvec_fused_neon
#define qwen_f32_matvec_fused_impl qwen_f32_matvec_fused_neon
#define qwen_q8_matvec_fused_impl qwen_q8_matvec_fused_neon
#define qwen_q4k_matvec_fused_impl qwen_q4k_matvec_fused_neon
#define qwen_q4k_matvec_preq_impl qwen_q4k_matvec_preq_neon
#define qwen_q4k_gemm_chunk_impl qwen_q4k_gemm_chunk_neon
#define qwen_q4k_argmax_range_impl qwen_q4k_argmax_range_neon
#define qwen_argmax_bf16_range_impl qwen_argmax_bf16_range_neon
#define qwen_argmax_q8_range_impl qwen_argmax_q8_range_neon
#define qwen_dot_f32_impl qwen_dot_f32_neon
#define qwen_vec_scale_inplace_impl qwen_vec_scale_inplace_neon
#define qwen_vec_axpy_inplace_impl qwen_vec_axpy_inplace_neon
#define qwen_vec_scale_add_impl qwen_vec_scale_add_neon

#elif defined(__AVX2__) && defined(__FMA__)
void qwen_bf16_matvec_fused_avx(float *y, const float *x, const uint16_t *W_bf16,
                                 const float *bias, int in_dim, int out_dim);
void qwen_f32_matvec_fused_avx(float *y, const float *x, const float *W,
                                const float *bias, int in_dim, int out_dim);
void qwen_argmax_bf16_range_avx(const float *x, const uint16_t *W_bf16,
                                 int in_dim, int start, int end,
                                 int *best_out, float *best_val_out);
float qwen_dot_f32_avx(const float *a, const float *b, int n);
void qwen_vec_scale_inplace_avx(float *dst, float scale, int n);
void qwen_vec_axpy_inplace_avx(float *dst, const float *src, float alpha, int n);
void qwen_vec_scale_add_avx(float *dst, const float *src, float correction, int n);

#define qwen_bf16_matvec_fused_impl qwen_bf16_matvec_fused_avx
#define qwen_f32_matvec_fused_impl qwen_f32_matvec_fused_avx
#define qwen_q8_matvec_fused_impl qwen_q8_matvec_fused_generic
#define qwen_q4k_matvec_fused_impl qwen_q4k_matvec_fused_generic
#define qwen_q4k_matvec_preq_impl qwen_q4k_matvec_preq_generic
#define qwen_q4k_gemm_chunk_impl qwen_q4k_gemm_chunk_generic
#define qwen_q4k_argmax_range_impl qwen_q4k_argmax_range_generic
#define qwen_argmax_bf16_range_impl qwen_argmax_bf16_range_avx
#define qwen_argmax_q8_range_impl qwen_argmax_q8_range_generic
#define qwen_dot_f32_impl qwen_dot_f32_avx
#define qwen_vec_scale_inplace_impl qwen_vec_scale_inplace_avx
#define qwen_vec_axpy_inplace_impl qwen_vec_axpy_inplace_avx
#define qwen_vec_scale_add_impl qwen_vec_scale_add_avx

#else
#define qwen_bf16_matvec_fused_impl qwen_bf16_matvec_fused_generic
#define qwen_f32_matvec_fused_impl qwen_f32_matvec_fused_generic
#define qwen_q8_matvec_fused_impl qwen_q8_matvec_fused_generic
#define qwen_q4k_matvec_fused_impl qwen_q4k_matvec_fused_generic
#define qwen_q4k_matvec_preq_impl qwen_q4k_matvec_preq_generic
#define qwen_q4k_gemm_chunk_impl qwen_q4k_gemm_chunk_generic
#define qwen_q4k_argmax_range_impl qwen_q4k_argmax_range_generic
#define qwen_argmax_bf16_range_impl qwen_argmax_bf16_range_generic
#define qwen_argmax_q8_range_impl qwen_argmax_q8_range_generic
#define qwen_dot_f32_impl qwen_dot_f32_generic
#define qwen_vec_scale_inplace_impl qwen_vec_scale_inplace_generic
#define qwen_vec_axpy_inplace_impl qwen_vec_axpy_inplace_generic
#define qwen_vec_scale_add_impl qwen_vec_scale_add_generic
#endif

#endif /* QWEN_ASR_KERNELS_IMPL_H */
