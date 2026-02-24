/*
 * qwen_asr_kernels_impl.h - NEON kernel declarations for ARM64
 */

#ifndef QWEN_ASR_KERNELS_IMPL_H
#define QWEN_ASR_KERNELS_IMPL_H

#include <stdint.h>
#include <stddef.h>
#include "qwen_asr_quant.h"

/* Thread pool parallel_for (defined in qwen_asr_kernels.c) */
typedef void (*parallel_fn_t)(int tid, int n_threads, void *arg);
void qwen_parallel_for(parallel_fn_t fn, void *arg);
int qwen_get_n_threads(void);

/* Q8_0 matvec */
void qwen_q8_matvec_fused_neon(float *y, const block_q8_0 *x_q8,
                                const block_q8_0 *W_q8, const float *bias,
                                int n_blocks, int out_dim);

/* Q4_K matvec / GEMM / argmax */
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

/* Vector ops */
float qwen_dot_f32_neon(const float *a, const float *b, int n);
void qwen_vec_scale_inplace_neon(float *dst, float scale, int n);
void qwen_vec_axpy_inplace_neon(float *dst, const float *src, float alpha, int n);
void qwen_vec_scale_add_neon(float *dst, const float *src, float correction, int n);

/* Dispatch macros: NEON only */
#define qwen_q8_matvec_fused_impl qwen_q8_matvec_fused_neon
#define qwen_q4k_matvec_fused_impl qwen_q4k_matvec_fused_neon
#define qwen_q4k_matvec_preq_impl qwen_q4k_matvec_preq_neon
#define qwen_q4k_gemm_chunk_impl qwen_q4k_gemm_chunk_neon
#define qwen_q4k_argmax_range_impl qwen_q4k_argmax_range_neon
#define qwen_dot_f32_impl qwen_dot_f32_neon
#define qwen_vec_scale_inplace_impl qwen_vec_scale_inplace_neon
#define qwen_vec_axpy_inplace_impl qwen_vec_axpy_inplace_neon
#define qwen_vec_scale_add_impl qwen_vec_scale_add_neon

#endif /* QWEN_ASR_KERNELS_IMPL_H */
