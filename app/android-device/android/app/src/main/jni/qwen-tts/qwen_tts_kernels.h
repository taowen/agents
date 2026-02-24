/*
 * qwen_tts_kernels.h - Math kernel declarations for Qwen3-TTS
 */

#ifndef QWEN_TTS_KERNELS_H
#define QWEN_TTS_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#include "qwen_tts_quant.h"

/* ========================================================================
 * BLAS configuration
 * ======================================================================== */

#ifdef USE_BLAS
  #ifdef ACCELERATE_NEW_LAPACK
    #include <Accelerate/Accelerate.h>
  #elif defined(USE_OPENBLAS)
    #include <cblas.h>
  #endif
#endif

/* ========================================================================
 * Core math operations
 * ======================================================================== */

/* RMSNorm: out[i] = weight[i] * (x[i] / sqrt(mean(x^2) + eps)) */
void kernel_rms_norm(float *out, const float *x, const float *weight, int dim, float eps);

/* RMSNorm in-place */
void kernel_rms_norm_inplace(float *x, const float *weight, int dim, float eps);

/* LayerNorm: out[i] = weight[i] * (x[i] - mean) / sqrt(var + eps) + bias[i] */
void kernel_layer_norm(float *out, const float *x, const float *weight, const float *bias, int dim, float eps);

/* Quantize float32 input vector to Q8_0 blocks for matvec.
 * n must be a multiple of QK8_0.
 * dst must have n/QK8_0 blocks allocated. */
void kernel_quantize_x_q8(const float *x, int n, block_q8_0 *dst);

/* Matrix-vector multiply with Q8_0 weights and Q8_0 input:
 * out[r] = sum_b(W_q8[r*n_blocks+b].scale * x_q8[b].scale * dot(W_q8[r*n_blocks+b].qs, x_q8[b].qs))
 * W_q8 is [rows, n_blocks] blocks, x_q8 is [n_blocks] blocks. */
void kernel_matvec_q8(float *out, const block_q8_0 *W_q8, const block_q8_0 *x_q8,
                       int rows, int n_blocks);

/* Fused SwiGLU with Q8_0 weights:
 * gate = first `intermediate` rows, up = next `intermediate` rows.
 * out[i] = silu(gate[i]) * up[i] */
void kernel_swiglu_matvec_q8(float *out, const block_q8_0 *gate_up_q8,
                               const block_q8_0 *x_q8, int intermediate, int n_blocks);

/* Matrix-vector multiply: out = A @ x, A is [rows, cols] F32 */
void kernel_matvec_f32(float *out, const float *A, const float *x, int rows, int cols);

/* Matrix-matrix multiply: C = A @ B^T
 * A is [M, K], B is [N, K], C is [M, N] - all F32 */
void kernel_matmul_f32(float *C, const float *A, const float *B, int M, int N, int K);

/* SiLU activation: x * sigmoid(x) */
void kernel_silu_inplace(float *x, int n);

/* GELU activation */
void kernel_gelu_inplace(float *x, int n);

/* SnakeBeta activation: x + inv_beta * sin^2(alpha * x)
 * alpha/inv_beta are preprocessed at model load time. */
void kernel_snake_beta(float *out, const float *x, const float *alpha,
                       const float *beta, int channels, int length);

/* Element-wise add: out[i] = a[i] + b[i] */
void kernel_add(float *out, const float *a, const float *b, int n);

/* Element-wise add in-place: a[i] += b[i] */
void kernel_add_inplace(float *a, const float *b, int n);

/* Element-wise multiply in-place: a[i] *= b[i] */
void kernel_mul_inplace(float *a, const float *b, int n);

/* Scale in-place: x[i] *= scale */
void kernel_scale_inplace(float *x, float scale, int n);

/* Softmax over n elements */
void kernel_softmax(float *x, int n);

/* Top-K sampling: returns sampled index */
int kernel_sample_top_k(const float *logits, int vocab_size, int top_k,
                        float top_p, float temperature, float *rng_state);

/* Apply repetition penalty to logits */
void kernel_apply_repetition_penalty(float *logits, const int *token_ids,
                                     int n_tokens, int vocab_size, float penalty);

/* RoPE: apply rotary position embedding to q/k vectors
 * q/k shape: [num_heads, head_dim] */
void kernel_rope_apply(float *q, float *k, const float *cos, const float *sin,
                       int num_heads, int head_dim);

/* M-RoPE: multimodal rotary position embedding
 * Uses 3 position streams (temporal, height, width) */
void kernel_mrope_apply(float *q, float *k, const float *cos, const float *sin,
                        int num_heads, int head_dim, const int *mrope_section);

/* Causal Conv1d forward pass */
void kernel_causal_conv1d(float *out, const float *input, const float *weight,
                          const float *bias, int in_channels, int out_channels,
                          int kernel_size, int length, int dilation, int groups);

/* Transposed Conv1d (for upsampling) */
void kernel_transposed_conv1d(float *out, const float *input, const float *weight,
                              const float *bias, int in_channels, int out_channels,
                              int kernel_size, int stride, int length, int *out_length);

/* Clamp values to [-1, 1] */
void kernel_clamp(float *x, int n, float min_val, float max_val);

/* Dot product */
float kernel_dot(const float *a, const float *b, int n);

/* Sum of squares */
float kernel_sum_sq(const float *x, int n);

/* Copy BF16 array to F32 */
void kernel_bf16_to_f32(float *out, const uint16_t *in, int n);

/* Zero buffer */
void kernel_zero(float *x, int n);

/* ========================================================================
 * Platform-specific dispatch
 * ======================================================================== */

/* Initialize kernel dispatch based on detected CPU features */
void kernel_init(void);

#endif /* QWEN_TTS_KERNELS_H */
