/*
 * qwen_asr_kernels.h - Math kernels for Qwen3-ASR inference
 *
 * Low-level math operations. All operate on float32 tensors in row-major order.
 * Adapted from voxtral-realtime project.
 */

#ifndef QWEN_ASR_KERNELS_H
#define QWEN_ASR_KERNELS_H

#include <stddef.h>
#include <stdint.h>
#include "qwen_asr_quant.h"

/* ========================================================================
 * Basic Operations
 * ======================================================================== */

void qwen_add_inplace(float *a, const float *b, int n);

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

/* Q8_0 weight variants */
void qwen_linear_q8(float *y, const float *x, const block_q8_0 *W_q8,
                    const float *b, int seq_len, int in_dim, int out_dim);

void qwen_linear_nobias_q8(float *y, const float *x, const block_q8_0 *W_q8,
                            int seq_len, int in_dim, int out_dim);

/* seq=1 decoder fast path: compute Q/K/V matvecs with one threaded dispatch */
void qwen_linear_nobias_q8_qkv(float *q, float *k, float *v, const float *x,
                                const block_q8_0 *Wq_q8,
                                const block_q8_0 *Wk_q8,
                                const block_q8_0 *Wv_q8,
                                int in_dim, int q_dim, int kv_dim);

/* Fused QKV GEMM: quantize input once, compute Q/K/V projections.
 * Falls back to matvec QKV path when seq_len=1. */
void qwen_linear_q8_qkv_batched(
    float *q, float *k, float *v,
    const float *x,
    const block_q8_0 *Wq_q8, const float *bq,
    const block_q8_0 *Wk_q8, const float *bk,
    const block_q8_0 *Wv_q8, const float *bv,
    int seq_len, int in_dim, int q_dim, int kv_dim
);

/* Free GEMM workspace (call from qwen_free) */
void qwen_gemm_workspace_free(void);

/* ========================================================================
 * 2D Convolution (for audio encoder conv stem)
 * ======================================================================== */

/*
 * 2D Convolution: out = conv2d(in, weight, bias)
 * in: [C_in, H, W]
 * weight: [C_out, C_in, kH, kW]
 * bias: [C_out] (can be NULL)
 * out: [C_out, H_out, W_out]
 * H_out = (H + 2*padding - kH) / stride + 1
 * W_out = (W + 2*padding - kW) / stride + 1
 */
void qwen_conv2d(float *out, const float *in, const float *weight, const float *bias,
                 int c_in, int c_out, int h_in, int w_in,
                 int kh, int kw, int stride, int padding);

/* ========================================================================
 * Normalization
 * ======================================================================== */

/* LayerNorm with bias: out = (x - mean) / sqrt(var + eps) * weight + bias */
void qwen_layer_norm(float *out, const float *x, const float *weight, const float *bias,
                     int seq_len, int hidden, float eps);

/* RMS Normalization: out = x / rms(x) * weight */
void qwen_rms_norm(float *out, const float *x, const float *weight,
                   int seq_len, int hidden, float eps);

/* Per-head RMS Normalization for Q/K norms in decoder
 * x: [seq, n_heads, head_dim], weight: [head_dim]
 * Normalizes each head independently */
void qwen_rms_norm_per_head(float *x, const float *weight,
                             int seq_len, int n_heads, int head_dim, float eps);

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

void qwen_gelu(float *x, int n);
void qwen_softmax(float *x, int rows, int cols);
/* out[seq,inter] = SiLU(gate_up[seq,2*inter][:,even]) * gate_up[:,odd] */
void qwen_swiglu_multiply(float *out, const float *gate_up, int seq_len, int intermediate);

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

/*
 * Bidirectional windowed attention (encoder).
 * Q, K, V: [seq, n_heads * head_dim]
 * out: [seq, n_heads * head_dim]
 * window_starts: array of window start positions
 * window_starts[n_windows] = seq (sentinel)
 * All heads have same dimensions (no GQA in encoder).
 */
void qwen_bidirectional_attention(float *out, const float *Q, const float *K,
                                   const float *V, int seq, int n_heads,
                                   int head_dim, float scale,
                                   const int *window_starts, int n_windows);

/*
 * Causal attention with GQA (decoder).
 * Q: [seq_q, n_heads * head_dim]
 * K_fp16: [seq_k, n_kv_heads * head_dim] as FP16 (uint16_t)
 * V_fp16: [seq_k, n_kv_heads * head_dim] as FP16 (uint16_t)
 * q_offset: global position of first query (for causal mask)
 */
void qwen_causal_attention(float *out, const float *Q,
                            const uint16_t *K_fp16, const uint16_t *V_fp16,
                            int seq_q, int seq_k, int n_heads, int n_kv_heads,
                            int head_dim, float scale, int q_offset);

/* ========================================================================
 * Position Embeddings
 * ======================================================================== */

/*
 * Sinusoidal position embeddings (encoder).
 * pe: output [n_pos, d_model]
 * First half = sin, second half = cos.
 */
void qwen_sinusoidal_pe(float *pe, int n_pos, int d_model);

/*
 * NeoX-style RoPE: compute cos/sin for positions.
 * cos_out, sin_out: [seq, head_dim]
 * cos[d] and cos[half+d] are the same (duplicated for full head_dim).
 */
void qwen_compute_rope_neox(float *cos_out, float *sin_out, const int *positions,
                              int seq, int head_dim, float theta);

/*
 * Apply NeoX-style RoPE to Q or K.
 * x: [seq, n_heads * head_dim] (in-place)
 * cos_vals, sin_vals: [seq, head_dim]
 */
void qwen_apply_rope_neox(float *x, const float *cos_vals, const float *sin_vals,
                            int seq, int n_heads, int head_dim);

/* ========================================================================
 * Q4_K Super-Block Weight Operations
 * ======================================================================== */

/* Q4_K linear: y = W_q4k @ x (no bias), seq_len=1 matvec or batched GEMM.
 * W_q4k: [out_dim * (in_dim/QK_K)] block_q4_k blocks.
 * Pre-quantizes x to int8, computes bsums, uses SDOT inner loop. */
void qwen_linear_nobias_q4k(float *y, const float *x, const block_q4_k *W_q4k,
                              int seq_len, int in_dim, int out_dim);

/* Q4_K fused QKV matvec for single-token decoder.
 * Quantizes x once, dispatches Q/K/V to thread pool. */
void qwen_linear_nobias_q4k_qkv(float *q, float *k, float *v, const float *x,
                                  const block_q4_k *Wq_q4k,
                                  const block_q4_k *Wk_q4k,
                                  const block_q4_k *Wv_q4k,
                                  int in_dim, int q_dim, int kv_dim);

/* Q4_K streaming argmax: finds argmax(W_q4k @ x) using Q4_K dot products. */
int qwen_argmax_matvec_q4k(const float *x, const block_q4_k *W_q4k,
                             int in_dim, int out_dim);

/* ========================================================================
 * FP16 Conversion
 * ======================================================================== */

/* Convert FP32 array to FP16 (stored as uint16_t). NEON-accelerated on ARM. */
void qwen_f32_to_f16(uint16_t *dst, const float *src, int n);

/* Convert FP16 (uint16_t) array to FP32. NEON-accelerated on ARM. */
void qwen_f16_to_f32(float *dst, const uint16_t *src, int n);

/* ========================================================================
 * Threading
 * ======================================================================== */

/* Set number of threads for parallel operations (default: 1).
 * Creates a persistent thread pool. Call before inference. */
void qwen_set_threads(int n);

/* Get number of available CPU cores */
int qwen_get_num_cpus(void);

/* Global verbose flag */
extern int qwen_verbose;

#endif /* QWEN_ASR_KERNELS_H */
