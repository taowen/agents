/*
 * qwen_tts_quant.h - Q4_K Weight Quantization for Qwen3-TTS
 *
 * Q4_K_M strategy (ggml standard):
 *   Talker:     all weights -> Q4_K (4.5 bpw, 256-element super-blocks)
 *   Sub-talker: all weights -> Q4_K
 */

#ifndef QWEN_TTS_QUANT_H
#define QWEN_TTS_QUANT_H

#include <stdint.h>
#include <stddef.h>
#include "qwen_tts_ggml_quants.h"

/* ========================================================================
 * Q4_K matvec kernels
 * ======================================================================== */

/* Q4_K matvec: out[rows] = A_q4k[rows * cols/QK_K blocks] @ x[cols]
 * x is quantized on-the-fly to Q8_K, then vec_dot is called per row. */
void kernel_matvec_q4k(float *out, const block_q4_K *A_q4k,
                       const float *x, int rows, int cols);

/* ========================================================================
 * Fused SwiGLU kernel (Q4_K)
 * ======================================================================== */

/* Fused SwiGLU with Q4_K gate+up weights:
 * gate_up_q4k has [2*intermediate] rows of [hidden] cols.
 * First half = gate, second half = up.
 * out[intermediate] = SiLU(gate @ x) * (up @ x) */
void kernel_swiglu_matvec_q4k(float *out, const block_q4_K *gate_up_q4k,
                               const float *x, int intermediate, int hidden);

/* ========================================================================
 * BF16 -> Q4_K conversion
 * ======================================================================== */

/* BF16 -> Q4_K quantization.
 * bf16[rows * cols] -> out_q4k[rows * (cols/QK_K) blocks]
 * cols must be divisible by QK_K (256). */
void quantize_bf16_to_q4k(const uint16_t *bf16, int rows, int cols,
                           block_q4_K **out_q4k);

/* ========================================================================
 * Weight cache API
 * ======================================================================== */

/* Forward-declare ctx type (defined in qwen_tts.h) */
struct qwen_tts_ctx;

/* Save pre-quantized weights to .qcache file */
int save_quantized_cache(struct qwen_tts_ctx *ctx);

/* Load pre-quantized weights from .qcache file; returns 0 on success */
int load_quantized_cache(struct qwen_tts_ctx *ctx);

#endif /* QWEN_TTS_QUANT_H */
