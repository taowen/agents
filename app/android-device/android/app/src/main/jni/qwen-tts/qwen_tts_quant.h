/*
 * qwen_tts_quant.h - INT8 Weight Quantization for Qwen3-TTS
 *
 * INT8-only strategy:
 *   Talker:     all weights -> INT8 (8-bit symmetric per-row)
 *   Sub-talker: all weights -> INT8
 */

#ifndef QWEN_TTS_QUANT_H
#define QWEN_TTS_QUANT_H

#include <stdint.h>
#include <stddef.h>

/* ========================================================================
 * INT8 kernels
 * ======================================================================== */

/* INT8 matvec: out[rows] = A_int8[rows,cols] @ x[cols], with per-row scales */
void kernel_matvec_int8(float *out, const int8_t *A_int8, const float *scales,
                         const float *x, int rows, int cols);

/* Quantize float vector x to int8 (symmetric, single global scale) */
void kernel_quantize_x_int8(const float *x, int cols,
                              int8_t *x_int8_out, float *x_scale_out);

/* INT8 matvec with pre-quantized x vector */
void kernel_matvec_int8_pq(float *out, const int8_t *A_int8, const float *scales,
                             const int8_t *x_int8, float x_scale, int rows, int cols);

/* ========================================================================
 * Fused SwiGLU kernels (quantized)
 * ======================================================================== */

/* Fused SwiGLU with INT8 gate+up weights */
void kernel_swiglu_matvec_int8(float *out, const int8_t *gate_up_int8,
                                const float *scales, const float *x,
                                int intermediate, int hidden);

/* ========================================================================
 * BF16 -> quantized format conversion
 * ======================================================================== */

/* BF16 -> INT8 per-row symmetric quantization */
void quantize_bf16_to_int8(const uint16_t *bf16, int rows, int cols,
                             int8_t **out_int8, float **out_scales);

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
