/*
 * qwen_tts_quant.h - INT8/Q4_K Weight Quantization for Qwen3-TTS
 *
 * Q4_K_M strategy:
 *   Talker:     QKV/gate_up -> Q4_K, wo/down -> INT8 (sensitive layers)
 *   Sub-talker: all -> Q4_K (lower precision requirements)
 */

#ifndef QWEN_TTS_QUANT_H
#define QWEN_TTS_QUANT_H

#include <stdint.h>
#include <stddef.h>

/* ========================================================================
 * Q4_K super-block quantization format
 * 256 elements per block, 8 sub-groups of 32
 * Dequant: weight ~ d * scales[g] * q - dmin * mins[g], q in [0,15]
 * ======================================================================== */

#define QK_K 256
#define Q4K_NUM_SUBS 8

typedef struct block_q4_k {
    float d;               /* super-block scale */
    float dmin;            /* super-block min offset */
    uint8_t scales[8];     /* per-sub-group scales */
    uint8_t mins[8];       /* per-sub-group mins */
    uint8_t qs[128];       /* 256 packed nibbles (lo=even, hi=odd) */
} block_q4_k;              /* 152 bytes / 256 elements */

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
 * Q4_K kernels
 * ======================================================================== */

/* Q4_K matvec: out[rows] = Q4K_weights[rows,cols] @ x[cols] */
void kernel_matvec_q4k(float *out, const block_q4_k *blocks,
                        const float *x, int rows, int cols);

/* ========================================================================
 * Fused SwiGLU kernels (quantized)
 * ======================================================================== */

/* Fused SwiGLU with INT8 gate+up weights */
void kernel_swiglu_matvec_int8(float *out, const int8_t *gate_up_int8,
                                const float *scales, const float *x,
                                int intermediate, int hidden);

/* Fused SwiGLU with Q4_K gate+up weights */
void kernel_swiglu_matvec_q4k(float *out, const block_q4_k *gate_up_blocks,
                                const float *x, int intermediate, int hidden);

/* ========================================================================
 * BF16 -> quantized format conversion
 * ======================================================================== */

/* BF16 -> INT8 per-row symmetric quantization */
void quantize_bf16_to_int8(const uint16_t *bf16, int rows, int cols,
                             int8_t **out_int8, float **out_scales);

/* BF16 -> Q4_K super-block quantization */
void quantize_bf16_to_q4k(const uint16_t *bf16, int rows, int cols,
                            block_q4_k **out_blocks);

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
