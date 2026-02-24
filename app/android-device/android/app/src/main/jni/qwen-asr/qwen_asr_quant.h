/*
 * qwen_asr_quant.h - Q8_0 quantization format for Qwen3-ASR
 *
 * Q8_0: 32 weights per block, symmetric quantization.
 * Each block stores a float scale and 32 int8 quantized values.
 * weight[i] = scale * qs[i]
 *
 * Memory: 36 bytes per 32 weights = 1.125 bytes/weight
 * vs FP32: 4 bytes/weight (3.56x compression)
 * vs BF16: 2 bytes/weight (1.78x compression)
 */

#ifndef QWEN_ASR_QUANT_H
#define QWEN_ASR_QUANT_H

#include <stddef.h>
#include <stdint.h>

#define QK8_0 32  /* block size */

typedef struct {
    float scale;           /* shared scale factor */
    int8_t qs[QK8_0];     /* quantized values */
} block_q8_0;             /* 36 bytes total */

/* ========================================================================
 * Q4_K Super-Block Quantization Format
 *
 * 256 weights per super-block, 8 sub-groups of 32.
 * Two-level quantization: super-block scale/min (float) + sub-group
 * integer scales/mins (uint8). Unsigned 4-bit quantized values [0,15].
 *
 * Dequantization: weight ≈ d * scales[g] * q - dmin * mins[g]
 *
 * Memory: 152 bytes per 256 weights = 0.59375 bytes/weight
 * vs Q8_0: 1.125 bytes/weight (1.89x compression)
 * ======================================================================== */

#define QK_K 256
#define Q4K_NUM_SUBS 8   /* QK_K / 32 */

typedef struct {
    float d;               /* 4B: super-block scale */
    float dmin;            /* 4B: super-block min (asymmetric offset) */
    uint8_t scales[8];     /* 8B: per-sub-group integer scales (0-255) */
    uint8_t mins[8];       /* 8B: per-sub-group integer mins (0-255) */
    uint8_t qs[128];       /* 128B: 256 unsigned int4 [0,15] packed nibbles */
} block_q4_k;              /* 152 bytes / 256 elements */

/* Quantize n float32 values to Q8_0 blocks.
 * n must be a multiple of QK8_0.
 * dst must have n/QK8_0 blocks allocated. */
void quantize_f32_to_q8_0(const float *src, block_q8_0 *dst, int n);

/* Quantize n bfloat16 values (stored as uint16_t) to Q8_0 blocks.
 * n must be a multiple of QK8_0.
 * dst must have n/QK8_0 blocks allocated. */
void quantize_bf16_to_q8_0(const uint16_t *src, block_q8_0 *dst, int n);

/* Dequantize Q8_0 blocks back to float32 (for verification).
 * n is the number of float values (must be multiple of QK8_0).
 * dst must have n floats allocated. */
void dequantize_q8_0_to_f32(const block_q8_0 *src, float *dst, int n);

/* Quantize n bfloat16 values (stored as uint16_t) to Q4_K super-blocks.
 * rows × cols must have cols divisible by QK_K (256).
 * *out_blocks is malloc'd and filled; caller must free.
 * On error (cols not divisible by QK_K, or OOM), *out_blocks is set to NULL. */
void quantize_bf16_to_q4k(const uint16_t *bf16, int rows, int cols,
                            block_q4_k **out_blocks);

/* Quantize X[M, K] row-wise to Q8_0 in transposed-block layout.
 * Output: X_q8t[n_blocks * M_pad] where n_blocks = K / QK8_0.
 * X_q8t[b * M_pad + m] = Q8_0 block for row m, K-block b.
 * Rows m >= M are zero-filled (scale=0, qs=0).
 * K must be a multiple of QK8_0. M_pad must be >= M, multiple of 4. */
void quantize_f32_rows_transpose_q8(
    const float *X,           /* [M, K] input */
    block_q8_0 *X_q8t,       /* [n_blocks * M_pad] output */
    int M, int K, int M_pad
);

#endif /* QWEN_ASR_QUANT_H */
