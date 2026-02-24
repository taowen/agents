/*
 * qwen_asr_quant.h - Quantization formats for Qwen3-ASR
 *
 * Q8_0: 32 weights per block, symmetric quantization.
 *   weight[i] = scale * qs[i]
 *   Memory: 36 bytes / 32 weights = 1.125 bytes/weight
 *
 * Q4_K: 256 weights per super-block, asymmetric (scale + min).
 *   weight[i] = d * q[i] - dmin * min[sub-block]
 *   Memory: 144 bytes / 256 weights = 0.5625 bytes/weight
 *
 * Q8_K: 256 weights per super-block, symmetric (runtime activation quant).
 *   Memory: 292 bytes / 256 weights
 */

#ifndef QWEN_ASR_QUANT_H
#define QWEN_ASR_QUANT_H

#include <stddef.h>
#include <stdint.h>

/* ======== Q8_0 (encoder weights, 32-element blocks) ======== */

#define QK8_0 32  /* block size */

typedef struct {
    float scale;           /* shared scale factor */
    int8_t qs[QK8_0];     /* quantized values */
} block_q8_0;             /* 36 bytes total */

/* ======== Q4_K / Q8_K (decoder weights, 256-element super-blocks) ======== */

#define QK_K 256           /* super-block size */
#define K_SCALE_SIZE 12    /* packed scales/mins bytes */

typedef uint16_t ggml_half;

/* Q4_K: 4-bit quantization with per-sub-block scales and mins.
 * 8 sub-blocks of 32 elements each. weight = d * q - dmin * min.
 * 144 bytes / 256 weights = 4.5 bits/weight effective. */
typedef struct {
    ggml_half d;              /* super-block scale (fp16) */
    ggml_half dmin;           /* super-block min (fp16) */
    uint8_t scales[K_SCALE_SIZE]; /* 6-bit packed scales/mins for sub-blocks */
    uint8_t qs[QK_K / 2];    /* 4-bit packed quants (128 bytes) */
} block_q4_K;                /* 144 bytes total */

/* Q8_K: 8-bit quantization for runtime activation quantization.
 * Used as the activation format when computing Q4_K × Q8_K dot products. */
typedef struct {
    float d;                  /* delta (scale) */
    int8_t qs[QK_K];         /* quantized values */
    int16_t bsums[QK_K / 16]; /* sum of quants in groups of 16 */
} block_q8_K;                /* 292 bytes total */

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

/* Quantize n float32 values to Q8_K super-blocks (256-element blocks).
 * n must be a multiple of QK_K.
 * dst must have n/QK_K blocks allocated. */
void quantize_f32_to_q8_K(const float *src, block_q8_K *dst, int n);

/* FP16 ↔ FP32 conversion helpers */
static inline float fp16_to_fp32(ggml_half h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp  = ((uint32_t)h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x03FFu;
    uint32_t f32;
    if (exp == 0) {
        if (mant == 0) { f32 = sign; }
        else {
            exp = 1;
            while (!(mant & 0x0400u)) { mant <<= 1; exp--; }
            mant &= 0x03FFu;
            f32 = sign | ((127 - 15 + exp) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f32 = sign | 0x7F800000u | (mant << 13);
    } else {
        f32 = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    __builtin_memcpy(&result, &f32, sizeof(float));
    return result;
}

static inline ggml_half fp32_to_fp16(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = ((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;
    if (exp <= 0) return (ggml_half)sign;
    if (exp >= 31) return (ggml_half)(sign | 0x7C00u);
    return (ggml_half)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

#endif /* QWEN_ASR_QUANT_H */
