/*
 * qwen_tts_quant.h - Q8_0 quantization format for Qwen3-TTS
 *
 * Q8_0: 32 weights per block, symmetric quantization.
 * Each block stores a float scale and 32 int8 quantized values.
 * weight[i] = scale * qs[i]
 *
 * Memory: 36 bytes per 32 weights = 1.125 bytes/weight
 */

#ifndef QWEN_TTS_QUANT_H
#define QWEN_TTS_QUANT_H

#include <stddef.h>
#include <stdint.h>

#define QK8_0 32  /* block size */

typedef struct {
    float scale;           /* shared scale factor */
    int8_t qs[QK8_0];     /* quantized values */
} block_q8_0;             /* 36 bytes total */

/* Quantize n float32 values to Q8_0 blocks.
 * n must be a multiple of QK8_0.
 * dst must have n/QK8_0 blocks allocated. */
void quantize_f32_to_q8_0(const float *src, block_q8_0 *dst, int n);

/* Quantize n bfloat16 values (stored as uint16_t) to Q8_0 blocks.
 * n must be a multiple of QK8_0.
 * dst must have n/QK8_0 blocks allocated. */
void quantize_bf16_to_q8_0(const uint16_t *src, block_q8_0 *dst, int n);

#endif /* QWEN_TTS_QUANT_H */
