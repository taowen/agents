/*
 * qwen_tts_quant.h - Q4_K quantization block definition for Qwen3-TTS
 */

#ifndef QWEN_TTS_QUANT_H
#define QWEN_TTS_QUANT_H

#include <stdint.h>

#define QK_K 256
#define Q4K_NUM_SUBS 8   /* QK_K / 32 */

typedef struct block_q4_k {
    float d;               /* 4B: super-block scale */
    float dmin;            /* 4B: super-block min (asymmetric offset) */
    uint8_t scales[8];     /* 8B: per-sub-group integer scales (0-255) */
    uint8_t mins[8];       /* 8B: per-sub-group integer mins (0-255) */
    uint8_t qs[128];       /* 128B: 256 unsigned int4 [0,15] packed nibbles */
} block_q4_k;              /* 152 bytes / 256 elements */

#endif /* QWEN_TTS_QUANT_H */
