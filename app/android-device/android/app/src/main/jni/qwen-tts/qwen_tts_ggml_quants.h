/*
 * qwen_tts_ggml_quants.h - Q4_K / Q8_K quantization (extracted from llama.cpp/ggml)
 *
 * Block structures and function declarations for ggml-style Q4_K_M quantization.
 * ARM NEON (dotprod) + scalar fallback only.
 */

#ifndef QWEN_TTS_GGML_QUANTS_H
#define QWEN_TTS_GGML_QUANTS_H

#include <stdint.h>
#include <stddef.h>

/* ========================================================================
 * Constants
 * ======================================================================== */

#define QK_K       256
#define K_SCALE_SIZE 12

/* ========================================================================
 * FP16 helpers (IEEE 754 half-precision)
 * ======================================================================== */

typedef uint16_t ggml_half;

static inline float ggml_fp16_to_fp32(ggml_half h) {
    /* Bit-exact software conversion (from ggml-impl.h) */
    uint32_t w = (uint32_t)h << 16;
    uint32_t sign = w & 0x80000000u;
    uint32_t two_w = w + w;

    uint32_t exp_offset = 0xE0u << 23;
    float exp_scale;
    {   /* 0x1.0p-112f = 2^-112 */
        uint32_t bits = 0x7800000u;
        __builtin_memcpy(&exp_scale, &bits, sizeof(float));
    }

    float normalized_value;
    {
        uint32_t bits = (two_w >> 4) + exp_offset;
        float fval;
        __builtin_memcpy(&fval, &bits, sizeof(float));
        normalized_value = fval * exp_scale;
    }

    uint32_t magic_mask = 126u << 23;
    float magic_bias = 0.5f;
    float denormalized_value;
    {
        uint32_t bits = (two_w >> 17) | magic_mask;
        float fval;
        __builtin_memcpy(&fval, &bits, sizeof(float));
        denormalized_value = fval - magic_bias;
    }

    uint32_t denormalized_cutoff = 1u << 27;
    uint32_t result;
    {
        uint32_t norm_bits, denorm_bits;
        __builtin_memcpy(&norm_bits, &normalized_value, sizeof(uint32_t));
        __builtin_memcpy(&denorm_bits, &denormalized_value, sizeof(uint32_t));
        result = sign | (two_w < denormalized_cutoff ? denorm_bits : norm_bits);
    }
    float out;
    __builtin_memcpy(&out, &result, sizeof(float));
    return out;
}

static inline ggml_half ggml_fp32_to_fp16(float f) {
    float scale_to_inf, scale_to_zero;
    {
        uint32_t bits = 0x77800000u;
        __builtin_memcpy(&scale_to_inf, &bits, sizeof(float));
        bits = 0x08800000u;
        __builtin_memcpy(&scale_to_zero, &bits, sizeof(float));
    }

    float base = (f >= 0 ? f : -f) * scale_to_inf;
    base = base * scale_to_zero;

    uint32_t w;
    __builtin_memcpy(&w, &f, sizeof(uint32_t));
    uint32_t shl1_w = w + w;
    uint32_t sign = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) bias = 0x71000000u;

    {
        uint32_t bits = (bias >> 1) + 0x07800000u;
        float fval;
        __builtin_memcpy(&fval, &bits, sizeof(float));
        base = fval + base;
    }

    uint32_t bits;
    __builtin_memcpy(&bits, &base, sizeof(uint32_t));
    uint32_t exp_bits = (bits >> 13) & 0x00007C00u;
    uint32_t mantissa_bits = bits & 0x00000FFFu;
    uint32_t nonsign = exp_bits + mantissa_bits;
    return (ggml_half)((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

#define GGML_FP16_TO_FP32(x) ggml_fp16_to_fp32(x)
#define GGML_FP32_TO_FP16(x) ggml_fp32_to_fp16(x)

/* ========================================================================
 * Block structures
 * ======================================================================== */

/*
 * Q4_K: 4-bit quantization with per-block scales and mins.
 * 8 sub-blocks of 32 elements each within a super-block of 256.
 * Effectively 4.5 bits per weight.
 * Size: 2*2 + 12 + 128 = 144 bytes per 256 elements.
 */
typedef struct {
    ggml_half d;                    /* super-block scale for quantized scales */
    ggml_half dmin;                 /* super-block scale for quantized mins */
    uint8_t scales[K_SCALE_SIZE];   /* scales and mins, quantized with 6 bits */
    uint8_t qs[QK_K / 2];          /* 4-bit quants */
} block_q4_K;

/*
 * Q8_K: 8-bit quantization (for intermediate activation quantization and dot products).
 * Size: 4 + 256 + 32 = 292 bytes per 256 elements.
 */
typedef struct {
    float   d;                      /* delta (scale) */
    int8_t  qs[QK_K];              /* quants */
    int16_t bsums[QK_K / 16];      /* sum of quants in groups of 16 */
} block_q8_K;

/* ========================================================================
 * Function declarations
 * ======================================================================== */

/* F32 -> Q4_K quantization (reference/scalar) */
void quantize_row_q4_K_ref(const float *x, block_q4_K *y, int64_t k);

/* F32 -> Q8_K quantization (NEON fast path + scalar fallback) */
void quantize_row_q8_K(const float *x, block_q8_K *y, int64_t k);

/* Q4_K -> F32 dequantization */
void dequantize_row_q4_K(const block_q4_K *x, float *y, int64_t k);

/* Dot product: Q4_K * Q8_K (NEON dotprod + scalar fallback) */
void vec_dot_q4_K_q8_K(int n, float *s, const block_q4_K *x, const block_q8_K *y);

#endif /* QWEN_TTS_GGML_QUANTS_H */
