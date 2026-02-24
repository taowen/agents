/*
 * qwen_tts_internal.h - Internal function declarations for Qwen3-TTS
 *
 * Cross-module declarations shared between qwen_tts.c, qwen_tts_talker.c,
 * and qwen_tts_codec.c.  Not part of the public API.
 */

#ifndef QWEN_TTS_INTERNAL_H
#define QWEN_TTS_INTERNAL_H

#include "qwen_tts.h"

/* ========================================================================
 * Timing helper (shared across translation units)
 * ======================================================================== */

double qwen_tts_time_ms(void);

/* ========================================================================
 * Talker
 * ======================================================================== */

/* Talker forward pass - prefill (multiple tokens) */
void qwen_tts_talker_prefill(qwen_tts_ctx_t *ctx, const float *input_embeds, int seq_len);

/* Talker forward pass - single token, returns logits */
void qwen_tts_talker_forward(qwen_tts_ctx_t *ctx, const float *input_embed, float *logits);

/* Sub-talker: generate remaining code groups given talker hidden + first codebook token */
void qwen_tts_subtalker_generate(
    qwen_tts_ctx_t *ctx,
    const float *talker_hidden,  /* [hidden] from talker's last hidden state */
    int first_code,              /* first codebook token from talker */
    int *out_codes               /* [num_code_groups] output - first slot = first_code */
);

/* ========================================================================
 * Codec decoder
 * ======================================================================== */

/* Codec decoder: convert codec tokens to waveform */
float *qwen_tts_codec_decode(
    qwen_tts_ctx_t *ctx,
    const int *codes,           /* [time_steps, num_quantizers] */
    int time_steps,
    int *out_samples
);

/* ========================================================================
 * Incremental Codec Decode State
 * ======================================================================== */

typedef struct {
    /* Pre-conv: CausalConv1d(512->1024, k=3, d=1), state_len = (3-1)*1 = 2 */
    float *pre_conv_state;      /* [512, 2] = 1024 floats */

    /* Codec transformer: position counter (KV cache uses ctx->codec_kv_*) */
    int transformer_pos;

    /* Upsample ConvNeXt dwconv states: k=7, d=1, groups=dim, state_len=6 */
    float *upsample_cn_state[2]; /* each [1024, 6] = 6144 floats */

    /* Vocoder pre-conv: CausalConv1d(1024->1536, k=7, d=1), state_len=6 */
    float *voc_preconv_state;   /* [1024, 6] = 6144 floats */

    /* Vocoder blocks (4 blocks) */
    struct {
        float *transconv_overlap;   /* [out_dim, K-stride] */
        float *ru_conv1_state[3];   /* [dim, (K-1)*dilation] for each ResUnit */
    } voc_blocks[4];

    /* Final conv: CausalConv1d(96->1, k=7, d=1), state_len=6 */
    float *final_conv_state;    /* [96, 6] = 576 floats */

    int n_processed;            /* tokens processed so far */
} qwen_tts_codec_stream_state_t;

/* Allocate and initialize incremental decode state (all buffers zeroed) */
qwen_tts_codec_stream_state_t *qwen_tts_codec_stream_init(qwen_tts_ctx_t *ctx);

/* Free incremental decode state */
void qwen_tts_codec_stream_free(qwen_tts_codec_stream_state_t *state);

/* Decode a single codec token incrementally; returns malloc'd PCM (caller frees).
 * codes: [num_quantizers] for 1 timestep.
 * *out_samples is set to 1920 on success. */
float *qwen_tts_codec_decode_step(
    qwen_tts_ctx_t *ctx,
    qwen_tts_codec_stream_state_t *state,
    const int *codes,
    int *out_samples
);

/* Verify incremental decode matches batch decode. Returns max absolute diff.
 * If max_diff < 1e-4, incremental decode is correct. */
int qwen_tts_codec_verify_incremental(qwen_tts_ctx_t *ctx,
                                        const int *all_codes,
                                        int n_tokens);

#endif /* QWEN_TTS_INTERNAL_H */
