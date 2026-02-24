/*
 * qwen_asr_internal.h - Shared declarations for split qwen_asr*.c files
 */

#ifndef QWEN_ASR_INTERNAL_H
#define QWEN_ASR_INTERNAL_H

#include "qwen_asr.h"
#include "qwen_asr_tokenizer.h"

/*
 * Prompt format constants (shared across transcribe/stream files)
 *   PREFIX_HEAD: [<|im_start|>, "system", "\n"]
 *   PREFIX_TAIL: [<|im_end|>, "\n", <|im_start|>, "user", "\n", <|audio_start|>]
 *   SUFFIX_BASE: [<|audio_end|>, <|im_end|>, "\n", <|im_start|>, "assistant", "\n"]
 */
static const int PROMPT_PREFIX_HEAD[] = {
    151644, 8948, 198
};
static const int PROMPT_PREFIX_TAIL[] = {
    151645, 198, 151644, 872, 198, 151669
};
static const int PROMPT_SUFFIX_BASE[] = {
    151670, 151645, 198, 151644, 77091, 198
};
#define PREFIX_HEAD_LEN 3
#define PREFIX_TAIL_LEN 6
#define SUFFIX_BASE_LEN 6

/* Shared helper functions (defined in qwen_asr.c) */
void tok_embed_bf16_to_f32(float *dst, const uint16_t *tok_emb_bf16, int token_id, int dim);
double get_time_ms(void);
int prepare_prompt_tokens(qwen_ctx_t *ctx, qwen_tokenizer_t *tokenizer);

/* Shared transcription helpers (defined in qwen_asr_transcribe.c) */
float *compact_silence(const float *samples, int n_samples, int *out_samples);
char *transcribe_segment(qwen_ctx_t *ctx, const float *samples, int n_samples,
                         qwen_tokenizer_t *tokenizer, const int *past_tokens,
                         int n_past_tokens, int *out_text_tokens);

#endif /* QWEN_ASR_INTERNAL_H */
