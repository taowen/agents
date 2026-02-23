/*
 * qwen_tts_generate.c - Generation logic for Qwen3-TTS
 *
 * Contains:
 *   - qwen_tts_generate()        — batch generation (prefill + AR + codec decode)
 *   - qwen_tts_generate_stream() — streaming generation with chunked codec decode
 */

#include "qwen_tts.h"
#include "qwen_tts_kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/time.h>

/* ========================================================================
 * Timing helpers
 * ======================================================================== */

static double time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ========================================================================
 * Text projection helper
 *
 * Projects text embeddings: text_hidden -> text_hidden (SiLU) -> hidden
 * ======================================================================== */

static void text_projection(qwen_tts_ctx_t *ctx, const float *text_embed,
                             float *out, int text_hidden, int hidden) {
    float *fc1_out = (float *)malloc(text_hidden * sizeof(float));
    kernel_matvec_bf16(fc1_out, ctx->talker.text_proj_fc1_bf16, text_embed, text_hidden, text_hidden);
    if (ctx->talker.text_proj_fc1_bias)
        kernel_add_inplace(fc1_out, ctx->talker.text_proj_fc1_bias, text_hidden);
    kernel_silu_inplace(fc1_out, text_hidden);
    kernel_matvec_bf16(out, ctx->talker.text_proj_fc2_bf16, fc1_out, hidden, text_hidden);
    if (ctx->talker.text_proj_fc2_bias)
        kernel_add_inplace(out, ctx->talker.text_proj_fc2_bias, hidden);
    free(fc1_out);
}

/* ========================================================================
 * Embed a text token: text_embedding -> text_projection
 * ======================================================================== */

static void embed_text_token(qwen_tts_ctx_t *ctx, int token_id, float *out) {
    int text_hidden = ctx->config.talker_text_hidden;
    int hidden = ctx->config.talker_hidden;
    float *text_embed = (float *)malloc(text_hidden * sizeof(float));
    kernel_bf16_to_f32(text_embed, ctx->talker.text_embedding_bf16 + (size_t)token_id * text_hidden, text_hidden);
    text_projection(ctx, text_embed, out, text_hidden, hidden);
    free(text_embed);
}

/* ========================================================================
 * Embed a codec token: lookup from codec_embedding
 * ======================================================================== */

static void embed_codec_token(qwen_tts_ctx_t *ctx, int token_id, float *out) {
    int hidden = ctx->config.talker_hidden;
    kernel_bf16_to_f32(out, ctx->talker.codec_embedding_bf16 + (size_t)token_id * hidden, hidden);
}

/* ========================================================================
 * Generate - CustomVoice Mode
 *
 * Builds the embedding sequence, runs talker prefill + autoregressive
 * generation, then decodes codec tokens to waveform.
 *
 * Input text_token_ids: pre-tokenized text in chat format:
 *   [im_start, assistant_id, \n, TEXT..., im_end, \n, im_start, assistant_id, \n]
 * ======================================================================== */

float *qwen_tts_generate(
    qwen_tts_ctx_t *ctx,
    const char *text,         /* unused for now, pass NULL */
    const char *speaker,
    const char *language,
    int *out_samples
) {
    if (!ctx || !out_samples) {
        return NULL;
    }
    *out_samples = 0;

    /* For now, we require pre-tokenized IDs stored in ctx.
     * This function implements the full generate flow with hardcoded token IDs.
     *
     * TODO: Add BPE tokenizer for direct text input.
     *
     * For demonstration, we accept text as a comma-separated list of token IDs.
     */

    /* Parse text as comma-separated token IDs */
    int *text_tokens = NULL;
    int n_text_tokens = 0;

    if (text) {
        /* Count tokens */
        const char *p = text;
        int count = 1;
        while (*p) { if (*p == ',') count++; p++; }

        text_tokens = (int *)malloc(count * sizeof(int));
        p = text;
        while (*p && n_text_tokens < count) {
            while (*p == ' ' || *p == ',') p++;
            if (*p == '\0') break;
            char *endp = NULL;
            long v = strtol(p, &endp, 10);
            if (endp == p) {
                fprintf(stderr, "Error: invalid token ID near '%s'\n", p);
                free(text_tokens);
                *out_samples = 0;
                return NULL;
            }
            text_tokens[n_text_tokens++] = (int)v;
            p = endp;
        }
    }

    if (n_text_tokens < 8) {
        fprintf(stderr, "Error: need at least 8 text tokens (chat template format)\n");
        free(text_tokens);
        *out_samples = 0;
        return NULL;
    }

    qwen_tts_config_t *cfg = &ctx->config;
    int hidden = cfg->talker_hidden;
    int num_groups = cfg->num_code_groups;

    double t_start = time_ms();

    /* ---- Look up speaker and language IDs ---- */
    int speaker_codec_id = -1;
    if (speaker && strlen(speaker) > 0) {
        for (int i = 0; i < cfg->n_speakers; i++) {
            if (strcasecmp(cfg->speaker_names[i], speaker) == 0) {
                speaker_codec_id = cfg->speaker_ids[i];
                break;
            }
        }
        if (speaker_codec_id < 0) {
            fprintf(stderr, "Warning: speaker '%s' not found, using no speaker embedding\n", speaker);
        }
    }

    int language_codec_id = -1;
    if (language && strlen(language) > 0 && strcasecmp(language, "auto") != 0) {
        for (int i = 0; i < cfg->n_languages; i++) {
            if (strcasecmp(cfg->language_names[i], language) == 0) {
                language_codec_id = cfg->language_ids[i];
                break;
            }
        }
        if (language_codec_id < 0) {
            fprintf(stderr, "Warning: language '%s' not found\n", language);
        }
    }

    /* ---- Build prefix embedding sequence ---- */

    /* Input format: [im_start, assistant, \n, TEXT..., im_end, \n, im_start, assistant, \n]
     * Positions: [0:3] = role, [3:-5] = content text, [-5:] = trailing template
     *
     * role_tokens = text_proj(text_embed(input[0:3]))   -- 3 tokens
     * first_text = text_proj(text_embed(input[3]))       -- 1 token
     * remaining_text = text_proj(text_embed(input[4:-5])) -- variable
     * trailing_text = [remaining_text..., tts_eos_embed]
     */

    int n_content = n_text_tokens - 8;  /* text tokens minus template (3 + 5) */
    if (n_content < 0) n_content = 0;

    /* Build codec prefix tokens */
    int codec_prefix[8];
    int n_codec_prefix = 0;

    if (language_codec_id < 0) {
        /* No language specified: nothink, think_bos, think_eos */
        codec_prefix[n_codec_prefix++] = cfg->codec_nothink_id;
        codec_prefix[n_codec_prefix++] = cfg->codec_think_bos_id;
        codec_prefix[n_codec_prefix++] = cfg->codec_think_eos_id;
    } else {
        /* Language specified: think, think_bos, language_id, think_eos */
        codec_prefix[n_codec_prefix++] = cfg->codec_think_id;
        codec_prefix[n_codec_prefix++] = cfg->codec_think_bos_id;
        codec_prefix[n_codec_prefix++] = language_codec_id;
        codec_prefix[n_codec_prefix++] = cfg->codec_think_eos_id;
    }

    if (speaker_codec_id >= 0) {
        codec_prefix[n_codec_prefix++] = speaker_codec_id;
    }
    codec_prefix[n_codec_prefix++] = cfg->codec_pad_id;
    codec_prefix[n_codec_prefix++] = cfg->codec_bos_id;

    /* Total prefill length:
     *   3 (role) + (n_codec_prefix - 1) (tts_pad/bos + codec without last) + 1 (first_text + codec_bos)
     *   = 3 + n_codec_prefix
     */
    int prefill_len = 3 + n_codec_prefix;

    float *input_embeds = (float *)calloc((size_t)prefill_len * hidden, sizeof(float));

    /* 1. Role tokens: text_proj(text_embed(role[0:3])) */
    for (int i = 0; i < 3; i++) {
        embed_text_token(ctx, text_tokens[i], input_embeds + i * hidden);
    }

    /* 2. Pad/bos section:
     *   Positions 3..3+n_codec_prefix-2:
     *     text side: tts_pad_embed (for all but last), tts_bos_embed (for the last in this group)
     *     codec side: codec_embed(codec_prefix[0..n_codec_prefix-2])
     *     These are added element-wise.
     */
    float *tts_pad_proj = (float *)malloc(hidden * sizeof(float));
    float *tts_bos_proj = (float *)malloc(hidden * sizeof(float));
    float *tts_eos_proj = (float *)malloc(hidden * sizeof(float));
    float *codec_emb_tmp = (float *)malloc(hidden * sizeof(float));
    embed_text_token(ctx, QWEN_TTS_TOKEN_TTS_PAD, tts_pad_proj);
    embed_text_token(ctx, QWEN_TTS_TOKEN_TTS_BOS, tts_bos_proj);
    embed_text_token(ctx, QWEN_TTS_TOKEN_TTS_EOS, tts_eos_proj);

    for (int i = 0; i < n_codec_prefix - 1; i++) {
        float *dst = input_embeds + (3 + i) * hidden;
        /* Text part: tts_pad for all except the last which gets tts_bos */
        if (i < n_codec_prefix - 2) {
            memcpy(dst, tts_pad_proj, hidden * sizeof(float));
        } else {
            memcpy(dst, tts_bos_proj, hidden * sizeof(float));
        }
        /* Codec part: add codec_embed(codec_prefix[i]) */
        embed_codec_token(ctx, codec_prefix[i], codec_emb_tmp);
        kernel_add_inplace(dst, codec_emb_tmp, hidden);
    }

    /* 3. First text token + codec_bos:
     *   text_proj(text_embed(text_tokens[3])) + codec_embed(codec_bos)
     */
    {
        int pos = 3 + n_codec_prefix - 1;
        float *dst = input_embeds + pos * hidden;
        embed_text_token(ctx, text_tokens[3], dst);
        embed_codec_token(ctx, cfg->codec_bos_id, codec_emb_tmp);
        kernel_add_inplace(dst, codec_emb_tmp, hidden);
    }
    free(codec_emb_tmp);

    /* Build trailing text embeddings (remaining text + tts_eos) */
    int n_trailing = (n_text_tokens - 4 - 5) + 1;  /* remaining text + eos */
    if (n_trailing < 1) n_trailing = 1;
    float *trailing_text = (float *)calloc((size_t)n_trailing * hidden, sizeof(float));
    for (int i = 0; i < n_trailing - 1; i++) {
        embed_text_token(ctx, text_tokens[4 + i], trailing_text + i * hidden);
    }
    memcpy(trailing_text + (n_trailing - 1) * hidden, tts_eos_proj, hidden * sizeof(float));

    /* ---- Prefill ---- */
    double t_prefill = time_ms();

    /* Reset KV cache */
    ctx->talker_kv_len = 0;
    qwen_tts_talker_prefill(ctx, input_embeds, prefill_len);

    double t_prefill_done = time_ms();
    if (qwen_tts_verbose >= 1)
        fprintf(stderr, "Prefill: %d tokens in %.1f ms\n", prefill_len, t_prefill_done - t_prefill);

    free(input_embeds);

    /* ---- Autoregressive generation ---- */
    int fixed_tokens = ctx->fixed_codec_tokens > 0 ? ctx->fixed_codec_tokens : 0;
    int max_tokens = fixed_tokens > 0 ? fixed_tokens : ctx->max_new_tokens;
    int *all_codes = (int *)calloc((size_t)max_tokens * num_groups, sizeof(int));
    int *generated_tokens = (int *)calloc(max_tokens, sizeof(int));
    int n_generated = 0;
    int stop_reason = 0; /* 1: eos, 2: max_tokens */
    int stop_step = max_tokens;

    float *logits = (float *)malloc(cfg->talker_vocab_size * sizeof(float));
    float *next_embed = (float *)malloc(hidden * sizeof(float));
    float *emb_tmp = (float *)malloc(hidden * sizeof(float));
    float rng_state = (float)ctx->sample_seed;

    /* Suppress tokens: [vocab-1024, vocab) except EOS */
    int suppress_start = cfg->talker_vocab_size - 1024;
    int *suppress_tokens = (int *)malloc(1024 * sizeof(int));
    int n_suppress = 0;
    for (int i = suppress_start; i < cfg->talker_vocab_size; i++) {
        if (i != cfg->codec_eos_id) suppress_tokens[n_suppress++] = i;
    }

    double t_gen = time_ms();

    for (int step = 0; step < max_tokens; step++) {
        /* If this is the first step after prefill, use the last hidden state from prefill */
        if (step == 0) {
            /* The prefill already stored the last hidden state; we need to compute logits.
             * Actually, the talker_forward function expects an input embedding.
             * After prefill, we need the next input embedding which is already prepared
             * as the prefill's output representation. Let's handle this:
             *
             * Actually, in the Python code, after prefill, the first generation step
             * uses the last hidden state to produce logits directly.
             * Then subsequent steps use the sampled token's embedding.
             *
             * For simplicity, let's compute logits from the last hidden state: */
            kernel_matvec_bf16(logits, ctx->talker.codec_head_bf16, ctx->tk_x,
                               cfg->talker_vocab_size, hidden);
        } else {
            /* Forward pass with the next embedding */
            qwen_tts_talker_forward(ctx, next_embed, logits);
        }

        /* Apply suppress tokens */
        for (int i = 0; i < n_suppress; i++) {
            logits[suppress_tokens[i]] = -1e9f;
        }

        /* Apply repetition penalty */
        kernel_apply_repetition_penalty(logits, generated_tokens, n_generated,
                                        cfg->talker_vocab_size, ctx->repetition_penalty);

        /* Sample */
        int token = kernel_sample_top_k(logits, cfg->talker_vocab_size, ctx->top_k,
                                         ctx->top_p, ctx->temperature, &rng_state);

        if (fixed_tokens > 0 && token == cfg->codec_eos_id && n_generated < fixed_tokens) {
            float eos_logit = logits[cfg->codec_eos_id];
            logits[cfg->codec_eos_id] = -1e9f;
            token = kernel_sample_top_k(logits, cfg->talker_vocab_size, ctx->top_k,
                                        ctx->top_p, ctx->temperature, &rng_state);
            logits[cfg->codec_eos_id] = eos_logit;
        }

        /* Check for EOS */
        if (fixed_tokens == 0 && token == cfg->codec_eos_id) {
            stop_reason = 1;
            stop_step = step;
            if (qwen_tts_verbose >= 1)
                fprintf(stderr, "EOS at step %d\n", step);
            break;
        }

        generated_tokens[n_generated] = token;

        /* Generate remaining code groups via sub-talker */
        int codes[QWEN_TTS_NUM_CODE_GROUPS];
        qwen_tts_subtalker_generate(ctx, ctx->tk_x, token, codes);

        /* Store all codes */
        memcpy(all_codes + n_generated * num_groups, codes, num_groups * sizeof(int));
        n_generated++;

        /* Build next input embedding:
         *   sum of all 32 group embeddings + trailing_text[step] or tts_pad_embed
         */
        memset(next_embed, 0, hidden * sizeof(float));

        /* Group 0: talker codec embedding */
        embed_codec_token(ctx, token, emb_tmp);
        kernel_add_inplace(next_embed, emb_tmp, hidden);

        /* Groups 1-31: sub-talker codec embeddings */
        for (int g = 1; g < num_groups; g++) {
            int emb_dim = hidden;  /* sub-talker embeddings have talker_hidden_size dim */
            kernel_bf16_to_f32(emb_tmp, ctx->subtalker.codec_embeddings_bf16[g - 1] +
                               (size_t)codes[g] * emb_dim, emb_dim);
            kernel_add_inplace(next_embed, emb_tmp, hidden);
        }
        /* Add trailing text embedding */
        if (step < n_trailing) {
            kernel_add_inplace(next_embed, trailing_text + step * hidden, hidden);
        } else {
            kernel_add_inplace(next_embed, tts_pad_proj, hidden);
        }

        /* Progress callback */
        if (ctx->progress_cb) {
            ctx->progress_cb(step + 1, max_tokens, ctx->progress_cb_userdata);
        }
        if (qwen_tts_verbose >= 1 && (n_generated % 10 == 0)) {
            double elapsed = time_ms() - t_gen;
            fprintf(stderr, "\r  Token %d (%.1f ms/token)...", n_generated, elapsed / n_generated);
        }
    }

    if (stop_reason == 0) {
        stop_reason = 2;
        stop_step = max_tokens;
    }

    double t_gen_done = time_ms();
    ctx->perf_talker_ms = t_gen_done - t_gen;
    ctx->perf_codec_tokens = n_generated;

    if (qwen_tts_verbose >= 1) {
        fprintf(stderr, "\r                                        \r");  /* clear progress line */
        fprintf(stderr, "Generated %d codec tokens in %.1f ms (%.1f ms/token)\n",
                n_generated, ctx->perf_talker_ms,
                n_generated > 0 ? ctx->perf_talker_ms / n_generated : 0);
        fprintf(stderr, "Stop: %s at step %d\n",
                stop_reason == 1 ? "eos" : "max_tokens", stop_step);
        if (qwen_tts_verbose >= 2) {
            fprintf(stderr, "Token trace:");
            for (int i = 0; i < n_generated; i++) {
                fprintf(stderr, "%s%d", i == 0 ? " " : ",", generated_tokens[i]);
            }
            fprintf(stderr, "\n");
        }
    }

    free(logits); free(generated_tokens); free(suppress_tokens); free(emb_tmp);
    free(trailing_text); free(tts_pad_proj); free(tts_bos_proj); free(tts_eos_proj);
    free(text_tokens);

    if (n_generated == 0) {
        free(all_codes); free(next_embed);
        *out_samples = 0;
        return NULL;
    }

    if (qwen_tts_ensure_codec_loaded(ctx) != 0) {
        fprintf(stderr,
                "Error: codec decoder weights are unavailable (missing /model/speech_tokenizer/*.safetensors)\n");
        free(all_codes); free(next_embed);
        *out_samples = 0;
        return NULL;
    }

    /* ---- Codec Decode ---- */
    double t_codec = time_ms();

    float *audio = qwen_tts_codec_decode(ctx, all_codes, n_generated, out_samples);
    if (!audio || *out_samples <= 0) {
        free(all_codes); free(next_embed);
        *out_samples = 0;
        return NULL;
    }

    double t_codec_done = time_ms();
    ctx->perf_codec_ms = t_codec_done - t_codec;
    ctx->perf_total_ms = t_codec_done - t_start;

    if (qwen_tts_verbose >= 1) {
        fprintf(stderr, "Codec decode: %d samples in %.1f ms\n", *out_samples, ctx->perf_codec_ms);
        fprintf(stderr, "Total: %.1f ms (%.2f s audio, %.2fx realtime)\n",
                ctx->perf_total_ms,
                *out_samples > 0 ? (float)*out_samples / QWEN_TTS_SAMPLE_RATE : 0,
                *out_samples > 0 ? ((float)*out_samples / QWEN_TTS_SAMPLE_RATE) / (ctx->perf_total_ms / 1000.0) : 0);
    }

    free(all_codes); free(next_embed);

    return audio;
}

/* ========================================================================
 * Streaming (Dual-Track Chunked) Generate
 *
 * Reuses the same prefill + AR loop as qwen_tts_generate(), but decodes
 * codec tokens in overlapping chunks during generation and delivers audio
 * via callback.
 *
 * chunk_size  = number of NEW codec frames per chunk (e.g. 25).
 *               0 means decode everything at the end (one callback).
 * left_context = chunk_size frames of overlap for continuity.
 *
 * Returns: 0 = success, -1 = error, 1 = aborted by callback.
 * ======================================================================== */

int qwen_tts_generate_stream(
    qwen_tts_ctx_t *ctx,
    const char *text,
    const char *speaker,
    const char *language,
    int chunk_size,
    qwen_tts_audio_cb audio_cb,
    void *userdata
) {
    if (!ctx || !audio_cb) {
        return -1;
    }

    int effective_chunk = chunk_size > 0 ? chunk_size : 0;
    int left_context = effective_chunk;  /* same as chunk_size */

    /* ---- Parse text as comma-separated token IDs ---- */
    int *text_tokens = NULL;
    int n_text_tokens = 0;

    if (text) {
        const char *p = text;
        int count = 1;
        while (*p) { if (*p == ',') count++; p++; }

        text_tokens = (int *)malloc(count * sizeof(int));
        p = text;
        while (*p && n_text_tokens < count) {
            while (*p == ' ' || *p == ',') p++;
            if (*p == '\0') break;
            char *endp = NULL;
            long v = strtol(p, &endp, 10);
            if (endp == p) {
                fprintf(stderr, "Error: invalid token ID near '%s'\n", p);
                free(text_tokens);
                return -1;
            }
            text_tokens[n_text_tokens++] = (int)v;
            p = endp;
        }
    }

    if (n_text_tokens < 8) {
        fprintf(stderr, "Error: need at least 8 text tokens (chat template format)\n");
        free(text_tokens);
        return -1;
    }

    qwen_tts_config_t *cfg = &ctx->config;
    int hidden = cfg->talker_hidden;
    int num_groups = cfg->num_code_groups;

    double t_start = time_ms();

    /* ---- Look up speaker and language IDs ---- */
    int speaker_codec_id = -1;
    if (speaker && strlen(speaker) > 0) {
        for (int i = 0; i < cfg->n_speakers; i++) {
            if (strcasecmp(cfg->speaker_names[i], speaker) == 0) {
                speaker_codec_id = cfg->speaker_ids[i];
                break;
            }
        }
        if (speaker_codec_id < 0) {
            fprintf(stderr, "Warning: speaker '%s' not found, using no speaker embedding\n", speaker);
        }
    }

    int language_codec_id = -1;
    if (language && strlen(language) > 0 && strcasecmp(language, "auto") != 0) {
        for (int i = 0; i < cfg->n_languages; i++) {
            if (strcasecmp(cfg->language_names[i], language) == 0) {
                language_codec_id = cfg->language_ids[i];
                break;
            }
        }
        if (language_codec_id < 0) {
            fprintf(stderr, "Warning: language '%s' not found\n", language);
        }
    }

    /* ---- Build prefix embedding sequence ---- */
    int n_content = n_text_tokens - 8;
    if (n_content < 0) n_content = 0;

    int codec_prefix[8];
    int n_codec_prefix = 0;

    if (language_codec_id < 0) {
        codec_prefix[n_codec_prefix++] = cfg->codec_nothink_id;
        codec_prefix[n_codec_prefix++] = cfg->codec_think_bos_id;
        codec_prefix[n_codec_prefix++] = cfg->codec_think_eos_id;
    } else {
        codec_prefix[n_codec_prefix++] = cfg->codec_think_id;
        codec_prefix[n_codec_prefix++] = cfg->codec_think_bos_id;
        codec_prefix[n_codec_prefix++] = language_codec_id;
        codec_prefix[n_codec_prefix++] = cfg->codec_think_eos_id;
    }

    if (speaker_codec_id >= 0) {
        codec_prefix[n_codec_prefix++] = speaker_codec_id;
    }
    codec_prefix[n_codec_prefix++] = cfg->codec_pad_id;
    codec_prefix[n_codec_prefix++] = cfg->codec_bos_id;

    int prefill_len = 3 + n_codec_prefix;
    float *input_embeds = (float *)calloc((size_t)prefill_len * hidden, sizeof(float));

    /* 1. Role tokens */
    for (int i = 0; i < 3; i++) {
        embed_text_token(ctx, text_tokens[i], input_embeds + i * hidden);
    }

    /* 2. Pad/bos section */
    float *tts_pad_proj = (float *)malloc(hidden * sizeof(float));
    float *tts_bos_proj = (float *)malloc(hidden * sizeof(float));
    float *tts_eos_proj = (float *)malloc(hidden * sizeof(float));
    float *codec_emb_tmp = (float *)malloc(hidden * sizeof(float));
    embed_text_token(ctx, QWEN_TTS_TOKEN_TTS_PAD, tts_pad_proj);
    embed_text_token(ctx, QWEN_TTS_TOKEN_TTS_BOS, tts_bos_proj);
    embed_text_token(ctx, QWEN_TTS_TOKEN_TTS_EOS, tts_eos_proj);

    for (int i = 0; i < n_codec_prefix - 1; i++) {
        float *dst = input_embeds + (3 + i) * hidden;
        if (i < n_codec_prefix - 2) {
            memcpy(dst, tts_pad_proj, hidden * sizeof(float));
        } else {
            memcpy(dst, tts_bos_proj, hidden * sizeof(float));
        }
        embed_codec_token(ctx, codec_prefix[i], codec_emb_tmp);
        kernel_add_inplace(dst, codec_emb_tmp, hidden);
    }

    /* 3. First text token + codec_bos */
    {
        int pos = 3 + n_codec_prefix - 1;
        float *dst = input_embeds + pos * hidden;
        embed_text_token(ctx, text_tokens[3], dst);
        embed_codec_token(ctx, cfg->codec_bos_id, codec_emb_tmp);
        kernel_add_inplace(dst, codec_emb_tmp, hidden);
    }
    free(codec_emb_tmp);

    /* Build trailing text embeddings */
    int n_trailing = (n_text_tokens - 4 - 5) + 1;
    if (n_trailing < 1) n_trailing = 1;
    float *trailing_text = (float *)calloc((size_t)n_trailing * hidden, sizeof(float));
    for (int i = 0; i < n_trailing - 1; i++) {
        embed_text_token(ctx, text_tokens[4 + i], trailing_text + i * hidden);
    }
    memcpy(trailing_text + (n_trailing - 1) * hidden, tts_eos_proj, hidden * sizeof(float));

    /* ---- Ensure codec is loaded BEFORE AR loop (needed for streaming decode) ---- */
    if (qwen_tts_ensure_codec_loaded(ctx) != 0) {
        fprintf(stderr,
                "Error: codec decoder weights are unavailable (missing /model/speech_tokenizer/*.safetensors)\n");
        free(input_embeds); free(trailing_text);
        free(tts_pad_proj); free(tts_bos_proj); free(tts_eos_proj);
        free(text_tokens);
        return -1;
    }

    /* ---- Prefill ---- */
    double t_prefill = time_ms();
    ctx->talker_kv_len = 0;
    qwen_tts_talker_prefill(ctx, input_embeds, prefill_len);

    double t_prefill_done = time_ms();
    if (qwen_tts_verbose >= 1)
        fprintf(stderr, "Stream prefill: %d tokens in %.1f ms\n", prefill_len, t_prefill_done - t_prefill);

    free(input_embeds);

    /* ---- Autoregressive generation with chunked decode ---- */
    int fixed_tokens = ctx->fixed_codec_tokens > 0 ? ctx->fixed_codec_tokens : 0;
    int max_tokens = fixed_tokens > 0 ? fixed_tokens : ctx->max_new_tokens;
    int *all_codes = (int *)calloc((size_t)max_tokens * num_groups, sizeof(int));
    int *generated_tokens = (int *)calloc(max_tokens, sizeof(int));
    int n_generated = 0;
    int stop_reason = 0;
    int chunks_sent = 0;
    int aborted = 0;

    float *logits = (float *)malloc(cfg->talker_vocab_size * sizeof(float));
    float *next_embed = (float *)malloc(hidden * sizeof(float));
    float *emb_tmp = (float *)malloc(hidden * sizeof(float));
    float rng_state = (float)ctx->sample_seed;

    /* Suppress tokens */
    int suppress_start = cfg->talker_vocab_size - 1024;
    int *suppress_tokens = (int *)malloc(1024 * sizeof(int));
    int n_suppress = 0;
    for (int i = suppress_start; i < cfg->talker_vocab_size; i++) {
        if (i != cfg->codec_eos_id) suppress_tokens[n_suppress++] = i;
    }

    double t_gen = time_ms();

    for (int step = 0; step < max_tokens; step++) {
        /* Compute logits */
        if (step == 0) {
            kernel_matvec_bf16(logits, ctx->talker.codec_head_bf16, ctx->tk_x,
                               cfg->talker_vocab_size, hidden);
        } else {
            qwen_tts_talker_forward(ctx, next_embed, logits);
        }

        /* Apply suppress tokens */
        for (int i = 0; i < n_suppress; i++) {
            logits[suppress_tokens[i]] = -1e9f;
        }

        /* Apply repetition penalty */
        kernel_apply_repetition_penalty(logits, generated_tokens, n_generated,
                                        cfg->talker_vocab_size, ctx->repetition_penalty);

        /* Sample */
        int token = kernel_sample_top_k(logits, cfg->talker_vocab_size, ctx->top_k,
                                         ctx->top_p, ctx->temperature, &rng_state);

        if (fixed_tokens > 0 && token == cfg->codec_eos_id && n_generated < fixed_tokens) {
            float eos_logit = logits[cfg->codec_eos_id];
            logits[cfg->codec_eos_id] = -1e9f;
            token = kernel_sample_top_k(logits, cfg->talker_vocab_size, ctx->top_k,
                                        ctx->top_p, ctx->temperature, &rng_state);
            logits[cfg->codec_eos_id] = eos_logit;
        }

        /* Check for EOS */
        if (fixed_tokens == 0 && token == cfg->codec_eos_id) {
            stop_reason = 1;
            if (qwen_tts_verbose >= 1)
                fprintf(stderr, "Stream EOS at step %d\n", step);
            break;
        }

        generated_tokens[n_generated] = token;

        /* Generate remaining code groups via sub-talker */
        int codes[QWEN_TTS_NUM_CODE_GROUPS];
        qwen_tts_subtalker_generate(ctx, ctx->tk_x, token, codes);

        /* Store all codes */
        memcpy(all_codes + n_generated * num_groups, codes, num_groups * sizeof(int));
        n_generated++;

        /* ---- Chunked decode: emit audio when we have a full chunk ---- */
        if (effective_chunk > 0 && n_generated % effective_chunk == 0) {
            int chunk_idx = chunks_sent;
            int chunk_start = chunk_idx * effective_chunk;
            int ctx_start = (chunks_sent > 0) ? chunk_start - left_context : 0;
            if (ctx_start < 0) ctx_start = 0;
            int ctx_frames = chunk_start - ctx_start;
            int total_frames = ctx_frames + effective_chunk;

            int n_audio = 0;
            float *audio = qwen_tts_codec_decode(ctx,
                all_codes + ctx_start * num_groups,
                total_frames, &n_audio);

            if (audio && n_audio > 0) {
                int trim = ctx_frames * QWEN_TTS_DECODE_UPSAMPLE;
                float *output = audio + trim;
                int output_len = n_audio - trim;
                if (output_len > 0) {
                    int ret = audio_cb(output, output_len, userdata);
                    if (ret != 0) {
                        aborted = 1;
                        free(audio);
                        break;
                    }
                }
                free(audio);
            }
            chunks_sent++;

            if (qwen_tts_verbose >= 1) {
                double elapsed = time_ms() - t_gen;
                fprintf(stderr, "  Stream chunk %d: %d frames (ctx %d), %.1f ms elapsed\n",
                        chunks_sent, effective_chunk, ctx_frames, elapsed);
            }
        }

        /* Build next input embedding */
        memset(next_embed, 0, hidden * sizeof(float));
        embed_codec_token(ctx, token, emb_tmp);
        kernel_add_inplace(next_embed, emb_tmp, hidden);

        for (int g = 1; g < num_groups; g++) {
            int emb_dim = hidden;
            kernel_bf16_to_f32(emb_tmp, ctx->subtalker.codec_embeddings_bf16[g - 1] +
                               (size_t)codes[g] * emb_dim, emb_dim);
            kernel_add_inplace(next_embed, emb_tmp, hidden);
        }
        if (step < n_trailing) {
            kernel_add_inplace(next_embed, trailing_text + step * hidden, hidden);
        } else {
            kernel_add_inplace(next_embed, tts_pad_proj, hidden);
        }

        /* Progress callback */
        if (ctx->progress_cb) {
            ctx->progress_cb(step + 1, max_tokens, ctx->progress_cb_userdata);
        }
        if (qwen_tts_verbose >= 1 && (n_generated % 10 == 0)) {
            double elapsed = time_ms() - t_gen;
            fprintf(stderr, "\r  Token %d (%.1f ms/token)...", n_generated, elapsed / n_generated);
        }
    }

    /* ---- Flush remaining frames ---- */
    if (!aborted && n_generated > 0) {
        int flushed_start = chunks_sent * effective_chunk;
        int remaining = n_generated - flushed_start;

        if (remaining > 0) {
            /* Decode remaining frames with left context */
            int ctx_start = (chunks_sent > 0) ? flushed_start - left_context : 0;
            if (ctx_start < 0) ctx_start = 0;
            int ctx_frames = flushed_start - ctx_start;
            int total_frames = ctx_frames + remaining;

            int n_audio = 0;
            float *audio = qwen_tts_codec_decode(ctx,
                all_codes + ctx_start * num_groups,
                total_frames, &n_audio);

            if (audio && n_audio > 0) {
                int trim = ctx_frames * QWEN_TTS_DECODE_UPSAMPLE;
                float *output = audio + trim;
                int output_len = n_audio - trim;
                if (output_len > 0) {
                    int ret = audio_cb(output, output_len, userdata);
                    if (ret != 0) aborted = 1;
                }
                free(audio);
            }

            if (qwen_tts_verbose >= 1) {
                fprintf(stderr, "  Stream flush: %d remaining frames (ctx %d)\n",
                        remaining, ctx_frames);
            }
        } else if (effective_chunk == 0) {
            /* chunk_size == 0: decode all at once, single callback */
            int n_audio = 0;
            float *audio = qwen_tts_codec_decode(ctx, all_codes, n_generated, &n_audio);
            if (audio && n_audio > 0) {
                int ret = audio_cb(audio, n_audio, userdata);
                if (ret != 0) aborted = 1;
                free(audio);
            }
        }
    }

    if (stop_reason == 0 && !aborted) {
        stop_reason = 2;
    }

    double t_done = time_ms();
    ctx->perf_talker_ms = t_done - t_gen;
    ctx->perf_codec_tokens = n_generated;
    ctx->perf_total_ms = t_done - t_start;

    if (qwen_tts_verbose >= 1) {
        fprintf(stderr, "\r                                        \r");
        fprintf(stderr, "Stream: %d codec tokens in %.1f ms (%.1f ms/token)\n",
                n_generated, ctx->perf_talker_ms,
                n_generated > 0 ? ctx->perf_talker_ms / n_generated : 0);
        fprintf(stderr, "Stop: %s, chunks sent: %d\n",
                aborted ? "aborted" : (stop_reason == 1 ? "eos" : "max_tokens"),
                chunks_sent);
    }

    free(logits); free(generated_tokens); free(suppress_tokens); free(emb_tmp);
    free(trailing_text); free(tts_pad_proj); free(tts_bos_proj); free(tts_eos_proj);
    free(text_tokens); free(all_codes); free(next_embed);

    if (aborted) return 1;
    return 0;
}
