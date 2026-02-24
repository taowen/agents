/*
 * qwen_asr_stream.c - Streaming transcription pipeline
 */

#include "qwen_asr_internal.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_audio.h"
#include "qwen_asr_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <sys/time.h>

/* Encode one audio span into encoder tokens. Caller owns out_enc_output. */
static int stream_encode_span(qwen_ctx_t *ctx, const float *samples, int n_samples,
                              float **out_enc_output, int *out_seq_len) {
    *out_enc_output = NULL;
    *out_seq_len = 0;
    if (n_samples <= 0) return 0;

    int mel_frames = 0;
    float *mel = qwen_mel_spectrogram(samples, n_samples, &mel_frames, NULL);
    if (!mel) return -1;

    int seq_len = 0;
    float *enc_output = qwen_encoder_forward(ctx, mel, mel_frames, &seq_len);
    free(mel);
    if (!enc_output) return -1;

    *out_enc_output = enc_output;
    *out_seq_len = seq_len;
    return 0;
}

/* Detect repeated token blocks at the sequence tail.
 * Returns max repetitions found (>=1), and stores period in out_period. */
static int stream_tail_repeat_blocks(const int *tokens, int n_tokens, int max_period,
                                     int *out_period) {
    if (out_period) *out_period = 0;
    if (!tokens || n_tokens < 2) return 1;

    int best_reps = 1;
    int best_period = 0;
    int period_cap = n_tokens / 2;
    if (max_period > 0 && period_cap > max_period) period_cap = max_period;

    for (int p = 1; p <= period_cap; p++) {
        int reps = 1;
        while ((reps + 1) * p <= n_tokens) {
            const int *a = tokens + n_tokens - (reps + 1) * p;
            const int *b = tokens + n_tokens - reps * p;
            if (memcmp(a, b, (size_t)p * sizeof(int)) != 0) break;
            reps++;
        }
        if (reps > best_reps) {
            best_reps = reps;
            best_period = p;
        }
    }

    if (out_period) *out_period = best_period;
    return best_reps;
}

typedef struct {
    int64_t start_sample;
    int n_samples;
    int seq_len;
    float *enc_output; /* [seq_len, dec_hidden] */
} stream_enc_window_t;

typedef struct {
    float *stem_output;   /* [n_tokens, d_model] */
    int n_tokens;
} stream_stem_entry_t;

static void stream_clear_stem_cache(stream_stem_entry_t *stem_cache,
                                    int *n_stem_cached,
                                    float *stem_mel_global_max) {
    if (stem_cache && n_stem_cached) {
        for (int i = 0; i < *n_stem_cached; i++) {
            free(stem_cache[i].stem_output);
            stem_cache[i].stem_output = NULL;
        }
        *n_stem_cached = 0;
    }
    if (stem_mel_global_max)
        *stem_mel_global_max = -1e30f;
}

static void stream_clear_enc_cache(stream_enc_window_t *enc_cache,
                                   int *n_enc_cache,
                                   int *enc_cache_start,
                                   int *enc_cached_seq_total,
                                   int64_t *next_window_start,
                                   int64_t new_start_sample) {
    if (!enc_cache || !n_enc_cache || !enc_cache_start ||
        !enc_cached_seq_total || !next_window_start) {
        return;
    }
    for (int i = *enc_cache_start; i < *n_enc_cache; i++) {
        free(enc_cache[i].enc_output);
        enc_cache[i].enc_output = NULL;
    }
    *n_enc_cache = 0;
    *enc_cache_start = 0;
    *enc_cached_seq_total = 0;
    *next_window_start = new_start_sample;
}

/* Encode audio span using stem cache for Conv2D reuse.
 * Reuses cached Conv2D stem outputs for mel chunks that haven't changed.
 * Returns encoder output [seq_len, output_dim] (caller owns), or NULL on failure. */
static float *stream_encode_stem_cached(
    qwen_ctx_t *ctx, const float *samples, int n_samples,
    stream_stem_entry_t **stem_cache_ptr, int *n_stem_cached_ptr,
    int *stem_cache_cap_ptr, float *stem_mel_global_max_ptr,
    int *out_seq_len, int *out_stem_hits, int *out_stem_total)
{
    *out_seq_len = 0;
    if (out_stem_hits) *out_stem_hits = 0;
    if (out_stem_total) *out_stem_total = 0;
    if (n_samples <= 0) return NULL;

    int mel_frames = 0;
    float *mel = qwen_mel_spectrogram(samples, n_samples, &mel_frames,
                                       stem_mel_global_max_ptr);
    if (!mel) return NULL;

    int mel_chunk_size = ctx->config.enc_chunk_size;
    int n_mel_chunks = (mel_frames + mel_chunk_size - 1) / mel_chunk_size;
    int d_model = ctx->config.enc_d_model;
    int stem_hits = 0;
    stream_stem_entry_t *sc = *stem_cache_ptr;
    int n_sc = *n_stem_cached_ptr;

    /* Grow cache if needed */
    if (n_mel_chunks > *stem_cache_cap_ptr) {
        int new_cap = *stem_cache_cap_ptr > 0 ? *stem_cache_cap_ptr : 8;
        while (new_cap < n_mel_chunks) new_cap *= 2;
        stream_stem_entry_t *tmp = (stream_stem_entry_t *)realloc(
            sc, (size_t)new_cap * sizeof(stream_stem_entry_t));
        if (tmp) {
            for (int i = *stem_cache_cap_ptr; i < new_cap; i++) {
                tmp[i].stem_output = NULL;
                tmp[i].n_tokens = 0;
            }
            sc = tmp;
            *stem_cache_ptr = sc;
            *stem_cache_cap_ptr = new_cap;
        }
    }

    /* Process each mel chunk with cache */
    int total_tokens = 0;
    for (int c = 0; c < n_mel_chunks; c++) {
        int cs = c * mel_chunk_size;
        int ce = cs + mel_chunk_size;
        if (ce > mel_frames) ce = mel_frames;
        int cw = ce - cs;

        /* Cache hit: all chunks except previously-last are stable
         * (reflect-padding only affects the tail chunk) */
        if (c < n_sc - 1 && sc[c].stem_output) {
            total_tokens += sc[c].n_tokens;
            stem_hits++;
        } else {
            if (c < n_sc && sc[c].stem_output) {
                free(sc[c].stem_output);
                sc[c].stem_output = NULL;
            }
            int n_tok = 0;
            float *sout = qwen_encoder_stem_chunk(
                ctx, mel, mel_frames, cs, cw, &n_tok);
            sc[c].stem_output = sout;
            sc[c].n_tokens = n_tok;
            total_tokens += n_tok;
        }
    }
    /* Free entries beyond current chunk count */
    for (int c = n_mel_chunks; c < n_sc; c++) {
        free(sc[c].stem_output);
        sc[c].stem_output = NULL;
        sc[c].n_tokens = 0;
    }
    *n_stem_cached_ptr = n_mel_chunks;

    /* Concatenate stem outputs */
    float *stem_x = (float *)calloc(
        (size_t)total_tokens * d_model, sizeof(float));
    if (!stem_x) { free(mel); return NULL; }
    int soff = 0;
    for (int c = 0; c < n_mel_chunks; c++) {
        memcpy(stem_x + (size_t)soff * d_model,
               sc[c].stem_output,
               (size_t)sc[c].n_tokens * d_model * sizeof(float));
        soff += sc[c].n_tokens;
    }

    /* Run transformer (consumes stem_x) */
    int seq_len = 0;
    float *enc_out = qwen_encoder_transformer(ctx, stem_x, total_tokens, &seq_len);
    free(mel);

    *out_seq_len = seq_len;
    if (out_stem_hits) *out_stem_hits = stem_hits;
    if (out_stem_total) *out_stem_total = n_mel_chunks;
    return enc_out;
}

/* Re-anchor stream text state to a short committed tail so decoding can
 * continue after a hard reset without replaying the full text history. */
static int stream_reanchor_text_state(qwen_ctx_t *ctx,
                                      const int *emitted_text_tokens,
                                      int n_emitted_text_tokens,
                                      int carry_text_tokens,
                                      int **raw_tokens,
                                      int *raw_tokens_cap,
                                      int *n_raw_tokens,
                                      int **stable_text_tokens,
                                      int *stable_text_cap,
                                      int *n_stable_text_tokens) {
    if (!ctx || !raw_tokens || !raw_tokens_cap || !n_raw_tokens ||
        !stable_text_tokens || !stable_text_cap || !n_stable_text_tokens) {
        return -1;
    }

    int carry = n_emitted_text_tokens;
    if (carry_text_tokens > 0 && carry > carry_text_tokens) carry = carry_text_tokens;
    if (carry < 0) carry = 0;

    int raw_lead = (ctx->n_force_prompt_tokens <= 0) ? 1 : 0; /* add <asr_text> marker */
    int raw_need = raw_lead + carry;

    if (raw_need > *raw_tokens_cap) {
        int new_cap = *raw_tokens_cap > 0 ? *raw_tokens_cap : 64;
        while (raw_need > new_cap) new_cap *= 2;
        int *tmp_raw = (int *)realloc(*raw_tokens, (size_t)new_cap * sizeof(int));
        if (!tmp_raw) return -1;
        *raw_tokens = tmp_raw;
        *raw_tokens_cap = new_cap;
    }
    if (carry > *stable_text_cap) {
        int new_cap = *stable_text_cap > 0 ? *stable_text_cap : 64;
        while (carry > new_cap) new_cap *= 2;
        int *tmp_stable = (int *)realloc(*stable_text_tokens, (size_t)new_cap * sizeof(int));
        if (!tmp_stable) return -1;
        *stable_text_tokens = tmp_stable;
        *stable_text_cap = new_cap;
    }

    int tail_off = n_emitted_text_tokens - carry;
    if (tail_off < 0) tail_off = 0;
    if (raw_lead) (*raw_tokens)[0] = QWEN_TOKEN_ASR_TEXT;
    if (carry > 0 && emitted_text_tokens) {
        memcpy((*raw_tokens) + raw_lead,
               emitted_text_tokens + tail_off,
               (size_t)carry * sizeof(int));
        memcpy(*stable_text_tokens,
               emitted_text_tokens + tail_off,
               (size_t)carry * sizeof(int));
    }

    *n_raw_tokens = raw_need;
    *n_stable_text_tokens = carry;
    return 0;
}

/* ========================================================================
 * Streaming Transcription (chunked rollback + encoder window cache)
 *
 * Decoder-side behavior follows the official streaming policy:
 * 1. Consume audio in fixed chunks (default 2 seconds).
 * 2. Use prefix rollback:
 *    - first N chunks: no text prefix,
 *    - later chunks: previous decoded tokens minus last K unfixed tokens.
 * 3. Decode only up to a bounded number of new tokens each step.
 * 4. Emit token deltas from the stable frontier.
 *
 * Encoder-side optimization:
 * - The encoder uses local attention windows, so completed windows are
 *   immutable.
 * - We cache completed window outputs once and only re-encode the current
 *   partial tail window.
 * - Decoder prefill still consumes all encoder tokens
 *   ([cached windows] + [current partial window]).
 * ======================================================================== */

/* Internal streaming implementation. When live!=NULL, audio is read
 * incrementally from the live buffer; when NULL, samples/n_samples
 * provide the complete audio upfront. */
static char *stream_impl(qwen_ctx_t *ctx, const float *samples, int n_samples,
                          qwen_live_audio_t *live) {
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int chunk_samples = (int)(ctx->stream_chunk_sec * QWEN_SAMPLE_RATE);
    int rollback = ctx->stream_rollback;
    int unfixed_chunks = ctx->stream_unfixed_chunks;
    int max_new_tokens = ctx->stream_max_new_tokens > 0 ? ctx->stream_max_new_tokens : 32;

    const float *audio_samples = samples;
    int64_t audio_n_samples = n_samples;
    float *compacted_samples = NULL;
    if (!live && ctx->skip_silence) {
        int compacted_n = n_samples;
        compacted_samples = compact_silence(samples, n_samples, &compacted_n);
        if (compacted_samples) audio_samples = compacted_samples;
        audio_n_samples = compacted_n;
        if (qwen_verbose >= 1) {
            float used_pct = 100.0f * (float)audio_n_samples /
                             (float)(n_samples > 0 ? n_samples : 1);
            float skipped_pct = 100.0f - used_pct;
            if (skipped_pct < 0.0f) skipped_pct = 0.0f;
            fprintf(stderr, "Silence skip: used %.1f%%, skipped %.1f%% (%d -> %lld samples)\n",
                    used_pct, skipped_pct, n_samples, (long long)audio_n_samples);
        }
    }

    /* For live mode, keep a local rolling buffer with global sample base. */
    float *local_samples = NULL;
    int64_t local_n_samples = 0;
    int64_t local_capacity = 0;
    int64_t local_base_sample = 0;
    int live_eof = 0;

    if (live) {
        /* Seed local buffer with whatever is available now. */
        pthread_mutex_lock(&live->mutex);
        int64_t live_start = live->sample_offset;
        int64_t live_count = live->n_samples;
        live_eof = live->eof;
        local_n_samples = live_count;
        local_base_sample = live_start;
        if (local_n_samples > 0) {
            local_capacity = local_n_samples + chunk_samples * 4;
            if ((uint64_t)local_capacity > (uint64_t)(SIZE_MAX / sizeof(float))) {
                pthread_mutex_unlock(&live->mutex);
                return NULL;
            }
            local_samples = (float *)malloc((size_t)local_capacity * sizeof(float));
            if (!local_samples) {
                pthread_mutex_unlock(&live->mutex);
                return NULL;
            }
            memcpy(local_samples, live->samples, (size_t)local_n_samples * sizeof(float));
        }
        /* Producer buffer is now mirrored locally: reset it to bound memory. */
        live->sample_offset = live_start + live_count;
        live->n_samples = 0;
        pthread_mutex_unlock(&live->mutex);
        audio_samples = local_samples;
        audio_n_samples = local_base_sample + local_n_samples;
    } else {
        if (audio_n_samples > INT_MAX) {
            free(compacted_samples);
            return NULL;
        }
        local_base_sample = 0;
        local_n_samples = audio_n_samples;
    }

    ctx->perf_total_ms = 0;
    ctx->perf_text_tokens = 0;
    ctx->perf_audio_ms = live ? 0.0 : 1000.0 * (double)n_samples / (double)QWEN_SAMPLE_RATE;
    ctx->perf_encode_ms = 0;
    ctx->perf_decode_ms = 0;
    int enc_window_frames = ctx->config.enc_n_window_infer;
    if (enc_window_frames < 100) enc_window_frames = 100;
    if (enc_window_frames > 800) enc_window_frames = 800;
    int enc_window_samples = enc_window_frames * QWEN_HOP_LENGTH;
    const char *no_cache_env = getenv("QWEN_STREAM_NO_ENC_CACHE");
    int use_enc_cache = 1;
    if (no_cache_env && no_cache_env[0] != '\0' && strcmp(no_cache_env, "0") != 0) {
        use_enc_cache = 0;
    }
    if (live && !use_enc_cache) {
        if (qwen_verbose >= 1) {
            fprintf(stderr, "Streaming (live): forcing encoder cache on (no-cache mode disabled)\n");
        }
        use_enc_cache = 1;
    }

    /* Sliding-window limits for long streams: bound encoder tokens and
     * prefix tokens fed to the decoder so memory/compute stay flat.
     * 4 windows × 8 s = 32 s of audio context; 150 prefix tokens ≈
     * 140 text tokens of decoder context.  raw_tokens array itself
     * grows unbounded (negligible memory) for correct text matching. */
    #define QWEN_STREAM_MAX_ENC_WINDOWS  4
    #define QWEN_STREAM_MAX_PREFIX_TOKENS 150
    #define QWEN_STREAM_MAX_REPEAT_TOKEN_RUN 12
    #define QWEN_STREAM_OVERLAP_MAX_TOKENS 48
    #define QWEN_STREAM_OVERLAP_MIN_TOKENS 4
    #define QWEN_STREAM_DEGEN_MAX_PERIOD 6
    #define QWEN_STREAM_DEGEN_MIN_REPEATS 4
    #define QWEN_STREAM_STALE_CHUNKS 4
    #define QWEN_STREAM_RESET_INTERVAL_CHUNKS 45
    #define QWEN_STREAM_RESET_CARRY_TOKENS 24

    if (qwen_verbose >= 2) {
        if (live)
            fprintf(stderr,
                    "Streaming (live): chunk=%.1f s, rollback=%d, "
                    "unfixed=%d, max_new=%d, enc_window=%.1fs, enc_cache=%s, prefix=%s, "
                    "max_enc_win=%d, max_prefix=%d\n",
                    ctx->stream_chunk_sec, rollback,
                    unfixed_chunks, max_new_tokens,
                    (float)enc_window_frames / 100.0f,
                    use_enc_cache ? "on" : "off",
                    ctx->past_text_conditioning ? "on" : "off",
                    QWEN_STREAM_MAX_ENC_WINDOWS, QWEN_STREAM_MAX_PREFIX_TOKENS);
        else
            fprintf(stderr,
                    "Streaming: %lld samples (%.1f s), chunk=%.1f s, rollback=%d, "
                    "unfixed=%d, max_new=%d, enc_window=%.1fs, enc_cache=%s, prefix=%s\n",
                    (long long)audio_n_samples, (float)audio_n_samples / QWEN_SAMPLE_RATE,
                    ctx->stream_chunk_sec, rollback,
                    unfixed_chunks, max_new_tokens,
                    (float)enc_window_frames / 100.0f,
                    use_enc_cache ? "on" : "off",
                    ctx->past_text_conditioning ? "on" : "off");
    }

    /* Load tokenizer */
    char vocab_path[1024];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", ctx->model_dir);
    qwen_tokenizer_t *tokenizer = qwen_tokenizer_load(vocab_path);
    if (!tokenizer) {
        free(compacted_samples);
        return NULL;
    }
    if (prepare_prompt_tokens(ctx, tokenizer) != 0) {
        qwen_tokenizer_free(tokenizer);
        free(compacted_samples);
        return NULL;
    }

    /* In non-interactive mode (no token callback) with pre-loaded audio,
     * streaming chunks are not externally consumed and the final answer is
     * already produced by a full refinement pass. Skip chunk-by-chunk
     * decoding entirely. (In live mode we must still use the chunked loop.) */
    if (!ctx->token_cb && !live) {
        if (qwen_verbose >= 2) {
            fprintf(stderr, "Streaming: no token callback, using direct final refinement\n");
        }
        if (audio_n_samples > INT_MAX) {
            qwen_tokenizer_free(tokenizer);
            free(compacted_samples);
            return NULL;
        }
        char *text = transcribe_segment(ctx, audio_samples, (int)audio_n_samples,
                                        tokenizer, NULL, 0, NULL);
        qwen_tokenizer_free(tokenizer);
        free(compacted_samples);
        return text;
    }

    /* Raw decoded history (language + <asr_text> + text), tokenized. */
    int *raw_tokens = (int *)malloc(8192 * sizeof(int));
    int n_raw_tokens = 0;
    int raw_tokens_cap = 8192;

    /* Stable committed text tokens already emitted to stdout. */
    int *stable_text_tokens = (int *)malloc(8192 * sizeof(int));
    int n_stable_text_tokens = 0;
    int stable_text_cap = 8192;
    int *emitted_text_tokens = (int *)malloc(8192 * sizeof(int));
    int n_emitted_text_tokens = 0;
    int emitted_text_cap = 8192;
    int stagnant_chunks = 0;
    /* Result text accumulator */
    size_t result_cap = 4096;
    size_t result_len = 0;
    char *result = (char *)malloc(result_cap);
    if (!raw_tokens || !stable_text_tokens || !emitted_text_tokens || !result) {
        free(raw_tokens);
        free(stable_text_tokens);
        free(emitted_text_tokens);
        free(result);
        qwen_tokenizer_free(tokenizer);
        free(compacted_samples);
        return NULL;
    }
    result[0] = '\0';

    /* Single-token decoder input buffer reused across all chunks. */
    float *tmp_embed = (float *)malloc(dim * sizeof(float));
    if (!tmp_embed) {
        free(raw_tokens);
        free(stable_text_tokens);
        free(emitted_text_tokens);
        free(result);
        qwen_tokenizer_free(tokenizer);
        free(compacted_samples);
        return NULL;
    }

    int chunk_idx = 0;
    int64_t audio_cursor = 0;
    stream_enc_window_t *enc_cache = NULL;
    int n_enc_cache = 0;
    int enc_cache_start = 0;  /* first live entry (older ones evicted) */
    int enc_cache_cap = 0;
    int enc_cached_seq_total = 0;
    int64_t next_window_start = 0;
    float *prev_prefill_embeds = NULL;
    int prev_prefill_len = 0;
    int prev_prefill_cap = 0;

    /* Stem cache for partial-window Conv2D reuse */
    stream_stem_entry_t *stem_cache = NULL;
    int n_stem_cached = 0;
    int stem_cache_cap = 0;
    float stem_mel_global_max = -1e30f;
    int prefill_total_tokens = 0;
    int prefill_reused_tokens = 0;

    while (audio_cursor < audio_n_samples || (live && !live_eof)) {
        /* Live mode: wait until we have enough data for the next chunk. */
        if (live) {
            int64_t want = audio_cursor + chunk_samples;
            pthread_mutex_lock(&live->mutex);
            while (live->sample_offset + live->n_samples < want && !live->eof)
                pthread_cond_wait(&live->cond, &live->mutex);

            int64_t live_start = live->sample_offset;
            int64_t live_count = live->n_samples;
            int64_t live_end = live_start + live_count;
            int is_eof_now = live->eof;

            int64_t local_end = local_base_sample + local_n_samples;
            if (local_end < live_start) {
                if (qwen_verbose >= 1) {
                    fprintf(stderr,
                            "Streaming (live): local buffer overrun, resyncing "
                            "(local_end=%lld, live_start=%lld)\n",
                            (long long)local_end, (long long)live_start);
                }
                local_base_sample = live_start;
                local_n_samples = 0;
                local_end = local_base_sample;
            }

            if (live_end > local_end) {
                int64_t delta64 = live_end - local_end;
                int64_t src_off64 = local_end - live_start;
                if (delta64 < 0 || src_off64 < 0 || src_off64 > live_count) {
                    pthread_mutex_unlock(&live->mutex);
                    break;
                }

                if (local_n_samples + delta64 > local_capacity) {
                    int64_t new_cap = local_capacity > 0 ? local_capacity : 32000;
                    while (new_cap < local_n_samples + delta64) new_cap *= 2;
                    if ((uint64_t)new_cap > (uint64_t)(SIZE_MAX / sizeof(float))) {
                        pthread_mutex_unlock(&live->mutex);
                        break;
                    }
                    float *tmp = (float *)realloc(local_samples,
                                                  (size_t)new_cap * sizeof(float));
                    if (!tmp) {
                        pthread_mutex_unlock(&live->mutex);
                        break;
                    }
                    local_samples = tmp;
                    local_capacity = new_cap;
                }
                memcpy(local_samples + (size_t)local_n_samples,
                       live->samples + (size_t)src_off64,
                       (size_t)delta64 * sizeof(float));
                local_n_samples += delta64;
            }

            /* Producer buffer is mirrored locally: reset it to bound memory. */
            live->sample_offset = live_end;
            live->n_samples = 0;
            live_eof = is_eof_now;
            pthread_mutex_unlock(&live->mutex);

            audio_samples = local_samples;
            audio_n_samples = local_base_sample + local_n_samples;
            ctx->perf_audio_ms = 1000.0 * (double)audio_n_samples / (double)QWEN_SAMPLE_RATE;
        }

        double chunk_t0 = get_time_ms();
        audio_cursor += chunk_samples;
        if (audio_cursor > audio_n_samples) audio_cursor = audio_n_samples;
        int is_final = live ? (live_eof && audio_cursor >= audio_n_samples)
                            : (audio_cursor >= audio_n_samples);

        /* Skip cold-start chunks entirely — their decode output is discarded
         * (candidate_len=0), so the encoder + prefill + decode work is wasted.
         * Starting fresh at chunk unfixed_chunks with more audio context
         * produces equivalent results with no wasted computation. */
        if (chunk_idx < unfixed_chunks && !is_final) {
            if (qwen_verbose >= 2)
                fprintf(stderr, "  Cold-start skip: chunk %d (%.1f s audio)\n",
                        chunk_idx, (float)audio_cursor / QWEN_SAMPLE_RATE);
            ctx->perf_total_ms += get_time_ms() - chunk_t0;
            chunk_idx++;
            continue;
        }

        /* Encoder path:
         * - default: cache completed local-attention windows and re-encode only
         *   the current partial tail window,
         * - debug fallback (`QWEN_STREAM_NO_ENC_CACHE=1`): re-encode full audio
         *   prefix every chunk. */
        double t0 = get_time_ms();
        int enc_seq_len = 0;
        float *enc_output = NULL;
        int64_t full_end = (audio_cursor / enc_window_samples) * (int64_t)enc_window_samples;

        if (!use_enc_cache) {
            if (audio_cursor > INT_MAX) {
                ctx->perf_total_ms += get_time_ms() - chunk_t0;
                chunk_idx++;
                continue;
            }
            if (stream_encode_span(ctx, audio_samples, (int)audio_cursor,
                                   &enc_output, &enc_seq_len) != 0 ||
                !enc_output || enc_seq_len <= 0) {
                free(enc_output);
                ctx->perf_total_ms += get_time_ms() - chunk_t0;
                chunk_idx++;
                continue;
            }
            double enc_ms = get_time_ms() - t0;
            ctx->perf_encode_ms += enc_ms;
            if (qwen_verbose >= 2) {
                fprintf(stderr,
                        "  Encoder: %d tokens from 0.0-%.1f s (full recompute, %.0f ms)\n",
                        enc_seq_len,
                        (float)audio_cursor / QWEN_SAMPLE_RATE,
                        enc_ms);
            }
        } else {
            int enc_failed = 0;
            int prev_n_enc_cache = n_enc_cache;

            while (next_window_start < full_end) {
                int64_t ws = next_window_start;
                int64_t ws_local_off = ws - local_base_sample;
                if (ws_local_off < 0 ||
                    ws_local_off + enc_window_samples > local_n_samples) {
                    enc_failed = 1;
                    break;
                }
                float *win_enc = NULL;
                int win_seq = 0;

                /* Use stem-cached encoding when cache is available
                 * (reuses Conv2D outputs from partial window processing) */
                int win_stem_hits = 0, win_stem_total = 0;
                if (n_stem_cached > 0) {
                    win_enc = stream_encode_stem_cached(
                        ctx, audio_samples + (size_t)ws_local_off,
                        enc_window_samples,
                        &stem_cache, &n_stem_cached, &stem_cache_cap,
                        &stem_mel_global_max,
                        &win_seq, &win_stem_hits, &win_stem_total);
                    /* Clear stem cache after encoding complete window
                     * (next partial window starts from new boundary) */
                    stream_clear_stem_cache(stem_cache, &n_stem_cached,
                                            &stem_mel_global_max);
                } else {
                    if (stream_encode_span(ctx,
                                           audio_samples + (size_t)ws_local_off,
                                           enc_window_samples,
                                           &win_enc, &win_seq) != 0) {
                        win_enc = NULL;
                    }
                }

                if (!win_enc || win_seq <= 0) {
                    free(win_enc);
                    enc_failed = 1;
                    break;
                }

                if (qwen_verbose >= 2 && win_stem_total > 0) {
                    fprintf(stderr,
                            "  Stem cache: %d/%d chunks cached, %d recomputed\n",
                            win_stem_hits, win_stem_total,
                            win_stem_total - win_stem_hits);
                }

                if (n_enc_cache == enc_cache_cap) {
                    int new_cap = enc_cache_cap > 0 ? enc_cache_cap * 2 : 8;
                    stream_enc_window_t *tmp = (stream_enc_window_t *)realloc(
                        enc_cache, (size_t)new_cap * sizeof(stream_enc_window_t));
                    if (!tmp) {
                        free(win_enc);
                        enc_failed = 1;
                        break;
                    }
                    enc_cache = tmp;
                    enc_cache_cap = new_cap;
                }

                enc_cache[n_enc_cache].start_sample = ws;
                enc_cache[n_enc_cache].n_samples = enc_window_samples;
                enc_cache[n_enc_cache].seq_len = win_seq;
                enc_cache[n_enc_cache].enc_output = win_enc;
                n_enc_cache++;
                enc_cached_seq_total += win_seq;
                next_window_start += enc_window_samples;
            }

            /* Partial window: use stem cache for Conv2D reuse */
            float *partial_enc = NULL;
            int partial_seq = 0;
            if (!enc_failed && full_end < audio_cursor) {
                int64_t partial_samples64 = audio_cursor - full_end;
                int64_t partial_off64 = full_end - local_base_sample;
                if (partial_samples64 > INT_MAX || partial_off64 < 0 ||
                    partial_off64 + partial_samples64 > local_n_samples) {
                    enc_failed = 1;
                } else {
                    int partial_stem_hits = 0, partial_stem_total = 0;
                    partial_enc = stream_encode_stem_cached(
                        ctx, audio_samples + (size_t)partial_off64,
                        (int)partial_samples64,
                        &stem_cache, &n_stem_cached, &stem_cache_cap,
                        &stem_mel_global_max,
                        &partial_seq, &partial_stem_hits, &partial_stem_total);
                    if (!partial_enc) enc_failed = 1;

                    if (qwen_verbose >= 2 && partial_stem_total > 0) {
                        fprintf(stderr,
                                "  Stem cache: %d/%d chunks cached, %d recomputed\n",
                                partial_stem_hits, partial_stem_total,
                                partial_stem_total - partial_stem_hits);
                    }
                }
            }

            if (enc_failed) {
                free(partial_enc);
                ctx->perf_total_ms += get_time_ms() - chunk_t0;
                chunk_idx++;
                continue;
            }

            /* Evict old encoder windows beyond the sliding-window limit
             * to keep decoder sequence length (and KV cache) bounded. */
            {
                int evicted = 0;
                while (n_enc_cache - enc_cache_start > QWEN_STREAM_MAX_ENC_WINDOWS) {
                    enc_cached_seq_total -= enc_cache[enc_cache_start].seq_len;
                    free(enc_cache[enc_cache_start].enc_output);
                    enc_cache[enc_cache_start].enc_output = NULL;
                    enc_cache_start++;
                    evicted++;
                }
                if (evicted && qwen_monitor) {
                    fprintf(stderr, "\xe2\x9f\xb3");  /* ⟳ = window eviction */
                    fflush(stderr);
                }
            }

            enc_seq_len = enc_cached_seq_total + partial_seq;
            if (enc_seq_len <= 0) {
                free(partial_enc);
                ctx->perf_total_ms += get_time_ms() - chunk_t0;
                chunk_idx++;
                continue;
            }

            enc_output = (float *)malloc((size_t)enc_seq_len * dim * sizeof(float));
            if (!enc_output) {
                free(partial_enc);
                ctx->perf_total_ms += get_time_ms() - chunk_t0;
                chunk_idx++;
                continue;
            }

            int enc_off = 0;
            for (int i = enc_cache_start; i < n_enc_cache; i++) {
                memcpy(enc_output + (size_t)enc_off * dim,
                       enc_cache[i].enc_output,
                       (size_t)enc_cache[i].seq_len * dim * sizeof(float));
                enc_off += enc_cache[i].seq_len;
            }
            if (partial_seq > 0 && partial_enc) {
                memcpy(enc_output + (size_t)enc_off * dim,
                       partial_enc, (size_t)partial_seq * dim * sizeof(float));
            }
            free(partial_enc);

            if (qwen_verbose >= 2) {
                double enc_ms = get_time_ms() - t0;
                ctx->perf_encode_ms += enc_ms;
                fprintf(stderr,
                        "  Encoder: %d tokens from 0.0-%.1f s (cached windows=%d, partial=%.1f s, %.0f ms)\n",
                        enc_seq_len,
                        (float)audio_cursor / QWEN_SAMPLE_RATE,
                        n_enc_cache - enc_cache_start,
                        (float)(audio_cursor - full_end) / QWEN_SAMPLE_RATE,
                        enc_ms);
            }
            if (qwen_verbose < 2) {
                double enc_ms = get_time_ms() - t0;
                ctx->perf_encode_ms += enc_ms;
            }
            if (qwen_monitor) {
                fprintf(stderr, "\xe2\x96\xb6");  /* ▶ = encoder */
                fflush(stderr);
            }
        }

        /* Prefix rollback state:
         * we feed previously decoded raw tokens minus last `rollback` tokens.
         * This mirrors official streaming and keeps boundary text stable. */
        int n_prefix_tokens_full = 0;
        int n_prefix_tokens = 0;
        int prefix_offset = 0;
        if (ctx->past_text_conditioning && chunk_idx >= unfixed_chunks && n_raw_tokens > 0) {
            n_prefix_tokens_full = n_raw_tokens - rollback;
            if (n_prefix_tokens_full < 0) n_prefix_tokens_full = 0;
            n_prefix_tokens = n_prefix_tokens_full;
            if (n_prefix_tokens > QWEN_STREAM_MAX_PREFIX_TOKENS) {
                n_prefix_tokens = QWEN_STREAM_MAX_PREFIX_TOKENS;
                prefix_offset = n_prefix_tokens_full - n_prefix_tokens;
            }
        }

        /* ---- Build input embeddings ---- */
        /* [PREFIX_HEAD] [prompt] [PREFIX_TAIL] [audio] [SUFFIX_BASE] [force-lang] [prefix_tokens] */
        int prefix_len = PREFIX_HEAD_LEN + ctx->n_prompt_tokens + PREFIX_TAIL_LEN;
        int suffix_len = SUFFIX_BASE_LEN + ctx->n_force_prompt_tokens;
        int total_seq = prefix_len + enc_seq_len + suffix_len + n_prefix_tokens;
        float *input_embeds = (float *)malloc((size_t)total_seq * dim * sizeof(float));
        if (!input_embeds) {
            free(enc_output);
            ctx->perf_total_ms += get_time_ms() - chunk_t0;
            chunk_idx++;
            continue;
        }

        int off = 0;
        for (int i = 0; i < PREFIX_HEAD_LEN; i++) {
            tok_embed_bf16_to_f32(input_embeds + off * dim,
                                  ctx->decoder.tok_embeddings_bf16,
                                  PROMPT_PREFIX_HEAD[i], dim);
            off++;
        }
        for (int i = 0; i < ctx->n_prompt_tokens; i++) {
            tok_embed_bf16_to_f32(input_embeds + off * dim,
                                  ctx->decoder.tok_embeddings_bf16,
                                  ctx->prompt_tokens[i], dim);
            off++;
        }
        for (int i = 0; i < PREFIX_TAIL_LEN; i++) {
            tok_embed_bf16_to_f32(input_embeds + off * dim,
                                  ctx->decoder.tok_embeddings_bf16,
                                  PROMPT_PREFIX_TAIL[i], dim);
            off++;
        }

        for (int i = 0; i < enc_seq_len; i++)
            memcpy(input_embeds + (prefix_len + i) * dim,
                   enc_output + i * dim, dim * sizeof(float));
        free(enc_output);
        enc_output = NULL;

        int suffix_off = prefix_len + enc_seq_len;
        for (int i = 0; i < SUFFIX_BASE_LEN; i++)
            tok_embed_bf16_to_f32(input_embeds + (suffix_off + i) * dim,
                                  ctx->decoder.tok_embeddings_bf16,
                                  PROMPT_SUFFIX_BASE[i], dim);

        for (int i = 0; i < ctx->n_force_prompt_tokens; i++)
            tok_embed_bf16_to_f32(input_embeds + (suffix_off + SUFFIX_BASE_LEN + i) * dim,
                                  ctx->decoder.tok_embeddings_bf16,
                                  ctx->force_prompt_tokens[i], dim);

        int text_off = suffix_off + suffix_len;
        for (int i = 0; i < n_prefix_tokens; i++)
            tok_embed_bf16_to_f32(input_embeds + (text_off + i) * dim,
                                  ctx->decoder.tok_embeddings_bf16,
                                  raw_tokens[prefix_offset + i], dim);

        /* ---- Decoder prefill + first token ---- */
        t0 = get_time_ms();
        int prefill_len = total_seq - 1;
        int reused_prefill = 0;
        if (prev_prefill_embeds && prev_prefill_len > 0) {
            int cmp_len = prefill_len < prev_prefill_len ? prefill_len : prev_prefill_len;
            size_t row_bytes = (size_t)dim * sizeof(float);
            while (reused_prefill < cmp_len) {
                const float *a = prev_prefill_embeds + (size_t)reused_prefill * dim;
                const float *b = input_embeds + (size_t)reused_prefill * dim;
                if (memcmp(a, b, row_bytes) != 0) break;
                reused_prefill++;
            }
        }
        /* Decoder KV reuse:
         * keep the longest unchanged prefill prefix and only prefill delta tokens. */
        ctx->kv_cache_len = reused_prefill;
        int delta_prefill = prefill_len - reused_prefill;
        if (delta_prefill > 0) {
            qwen_decoder_prefill(ctx,
                                 input_embeds + (size_t)reused_prefill * dim,
                                 delta_prefill);
        }
        prefill_total_tokens += prefill_len;
        prefill_reused_tokens += reused_prefill;

        float *last_embed = input_embeds + (size_t)prefill_len * dim;
        int token = qwen_decoder_forward(ctx, last_embed);

        if (prefill_len > prev_prefill_cap) {
            int new_cap = prev_prefill_cap > 0 ? prev_prefill_cap : 64;
            while (new_cap < prefill_len) new_cap *= 2;
            float *tmp_prev = (float *)realloc(prev_prefill_embeds,
                                               (size_t)new_cap * dim * sizeof(float));
            if (tmp_prev) {
                prev_prefill_embeds = tmp_prev;
                prev_prefill_cap = new_cap;
            } else {
                prev_prefill_len = 0;
            }
        }
        if (prev_prefill_embeds && prev_prefill_cap >= prefill_len) {
            memcpy(prev_prefill_embeds, input_embeds,
                   (size_t)prefill_len * dim * sizeof(float));
            prev_prefill_len = prefill_len;
        } else {
            prev_prefill_len = 0;
        }
        free(input_embeds);

        double prefill_ms = get_time_ms() - t0;
        ctx->perf_decode_ms += prefill_ms;
        if (qwen_verbose >= 2)
            fprintf(stderr, "  Prefill: %d tokens (%d prefix, reused %d) (%.0f ms)\n",
                    total_seq, n_prefix_tokens, reused_prefill, prefill_ms);
        if (qwen_monitor) {
            fprintf(stderr, "\xc2\xb7");  /* · = prefill */
            fflush(stderr);
        }

        /* ---- Autoregressive decode ---- */
        t0 = get_time_ms();
        int n_generated = 0;

        /* Cold-start: minimal decode during unfixed chunks (results are discarded).
         * 5 tokens is enough to detect language + <asr_text>. */
        int effective_max_new = max_new_tokens;
        if (chunk_idx < unfixed_chunks)
            effective_max_new = 5;

        /* Collect ALL generated tokens (including language, <asr_text>, etc.) */
        int *chunk_tokens = (int *)malloc((size_t)effective_max_new * sizeof(int));
        if (!chunk_tokens) {
            ctx->perf_total_ms += get_time_ms() - chunk_t0;
            chunk_idx++;
            continue;
        }
        int n_chunk_tokens = 0;

        while (n_generated < effective_max_new) {
            n_generated++;
            if (token == QWEN_TOKEN_ENDOFTEXT || token == QWEN_TOKEN_IM_END) break;

            chunk_tokens[n_chunk_tokens++] = token;

            tok_embed_bf16_to_f32(tmp_embed, ctx->decoder.tok_embeddings_bf16, token, dim);
            token = qwen_decoder_forward(ctx, tmp_embed);
        }

        double decode_ms = get_time_ms() - t0;
        ctx->perf_decode_ms += decode_ms;
        if (qwen_verbose >= 2)
            fprintf(stderr, "  Decode: %d tokens (%.0f ms, %.1f ms/token%s)\n",
                    n_generated, decode_ms,
                    n_generated > 0 ? decode_ms / n_generated : 0,
                    (n_generated >= effective_max_new &&
                     token != QWEN_TOKEN_ENDOFTEXT &&
                     token != QWEN_TOKEN_IM_END) ? ", hit max_new" : "");
        if (qwen_monitor) {
            /* ▪ = normal decode, ▸ = slow (>30ms/token) */
            double ms_per_tok = n_generated > 0 ? decode_ms / n_generated : 0;
            fprintf(stderr, "%s", ms_per_tok > 30 ? "\xe2\x96\xb8" : "\xe2\x96\xaa");
            fflush(stderr);
        }

        /* Update raw token history = full prefix + newly generated continuation.
         * Uses n_prefix_tokens_full (uncapped) so raw_tokens keeps the complete
         * token sequence for correct text-level matching in the commit phase. */
        int dropped_repeat_tokens = 0;
        if (n_chunk_tokens > 0) {
            int prev_tok = -1;
            int prev_run = 0;
            if (n_prefix_tokens_full > 0) {
                prev_tok = raw_tokens[n_prefix_tokens_full - 1];
                prev_run = 1;
                for (int j = n_prefix_tokens_full - 2; j >= 0; j--) {
                    if (raw_tokens[j] != prev_tok) break;
                    prev_run++;
                    if (prev_run >= QWEN_STREAM_MAX_REPEAT_TOKEN_RUN) break;
                }
            }

            int out = 0;
            for (int i = 0; i < n_chunk_tokens; i++) {
                int tok = chunk_tokens[i];
                int suppress = 0;

                if (tok == prev_tok) {
                    prev_run++;
                    if (prev_run > QWEN_STREAM_MAX_REPEAT_TOKEN_RUN) suppress = 1;
                } else {
                    prev_tok = tok;
                    prev_run = 1;
                }

                if (suppress) {
                    dropped_repeat_tokens++;
                    continue;
                }
                chunk_tokens[out++] = tok;
            }
            n_chunk_tokens = out;
        }

        int n_raw_new = n_prefix_tokens_full + n_chunk_tokens;
        if (n_raw_new > raw_tokens_cap) {
            while (n_raw_new > raw_tokens_cap) raw_tokens_cap *= 2;
            int *tmp_raw = (int *)realloc(raw_tokens, (size_t)raw_tokens_cap * sizeof(int));
            if (!tmp_raw) {
                free(chunk_tokens);
                ctx->perf_total_ms += get_time_ms() - chunk_t0;
                chunk_idx++;
                continue;
            }
            raw_tokens = tmp_raw;
        }
        if (n_chunk_tokens > 0) {
            memcpy(raw_tokens + n_prefix_tokens_full, chunk_tokens,
                   (size_t)n_chunk_tokens * sizeof(int));
        }
        n_raw_tokens = n_raw_new;
        free(chunk_tokens);
        if (dropped_repeat_tokens > 0 && qwen_verbose >= 2) {
            fprintf(stderr, "  Decode: dropped %d repeated tokens\n", dropped_repeat_tokens);
        }

        /* Parse text region from raw stream output:
         * - default: language ... <asr_text> TEXT
         * - forced language: prompt already anchors language, so generated stream is TEXT. */
        int text_start = 0;
        if (ctx->n_force_prompt_tokens <= 0) {
            int asr_text_pos = -1;
            for (int i = 0; i < n_raw_tokens; i++) {
                if (raw_tokens[i] == QWEN_TOKEN_ASR_TEXT) {
                    asr_text_pos = i;
                    break;
                }
            }
            text_start = (asr_text_pos >= 0) ? asr_text_pos + 1 : 0;
        }
        if (text_start < 0) text_start = 0;
        if (text_start > n_raw_tokens) text_start = n_raw_tokens;
        int n_text_tokens = n_raw_tokens - text_start;

        /* "Fixed" frontier for this chunk:
         * - cold-start chunks: emit nothing,
         * - intermediate chunks: keep last `rollback` text tokens unfixed,
         *   but if text is shorter than rollback keep only 1 token unfixed so
         *   streaming still advances,
         * - final chunk: emit everything. */
        int candidate_len = 0;
        if (is_final) {
            candidate_len = n_text_tokens;
        } else if (chunk_idx >= unfixed_chunks) {
            candidate_len = n_text_tokens - rollback;
            if (candidate_len <= 0 && n_text_tokens > 0) candidate_len = n_text_tokens - 1;
            if (candidate_len < 0) candidate_len = 0;
        }

        /* Streaming commit: emit token delta against the previous candidate.
         * We do not attempt monotonic byte-level reconciliation here. */
        int *candidate_tokens = raw_tokens + text_start;
        int did_recovery_reset = 0;
        int did_periodic_reset = 0;
        {
            int tail_period = 0;
            int tail_reps = stream_tail_repeat_blocks(candidate_tokens, candidate_len,
                                                      QWEN_STREAM_DEGEN_MAX_PERIOD,
                                                      &tail_period);
            int candidate_advance = candidate_len - n_stable_text_tokens;
            if (!is_final && n_generated >= effective_max_new && candidate_advance <= 1) {
                stagnant_chunks++;
            } else {
                stagnant_chunks = 0;
            }
            int recovery_reset = 0;
            if (tail_period > 0 && tail_reps >= QWEN_STREAM_DEGEN_MIN_REPEATS) {
                recovery_reset = 1;
            }
            if (stagnant_chunks >= QWEN_STREAM_STALE_CHUNKS) {
                recovery_reset = 1;
            }
            if (dropped_repeat_tokens >= 8) {
                recovery_reset = 1;
            }

            if (recovery_reset) {
                if (stream_reanchor_text_state(ctx,
                                               emitted_text_tokens,
                                               n_emitted_text_tokens,
                                               QWEN_STREAM_RESET_CARRY_TOKENS,
                                               &raw_tokens, &raw_tokens_cap, &n_raw_tokens,
                                               &stable_text_tokens, &stable_text_cap,
                                               &n_stable_text_tokens) != 0) {
                    n_raw_tokens = 0;
                    n_stable_text_tokens = 0;
                }
                prev_prefill_len = 0;
                stream_clear_enc_cache(enc_cache,
                                       &n_enc_cache,
                                       &enc_cache_start,
                                       &enc_cached_seq_total,
                                       &next_window_start,
                                       full_end);
                stream_clear_stem_cache(stem_cache, &n_stem_cached, &stem_mel_global_max);
                stagnant_chunks = 0;
                did_recovery_reset = 1;
                if (qwen_monitor) {
                    fprintf(stderr, "!");
                    fflush(stderr);
                }
            } else {
                if (candidate_len > stable_text_cap) {
                    while (candidate_len > stable_text_cap) stable_text_cap *= 2;
                    int *tmp_stable = (int *)realloc(stable_text_tokens,
                                                     (size_t)stable_text_cap * sizeof(int));
                    if (!tmp_stable) {
                        candidate_len = n_stable_text_tokens;
                    } else {
                        stable_text_tokens = tmp_stable;
                    }
                }

                int lcp = 0;
                while (lcp < n_stable_text_tokens &&
                       lcp < candidate_len &&
                       stable_text_tokens[lcp] == candidate_tokens[lcp]) {
                    lcp++;
                }
                for (int i = lcp; i < candidate_len; i++) {
                    stable_text_tokens[i] = candidate_tokens[i];
                }

                int emit_start = lcp;
                if (emit_start < candidate_len && n_emitted_text_tokens > 0) {
                    int max_overlap = candidate_len - emit_start;
                    if (max_overlap > n_emitted_text_tokens) max_overlap = n_emitted_text_tokens;
                    if (max_overlap > QWEN_STREAM_OVERLAP_MAX_TOKENS)
                        max_overlap = QWEN_STREAM_OVERLAP_MAX_TOKENS;
                    for (int k = max_overlap; k >= QWEN_STREAM_OVERLAP_MIN_TOKENS; k--) {
                        if (memcmp(emitted_text_tokens + (n_emitted_text_tokens - k),
                                   candidate_tokens + emit_start,
                                   (size_t)k * sizeof(int)) == 0) {
                            emit_start += k;
                            break;
                        }
                    }
                }

                for (int i = emit_start; i < candidate_len; i++) {
                    int tok = stable_text_tokens[i];
                    const char *piece = qwen_tokenizer_decode(tokenizer, tok);
                    if (ctx->token_cb) ctx->token_cb(piece, ctx->token_cb_userdata);

                    size_t plen = strlen(piece);
                    if (result_len + plen + 1 > result_cap) {
                        while (result_len + plen + 1 > result_cap) result_cap *= 2;
                        result = (char *)realloc(result, result_cap);
                    }
                    memcpy(result + result_len, piece, plen);
                    result_len += plen;
                    result[result_len] = '\0';
                    ctx->perf_text_tokens++;

                    if (n_emitted_text_tokens == emitted_text_cap) {
                        int new_cap = emitted_text_cap * 2;
                        int *tmp_emit = (int *)realloc(emitted_text_tokens,
                                                       (size_t)new_cap * sizeof(int));
                        if (tmp_emit) {
                            emitted_text_tokens = tmp_emit;
                            emitted_text_cap = new_cap;
                        }
                    }
                    if (n_emitted_text_tokens < emitted_text_cap) {
                        emitted_text_tokens[n_emitted_text_tokens++] = tok;
                    }
                }

                n_stable_text_tokens = candidate_len;

                int periodic_reset =
                    (!is_final &&
                     ctx->past_text_conditioning &&
                     chunk_idx >= unfixed_chunks &&
                     ((chunk_idx + 1) % QWEN_STREAM_RESET_INTERVAL_CHUNKS == 0));
                if (periodic_reset) {
                    if (stream_reanchor_text_state(ctx,
                                                   emitted_text_tokens,
                                                   n_emitted_text_tokens,
                                                   QWEN_STREAM_RESET_CARRY_TOKENS,
                                                   &raw_tokens, &raw_tokens_cap, &n_raw_tokens,
                                                   &stable_text_tokens, &stable_text_cap,
                                                   &n_stable_text_tokens) != 0) {
                        n_raw_tokens = 0;
                        n_stable_text_tokens = 0;
                    }
                    prev_prefill_len = 0;
                    stream_clear_enc_cache(enc_cache,
                                           &n_enc_cache,
                                           &enc_cache_start,
                                           &enc_cached_seq_total,
                                           &next_window_start,
                                           full_end);
                    stream_clear_stem_cache(stem_cache, &n_stem_cached, &stem_mel_global_max);
                    did_periodic_reset = 1;
                }
            }
        }

        if (qwen_verbose >= 2) {
            if (prefix_offset > 0)
                fprintf(stderr, "  Prefix window: %d/%d tokens (offset %d)\n",
                        n_prefix_tokens, n_prefix_tokens_full, prefix_offset);
            if (did_recovery_reset) {
                fprintf(stderr, "  Recovery reset applied\n");
            } else if (did_periodic_reset) {
                fprintf(stderr, "  Periodic reset applied\n");
            }
            fprintf(stderr, "  Commit: candidate=%d tokens, emitted_total=%d\n",
                    candidate_len, n_stable_text_tokens);
        }

        if (live && use_enc_cache) {
            /* Keep only the current partial tail [full_end, audio_n_samples). */
            int64_t keep_from = full_end;
            if (keep_from > local_base_sample) {
                int64_t drop64 = keep_from - local_base_sample;
                if (drop64 > local_n_samples) drop64 = local_n_samples;
                if (drop64 > 0) {
                    int64_t remain = local_n_samples - drop64;
                    if (remain > 0) {
                        memmove(local_samples, local_samples + (size_t)drop64,
                                (size_t)remain * sizeof(float));
                    }
                    local_n_samples = remain;
                    local_base_sample += drop64;
                    audio_samples = local_samples;
                    audio_n_samples = local_base_sample + local_n_samples;
                }
            }
        }

        ctx->perf_total_ms += get_time_ms() - chunk_t0;
        chunk_idx++;
    }

    free(tmp_embed);
    for (int i = enc_cache_start; i < n_enc_cache; i++) {
        free(enc_cache[i].enc_output);
    }
    free(enc_cache);
    stream_clear_stem_cache(stem_cache, &n_stem_cached, &stem_mel_global_max);
    free(stem_cache);
    if (qwen_verbose >= 2 && prefill_total_tokens > 0) {
        double reuse_pct = 100.0 * (double)prefill_reused_tokens / (double)prefill_total_tokens;
        fprintf(stderr, "  Prefill reuse: %d/%d tokens (%.1f%%)\n",
                prefill_reused_tokens, prefill_total_tokens, reuse_pct);
    }
    free(prev_prefill_embeds);
    free(raw_tokens);
    free(stable_text_tokens);
    free(emitted_text_tokens);
    qwen_tokenizer_free(tokenizer);
    free(compacted_samples);
    free(local_samples);

    /* Trim whitespace */
    size_t rlen = strlen(result);
    while (rlen > 0 && isspace((unsigned char)result[rlen - 1])) result[--rlen] = '\0';
    char *start = result;
    while (*start && isspace((unsigned char)*start)) start++;
    if (start != result) memmove(result, start, strlen(start) + 1);

    return result;
}

char *qwen_transcribe_stream_live(qwen_ctx_t *ctx, qwen_live_audio_t *live) {
    return stream_impl(ctx, NULL, 0, live);
}
