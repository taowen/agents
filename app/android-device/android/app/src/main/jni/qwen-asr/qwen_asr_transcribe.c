/*
 * qwen_asr_transcribe.c - Batch transcription pipeline
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

static int cmp_float_asc(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

/* Drop long silent spans while preserving short pauses for readability.
 * Uses adaptive RMS gating with spike rejection for noisy backgrounds. */
float *compact_silence(const float *samples, int n_samples, int *out_samples) {
    if (!samples || n_samples <= 0 || !out_samples) return NULL;

    const int win = 160;               /* 10 ms at 16kHz */
    const float base_thresh = 0.002f;  /* ~ -54 dBFS */
    const float max_thresh = 0.025f;   /* avoid over-aggressive clipping */
    const float smooth_alpha = 0.2f;   /* smooth frame-level RMS */
    const int min_voice_windows = 5;   /* reject <50ms spikes as noise */
    const int pad_voice_windows = 3;   /* keep 30ms around speech edges */
    const int pass_windows = 60;       /* keep first 600ms of silence */

    int n_win = (n_samples + win - 1) / win;
    float *rms_vals = (float *)malloc((size_t)n_win * sizeof(float));
    float *sorted = (float *)malloc((size_t)n_win * sizeof(float));
    float *smooth_vals = (float *)malloc((size_t)n_win * sizeof(float));
    unsigned char *is_voice = (unsigned char *)malloc((size_t)n_win);
    if (!rms_vals || !sorted || !smooth_vals || !is_voice) {
        free(rms_vals);
        free(sorted);
        free(smooth_vals);
        free(is_voice);
        return NULL;
    }

    for (int w = 0; w < n_win; w++) {
        int start = w * win;
        int end = start + win;
        if (end > n_samples) end = n_samples;
        int len = end - start;
        float energy = 0.0f;
        for (int i = 0; i < len; i++) {
            float v = samples[start + i];
            energy += v * v;
        }
        rms_vals[w] = sqrtf(energy / (float)(len > 0 ? len : 1));
    }

    /* Smooth RMS so tiny impulsive noise does not flip decisions. */
    float smooth = rms_vals[0];
    for (int w = 0; w < n_win; w++) {
        smooth = (1.0f - smooth_alpha) * smooth + smooth_alpha * rms_vals[w];
        smooth_vals[w] = smooth;
    }

    memcpy(sorted, smooth_vals, (size_t)n_win * sizeof(float));
    qsort(sorted, (size_t)n_win, sizeof(float), cmp_float_asc);

    /* Adaptive threshold from low-energy percentile (robust to loud clips). */
    int p25 = (int)((n_win - 1) * 0.25f);
    float noise_floor = sorted[p25];
    float thresh = noise_floor * 1.8f;
    if (thresh < base_thresh) thresh = base_thresh;
    if (thresh > max_thresh) thresh = max_thresh;
    free(sorted);

    for (int w = 0; w < n_win; w++) {
        is_voice[w] = (smooth_vals[w] > thresh) ? 1 : 0;
    }
    free(smooth_vals);

    /* Remove very short voice bursts (usually clicks/hiss spikes). */
    for (int i = 0; i < n_win; ) {
        if (!is_voice[i]) { i++; continue; }
        int j = i + 1;
        while (j < n_win && is_voice[j]) j++;
        if (j - i < min_voice_windows) {
            memset(is_voice + i, 0, (size_t)(j - i));
        }
        i = j;
    }

    /* Add a small speech edge pad to avoid clipping word boundaries. */
    unsigned char *padded = (unsigned char *)calloc((size_t)n_win, 1);
    if (!padded) {
        free(is_voice);
        free(rms_vals);
        return NULL;
    }
    for (int w = 0; w < n_win; w++) {
        if (!is_voice[w]) continue;
        int a = w - pad_voice_windows;
        int b = w + pad_voice_windows;
        if (a < 0) a = 0;
        if (b >= n_win) b = n_win - 1;
        for (int k = a; k <= b; k++) padded[k] = 1;
    }
    free(is_voice);

    float *out = (float *)malloc((size_t)n_samples * sizeof(float));
    if (!out) {
        free(rms_vals);
        free(padded);
        return NULL;
    }

    int out_n = 0;
    int silence_count = 0;
    for (int w = 0; w < n_win; w++) {
        int start = w * win;
        int end = start + win;
        if (end > n_samples) end = n_samples;
        int len = end - start;

        if (padded[w]) {
            memcpy(out + out_n, samples + start, (size_t)len * sizeof(float));
            out_n += len;
            silence_count = 0;
        } else {
            silence_count++;
            if (silence_count <= pass_windows) {
                memcpy(out + out_n, samples + start, (size_t)len * sizeof(float));
                out_n += len;
            }
        }
    }
    free(padded);
    free(rms_vals);

    if (out_n == 0) {
        int keep = n_samples;
        int min_keep = QWEN_SAMPLE_RATE / 2;
        if (keep > min_keep) keep = min_keep;
        memcpy(out, samples, (size_t)keep * sizeof(float));
        out_n = keep;
    }

    *out_samples = out_n;
    return out;
}

/* ---- Segment-based transcription ---- */

#define ENERGY_WINDOW_MS    100

/*
 * Find the best split point near target_sample by looking for the
 * lowest-energy 100ms window within +/-search_sec seconds.
 */
static int find_split_point(const float *samples, int n_samples,
                            int target_sample, float search_sec) {
    int search_half = (int)(search_sec * QWEN_SAMPLE_RATE);
    int lo = target_sample - search_half;
    int hi = target_sample + search_half;
    if (lo < 0) lo = 0;
    if (hi > n_samples) hi = n_samples;

    int win_samples = (ENERGY_WINDOW_MS * QWEN_SAMPLE_RATE) / 1000; /* 1600 */
    float best_energy = 1e30f;
    int best_center = target_sample;

    for (int pos = lo; pos + win_samples <= hi; pos += win_samples / 2) {
        float energy = 0;
        int end = pos + win_samples;
        if (end > n_samples) end = n_samples;
        for (int j = pos; j < end; j++) {
            energy += samples[j] * samples[j];
        }
        energy /= (end - pos);
        if (energy < best_energy) {
            best_energy = energy;
            best_center = pos + (end - pos) / 2;
        }
    }
    return best_center;
}

/*
 * Transcribe a single audio segment. Returns malloc'd text or NULL.
 * The tokenizer is passed in so we only load it once.
 */
char *transcribe_segment(qwen_ctx_t *ctx, const float *samples,
                                int n_samples, qwen_tokenizer_t *tokenizer,
                                const int *past_tokens, int n_past_tokens,
                                int *out_text_tokens) {
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    double seg_t0 = get_time_ms();
    int n_text_tokens = 0;

    /* ---- Mel spectrogram ---- */
    double t0 = get_time_ms();
    int mel_frames = 0;
    float *mel = qwen_mel_spectrogram(samples, n_samples, &mel_frames);
    if (!mel) return NULL;
    double mel_ms = get_time_ms() - t0;

    if (qwen_verbose >= 2)
        fprintf(stderr, "  Mel: %d frames (%.0f ms)\n", mel_frames, mel_ms);

    /* ---- Encoder ---- */
    t0 = get_time_ms();
    int enc_seq_len = 0;
    float *enc_output = qwen_encoder_forward(ctx, mel, mel_frames, &enc_seq_len);
    free(mel);
    if (!enc_output) return NULL;
    double enc_ms = get_time_ms() - t0;

    if (qwen_verbose >= 2)
        fprintf(stderr, "  Encoder: %d tokens (%.0f ms)\n", enc_seq_len, enc_ms);

    if (prepare_prompt_tokens(ctx, tokenizer) != 0) {
        free(enc_output);
        return NULL;
    }

    /* ---- Build input embeddings ---- */
    int prefix_len = PREFIX_HEAD_LEN + ctx->n_prompt_tokens + PREFIX_TAIL_LEN;
    int suffix_len = SUFFIX_BASE_LEN + ctx->n_force_prompt_tokens;
    int n_past_prompt_tokens = (n_past_tokens > 0) ? (n_past_tokens + 1) : 0; /* + <asr_text> */
    int total_seq = prefix_len + enc_seq_len + suffix_len + n_past_prompt_tokens;
    float *input_embeds = (float *)malloc((size_t)total_seq * dim * sizeof(float));
    float *tmp_embed = (float *)malloc(dim * sizeof(float));
    if (!input_embeds || !tmp_embed) {
        free(enc_output);
        free(input_embeds);
        free(tmp_embed);
        return NULL;
    }

    /* Embed prefix head: <|im_start|>system\n */
    int off = 0;
    for (int i = 0; i < PREFIX_HEAD_LEN; i++) {
        tok_embed_bf16_to_f32(input_embeds + off * dim,
                              ctx->decoder.tok_embeddings_bf16,
                              PROMPT_PREFIX_HEAD[i], dim);
        off++;
    }

    /* Embed optional prompt text (system content) */
    for (int i = 0; i < ctx->n_prompt_tokens; i++) {
        tok_embed_bf16_to_f32(input_embeds + off * dim,
                              ctx->decoder.tok_embeddings_bf16,
                              ctx->prompt_tokens[i], dim);
        off++;
    }

    /* Embed prefix tail: <|im_end|>\n<|im_start|>user\n<|audio_start|> */
    for (int i = 0; i < PREFIX_TAIL_LEN; i++) {
        tok_embed_bf16_to_f32(input_embeds + off * dim,
                              ctx->decoder.tok_embeddings_bf16,
                              PROMPT_PREFIX_TAIL[i], dim);
        off++;
    }

    /* Replace audio_pad positions with encoder output */
    for (int i = 0; i < enc_seq_len; i++) {
        memcpy(input_embeds + (prefix_len + i) * dim,
               enc_output + i * dim,
               dim * sizeof(float));
    }
    free(enc_output);

    /* Embed suffix base: <|audio_end|><|im_end|>\n<|im_start|>assistant\n */
    int suffix_off = prefix_len + enc_seq_len;
    for (int i = 0; i < SUFFIX_BASE_LEN; i++) {
        tok_embed_bf16_to_f32(input_embeds + (suffix_off + i) * dim,
                              ctx->decoder.tok_embeddings_bf16,
                              PROMPT_SUFFIX_BASE[i], dim);
    }

    /* Optional forced-language suffix: "language X" + <asr_text> */
    for (int i = 0; i < ctx->n_force_prompt_tokens; i++) {
        tok_embed_bf16_to_f32(input_embeds + (suffix_off + SUFFIX_BASE_LEN + i) * dim,
                              ctx->decoder.tok_embeddings_bf16,
                              ctx->force_prompt_tokens[i], dim);
    }

    /* Optional past-text conditioning tokens (for segmented mode).
     * Put a fresh <asr_text> marker AFTER the past text so generation
     * restarts from a new ASR span instead of terminating immediately. */
    int past_off = suffix_off + suffix_len;
    for (int i = 0; i < n_past_tokens; i++) {
        tok_embed_bf16_to_f32(input_embeds + (past_off + i) * dim,
                              ctx->decoder.tok_embeddings_bf16,
                              past_tokens[i], dim);
    }
    if (n_past_tokens > 0) {
        tok_embed_bf16_to_f32(input_embeds + (past_off + n_past_tokens) * dim,
                              ctx->decoder.tok_embeddings_bf16,
                              QWEN_TOKEN_ASR_TEXT, dim);
    }

    /* ---- Decoder prefill ---- */
    t0 = get_time_ms();
    ctx->kv_cache_len = 0; /* Reset KV cache for this segment */
    int prefill_len = total_seq - 1; /* prefill all but last */
    qwen_decoder_prefill(ctx, input_embeds, prefill_len);

    /* First token from last prefill position */
    float *last_embed = input_embeds + (size_t)prefill_len * dim;
    int token = qwen_decoder_forward(ctx, last_embed);
    free(input_embeds);

    double prefill_ms = get_time_ms() - t0;
    if (qwen_verbose >= 2)
        fprintf(stderr, "  Prefill: %d tokens (%.0f ms)\n", total_seq, prefill_ms);

    /* ---- Autoregressive decode ---- */
    t0 = get_time_ms();
    int max_tokens = 2048;
    int n_generated = 0;
    /* If language is forced, <asr_text> is already part of prompt suffix. */
    int past_asr_text = (ctx->n_force_prompt_tokens > 0 || n_past_tokens > 0) ? 1 : 0;

    size_t text_cap = 4096;
    size_t text_len = 0;
    char *text = (char *)malloc(text_cap);
    text[0] = '\0';

    while (n_generated < max_tokens) {
        n_generated++;

        /* Check EOS */
        if (token == QWEN_TOKEN_ENDOFTEXT || token == QWEN_TOKEN_IM_END) break;

        /* Track <asr_text> marker */
        if (token == QWEN_TOKEN_ASR_TEXT) {
            past_asr_text = 1;
        } else if (past_asr_text) {
            /* Decode and emit this text token */
            const char *piece = qwen_tokenizer_decode(tokenizer, token);
            size_t piece_len = strlen(piece);
            if (text_len + piece_len + 1 > text_cap) {
                while (text_len + piece_len + 1 > text_cap) text_cap *= 2;
                text = (char *)realloc(text, text_cap);
            }
            memcpy(text + text_len, piece, piece_len);
            text_len += piece_len;
            text[text_len] = '\0';
            n_text_tokens++;

            /* Stream token via callback */
            if (ctx->token_cb)
                ctx->token_cb(piece, ctx->token_cb_userdata);
        }

        /* Embed and generate next token */
        tok_embed_bf16_to_f32(tmp_embed, ctx->decoder.tok_embeddings_bf16, token, dim);
        token = qwen_decoder_forward(ctx, tmp_embed);
    }

    double decode_ms = get_time_ms() - t0;
    if (qwen_verbose >= 2)
        fprintf(stderr, "  Decode: %d tokens (%.0f ms, %.1f ms/token)\n",
                n_generated, decode_ms,
                n_generated > 0 ? decode_ms / n_generated : 0);

    free(tmp_embed);

    /* Trim whitespace */
    size_t rlen = strlen(text);
    while (rlen > 0 && isspace((unsigned char)text[rlen - 1])) text[--rlen] = '\0';
    char *start = text;
    while (*start && isspace((unsigned char)*start)) start++;
    if (start != text) memmove(text, start, strlen(start) + 1);

    ctx->perf_total_ms += get_time_ms() - seg_t0;
    ctx->perf_text_tokens += n_text_tokens;
    ctx->perf_encode_ms += mel_ms + enc_ms;
    ctx->perf_decode_ms += prefill_ms + decode_ms;
    if (out_text_tokens) *out_text_tokens = n_text_tokens;

    return text;
}

static int should_retry_unconditioned_segment(const char *full_result,
                                              const char *seg_text,
                                              int core_samples,
                                              int n_text_tokens) {
    if (!seg_text || seg_text[0] == '\0') return 1;

    /* A segment producing very few tokens under conditioning is usually
     * a collapse (model repeats/terminates early instead of following audio).
     * Use stricter checks from ~8s upward to catch common -S 10 failures. */
    float core_sec = (float)core_samples / (float)QWEN_SAMPLE_RATE;
    if (core_sec >= 8.0f) {
        int min_tokens = (int)(core_sec * 1.75f);
        if (min_tokens < 12) min_tokens = 12;
        if (n_text_tokens < min_tokens) return 1;
    }

    /* Exact duplicate span already present in accumulated text: likely drift. */
    if (full_result && full_result[0] != '\0') {
        size_t seg_len = strlen(seg_text);
        if (seg_len >= 48 && strstr(full_result, seg_text) != NULL) return 1;
    }

    return 0;
}

static int should_insert_boundary_space(int prev_ch, int next_ch) {
    if (prev_ch <= 0 || next_ch <= 0) return 0;
    if (isspace((unsigned char)prev_ch)) return 0;
    if (isspace((unsigned char)next_ch)) return 0;
    if (ispunct((unsigned char)next_ch)) return 0;
    return 1;
}

typedef struct {
    qwen_token_cb downstream_cb;
    void *downstream_userdata;
    int maybe_prepend_space;
    int saw_first_piece;
} segment_emit_state_t;

static void segment_emit_cb(const char *piece, void *userdata) {
    segment_emit_state_t *st = (segment_emit_state_t *)userdata;
    if (!st || !st->downstream_cb || !piece) return;

    if (!st->saw_first_piece) {
        st->saw_first_piece = 1;
        if (st->maybe_prepend_space) {
            unsigned char c0 = (unsigned char)piece[0];
            if (c0 != '\0' && !isspace(c0) && !ispunct(c0)) {
                st->downstream_cb(" ", st->downstream_userdata);
            }
        }
    }
    st->downstream_cb(piece, st->downstream_userdata);
}

char *qwen_transcribe_audio(qwen_ctx_t *ctx, const float *samples, int n_samples) {
    ctx->perf_total_ms = 0;
    ctx->perf_text_tokens = 0;
    ctx->perf_audio_ms = 1000.0 * (double)n_samples / (double)QWEN_SAMPLE_RATE;
    ctx->perf_encode_ms = 0;
    ctx->perf_decode_ms = 0;

    const float *audio_samples = samples;
    int audio_n_samples = n_samples;
    float *compacted_samples = NULL;
    if (ctx->skip_silence) {
        compacted_samples = compact_silence(samples, n_samples, &audio_n_samples);
        if (compacted_samples) audio_samples = compacted_samples;
        if (qwen_verbose >= 1) {
            float used_pct = 100.0f * (float)audio_n_samples /
                             (float)(n_samples > 0 ? n_samples : 1);
            float skipped_pct = 100.0f - used_pct;
            if (skipped_pct < 0.0f) skipped_pct = 0.0f;
            fprintf(stderr, "Silence skip: used %.1f%%, skipped %.1f%% (%d -> %d samples)\n",
                    used_pct, skipped_pct, n_samples, audio_n_samples);
        }
    }

    if (qwen_verbose >= 2)
        fprintf(stderr, "Audio: %d samples (%.1f seconds)\n",
                audio_n_samples, (float)audio_n_samples / QWEN_SAMPLE_RATE);

    /* Load tokenizer once for all segments */
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

    /* Determine segment boundaries.
     * Clamp search window to half the segment size so split points
     * can never overlap and produce zero-length segments. */
    float search = ctx->search_sec;
    if (search > ctx->segment_sec / 2.0f) search = ctx->segment_sec / 2.0f;
    int target_samples = (int)(ctx->segment_sec * QWEN_SAMPLE_RATE);
    int margin_samples = (int)(search * QWEN_SAMPLE_RATE);

    /* No splitting if segment_sec is 0 or audio fits in one segment */
    if (ctx->segment_sec <= 0 || audio_n_samples <= target_samples + margin_samples) {
        char *text = transcribe_segment(ctx, audio_samples, audio_n_samples, tokenizer, NULL, 0, NULL);
        qwen_tokenizer_free(tokenizer);
        free(compacted_samples);
        return text;
    }

    /* Build split points */
    int splits[128]; /* max 128 segments */
    int n_splits = 0;
    splits[n_splits++] = 0;

    int pos = 0;
    while (pos + target_samples + margin_samples < audio_n_samples) {
        int split = find_split_point(audio_samples, audio_n_samples,
                                     pos + target_samples, search);
        splits[n_splits++] = split;
        pos = split;
        if (n_splits >= 127) break; /* safety */
    }
    splits[n_splits] = audio_n_samples; /* end sentinel */

    if (qwen_verbose >= 2)
        fprintf(stderr, "Splitting into %d segments\n", n_splits);

    /* Transcribe each segment and concatenate */
    size_t result_cap = 4096;
    size_t result_len = 0;
    char *result = (char *)malloc(result_cap);
    result[0] = '\0';
    int min_samples = QWEN_SAMPLE_RATE / 2; /* 0.5s minimum, like official */
    int do_boundary_cleanup = (ctx->past_text_conditioning != 0);
    int use_past_conditioning = ctx->past_text_conditioning;
    int conditioning_collapses = 0;
    qwen_token_cb saved_cb = ctx->token_cb;
    void *saved_cb_userdata = ctx->token_cb_userdata;

    for (int s = 0; s < n_splits; s++) {
        int core_start = splits[s];
        int core_end = splits[s + 1];
        int seg_start = core_start;
        int seg_end = core_end;
        int seg_samples = seg_end - seg_start;

        if (qwen_verbose >= 2)
            fprintf(stderr, "Segment %d/%d: core %.1f-%.1fs, decode %.1f-%.1fs (%d samples)\n",
                    s + 1, n_splits,
                    (float)core_start / QWEN_SAMPLE_RATE,
                    (float)core_end / QWEN_SAMPLE_RATE,
                    (float)seg_start / QWEN_SAMPLE_RATE,
                    (float)seg_end / QWEN_SAMPLE_RATE,
                    seg_samples);

        /* Pad short segments to 0.5s with zeros (like official pipeline) */
        float *seg_buf = NULL;
        const float *seg_ptr = audio_samples + seg_start;
        if (seg_samples < min_samples) {
            seg_buf = (float *)calloc(min_samples, sizeof(float));
            memcpy(seg_buf, seg_ptr, seg_samples * sizeof(float));
            seg_ptr = seg_buf;
            seg_samples = min_samples;
        }

        int *past_tokens = NULL;
        int n_past_tokens = 0;
        if (use_past_conditioning && result_len > 0) {
            past_tokens = qwen_tokenizer_encode(tokenizer, result, &n_past_tokens);
            if (!past_tokens) n_past_tokens = 0;
        }

        segment_emit_state_t emit_state = {0};
        if (do_boundary_cleanup) {
            /* Cleanup mode buffers segment output and emits finalized text only. */
            ctx->token_cb = NULL;
            ctx->token_cb_userdata = NULL;
        } else if (saved_cb) {
            /* Fast segmented mode: emit each generated token immediately.
             * Add one separating space before the first token of the segment
             * only when needed and only if the first piece does not already
             * begin with whitespace/punctuation. */
            emit_state.downstream_cb = saved_cb;
            emit_state.downstream_userdata = saved_cb_userdata;
            emit_state.maybe_prepend_space =
                (result_len > 0 && !isspace((unsigned char)result[result_len - 1]));
            emit_state.saw_first_piece = 0;
            ctx->token_cb = segment_emit_cb;
            ctx->token_cb_userdata = &emit_state;
        }

        int seg_text_tokens = 0;
        char *seg_text = transcribe_segment(ctx, seg_ptr, seg_samples, tokenizer,
                                            past_tokens, n_past_tokens,
                                            &seg_text_tokens);
        if (do_boundary_cleanup &&
            use_past_conditioning && n_past_tokens > 0 &&
            should_retry_unconditioned_segment(result, seg_text,
                                               core_end - core_start,
                                               seg_text_tokens)) {
            conditioning_collapses++;
            if (qwen_verbose >= 2) {
                fprintf(stderr,
                        "Segment mode: retrying segment %d/%d without past-text conditioning "
                        "(core=%.1fs, tokens=%d)\n",
                        s + 1, n_splits,
                        (float)(core_end - core_start) / QWEN_SAMPLE_RATE,
                        seg_text_tokens);
            }
            /* Guardrail: if conditioned decode collapses or drifts,
             * retry this segment without past-text conditioning. */
            free(seg_text);
            seg_text = transcribe_segment(ctx, seg_ptr, seg_samples, tokenizer, NULL, 0,
                                          &seg_text_tokens);
            if (conditioning_collapses >= 2) {
                use_past_conditioning = 0;
                if (qwen_verbose >= 2) {
                    fprintf(stderr, "Segment mode: disabling past text conditioning after %d collapses\n",
                            conditioning_collapses);
                }
            }
        }
        ctx->token_cb = saved_cb;
        ctx->token_cb_userdata = saved_cb_userdata;

        free(past_tokens);
        free(seg_buf);
        if (!seg_text) continue;
        if (seg_text[0] == '\0') { free(seg_text); continue; }

        int cut_pos = 0;
        if (do_boundary_cleanup) {
            while (seg_text[cut_pos] != '\0' && isspace((unsigned char)seg_text[cut_pos])) cut_pos++;
        }
        if (seg_text[cut_pos] == '\0') {
            free(seg_text);
            continue;
        }

        size_t add_len = strlen(seg_text + cut_pos);
        int need_space = should_insert_boundary_space(
            result_len > 0 ? (int)(unsigned char)result[result_len - 1] : 0,
            (int)(unsigned char)seg_text[cut_pos]);
        size_t need = result_len + add_len + (size_t)(need_space ? 2 : 1);
        if (need > result_cap) {
            while (need > result_cap) result_cap *= 2;
            result = (char *)realloc(result, result_cap);
        }
        if (need_space) {
            result[result_len++] = ' ';
            if (do_boundary_cleanup && saved_cb) saved_cb(" ", saved_cb_userdata);
        }
        memcpy(result + result_len, seg_text + cut_pos, add_len);
        result_len += add_len;
        result[result_len] = '\0';
        if (do_boundary_cleanup && saved_cb) saved_cb(seg_text + cut_pos, saved_cb_userdata);
        free(seg_text);
    }

    ctx->token_cb = saved_cb;
    ctx->token_cb_userdata = saved_cb_userdata;
    qwen_tokenizer_free(tokenizer);
    free(compacted_samples);
    return result;
}
