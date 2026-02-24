/*
 * qwen_asr.c - Main API for Qwen3-ASR inference
 *
 * Pipeline: Load weights -> WAV -> Mel -> Encoder -> Build prompt ->
 *           Prefill decoder -> Autoregressive decode -> Tokenizer -> Text
 */

#include "qwen_asr.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_safetensors.h"
#include "qwen_asr_audio.h"
#include "qwen_asr_tokenizer.h"
#include "qwen_asr_internal.h"
#include "qwen_asr_quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <limits.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

/* Global verbose flag */
int qwen_verbose = 0;
int qwen_monitor = 0;

/* Global cache directory for .qcache files */
static char g_cache_dir[512] = {0};

void qwen_set_cache_dir(const char *dir) {
    if (dir && dir[0]) {
        snprintf(g_cache_dir, sizeof(g_cache_dir), "%s", dir);
    } else {
        g_cache_dir[0] = '\0';
    }
}

static const char *get_cache_dir(const qwen_ctx_t *ctx) {
    return g_cache_dir[0] ? g_cache_dir : ctx->model_dir;
}

void qwen_set_token_callback(qwen_ctx_t *ctx, qwen_token_cb cb, void *userdata) {
    ctx->token_cb = cb;
    ctx->token_cb_userdata = userdata;
}

static void reset_prompt_cache(qwen_ctx_t *ctx) {
    free(ctx->prompt_tokens);
    ctx->prompt_tokens = NULL;
    ctx->n_prompt_tokens = 0;

    free(ctx->force_prompt_tokens);
    ctx->force_prompt_tokens = NULL;
    ctx->n_force_prompt_tokens = 0;

    ctx->prompt_tokens_ready = 0;
}

/* ========================================================================
 * Internal load functions (defined in encoder/decoder .c files)
 * ======================================================================== */

extern int qwen_encoder_load(qwen_encoder_t *enc, multi_safetensors_t *ms,
                              const qwen_config_t *cfg);
extern int qwen_decoder_load(qwen_decoder_t *dec, multi_safetensors_t *ms,
                              const qwen_config_t *cfg);

/* ========================================================================
 * Config Detection
 * ======================================================================== */

/* Detect model variant from config.json or heuristics */
static int detect_config(qwen_ctx_t *ctx) {
    qwen_config_t *cfg = &ctx->config;

    /* Try to detect from number of shards:
     * 1.7B has 2 shards, 0.6B has 1 shard
     * But we can also check a specific weight shape. */

    /* Check if thinker.audio_tower.layers.17 exists (0.6B has 18 layers, 1.7B has 24) */
    multi_safetensors_t *ms = (multi_safetensors_t *)ctx->safetensors;

    /* Check for layer 18 (0-indexed) in encoder - if it exists, it's 1.7B */
    const safetensor_t *test = multi_safetensors_find(ms,
        "thinker.audio_tower.layers.18.self_attn.q_proj.weight", NULL);

    if (test) {
        /* 1.7B model */
        cfg->enc_d_model = 1024;
        cfg->enc_layers = 24;
        cfg->enc_heads = 16;
        cfg->enc_head_dim = 64;
        cfg->enc_ffn_dim = 4096;
        cfg->enc_output_dim = 2048;
        cfg->dec_hidden = 2048;
        cfg->dec_layers = 28;
        cfg->dec_heads = 16;
        cfg->dec_kv_heads = 8;
        cfg->dec_head_dim = 128;
        cfg->dec_intermediate = 6144;
        if (qwen_verbose >= 1) fprintf(stderr, "Detected: Qwen3-ASR-1.7B\n");
    } else {
        /* 0.6B model */
        cfg->enc_d_model = 896;
        cfg->enc_layers = 18;
        cfg->enc_heads = 14;
        cfg->enc_head_dim = 64;
        cfg->enc_ffn_dim = 3584;
        cfg->enc_output_dim = 1024;
        cfg->dec_hidden = 1024;
        cfg->dec_layers = 28;
        cfg->dec_heads = 16;
        cfg->dec_kv_heads = 8;
        cfg->dec_head_dim = 128;
        cfg->dec_intermediate = 3072;
        if (qwen_verbose >= 1) fprintf(stderr, "Detected: Qwen3-ASR-0.6B\n");
    }

    /* Common parameters */
    cfg->enc_n_window = 50;
    cfg->enc_n_window_infer = 800;
    cfg->enc_chunk_size = cfg->enc_n_window * 2; /* 100 */
    cfg->enc_conv_proj_dim = QWEN_CONV_HIDDEN * 16; /* 7680 */
    cfg->vocab_size = QWEN_VOCAB_SIZE;
    cfg->dec_rms_norm_eps = 1e-6f;
    cfg->dec_rope_theta = 1e6f;

    return 0;
}

/* ========================================================================
 * Pre-quantized Weight Cache (.qcache)
 *
 * After first-time BF16→Q4_K/Q8_0 quantization, serialize all quantized
 * projection weights to a binary cache file. Subsequent loads mmap the
 * cache, avoiding the expensive quantization step.
 *
 * Cache is saved alongside safetensors in model_dir.
 * Invalidated when safetensors total file size changes.
 *
 * Cache format:
 *   header (asr_qcache_header_t)
 *   for each encoder layer:
 *     wq_q8 | wk_q8 | wv_q8 | wo_q8 | fc1_q8 | fc2_q8
 *   encoder: conv_out_q8 | proj1_q8 | proj2_q8
 *   for each decoder layer:
 *     wq_q4k | wk_q4k | wv_q4k | wo_q4k | gate_up_q4k | down_q4k
 *   decoder: tok_embeddings_q4k
 * ======================================================================== */

#define ASR_QCACHE_MAGIC   0x31435141  /* "AQC1" */
#define ASR_QCACHE_VERSION 1

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t source_size;           /* safetensors total file size for validation */
    uint32_t n_enc_layers;
    uint32_t n_dec_layers;
    /* Encoder per-layer Q8_0 sizes */
    uint32_t enc_wq_q8_bytes;       /* per layer */
    uint32_t enc_wk_q8_bytes;
    uint32_t enc_wv_q8_bytes;
    uint32_t enc_wo_q8_bytes;
    uint32_t enc_fc1_q8_bytes;
    uint32_t enc_fc2_q8_bytes;
    /* Encoder one-time Q8_0 sizes */
    uint32_t enc_conv_out_q8_bytes;
    uint32_t enc_proj1_q8_bytes;
    uint32_t enc_proj2_q8_bytes;
    /* Decoder per-layer Q4_K sizes */
    uint32_t dec_wq_q4k_bytes;
    uint32_t dec_wk_q4k_bytes;
    uint32_t dec_wv_q4k_bytes;
    uint32_t dec_wo_q4k_bytes;
    uint32_t dec_gate_up_q4k_bytes;
    uint32_t dec_down_q4k_bytes;
    /* Decoder one-time Q4_K sizes */
    uint32_t dec_tok_emb_q4k_bytes;
    uint32_t reserved[3];
} asr_qcache_header_t;

static uint64_t get_safetensors_size(const char *model_dir) {
    char path[1024];
    uint64_t total = 0;
    struct stat st;

    snprintf(path, sizeof(path), "%s/model.safetensors", model_dir);
    if (stat(path, &st) == 0) total += (uint64_t)st.st_size;

    for (int i = 1; i <= 10; i++) {
        snprintf(path, sizeof(path), "%s/model-%05d-of-00002.safetensors", model_dir, i);
        if (stat(path, &st) == 0) total += (uint64_t)st.st_size;
        snprintf(path, sizeof(path), "%s/model-%05d-of-00003.safetensors", model_dir, i);
        if (stat(path, &st) == 0) total += (uint64_t)st.st_size;
    }
    return total;
}

static int save_asr_qcache(qwen_ctx_t *ctx) {
    const qwen_config_t *cfg = &ctx->config;
    char path[1024];
    snprintf(path, sizeof(path), "%s/model.qcache", get_cache_dir(ctx));

    /* Compute per-layer sizes */
    int d = cfg->enc_d_model;
    int ffn = cfg->enc_ffn_dim;
    int enc_d_blocks = d / QK8_0;

    uint32_t enc_wq_q8_bytes = (uint32_t)((size_t)d * enc_d_blocks * sizeof(block_q8_0));
    uint32_t enc_wk_q8_bytes = enc_wq_q8_bytes;
    uint32_t enc_wv_q8_bytes = enc_wq_q8_bytes;
    uint32_t enc_wo_q8_bytes = enc_wq_q8_bytes;
    uint32_t enc_fc1_q8_bytes = (uint32_t)((size_t)ffn * enc_d_blocks * sizeof(block_q8_0));
    uint32_t enc_fc2_q8_bytes = (uint32_t)((size_t)d * (ffn / QK8_0) * sizeof(block_q8_0));

    int conv_proj_dim = cfg->enc_conv_proj_dim;
    uint32_t enc_conv_out_q8_bytes = (uint32_t)((size_t)d * (conv_proj_dim / QK8_0) * sizeof(block_q8_0));
    uint32_t enc_proj1_q8_bytes = (uint32_t)((size_t)d * enc_d_blocks * sizeof(block_q8_0));
    uint32_t enc_proj2_q8_bytes = (uint32_t)((size_t)cfg->enc_output_dim * enc_d_blocks * sizeof(block_q8_0));

    int hidden = cfg->dec_hidden;
    int q_dim = cfg->dec_heads * cfg->dec_head_dim;
    int kv_dim = cfg->dec_kv_heads * cfg->dec_head_dim;
    int inter = cfg->dec_intermediate;
    int h_bpr = hidden / QK_K;  /* blocks per row for hidden dim */
    int q_bpr = q_dim / QK_K;
    int i_bpr = inter / QK_K;

    uint32_t dec_wq_q4k_bytes = (uint32_t)((size_t)q_dim * h_bpr * sizeof(block_q4_k));
    uint32_t dec_wk_q4k_bytes = (uint32_t)((size_t)kv_dim * h_bpr * sizeof(block_q4_k));
    uint32_t dec_wv_q4k_bytes = (uint32_t)((size_t)kv_dim * h_bpr * sizeof(block_q4_k));
    uint32_t dec_wo_q4k_bytes = (uint32_t)((size_t)hidden * q_bpr * sizeof(block_q4_k));
    uint32_t dec_gate_up_q4k_bytes = (uint32_t)((size_t)(2 * inter) * h_bpr * sizeof(block_q4_k));
    uint32_t dec_down_q4k_bytes = (uint32_t)((size_t)hidden * i_bpr * sizeof(block_q4_k));
    uint32_t dec_tok_emb_q4k_bytes = (uint32_t)((size_t)cfg->vocab_size * h_bpr * sizeof(block_q4_k));

    /* Build header */
    asr_qcache_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = ASR_QCACHE_MAGIC;
    hdr.version = ASR_QCACHE_VERSION;
    hdr.source_size = get_safetensors_size(ctx->model_dir);
    hdr.n_enc_layers = (uint32_t)cfg->enc_layers;
    hdr.n_dec_layers = (uint32_t)cfg->dec_layers;
    hdr.enc_wq_q8_bytes = enc_wq_q8_bytes;
    hdr.enc_wk_q8_bytes = enc_wk_q8_bytes;
    hdr.enc_wv_q8_bytes = enc_wv_q8_bytes;
    hdr.enc_wo_q8_bytes = enc_wo_q8_bytes;
    hdr.enc_fc1_q8_bytes = enc_fc1_q8_bytes;
    hdr.enc_fc2_q8_bytes = enc_fc2_q8_bytes;
    hdr.enc_conv_out_q8_bytes = enc_conv_out_q8_bytes;
    hdr.enc_proj1_q8_bytes = enc_proj1_q8_bytes;
    hdr.enc_proj2_q8_bytes = enc_proj2_q8_bytes;
    hdr.dec_wq_q4k_bytes = dec_wq_q4k_bytes;
    hdr.dec_wk_q4k_bytes = dec_wk_q4k_bytes;
    hdr.dec_wv_q4k_bytes = dec_wv_q4k_bytes;
    hdr.dec_wo_q4k_bytes = dec_wo_q4k_bytes;
    hdr.dec_gate_up_q4k_bytes = dec_gate_up_q4k_bytes;
    hdr.dec_down_q4k_bytes = dec_down_q4k_bytes;
    hdr.dec_tok_emb_q4k_bytes = dec_tok_emb_q4k_bytes;

    FILE *f = fopen(path, "wb");
    if (!f) {
        if (qwen_verbose >= 1)
            fprintf(stderr, "Warning: cannot create qcache at %s\n", path);
        return -1;
    }

    fwrite(&hdr, sizeof(hdr), 1, f);

    #define WRITE_OR_ZERO(ptr, nbytes) do { \
        if (ptr) fwrite(ptr, 1, nbytes, f); \
        else { void *z = calloc(1, nbytes); fwrite(z, 1, nbytes, f); free(z); } \
    } while(0)

    /* Write encoder layers */
    for (int i = 0; i < cfg->enc_layers; i++) {
        qwen_enc_layer_t *l = &ctx->encoder.layers[i];
        WRITE_OR_ZERO(l->wq_weight_q8, enc_wq_q8_bytes);
        WRITE_OR_ZERO(l->wk_weight_q8, enc_wk_q8_bytes);
        WRITE_OR_ZERO(l->wv_weight_q8, enc_wv_q8_bytes);
        WRITE_OR_ZERO(l->wo_weight_q8, enc_wo_q8_bytes);
        WRITE_OR_ZERO(l->fc1_weight_q8, enc_fc1_q8_bytes);
        WRITE_OR_ZERO(l->fc2_weight_q8, enc_fc2_q8_bytes);
    }

    /* Write encoder one-time weights */
    WRITE_OR_ZERO(ctx->encoder.conv_out_weight_q8, enc_conv_out_q8_bytes);
    WRITE_OR_ZERO(ctx->encoder.proj1_weight_q8, enc_proj1_q8_bytes);
    WRITE_OR_ZERO(ctx->encoder.proj2_weight_q8, enc_proj2_q8_bytes);

    /* Write decoder layers */
    for (int i = 0; i < cfg->dec_layers; i++) {
        qwen_dec_layer_t *l = &ctx->decoder.layers[i];
        WRITE_OR_ZERO(l->wq_weight_q4k, dec_wq_q4k_bytes);
        WRITE_OR_ZERO(l->wk_weight_q4k, dec_wk_q4k_bytes);
        WRITE_OR_ZERO(l->wv_weight_q4k, dec_wv_q4k_bytes);
        WRITE_OR_ZERO(l->wo_weight_q4k, dec_wo_q4k_bytes);
        WRITE_OR_ZERO(l->gate_up_fused_q4k, dec_gate_up_q4k_bytes);
        WRITE_OR_ZERO(l->down_weight_q4k, dec_down_q4k_bytes);
    }

    /* Write decoder token embeddings Q4_K */
    WRITE_OR_ZERO(ctx->decoder.tok_embeddings_q4k, dec_tok_emb_q4k_bytes);

    #undef WRITE_OR_ZERO

    fclose(f);
    if (qwen_verbose >= 1)
        fprintf(stderr, "Saved quantized cache to %s\n", path);
    return 0;
}

static int load_asr_qcache(qwen_ctx_t *ctx) {
    const qwen_config_t *cfg = &ctx->config;
    char path[1024];
    snprintf(path, sizeof(path), "%s/model.qcache", get_cache_dir(ctx));

    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return -1; }
    size_t file_size = (size_t)st.st_size;
    if (file_size < sizeof(asr_qcache_header_t)) { close(fd); return -1; }

    void *mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) return -1;

    const asr_qcache_header_t *hdr = (const asr_qcache_header_t *)mapped;

    /* Validate header */
    if (hdr->magic != ASR_QCACHE_MAGIC || hdr->version != ASR_QCACHE_VERSION) {
        munmap(mapped, file_size);
        return -1;
    }
    if ((int)hdr->n_enc_layers != cfg->enc_layers ||
        (int)hdr->n_dec_layers != cfg->dec_layers) {
        munmap(mapped, file_size);
        return -1;
    }

    /* Validate source file size */
    uint64_t expected_src = get_safetensors_size(ctx->model_dir);
    if (hdr->source_size != expected_src) {
        if (qwen_verbose >= 1)
            fprintf(stderr, "qcache: source size mismatch (cache=%llu, actual=%llu), re-quantizing\n",
                    (unsigned long long)hdr->source_size, (unsigned long long)expected_src);
        munmap(mapped, file_size);
        return -1;
    }

    /* Validate total file size */
    size_t enc_per_layer = (size_t)hdr->enc_wq_q8_bytes + hdr->enc_wk_q8_bytes +
                            hdr->enc_wv_q8_bytes + hdr->enc_wo_q8_bytes +
                            hdr->enc_fc1_q8_bytes + hdr->enc_fc2_q8_bytes;
    size_t dec_per_layer = (size_t)hdr->dec_wq_q4k_bytes + hdr->dec_wk_q4k_bytes +
                            hdr->dec_wv_q4k_bytes + hdr->dec_wo_q4k_bytes +
                            hdr->dec_gate_up_q4k_bytes + hdr->dec_down_q4k_bytes;
    size_t expected_size = sizeof(asr_qcache_header_t) +
                           enc_per_layer * hdr->n_enc_layers +
                           (size_t)hdr->enc_conv_out_q8_bytes +
                           hdr->enc_proj1_q8_bytes + hdr->enc_proj2_q8_bytes +
                           dec_per_layer * hdr->n_dec_layers +
                           hdr->dec_tok_emb_q4k_bytes;
    if (file_size < expected_size) {
        munmap(mapped, file_size);
        return -1;
    }

    /* Copy weights from mmap into malloc'd buffers */
    const uint8_t *ptr = (const uint8_t *)mapped + sizeof(asr_qcache_header_t);

    #define CACHE_COPY(dst, type, n_bytes) do { \
        if ((n_bytes) > 0) { \
            dst = (type)malloc(n_bytes); \
            if (dst) memcpy(dst, ptr, n_bytes); \
            ptr += (n_bytes); \
        } \
    } while(0)

    /* Read encoder layers */
    for (int i = 0; i < cfg->enc_layers; i++) {
        qwen_enc_layer_t *l = &ctx->encoder.layers[i];
        CACHE_COPY(l->wq_weight_q8, block_q8_0 *, hdr->enc_wq_q8_bytes);
        CACHE_COPY(l->wk_weight_q8, block_q8_0 *, hdr->enc_wk_q8_bytes);
        CACHE_COPY(l->wv_weight_q8, block_q8_0 *, hdr->enc_wv_q8_bytes);
        CACHE_COPY(l->wo_weight_q8, block_q8_0 *, hdr->enc_wo_q8_bytes);
        CACHE_COPY(l->fc1_weight_q8, block_q8_0 *, hdr->enc_fc1_q8_bytes);
        CACHE_COPY(l->fc2_weight_q8, block_q8_0 *, hdr->enc_fc2_q8_bytes);
    }

    /* Read encoder one-time weights */
    CACHE_COPY(ctx->encoder.conv_out_weight_q8, block_q8_0 *, hdr->enc_conv_out_q8_bytes);
    CACHE_COPY(ctx->encoder.proj1_weight_q8, block_q8_0 *, hdr->enc_proj1_q8_bytes);
    CACHE_COPY(ctx->encoder.proj2_weight_q8, block_q8_0 *, hdr->enc_proj2_q8_bytes);

    /* Read decoder layers */
    for (int i = 0; i < cfg->dec_layers; i++) {
        qwen_dec_layer_t *l = &ctx->decoder.layers[i];
        CACHE_COPY(l->wq_weight_q4k, block_q4_k *, hdr->dec_wq_q4k_bytes);
        CACHE_COPY(l->wk_weight_q4k, block_q4_k *, hdr->dec_wk_q4k_bytes);
        CACHE_COPY(l->wv_weight_q4k, block_q4_k *, hdr->dec_wv_q4k_bytes);
        CACHE_COPY(l->wo_weight_q4k, block_q4_k *, hdr->dec_wo_q4k_bytes);
        CACHE_COPY(l->gate_up_fused_q4k, block_q4_k *, hdr->dec_gate_up_q4k_bytes);
        CACHE_COPY(l->down_weight_q4k, block_q4_k *, hdr->dec_down_q4k_bytes);
    }

    /* Read decoder token embeddings */
    CACHE_COPY(ctx->decoder.tok_embeddings_q4k, block_q4_k *, hdr->dec_tok_emb_q4k_bytes);

    #undef CACHE_COPY

    munmap(mapped, file_size);

    if (qwen_verbose >= 1)
        fprintf(stderr, "Loaded quantized cache from %s\n", path);
    return 0;
}

/* ========================================================================
 * Model Loading
 * ======================================================================== */

qwen_ctx_t *qwen_load(const char *model_dir) {
    qwen_ctx_t *ctx = (qwen_ctx_t *)calloc(1, sizeof(qwen_ctx_t));
    if (!ctx) return NULL;
    snprintf(ctx->model_dir, sizeof(ctx->model_dir), "%s", model_dir);

    /* Open safetensors (multi-shard) */
    if (qwen_verbose >= 1)
        fprintf(stderr, "Loading model from %s\n", model_dir);

    multi_safetensors_t *ms = multi_safetensors_open(model_dir);
    if (!ms) {
        fprintf(stderr, "qwen_load: cannot open safetensors in %s\n", model_dir);
        free(ctx);
        return NULL;
    }
    ctx->safetensors = ms;

    /* Detect model configuration */
    detect_config(ctx);

    /* Try loading quantized weight cache */
    int cache_ok = load_asr_qcache(ctx);
    if (cache_ok == 0 && qwen_verbose >= 1) {
        fprintf(stderr, "Loaded quantized cache, skipping quantization\n");
    }

    /* Load encoder weights (skips quantization for weights already in cache) */
    if (qwen_verbose >= 1) fprintf(stderr, "Loading encoder weights...\n");
    if (qwen_encoder_load(&ctx->encoder, ms, &ctx->config) != 0) {
        fprintf(stderr, "qwen_load: failed to load encoder\n");
        qwen_free(ctx);
        return NULL;
    }

    /* Load decoder weights (skips quantization for weights already in cache) */
    if (qwen_verbose >= 1) fprintf(stderr, "Loading decoder weights...\n");
    if (qwen_decoder_load(&ctx->decoder, ms, &ctx->config) != 0) {
        fprintf(stderr, "qwen_load: failed to load decoder\n");
        qwen_free(ctx);
        return NULL;
    }

    /* Save cache if it wasn't loaded (first-time quantization) */
    if (cache_ok != 0) {
        save_asr_qcache(ctx);
    }

    /* Default transcription mode: full-audio offline decode (no splitting). */
    ctx->segment_sec = 0.0f;
    ctx->search_sec = 3.0f;

    /* Default streaming parameters */
    ctx->stream_chunk_sec = 2.0f;
    ctx->stream_rollback = 5;
    ctx->stream_unfixed_chunks = 2;
    ctx->stream_max_new_tokens = 32;
    ctx->past_text_conditioning = 1;
    ctx->skip_silence = 0;

    if (qwen_verbose >= 1) fprintf(stderr, "Model loaded.\n");
    return ctx;
}

/* ========================================================================
 * Free
 * ======================================================================== */

void qwen_free(qwen_ctx_t *ctx) {
    if (!ctx) return;

    #define FREE0(p) do { free(p); (p) = NULL; } while (0)

    /* Encoder conv stem */
    FREE0(ctx->encoder.conv1_weight); FREE0(ctx->encoder.conv1_bias);
    FREE0(ctx->encoder.conv2_weight); FREE0(ctx->encoder.conv2_bias);
    FREE0(ctx->encoder.conv3_weight); FREE0(ctx->encoder.conv3_bias);
    FREE0(ctx->encoder.conv_out_weight_q8);

    /* Encoder layers (weights are Q8_0, all allocated) */
    for (int i = 0; i < ctx->config.enc_layers; i++) {
        qwen_enc_layer_t *l = &ctx->encoder.layers[i];
        FREE0(l->wq_weight_q8); FREE0(l->wq_bias);
        FREE0(l->wk_weight_q8); FREE0(l->wk_bias);
        FREE0(l->wv_weight_q8); FREE0(l->wv_bias);
        FREE0(l->wo_weight_q8); FREE0(l->wo_bias);
        FREE0(l->attn_norm_weight); FREE0(l->attn_norm_bias);
        FREE0(l->fc1_weight_q8); FREE0(l->fc1_bias);
        FREE0(l->fc2_weight_q8); FREE0(l->fc2_bias);
        FREE0(l->ffn_norm_weight); FREE0(l->ffn_norm_bias);
    }
    FREE0(ctx->encoder.ln_post_weight); FREE0(ctx->encoder.ln_post_bias);
    FREE0(ctx->encoder.proj1_weight_q8); FREE0(ctx->encoder.proj1_bias);
    FREE0(ctx->encoder.proj2_weight_q8); FREE0(ctx->encoder.proj2_bias);

    /* Decoder layers (Q4_K weights are all malloc'd, must be freed) */
    for (int i = 0; i < ctx->config.dec_layers; i++) {
        qwen_dec_layer_t *l = &ctx->decoder.layers[i];
        FREE0(l->wq_weight_q4k); FREE0(l->wk_weight_q4k);
        FREE0(l->wv_weight_q4k); FREE0(l->wo_weight_q4k);
        FREE0(l->q_norm_weight); FREE0(l->k_norm_weight);
        FREE0(l->input_norm); FREE0(l->post_attn_norm);
        FREE0(l->down_weight_q4k);
        FREE0(l->gate_up_fused_q4k);
    }
    FREE0(ctx->decoder.norm);
    FREE0(ctx->decoder.tok_embeddings_q4k);

    #undef FREE0

    /* KV cache */
    free(ctx->kv_cache_k);
    free(ctx->kv_cache_v);

    /* Persistent decoder buffers */
    free(ctx->dec_x); free(ctx->dec_x_norm);
    free(ctx->dec_q); free(ctx->dec_k); free(ctx->dec_v);
    free(ctx->dec_attn_out); free(ctx->dec_proj_out);
    free(ctx->dec_gate); free(ctx->dec_up); free(ctx->dec_ffn_out);
    free(ctx->dec_rope_cos); free(ctx->dec_rope_sin);

    /* Persistent decoder prefill buffers */
    free(ctx->pref_x); free(ctx->pref_x_norm);
    free(ctx->pref_q); free(ctx->pref_k); free(ctx->pref_v);
    free(ctx->pref_attn_out); free(ctx->pref_proj_out); free(ctx->pref_ffn_out);
    free(ctx->pref_gate); free(ctx->pref_gate_up);

    /* Decoder RoPE caches */
    free(ctx->rope_cache_cos); free(ctx->rope_cache_sin);
    free(ctx->rope_inv_freq);

    /* GEMM workspace */
    qwen_gemm_workspace_free();

    /* Prompt/language options */
    free(ctx->prompt);
    free(ctx->force_language);
    free(ctx->prompt_tokens);
    free(ctx->force_prompt_tokens);

    /* Close safetensors */
    if (ctx->safetensors) {
        multi_safetensors_close((multi_safetensors_t *)ctx->safetensors);
    }

    free(ctx);
}

/* ========================================================================
 * Transcription
 * ======================================================================== */

/* Convert a single token embedding from bf16 to f32 */
void tok_embed_bf16_to_f32(float *dst, const uint16_t *tok_emb_bf16,
                                  int token_id, int dim) {
    const uint16_t *src = tok_emb_bf16 + (size_t)token_id * dim;
    for (int i = 0; i < dim; i++) {
        uint32_t f32_bits = ((uint32_t)src[i]) << 16;
        memcpy(&dst[i], &f32_bits, sizeof(float));
    }
}

double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* Prepare cached prompt-related tokens once per context. */
int prepare_prompt_tokens(qwen_ctx_t *ctx, qwen_tokenizer_t *tokenizer) {
    if (ctx->prompt_tokens_ready) return 0;

    reset_prompt_cache(ctx);

    if (ctx->prompt && ctx->prompt[0] != '\0') {
        ctx->prompt_tokens = qwen_tokenizer_encode(tokenizer, ctx->prompt, &ctx->n_prompt_tokens);
        if (!ctx->prompt_tokens) {
            fprintf(stderr, "qwen: failed to encode --prompt text\n");
            return -1;
        }
    }

    if (ctx->force_language && ctx->force_language[0] != '\0') {
        char force_text[128];
        snprintf(force_text, sizeof(force_text), "language %s", ctx->force_language);

        int n_lang_txt = 0;
        int *lang_txt_tokens = qwen_tokenizer_encode(tokenizer, force_text, &n_lang_txt);
        if (!lang_txt_tokens) {
            fprintf(stderr, "qwen: failed to encode --language text\n");
            return -1;
        }

        ctx->n_force_prompt_tokens = n_lang_txt + 1; /* + <asr_text> marker */
        ctx->force_prompt_tokens = (int *)malloc((size_t)ctx->n_force_prompt_tokens * sizeof(int));
        if (!ctx->force_prompt_tokens) {
            free(lang_txt_tokens);
            return -1;
        }
        if (n_lang_txt > 0) {
            memcpy(ctx->force_prompt_tokens, lang_txt_tokens, (size_t)n_lang_txt * sizeof(int));
        }
        ctx->force_prompt_tokens[n_lang_txt] = QWEN_TOKEN_ASR_TEXT;
        free(lang_txt_tokens);
    }

    ctx->prompt_tokens_ready = 1;
    return 0;
}
