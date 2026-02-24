/*
 * qwen_tts.c - Main API for Qwen3-TTS C inference engine
 *
 * Contains:
 *   - Minimal JSON helpers for config.json parsing
 *   - Config loading (talker + speech_tokenizer)
 *   - Weight loading from SafeTensors (mmap)
 *   - qwen_tts_load() / qwen_tts_free()
 *
 * Generation logic (generate / generate_stream) is in qwen_tts_generate.c.
 */

#include "qwen_tts.h"
#include "qwen_tts_kernels.h"
#include "qwen_tts_quant.h"
#include "qwen_tts_safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef __EMSCRIPTEN__
#include <dirent.h>
#include <unistd.h>
#endif

int qwen_tts_verbose = 0;
const char *qwen_tts_cache_dir_override = NULL;

/* ========================================================================
 * Timing helpers
 * ======================================================================== */

static double time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ========================================================================
 * Minimal JSON helpers
 *
 * These work on raw JSON text, finding keys at a given nesting level.
 * Not a full parser — just enough for config.json.
 * ======================================================================== */

static void jskip_ws(const char **p) {
    while (**p == ' ' || **p == '\n' || **p == '\r' || **p == '\t') (*p)++;
}

/* Skip a JSON value (string, number, object, array, bool, null) */
static void jskip_value(const char **p) {
    jskip_ws(p);
    if (**p == '"') {
        (*p)++;
        while (**p && !(**p == '"' && *(*p-1) != '\\')) (*p)++;
        if (**p == '"') (*p)++;
    } else if (**p == '{') {
        int depth = 1; (*p)++;
        while (**p && depth > 0) {
            if (**p == '{') depth++;
            else if (**p == '}') depth--;
            else if (**p == '"') { (*p)++; while (**p && !(**p == '"' && *(*p-1) != '\\')) (*p)++; }
            (*p)++;
        }
    } else if (**p == '[') {
        int depth = 1; (*p)++;
        while (**p && depth > 0) {
            if (**p == '[') depth++;
            else if (**p == ']') depth--;
            else if (**p == '"') { (*p)++; while (**p && !(**p == '"' && *(*p-1) != '\\')) (*p)++; }
            (*p)++;
        }
    } else {
        while (**p && **p != ',' && **p != '}' && **p != ']') (*p)++;
    }
}

/* Find a key in the current JSON object level. Returns pointer to the value or NULL. */
static const char *jfind_key(const char *json, const char *key) {
    const char *p = json;
    jskip_ws(&p);
    if (*p != '{') return NULL;
    p++;
    while (1) {
        jskip_ws(&p);
        if (*p == '}' || *p == '\0') return NULL;
        /* Parse key string */
        if (*p != '"') return NULL;
        p++;
        const char *ks = p;
        while (*p && *p != '"') { if (*p == '\\') p++; p++; }
        int klen = (int)(p - ks);
        if (*p == '"') p++;
        jskip_ws(&p);
        if (*p == ':') p++;
        jskip_ws(&p);
        if (klen == (int)strlen(key) && memcmp(ks, key, klen) == 0) {
            return p;  /* Points to start of value */
        }
        jskip_value(&p);
        jskip_ws(&p);
        if (*p == ',') p++;
    }
}

/* Navigate nested path: "talker_config.vocab_size" */
static const char *jfind_path(const char *json, const char *path) {
    char buf[256];
    strncpy(buf, path, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    const char *p = json;
    char *tok = strtok(buf, ".");
    while (tok) {
        p = jfind_key(p, tok);
        if (!p) return NULL;
        char *next = strtok(NULL, ".");
        if (!next) return p;
        tok = next;
    }
    return p;
}

static int jget_int(const char *json, const char *path, int def) {
    const char *v = jfind_path(json, path);
    if (!v) return def;
    return (int)strtol(v, NULL, 10);
}

static float jget_float(const char *json, const char *path, float def) {
    const char *v = jfind_path(json, path);
    if (!v) return def;
    return strtof(v, NULL);
}

/* Get a JSON string value. Returns pointer to out buffer. */
__attribute__((unused))
static const char *jget_str(const char *json, const char *path, char *out, int max_len) {
    const char *v = jfind_path(json, path);
    if (!v || *v != '"') { out[0] = '\0'; return out; }
    v++;
    int i = 0;
    while (*v && *v != '"' && i < max_len - 1) {
        if (*v == '\\') { v++; }
        out[i++] = *v++;
    }
    out[i] = '\0';
    return out;
}

/* Get a JSON integer array value. Returns number of elements parsed. */
static int jget_int_array(const char *json, const char *path, int *out, int max_n) {
    const char *v = jfind_path(json, path);
    if (!v || *v != '[') return 0;
    v++;
    int n = 0;
    while (*v && *v != ']' && n < max_n) {
        jskip_ws(&v);
        if (*v == ']') break;
        out[n++] = (int)strtol(v, (char **)&v, 10);
        jskip_ws(&v);
        if (*v == ',') v++;
    }
    return n;
}

/* Parse speaker map: "spk_id": {"name": [id1, id2, ...], ...} */
static void jparse_speaker_map(const char *json, const char *path,
                                int *n_speakers, char ***names, int **ids) {
    const char *v = jfind_path(json, path);
    if (!v || *v != '{') { *n_speakers = 0; return; }
    const char *p = v + 1;

    /* Count speakers first */
    int count = 0;
    const char *pp = p;
    while (*pp && *pp != '}') {
        jskip_ws(&pp);
        if (*pp == '"') { jskip_value(&pp); jskip_ws(&pp); if (*pp == ':') pp++; jskip_value(&pp); count++; }
        jskip_ws(&pp);
        if (*pp == ',') pp++;
    }

    *n_speakers = count;
    *names = (char **)calloc(count, sizeof(char *));
    *ids = (int *)calloc(count, sizeof(int));

    p = v + 1;
    for (int i = 0; i < count; i++) {
        jskip_ws(&p);
        if (*p != '"') break;
        p++;
        const char *ns = p;
        while (*p && *p != '"') p++;
        int nlen = (int)(p - ns);
        (*names)[i] = (char *)malloc(nlen + 1);
        memcpy((*names)[i], ns, nlen);
        (*names)[i][nlen] = '\0';
        if (*p == '"') p++;
        jskip_ws(&p);
        if (*p == ':') p++;
        jskip_ws(&p);
        /* Value can be an integer or array; take first int from array or the integer */
        if (*p == '[') {
            p++;
            jskip_ws(&p);
            (*ids)[i] = (int)strtol(p, (char **)&p, 10);
            while (*p && *p != ']') p++;
            if (*p == ']') p++;
        } else {
            (*ids)[i] = (int)strtol(p, (char **)&p, 10);
        }
        jskip_ws(&p);
        if (*p == ',') p++;
    }
}

/* Read entire file into malloc'd buffer, null-terminated. */
static char *read_file_text(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(size + 1);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, size, f);
    buf[size] = '\0';
    fclose(f);
    return buf;
}

/* ========================================================================
 * Config Loading
 * ======================================================================== */

static int load_config(qwen_tts_ctx_t *ctx) {
    char path[1024];
    qwen_tts_config_t *cfg = &ctx->config;

    /* ---- Load main config.json ---- */
    snprintf(path, sizeof(path), "%s/config.json", ctx->model_dir);
    char *json = read_file_text(path);
    if (!json) {
        fprintf(stderr, "Error: cannot read %s\n", path);
        return -1;
    }

    /* Talker config */
    cfg->talker_vocab_size      = jget_int(json, "talker_config.vocab_size", QWEN_TTS_TALKER_VOCAB);
    cfg->talker_hidden          = jget_int(json, "talker_config.hidden_size", QWEN_TTS_TALKER_HIDDEN);
    cfg->talker_intermediate    = jget_int(json, "talker_config.intermediate_size", QWEN_TTS_TALKER_INTERMEDIATE);
    cfg->talker_layers          = jget_int(json, "talker_config.num_hidden_layers", QWEN_TTS_TALKER_LAYERS);
    cfg->talker_heads           = jget_int(json, "talker_config.num_attention_heads", QWEN_TTS_TALKER_HEADS);
    cfg->talker_kv_heads        = jget_int(json, "talker_config.num_key_value_heads", QWEN_TTS_TALKER_KV_HEADS);
    cfg->talker_head_dim        = jget_int(json, "talker_config.head_dim", 0);
    if (cfg->talker_head_dim <= 0 && cfg->talker_heads > 0) {
        cfg->talker_head_dim = cfg->talker_hidden / cfg->talker_heads;
    }
    cfg->talker_text_hidden     = jget_int(json, "talker_config.text_hidden_size", QWEN_TTS_TALKER_TEXT_HIDDEN);
    cfg->talker_text_vocab      = jget_int(json, "talker_config.text_vocab_size", QWEN_TTS_TALKER_TEXT_VOCAB);
    cfg->num_code_groups        = jget_int(json, "talker_config.num_code_groups", QWEN_TTS_NUM_CODE_GROUPS);
    cfg->talker_rms_norm_eps    = jget_float(json, "talker_config.rms_norm_eps", 1e-6f);
    cfg->talker_rope_theta      = jget_float(json, "talker_config.rope_theta", 10000.0f);

    /* M-RoPE sections */
    cfg->mrope_section[0] = 16; cfg->mrope_section[1] = 16; cfg->mrope_section[2] = 0;
    jget_int_array(json, "talker_config.rope_scaling.mrope_section", cfg->mrope_section, 3);

    /* Sub-talker config */
    cfg->subtalker_vocab_size   = jget_int(json, "talker_config.code_predictor_config.vocab_size", QWEN_TTS_SUBTALKER_VOCAB);
    cfg->subtalker_hidden       = jget_int(json, "talker_config.code_predictor_config.hidden_size", QWEN_TTS_SUBTALKER_HIDDEN);
    cfg->subtalker_intermediate = jget_int(json, "talker_config.code_predictor_config.intermediate_size", QWEN_TTS_SUBTALKER_INTERMEDIATE);
    cfg->subtalker_layers       = jget_int(json, "talker_config.code_predictor_config.num_hidden_layers", QWEN_TTS_SUBTALKER_LAYERS);
    cfg->subtalker_heads        = jget_int(json, "talker_config.code_predictor_config.num_attention_heads", QWEN_TTS_SUBTALKER_HEADS);
    cfg->subtalker_kv_heads     = jget_int(json, "talker_config.code_predictor_config.num_key_value_heads", QWEN_TTS_SUBTALKER_KV_HEADS);
    cfg->subtalker_head_dim     = jget_int(json, "talker_config.code_predictor_config.head_dim", QWEN_TTS_SUBTALKER_HEAD_DIM);

    /* Codec special token IDs */
    cfg->codec_pad_id       = jget_int(json, "talker_config.codec_pad_id", QWEN_TTS_CODEC_PAD);
    cfg->codec_bos_id       = jget_int(json, "talker_config.codec_bos_id", QWEN_TTS_CODEC_BOS);
    cfg->codec_eos_id       = jget_int(json, "talker_config.codec_eos_token_id", QWEN_TTS_CODEC_EOS);
    cfg->codec_nothink_id   = jget_int(json, "talker_config.codec_nothink_id", QWEN_TTS_CODEC_NOTHINK);
    cfg->codec_think_id     = jget_int(json, "talker_config.codec_think_id", QWEN_TTS_CODEC_THINK);
    cfg->codec_think_bos_id = jget_int(json, "talker_config.codec_think_bos_id", QWEN_TTS_CODEC_THINK_BOS);
    cfg->codec_think_eos_id = jget_int(json, "talker_config.codec_think_eos_id", QWEN_TTS_CODEC_THINK_EOS);

    /* Speaker and language maps */
    jparse_speaker_map(json, "talker_config.spk_id", &cfg->n_speakers, &cfg->speaker_names, &cfg->speaker_ids);
    jparse_speaker_map(json, "talker_config.codec_language_id", &cfg->n_languages, &cfg->language_names, &cfg->language_ids);

    free(json);

    /* Basic shape/config sanity checks to avoid silent model mismatch. */
    if (cfg->talker_heads <= 0 || cfg->talker_kv_heads <= 0 || cfg->talker_head_dim <= 0) {
        fprintf(stderr, "Error: invalid talker attention config (heads=%d kv_heads=%d head_dim=%d)\n",
                cfg->talker_heads, cfg->talker_kv_heads, cfg->talker_head_dim);
        return -1;
    }
    if (cfg->talker_heads % cfg->talker_kv_heads != 0) {
        fprintf(stderr, "Error: talker heads (%d) must be divisible by kv heads (%d)\n",
                cfg->talker_heads, cfg->talker_kv_heads);
        return -1;
    }
    if (cfg->talker_head_dim > 512 || cfg->subtalker_head_dim > 512) {
        fprintf(stderr, "Error: unsupported head_dim (talker=%d subtalker=%d, max=512)\n",
                cfg->talker_head_dim, cfg->subtalker_head_dim);
        return -1;
    }

    /* ---- Load speech_tokenizer config ---- */
    snprintf(path, sizeof(path), "%s/speech_tokenizer/config.json", ctx->model_dir);
    json = read_file_text(path);
    if (!json) {
        fprintf(stderr, "Error: cannot read %s\n", path);
        return -1;
    }

    cfg->codec_num_quantizers   = jget_int(json, "decoder_config.num_quantizers", QWEN_TTS_CODEC_NUM_QUANTIZERS);
    cfg->codec_codebook_size    = jget_int(json, "decoder_config.codebook_size", QWEN_TTS_CODEC_CODEBOOK_SIZE);
    cfg->codec_codebook_dim     = jget_int(json, "decoder_config.codebook_dim", 128);
    cfg->codec_hidden           = jget_int(json, "decoder_config.hidden_size", QWEN_TTS_CODEC_HIDDEN);
    cfg->codec_latent           = jget_int(json, "decoder_config.latent_dim", QWEN_TTS_CODEC_LATENT);
    cfg->codec_layers           = jget_int(json, "decoder_config.num_hidden_layers", QWEN_TTS_CODEC_LAYERS);
    cfg->codec_heads            = jget_int(json, "decoder_config.num_attention_heads", QWEN_TTS_CODEC_HEADS);
    cfg->codec_kv_heads         = jget_int(json, "decoder_config.num_key_value_heads", QWEN_TTS_CODEC_KV_HEADS);
    cfg->codec_intermediate     = jget_int(json, "decoder_config.intermediate_size", QWEN_TTS_CODEC_INTERMEDIATE);
    cfg->codec_sliding_window   = jget_int(json, "decoder_config.sliding_window", QWEN_TTS_CODEC_SLIDING_WINDOW);
    cfg->codec_decoder_dim      = jget_int(json, "decoder_config.decoder_dim", QWEN_TTS_CODEC_DECODER_DIM);
    cfg->codec_rms_norm_eps     = jget_float(json, "decoder_config.rms_norm_eps", 1e-5f);
    cfg->codec_layer_scale      = jget_float(json, "decoder_config.layer_scale_initial_scale", 0.01f);

    int rates[4] = {8, 5, 4, 3};
    jget_int_array(json, "decoder_config.upsample_rates", rates, 4);
    memcpy(cfg->codec_upsample_rates, rates, sizeof(rates));

    int ratios[2] = {2, 2};
    jget_int_array(json, "decoder_config.upsampling_ratios", ratios, 2);
    memcpy(cfg->codec_upsampling_ratios, ratios, sizeof(ratios));

    free(json);

    if (qwen_tts_verbose >= 1) {
        fprintf(stderr, "Config loaded:\n");
        fprintf(stderr, "  Talker: %d layers, hidden=%d, heads=%d/%d, head_dim=%d\n",
                cfg->talker_layers, cfg->talker_hidden, cfg->talker_heads, cfg->talker_kv_heads, cfg->talker_head_dim);
        fprintf(stderr, "  Sub-talker: %d layers, hidden=%d, heads=%d/%d, head_dim=%d\n",
                cfg->subtalker_layers, cfg->subtalker_hidden, cfg->subtalker_heads, cfg->subtalker_kv_heads, cfg->subtalker_head_dim);
        fprintf(stderr, "  Codec: %d layers, hidden=%d, codebook_dim=%d, decoder_dim=%d\n",
                cfg->codec_layers, cfg->codec_hidden, cfg->codec_codebook_dim, cfg->codec_decoder_dim);
        fprintf(stderr, "  M-RoPE sections: [%d, %d, %d]\n",
                cfg->mrope_section[0], cfg->mrope_section[1], cfg->mrope_section[2]);
        fprintf(stderr, "  Speakers: %d, Languages: %d\n", cfg->n_speakers, cfg->n_languages);
    }

    return 0;
}

/* ========================================================================
 * Weight Loading Helpers
 * ======================================================================== */

/* Convenience macros for loading weights from safetensors */
#define GET_BF16(ms, name)      (uint16_t *)multi_safetensors_get_bf16(ms, name, NULL, NULL)
#define GET_F32(ms, name)       (float *)multi_safetensors_get_f32(ms, name, NULL, NULL)
#define LOAD_F32(ms, name)      multi_safetensors_load_f32(ms, name, NULL, NULL)

#define GET_BF16_CHECK(dst, ms, name) do { \
    dst = GET_BF16(ms, name); \
    if (!dst && qwen_tts_verbose >= 2) fprintf(stderr, "  Warning: tensor not found: %s\n", name); \
} while(0)

#define GET_F32_CHECK(dst, ms, name) do { \
    dst = GET_F32(ms, name); \
    if (!dst && qwen_tts_verbose >= 2) fprintf(stderr, "  Warning: tensor not found: %s\n", name); \
} while(0)

#define LOAD_F32_CHECK(dst, ms, name) do { \
    dst = LOAD_F32(ms, name); \
    if (!dst && qwen_tts_verbose >= 2) fprintf(stderr, "  Warning: tensor not found: %s\n", name); \
} while(0)

static int expect_tensor_bf16_2d(const multi_safetensors_t *ms, const char *name,
                                 int64_t dim0, int64_t dim1) {
    void *data = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &data);
    if (!t || !data) {
        fprintf(stderr, "Error: missing required tensor: %s\n", name);
        return -1;
    }
    if (!t->dtype || strcmp(t->dtype, "BF16") != 0) {
        fprintf(stderr, "Error: tensor %s dtype mismatch: expected BF16, got %s\n",
                name, t->dtype ? t->dtype : "(null)");
        return -1;
    }
    if (t->ndim != 2 || t->shape[0] != dim0 || t->shape[1] != dim1) {
        fprintf(stderr,
                "Error: tensor %s shape mismatch: expected [%lld, %lld], got [%lld, %lld]\n",
                name,
                (long long)dim0, (long long)dim1,
                (long long)(t->ndim > 0 ? t->shape[0] : -1),
                (long long)(t->ndim > 1 ? t->shape[1] : -1));
        return -1;
    }
    return 0;
}

static int validate_talker_attention_shapes(const multi_safetensors_t *ms,
                                            const qwen_tts_config_t *cfg,
                                            int layer_idx) {
    char name[512];
    int64_t q_out = (int64_t)cfg->talker_heads * cfg->talker_head_dim;
    int64_t kv_out = (int64_t)cfg->talker_kv_heads * cfg->talker_head_dim;
    int64_t hidden = cfg->talker_hidden;

    snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.q_proj.weight", layer_idx);
    if (expect_tensor_bf16_2d(ms, name, q_out, hidden) != 0) return -1;

    snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.k_proj.weight", layer_idx);
    if (expect_tensor_bf16_2d(ms, name, kv_out, hidden) != 0) return -1;

    snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.v_proj.weight", layer_idx);
    if (expect_tensor_bf16_2d(ms, name, kv_out, hidden) != 0) return -1;

    snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.o_proj.weight", layer_idx);
    if (expect_tensor_bf16_2d(ms, name, hidden, q_out) != 0) return -1;

    return 0;
}

/* ========================================================================
 * Load Talker Weights
 * ======================================================================== */

static int load_talker_weights(qwen_tts_ctx_t *ctx, const multi_safetensors_t *ms) {
    qwen_tts_config_t *cfg = &ctx->config;
    char name[512];

    if (qwen_tts_verbose >= 1) fprintf(stderr, "Loading talker weights...\n");

    /* Embeddings */
    GET_BF16_CHECK(ctx->talker.codec_embedding_bf16, ms, "talker.model.codec_embedding.weight");
    GET_BF16_CHECK(ctx->talker.text_embedding_bf16, ms, "talker.model.text_embedding.weight");

    /* Text projection MLP */
    GET_BF16_CHECK(ctx->talker.text_proj_fc1_bf16, ms, "talker.text_projection.linear_fc1.weight");
    LOAD_F32_CHECK(ctx->talker.text_proj_fc1_bias, ms, "talker.text_projection.linear_fc1.bias");
    GET_BF16_CHECK(ctx->talker.text_proj_fc2_bf16, ms, "talker.text_projection.linear_fc2.weight");
    LOAD_F32_CHECK(ctx->talker.text_proj_fc2_bias, ms, "talker.text_projection.linear_fc2.bias");

    /* Transformer layers */
    for (int i = 0; i < cfg->talker_layers; i++) {
        qwen_tts_talker_layer_t *l = &ctx->talker.layers[i];

        if (validate_talker_attention_shapes(ms, cfg, i) != 0) return -1;

        snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.q_proj.weight", i);
        GET_BF16_CHECK(l->wq_bf16, ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.k_proj.weight", i);
        GET_BF16_CHECK(l->wk_bf16, ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.v_proj.weight", i);
        GET_BF16_CHECK(l->wv_bf16, ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.o_proj.weight", i);
        GET_BF16_CHECK(l->wo_bf16, ms, name);

        snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.q_norm.weight", i);
        LOAD_F32_CHECK(l->q_norm_weight, ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.k_norm.weight", i);
        LOAD_F32_CHECK(l->k_norm_weight, ms, name);

        snprintf(name, sizeof(name), "talker.model.layers.%d.input_layernorm.weight", i);
        LOAD_F32_CHECK(l->input_norm, ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.post_attention_layernorm.weight", i);
        LOAD_F32_CHECK(l->post_attn_norm, ms, name);

        snprintf(name, sizeof(name), "talker.model.layers.%d.mlp.gate_proj.weight", i);
        GET_BF16_CHECK(l->gate_bf16, ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.mlp.up_proj.weight", i);
        GET_BF16_CHECK(l->up_bf16, ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.mlp.down_proj.weight", i);
        GET_BF16_CHECK(l->down_bf16, ms, name);

        /* Create fused gate+up weights for faster single-token SwiGLU MLP */
        {
            size_t gu_size = (size_t)cfg->talker_intermediate * cfg->talker_hidden;
            l->gate_up_fused_bf16 = (uint16_t *)malloc(2 * gu_size * sizeof(uint16_t));
            if (l->gate_up_fused_bf16) {
                memcpy(l->gate_up_fused_bf16, l->gate_bf16, gu_size * sizeof(uint16_t));
                memcpy(l->gate_up_fused_bf16 + gu_size, l->up_bf16, gu_size * sizeof(uint16_t));
            }
        }

        /* Create fused Q+K+V weights */
        {
            int q_rows = cfg->talker_heads * cfg->talker_head_dim;
            int kv_rows = cfg->talker_kv_heads * cfg->talker_head_dim;
            int total_rows = q_rows + kv_rows + kv_rows;
            size_t row_elems = (size_t)cfg->talker_hidden;
            l->wqkv_fused_bf16 = (uint16_t *)malloc((size_t)total_rows * row_elems * sizeof(uint16_t));
            if (l->wqkv_fused_bf16) {
                memcpy(l->wqkv_fused_bf16,
                       l->wq_bf16, (size_t)q_rows * row_elems * sizeof(uint16_t));
                memcpy(l->wqkv_fused_bf16 + (size_t)q_rows * row_elems,
                       l->wk_bf16, (size_t)kv_rows * row_elems * sizeof(uint16_t));
                memcpy(l->wqkv_fused_bf16 + (size_t)(q_rows + kv_rows) * row_elems,
                       l->wv_bf16, (size_t)kv_rows * row_elems * sizeof(uint16_t));
            }
        }

        /* Q4_K quantize QKV (skip if loaded from cache) */
        if (l->wqkv_fused_bf16 && !l->wqkv_q4k) {
            int q_rows = cfg->talker_heads * cfg->talker_head_dim;
            int kv_rows = cfg->talker_kv_heads * cfg->talker_head_dim;
            int total_rows = q_rows + kv_rows + kv_rows;
            quantize_bf16_to_q4k(l->wqkv_fused_bf16, total_rows, cfg->talker_hidden,
                                 &l->wqkv_q4k);
        }

        /* Q4_K quantize gate_up (skip if loaded from cache) */
        if (l->gate_up_fused_bf16 && !l->gate_up_q4k) {
            int gu_rows = 2 * cfg->talker_intermediate;
            quantize_bf16_to_q4k(l->gate_up_fused_bf16, gu_rows, cfg->talker_hidden,
                                 &l->gate_up_q4k);
        }

        /* Q4_K quantize wo (skip if loaded from cache) */
        if (l->wo_bf16 && !l->wo_q4k) {
            int q_dim = cfg->talker_heads * cfg->talker_head_dim;
            quantize_bf16_to_q4k(l->wo_bf16, cfg->talker_hidden, q_dim,
                                 &l->wo_q4k);
        }

        /* Q4_K quantize down (skip if loaded from cache) */
        if (l->down_bf16 && !l->down_q4k) {
            quantize_bf16_to_q4k(l->down_bf16, cfg->talker_hidden, cfg->talker_intermediate,
                                 &l->down_q4k);
        }
    }

    /* Final norm */
    LOAD_F32_CHECK(ctx->talker.norm, ms, "talker.model.norm.weight");

    /* Codec head */
    GET_BF16_CHECK(ctx->talker.codec_head_bf16, ms, "talker.codec_head.weight");

    if (qwen_tts_verbose >= 1) fprintf(stderr, "  Talker: %d layers loaded\n", cfg->talker_layers);
    return 0;
}

/* ========================================================================
 * Load Sub-Talker (Code Predictor) Weights
 * ======================================================================== */

static void load_subtalker_weights(qwen_tts_ctx_t *ctx, const multi_safetensors_t *ms) {
    qwen_tts_config_t *cfg = &ctx->config;
    char name[512];

    if (qwen_tts_verbose >= 1) fprintf(stderr, "Loading sub-talker weights...\n");

    /* 31 codec embeddings (groups 1-31) */
    for (int g = 0; g < cfg->num_code_groups - 1; g++) {
        snprintf(name, sizeof(name), "talker.code_predictor.model.codec_embedding.%d.weight", g);
        GET_BF16_CHECK(ctx->subtalker.codec_embeddings_bf16[g], ms, name);
    }

    /* Input projection */
    GET_BF16_CHECK(ctx->subtalker.input_proj_bf16, ms, "talker.code_predictor.small_to_mtp_projection.weight");
    LOAD_F32_CHECK(ctx->subtalker.input_proj_bias, ms, "talker.code_predictor.small_to_mtp_projection.bias");

    /* Transformer layers */
    for (int i = 0; i < cfg->subtalker_layers; i++) {
        qwen_tts_subtalker_layer_t *l = &ctx->subtalker.layers[i];

        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.q_proj.weight", i);
        GET_BF16_CHECK(l->wq_bf16, ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.k_proj.weight", i);
        GET_BF16_CHECK(l->wk_bf16, ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.v_proj.weight", i);
        GET_BF16_CHECK(l->wv_bf16, ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.o_proj.weight", i);
        GET_BF16_CHECK(l->wo_bf16, ms, name);

        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.q_norm.weight", i);
        LOAD_F32_CHECK(l->q_norm_weight, ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.k_norm.weight", i);
        LOAD_F32_CHECK(l->k_norm_weight, ms, name);

        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.input_layernorm.weight", i);
        LOAD_F32_CHECK(l->input_norm, ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.post_attention_layernorm.weight", i);
        LOAD_F32_CHECK(l->post_attn_norm, ms, name);

        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.mlp.gate_proj.weight", i);
        GET_BF16_CHECK(l->gate_bf16, ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.mlp.up_proj.weight", i);
        GET_BF16_CHECK(l->up_bf16, ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.mlp.down_proj.weight", i);
        GET_BF16_CHECK(l->down_bf16, ms, name);

        /* Optional fused gate+up weights for faster single-token subtalker MLP. */
        size_t gu_size = (size_t)cfg->subtalker_intermediate * cfg->subtalker_hidden;
        l->gate_up_fused_bf16 = (uint16_t *)malloc(2 * gu_size * sizeof(uint16_t));
        if (l->gate_up_fused_bf16) {
            memcpy(l->gate_up_fused_bf16, l->gate_bf16, gu_size * sizeof(uint16_t));
            memcpy(l->gate_up_fused_bf16 + gu_size, l->up_bf16, gu_size * sizeof(uint16_t));
        }

        /* Create fused Q+K+V weights */
        {
            int q_rows = cfg->subtalker_heads * cfg->subtalker_head_dim;
            int kv_rows = cfg->subtalker_kv_heads * cfg->subtalker_head_dim;
            int total_rows = q_rows + kv_rows + kv_rows;
            size_t row_elems = (size_t)cfg->subtalker_hidden;
            l->wqkv_fused_bf16 = (uint16_t *)malloc((size_t)total_rows * row_elems * sizeof(uint16_t));
            if (l->wqkv_fused_bf16) {
                memcpy(l->wqkv_fused_bf16,
                       l->wq_bf16, (size_t)q_rows * row_elems * sizeof(uint16_t));
                memcpy(l->wqkv_fused_bf16 + (size_t)q_rows * row_elems,
                       l->wk_bf16, (size_t)kv_rows * row_elems * sizeof(uint16_t));
                memcpy(l->wqkv_fused_bf16 + (size_t)(q_rows + kv_rows) * row_elems,
                       l->wv_bf16, (size_t)kv_rows * row_elems * sizeof(uint16_t));
            }
        }

        /* Q4_K quantize all weights (skip if loaded from cache) */
        if (l->wqkv_fused_bf16 && !l->wqkv_q4k) {
            int q_rows = cfg->subtalker_heads * cfg->subtalker_head_dim;
            int kv_rows = cfg->subtalker_kv_heads * cfg->subtalker_head_dim;
            int total_rows = q_rows + kv_rows + kv_rows;
            quantize_bf16_to_q4k(l->wqkv_fused_bf16, total_rows, cfg->subtalker_hidden,
                                 &l->wqkv_q4k);
        }
        if (l->gate_up_fused_bf16 && !l->gate_up_q4k) {
            int gu_rows = 2 * cfg->subtalker_intermediate;
            quantize_bf16_to_q4k(l->gate_up_fused_bf16, gu_rows, cfg->subtalker_hidden,
                                 &l->gate_up_q4k);
        }
        if (l->wo_bf16 && !l->wo_q4k) {
            int q_dim = cfg->subtalker_heads * cfg->subtalker_head_dim;
            quantize_bf16_to_q4k(l->wo_bf16, cfg->subtalker_hidden, q_dim,
                                 &l->wo_q4k);
        }
        if (l->down_bf16 && !l->down_q4k) {
            quantize_bf16_to_q4k(l->down_bf16, cfg->subtalker_hidden, cfg->subtalker_intermediate,
                                 &l->down_q4k);
        }
    }

    /* Final norm */
    LOAD_F32_CHECK(ctx->subtalker.norm, ms, "talker.code_predictor.model.norm.weight");

    /* 31 LM heads */
    for (int g = 0; g < cfg->num_code_groups - 1; g++) {
        snprintf(name, sizeof(name), "talker.code_predictor.lm_head.%d.weight", g);
        GET_BF16_CHECK(ctx->subtalker.lm_heads_bf16[g], ms, name);
    }

    if (qwen_tts_verbose >= 1) fprintf(stderr, "  Sub-talker: %d layers loaded\n", cfg->subtalker_layers);
}

/* ========================================================================
 * Load Codec Decoder (Speech Tokenizer) Weights
 * ======================================================================== */

static void build_codec_codebook_embeddings(qwen_tts_codebook_t *cb, int codebook_size, int codebook_dim) {
    if (!cb || !cb->cluster_usage || !cb->embedding_sum) return;

    size_t total = (size_t)codebook_size * codebook_dim;
    cb->embeddings = (float *)malloc(total * sizeof(float));
    if (!cb->embeddings) return;

    for (int c = 0; c < codebook_size; c++) {
        float usage = cb->cluster_usage[c];
        if (usage < 1e-5f) usage = 1e-5f;
        float inv_usage = 1.0f / usage;
        float *dst = cb->embeddings + (size_t)c * codebook_dim;
        const float *src = cb->embedding_sum + (size_t)c * codebook_dim;
        for (int d = 0; d < codebook_dim; d++) {
            dst[d] = src[d] * inv_usage;
        }
    }
}

static void preprocess_snakebeta_params(float *alpha, float *beta, int n) {
    if (!alpha || !beta) return;
    for (int i = 0; i < n; i++) {
        alpha[i] = expf(alpha[i]);
        beta[i] = 1.0f / (expf(beta[i]) + 1e-9f);
    }
}

static void load_codec_weights(qwen_tts_ctx_t *ctx, const multi_safetensors_t *ms) {
    qwen_tts_config_t *cfg = &ctx->config;
    qwen_tts_codec_decoder_t *codec = &ctx->codec;
    char name[512];

    if (qwen_tts_verbose >= 1) fprintf(stderr, "Loading codec decoder weights...\n");

    /* ---- RVQ: SplitResidualVectorQuantizer ---- */

    /* Semantic codebook (quantizer 0): rvq_first has 1 VQ layer */
    LOAD_F32_CHECK(codec->rvq.semantic_codebooks[0].cluster_usage, ms,
                   "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage");
    LOAD_F32_CHECK(codec->rvq.semantic_codebooks[0].embedding_sum, ms,
                   "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum");
    build_codec_codebook_embeddings(&codec->rvq.semantic_codebooks[0],
                                    cfg->codec_codebook_size, cfg->codec_codebook_dim / 2);

    /* Semantic output_proj: Conv1d(vq_dim, codebook_dim, 1) */
    LOAD_F32_CHECK(codec->rvq.semantic_output_proj, ms,
                   "decoder.quantizer.rvq_first.output_proj.weight");

    /* Acoustic codebooks (quantizers 1-15): rvq_rest has 15 VQ layers */
    for (int q = 0; q < cfg->codec_num_quantizers - 1; q++) {
        snprintf(name, sizeof(name), "decoder.quantizer.rvq_rest.vq.layers.%d._codebook.cluster_usage", q);
        LOAD_F32_CHECK(codec->rvq.acoustic_codebooks[q].cluster_usage, ms, name);
        snprintf(name, sizeof(name), "decoder.quantizer.rvq_rest.vq.layers.%d._codebook.embedding_sum", q);
        LOAD_F32_CHECK(codec->rvq.acoustic_codebooks[q].embedding_sum, ms, name);
        build_codec_codebook_embeddings(&codec->rvq.acoustic_codebooks[q],
                                        cfg->codec_codebook_size, cfg->codec_codebook_dim / 2);
    }

    LOAD_F32_CHECK(codec->rvq.acoustic_output_proj, ms,
                   "decoder.quantizer.rvq_rest.output_proj.weight");

    /* ---- Pre-conv ---- */
    LOAD_F32_CHECK(codec->pre_conv_weight, ms, "decoder.pre_conv.conv.weight");
    LOAD_F32_CHECK(codec->pre_conv_bias, ms, "decoder.pre_conv.conv.bias");

    /* ---- Transformer ---- */
    LOAD_F32_CHECK(codec->transformer_input_proj_weight, ms, "decoder.pre_transformer.input_proj.weight");
    LOAD_F32_CHECK(codec->transformer_input_proj_bias, ms, "decoder.pre_transformer.input_proj.bias");
    LOAD_F32_CHECK(codec->transformer_output_proj_weight, ms, "decoder.pre_transformer.output_proj.weight");
    LOAD_F32_CHECK(codec->transformer_output_proj_bias, ms, "decoder.pre_transformer.output_proj.bias");
    LOAD_F32_CHECK(codec->transformer_norm, ms, "decoder.pre_transformer.norm.weight");

    for (int i = 0; i < cfg->codec_layers; i++) {
        qwen_tts_codec_transformer_layer_t *l = &codec->transformer_layers[i];

        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.input_layernorm.weight", i);
        LOAD_F32_CHECK(l->input_norm, ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.post_attention_layernorm.weight", i);
        LOAD_F32_CHECK(l->post_attn_norm, ms, name);

        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn_layer_scale.scale", i);
        LOAD_F32_CHECK(l->attn_layer_scale, ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp_layer_scale.scale", i);
        LOAD_F32_CHECK(l->mlp_layer_scale, ms, name);

        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.q_proj.weight", i);
        LOAD_F32_CHECK(l->wq, ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.k_proj.weight", i);
        LOAD_F32_CHECK(l->wk, ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.v_proj.weight", i);
        LOAD_F32_CHECK(l->wv, ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.o_proj.weight", i);
        LOAD_F32_CHECK(l->wo, ms, name);

        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp.gate_proj.weight", i);
        LOAD_F32_CHECK(l->gate, ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp.up_proj.weight", i);
        LOAD_F32_CHECK(l->up, ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp.down_proj.weight", i);
        LOAD_F32_CHECK(l->down, ms, name);
    }

    /* ---- Upsample stages ---- */
    for (int s = 0; s < 2; s++) {
        snprintf(name, sizeof(name), "decoder.upsample.%d.0.conv.weight", s);
        LOAD_F32_CHECK(codec->upsample_transconv_weight[s], ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.0.conv.bias", s);
        LOAD_F32_CHECK(codec->upsample_transconv_bias[s], ms, name);

        qwen_tts_convnext_block_t *cn = &codec->upsample_convnext[s];
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.dwconv.conv.weight", s);
        LOAD_F32_CHECK(cn->dwconv_weight, ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.dwconv.conv.bias", s);
        LOAD_F32_CHECK(cn->dwconv_bias, ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.norm.weight", s);
        LOAD_F32_CHECK(cn->norm_weight, ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.norm.bias", s);
        LOAD_F32_CHECK(cn->norm_bias, ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.pwconv1.weight", s);
        LOAD_F32_CHECK(cn->pwconv1_weight, ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.pwconv1.bias", s);
        LOAD_F32_CHECK(cn->pwconv1_bias, ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.pwconv2.weight", s);
        LOAD_F32_CHECK(cn->pwconv2_weight, ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.pwconv2.bias", s);
        LOAD_F32_CHECK(cn->pwconv2_bias, ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.gamma", s);
        LOAD_F32_CHECK(cn->gamma, ms, name);
    }

    /* ---- Vocoder ---- */
    /* decoder.decoder.[0..6] in Python's ModuleList:
     *   [0] = initial CausalConv (latent -> decoder_dim, k=7)
     *   [1..4] = DecoderBlock (each has .block = [SnakeBeta, TransConv, ResUnit, ResUnit, ResUnit])
     *   [5] = final SnakeBeta
     *   [6] = final CausalConv (out_dim -> 1, k=7)
     */
    LOAD_F32_CHECK(codec->vocoder_pre_conv_weight, ms, "decoder.decoder.0.conv.weight");
    LOAD_F32_CHECK(codec->vocoder_pre_conv_bias, ms, "decoder.decoder.0.conv.bias");

    for (int b = 0; b < 4; b++) {
        qwen_tts_vocoder_block_t *vb = &codec->vocoder_blocks[b];
        int idx = b + 1;  /* Python module index: decoder.decoder.{b+1} */

        /* SnakeBeta activation at block[0] */
        snprintf(name, sizeof(name), "decoder.decoder.%d.block.0.alpha", idx);
        LOAD_F32_CHECK(vb->act_alpha, ms, name);
        snprintf(name, sizeof(name), "decoder.decoder.%d.block.0.beta", idx);
        LOAD_F32_CHECK(vb->act_beta, ms, name);
        preprocess_snakebeta_params(vb->act_alpha, vb->act_beta, cfg->codec_decoder_dim >> b);

        /* Transposed conv at block[1] */
        snprintf(name, sizeof(name), "decoder.decoder.%d.block.1.conv.weight", idx);
        LOAD_F32_CHECK(vb->transconv_weight, ms, name);
        snprintf(name, sizeof(name), "decoder.decoder.%d.block.1.conv.bias", idx);
        LOAD_F32_CHECK(vb->transconv_bias, ms, name);

        /* 3 residual units at block[2], block[3], block[4] */
        for (int r = 0; r < 3; r++) {
            qwen_tts_vocoder_resunit_t *ru = &vb->resunits[r];
            int ridx = r + 2;

            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.act1.alpha", idx, ridx);
            LOAD_F32_CHECK(ru->act1_alpha, ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.act1.beta", idx, ridx);
            LOAD_F32_CHECK(ru->act1_beta, ms, name);
            preprocess_snakebeta_params(ru->act1_alpha, ru->act1_beta, cfg->codec_decoder_dim >> (b + 1));
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.conv1.conv.weight", idx, ridx);
            LOAD_F32_CHECK(ru->conv1_weight, ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.conv1.conv.bias", idx, ridx);
            LOAD_F32_CHECK(ru->conv1_bias, ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.act2.alpha", idx, ridx);
            LOAD_F32_CHECK(ru->act2_alpha, ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.act2.beta", idx, ridx);
            LOAD_F32_CHECK(ru->act2_beta, ms, name);
            preprocess_snakebeta_params(ru->act2_alpha, ru->act2_beta, cfg->codec_decoder_dim >> (b + 1));
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.conv2.conv.weight", idx, ridx);
            LOAD_F32_CHECK(ru->conv2_weight, ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.conv2.conv.bias", idx, ridx);
            LOAD_F32_CHECK(ru->conv2_bias, ms, name);
        }
    }

    /* Final SnakeBeta + Conv (decoder.decoder.5 and decoder.decoder.6) */
    LOAD_F32_CHECK(codec->vocoder_final_act_alpha, ms, "decoder.decoder.5.alpha");
    LOAD_F32_CHECK(codec->vocoder_final_act_beta, ms, "decoder.decoder.5.beta");
    preprocess_snakebeta_params(codec->vocoder_final_act_alpha, codec->vocoder_final_act_beta,
                                cfg->codec_decoder_dim / 16);
    LOAD_F32_CHECK(codec->vocoder_final_conv_weight, ms, "decoder.decoder.6.conv.weight");
    LOAD_F32_CHECK(codec->vocoder_final_conv_bias, ms, "decoder.decoder.6.conv.bias");

    if (qwen_tts_verbose >= 1) fprintf(stderr, "  Codec decoder loaded\n");
}

int qwen_tts_ensure_codec_loaded(qwen_tts_ctx_t *ctx) {
    if (!ctx) return -1;
    if (ctx->codec_safetensors) return 0;

#ifdef __EMSCRIPTEN__
    /* In browser/WASM, keep peak memory lower by dropping talker mapping first. */
    if (ctx->safetensors) {
        if (qwen_tts_verbose >= 1) {
            fprintf(stderr, "WASM: releasing talker safetensors before codec load\n");
        }
        multi_safetensors_close((multi_safetensors_t *)ctx->safetensors);
        ctx->safetensors = NULL;
    }

    /* Free large root talker safetensors files from MEMFS before codec load. */
    DIR *d = opendir(ctx->model_dir);
    if (d) {
        struct dirent *ent;
        while ((ent = readdir(d)) != NULL) {
            const char *name = ent->d_name;
            const char *ext = strrchr(name, '.');
            int remove_file = 0;
            if (ext && strcmp(ext, ".safetensors") == 0) remove_file = 1;
            if (strstr(name, ".safetensors.index.json")) remove_file = 1;
            if (remove_file) {
                char path[1024];
                snprintf(path, sizeof(path), "%s/%s", ctx->model_dir, name);
                unlink(path);
            }
        }
        closedir(d);
    }
#endif

    char codec_dir[1024];
    snprintf(codec_dir, sizeof(codec_dir), "%s/speech_tokenizer", ctx->model_dir);
    multi_safetensors_t *cms = multi_safetensors_open(codec_dir);
    if (!cms) {
        fprintf(stderr, "Error: cannot open speech_tokenizer safetensors in %s\n", codec_dir);
        return -1;
    }
    ctx->codec_safetensors = cms;
    load_codec_weights(ctx, cms);
    return 0;
}

/* ========================================================================
 * Load Model
 * ======================================================================== */

qwen_tts_ctx_t *qwen_tts_load(const char *model_dir) {
    double t0 = time_ms();

    qwen_tts_ctx_t *ctx = (qwen_tts_ctx_t *)calloc(1, sizeof(qwen_tts_ctx_t));
    if (!ctx) return NULL;

    strncpy(ctx->model_dir, model_dir, sizeof(ctx->model_dir) - 1);

    /* Set cache directory (override from JNI or model_dir) */
    if (qwen_tts_cache_dir_override && qwen_tts_cache_dir_override[0]) {
        strncpy(ctx->cache_dir, qwen_tts_cache_dir_override, sizeof(ctx->cache_dir) - 1);
    } else {
        strncpy(ctx->cache_dir, model_dir, sizeof(ctx->cache_dir) - 1);
    }

    /* Set default generation parameters */
    ctx->temperature = 0.9f;
    ctx->subtalker_temperature = 0.9f;
    ctx->top_k = 50;
    ctx->subtalker_top_k = 50;
    ctx->top_p = 1.0f;
    ctx->subtalker_top_p = 1.0f;
    ctx->repetition_penalty = 1.05f;
    ctx->max_new_tokens = 4096;
    ctx->fixed_codec_tokens = 0;
    ctx->sample_seed = 42;

    /* Load config */
    if (load_config(ctx) != 0) {
        free(ctx);
        return NULL;
    }

    /* Open talker safetensors */
    multi_safetensors_t *ms = multi_safetensors_open(model_dir);
    if (!ms) {
        fprintf(stderr, "Error: cannot open model safetensors in %s\n", model_dir);
        free(ctx);
        return NULL;
    }
    ctx->safetensors = ms;

    /* Try loading pre-quantized weights from cache */
    int cache_loaded = load_quantized_cache(ctx);

    if (load_talker_weights(ctx, ms) != 0) {
        qwen_tts_free(ctx);
        return NULL;
    }
    load_subtalker_weights(ctx, ms);

    /* Save quantized cache if we had to quantize (cache miss) */
    if (cache_loaded != 0) {
        save_quantized_cache(ctx);
    }

    /* Open codec decoder safetensors */
#ifndef __EMSCRIPTEN__
    if (qwen_tts_ensure_codec_loaded(ctx) != 0) {
        /* Continue without codec (can still generate tokens). */
    }
#else
    if (qwen_tts_verbose >= 1) {
        fprintf(stderr, "WASM: deferring codec decoder load until decode stage\n");
    }
#endif

    kernel_init();

    double t1 = time_ms();
    if (qwen_tts_verbose >= 1)
        fprintf(stderr, "Model loaded in %.1f ms\n", t1 - t0);

    return ctx;
}

/* ========================================================================
 * Free
 * ======================================================================== */

void qwen_tts_free(qwen_tts_ctx_t *ctx) {
    if (!ctx) return;

    /* Close safetensors (this frees mmap'd weights) */
    if (ctx->safetensors) multi_safetensors_close((multi_safetensors_t *)ctx->safetensors);
    if (ctx->codec_safetensors) multi_safetensors_close((multi_safetensors_t *)ctx->codec_safetensors);

    /* Free codec weights that were LOAD_F32'd (malloc'd copies) */
    /* RVQ */
    for (int i = 0; i < 1; i++) {
        free(ctx->codec.rvq.semantic_codebooks[i].cluster_usage);
        free(ctx->codec.rvq.semantic_codebooks[i].embedding_sum);
        free(ctx->codec.rvq.semantic_codebooks[i].embeddings);
    }
    for (int i = 0; i < ctx->config.codec_num_quantizers - 1; i++) {
        free(ctx->codec.rvq.acoustic_codebooks[i].cluster_usage);
        free(ctx->codec.rvq.acoustic_codebooks[i].embedding_sum);
        free(ctx->codec.rvq.acoustic_codebooks[i].embeddings);
    }
    free(ctx->codec.rvq.semantic_output_proj);
    free(ctx->codec.rvq.acoustic_output_proj);
    free(ctx->codec.pre_conv_weight);
    free(ctx->codec.pre_conv_bias);
    free(ctx->codec.transformer_input_proj_weight);
    free(ctx->codec.transformer_input_proj_bias);
    free(ctx->codec.transformer_output_proj_weight);
    free(ctx->codec.transformer_output_proj_bias);
    free(ctx->codec.transformer_norm);

    for (int i = 0; i < ctx->config.codec_layers; i++) {
        qwen_tts_codec_transformer_layer_t *l = &ctx->codec.transformer_layers[i];
        free(l->input_norm); free(l->post_attn_norm);
        free(l->attn_layer_scale); free(l->mlp_layer_scale);
        free(l->wq); free(l->wk); free(l->wv); free(l->wo);
        free(l->gate); free(l->up); free(l->down);
    }

    for (int s = 0; s < 2; s++) {
        free(ctx->codec.upsample_transconv_weight[s]);
        free(ctx->codec.upsample_transconv_bias[s]);
        qwen_tts_convnext_block_t *cn = &ctx->codec.upsample_convnext[s];
        free(cn->dwconv_weight); free(cn->dwconv_bias);
        free(cn->norm_weight); free(cn->norm_bias);
        free(cn->pwconv1_weight); free(cn->pwconv1_bias);
        free(cn->pwconv2_weight); free(cn->pwconv2_bias);
        free(cn->gamma);
    }

    free(ctx->codec.vocoder_pre_conv_weight); free(ctx->codec.vocoder_pre_conv_bias);
    for (int b = 0; b < 4; b++) {
        qwen_tts_vocoder_block_t *vb = &ctx->codec.vocoder_blocks[b];
        free(vb->act_alpha); free(vb->act_beta);
        free(vb->transconv_weight); free(vb->transconv_bias);
        for (int r = 0; r < 3; r++) {
            qwen_tts_vocoder_resunit_t *ru = &vb->resunits[r];
            free(ru->act1_alpha); free(ru->act1_beta);
            free(ru->conv1_weight); free(ru->conv1_bias);
            free(ru->act2_alpha); free(ru->act2_beta);
            free(ru->conv2_weight); free(ru->conv2_bias);
        }
    }
    free(ctx->codec.vocoder_final_act_alpha); free(ctx->codec.vocoder_final_act_beta);
    free(ctx->codec.vocoder_final_conv_weight); free(ctx->codec.vocoder_final_conv_bias);

    /* Free talker LOAD_F32'd weights (norms + biases) + quantized */
    free(ctx->talker.text_proj_fc1_bias);
    free(ctx->talker.text_proj_fc2_bias);
    free(ctx->talker.norm);
    for (int i = 0; i < ctx->config.talker_layers; i++) {
        qwen_tts_talker_layer_t *l = &ctx->talker.layers[i];
        free(l->q_norm_weight); free(l->k_norm_weight);
        free(l->input_norm); free(l->post_attn_norm);
        free(l->gate_up_fused_bf16);
        free(l->wqkv_fused_bf16);
        free(l->wqkv_q4k);
        free(l->gate_up_q4k);
        free(l->wo_q4k);
        free(l->down_q4k);
    }

    /* Free subtalker LOAD_F32'd weights (norms + biases) + quantized */
    free(ctx->subtalker.input_proj_bias);
    free(ctx->subtalker.norm);
    for (int i = 0; i < ctx->config.subtalker_layers; i++) {
        qwen_tts_subtalker_layer_t *l = &ctx->subtalker.layers[i];
        free(l->q_norm_weight); free(l->k_norm_weight);
        free(l->input_norm); free(l->post_attn_norm);
        free(l->gate_up_fused_bf16);
        free(l->wqkv_fused_bf16);
        free(l->wqkv_q4k);
        free(l->gate_up_q4k);
        free(l->wo_q4k);
        free(l->down_q4k);
    }

    /* Free KV caches and scratch buffers */
    free(ctx->talker_kv_k); free(ctx->talker_kv_v);
    free(ctx->subtalker_kv_k); free(ctx->subtalker_kv_v);
    free(ctx->codec_kv_k); free(ctx->codec_kv_v);
    free(ctx->tk_qkv);
    free(ctx->tk_x); free(ctx->tk_x_norm);
    free(ctx->tk_q); free(ctx->tk_k); free(ctx->tk_v);
    free(ctx->tk_attn_out); free(ctx->tk_proj_out);
    free(ctx->tk_gate); free(ctx->tk_up); free(ctx->tk_ffn_out);
    free(ctx->tk_scores);
    free(ctx->tk_rope_cos); free(ctx->tk_rope_sin);
    free(ctx->tk_pref_x); free(ctx->tk_pref_x_norm);
    free(ctx->tk_pref_q); free(ctx->tk_pref_k); free(ctx->tk_pref_v);
    free(ctx->tk_pref_attn_out); free(ctx->tk_pref_proj_out);
    free(ctx->tk_pref_gate); free(ctx->tk_pref_gate_up); free(ctx->tk_pref_ffn_out);
    free(ctx->st_qkv);
    free(ctx->st_x); free(ctx->st_x_norm);
    free(ctx->st_q); free(ctx->st_k); free(ctx->st_v);
    free(ctx->st_attn_out); free(ctx->st_logits);
    free(ctx->st_gate); free(ctx->st_up);
    free(ctx->st_embed); free(ctx->st_proj_hidden);
    free(ctx->st_scores);
    free(ctx->st_rope_cos); free(ctx->st_rope_sin);
    free(ctx->talker_rope_cos_cache); free(ctx->talker_rope_sin_cache);

    /* Free config maps */
    for (int i = 0; i < ctx->config.n_speakers; i++) free(ctx->config.speaker_names[i]);
    free(ctx->config.speaker_names); free(ctx->config.speaker_ids);
    for (int i = 0; i < ctx->config.n_languages; i++) free(ctx->config.language_names[i]);
    free(ctx->config.language_names); free(ctx->config.language_ids);

    free(ctx);
}

void qwen_tts_set_progress_callback(qwen_tts_ctx_t *ctx, qwen_tts_progress_cb cb, void *userdata) {
    ctx->progress_cb = cb;
    ctx->progress_cb_userdata = userdata;
}
