/*
 * qwen_tts.c - Main API for Qwen3-TTS C inference engine
 *
 * Contains:
 *   - Minimal JSON helpers for config.json parsing
 *   - Config loading (talker + speech_tokenizer)
 *   - Weight loading from SafeTensors (mmap)
 *   - generate() function (embedding construction + autoregressive loop)
 *   - Free / cleanup
 */

#include "qwen_tts.h"
#include "qwen_tts_internal.h"
#include "qwen_tts_kernels.h"
#include "qwen_tts_safetensors.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef __EMSCRIPTEN__
#include <dirent.h>
#include <unistd.h>
#endif

int qwen_tts_verbose = 0;
const char *qwen_tts_cache_dir_override = NULL;  /* set before qwen_tts_load() */

/* ========================================================================
 * Timing helpers
 * ======================================================================== */

double qwen_tts_time_ms(void) {
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
 * Q8_0 Quantization Helpers
 *
 * Allocate and quantize a weight matrix [rows, cols] to Q8_0 blocks.
 * Returns malloc'd block array. Caller must free.
 * ======================================================================== */

static block_q8_0 *quantize_weight_bf16(const uint16_t *bf16, int rows, int cols) {
    size_t total_elems = (size_t)rows * cols;
    size_t n_blocks = total_elems / QK8_0;
    block_q8_0 *dst = (block_q8_0 *)malloc(n_blocks * sizeof(block_q8_0));
    if (!dst) return NULL;
    quantize_bf16_to_q8_0(bf16, dst, (int)total_elems);
    return dst;
}

static block_q8_0 *quantize_weight_f32(const float *f32, int rows, int cols) {
    size_t total_elems = (size_t)rows * cols;
    size_t n_blocks = total_elems / QK8_0;
    block_q8_0 *dst = (block_q8_0 *)malloc(n_blocks * sizeof(block_q8_0));
    if (!dst) return NULL;
    quantize_f32_to_q8_0(f32, dst, (int)total_elems);
    return dst;
}

/* ========================================================================
 * Pre-quantized Weight Cache (Q8_0)
 *
 * After first-time BF16/F32→Q8_0 quantization, serialize the quantized
 * weights to a binary cache file. Subsequent loads mmap the cache,
 * avoiding the expensive quantization step.
 *
 * Cache format:
 *   header (qcache_header_t)
 *   for each talker layer:
 *     wqkv_q8 | wo_q8 | gate_up_q8 | down_q8
 *   for each subtalker layer:
 *     wqkv_q8 | wo_q8 | gate_up_q8 | down_q8
 *   non-layer weights:
 *     text_proj_fc1_q8 | text_proj_fc2_q8 | codec_head_q8
 *     input_proj_q8 | lm_heads_q8[31]
 * ======================================================================== */

#ifndef __EMSCRIPTEN__
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#define QCACHE_MAGIC   0x33435151   /* "QQC3" */
#define QCACHE_VERSION 3

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t source_size;         /* original safetensors total file size for validation */
    uint32_t n_talker_layers;
    uint32_t n_subtalker_layers;
    /* Talker per-layer Q8_0 byte sizes */
    uint32_t tk_wqkv_q8_bytes;   /* per layer */
    uint32_t tk_wo_q8_bytes;
    uint32_t tk_gate_up_q8_bytes;
    uint32_t tk_down_q8_bytes;
    /* Subtalker per-layer Q8_0 byte sizes */
    uint32_t st_wqkv_q8_bytes;
    uint32_t st_wo_q8_bytes;
    uint32_t st_gate_up_q8_bytes;
    uint32_t st_down_q8_bytes;
    /* Non-layer weight sizes */
    uint32_t text_proj_fc1_q8_bytes;
    uint32_t text_proj_fc2_q8_bytes;
    uint32_t codec_head_q8_bytes;
    uint32_t input_proj_q8_bytes;
    uint32_t lm_head_q8_bytes;   /* per lm_head */
    uint32_t reserved[3];         /* future use */
} qcache_header_t;

#ifndef __EMSCRIPTEN__

static uint64_t get_safetensors_size(const char *model_dir) {
    /* Sum the sizes of all .safetensors files in the model_dir */
    char path[1024];
    uint64_t total = 0;
    struct stat st;

    /* Try common patterns: model.safetensors, model-00001-of-NNNNN.safetensors */
    snprintf(path, sizeof(path), "%s/model.safetensors", model_dir);
    if (stat(path, &st) == 0) {
        total += (uint64_t)st.st_size;
    }
    for (int i = 1; i <= 10; i++) {
        snprintf(path, sizeof(path), "%s/model-%05d-of-00002.safetensors", model_dir, i);
        if (stat(path, &st) == 0) {
            total += (uint64_t)st.st_size;
        }
        snprintf(path, sizeof(path), "%s/model-%05d-of-00003.safetensors", model_dir, i);
        if (stat(path, &st) == 0) {
            total += (uint64_t)st.st_size;
        }
    }
    return total;
}

static inline uint32_t q8_bytes(int rows, int cols) {
    return (uint32_t)(((size_t)rows * cols / QK8_0) * sizeof(block_q8_0));
}

static int save_quantized_cache(qwen_tts_ctx_t *ctx) {
    qwen_tts_config_t *cfg = &ctx->config;
    char path[1024];
    snprintf(path, sizeof(path), "%s/model.qcache", ctx->cache_dir);

    /* Compute per-layer Q8_0 sizes */
    int tk_qkv_rows = cfg->talker_heads * cfg->talker_head_dim +
                       2 * cfg->talker_kv_heads * cfg->talker_head_dim;
    int tk_q_dim = cfg->talker_heads * cfg->talker_head_dim;

    int st_qkv_rows = cfg->subtalker_heads * cfg->subtalker_head_dim +
                       2 * cfg->subtalker_kv_heads * cfg->subtalker_head_dim;
    int st_q_dim = cfg->subtalker_heads * cfg->subtalker_head_dim;

    qcache_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = QCACHE_MAGIC;
    hdr.version = QCACHE_VERSION;
    hdr.source_size = get_safetensors_size(ctx->model_dir);
    hdr.n_talker_layers = (uint32_t)cfg->talker_layers;
    hdr.n_subtalker_layers = (uint32_t)cfg->subtalker_layers;
    hdr.tk_wqkv_q8_bytes = q8_bytes(tk_qkv_rows, cfg->talker_hidden);
    hdr.tk_wo_q8_bytes = q8_bytes(cfg->talker_hidden, tk_q_dim);
    hdr.tk_gate_up_q8_bytes = q8_bytes(2 * cfg->talker_intermediate, cfg->talker_hidden);
    hdr.tk_down_q8_bytes = q8_bytes(cfg->talker_hidden, cfg->talker_intermediate);
    hdr.st_wqkv_q8_bytes = q8_bytes(st_qkv_rows, cfg->subtalker_hidden);
    hdr.st_wo_q8_bytes = q8_bytes(cfg->subtalker_hidden, st_q_dim);
    hdr.st_gate_up_q8_bytes = q8_bytes(2 * cfg->subtalker_intermediate, cfg->subtalker_hidden);
    hdr.st_down_q8_bytes = q8_bytes(cfg->subtalker_hidden, cfg->subtalker_intermediate);
    hdr.text_proj_fc1_q8_bytes = q8_bytes(cfg->talker_text_hidden, cfg->talker_text_hidden);
    hdr.text_proj_fc2_q8_bytes = q8_bytes(cfg->talker_hidden, cfg->talker_text_hidden);
    hdr.codec_head_q8_bytes = q8_bytes(cfg->talker_vocab_size, cfg->talker_hidden);
    /* Store 0 for absent optional weights so they stay NULL on load */
    hdr.input_proj_q8_bytes = ctx->subtalker.input_proj_q8
        ? q8_bytes(cfg->subtalker_hidden, cfg->talker_hidden) : 0;
    hdr.lm_head_q8_bytes = q8_bytes(cfg->subtalker_vocab_size, cfg->subtalker_hidden);

    FILE *f = fopen(path, "wb");
    if (!f) {
        if (qwen_tts_verbose >= 1)
            fprintf(stderr, "Warning: cannot create qcache at %s\n", path);
        return -1;
    }

    fwrite(&hdr, sizeof(hdr), 1, f);

    /* Only write data for non-NULL pointers; header byte sizes are 0 for absent weights */
    #define WRITE_Q8(ptr, n_bytes) do { \
        if ((ptr) && (n_bytes) > 0) fwrite(ptr, 1, n_bytes, f); \
    } while(0)

    for (int i = 0; i < cfg->talker_layers; i++) {
        qwen_tts_talker_layer_t *l = &ctx->talker.layers[i];
        WRITE_Q8(l->wqkv_q8, hdr.tk_wqkv_q8_bytes);
        WRITE_Q8(l->wo_q8, hdr.tk_wo_q8_bytes);
        WRITE_Q8(l->gate_up_q8, hdr.tk_gate_up_q8_bytes);
        WRITE_Q8(l->down_q8, hdr.tk_down_q8_bytes);
    }
    for (int i = 0; i < cfg->subtalker_layers; i++) {
        qwen_tts_subtalker_layer_t *l = &ctx->subtalker.layers[i];
        WRITE_Q8(l->wqkv_q8, hdr.st_wqkv_q8_bytes);
        WRITE_Q8(l->wo_q8, hdr.st_wo_q8_bytes);
        WRITE_Q8(l->gate_up_q8, hdr.st_gate_up_q8_bytes);
        WRITE_Q8(l->down_q8, hdr.st_down_q8_bytes);
    }
    /* Non-layer weights */
    WRITE_Q8(ctx->talker.text_proj_fc1_q8, hdr.text_proj_fc1_q8_bytes);
    WRITE_Q8(ctx->talker.text_proj_fc2_q8, hdr.text_proj_fc2_q8_bytes);
    WRITE_Q8(ctx->talker.codec_head_q8, hdr.codec_head_q8_bytes);
    WRITE_Q8(ctx->subtalker.input_proj_q8, hdr.input_proj_q8_bytes);
    for (int g = 0; g < cfg->num_code_groups - 1; g++) {
        WRITE_Q8(ctx->subtalker.lm_heads_q8[g], hdr.lm_head_q8_bytes);
    }

    #undef WRITE_Q8

    fclose(f);
    if (qwen_tts_verbose >= 1)
        fprintf(stderr, "Saved quantized cache to %s (input_proj: %s)\n",
                path, ctx->subtalker.input_proj_q8 ? "present" : "absent");
    return 0;
}

static int load_quantized_cache(qwen_tts_ctx_t *ctx) {
    qwen_tts_config_t *cfg = &ctx->config;
    char path[1024];
    snprintf(path, sizeof(path), "%s/model.qcache", ctx->cache_dir);

    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return -1; }
    size_t file_size = (size_t)st.st_size;
    if (file_size < sizeof(qcache_header_t)) { close(fd); return -1; }

    void *mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) return -1;

    const qcache_header_t *hdr = (const qcache_header_t *)mapped;

    if (hdr->magic != QCACHE_MAGIC || hdr->version != QCACHE_VERSION) {
        munmap(mapped, file_size);
        return -1;
    }
    if ((int)hdr->n_talker_layers != cfg->talker_layers ||
        (int)hdr->n_subtalker_layers != cfg->subtalker_layers) {
        munmap(mapped, file_size);
        return -1;
    }

    uint64_t expected_src = get_safetensors_size(ctx->model_dir);
    if (hdr->source_size != expected_src) {
        if (qwen_tts_verbose >= 1)
            fprintf(stderr, "qcache: source size mismatch (cache=%llu, actual=%llu), re-quantizing\n",
                    (unsigned long long)hdr->source_size, (unsigned long long)expected_src);
        munmap(mapped, file_size);
        return -1;
    }

    /* Validate total file size */
    size_t tk_per_layer = (size_t)hdr->tk_wqkv_q8_bytes + hdr->tk_wo_q8_bytes +
                          hdr->tk_gate_up_q8_bytes + hdr->tk_down_q8_bytes;
    size_t st_per_layer = (size_t)hdr->st_wqkv_q8_bytes + hdr->st_wo_q8_bytes +
                          hdr->st_gate_up_q8_bytes + hdr->st_down_q8_bytes;
    size_t non_layer = (size_t)hdr->text_proj_fc1_q8_bytes + hdr->text_proj_fc2_q8_bytes +
                       hdr->codec_head_q8_bytes + hdr->input_proj_q8_bytes +
                       (size_t)hdr->lm_head_q8_bytes * (cfg->num_code_groups - 1);
    size_t expected_size = sizeof(qcache_header_t) +
                           tk_per_layer * hdr->n_talker_layers +
                           st_per_layer * hdr->n_subtalker_layers +
                           non_layer;
    if (file_size < expected_size) {
        munmap(mapped, file_size);
        return -1;
    }

    const uint8_t *ptr = (const uint8_t *)mapped + sizeof(qcache_header_t);

    #define CACHE_COPY(dst, type, n_bytes) do { \
        if ((n_bytes) > 0) { \
            dst = (type)malloc(n_bytes); \
            if (dst) memcpy(dst, ptr, n_bytes); \
            ptr += (n_bytes); \
        } \
    } while(0)

    for (int i = 0; i < cfg->talker_layers; i++) {
        qwen_tts_talker_layer_t *l = &ctx->talker.layers[i];
        CACHE_COPY(l->wqkv_q8, block_q8_0 *, hdr->tk_wqkv_q8_bytes);
        CACHE_COPY(l->wo_q8, block_q8_0 *, hdr->tk_wo_q8_bytes);
        CACHE_COPY(l->gate_up_q8, block_q8_0 *, hdr->tk_gate_up_q8_bytes);
        CACHE_COPY(l->down_q8, block_q8_0 *, hdr->tk_down_q8_bytes);
    }
    for (int i = 0; i < cfg->subtalker_layers; i++) {
        qwen_tts_subtalker_layer_t *l = &ctx->subtalker.layers[i];
        CACHE_COPY(l->wqkv_q8, block_q8_0 *, hdr->st_wqkv_q8_bytes);
        CACHE_COPY(l->wo_q8, block_q8_0 *, hdr->st_wo_q8_bytes);
        CACHE_COPY(l->gate_up_q8, block_q8_0 *, hdr->st_gate_up_q8_bytes);
        CACHE_COPY(l->down_q8, block_q8_0 *, hdr->st_down_q8_bytes);
    }
    /* Non-layer weights */
    CACHE_COPY(ctx->talker.text_proj_fc1_q8, block_q8_0 *, hdr->text_proj_fc1_q8_bytes);
    CACHE_COPY(ctx->talker.text_proj_fc2_q8, block_q8_0 *, hdr->text_proj_fc2_q8_bytes);
    CACHE_COPY(ctx->talker.codec_head_q8, block_q8_0 *, hdr->codec_head_q8_bytes);
    CACHE_COPY(ctx->subtalker.input_proj_q8, block_q8_0 *, hdr->input_proj_q8_bytes);
    for (int g = 0; g < cfg->num_code_groups - 1; g++) {
        CACHE_COPY(ctx->subtalker.lm_heads_q8[g], block_q8_0 *, hdr->lm_head_q8_bytes);
    }

    #undef CACHE_COPY

    munmap(mapped, file_size);

    if (qwen_tts_verbose >= 1)
        fprintf(stderr, "Loaded quantized cache from %s (input_proj: %s)\n",
                path, ctx->subtalker.input_proj_q8 ? "present" : "absent");
    return 0;
}

#else /* __EMSCRIPTEN__ */

static int save_quantized_cache(qwen_tts_ctx_t *ctx) { (void)ctx; return -1; }
static int load_quantized_cache(qwen_tts_ctx_t *ctx) { (void)ctx; return -1; }

#endif /* __EMSCRIPTEN__ */

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

/* Load F32 from safetensors, convert to FP16, free F32 original */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static __fp16 *load_f16_from_safetensors(const multi_safetensors_t *ms, const char *name) {
    int64_t shape[8];
    int ndim = 0;
    float *f32 = multi_safetensors_load_f32(ms, name, shape, &ndim);
    if (!f32) return NULL;
    size_t n_elems = 1;
    for (int d = 0; d < ndim; d++) n_elems *= (size_t)shape[d];
    __fp16 *f16 = (__fp16 *)malloc(n_elems * sizeof(__fp16));
    if (f16) kernel_f32_to_f16(f16, f32, (int)n_elems);
    free(f32);
    return f16;
}

#define LOAD_F16_CHECK(dst, ms, name) do { \
    dst = load_f16_from_safetensors(ms, name); \
    if (!dst && qwen_tts_verbose >= 2) fprintf(stderr, "  Warning: tensor not found: %s\n", name); \
} while(0)
#endif

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

    /* Embeddings (BF16, row-lookup, kept as-is) */
    GET_BF16_CHECK(ctx->talker.codec_embedding_bf16, ms, "talker.model.codec_embedding.weight");
    GET_BF16_CHECK(ctx->talker.text_embedding_bf16, ms, "talker.model.text_embedding.weight");

    /* Text projection MLP: BF16 -> Q8_0 (skip if loaded from cache) */
    {
        uint16_t *fc1_bf16 = NULL, *fc2_bf16 = NULL;
        GET_BF16_CHECK(fc1_bf16, ms, "talker.text_projection.linear_fc1.weight");
        LOAD_F32_CHECK(ctx->talker.text_proj_fc1_bias, ms, "talker.text_projection.linear_fc1.bias");
        GET_BF16_CHECK(fc2_bf16, ms, "talker.text_projection.linear_fc2.weight");
        LOAD_F32_CHECK(ctx->talker.text_proj_fc2_bias, ms, "talker.text_projection.linear_fc2.bias");
        if (!ctx->talker.text_proj_fc1_q8 && fc1_bf16)
            ctx->talker.text_proj_fc1_q8 = quantize_weight_bf16(fc1_bf16, cfg->talker_text_hidden, cfg->talker_text_hidden);
        if (!ctx->talker.text_proj_fc2_q8 && fc2_bf16)
            ctx->talker.text_proj_fc2_q8 = quantize_weight_bf16(fc2_bf16, cfg->talker_hidden, cfg->talker_text_hidden);
    }

    /* Transformer layers */
    for (int i = 0; i < cfg->talker_layers; i++) {
        qwen_tts_talker_layer_t *l = &ctx->talker.layers[i];

        if (validate_talker_attention_shapes(ms, cfg, i) != 0) return -1;

        /* Load norms */
        snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.q_norm.weight", i);
        LOAD_F32_CHECK(l->q_norm_weight, ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.k_norm.weight", i);
        LOAD_F32_CHECK(l->k_norm_weight, ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.input_layernorm.weight", i);
        LOAD_F32_CHECK(l->input_norm, ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.post_attention_layernorm.weight", i);
        LOAD_F32_CHECK(l->post_attn_norm, ms, name);

        /* Fuse Q+K+V -> Q8_0 (skip if loaded from cache) */
        if (!l->wqkv_q8) {
            uint16_t *wq = NULL, *wk = NULL, *wv = NULL;
            snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.q_proj.weight", i);
            GET_BF16_CHECK(wq, ms, name);
            snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.k_proj.weight", i);
            GET_BF16_CHECK(wk, ms, name);
            snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.v_proj.weight", i);
            GET_BF16_CHECK(wv, ms, name);

            int q_rows = cfg->talker_heads * cfg->talker_head_dim;
            int kv_rows = cfg->talker_kv_heads * cfg->talker_head_dim;
            int total_rows = q_rows + kv_rows + kv_rows;
            size_t row_elems = (size_t)cfg->talker_hidden;
            uint16_t *fused = (uint16_t *)malloc((size_t)total_rows * row_elems * sizeof(uint16_t));
            if (fused && wq && wk && wv) {
                memcpy(fused, wq, (size_t)q_rows * row_elems * sizeof(uint16_t));
                memcpy(fused + (size_t)q_rows * row_elems, wk, (size_t)kv_rows * row_elems * sizeof(uint16_t));
                memcpy(fused + (size_t)(q_rows + kv_rows) * row_elems, wv, (size_t)kv_rows * row_elems * sizeof(uint16_t));
                l->wqkv_q8 = quantize_weight_bf16(fused, total_rows, (int)row_elems);
            }
            free(fused);
        }

        /* wo -> Q8_0 */
        if (!l->wo_q8) {
            uint16_t *wo = NULL;
            snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.o_proj.weight", i);
            GET_BF16_CHECK(wo, ms, name);
            if (wo)
                l->wo_q8 = quantize_weight_bf16(wo, cfg->talker_hidden, cfg->talker_heads * cfg->talker_head_dim);
        }

        /* Fuse gate+up -> Q8_0 */
        if (!l->gate_up_q8) {
            uint16_t *gate = NULL, *up = NULL;
            snprintf(name, sizeof(name), "talker.model.layers.%d.mlp.gate_proj.weight", i);
            GET_BF16_CHECK(gate, ms, name);
            snprintf(name, sizeof(name), "talker.model.layers.%d.mlp.up_proj.weight", i);
            GET_BF16_CHECK(up, ms, name);

            size_t gu_size = (size_t)cfg->talker_intermediate * cfg->talker_hidden;
            uint16_t *fused = (uint16_t *)malloc(2 * gu_size * sizeof(uint16_t));
            if (fused && gate && up) {
                memcpy(fused, gate, gu_size * sizeof(uint16_t));
                memcpy(fused + gu_size, up, gu_size * sizeof(uint16_t));
                l->gate_up_q8 = quantize_weight_bf16(fused, 2 * cfg->talker_intermediate, cfg->talker_hidden);
            }
            free(fused);
        }

        /* down -> Q8_0 */
        if (!l->down_q8) {
            uint16_t *down = NULL;
            snprintf(name, sizeof(name), "talker.model.layers.%d.mlp.down_proj.weight", i);
            GET_BF16_CHECK(down, ms, name);
            if (down)
                l->down_q8 = quantize_weight_bf16(down, cfg->talker_hidden, cfg->talker_intermediate);
        }
    }

    /* Final norm */
    LOAD_F32_CHECK(ctx->talker.norm, ms, "talker.model.norm.weight");

    /* Codec head: BF16 -> Q8_0 */
    if (!ctx->talker.codec_head_q8) {
        uint16_t *ch_bf16 = NULL;
        GET_BF16_CHECK(ch_bf16, ms, "talker.codec_head.weight");
        if (ch_bf16)
            ctx->talker.codec_head_q8 = quantize_weight_bf16(ch_bf16, cfg->talker_vocab_size, cfg->talker_hidden);
    }

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

    /* 31 codec embeddings (groups 1-31, BF16 row-lookup) */
    for (int g = 0; g < cfg->num_code_groups - 1; g++) {
        snprintf(name, sizeof(name), "talker.code_predictor.model.codec_embedding.%d.weight", g);
        GET_BF16_CHECK(ctx->subtalker.codec_embeddings_bf16[g], ms, name);
    }

    /* Input projection: BF16 -> Q8_0 (skip if loaded from cache) */
    if (!ctx->subtalker.input_proj_q8) {
        uint16_t *ip_bf16 = NULL;
        GET_BF16_CHECK(ip_bf16, ms, "talker.code_predictor.small_to_mtp_projection.weight");
        if (ip_bf16)
            ctx->subtalker.input_proj_q8 = quantize_weight_bf16(ip_bf16, cfg->subtalker_hidden, cfg->talker_hidden);
    }
    LOAD_F32_CHECK(ctx->subtalker.input_proj_bias, ms, "talker.code_predictor.small_to_mtp_projection.bias");

    /* Transformer layers */
    for (int i = 0; i < cfg->subtalker_layers; i++) {
        qwen_tts_subtalker_layer_t *l = &ctx->subtalker.layers[i];

        /* Load norms */
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.q_norm.weight", i);
        LOAD_F32_CHECK(l->q_norm_weight, ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.k_norm.weight", i);
        LOAD_F32_CHECK(l->k_norm_weight, ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.input_layernorm.weight", i);
        LOAD_F32_CHECK(l->input_norm, ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.post_attention_layernorm.weight", i);
        LOAD_F32_CHECK(l->post_attn_norm, ms, name);

        /* Fuse Q+K+V -> Q8_0 (skip if loaded from cache) */
        if (!l->wqkv_q8) {
            uint16_t *wq = NULL, *wk = NULL, *wv = NULL;
            snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.q_proj.weight", i);
            GET_BF16_CHECK(wq, ms, name);
            snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.k_proj.weight", i);
            GET_BF16_CHECK(wk, ms, name);
            snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.v_proj.weight", i);
            GET_BF16_CHECK(wv, ms, name);

            int q_rows = cfg->subtalker_heads * cfg->subtalker_head_dim;
            int kv_rows = cfg->subtalker_kv_heads * cfg->subtalker_head_dim;
            int total_rows = q_rows + kv_rows + kv_rows;
            size_t row_elems = (size_t)cfg->subtalker_hidden;
            uint16_t *fused = (uint16_t *)malloc((size_t)total_rows * row_elems * sizeof(uint16_t));
            if (fused && wq && wk && wv) {
                memcpy(fused, wq, (size_t)q_rows * row_elems * sizeof(uint16_t));
                memcpy(fused + (size_t)q_rows * row_elems, wk, (size_t)kv_rows * row_elems * sizeof(uint16_t));
                memcpy(fused + (size_t)(q_rows + kv_rows) * row_elems, wv, (size_t)kv_rows * row_elems * sizeof(uint16_t));
                l->wqkv_q8 = quantize_weight_bf16(fused, total_rows, (int)row_elems);
            }
            free(fused);
        }

        /* wo -> Q8_0 */
        if (!l->wo_q8) {
            uint16_t *wo = NULL;
            snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.o_proj.weight", i);
            GET_BF16_CHECK(wo, ms, name);
            if (wo)
                l->wo_q8 = quantize_weight_bf16(wo, cfg->subtalker_hidden, cfg->subtalker_heads * cfg->subtalker_head_dim);
        }

        /* Fuse gate+up -> Q8_0 */
        if (!l->gate_up_q8) {
            uint16_t *gate = NULL, *up = NULL;
            snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.mlp.gate_proj.weight", i);
            GET_BF16_CHECK(gate, ms, name);
            snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.mlp.up_proj.weight", i);
            GET_BF16_CHECK(up, ms, name);

            size_t gu_size = (size_t)cfg->subtalker_intermediate * cfg->subtalker_hidden;
            uint16_t *fused = (uint16_t *)malloc(2 * gu_size * sizeof(uint16_t));
            if (fused && gate && up) {
                memcpy(fused, gate, gu_size * sizeof(uint16_t));
                memcpy(fused + gu_size, up, gu_size * sizeof(uint16_t));
                l->gate_up_q8 = quantize_weight_bf16(fused, 2 * cfg->subtalker_intermediate, cfg->subtalker_hidden);
            }
            free(fused);
        }

        /* down -> Q8_0 */
        if (!l->down_q8) {
            uint16_t *down = NULL;
            snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.mlp.down_proj.weight", i);
            GET_BF16_CHECK(down, ms, name);
            if (down)
                l->down_q8 = quantize_weight_bf16(down, cfg->subtalker_hidden, cfg->subtalker_intermediate);
        }
    }

    /* Final norm */
    LOAD_F32_CHECK(ctx->subtalker.norm, ms, "talker.code_predictor.model.norm.weight");

    /* 31 LM heads: BF16 -> Q8_0 (skip if loaded from cache) */
    for (int g = 0; g < cfg->num_code_groups - 1; g++) {
        if (!ctx->subtalker.lm_heads_q8[g]) {
            uint16_t *lm_bf16 = NULL;
            snprintf(name, sizeof(name), "talker.code_predictor.lm_head.%d.weight", g);
            GET_BF16_CHECK(lm_bf16, ms, name);
            if (lm_bf16)
                ctx->subtalker.lm_heads_q8[g] = quantize_weight_bf16(lm_bf16, cfg->subtalker_vocab_size, cfg->subtalker_hidden);
        }
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

        /* Load F32 weights into locals, fuse, convert to FP16 (primary) + Q8_0 (fallback) */
        {
            float *wq = NULL, *wk = NULL, *wv = NULL, *wo = NULL;
            float *gate = NULL, *up = NULL, *down = NULL;

            snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.q_proj.weight", i);
            LOAD_F32_CHECK(wq, ms, name);
            snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.k_proj.weight", i);
            LOAD_F32_CHECK(wk, ms, name);
            snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.v_proj.weight", i);
            LOAD_F32_CHECK(wv, ms, name);
            snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.o_proj.weight", i);
            LOAD_F32_CHECK(wo, ms, name);

            snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp.gate_proj.weight", i);
            LOAD_F32_CHECK(gate, ms, name);
            snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp.up_proj.weight", i);
            LOAD_F32_CHECK(up, ms, name);
            snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp.down_proj.weight", i);
            LOAD_F32_CHECK(down, ms, name);

            int q_dim = cfg->codec_heads * (cfg->codec_hidden / cfg->codec_heads);
            int kv_dim = cfg->codec_kv_heads * (cfg->codec_hidden / cfg->codec_heads);
            int codec_hidden = cfg->codec_hidden;
            int intermediate = cfg->codec_intermediate;

            /* Fused QKV -> FP16 (primary) + Q8_0 (fallback), free F32 */
            if (wq && wk && wv) {
                int total_rows = q_dim + kv_dim + kv_dim;
                size_t n_elems = (size_t)total_rows * codec_hidden;
                float *fused = (float *)malloc(n_elems * sizeof(float));
                if (fused) {
                    memcpy(fused, wq, (size_t)q_dim * codec_hidden * sizeof(float));
                    memcpy(fused + (size_t)q_dim * codec_hidden, wk, (size_t)kv_dim * codec_hidden * sizeof(float));
                    memcpy(fused + (size_t)(q_dim + kv_dim) * codec_hidden, wv, (size_t)kv_dim * codec_hidden * sizeof(float));
                    l->wqkv_q8 = quantize_weight_f32(fused, total_rows, codec_hidden);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                    l->wqkv_f16 = (__fp16 *)malloc(n_elems * sizeof(__fp16));
                    if (l->wqkv_f16) kernel_f32_to_f16(l->wqkv_f16, fused, (int)n_elems);
#endif
                    free(fused);
                }
            }

            /* wo -> FP16 + Q8_0, free F32 */
            if (wo) {
                size_t n_elems = (size_t)codec_hidden * q_dim;
                l->wo_q8 = quantize_weight_f32(wo, codec_hidden, q_dim);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                l->wo_f16 = (__fp16 *)malloc(n_elems * sizeof(__fp16));
                if (l->wo_f16) kernel_f32_to_f16(l->wo_f16, wo, (int)n_elems);
#endif
                free(wo); wo = NULL;
            }

            /* Fused gate+up -> FP16 + Q8_0, free F32 */
            if (gate && up) {
                int gu_rows = 2 * intermediate;
                size_t n_elems = (size_t)gu_rows * codec_hidden;
                float *fused = (float *)malloc(n_elems * sizeof(float));
                if (fused) {
                    memcpy(fused, gate, (size_t)intermediate * codec_hidden * sizeof(float));
                    memcpy(fused + (size_t)intermediate * codec_hidden, up, (size_t)intermediate * codec_hidden * sizeof(float));
                    l->gate_up_q8 = quantize_weight_f32(fused, gu_rows, codec_hidden);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                    l->gate_up_f16 = (__fp16 *)malloc(n_elems * sizeof(__fp16));
                    if (l->gate_up_f16) kernel_f32_to_f16(l->gate_up_f16, fused, (int)n_elems);
#endif
                    free(fused);
                }
            }

            /* down -> FP16 + Q8_0, free F32 */
            if (down) {
                size_t n_elems = (size_t)codec_hidden * intermediate;
                l->down_q8 = quantize_weight_f32(down, codec_hidden, intermediate);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                l->down_f16 = (__fp16 *)malloc(n_elems * sizeof(__fp16));
                if (l->down_f16) kernel_f32_to_f16(l->down_f16, down, (int)n_elems);
#endif
                free(down); down = NULL;
            }

            /* Free original F32 weights */
            free(wq); free(wk); free(wv); free(wo);
            free(gate); free(up); free(down);
        }
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
     *
     * Vocoder uses F32 weights and activations (FP16 vocoder was slower on
     * test devices due to poor FP16 NEON throughput on some ARM cores).
     */
    LOAD_F32_CHECK(codec->vocoder_pre_conv_weight, ms, "decoder.decoder.0.conv.weight");
    LOAD_F32_CHECK(codec->vocoder_pre_conv_bias, ms, "decoder.decoder.0.conv.bias");

    for (int b = 0; b < 4; b++) {
        qwen_tts_vocoder_block_t *vb = &codec->vocoder_blocks[b];
        int idx = b + 1;  /* Python module index: decoder.decoder.{b+1} */

        /* SnakeBeta activation at block[0] — always F32 */
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

            /* SnakeBeta — always F32 */
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

static int ensure_codec_loaded(qwen_tts_ctx_t *ctx) {
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
 * Text projection helper
 *
 * Projects text embeddings: text_hidden -> text_hidden (SiLU) -> hidden
 * ======================================================================== */

static void ensure_text_scratch(qwen_tts_ctx_t *ctx, int text_hidden) {
    if (ctx->scratch_text_hidden_cap >= text_hidden) return;
    ctx->scratch_text_hidden = (float *)realloc(ctx->scratch_text_hidden,
                                                 text_hidden * sizeof(float));
    ctx->scratch_text_embed = (float *)realloc(ctx->scratch_text_embed,
                                                text_hidden * sizeof(float));
    ctx->scratch_text_hidden_cap = text_hidden;
}

static void text_projection(qwen_tts_ctx_t *ctx, const float *text_embed,
                             float *out, int text_hidden, int hidden) {
    ensure_text_scratch(ctx, text_hidden);
    float *fc1_out = ctx->scratch_text_hidden;
    int n_blocks = text_hidden / QK8_0;
    block_q8_0 x_q8[n_blocks]; /* text_hidden=2048 -> 64 blocks -> 2304 bytes on stack */
    kernel_quantize_x_q8(text_embed, text_hidden, x_q8);
    kernel_matvec_q8(fc1_out, ctx->talker.text_proj_fc1_q8, x_q8, text_hidden, n_blocks);
    if (ctx->talker.text_proj_fc1_bias)
        kernel_add_inplace(fc1_out, ctx->talker.text_proj_fc1_bias, text_hidden);
    kernel_silu_inplace(fc1_out, text_hidden);
    block_q8_0 fc1_q8[n_blocks];
    kernel_quantize_x_q8(fc1_out, text_hidden, fc1_q8);
    kernel_matvec_q8(out, ctx->talker.text_proj_fc2_q8, fc1_q8, hidden, n_blocks);
    if (ctx->talker.text_proj_fc2_bias)
        kernel_add_inplace(out, ctx->talker.text_proj_fc2_bias, hidden);
}

/* ========================================================================
 * Embed a text token: text_embedding -> text_projection
 * ======================================================================== */

static void embed_text_token(qwen_tts_ctx_t *ctx, int token_id, float *out) {
    int text_hidden = ctx->config.talker_text_hidden;
    int hidden = ctx->config.talker_hidden;
    ensure_text_scratch(ctx, text_hidden);
    float *text_embed = ctx->scratch_text_embed;
    kernel_bf16_to_f32(text_embed, ctx->talker.text_embedding_bf16 + (size_t)token_id * text_hidden, text_hidden);
    text_projection(ctx, text_embed, out, text_hidden, hidden);
}

/* ========================================================================
 * Embed a codec token: lookup from codec_embedding
 * ======================================================================== */

static void embed_codec_token(qwen_tts_ctx_t *ctx, int token_id, float *out) {
    int hidden = ctx->config.talker_hidden;
    kernel_bf16_to_f32(out, ctx->talker.codec_embedding_bf16 + (size_t)token_id * hidden, hidden);
}

/* ========================================================================
 * Load Model
 * ======================================================================== */

qwen_tts_ctx_t *qwen_tts_load(const char *model_dir) {
    double t0 = qwen_tts_time_ms();

    qwen_tts_ctx_t *ctx = (qwen_tts_ctx_t *)calloc(1, sizeof(qwen_tts_ctx_t));
    if (!ctx) return NULL;

    strncpy(ctx->model_dir, model_dir, sizeof(ctx->model_dir) - 1);
    /* Use override cache_dir if set, otherwise default to model_dir */
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

    /* Try loading pre-quantized weight cache first */
    int cache_loaded = load_quantized_cache(ctx);

    if (load_talker_weights(ctx, ms) != 0) {
        qwen_tts_free(ctx);
        return NULL;
    }
    load_subtalker_weights(ctx, ms);

    /* Save cache if we didn't load from cache */
    if (cache_loaded != 0) {
        save_quantized_cache(ctx);
    }

    /* Open codec decoder safetensors */
#ifndef __EMSCRIPTEN__
    if (ensure_codec_loaded(ctx) != 0) {
        /* Continue without codec (can still generate tokens). */
    }
#else
    if (qwen_tts_verbose >= 1) {
        fprintf(stderr, "WASM: deferring codec decoder load until decode stage\n");
    }
#endif

    kernel_init();

    double t1 = qwen_tts_time_ms();
    if (qwen_tts_verbose >= 1) {
        fprintf(stderr, "Model loaded in %.1f ms\n", t1 - t0);
    }

    return ctx;
}

void qwen_tts_set_cache_dir(qwen_tts_ctx_t *ctx, const char *cache_dir) {
    if (ctx && cache_dir)
        strncpy(ctx->cache_dir, cache_dir, sizeof(ctx->cache_dir) - 1);
}

int qwen_tts_save_cache(qwen_tts_ctx_t *ctx) {
#ifndef __EMSCRIPTEN__
    return save_quantized_cache(ctx);
#else
    return -1;
#endif
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
        free(l->wqkv_q8); free(l->wo_q8);
        free(l->gate_up_q8); free(l->down_q8);
        free(l->wqkv_f16); free(l->wo_f16);
        free(l->gate_up_f16); free(l->down_f16);
        free(l->wqkv_f32); free(l->wo_f32);
        free(l->gate_up_f32); free(l->down_f32);
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

    /* Free talker weights (norms, biases, Q8_0) */
    free(ctx->talker.text_proj_fc1_bias);
    free(ctx->talker.text_proj_fc2_bias);
    free(ctx->talker.text_proj_fc1_q8);
    free(ctx->talker.text_proj_fc2_q8);
    free(ctx->talker.codec_head_q8);
    free(ctx->talker.norm);
    for (int i = 0; i < ctx->config.talker_layers; i++) {
        qwen_tts_talker_layer_t *l = &ctx->talker.layers[i];
        free(l->q_norm_weight); free(l->k_norm_weight);
        free(l->input_norm); free(l->post_attn_norm);
        free(l->wqkv_q8); free(l->wo_q8);
        free(l->gate_up_q8); free(l->down_q8);
    }

    /* Free subtalker weights (norms, biases, Q8_0) */
    free(ctx->subtalker.input_proj_bias);
    free(ctx->subtalker.input_proj_q8);
    free(ctx->subtalker.norm);
    for (int g = 0; g < ctx->config.num_code_groups - 1; g++)
        free(ctx->subtalker.lm_heads_q8[g]);
    for (int i = 0; i < ctx->config.subtalker_layers; i++) {
        qwen_tts_subtalker_layer_t *l = &ctx->subtalker.layers[i];
        free(l->q_norm_weight); free(l->k_norm_weight);
        free(l->input_norm); free(l->post_attn_norm);
        free(l->wqkv_q8); free(l->wo_q8);
        free(l->gate_up_q8); free(l->down_q8);
    }

    /* Free KV caches and scratch buffers */
    free(ctx->talker_kv_k); free(ctx->talker_kv_v);
    free(ctx->subtalker_kv_k); free(ctx->subtalker_kv_v);
    free(ctx->codec_kv_k); free(ctx->codec_kv_v);
    free(ctx->tk_x); free(ctx->tk_x_norm);
    free(ctx->tk_q); free(ctx->tk_k); free(ctx->tk_v);
    free(ctx->tk_qkv);
    free(ctx->tk_attn_out); free(ctx->tk_proj_out);
    free(ctx->tk_gate); free(ctx->tk_up); free(ctx->tk_ffn_out);
    free(ctx->tk_scores);
    free(ctx->tk_rope_cos); free(ctx->tk_rope_sin);
    free(ctx->tk_pref_x); free(ctx->tk_pref_x_norm);
    free(ctx->tk_pref_q); free(ctx->tk_pref_k); free(ctx->tk_pref_v);
    free(ctx->tk_pref_attn_out); free(ctx->tk_pref_proj_out);
    free(ctx->tk_pref_gate); free(ctx->tk_pref_gate_up); free(ctx->tk_pref_ffn_out);
    free(ctx->st_x); free(ctx->st_x_norm);
    free(ctx->st_q); free(ctx->st_k); free(ctx->st_v);
    free(ctx->st_qkv);
    free(ctx->st_attn_out); free(ctx->st_logits);
    free(ctx->st_gate); free(ctx->st_up);
    free(ctx->st_embed); free(ctx->st_proj_hidden);
    free(ctx->st_scores);
    free(ctx->st_rope_cos); free(ctx->st_rope_sin);
    free(ctx->talker_rope_cos_cache); free(ctx->talker_rope_sin_cache);
    free(ctx->scratch_text_hidden); free(ctx->scratch_text_embed);

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

    double t_start = qwen_tts_time_ms();

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
    double t_prefill = qwen_tts_time_ms();

    /* Reset KV cache */
    ctx->talker_kv_len = 0;
    qwen_tts_talker_prefill(ctx, input_embeds, prefill_len);

    double t_prefill_done = qwen_tts_time_ms();
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

    double t_gen = qwen_tts_time_ms();
    ctx->perf_subtalker_ms = 0.0;

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
            {
                int n_blocks = hidden / QK8_0;
                block_q8_0 x_q8[n_blocks];
                kernel_quantize_x_q8(ctx->tk_x, hidden, x_q8);
                kernel_matvec_q8(logits, ctx->talker.codec_head_q8, x_q8, cfg->talker_vocab_size, n_blocks);
            }
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
        double t_st = qwen_tts_time_ms();
        qwen_tts_subtalker_generate(ctx, ctx->tk_x, token, codes);
        ctx->perf_subtalker_ms += qwen_tts_time_ms() - t_st;

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
            double elapsed = qwen_tts_time_ms() - t_gen;
            fprintf(stderr, "\r  Token %d (%.1f ms/token)...", n_generated, elapsed / n_generated);
        }
    }

    if (stop_reason == 0) {
        stop_reason = 2;
        stop_step = max_tokens;
    }

    double t_gen_done = qwen_tts_time_ms();
    ctx->perf_talker_ms = t_gen_done - t_gen;
    ctx->perf_codec_tokens = n_generated;

    if (qwen_tts_verbose >= 1) {
        fprintf(stderr, "\r                                        \r");  /* clear progress line */
        fprintf(stderr, "Generated %d codec tokens in %.1f ms (%.1f ms/token)\n",
                n_generated, ctx->perf_talker_ms,
                n_generated > 0 ? ctx->perf_talker_ms / n_generated : 0);
        /* Time decomposition: talker (pure) vs sub-talker */
        {
            double talker_pure_ms = ctx->perf_talker_ms - ctx->perf_subtalker_ms;
            double total_gen = ctx->perf_talker_ms;
            if (total_gen > 0) {
                fprintf(stderr, "Talker: %.0fms (%.1f%%) | Sub-talker: %.0fms (%.1f%%)\n",
                        talker_pure_ms, 100.0 * talker_pure_ms / total_gen,
                        ctx->perf_subtalker_ms, 100.0 * ctx->perf_subtalker_ms / total_gen);
            }
        }
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

    if (ensure_codec_loaded(ctx) != 0) {
        fprintf(stderr,
                "Error: codec decoder weights are unavailable (missing /model/speech_tokenizer/*.safetensors)\n");
        free(all_codes); free(next_embed);
        *out_samples = 0;
        return NULL;
    }

    /* ---- Codec Decode ---- */
    double t_codec = qwen_tts_time_ms();

    float *audio = qwen_tts_codec_decode(ctx, all_codes, n_generated, out_samples);
    if (!audio || *out_samples <= 0) {
        free(all_codes); free(next_embed);
        *out_samples = 0;
        return NULL;
    }

    double t_codec_done = qwen_tts_time_ms();
    ctx->perf_codec_ms = t_codec_done - t_codec;
    ctx->perf_total_ms = t_codec_done - t_start;

    if (qwen_tts_verbose >= 1) {
        fprintf(stderr, "Codec decode: %d samples in %.1f ms\n", *out_samples, ctx->perf_codec_ms);
        /* Full time decomposition */
        {
            double talker_pure_ms = ctx->perf_talker_ms - ctx->perf_subtalker_ms;
            double total = ctx->perf_total_ms;
            if (total > 0) {
                fprintf(stderr, "Talker: %.0fms (%.1f%%) | Sub-talker: %.0fms (%.1f%%) | Codec: %.0fms (%.1f%%)\n",
                        talker_pure_ms, 100.0 * talker_pure_ms / total,
                        ctx->perf_subtalker_ms, 100.0 * ctx->perf_subtalker_ms / total,
                        ctx->perf_codec_ms, 100.0 * ctx->perf_codec_ms / total);
            }
        }
        fprintf(stderr, "Total: %.1f ms (%.2f s audio, %.2fx realtime)\n",
                ctx->perf_total_ms,
                *out_samples > 0 ? (float)*out_samples / QWEN_TTS_SAMPLE_RATE : 0,
                *out_samples > 0 ? ((float)*out_samples / QWEN_TTS_SAMPLE_RATE) / (ctx->perf_total_ms / 1000.0) : 0);
    }

    free(all_codes); free(next_embed);

    return audio;
}

/* ========================================================================
 * Streaming Generate
 *
 * Same logic as qwen_tts_generate() but periodically decodes accumulated
 * codec tokens and delivers new PCM samples via callback.
 * Pipelined: vocoder runs on a background thread while talker generates
 * the next chunk of tokens, overlapping compute.
 * ======================================================================== */

/* Vocoder pipeline: background thread job for batch codec decode */
typedef struct {
    qwen_tts_ctx_t *ctx;
    const int *codes;      /* pointer into all_codes (stable, no realloc) */
    int num_groups;        /* quantizers per frame */
    int decode_len;        /* total tokens to decode (new + context) */
    int n_ctx;             /* context frames to trim from output */
    float *audio;          /* output: malloc'd audio buffer */
    int n_audio;           /* output: number of audio samples */
    double decode_ms;      /* output: decode time */
} vocoder_job_t;

static void *vocoder_thread_fn(void *arg) {
    vocoder_job_t *job = (vocoder_job_t *)arg;
    double t0 = qwen_tts_time_ms();
    job->audio = qwen_tts_codec_decode(job->ctx, job->codes,
                                        job->decode_len, &job->n_audio);
    job->decode_ms = qwen_tts_time_ms() - t0;
    return NULL;
}

int qwen_tts_generate_stream(
    qwen_tts_ctx_t *ctx,
    const char *text,
    const char *speaker,
    const char *language,
    int chunk_size,
    qwen_tts_audio_cb audio_cb,
    void *userdata
) {
    if (!ctx || !audio_cb) return -1;

    /* Parse text as comma-separated token IDs (same as qwen_tts_generate) */
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

    double t_start = qwen_tts_time_ms();

    /* ---- Look up speaker and language IDs ---- */
    int speaker_codec_id = -1;
    if (speaker && strlen(speaker) > 0) {
        for (int i = 0; i < cfg->n_speakers; i++) {
            if (strcasecmp(cfg->speaker_names[i], speaker) == 0) {
                speaker_codec_id = cfg->speaker_ids[i];
                break;
            }
        }
        if (speaker_codec_id < 0)
            fprintf(stderr, "Warning: speaker '%s' not found, using no speaker embedding\n", speaker);
    }

    int language_codec_id = -1;
    if (language && strlen(language) > 0 && strcasecmp(language, "auto") != 0) {
        for (int i = 0; i < cfg->n_languages; i++) {
            if (strcasecmp(cfg->language_names[i], language) == 0) {
                language_codec_id = cfg->language_ids[i];
                break;
            }
        }
        if (language_codec_id < 0)
            fprintf(stderr, "Warning: language '%s' not found\n", language);
    }

    /* ---- Build prefix embedding sequence (same as qwen_tts_generate) ---- */
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
    if (speaker_codec_id >= 0)
        codec_prefix[n_codec_prefix++] = speaker_codec_id;
    codec_prefix[n_codec_prefix++] = cfg->codec_pad_id;
    codec_prefix[n_codec_prefix++] = cfg->codec_bos_id;

    int prefill_len = 3 + n_codec_prefix;
    float *input_embeds = (float *)calloc((size_t)prefill_len * hidden, sizeof(float));

    for (int i = 0; i < 3; i++)
        embed_text_token(ctx, text_tokens[i], input_embeds + i * hidden);

    float *tts_pad_proj = (float *)malloc(hidden * sizeof(float));
    float *tts_bos_proj = (float *)malloc(hidden * sizeof(float));
    float *tts_eos_proj = (float *)malloc(hidden * sizeof(float));
    float *codec_emb_tmp = (float *)malloc(hidden * sizeof(float));
    embed_text_token(ctx, QWEN_TTS_TOKEN_TTS_PAD, tts_pad_proj);
    embed_text_token(ctx, QWEN_TTS_TOKEN_TTS_BOS, tts_bos_proj);
    embed_text_token(ctx, QWEN_TTS_TOKEN_TTS_EOS, tts_eos_proj);

    for (int i = 0; i < n_codec_prefix - 1; i++) {
        float *dst = input_embeds + (3 + i) * hidden;
        if (i < n_codec_prefix - 2)
            memcpy(dst, tts_pad_proj, hidden * sizeof(float));
        else
            memcpy(dst, tts_bos_proj, hidden * sizeof(float));
        embed_codec_token(ctx, codec_prefix[i], codec_emb_tmp);
        kernel_add_inplace(dst, codec_emb_tmp, hidden);
    }

    {
        int pos = 3 + n_codec_prefix - 1;
        float *dst = input_embeds + pos * hidden;
        embed_text_token(ctx, text_tokens[3], dst);
        embed_codec_token(ctx, cfg->codec_bos_id, codec_emb_tmp);
        kernel_add_inplace(dst, codec_emb_tmp, hidden);
    }
    free(codec_emb_tmp);

    int n_trailing = (n_text_tokens - 4 - 5) + 1;
    if (n_trailing < 1) n_trailing = 1;
    float *trailing_text = (float *)calloc((size_t)n_trailing * hidden, sizeof(float));
    for (int i = 0; i < n_trailing - 1; i++)
        embed_text_token(ctx, text_tokens[4 + i], trailing_text + i * hidden);
    memcpy(trailing_text + (n_trailing - 1) * hidden, tts_eos_proj, hidden * sizeof(float));

    /* ---- Prefill ---- */
    ctx->talker_kv_len = 0;
    qwen_tts_talker_prefill(ctx, input_embeds, prefill_len);
    free(input_embeds);

    /* ---- Autoregressive generation with streaming decode ---- */
    int fixed_tokens = ctx->fixed_codec_tokens > 0 ? ctx->fixed_codec_tokens : 0;
    int max_tokens = fixed_tokens > 0 ? fixed_tokens : ctx->max_new_tokens;
    int *all_codes = (int *)calloc((size_t)max_tokens * num_groups, sizeof(int));
    int *generated_tokens = (int *)calloc(max_tokens, sizeof(int));
    int n_generated = 0;
    int aborted = 0;

    float *logits = (float *)malloc(cfg->talker_vocab_size * sizeof(float));
    float *next_embed = (float *)malloc(hidden * sizeof(float));
    float *emb_tmp = (float *)malloc(hidden * sizeof(float));
    float rng_state = (float)ctx->sample_seed;

    int suppress_start = cfg->talker_vocab_size - 1024;
    int *suppress_tokens = (int *)malloc(1024 * sizeof(int));
    int n_suppress = 0;
    for (int i = suppress_start; i < cfg->talker_vocab_size; i++) {
        if (i != cfg->codec_eos_id) suppress_tokens[n_suppress++] = i;
    }

    /* Ensure codec is loaded before starting (needed for streaming decode) */
    if (ensure_codec_loaded(ctx) != 0) {
        fprintf(stderr, "Error: codec decoder weights are unavailable\n");
        free(all_codes); free(generated_tokens); free(logits); free(next_embed);
        free(emb_tmp); free(suppress_tokens); free(trailing_text);
        free(tts_pad_proj); free(tts_bos_proj); free(tts_eos_proj); free(text_tokens);
        return -1;
    }

    int prev_audio_len = 0;    /* samples already sent via callback */
    double t_gen = qwen_tts_time_ms();
    ctx->perf_subtalker_ms = 0.0;

    /* chunk_size > 0: batch-decode every chunk_size tokens with pipelined vocoder
     * chunk_size == 0: batch mode (decode all at EOS) */
    int effective_chunk = (chunk_size > 0) ? chunk_size : 0;

    /* Sliding window context: re-decode this many previous frames for boundary quality.
     * 0 = no context (fastest), 2-4 = smoother boundaries (costs ~170ms per context frame). */
    int context_frames = 0;
    int accumulated_new = 0;    /* tokens accumulated since last batch decode */
    int prev_decoded_end = 0;   /* how many tokens have been decoded so far */

    /* Pipeline: vocoder runs on background thread */
    vocoder_job_t voc_job = {0};
    pthread_t voc_thread;
    int voc_thread_active = 0;

    for (int step = 0; step < max_tokens; step++) {
        if (step == 0) {
            {
                int n_blocks = hidden / QK8_0;
                block_q8_0 x_q8[n_blocks];
                kernel_quantize_x_q8(ctx->tk_x, hidden, x_q8);
                kernel_matvec_q8(logits, ctx->talker.codec_head_q8, x_q8, cfg->talker_vocab_size, n_blocks);
            }
        } else {
            qwen_tts_talker_forward(ctx, next_embed, logits);
        }

        for (int i = 0; i < n_suppress; i++)
            logits[suppress_tokens[i]] = -1e9f;

        kernel_apply_repetition_penalty(logits, generated_tokens, n_generated,
                                        cfg->talker_vocab_size, ctx->repetition_penalty);

        int token = kernel_sample_top_k(logits, cfg->talker_vocab_size, ctx->top_k,
                                         ctx->top_p, ctx->temperature, &rng_state);

        if (fixed_tokens > 0 && token == cfg->codec_eos_id && n_generated < fixed_tokens) {
            float eos_logit = logits[cfg->codec_eos_id];
            logits[cfg->codec_eos_id] = -1e9f;
            token = kernel_sample_top_k(logits, cfg->talker_vocab_size, ctx->top_k,
                                        ctx->top_p, ctx->temperature, &rng_state);
            logits[cfg->codec_eos_id] = eos_logit;
        }

        int is_eos = (fixed_tokens == 0 && token == cfg->codec_eos_id);
        if (is_eos) {
            if (qwen_tts_verbose >= 1)
                fprintf(stderr, "EOS at step %d\n", step);
        }

        if (!is_eos) {
            generated_tokens[n_generated] = token;

            int codes[QWEN_TTS_NUM_CODE_GROUPS];
            double t_st = qwen_tts_time_ms();
            qwen_tts_subtalker_generate(ctx, ctx->tk_x, token, codes);
            ctx->perf_subtalker_ms += qwen_tts_time_ms() - t_st;

            memcpy(all_codes + n_generated * num_groups, codes, num_groups * sizeof(int));
            n_generated++;
            accumulated_new++;

            /* Launch pipelined batch decode when we've accumulated enough tokens.
             * Vocoder runs on background thread while talker continues generating. */
            if (effective_chunk > 0 && accumulated_new >= effective_chunk) {
                /* Join previous vocoder thread and deliver its audio */
                if (voc_thread_active) {
                    pthread_join(voc_thread, NULL);
                    voc_thread_active = 0;
                    if (voc_job.audio && voc_job.n_audio > 0) {
                        int ctx_samples = voc_job.n_ctx * QWEN_TTS_DECODE_UPSAMPLE;
                        int emit_len = voc_job.n_audio - ctx_samples;
                        if (emit_len > 0) {
                            prev_audio_len += emit_len;
                            if (qwen_tts_verbose >= 1)
                                fprintf(stderr, "Stream chunk: %d samples (%d total, %.2fs), decode %.0fms\n",
                                        emit_len, prev_audio_len,
                                        (float)prev_audio_len / QWEN_TTS_SAMPLE_RATE, voc_job.decode_ms);
                            int ret = audio_cb(voc_job.audio + ctx_samples, emit_len, userdata);
                            if (ret != 0) {
                                aborted = 1;
                                free(voc_job.audio);
                                break;
                            }
                        }
                    }
                    free(voc_job.audio);
                    voc_job.audio = NULL;
                }

                /* Launch new vocoder decode on background thread */
                int n_ctx = (prev_decoded_end > 0) ? context_frames : 0;
                if (n_ctx > prev_decoded_end) n_ctx = prev_decoded_end;
                int decode_start = n_generated - accumulated_new - n_ctx;

                voc_job.ctx = ctx;
                voc_job.codes = all_codes + decode_start * num_groups;
                voc_job.num_groups = num_groups;
                voc_job.decode_len = accumulated_new + n_ctx;
                voc_job.n_ctx = n_ctx;
                voc_job.audio = NULL;
                voc_job.n_audio = 0;
                voc_job.decode_ms = 0;

                if (qwen_tts_verbose >= 1)
                    fprintf(stderr, "Pipeline: launch vocoder (%d new +%d ctx), talker continues...\n",
                            accumulated_new, n_ctx);

                pthread_create(&voc_thread, NULL, vocoder_thread_fn, &voc_job);
                voc_thread_active = 1;

                prev_decoded_end = n_generated;
                accumulated_new = 0;
            }
        }

        /* At EOS or loop end: join vocoder, flush remaining tokens */
        if (is_eos) {
            /* Join any running vocoder thread and deliver its audio */
            if (voc_thread_active) {
                pthread_join(voc_thread, NULL);
                voc_thread_active = 0;
                if (voc_job.audio && voc_job.n_audio > 0) {
                    int ctx_samples = voc_job.n_ctx * QWEN_TTS_DECODE_UPSAMPLE;
                    int emit_len = voc_job.n_audio - ctx_samples;
                    if (emit_len > 0) {
                        prev_audio_len += emit_len;
                        if (qwen_tts_verbose >= 1)
                            fprintf(stderr, "Stream chunk: %d samples (%d total, %.2fs), decode %.0fms\n",
                                    emit_len, prev_audio_len,
                                    (float)prev_audio_len / QWEN_TTS_SAMPLE_RATE, voc_job.decode_ms);
                        int ret = audio_cb(voc_job.audio + ctx_samples, emit_len, userdata);
                        if (ret != 0) aborted = 1;
                    }
                }
                free(voc_job.audio);
                voc_job.audio = NULL;
            }

            /* Decode remaining accumulated tokens (synchronous) */
            if (effective_chunk > 0 && accumulated_new > 0) {
                int n_ctx = (prev_decoded_end > 0) ? context_frames : 0;
                if (n_ctx > prev_decoded_end) n_ctx = prev_decoded_end;
                int decode_start = n_generated - accumulated_new - n_ctx;
                int decode_len = accumulated_new + n_ctx;

                int n_audio = 0;
                float *audio = qwen_tts_codec_decode(ctx, all_codes + decode_start * num_groups,
                                                      decode_len, &n_audio);
                if (audio && n_audio > 0) {
                    int ctx_samples = n_ctx * QWEN_TTS_DECODE_UPSAMPLE;
                    int emit_len = n_audio - ctx_samples;
                    if (emit_len > 0) {
                        prev_audio_len += emit_len;
                        if (qwen_tts_verbose >= 1)
                            fprintf(stderr, "Stream final: %d new (+%d ctx), %d samples (%d total, %.2fs)\n",
                                    accumulated_new, n_ctx, emit_len, prev_audio_len,
                                    (float)prev_audio_len / QWEN_TTS_SAMPLE_RATE);
                        int ret = audio_cb(audio + ctx_samples, emit_len, userdata);
                        if (ret != 0) aborted = 1;
                    }
                }
                free(audio);
            }
        }

        /* Batch mode (chunk_size==0): decode all at EOS */
        if (effective_chunk == 0 && is_eos && n_generated > 0) {
            int n_audio = 0;
            float *audio = qwen_tts_codec_decode(ctx, all_codes, n_generated, &n_audio);
            if (audio && n_audio > 0) {
                if (qwen_tts_verbose >= 1)
                    fprintf(stderr, "Batch decode: %d samples (%.2fs)\n",
                            n_audio, (float)n_audio / QWEN_TTS_SAMPLE_RATE);
                int ret = audio_cb(audio, n_audio, userdata);
                prev_audio_len = n_audio;
                if (ret != 0) aborted = 1;
            }
            free(audio);
        }

        if (is_eos) break;

        /* Build next input embedding */
        if (!is_eos) {
            memset(next_embed, 0, hidden * sizeof(float));
            embed_codec_token(ctx, token, emb_tmp);
            kernel_add_inplace(next_embed, emb_tmp, hidden);
            for (int g = 1; g < num_groups; g++) {
                int emb_dim = hidden;
                int code_g = all_codes[(n_generated - 1) * num_groups + g];
                kernel_bf16_to_f32(emb_tmp, ctx->subtalker.codec_embeddings_bf16[g - 1] +
                                   (size_t)code_g * emb_dim, emb_dim);
                kernel_add_inplace(next_embed, emb_tmp, hidden);
            }
            if (step < n_trailing)
                kernel_add_inplace(next_embed, trailing_text + step * hidden, hidden);
            else
                kernel_add_inplace(next_embed, tts_pad_proj, hidden);
        }

        if (ctx->progress_cb)
            ctx->progress_cb(step + 1, max_tokens, ctx->progress_cb_userdata);
        if (qwen_tts_verbose >= 1 && (n_generated % 10 == 0)) {
            double elapsed = qwen_tts_time_ms() - t_gen;
            fprintf(stderr, "\r  Token %d (%.1f ms/token)...", n_generated, elapsed / n_generated);
        }
    }

    /* Safety: join vocoder thread if still active (e.g., loop aborted or max_tokens hit) */
    if (voc_thread_active) {
        pthread_join(voc_thread, NULL);
        voc_thread_active = 0;
        /* Deliver remaining audio if not aborted */
        if (!aborted && voc_job.audio && voc_job.n_audio > 0) {
            int ctx_samples = voc_job.n_ctx * QWEN_TTS_DECODE_UPSAMPLE;
            int emit_len = voc_job.n_audio - ctx_samples;
            if (emit_len > 0) {
                prev_audio_len += emit_len;
                audio_cb(voc_job.audio + ctx_samples, emit_len, userdata);
            }
        }
        free(voc_job.audio);
        voc_job.audio = NULL;
    }

    double t_gen_done = qwen_tts_time_ms();
    ctx->perf_talker_ms = t_gen_done - t_gen;
    ctx->perf_codec_tokens = n_generated;
    ctx->perf_total_ms = t_gen_done - t_start;

    if (qwen_tts_verbose >= 1) {
        fprintf(stderr, "\r                                        \r");
        fprintf(stderr, "Stream generate: %d codec tokens, %d audio samples sent, total %.1f ms\n",
                n_generated, prev_audio_len, ctx->perf_total_ms);
    }

    free(all_codes); free(generated_tokens); free(logits); free(next_embed);
    free(emb_tmp); free(suppress_tokens); free(trailing_text);
    free(tts_pad_proj); free(tts_bos_proj); free(tts_eos_proj); free(text_tokens);

    if (aborted) return 1;
    if (n_generated == 0) return -1;
    return 0;
}
