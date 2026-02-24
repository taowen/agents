/*
 * qwen_tts_quant.c - Q4_K Weight Quantization for Qwen3-TTS
 *
 * Contains:
 *   - Q4_K matvec kernel (quantize x to Q8_K, then vec_dot per row)
 *   - Fused SwiGLU variant for Q4_K
 *   - BF16 -> Q4_K quantization function
 *   - Weight cache save/load (binary .qcache format, version 3)
 */

#include "qwen_tts_quant.h"
#include "qwen_tts.h"
#include "qwen_tts_kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int qwen_tts_verbose;

/* ========================================================================
 * Q4_K matvec: out[rows] = A_q4k @ x
 *
 * Strategy: quantize x to Q8_K once, then call vec_dot per row.
 * Each row has (cols / QK_K) Q4_K blocks.
 * ======================================================================== */

void kernel_matvec_q4k(float *out, const block_q4_K *A_q4k,
                       const float *x, int rows, int cols) {
    /* Quantize x vector to Q8_K */
    int blocks_per_row = cols / QK_K;

    /* Use thread-local static buffers for x quantization */
    static block_q8_K *x_q8k = NULL;
    static int x_q8k_cap = 0;  /* in blocks */
    if (blocks_per_row > x_q8k_cap) {
        free(x_q8k);
        x_q8k = (block_q8_K *)malloc((size_t)blocks_per_row * sizeof(block_q8_K));
        x_q8k_cap = blocks_per_row;
    }

    quantize_row_q8_K(x, x_q8k, (int64_t)cols);

    /* Compute dot product for each row */
#ifdef USE_OPENMP
    #pragma omp parallel for schedule(static) num_threads(2) if(rows >= 512)
#endif
    for (int r = 0; r < rows; r++) {
        const block_q4_K *row = A_q4k + (size_t)r * blocks_per_row;
        vec_dot_q4_K_q8_K(cols, &out[r], row, x_q8k);
    }
}

/* ========================================================================
 * Fused SwiGLU with Q4_K weights
 *
 * gate_up_q4k layout: [2*intermediate, hidden] in Q4_K blocks
 * First intermediate rows = gate, next intermediate rows = up.
 * ======================================================================== */

void kernel_swiglu_matvec_q4k(float *out, const block_q4_K *gate_up_q4k,
                               const float *x, int intermediate, int hidden) {
    static float *up_scratch = NULL;
    static size_t up_scratch_cap = 0;

    if ((size_t)intermediate > up_scratch_cap) {
        free(up_scratch);
        up_scratch = (float *)malloc((size_t)intermediate * sizeof(float));
        up_scratch_cap = (size_t)intermediate;
    }

    int blocks_per_row = hidden / QK_K;

    /* Quantize x to Q8_K once, reuse for both gate and up */
    static block_q8_K *x_q8k = NULL;
    static int x_q8k_cap = 0;
    if (blocks_per_row > x_q8k_cap) {
        free(x_q8k);
        x_q8k = (block_q8_K *)malloc((size_t)blocks_per_row * sizeof(block_q8_K));
        x_q8k_cap = blocks_per_row;
    }
    quantize_row_q8_K(x, x_q8k, (int64_t)hidden);

    /* Gate (first half of rows) */
    for (int r = 0; r < intermediate; r++) {
        const block_q4_K *row = gate_up_q4k + (size_t)r * blocks_per_row;
        vec_dot_q4_K_q8_K(hidden, &out[r], row, x_q8k);
    }

    /* Up (second half of rows) */
    for (int r = 0; r < intermediate; r++) {
        const block_q4_K *row = gate_up_q4k + (size_t)(intermediate + r) * blocks_per_row;
        vec_dot_q4_K_q8_K(hidden, &up_scratch[r], row, x_q8k);
    }

    /* SiLU(gate) * up */
    for (int i = 0; i < intermediate; i++) {
        float g = out[i];
        out[i] = (g / (1.0f + expf(-g))) * up_scratch[i];
    }
}

/* ========================================================================
 * BF16 -> Q4_K quantization
 *
 * Converts a [rows, cols] BF16 weight matrix to Q4_K blocks.
 * Each row has (cols / QK_K) blocks.
 * ======================================================================== */

void quantize_bf16_to_q4k(const uint16_t *bf16, int rows, int cols,
                           block_q4_K **out_q4k) {
    int blocks_per_row = cols / QK_K;
    size_t total_blocks = (size_t)rows * blocks_per_row;

    *out_q4k = (block_q4_K *)malloc(total_blocks * sizeof(block_q4_K));
    if (!*out_q4k) return;

    /* Temp buffer for one row of F32 */
    float *row_f32 = (float *)malloc((size_t)cols * sizeof(float));
    if (!row_f32) {
        free(*out_q4k);
        *out_q4k = NULL;
        return;
    }

    for (int r = 0; r < rows; r++) {
        const uint16_t *src = bf16 + (size_t)r * cols;

        /* BF16 -> F32 */
        for (int c = 0; c < cols; c++) {
            uint32_t bits = ((uint32_t)src[c]) << 16;
            __builtin_memcpy(&row_f32[c], &bits, sizeof(float));
        }

        /* F32 -> Q4_K */
        quantize_row_q4_K_ref(row_f32, *out_q4k + (size_t)r * blocks_per_row, (int64_t)cols);
    }

    free(row_f32);
}

/* ========================================================================
 * Pre-quantized Weight Cache
 *
 * Binary cache file format (.qcache) version 3:
 *   header (qcache_header_t)
 *   for each talker layer:
 *     wqkv_q4k | gate_up_q4k | wo_q4k | down_q4k
 *   for each subtalker layer:
 *     wqkv_q4k | gate_up_q4k | wo_q4k | down_q4k
 * ======================================================================== */

#ifndef __EMSCRIPTEN__
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#define QCACHE_MAGIC   0x31435151   /* "QQC1" */
#define QCACHE_VERSION 3

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t source_size;         /* original safetensors total file size for validation */
    uint32_t n_talker_layers;
    uint32_t n_subtalker_layers;
    /* Talker per-layer sizes (Q4_K byte counts) */
    uint32_t tk_wqkv_q4k_bytes;
    uint32_t tk_gate_up_q4k_bytes;
    uint32_t tk_wo_q4k_bytes;
    uint32_t tk_down_q4k_bytes;
    /* Subtalker per-layer sizes (Q4_K byte counts) */
    uint32_t st_wqkv_q4k_bytes;
    uint32_t st_gate_up_q4k_bytes;
    uint32_t st_wo_q4k_bytes;
    uint32_t st_down_q4k_bytes;
    uint32_t reserved[8];
} qcache_header_t;

#ifndef __EMSCRIPTEN__

static uint64_t get_safetensors_size(const char *model_dir) {
    char path[1024];
    uint64_t total = 0;
    struct stat st;

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

/* Helper to compute Q4_K byte count for [rows, cols] */
static inline uint32_t q4k_bytes(int rows, int cols) {
    int blocks_per_row = cols / QK_K;
    return (uint32_t)((size_t)rows * blocks_per_row * sizeof(block_q4_K));
}

int save_quantized_cache(struct qwen_tts_ctx *ctx) {
    qwen_tts_config_t *cfg = &ctx->config;
    char path[1024];

    snprintf(path, sizeof(path), "%s/model.qcache", ctx->cache_dir);

    /* Compute per-layer Q4_K sizes - Talker */
    int tk_qkv_rows = cfg->talker_heads * cfg->talker_head_dim +
                       2 * cfg->talker_kv_heads * cfg->talker_head_dim;
    uint32_t tk_wqkv_bytes = q4k_bytes(tk_qkv_rows, cfg->talker_hidden);

    int tk_gu_rows = 2 * cfg->talker_intermediate;
    uint32_t tk_gate_up_bytes = q4k_bytes(tk_gu_rows, cfg->talker_hidden);

    int tk_wo_rows = cfg->talker_hidden;
    int tk_wo_cols = cfg->talker_heads * cfg->talker_head_dim;
    uint32_t tk_wo_bytes = q4k_bytes(tk_wo_rows, tk_wo_cols);

    int tk_down_rows = cfg->talker_hidden;
    int tk_down_cols = cfg->talker_intermediate;
    uint32_t tk_down_bytes = q4k_bytes(tk_down_rows, tk_down_cols);

    /* Compute per-layer Q4_K sizes - Subtalker */
    int st_qkv_rows = cfg->subtalker_heads * cfg->subtalker_head_dim +
                       2 * cfg->subtalker_kv_heads * cfg->subtalker_head_dim;
    uint32_t st_wqkv_bytes = q4k_bytes(st_qkv_rows, cfg->subtalker_hidden);

    int st_gu_rows = 2 * cfg->subtalker_intermediate;
    uint32_t st_gate_up_bytes = q4k_bytes(st_gu_rows, cfg->subtalker_hidden);

    int st_wo_rows = cfg->subtalker_hidden;
    int st_wo_cols = cfg->subtalker_heads * cfg->subtalker_head_dim;
    uint32_t st_wo_bytes = q4k_bytes(st_wo_rows, st_wo_cols);

    int st_down_rows = cfg->subtalker_hidden;
    int st_down_cols = cfg->subtalker_intermediate;
    uint32_t st_down_bytes = q4k_bytes(st_down_rows, st_down_cols);

    /* Build header */
    qcache_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = QCACHE_MAGIC;
    hdr.version = QCACHE_VERSION;
    hdr.source_size = get_safetensors_size(ctx->model_dir);
    hdr.n_talker_layers = (uint32_t)cfg->talker_layers;
    hdr.n_subtalker_layers = (uint32_t)cfg->subtalker_layers;
    hdr.tk_wqkv_q4k_bytes = tk_wqkv_bytes;
    hdr.tk_gate_up_q4k_bytes = tk_gate_up_bytes;
    hdr.tk_wo_q4k_bytes = tk_wo_bytes;
    hdr.tk_down_q4k_bytes = tk_down_bytes;
    hdr.st_wqkv_q4k_bytes = st_wqkv_bytes;
    hdr.st_gate_up_q4k_bytes = st_gate_up_bytes;
    hdr.st_wo_q4k_bytes = st_wo_bytes;
    hdr.st_down_q4k_bytes = st_down_bytes;

    FILE *f = fopen(path, "wb");
    if (!f) {
        if (qwen_tts_verbose >= 1)
            fprintf(stderr, "Warning: cannot create qcache at %s\n", path);
        return -1;
    }

    fwrite(&hdr, sizeof(hdr), 1, f);

    #define WRITE_OR_ZERO(ptr, n_bytes) do { \
        if (ptr) fwrite(ptr, 1, n_bytes, f); \
        else { void *z = calloc(1, n_bytes); fwrite(z, 1, n_bytes, f); free(z); } \
    } while(0)

    /* Write talker layers */
    for (int i = 0; i < cfg->talker_layers; i++) {
        qwen_tts_talker_layer_t *l = &ctx->talker.layers[i];
        WRITE_OR_ZERO(l->wqkv_q4k, tk_wqkv_bytes);
        WRITE_OR_ZERO(l->gate_up_q4k, tk_gate_up_bytes);
        WRITE_OR_ZERO(l->wo_q4k, tk_wo_bytes);
        WRITE_OR_ZERO(l->down_q4k, tk_down_bytes);
    }

    /* Write subtalker layers */
    for (int i = 0; i < cfg->subtalker_layers; i++) {
        qwen_tts_subtalker_layer_t *l = &ctx->subtalker.layers[i];
        WRITE_OR_ZERO(l->wqkv_q4k, st_wqkv_bytes);
        WRITE_OR_ZERO(l->gate_up_q4k, st_gate_up_bytes);
        WRITE_OR_ZERO(l->wo_q4k, st_wo_bytes);
        WRITE_OR_ZERO(l->down_q4k, st_down_bytes);
    }

    #undef WRITE_OR_ZERO

    fclose(f);
    if (qwen_tts_verbose >= 1)
        fprintf(stderr, "Saved quantized cache to %s\n", path);
    return 0;
}

int load_quantized_cache(struct qwen_tts_ctx *ctx) {
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

    /* Validate header */
    if (hdr->magic != QCACHE_MAGIC || hdr->version != QCACHE_VERSION) {
        munmap(mapped, file_size);
        return -1;
    }
    if ((int)hdr->n_talker_layers != cfg->talker_layers ||
        (int)hdr->n_subtalker_layers != cfg->subtalker_layers) {
        munmap(mapped, file_size);
        return -1;
    }

    /* Validate source file size */
    uint64_t expected_src = get_safetensors_size(ctx->model_dir);
    if (hdr->source_size != expected_src) {
        if (qwen_tts_verbose >= 1)
            fprintf(stderr, "qcache: source size mismatch (cache=%llu, actual=%llu), re-quantizing\n",
                    (unsigned long long)hdr->source_size, (unsigned long long)expected_src);
        munmap(mapped, file_size);
        return -1;
    }

    /* Validate total file size */
    size_t tk_per_layer = (size_t)hdr->tk_wqkv_q4k_bytes + hdr->tk_gate_up_q4k_bytes +
                          hdr->tk_wo_q4k_bytes + hdr->tk_down_q4k_bytes;
    size_t st_per_layer = (size_t)hdr->st_wqkv_q4k_bytes + hdr->st_gate_up_q4k_bytes +
                          hdr->st_wo_q4k_bytes + hdr->st_down_q4k_bytes;
    size_t expected_size = sizeof(qcache_header_t) +
                           tk_per_layer * hdr->n_talker_layers +
                           st_per_layer * hdr->n_subtalker_layers;
    if (file_size < expected_size) {
        munmap(mapped, file_size);
        return -1;
    }

    /* Copy weights from mmap into malloc'd buffers */
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
        CACHE_COPY(l->wqkv_q4k, block_q4_K *, hdr->tk_wqkv_q4k_bytes);
        CACHE_COPY(l->gate_up_q4k, block_q4_K *, hdr->tk_gate_up_q4k_bytes);
        CACHE_COPY(l->wo_q4k, block_q4_K *, hdr->tk_wo_q4k_bytes);
        CACHE_COPY(l->down_q4k, block_q4_K *, hdr->tk_down_q4k_bytes);
    }

    for (int i = 0; i < cfg->subtalker_layers; i++) {
        qwen_tts_subtalker_layer_t *l = &ctx->subtalker.layers[i];
        CACHE_COPY(l->wqkv_q4k, block_q4_K *, hdr->st_wqkv_q4k_bytes);
        CACHE_COPY(l->gate_up_q4k, block_q4_K *, hdr->st_gate_up_q4k_bytes);
        CACHE_COPY(l->wo_q4k, block_q4_K *, hdr->st_wo_q4k_bytes);
        CACHE_COPY(l->down_q4k, block_q4_K *, hdr->st_down_q4k_bytes);
    }

    #undef CACHE_COPY

    munmap(mapped, file_size);

    if (qwen_tts_verbose >= 1)
        fprintf(stderr, "Loaded quantized cache from %s\n", path);
    return 0;
}

#else /* __EMSCRIPTEN__ */

int save_quantized_cache(struct qwen_tts_ctx *ctx) { (void)ctx; return -1; }
int load_quantized_cache(struct qwen_tts_ctx *ctx) { (void)ctx; return -1; }

#endif /* __EMSCRIPTEN__ */
