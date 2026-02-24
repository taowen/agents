/*
 * qwen_asr_gguf.c - Minimal GGUF v3 reader (mmap-based, no ggml dependency)
 *
 * GGUF v3 file layout:
 *   [4B magic "GGUF"] [4B version=3]
 *   [8B n_tensors] [8B n_kv]
 *   [KV pairs...]
 *   [Tensor infos...]
 *   [alignment padding to GGUF_DEFAULT_ALIGNMENT]
 *   [Tensor data blob...]
 */

#include "qwen_asr_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

#define GGUF_MAGIC      0x46554747  /* "GGUF" in little-endian */
#define GGUF_VERSION    3
#define GGUF_ALIGNMENT  32

/* GGUF value types */
enum {
    GGUF_VAL_UINT8   = 0,
    GGUF_VAL_INT8    = 1,
    GGUF_VAL_UINT16  = 2,
    GGUF_VAL_INT16   = 3,
    GGUF_VAL_UINT32  = 4,
    GGUF_VAL_INT32   = 5,
    GGUF_VAL_FLOAT32 = 6,
    GGUF_VAL_BOOL    = 7,
    GGUF_VAL_STRING  = 8,
    GGUF_VAL_ARRAY   = 9,
    GGUF_VAL_UINT64  = 10,
    GGUF_VAL_INT64   = 11,
    GGUF_VAL_FLOAT64 = 12,
};

/* GGML type → block size / type size lookup */
static size_t ggml_type_size(uint32_t type) {
    switch (type) {
        case 0:  return 4;     /* F32 */
        case 1:  return 2;     /* F16 */
        case 8:  return 36;    /* Q8_0: 4 + 32 = 36 bytes per 32 elements */
        case 12: return 144;   /* Q4_K: 2+2+12+128 = 144 bytes per 256 elements */
        default: return 0;
    }
}

static int ggml_type_block_size(uint32_t type) {
    switch (type) {
        case 0:  return 1;     /* F32 */
        case 1:  return 1;     /* F16 */
        case 8:  return 32;    /* Q8_0 */
        case 12: return 256;   /* Q4_K */
        default: return 0;
    }
}

/* Cursor-based reader helpers */
typedef struct {
    const uint8_t *ptr;
    const uint8_t *end;
} cursor_t;

static int cursor_ok(cursor_t *c, size_t n) {
    return c->ptr + n <= c->end;
}

static uint8_t read_u8(cursor_t *c) {
    uint8_t v = *c->ptr;
    c->ptr += 1;
    return v;
}

static uint32_t read_u32(cursor_t *c) {
    uint32_t v;
    memcpy(&v, c->ptr, 4);
    c->ptr += 4;
    return v;
}

static uint64_t read_u64(cursor_t *c) {
    uint64_t v;
    memcpy(&v, c->ptr, 8);
    c->ptr += 8;
    return v;
}

static float read_f32(cursor_t *c) {
    float v;
    memcpy(&v, c->ptr, 4);
    c->ptr += 4;
    return v;
}

static double read_f64(cursor_t *c) {
    double v;
    memcpy(&v, c->ptr, 8);
    c->ptr += 8;
    return v;
}

/* Read a GGUF string: [uint64 len] [char data...] (no null terminator in file) */
static int read_string(cursor_t *c, char *buf, size_t buf_size) {
    if (!cursor_ok(c, 8)) return -1;
    uint64_t len = read_u64(c);
    if (!cursor_ok(c, (size_t)len)) return -1;
    size_t copy = len < buf_size - 1 ? (size_t)len : buf_size - 1;
    memcpy(buf, c->ptr, copy);
    buf[copy] = '\0';
    c->ptr += len;
    return 0;
}

/* Skip a GGUF value (for array elements or values we don't care about) */
static int skip_value(cursor_t *c, uint32_t type) {
    switch (type) {
        case GGUF_VAL_UINT8:
        case GGUF_VAL_INT8:
        case GGUF_VAL_BOOL:
            if (!cursor_ok(c, 1)) return -1;
            c->ptr += 1;
            return 0;
        case GGUF_VAL_UINT16:
        case GGUF_VAL_INT16:
            if (!cursor_ok(c, 2)) return -1;
            c->ptr += 2;
            return 0;
        case GGUF_VAL_UINT32:
        case GGUF_VAL_INT32:
        case GGUF_VAL_FLOAT32:
            if (!cursor_ok(c, 4)) return -1;
            c->ptr += 4;
            return 0;
        case GGUF_VAL_UINT64:
        case GGUF_VAL_INT64:
        case GGUF_VAL_FLOAT64:
            if (!cursor_ok(c, 8)) return -1;
            c->ptr += 8;
            return 0;
        case GGUF_VAL_STRING: {
            if (!cursor_ok(c, 8)) return -1;
            uint64_t len = read_u64(c);
            if (!cursor_ok(c, (size_t)len)) return -1;
            c->ptr += len;
            return 0;
        }
        case GGUF_VAL_ARRAY: {
            if (!cursor_ok(c, 12)) return -1;
            uint32_t arr_type = read_u32(c);
            uint64_t arr_count = read_u64(c);
            for (uint64_t i = 0; i < arr_count; i++) {
                if (skip_value(c, arr_type) != 0) return -1;
            }
            return 0;
        }
        default:
            return -1;
    }
}

gguf_ctx_t *gguf_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "gguf_open: cannot open %s\n", path);
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return NULL;
    }
    size_t file_size = (size_t)st.st_size;

    void *map = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) {
        fprintf(stderr, "gguf_open: mmap failed for %s\n", path);
        return NULL;
    }

    cursor_t c = { .ptr = (const uint8_t *)map, .end = (const uint8_t *)map + file_size };

    /* Header: magic + version */
    if (!cursor_ok(&c, 16)) goto fail;
    uint32_t magic = read_u32(&c);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "gguf_open: bad magic 0x%08X (expected 0x%08X)\n", magic, GGUF_MAGIC);
        goto fail;
    }
    uint32_t version = read_u32(&c);
    if (version < 2 || version > 3) {
        fprintf(stderr, "gguf_open: unsupported version %u\n", version);
        goto fail;
    }

    uint64_t n_tensors = read_u64(&c);
    uint64_t n_kv = read_u64(&c);

    gguf_ctx_t *ctx = (gguf_ctx_t *)calloc(1, sizeof(gguf_ctx_t));
    ctx->mmap_base = map;
    ctx->mmap_size = file_size;
    ctx->version = version;
    ctx->n_tensors = n_tensors;
    ctx->n_kv = n_kv;

    /* Parse KV pairs */
    ctx->kvs = (gguf_kv_t *)calloc((size_t)n_kv, sizeof(gguf_kv_t));
    for (uint64_t i = 0; i < n_kv; i++) {
        gguf_kv_t *kv = &ctx->kvs[i];
        if (read_string(&c, kv->key, sizeof(kv->key)) != 0) goto fail_ctx;
        if (!cursor_ok(&c, 4)) goto fail_ctx;
        kv->type = read_u32(&c);

        switch (kv->type) {
            case GGUF_VAL_UINT8:   if (!cursor_ok(&c, 1)) goto fail_ctx; kv->value.u8 = read_u8(&c); break;
            case GGUF_VAL_INT8:    if (!cursor_ok(&c, 1)) goto fail_ctx; kv->value.i8 = (int8_t)read_u8(&c); break;
            case GGUF_VAL_UINT16:  if (!cursor_ok(&c, 2)) goto fail_ctx; memcpy(&kv->value.u16, c.ptr, 2); c.ptr += 2; break;
            case GGUF_VAL_INT16:   if (!cursor_ok(&c, 2)) goto fail_ctx; memcpy(&kv->value.i16, c.ptr, 2); c.ptr += 2; break;
            case GGUF_VAL_UINT32:  if (!cursor_ok(&c, 4)) goto fail_ctx; kv->value.u32 = read_u32(&c); break;
            case GGUF_VAL_INT32:   if (!cursor_ok(&c, 4)) goto fail_ctx; memcpy(&kv->value.i32, c.ptr, 4); c.ptr += 4; break;
            case GGUF_VAL_FLOAT32: if (!cursor_ok(&c, 4)) goto fail_ctx; kv->value.f32 = read_f32(&c); break;
            case GGUF_VAL_BOOL:    if (!cursor_ok(&c, 1)) goto fail_ctx; kv->value.u8 = read_u8(&c); break;
            case GGUF_VAL_UINT64:  if (!cursor_ok(&c, 8)) goto fail_ctx; kv->value.u64 = read_u64(&c); break;
            case GGUF_VAL_INT64:   if (!cursor_ok(&c, 8)) goto fail_ctx; memcpy(&kv->value.i64, c.ptr, 8); c.ptr += 8; break;
            case GGUF_VAL_FLOAT64: if (!cursor_ok(&c, 8)) goto fail_ctx; kv->value.f64 = read_f64(&c); break;
            case GGUF_VAL_STRING: {
                if (!cursor_ok(&c, 8)) goto fail_ctx;
                uint64_t slen = read_u64(&c);
                if (!cursor_ok(&c, (size_t)slen)) goto fail_ctx;
                /* Point directly into mmap for string */
                kv->value.string.len = slen;
                kv->value.string.str = (char *)c.ptr;
                c.ptr += slen;
                break;
            }
            case GGUF_VAL_ARRAY: {
                /* Skip arrays: read element type + count, then skip each element */
                if (!cursor_ok(&c, 12)) goto fail_ctx;
                uint32_t arr_type = read_u32(&c);
                uint64_t arr_count = read_u64(&c);
                for (uint64_t ai = 0; ai < arr_count; ai++) {
                    if (skip_value(&c, arr_type) != 0) goto fail_ctx;
                }
                break;
            }
            default:
                fprintf(stderr, "gguf_open: unknown KV type %u for key '%s'\n", kv->type, kv->key);
                goto fail_ctx;
        }
    }

    /* Parse tensor infos */
    ctx->tensors = (gguf_tensor_t *)calloc((size_t)n_tensors, sizeof(gguf_tensor_t));
    for (uint64_t i = 0; i < n_tensors; i++) {
        gguf_tensor_t *t = &ctx->tensors[i];
        if (read_string(&c, t->name, sizeof(t->name)) != 0) goto fail_ctx;
        if (!cursor_ok(&c, 4)) goto fail_ctx;
        t->ndim = read_u32(&c);
        if (t->ndim > 4) goto fail_ctx;
        if (!cursor_ok(&c, t->ndim * 8)) goto fail_ctx;
        for (uint32_t d = 0; d < t->ndim; d++) {
            t->shape[d] = read_u64(&c);
        }
        if (!cursor_ok(&c, 4 + 8)) goto fail_ctx;
        t->type = read_u32(&c);
        uint64_t offset = read_u64(&c);

        /* Compute data size */
        uint64_t n_elems = 1;
        for (uint32_t d = 0; d < t->ndim; d++) n_elems *= t->shape[d];

        int block_size = ggml_type_block_size(t->type);
        size_t type_sz = ggml_type_size(t->type);
        if (block_size == 0 || type_sz == 0) {
            fprintf(stderr, "gguf_open: unsupported tensor type %u for '%s'\n", t->type, t->name);
            goto fail_ctx;
        }

        t->nbytes = (n_elems / block_size) * type_sz;

        /* Store relative offset; will resolve after alignment */
        t->data = (void *)(uintptr_t)offset;
    }

    /* Resolve data pointers: data blob starts after header, aligned */
    size_t header_end = (size_t)(c.ptr - (const uint8_t *)map);
    size_t data_start = (header_end + GGUF_ALIGNMENT - 1) & ~(size_t)(GGUF_ALIGNMENT - 1);

    for (uint64_t i = 0; i < n_tensors; i++) {
        uint64_t offset = (uint64_t)(uintptr_t)ctx->tensors[i].data;
        ctx->tensors[i].data = (uint8_t *)map + data_start + offset;

        /* Bounds check */
        if ((uint8_t *)ctx->tensors[i].data + ctx->tensors[i].nbytes > (uint8_t *)map + file_size) {
            fprintf(stderr, "gguf_open: tensor '%s' data out of bounds\n", ctx->tensors[i].name);
            goto fail_ctx;
        }
    }

    return ctx;

fail_ctx:
    free(ctx->kvs);
    free(ctx->tensors);
    free(ctx);
fail:
    munmap(map, file_size);
    return NULL;
}

gguf_tensor_t *gguf_find(gguf_ctx_t *ctx, const char *name) {
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0) {
            return &ctx->tensors[i];
        }
    }
    return NULL;
}

int gguf_get_u32(gguf_ctx_t *ctx, const char *key, uint32_t *out) {
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        if (strcmp(ctx->kvs[i].key, key) == 0 && ctx->kvs[i].type == GGUF_VAL_UINT32) {
            *out = ctx->kvs[i].value.u32;
            return 0;
        }
    }
    return -1;
}

int gguf_get_i32(gguf_ctx_t *ctx, const char *key, int32_t *out) {
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        if (strcmp(ctx->kvs[i].key, key) == 0) {
            if (ctx->kvs[i].type == GGUF_VAL_INT32) {
                *out = ctx->kvs[i].value.i32;
                return 0;
            }
            if (ctx->kvs[i].type == GGUF_VAL_UINT32) {
                *out = (int32_t)ctx->kvs[i].value.u32;
                return 0;
            }
        }
    }
    return -1;
}

int gguf_get_f32(gguf_ctx_t *ctx, const char *key, float *out) {
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        if (strcmp(ctx->kvs[i].key, key) == 0 && ctx->kvs[i].type == GGUF_VAL_FLOAT32) {
            *out = ctx->kvs[i].value.f32;
            return 0;
        }
    }
    return -1;
}

int gguf_get_string(gguf_ctx_t *ctx, const char *key, const char **out) {
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        if (strcmp(ctx->kvs[i].key, key) == 0 && ctx->kvs[i].type == GGUF_VAL_STRING) {
            *out = ctx->kvs[i].value.string.str;
            return 0;
        }
    }
    return -1;
}

void gguf_close(gguf_ctx_t *ctx) {
    if (!ctx) return;
    if (ctx->mmap_base) {
        munmap(ctx->mmap_base, ctx->mmap_size);
    }
    free(ctx->kvs);
    free(ctx->tensors);
    free(ctx);
}
