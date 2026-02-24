/*
 * qwen_asr_gguf.h - Minimal GGUF v3 reader (mmap-based, no ggml dependency)
 */

#ifndef QWEN_ASR_GGUF_H
#define QWEN_ASR_GGUF_H

#include <stddef.h>
#include <stdint.h>

/* GGML tensor types we care about */
#define GGUF_TYPE_F32   0
#define GGUF_TYPE_F16   1
#define GGUF_TYPE_Q8_0  8
#define GGUF_TYPE_Q4_K  12

/* Tensor descriptor */
typedef struct {
    char name[256];
    uint32_t type;       /* GGML type enum */
    uint32_t ndim;
    uint64_t shape[4];
    void *data;          /* pointer into mmap'd region */
    uint64_t nbytes;     /* total data size in bytes */
} gguf_tensor_t;

/* KV metadata entry */
typedef struct {
    char key[256];
    uint32_t type;       /* GGUF value type */
    union {
        uint8_t  u8;
        int8_t   i8;
        uint16_t u16;
        int16_t  i16;
        uint32_t u32;
        int32_t  i32;
        float    f32;
        uint64_t u64;
        int64_t  i64;
        double   f64;
        struct { uint64_t len; char *str; } string;
    } value;
} gguf_kv_t;

/* GGUF context */
typedef struct {
    void *mmap_base;
    size_t mmap_size;

    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;

    gguf_kv_t *kvs;        /* [n_kv] */
    gguf_tensor_t *tensors; /* [n_tensors] */
} gguf_ctx_t;

/* Open a GGUF file (mmap + parse header + tensor index) */
gguf_ctx_t *gguf_open(const char *path);

/* Find a tensor by name. Returns NULL if not found. */
gguf_tensor_t *gguf_find(gguf_ctx_t *ctx, const char *name);

/* Read KV metadata values. Returns 0 on success, -1 if not found. */
int gguf_get_u32(gguf_ctx_t *ctx, const char *key, uint32_t *out);
int gguf_get_i32(gguf_ctx_t *ctx, const char *key, int32_t *out);
int gguf_get_f32(gguf_ctx_t *ctx, const char *key, float *out);
int gguf_get_string(gguf_ctx_t *ctx, const char *key, const char **out);

/* Close and munmap */
void gguf_close(gguf_ctx_t *ctx);

#endif /* QWEN_ASR_GGUF_H */
