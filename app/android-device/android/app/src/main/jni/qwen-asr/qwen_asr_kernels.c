/*
 * qwen_asr_kernels.c - Math kernels for Qwen3-ASR inference
 * Adapted from voxtral-realtime project.
 */

#include "qwen_asr_kernels.h"
#include "qwen_asr_kernels_impl.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ========================================================================
 * Thread Pool
 * ======================================================================== */

#define QWEN_MAX_THREADS 16

static struct {
    pthread_t threads[QWEN_MAX_THREADS - 1];
    int tids[QWEN_MAX_THREADS - 1];
    int n_threads;
    int shutdown;

    parallel_fn_t fn;
    void *arg;
    int generation;

    pthread_mutex_t mutex;
    pthread_cond_t cond_work;
    pthread_cond_t cond_done;
    int n_done;
} tp = {
    .n_threads = 1,
    .shutdown = 0,
    .generation = 0,
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .cond_work = PTHREAD_COND_INITIALIZER,
    .cond_done = PTHREAD_COND_INITIALIZER,
};

static void *worker_loop(void *arg) {
    int tid = *(int *)arg;
    int my_gen = 0;

    for (;;) {
        pthread_mutex_lock(&tp.mutex);
        while (tp.generation == my_gen && !tp.shutdown)
            pthread_cond_wait(&tp.cond_work, &tp.mutex);
        if (tp.shutdown) {
            pthread_mutex_unlock(&tp.mutex);
            return NULL;
        }
        my_gen = tp.generation;
        parallel_fn_t fn = tp.fn;
        void *a = tp.arg;
        int nt = tp.n_threads;
        pthread_mutex_unlock(&tp.mutex);

        fn(tid, nt, a);

        pthread_mutex_lock(&tp.mutex);
        if (++tp.n_done >= tp.n_threads - 1)
            pthread_cond_signal(&tp.cond_done);
        pthread_mutex_unlock(&tp.mutex);
    }
}

void qwen_set_threads(int n) {
    if (n < 1) n = 1;
    if (n > QWEN_MAX_THREADS) n = QWEN_MAX_THREADS;

    /* Shutdown existing workers */
    if (tp.n_threads > 1) {
        pthread_mutex_lock(&tp.mutex);
        tp.shutdown = 1;
        pthread_cond_broadcast(&tp.cond_work);
        pthread_mutex_unlock(&tp.mutex);
        for (int i = 0; i < tp.n_threads - 1; i++)
            pthread_join(tp.threads[i], NULL);
        tp.shutdown = 0;
        tp.generation = 0;
    }

    tp.n_threads = n;
    if (n <= 1) return;

    for (int i = 0; i < n - 1; i++) {
        tp.tids[i] = i + 1;
        pthread_create(&tp.threads[i], NULL, worker_loop, &tp.tids[i]);
    }

    if (qwen_verbose >= 2)
        fprintf(stderr, "Thread pool: %d threads\n", n);
}

int qwen_get_num_cpus(void) {
    int n = (int)sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? n : 1;
}

int qwen_get_n_threads(void) { return tp.n_threads; }

/* Dispatch work to all threads; main thread is tid=0 */
void qwen_parallel_for(parallel_fn_t fn, void *arg) {
    if (tp.n_threads <= 1) {
        fn(0, 1, arg);
        return;
    }

    pthread_mutex_lock(&tp.mutex);
    tp.fn = fn;
    tp.arg = arg;
    tp.n_done = 0;
    tp.generation++;
    pthread_cond_broadcast(&tp.cond_work);
    pthread_mutex_unlock(&tp.mutex);

    fn(0, tp.n_threads, arg);

    pthread_mutex_lock(&tp.mutex);
    while (tp.n_done < tp.n_threads - 1)
        pthread_cond_wait(&tp.cond_done, &tp.mutex);
    pthread_mutex_unlock(&tp.mutex);
}

/* ========================================================================
 * Basic Element-wise Operations
 * ======================================================================== */

void qwen_add_inplace(float *a, const float *b, int n) {
#ifdef __ARM_NEON
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(a + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
        vst1q_f32(a + i + 4, vaddq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4)));
    }
    for (; i < n; i++) a[i] += b[i];
#else
    for (int i = 0; i < n; i++) a[i] += b[i];
#endif
}

/* Transpose Yt[N, M_pad] → Y[M, N] */
static void transpose_back(float *Y, const float *Yt, int M, int N, int M_pad) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            Y[(size_t)m * N + n] = Yt[(size_t)n * M_pad + m];
        }
    }
}

/* ========================================================================
 * Q8_0 Quantized Weight Operations
 * ======================================================================== */

/* Forward declarations */
static void q8_matvec_threaded(float *y, const float *x, const block_q8_0 *W_q8,
                                const float *bias, int in_dim, int out_dim);

/* GEMM workspace: pre-allocated, lazily grown, never shrunk.
 * Eliminates ~650 malloc/free per inference (~325 GEMM calls × 2). */
static struct {
    block_q8_0 *x_q8t;     /* [n_blocks * M_pad] transposed quantized input */
    float *yt;              /* [N * M_pad] transposed output */
    size_t x_q8t_cap;      /* allocated block_q8_0 count */
    size_t yt_cap;          /* allocated float count */
} gemm_ws = {0};

static void gemm_ws_ensure(int n_blocks, int M_pad, int N) {
    size_t need_q8 = (size_t)n_blocks * M_pad;
    if (need_q8 > gemm_ws.x_q8t_cap) {
        free(gemm_ws.x_q8t);
        gemm_ws.x_q8t = (block_q8_0 *)malloc(need_q8 * sizeof(block_q8_0));
        gemm_ws.x_q8t_cap = need_q8;
    }
    size_t need_yt = (size_t)N * M_pad;
    if (need_yt > gemm_ws.yt_cap) {
        free(gemm_ws.yt);
        gemm_ws.yt = (float *)malloc(need_yt * sizeof(float));
        gemm_ws.yt_cap = need_yt;
    }
}

void qwen_gemm_workspace_free(void) {
    free(gemm_ws.x_q8t); gemm_ws.x_q8t = NULL; gemm_ws.x_q8t_cap = 0;
    free(gemm_ws.yt);    gemm_ws.yt = NULL;    gemm_ws.yt_cap = 0;
}

/* Q8_0 INT8 GEMM worker: INT8×INT8 dot products via vdotq_s32.
 * X_q8t: [n_blocks, M_pad] (transposed quantized input)
 * W_q8:  [N, n_blocks] (quantized weights)
 * Yt:    [N, M_pad] (output, pre-initialized with bias) */
typedef struct {
    float *Yt;                  /* [N, M_pad] */
    const block_q8_0 *X_q8t;   /* [n_blocks, M_pad] */
    const block_q8_0 *W_q8;    /* [N, n_blocks] */
    int M_pad, N, n_blocks;
} q8_gemm_task_t;

static void q8_gemm_worker(int tid, int n_threads, void *arg) {
    q8_gemm_task_t *t = (q8_gemm_task_t *)arg;
    int chunk = (t->N + n_threads - 1) / n_threads;
    int n_start = tid * chunk;
    int n_end = n_start + chunk;
    if (n_end > t->N) n_end = t->N;
    if (n_start >= n_end) return;

    int M_pad = t->M_pad;
    int n_blocks = t->n_blocks;

    /* N-tiling: tile the N dimension so Yt[Nc, M_pad] fits in L1D (~32KB).
     * Nc = 32768 / (M_pad * sizeof(float)).
     * Trade-off: x_col is re-read from L2 for each N-tile, but x_col total
     * is small (~140KB) and stays in L2. */
    int Nc = 32768 / (M_pad * (int)sizeof(float));
    if (Nc < 4) Nc = 4;
    if (Nc > (n_end - n_start)) Nc = n_end - n_start;

    /* N-tile outer, K-outer, N-inner, M-inner */
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    for (int n_base = n_start; n_base < n_end; n_base += Nc) {
        int n_tile_end = n_base + Nc;
        if (n_tile_end > n_end) n_tile_end = n_end;

        for (int kb = 0; kb < n_blocks; kb++) {
            const block_q8_0 *x_col = t->X_q8t + (size_t)kb * M_pad;

            for (int n = n_base; n < n_tile_end; n++) {
                float *yt_row = t->Yt + (size_t)n * M_pad;
                const block_q8_0 *wb = &t->W_q8[(size_t)n * n_blocks + kb];
                float w_scale = wb->scale;

                int8x16_t w_lo = vld1q_s8(wb->qs);
                int8x16_t w_hi = vld1q_s8(wb->qs + 16);

                for (int m = 0; m < M_pad; m += 4) {
                    const block_q8_0 *xb0 = &x_col[m];
                    const block_q8_0 *xb1 = &x_col[m + 1];
                    const block_q8_0 *xb2 = &x_col[m + 2];
                    const block_q8_0 *xb3 = &x_col[m + 3];

                    int32x4_t d0 = vdotq_s32(vdupq_n_s32(0), w_lo, vld1q_s8(xb0->qs));
                    d0 = vdotq_s32(d0, w_hi, vld1q_s8(xb0->qs + 16));
                    int32x4_t d1 = vdotq_s32(vdupq_n_s32(0), w_lo, vld1q_s8(xb1->qs));
                    d1 = vdotq_s32(d1, w_hi, vld1q_s8(xb1->qs + 16));
                    int32x4_t d2 = vdotq_s32(vdupq_n_s32(0), w_lo, vld1q_s8(xb2->qs));
                    d2 = vdotq_s32(d2, w_hi, vld1q_s8(xb2->qs + 16));
                    int32x4_t d3 = vdotq_s32(vdupq_n_s32(0), w_lo, vld1q_s8(xb3->qs));
                    d3 = vdotq_s32(d3, w_hi, vld1q_s8(xb3->qs + 16));

                    int32x4_t p01 = vpaddq_s32(d0, d1);
                    int32x4_t p23 = vpaddq_s32(d2, d3);
                    int32x4_t all4 = vpaddq_s32(p01, p23);

                    float32x4_t dots_f = vcvtq_f32_s32(all4);
                    float32x4_t xs = {xb0->scale, xb1->scale, xb2->scale, xb3->scale};
                    float32x4_t scaled = vmulq_f32(vmulq_n_f32(dots_f, w_scale), xs);

                    float32x4_t acc = vld1q_f32(yt_row + m);
                    vst1q_f32(yt_row + m, vaddq_f32(acc, scaled));
                }
            }
        }
    }
#elif defined(__ARM_NEON)
    /* NEON fallback without dotprod: use vmovl_s8 + vmlal_s16 */
    for (int n_base = n_start; n_base < n_end; n_base += Nc) {
        int n_tile_end = n_base + Nc;
        if (n_tile_end > n_end) n_tile_end = n_end;

        for (int kb = 0; kb < n_blocks; kb++) {
            const block_q8_0 *x_col = t->X_q8t + (size_t)kb * M_pad;

            for (int n = n_base; n < n_tile_end; n++) {
                float *yt_row = t->Yt + (size_t)n * M_pad;
                const block_q8_0 *wb = &t->W_q8[(size_t)n * n_blocks + kb];
                float w_scale = wb->scale;

                for (int m = 0; m < M_pad; m += 4) {
                    const block_q8_0 *xb0 = &x_col[m];
                    const block_q8_0 *xb1 = &x_col[m + 1];
                    const block_q8_0 *xb2 = &x_col[m + 2];
                    const block_q8_0 *xb3 = &x_col[m + 3];

                    int32x4_t sum0 = vdupq_n_s32(0), sum1 = vdupq_n_s32(0);
                    int32x4_t sum2 = vdupq_n_s32(0), sum3 = vdupq_n_s32(0);

                    for (int j = 0; j < QK8_0; j += 8) {
                        int8x8_t wq = vld1_s8(wb->qs + j);
                        int16x8_t wq16 = vmovl_s8(wq);
                        int16x4_t wq_lo = vget_low_s16(wq16);
                        int16x4_t wq_hi = vget_high_s16(wq16);

                        int16x8_t x0_16 = vmovl_s8(vld1_s8(xb0->qs + j));
                        sum0 = vmlal_s16(sum0, wq_lo, vget_low_s16(x0_16));
                        sum0 = vmlal_s16(sum0, wq_hi, vget_high_s16(x0_16));

                        int16x8_t x1_16 = vmovl_s8(vld1_s8(xb1->qs + j));
                        sum1 = vmlal_s16(sum1, wq_lo, vget_low_s16(x1_16));
                        sum1 = vmlal_s16(sum1, wq_hi, vget_high_s16(x1_16));

                        int16x8_t x2_16 = vmovl_s8(vld1_s8(xb2->qs + j));
                        sum2 = vmlal_s16(sum2, wq_lo, vget_low_s16(x2_16));
                        sum2 = vmlal_s16(sum2, wq_hi, vget_high_s16(x2_16));

                        int16x8_t x3_16 = vmovl_s8(vld1_s8(xb3->qs + j));
                        sum3 = vmlal_s16(sum3, wq_lo, vget_low_s16(x3_16));
                        sum3 = vmlal_s16(sum3, wq_hi, vget_high_s16(x3_16));
                    }

                    int32x4_t p01 = vpaddq_s32(sum0, sum1);
                    int32x4_t p23 = vpaddq_s32(sum2, sum3);
                    int32x4_t all4 = vpaddq_s32(p01, p23);

                    float32x4_t dots_f = vcvtq_f32_s32(all4);
                    float32x4_t xs = {xb0->scale, xb1->scale, xb2->scale, xb3->scale};
                    float32x4_t scaled = vmulq_f32(vmulq_n_f32(dots_f, w_scale), xs);

                    float32x4_t acc = vld1q_f32(yt_row + m);
                    vst1q_f32(yt_row + m, vaddq_f32(acc, scaled));
                }
            }
        }
    }
#else
    /* Scalar fallback */
    for (int n_base = n_start; n_base < n_end; n_base += Nc) {
        int n_tile_end = n_base + Nc;
        if (n_tile_end > n_end) n_tile_end = n_end;

        for (int kb = 0; kb < n_blocks; kb++) {
            const block_q8_0 *x_col = t->X_q8t + (size_t)kb * M_pad;

            for (int n = n_base; n < n_tile_end; n++) {
                float *yt_row = t->Yt + (size_t)n * M_pad;
                const block_q8_0 *wb = &t->W_q8[(size_t)n * n_blocks + kb];
                float w_scale = wb->scale;

                for (int m = 0; m < M_pad; m++) {
                    const block_q8_0 *xb = &x_col[m];
                    int32_t dot = 0;
                    for (int j = 0; j < QK8_0; j++)
                        dot += (int32_t)wb->qs[j] * (int32_t)xb->qs[j];
                    yt_row[m] += w_scale * xb->scale * (float)dot;
                }
            }
        }
    }
#endif
}

/* Batched Q8_0 GEMM: Y[M,N] = X[M,K] @ W_q8[N,K/32 blocks]^T + bias[N]
 * Uses INT8 dot products: quantizes X to Q8_0, then vdotq_s32 for GEMM. */
static void q8_gemm_batched(float *Y, const float *X, const block_q8_0 *W_q8,
                              const float *bias, int M, int K, int N) {
    int M_pad = (M + 3) & ~3;
    int n_blocks = K / QK8_0;

    gemm_ws_ensure(n_blocks, M_pad, N);
    quantize_f32_rows_transpose_q8(X, gemm_ws.x_q8t, M, K, M_pad);

    /* Initialize Yt with bias (or zero) */
    for (int n = 0; n < N; n++) {
        float b = bias ? bias[n] : 0.0f;
        float *yt_row = gemm_ws.yt + (size_t)n * M_pad;
        for (int m = 0; m < M_pad; m++)
            yt_row[m] = (m < M) ? b : 0.0f;
    }

    q8_gemm_task_t task = { gemm_ws.yt, gemm_ws.x_q8t, W_q8, M_pad, N, n_blocks };
    if (tp.n_threads <= 1) {
        q8_gemm_worker(0, 1, &task);
    } else {
        qwen_parallel_for(q8_gemm_worker, &task);
    }

    transpose_back(Y, gemm_ws.yt, M, N, M_pad);
}

/* Q8_0 Threaded MatVec: quantize x once, then dispatch to threads */
typedef struct {
    float *y;
    const block_q8_0 *x_q8;
    const block_q8_0 *W_q8;
    const float *bias;
    int n_blocks;
    int out_dim;
} q8_matvec_task_t;

static void q8_matvec_fused(float *y, const block_q8_0 *x_q8, const block_q8_0 *W_q8,
                              const float *bias, int n_blocks, int out_dim) {
    qwen_q8_matvec_fused_impl(y, x_q8, W_q8, bias, n_blocks, out_dim);
}

static void q8_matvec_worker(int tid, int n_threads, void *arg) {
    q8_matvec_task_t *t = (q8_matvec_task_t *)arg;
    int chunk = (t->out_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->out_dim) end = t->out_dim;
    if (start >= end) return;

    q8_matvec_fused(t->y + start, t->x_q8,
                     t->W_q8 + (size_t)start * t->n_blocks,
                     t->bias ? t->bias + start : NULL,
                     t->n_blocks, end - start);
}

static void q8_matvec_threaded(float *y, const float *x, const block_q8_0 *W_q8,
                                const float *bias, int in_dim, int out_dim) {
    int n_blocks = in_dim / QK8_0;

    /* Quantize input x to Q8_0 once, reused for all output rows */
    block_q8_0 *x_q8 = (block_q8_0 *)malloc((size_t)n_blocks * sizeof(block_q8_0));
    quantize_f32_to_q8_0(x, x_q8, in_dim);

    if (tp.n_threads <= 1) {
        q8_matvec_fused(y, x_q8, W_q8, bias, n_blocks, out_dim);
    } else {
        q8_matvec_task_t task = { y, x_q8, W_q8, bias, n_blocks, out_dim };
        qwen_parallel_for(q8_matvec_worker, &task);
    }

    free(x_q8);
}

/* Q8_0 QKV fused matvec for single-token decoder */
typedef struct {
    float *q;
    float *k;
    float *v;
    const block_q8_0 *x_q8;
    const block_q8_0 *Wq_q8;
    const block_q8_0 *Wk_q8;
    const block_q8_0 *Wv_q8;
    int n_blocks;
    int q_dim;
    int kv_dim;
    int total_dim;
} q8_qkv_matvec_task_t;

static void q8_qkv_matvec_worker(int tid, int n_threads, void *arg) {
    q8_qkv_matvec_task_t *t = (q8_qkv_matvec_task_t *)arg;
    int chunk = (t->total_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->total_dim) end = t->total_dim;
    if (start >= end) return;

    int q_end = t->q_dim;
    int k_end = q_end + t->kv_dim;
    int v_end = k_end + t->kv_dim;

    if (start < q_end) {
        int s = start;
        int e = end < q_end ? end : q_end;
        if (s < e) {
            q8_matvec_fused(t->q + s, t->x_q8,
                             t->Wq_q8 + (size_t)s * t->n_blocks,
                             NULL, t->n_blocks, e - s);
        }
    }

    if (end > q_end && start < k_end) {
        int s = start > q_end ? start - q_end : 0;
        int e_abs = end < k_end ? end : k_end;
        int e = e_abs - q_end;
        if (s < e) {
            q8_matvec_fused(t->k + s, t->x_q8,
                             t->Wk_q8 + (size_t)s * t->n_blocks,
                             NULL, t->n_blocks, e - s);
        }
    }

    if (end > k_end && start < v_end) {
        int s = start > k_end ? start - k_end : 0;
        int e_abs = end < v_end ? end : v_end;
        int e = e_abs - k_end;
        if (s < e) {
            q8_matvec_fused(t->v + s, t->x_q8,
                             t->Wv_q8 + (size_t)s * t->n_blocks,
                             NULL, t->n_blocks, e - s);
        }
    }
}

void qwen_linear_nobias_q8_qkv(float *q, float *k, float *v, const float *x,
                                const block_q8_0 *Wq_q8,
                                const block_q8_0 *Wk_q8,
                                const block_q8_0 *Wv_q8,
                                int in_dim, int q_dim, int kv_dim) {
    int n_blocks = in_dim / QK8_0;

    /* Quantize x once, shared across Q/K/V */
    block_q8_0 *x_q8 = (block_q8_0 *)malloc((size_t)n_blocks * sizeof(block_q8_0));
    quantize_f32_to_q8_0(x, x_q8, in_dim);

    if (tp.n_threads <= 1) {
        q8_matvec_fused(q, x_q8, Wq_q8, NULL, n_blocks, q_dim);
        q8_matvec_fused(k, x_q8, Wk_q8, NULL, n_blocks, kv_dim);
        q8_matvec_fused(v, x_q8, Wv_q8, NULL, n_blocks, kv_dim);
        free(x_q8);
        return;
    }

    q8_qkv_matvec_task_t task = {
        .q = q,
        .k = k,
        .v = v,
        .x_q8 = x_q8,
        .Wq_q8 = Wq_q8,
        .Wk_q8 = Wk_q8,
        .Wv_q8 = Wv_q8,
        .n_blocks = n_blocks,
        .q_dim = q_dim,
        .kv_dim = kv_dim,
        .total_dim = q_dim + 2 * kv_dim,
    };
    qwen_parallel_for(q8_qkv_matvec_worker, &task);
    free(x_q8);
}

void qwen_linear_nobias_q8(float *y, const float *x, const block_q8_0 *W_q8,
                            int seq_len, int in_dim, int out_dim) {
    if (seq_len > 1) {
        q8_gemm_batched(y, x, W_q8, NULL, seq_len, in_dim, out_dim);
    } else {
        q8_matvec_threaded(y, x, W_q8, NULL, in_dim, out_dim);
    }
}

void qwen_linear_q8(float *y, const float *x, const block_q8_0 *W_q8,
                    const float *b, int seq_len, int in_dim, int out_dim) {
    if (seq_len > 1) {
        q8_gemm_batched(y, x, W_q8, b, seq_len, in_dim, out_dim);
    } else {
        q8_matvec_threaded(y, x, W_q8, b, in_dim, out_dim);
    }
}

/* Fused QKV GEMM: quantize input once, run Q/K/V projections sharing the
 * quantized input. Saves 2× redundant quantizations for encoder/prefill. */
static void q8_gemm_batched_with_q8t(float *Y, const block_q8_0 *X_q8t,
                                       const block_q8_0 *W_q8, const float *bias,
                                       int M, int M_pad, int n_blocks, int N) {
    /* Initialize Yt with bias (or zero) */
    size_t yt_need = (size_t)N * M_pad;
    if (yt_need > gemm_ws.yt_cap) {
        free(gemm_ws.yt);
        gemm_ws.yt = (float *)malloc(yt_need * sizeof(float));
        gemm_ws.yt_cap = yt_need;
    }

    for (int n = 0; n < N; n++) {
        float b = bias ? bias[n] : 0.0f;
        float *yt_row = gemm_ws.yt + (size_t)n * M_pad;
        for (int m = 0; m < M_pad; m++)
            yt_row[m] = (m < M) ? b : 0.0f;
    }

    q8_gemm_task_t task = { gemm_ws.yt, X_q8t, W_q8, M_pad, N, n_blocks };
    if (tp.n_threads <= 1) {
        q8_gemm_worker(0, 1, &task);
    } else {
        qwen_parallel_for(q8_gemm_worker, &task);
    }

    transpose_back(Y, gemm_ws.yt, M, N, M_pad);
}

void qwen_linear_q8_qkv_batched(
    float *q, float *k, float *v,
    const float *x,
    const block_q8_0 *Wq_q8, const float *bq,
    const block_q8_0 *Wk_q8, const float *bk,
    const block_q8_0 *Wv_q8, const float *bv,
    int seq_len, int in_dim, int q_dim, int kv_dim
) {
    /* seq_len=1: use existing matvec QKV path */
    if (seq_len <= 1) {
        if (bq || bk || bv) {
            /* Encoder path with biases: separate matvec calls */
            q8_matvec_threaded(q, x, Wq_q8, bq, in_dim, q_dim);
            q8_matvec_threaded(k, x, Wk_q8, bk, in_dim, kv_dim);
            q8_matvec_threaded(v, x, Wv_q8, bv, in_dim, kv_dim);
        } else {
            qwen_linear_nobias_q8_qkv(q, k, v, x, Wq_q8, Wk_q8, Wv_q8,
                                        in_dim, q_dim, kv_dim);
        }
        return;
    }

    int M_pad = (seq_len + 3) & ~3;
    int n_blocks = in_dim / QK8_0;

    /* Quantize input once */
    gemm_ws_ensure(n_blocks, M_pad, q_dim > kv_dim ? q_dim : kv_dim);
    quantize_f32_rows_transpose_q8(x, gemm_ws.x_q8t, seq_len, in_dim, M_pad);

    /* Run Q/K/V GEMMs sharing the quantized input */
    q8_gemm_batched_with_q8t(q, gemm_ws.x_q8t, Wq_q8, bq, seq_len, M_pad, n_blocks, q_dim);
    q8_gemm_batched_with_q8t(k, gemm_ws.x_q8t, Wk_q8, bk, seq_len, M_pad, n_blocks, kv_dim);
    q8_gemm_batched_with_q8t(v, gemm_ws.x_q8t, Wv_q8, bv, seq_len, M_pad, n_blocks, kv_dim);
}

/* ========================================================================
 * Q4_K Super-Block Weight Operations
 * ======================================================================== */

/* Q4_K threaded matvec: split output rows across threads */
typedef struct {
    float *y;
    const block_q4_k *W_q4k;
    const float *x;
    int in_dim;
    int out_dim;
} q4k_matvec_task_t;

static void q4k_matvec_worker(int tid, int n_threads, void *arg) {
    q4k_matvec_task_t *t = (q4k_matvec_task_t *)arg;
    int chunk = (t->out_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->out_dim) end = t->out_dim;
    if (start >= end) return;

    int blocks_per_row = t->in_dim / QK_K;
    qwen_q4k_matvec_fused_impl(t->y + start,
                                 t->W_q4k + (size_t)start * blocks_per_row,
                                 t->x, end - start, t->in_dim);
}

static void q4k_matvec_threaded(float *y, const float *x, const block_q4_k *W_q4k,
                                  int in_dim, int out_dim) {
    if (tp.n_threads <= 1) {
        qwen_q4k_matvec_fused_impl(y, W_q4k, x, out_dim, in_dim);
        return;
    }
    q4k_matvec_task_t task = { y, W_q4k, x, in_dim, out_dim };
    qwen_parallel_for(q4k_matvec_worker, &task);
}

/* ---- Q4_K Batched GEMM for prefill (seq_len > 1) ----
 * Pre-quantizes all M tokens to int8 once, then processes all tokens
 * in a single threaded dispatch to avoid per-token thread overhead. */

/* Workspace for Q4_K GEMM pre-quantized data */
static struct {
    int8_t  *x_int8;    /* [M_cap * K_cap] */
    float   *x_scales;  /* [M_cap] */
    int32_t *bsums;     /* [M_cap * subs_cap] */
    int M_cap;
    int K_cap;
    int subs_cap;
} q4k_gemm_ws;

static void q4k_gemm_ws_ensure(int M, int K) {
    int total_subs = K / 32;
    if (M <= q4k_gemm_ws.M_cap && K <= q4k_gemm_ws.K_cap) return;
    int new_M = q4k_gemm_ws.M_cap;
    int new_K = q4k_gemm_ws.K_cap;
    if (new_M < M) { new_M = M; if (new_M < 256) new_M = 256; }
    if (new_K < K) { new_K = K; if (new_K < 1024) new_K = 1024; }
    int new_subs = new_K / 32;

    free(q4k_gemm_ws.x_int8);
    free(q4k_gemm_ws.x_scales);
    free(q4k_gemm_ws.bsums);
    q4k_gemm_ws.x_int8  = (int8_t *)malloc((size_t)new_M * new_K);
    q4k_gemm_ws.x_scales = (float *)malloc((size_t)new_M * sizeof(float));
    q4k_gemm_ws.bsums    = (int32_t *)malloc((size_t)new_M * new_subs * sizeof(int32_t));
    q4k_gemm_ws.M_cap = new_M;
    q4k_gemm_ws.K_cap = new_K;
    q4k_gemm_ws.subs_cap = new_subs;
}

/* Quantize x[m] to int8 and compute bsums (scalar, used on non-NEON) */
static void q4k_quantize_x_int8_scalar(const float *x, int cols,
                                         int8_t *x_int8, float *x_scale_out) {
    float x_absmax = 0.0f;
    for (int i = 0; i < cols; i++) {
        float a = x[i] > 0 ? x[i] : -x[i];
        if (a > x_absmax) x_absmax = a;
    }
    *x_scale_out = x_absmax / 127.0f;
    float inv = (x_absmax > 0.0f) ? 127.0f / x_absmax : 0.0f;
    for (int i = 0; i < cols; i++) {
        float v = x[i] * inv;
        int iv = (int)(v + (v > 0 ? 0.5f : -0.5f));
        if (iv > 127) iv = 127;
        if (iv < -128) iv = -128;
        x_int8[i] = (int8_t)iv;
    }
}

#ifdef __ARM_NEON
static void q4k_quantize_x_int8_neon(const float *x, int cols,
                                       int8_t *x_int8, float *x_scale_out) {
    float x_absmax = 0.0f;
    float32x4_t vabsmax = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < cols; i += 4)
        vabsmax = vmaxq_f32(vabsmax, vabsq_f32(vld1q_f32(x + i)));
    x_absmax = vmaxvq_f32(vabsmax);
    for (; i < cols; i++) {
        float a = x[i] > 0 ? x[i] : -x[i];
        if (a > x_absmax) x_absmax = a;
    }
    *x_scale_out = x_absmax / 127.0f;
    float inv = (x_absmax > 0.0f) ? 127.0f / x_absmax : 0.0f;
    float32x4_t vscale = vdupq_n_f32(inv);
    int c = 0;
    for (; c + 7 < cols; c += 8) {
        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + c), vscale));
        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + c + 4), vscale));
        int16x4_t s0 = vqmovn_s32(i0);
        int16x4_t s1 = vqmovn_s32(i1);
        int8x8_t b = vqmovn_s16(vcombine_s16(s0, s1));
        vst1_s8(x_int8 + c, b);
    }
    for (; c < cols; c++) {
        float v = x[c] * inv;
        int iv = (int)(v + (v > 0 ? 0.5f : -0.5f));
        if (iv > 127) iv = 127;
        if (iv < -128) iv = -128;
        x_int8[c] = (int8_t)iv;
    }
}

static void q4k_compute_bsums_neon(const int8_t *x_int8, int cols, int32_t *bsums) {
    int total_subs = cols / 32;
#ifdef __ARM_FEATURE_DOTPROD
    int8x16_t ones = vdupq_n_s8(1);
    for (int s = 0; s < total_subs; s++) {
        const int8_t *xg = x_int8 + s * 32;
        int32x4_t sum4 = vdupq_n_s32(0);
        sum4 = vdotq_s32(sum4, vld1q_s8(xg), ones);
        sum4 = vdotq_s32(sum4, vld1q_s8(xg + 16), ones);
        bsums[s] = vaddvq_s32(sum4);
    }
#else
    for (int s = 0; s < total_subs; s++) {
        int32_t sum = 0;
        const int8_t *xg = x_int8 + s * 32;
        for (int i = 0; i < 32; i++) sum += (int32_t)xg[i];
        bsums[s] = sum;
    }
#endif
}
#endif /* __ARM_NEON */

static void q4k_compute_bsums_scalar(const int8_t *x_int8, int cols, int32_t *bsums) {
    int total_subs = cols / 32;
    for (int s = 0; s < total_subs; s++) {
        int32_t sum = 0;
        const int8_t *xg = x_int8 + s * 32;
        for (int i = 0; i < 32; i++) sum += (int32_t)xg[i];
        bsums[s] = sum;
    }
}

/* Pre-quantize M tokens and compute bsums */
static void q4k_preq_batch(const float *X, int M, int K) {
    int total_subs = K / 32;
    for (int m = 0; m < M; m++) {
        int8_t *xi = q4k_gemm_ws.x_int8 + (size_t)m * K;
#ifdef __ARM_NEON
        q4k_quantize_x_int8_neon(X + (size_t)m * K, K, xi, &q4k_gemm_ws.x_scales[m]);
        q4k_compute_bsums_neon(xi, K, q4k_gemm_ws.bsums + (size_t)m * total_subs);
#else
        q4k_quantize_x_int8_scalar(X + (size_t)m * K, K, xi, &q4k_gemm_ws.x_scales[m]);
        q4k_compute_bsums_scalar(xi, K, q4k_gemm_ws.bsums + (size_t)m * total_subs);
#endif
    }
}

/* GEMM worker: each thread processes a chunk of output rows for ALL tokens */
typedef struct {
    float *Y;                     /* [M, N] row-major output */
    const block_q4_k *W_q4k;     /* [N, K/QK_K blocks] */
    const int8_t *x_int8;        /* [M, K] */
    const float *x_scales;       /* [M] */
    const int32_t *bsums;        /* [M, total_subs] */
    int M, K, N;
    int blocks_per_row;
    int total_subs;
} q4k_gemm_task_t;

static void q4k_gemm_worker(int tid, int n_threads, void *arg) {
    q4k_gemm_task_t *t = (q4k_gemm_task_t *)arg;
    int chunk = (t->N + n_threads - 1) / n_threads;
    int r_start = tid * chunk;
    int r_end = r_start + chunk;
    if (r_end > t->N) r_end = t->N;
    if (r_start >= r_end) return;

    qwen_q4k_gemm_chunk_impl(
        t->Y, t->N,
        t->W_q4k, t->blocks_per_row,
        t->x_int8, t->K,
        t->x_scales,
        t->bsums, t->total_subs,
        t->M, r_start, r_end);
}

static void q4k_gemm_batched(float *Y, const float *X, const block_q4_k *W_q4k,
                                int M, int K, int N) {
    q4k_gemm_ws_ensure(M, K);
    q4k_preq_batch(X, M, K);

    q4k_gemm_task_t task = {
        .Y = Y, .W_q4k = W_q4k,
        .x_int8 = q4k_gemm_ws.x_int8,
        .x_scales = q4k_gemm_ws.x_scales,
        .bsums = q4k_gemm_ws.bsums,
        .M = M, .K = K, .N = N,
        .blocks_per_row = K / QK_K,
        .total_subs = K / 32,
    };

    if (tp.n_threads <= 1) {
        q4k_gemm_worker(0, 1, &task);
    } else {
        qwen_parallel_for(q4k_gemm_worker, &task);
    }
}

void qwen_linear_nobias_q4k(float *y, const float *x, const block_q4_k *W_q4k,
                              int seq_len, int in_dim, int out_dim) {
    if (seq_len <= 1) {
        q4k_matvec_threaded(y, x, W_q4k, in_dim, out_dim);
    } else {
        q4k_gemm_batched(y, x, W_q4k, seq_len, in_dim, out_dim);
    }
}

/* Q4_K QKV fused matvec for single-token decoder */
typedef struct {
    float *q;
    float *k;
    float *v;
    const block_q4_k *Wq_q4k;
    const block_q4_k *Wk_q4k;
    const block_q4_k *Wv_q4k;
    const float *x;
    int in_dim;
    int q_dim;
    int kv_dim;
    int total_dim;
} q4k_qkv_matvec_task_t;

static void q4k_qkv_matvec_worker(int tid, int n_threads, void *arg) {
    q4k_qkv_matvec_task_t *t = (q4k_qkv_matvec_task_t *)arg;
    int chunk = (t->total_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->total_dim) end = t->total_dim;
    if (start >= end) return;

    int blocks_per_row = t->in_dim / QK_K;
    int q_end = t->q_dim;
    int k_end = q_end + t->kv_dim;
    int v_end = k_end + t->kv_dim;

    if (start < q_end) {
        int s = start;
        int e = end < q_end ? end : q_end;
        if (s < e) {
            qwen_q4k_matvec_fused_impl(t->q + s,
                                         t->Wq_q4k + (size_t)s * blocks_per_row,
                                         t->x, e - s, t->in_dim);
        }
    }

    if (end > q_end && start < k_end) {
        int s = start > q_end ? start - q_end : 0;
        int e_abs = end < k_end ? end : k_end;
        int e = e_abs - q_end;
        if (s < e) {
            qwen_q4k_matvec_fused_impl(t->k + s,
                                         t->Wk_q4k + (size_t)s * blocks_per_row,
                                         t->x, e - s, t->in_dim);
        }
    }

    if (end > k_end && start < v_end) {
        int s = start > k_end ? start - k_end : 0;
        int e_abs = end < v_end ? end : v_end;
        int e = e_abs - k_end;
        if (s < e) {
            qwen_q4k_matvec_fused_impl(t->v + s,
                                         t->Wv_q4k + (size_t)s * blocks_per_row,
                                         t->x, e - s, t->in_dim);
        }
    }
}

void qwen_linear_nobias_q4k_qkv(float *q, float *k, float *v, const float *x,
                                  const block_q4_k *Wq_q4k,
                                  const block_q4_k *Wk_q4k,
                                  const block_q4_k *Wv_q4k,
                                  int in_dim, int q_dim, int kv_dim) {
    if (tp.n_threads <= 1) {
        qwen_q4k_matvec_fused_impl(q, Wq_q4k, x, q_dim, in_dim);
        qwen_q4k_matvec_fused_impl(k, Wk_q4k, x, kv_dim, in_dim);
        qwen_q4k_matvec_fused_impl(v, Wv_q4k, x, kv_dim, in_dim);
        return;
    }

    q4k_qkv_matvec_task_t task = {
        .q = q, .k = k, .v = v,
        .Wq_q4k = Wq_q4k, .Wk_q4k = Wk_q4k, .Wv_q4k = Wv_q4k,
        .x = x,
        .in_dim = in_dim, .q_dim = q_dim, .kv_dim = kv_dim,
        .total_dim = q_dim + 2 * kv_dim,
    };
    qwen_parallel_for(q4k_qkv_matvec_worker, &task);
}

/* Q4_K argmax */
typedef struct {
    const block_q4_k *W_q4k;
    const float *x;
    int in_dim;
    int out_dim;
    int best_idx[QWEN_MAX_THREADS];
    float best_val[QWEN_MAX_THREADS];
} q4k_argmax_task_t;

static void q4k_argmax_worker(int tid, int n_threads, void *arg) {
    q4k_argmax_task_t *t = (q4k_argmax_task_t *)arg;
    int chunk = (t->out_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->out_dim) end = t->out_dim;
    if (start >= end) {
        t->best_val[tid] = -1e30f;
        t->best_idx[tid] = 0;
        return;
    }
    qwen_q4k_argmax_range_impl(t->W_q4k, t->x, t->in_dim, start, end,
                                &t->best_idx[tid], &t->best_val[tid]);
}

int qwen_argmax_matvec_q4k(const float *x, const block_q4_k *W_q4k,
                             int in_dim, int out_dim) {
    if (tp.n_threads <= 1) {
        int best;
        float best_val;
        qwen_q4k_argmax_range_impl(W_q4k, x, in_dim, 0, out_dim, &best, &best_val);
        return best;
    }

    q4k_argmax_task_t task;
    task.W_q4k = W_q4k;
    task.x = x;
    task.in_dim = in_dim;
    task.out_dim = out_dim;
    qwen_parallel_for(q4k_argmax_worker, &task);

    int best = task.best_idx[0];
    float best_val = task.best_val[0];
    for (int i = 1; i < tp.n_threads; i++) {
        if (task.best_val[i] > best_val) {
            best_val = task.best_val[i];
            best = task.best_idx[i];
        }
    }
    return best;
}

/* ========================================================================
 * 2D Convolution (im2col + BLAS sgemm)
 * ======================================================================== */

/*
 * im2col: Unroll input patches into a column matrix for GEMM-based conv2d.
 * Input: [C_in, H_in, W_in]
 * Output columns: [C_in * kH * kW, H_out * W_out]
 */
static void im2col(const float *in, float *cols,
                   int c_in, int h_in, int w_in,
                   int kh, int kw, int stride, int padding,
                   int h_out, int w_out) {
    int col_len = h_out * w_out;
    for (int ic = 0; ic < c_in; ic++) {
        for (int ki = 0; ki < kh; ki++) {
            for (int kj = 0; kj < kw; kj++) {
                int col_row = (ic * kh + ki) * kw + kj;
                float *col_ptr = cols + (size_t)col_row * col_len;
                for (int oh = 0; oh < h_out; oh++) {
                    int ih = oh * stride - padding + ki;
                    for (int ow = 0; ow < w_out; ow++) {
                        int iw = ow * stride - padding + kj;
                        if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
                            col_ptr[oh * w_out + ow] = in[ic * h_in * w_in + ih * w_in + iw];
                        } else {
                            col_ptr[oh * w_out + ow] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

typedef struct {
    float *out;
    const float *weight;
    const float *cols;
    const float *bias;
    int c_out;
    int patch_size;
    int spatial_out;
} conv2d_gemm_task_t;

static void conv2d_gemm_worker(int tid, int n_threads, void *arg) {
    conv2d_gemm_task_t *t = (conv2d_gemm_task_t *)arg;
    int chunk = (t->c_out + n_threads - 1) / n_threads;
    int oc_start = tid * chunk;
    int oc_end = oc_start + chunk;
    if (oc_end > t->c_out) oc_end = t->c_out;
    if (oc_start >= oc_end) return;

    int patch_size = t->patch_size;
    int spatial_out = t->spatial_out;
    const float *cols = t->cols;
    const float *weight = t->weight;

    /* Initialize all output channels with bias */
    for (int oc = oc_start; oc < oc_end; oc++) {
        float *out_row = t->out + (size_t)oc * spatial_out;
        float b = t->bias ? t->bias[oc] : 0.0f;
        int s = 0;
#ifdef __ARM_NEON
        float32x4_t bv = vdupq_n_f32(b);
        for (; s + 3 < spatial_out; s += 4)
            vst1q_f32(out_row + s, bv);
#endif
        for (; s < spatial_out; s++)
            out_row[s] = b;
    }

    /* p-outer, oc-inner: each cols group is loaded from DRAM once and reused
     * for all output channels.  cols total (e.g. 13.8MB) is read once instead
     * of c_out times (480×13.8MB = 6.6GB).  Each thread's output block
     * (~384KB) stays in L2. */
#ifdef __ARM_NEON
    int p = 0;
    for (; p + 7 < patch_size; p += 8) {
        const float *cr0 = cols + (size_t)(p + 0) * spatial_out;
        const float *cr1 = cols + (size_t)(p + 1) * spatial_out;
        const float *cr2 = cols + (size_t)(p + 2) * spatial_out;
        const float *cr3 = cols + (size_t)(p + 3) * spatial_out;
        const float *cr4 = cols + (size_t)(p + 4) * spatial_out;
        const float *cr5 = cols + (size_t)(p + 5) * spatial_out;
        const float *cr6 = cols + (size_t)(p + 6) * spatial_out;
        const float *cr7 = cols + (size_t)(p + 7) * spatial_out;

        for (int oc = oc_start; oc < oc_end; oc++) {
            const float *w_row = weight + (size_t)oc * patch_size;
            float *out_row = t->out + (size_t)oc * spatial_out;
            float32x4_t wv0 = vld1q_f32(w_row + p);
            float32x4_t wv1 = vld1q_f32(w_row + p + 4);

            int s = 0;
            for (; s + 3 < spatial_out; s += 4) {
                float32x4_t acc = vld1q_f32(out_row + s);
                acc = vfmaq_laneq_f32(acc, vld1q_f32(cr0 + s), wv0, 0);
                acc = vfmaq_laneq_f32(acc, vld1q_f32(cr1 + s), wv0, 1);
                acc = vfmaq_laneq_f32(acc, vld1q_f32(cr2 + s), wv0, 2);
                acc = vfmaq_laneq_f32(acc, vld1q_f32(cr3 + s), wv0, 3);
                acc = vfmaq_laneq_f32(acc, vld1q_f32(cr4 + s), wv1, 0);
                acc = vfmaq_laneq_f32(acc, vld1q_f32(cr5 + s), wv1, 1);
                acc = vfmaq_laneq_f32(acc, vld1q_f32(cr6 + s), wv1, 2);
                acc = vfmaq_laneq_f32(acc, vld1q_f32(cr7 + s), wv1, 3);
                vst1q_f32(out_row + s, acc);
            }
            for (; s < spatial_out; s++) {
                out_row[s] += w_row[p+0] * cr0[s] + w_row[p+1] * cr1[s]
                           +  w_row[p+2] * cr2[s] + w_row[p+3] * cr3[s]
                           +  w_row[p+4] * cr4[s] + w_row[p+5] * cr5[s]
                           +  w_row[p+6] * cr6[s] + w_row[p+7] * cr7[s];
            }
        }
    }
    /* Tail: remaining p values */
    for (; p < patch_size; p++) {
        const float *cr = cols + (size_t)p * spatial_out;
        for (int oc = oc_start; oc < oc_end; oc++) {
            float wp = weight[(size_t)oc * patch_size + p];
            float *out_row = t->out + (size_t)oc * spatial_out;
            int s = 0;
            float32x4_t wpv = vdupq_n_f32(wp);
            for (; s + 3 < spatial_out; s += 4) {
                float32x4_t acc = vld1q_f32(out_row + s);
                acc = vfmaq_f32(acc, vld1q_f32(cr + s), wpv);
                vst1q_f32(out_row + s, acc);
            }
            for (; s < spatial_out; s++)
                out_row[s] += wp * cr[s];
        }
    }
#else
    /* Scalar fallback: p-outer, oc-inner */
    for (int p = 0; p < patch_size; p++) {
        const float *cr = cols + (size_t)p * spatial_out;
        for (int oc = oc_start; oc < oc_end; oc++) {
            float wp = weight[(size_t)oc * patch_size + p];
            float *out_row = t->out + (size_t)oc * spatial_out;
            for (int s = 0; s < spatial_out; s++)
                out_row[s] += wp * cr[s];
        }
    }
#endif
}

void qwen_conv2d(float *out, const float *in, const float *weight, const float *bias,
                 int c_in, int c_out, int h_in, int w_in,
                 int kh, int kw, int stride, int padding) {
    int h_out = (h_in + 2 * padding - kh) / stride + 1;
    int w_out = (w_in + 2 * padding - kw) / stride + 1;
    int patch_size = c_in * kh * kw;
    int spatial_out = h_out * w_out;

    /* im2col: input -> column matrix [patch_size, spatial_out] */
    float *cols = (float *)malloc((size_t)patch_size * spatial_out * sizeof(float));
    im2col(in, cols, c_in, h_in, w_in, kh, kw, stride, padding, h_out, w_out);

    /* GEMM: weight[c_out, patch_size] @ cols[patch_size, spatial_out] = out[c_out, spatial_out] */
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                c_out, spatial_out, patch_size,
                1.0f, weight, patch_size, cols, spatial_out,
                0.0f, out, spatial_out);
    free(cols);
    /* Add bias */
    if (bias) {
        for (int oc = 0; oc < c_out; oc++) {
            float b = bias[oc];
            float *row = out + oc * spatial_out;
            for (int s = 0; s < spatial_out; s++) {
                row[s] += b;
            }
        }
    }
#else
    /* NEON-vectorized GEMM with bias fused in, parallelized over output channels */
    conv2d_gemm_task_t task = {
        .out = out, .weight = weight, .cols = cols, .bias = bias,
        .c_out = c_out, .patch_size = patch_size, .spatial_out = spatial_out
    };
    qwen_parallel_for(conv2d_gemm_worker, &task);
    free(cols);
#endif
}
