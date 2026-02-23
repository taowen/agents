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
#if (defined(__AVX512F__) || defined(__AVX2__)) && (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#endif
#ifdef __APPLE__
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================
 * Thread Pool
 * ======================================================================== */

#define QWEN_MAX_THREADS 16

typedef void (*parallel_fn_t)(int tid, int n_threads, void *arg);

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
#ifdef __APPLE__
    int n = 0;
    size_t len = sizeof(n);
    sysctlbyname("hw.ncpu", &n, &len, NULL, 0);
    return n > 0 ? n : 1;
#else
    int n = (int)sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? n : 1;
#endif
}

/* Dispatch work to all threads; main thread is tid=0 */
static void parallel_for(parallel_fn_t fn, void *arg) {
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

void qwen_mul_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] *= b[i];
}

void qwen_scale(float *x, float s, int n) {
    for (int i = 0; i < n; i++) x[i] *= s;
}

void qwen_copy(float *dst, const float *src, int n) {
    memcpy(dst, src, n * sizeof(float));
}

/* Forward declarations for threaded matvec helpers */
static void f32_matvec_threaded(float *y, const float *x, const float *W,
                                 const float *bias, int in_dim, int out_dim);
static void bf16_matvec_threaded(float *y, const float *x, const uint16_t *W_bf16,
                                  const float *bias, int in_dim, int out_dim);

/* ========================================================================
 * Batched GEMM for seq_len > 1
 *
 * Problem: per-row matvec re-reads the entire weight matrix for each input
 * row, leading to M * |W| bytes of DRAM traffic. For M=143, K=N=1280:
 * 143 * 6.25MB = 894MB, bottlenecked on memory bandwidth.
 *
 * Solution: transpose X to Xt[K,M], then for each W row n, accumulate
 * W[n,k] * Xt[k,:] into Yt[n,:]. Xt fits in L2 (~713KB for M=143, K=1280)
 * so W is read only once (~6.25MB total) — 128× less memory traffic.
 * ======================================================================== */

/* FP32 batched GEMM worker: processes a slice of output rows [n_start, n_end) */
typedef struct {
    float *Yt;        /* [N, M_pad] transposed output */
    const float *Xt;  /* [K, M_pad] transposed input */
    const float *W;   /* [N, K] weight matrix */
    const float *bias; /* [N] or NULL */
    int M_pad;        /* padded M (multiple of 4) */
    int K;
    int N;
} f32_gemm_task_t;

static void f32_gemm_worker(int tid, int n_threads, void *arg) {
    f32_gemm_task_t *t = (f32_gemm_task_t *)arg;
    int chunk = (t->N + n_threads - 1) / n_threads;
    int n_start = tid * chunk;
    int n_end = n_start + chunk;
    if (n_end > t->N) n_end = t->N;
    if (n_start >= n_end) return;

    int M_pad = t->M_pad;
    int K = t->K;

    /* Initialize all Yt rows with bias (or zero) */
    for (int n = n_start; n < n_end; n++) {
        float *yt_row = t->Yt + (size_t)n * M_pad;
        if (t->bias) {
            float b = t->bias[n];
#ifdef __ARM_NEON
            float32x4_t bv = vdupq_n_f32(b);
            for (int m = 0; m < M_pad; m += 4)
                vst1q_f32(yt_row + m, bv);
#else
            for (int m = 0; m < M_pad; m++) yt_row[m] = b;
#endif
        } else {
            memset(yt_row, 0, M_pad * sizeof(float));
        }
    }

    /* K-outer, N-inner: Xt rows loaded once from DRAM, reused for all N.
     * Thread's Yt block stays in L2 (e.g. 768×160×4 = 480KB). */
    int k = 0;
#ifdef __ARM_NEON
    for (; k + 7 < K; k += 8) {
        const float *xt0 = t->Xt + (size_t)k * M_pad;
        const float *xt4 = t->Xt + (size_t)(k + 4) * M_pad;

        for (int n = n_start; n < n_end; n++) {
            float *yt_row = t->Yt + (size_t)n * M_pad;
            const float *w_row = t->W + (size_t)n * K;
            float32x4_t wv0 = vld1q_f32(w_row + k);
            float32x4_t wv1 = vld1q_f32(w_row + k + 4);

            for (int m = 0; m < M_pad; m += 4) {
                float32x4_t yt = vld1q_f32(yt_row + m);
                yt = vfmaq_laneq_f32(yt, vld1q_f32(xt0 + m), wv0, 0);
                yt = vfmaq_laneq_f32(yt, vld1q_f32(xt0 + M_pad + m), wv0, 1);
                yt = vfmaq_laneq_f32(yt, vld1q_f32(xt0 + 2*M_pad + m), wv0, 2);
                yt = vfmaq_laneq_f32(yt, vld1q_f32(xt0 + 3*M_pad + m), wv0, 3);
                yt = vfmaq_laneq_f32(yt, vld1q_f32(xt4 + m), wv1, 0);
                yt = vfmaq_laneq_f32(yt, vld1q_f32(xt4 + M_pad + m), wv1, 1);
                yt = vfmaq_laneq_f32(yt, vld1q_f32(xt4 + 2*M_pad + m), wv1, 2);
                yt = vfmaq_laneq_f32(yt, vld1q_f32(xt4 + 3*M_pad + m), wv1, 3);
                vst1q_f32(yt_row + m, yt);
            }
        }
    }
    for (; k < K; k++) {
        const float *xt_col = t->Xt + (size_t)k * M_pad;
        for (int n = n_start; n < n_end; n++) {
            float *yt_row = t->Yt + (size_t)n * M_pad;
            float32x4_t wv = vdupq_n_f32(t->W[(size_t)n * K + k]);
            for (int m = 0; m < M_pad; m += 4) {
                float32x4_t yt = vld1q_f32(yt_row + m);
                yt = vfmaq_f32(yt, vld1q_f32(xt_col + m), wv);
                vst1q_f32(yt_row + m, yt);
            }
        }
    }
#else
    for (; k < K; k++) {
        const float *xt_col = t->Xt + (size_t)k * M_pad;
        for (int n = n_start; n < n_end; n++) {
            float wk = t->W[(size_t)n * K + k];
            float *yt_row = t->Yt + (size_t)n * M_pad;
            for (int m = 0; m < M_pad; m++)
                yt_row[m] += wk * xt_col[m];
        }
    }
#endif
}

/* BF16 batched GEMM worker */
typedef struct {
    float *Yt;
    const float *Xt;
    const uint16_t *W_bf16;
    const float *bias;
    int M_pad;
    int K;
    int N;
} bf16_gemm_task_t;

static void bf16_gemm_worker(int tid, int n_threads, void *arg) {
    bf16_gemm_task_t *t = (bf16_gemm_task_t *)arg;
    int chunk = (t->N + n_threads - 1) / n_threads;
    int n_start = tid * chunk;
    int n_end = n_start + chunk;
    if (n_end > t->N) n_end = t->N;
    if (n_start >= n_end) return;

    int M_pad = t->M_pad;
    int K = t->K;

    /* Initialize all Yt rows with bias (or zero) */
    for (int n = n_start; n < n_end; n++) {
        float *yt_row = t->Yt + (size_t)n * M_pad;
        if (t->bias) {
            float b = t->bias[n];
#ifdef __ARM_NEON
            float32x4_t bv = vdupq_n_f32(b);
            for (int m = 0; m < M_pad; m += 4)
                vst1q_f32(yt_row + m, bv);
#else
            for (int m = 0; m < M_pad; m++) yt_row[m] = b;
#endif
        } else {
            memset(yt_row, 0, M_pad * sizeof(float));
        }
    }

    /* K-outer, N-inner: Xt rows loaded once, reused for all N rows.
     * Thread's Yt block stays in L2. */
    int k = 0;
#ifdef __ARM_NEON
    for (; k + 7 < K; k += 8) {
        const float *xt0 = t->Xt + (size_t)k * M_pad;
        const float *xt4 = t->Xt + (size_t)(k + 4) * M_pad;

        for (int n = n_start; n < n_end; n++) {
            float *yt_row = t->Yt + (size_t)n * M_pad;
            const uint16_t *w_row = t->W_bf16 + (size_t)n * K;
            uint16x8_t raw = vld1q_u16(w_row + k);
            float32x4_t wv0 = vreinterpretq_f32_u32(
                vshlq_n_u32(vmovl_u16(vget_low_u16(raw)), 16));
            float32x4_t wv1 = vreinterpretq_f32_u32(
                vshlq_n_u32(vmovl_u16(vget_high_u16(raw)), 16));

            for (int m = 0; m < M_pad; m += 4) {
                float32x4_t yt = vld1q_f32(yt_row + m);
                yt = vfmaq_laneq_f32(yt, vld1q_f32(xt0 + m), wv0, 0);
                yt = vfmaq_laneq_f32(yt, vld1q_f32(xt0 + M_pad + m), wv0, 1);
                yt = vfmaq_laneq_f32(yt, vld1q_f32(xt0 + 2*M_pad + m), wv0, 2);
                yt = vfmaq_laneq_f32(yt, vld1q_f32(xt0 + 3*M_pad + m), wv0, 3);
                yt = vfmaq_laneq_f32(yt, vld1q_f32(xt4 + m), wv1, 0);
                yt = vfmaq_laneq_f32(yt, vld1q_f32(xt4 + M_pad + m), wv1, 1);
                yt = vfmaq_laneq_f32(yt, vld1q_f32(xt4 + 2*M_pad + m), wv1, 2);
                yt = vfmaq_laneq_f32(yt, vld1q_f32(xt4 + 3*M_pad + m), wv1, 3);
                vst1q_f32(yt_row + m, yt);
            }
        }
    }
    for (; k < K; k++) {
        const float *xt_col = t->Xt + (size_t)k * M_pad;
        for (int n = n_start; n < n_end; n++) {
            float *yt_row = t->Yt + (size_t)n * M_pad;
            uint32_t bits = ((uint32_t)t->W_bf16[(size_t)n * K + k]) << 16;
            float wk; memcpy(&wk, &bits, 4);
            float32x4_t wv = vdupq_n_f32(wk);
            for (int m = 0; m < M_pad; m += 4) {
                float32x4_t yt = vld1q_f32(yt_row + m);
                yt = vfmaq_f32(yt, vld1q_f32(xt_col + m), wv);
                vst1q_f32(yt_row + m, yt);
            }
        }
    }
#else
    for (; k < K; k++) {
        const float *xt_col = t->Xt + (size_t)k * M_pad;
        for (int n = n_start; n < n_end; n++) {
            uint32_t bits = ((uint32_t)t->W_bf16[(size_t)n * K + k]) << 16;
            float wk; memcpy(&wk, &bits, 4);
            float *yt_row = t->Yt + (size_t)n * M_pad;
            for (int m = 0; m < M_pad; m++)
                yt_row[m] += wk * xt_col[m];
        }
    }
#endif
}

/* Transpose X[M,K] → Xt[K, M_pad] (zero-padded to multiple of 4) */
static void transpose_pad(float *Xt, const float *X, int M, int K, int M_pad) {
    if (M_pad > M) {
        /* Zero the padding columns for all K rows */
        for (int k = 0; k < K; k++) {
            for (int m = M; m < M_pad; m++) {
                Xt[(size_t)k * M_pad + m] = 0.0f;
            }
        }
    }
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            Xt[(size_t)k * M_pad + m] = X[(size_t)m * K + k];
        }
    }
}

/* Transpose Yt[N, M_pad] → Y[M, N] */
static void transpose_back(float *Y, const float *Yt, int M, int N, int M_pad) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            Y[(size_t)m * N + n] = Yt[(size_t)n * M_pad + m];
        }
    }
}

/* Batched FP32 GEMM: Y[M,N] = X[M,K] @ W[N,K]^T + bias[N] */
static void f32_gemm_batched(float *Y, const float *X, const float *W,
                              const float *bias, int M, int K, int N) {
    int M_pad = (M + 3) & ~3; /* round up to multiple of 4 */

    float *Xt = (float *)malloc((size_t)K * M_pad * sizeof(float));
    float *Yt = (float *)malloc((size_t)N * M_pad * sizeof(float));

    transpose_pad(Xt, X, M, K, M_pad);

    f32_gemm_task_t task = { Yt, Xt, W, bias, M_pad, K, N };
    if (tp.n_threads <= 1) {
        f32_gemm_worker(0, 1, &task);
    } else {
        parallel_for(f32_gemm_worker, &task);
    }

    transpose_back(Y, Yt, M, N, M_pad);

    free(Xt);
    free(Yt);
}

/* Batched BF16 GEMM: Y[M,N] = X[M,K] @ W_bf16[N,K]^T + bias[N] */
static void bf16_gemm_batched(float *Y, const float *X, const uint16_t *W_bf16,
                               const float *bias, int M, int K, int N) {
    int M_pad = (M + 3) & ~3;

    float *Xt = (float *)malloc((size_t)K * M_pad * sizeof(float));
    float *Yt = (float *)malloc((size_t)N * M_pad * sizeof(float));

    transpose_pad(Xt, X, M, K, M_pad);

    bf16_gemm_task_t task = { Yt, Xt, W_bf16, bias, M_pad, K, N };
    if (tp.n_threads <= 1) {
        bf16_gemm_worker(0, 1, &task);
    } else {
        parallel_for(bf16_gemm_worker, &task);
    }

    transpose_back(Y, Yt, M, N, M_pad);

    free(Xt);
    free(Yt);
}

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

void qwen_matmul_t(float *C, const float *A, const float *B, int M, int K, int N) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
#else
    if (M > 1) {
        f32_gemm_batched(C, A, B, NULL, M, K, N);
    } else {
        f32_matvec_threaded(C, A, B, NULL, K, N);
    }
#endif
}

void qwen_linear(float *y, const float *x, const float *W, const float *b,
                 int seq_len, int in_dim, int out_dim) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, out_dim, in_dim,
                1.0f, x, in_dim, W, in_dim,
                0.0f, y, out_dim);
    if (b != NULL) {
        for (int s = 0; s < seq_len; s++) {
            for (int o = 0; o < out_dim; o++) {
                y[s * out_dim + o] += b[o];
            }
        }
    }
#else
    if (seq_len > 1) {
        f32_gemm_batched(y, x, W, b, seq_len, in_dim, out_dim);
    } else {
        f32_matvec_threaded(y, x, W, b, in_dim, out_dim);
    }
#endif
}

void qwen_linear_nobias(float *y, const float *x, const float *W,
                         int seq_len, int in_dim, int out_dim) {
    qwen_linear(y, x, W, NULL, seq_len, in_dim, out_dim);
}

/*
 * Fused BF16 matvec: y[out_dim] = W_bf16[out_dim, in_dim] @ x[in_dim] + bias
 * Processes 2 output rows at a time to amortize x vector loads.
 */
static void bf16_matvec_fused(float *y, const float *x, const uint16_t *W_bf16,
                               const float *bias, int in_dim, int out_dim) {
    qwen_bf16_matvec_fused_impl(y, x, W_bf16, bias, in_dim, out_dim);
}

/*
 * Fused FP32 matvec: y[out_dim] = W[out_dim, in_dim] @ x[in_dim] + bias
 */
static void f32_matvec_fused(float *y, const float *x, const float *W,
                              const float *bias, int in_dim, int out_dim) {
    qwen_f32_matvec_fused_impl(y, x, W, bias, in_dim, out_dim);
}

/* Threaded matvec: split output rows across threads */
typedef struct {
    float *y;
    const float *x;
    const uint16_t *W_bf16;
    const float *bias;
    int in_dim;
    int out_dim;
} matvec_task_t;

static void matvec_worker(int tid, int n_threads, void *arg) {
    matvec_task_t *t = (matvec_task_t *)arg;
    int chunk = (t->out_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->out_dim) end = t->out_dim;
    if (start >= end) return;

    bf16_matvec_fused(t->y + start, t->x,
                      t->W_bf16 + (size_t)start * t->in_dim,
                      t->bias ? t->bias + start : NULL,
                      t->in_dim, end - start);
}

static void bf16_matvec_threaded(float *y, const float *x, const uint16_t *W_bf16,
                                  const float *bias, int in_dim, int out_dim) {
    if (tp.n_threads <= 1) {
        bf16_matvec_fused(y, x, W_bf16, bias, in_dim, out_dim);
        return;
    }
    matvec_task_t task = { y, x, W_bf16, bias, in_dim, out_dim };
    parallel_for(matvec_worker, &task);
}

/* Threaded FP32 matvec: split output rows across threads */
typedef struct {
    float *y;
    const float *x;
    const float *W;
    const float *bias;
    int in_dim;
    int out_dim;
} f32_matvec_task_t;

static void f32_matvec_worker(int tid, int n_threads, void *arg) {
    f32_matvec_task_t *t = (f32_matvec_task_t *)arg;
    int chunk = (t->out_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->out_dim) end = t->out_dim;
    if (start >= end) return;

    f32_matvec_fused(t->y + start, t->x,
                     t->W + (size_t)start * t->in_dim,
                     t->bias ? t->bias + start : NULL,
                     t->in_dim, end - start);
}

static void f32_matvec_threaded(float *y, const float *x, const float *W,
                                 const float *bias, int in_dim, int out_dim) {
    if (tp.n_threads <= 1) {
        f32_matvec_fused(y, x, W, bias, in_dim, out_dim);
        return;
    }
    f32_matvec_task_t task = { y, x, W, bias, in_dim, out_dim };
    parallel_for(f32_matvec_worker, &task);
}

typedef struct {
    float *q;
    float *k;
    float *v;
    const float *x;
    const uint16_t *Wq_bf16;
    const uint16_t *Wk_bf16;
    const uint16_t *Wv_bf16;
    int in_dim;
    int q_dim;
    int kv_dim;
    int total_dim;
} qkv_matvec_task_t;

static void qkv_matvec_worker(int tid, int n_threads, void *arg) {
    qkv_matvec_task_t *t = (qkv_matvec_task_t *)arg;
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
            bf16_matvec_fused(t->q + s, t->x,
                              t->Wq_bf16 + (size_t)s * t->in_dim,
                              NULL, t->in_dim, e - s);
        }
    }

    if (end > q_end && start < k_end) {
        int s = start > q_end ? start - q_end : 0;
        int e_abs = end < k_end ? end : k_end;
        int e = e_abs - q_end;
        if (s < e) {
            bf16_matvec_fused(t->k + s, t->x,
                              t->Wk_bf16 + (size_t)s * t->in_dim,
                              NULL, t->in_dim, e - s);
        }
    }

    if (end > k_end && start < v_end) {
        int s = start > k_end ? start - k_end : 0;
        int e_abs = end < v_end ? end : v_end;
        int e = e_abs - k_end;
        if (s < e) {
            bf16_matvec_fused(t->v + s, t->x,
                              t->Wv_bf16 + (size_t)s * t->in_dim,
                              NULL, t->in_dim, e - s);
        }
    }
}

void qwen_linear_nobias_bf16_qkv(float *q, float *k, float *v, const float *x,
                                 const uint16_t *Wq_bf16,
                                 const uint16_t *Wk_bf16,
                                 const uint16_t *Wv_bf16,
                                 int in_dim, int q_dim, int kv_dim) {
    if (tp.n_threads <= 1) {
        bf16_matvec_fused(q, x, Wq_bf16, NULL, in_dim, q_dim);
        bf16_matvec_fused(k, x, Wk_bf16, NULL, in_dim, kv_dim);
        bf16_matvec_fused(v, x, Wv_bf16, NULL, in_dim, kv_dim);
        return;
    }

    qkv_matvec_task_t task = {
        .q = q,
        .k = k,
        .v = v,
        .x = x,
        .Wq_bf16 = Wq_bf16,
        .Wk_bf16 = Wk_bf16,
        .Wv_bf16 = Wv_bf16,
        .in_dim = in_dim,
        .q_dim = q_dim,
        .kv_dim = kv_dim,
        .total_dim = q_dim + 2 * kv_dim,
    };
    parallel_for(qkv_matvec_worker, &task);
}

void qwen_linear_nobias_bf16(float *y, const float *x, const uint16_t *W_bf16,
                              int seq_len, int in_dim, int out_dim) {
    if (seq_len > 1) {
        bf16_gemm_batched(y, x, W_bf16, NULL, seq_len, in_dim, out_dim);
    } else {
        bf16_matvec_threaded(y, x, W_bf16, NULL, in_dim, out_dim);
    }
}

void qwen_linear_bf16(float *y, const float *x, const uint16_t *W_bf16,
                      const float *b, int seq_len, int in_dim, int out_dim) {
    if (seq_len > 1) {
        bf16_gemm_batched(y, x, W_bf16, b, seq_len, in_dim, out_dim);
    } else {
        bf16_matvec_threaded(y, x, W_bf16, b, in_dim, out_dim);
    }
}

/* Find argmax over a range of output rows [start, end).
 * Uses 2-row processing to amortize x vector loads (same as bf16_matvec_fused). */
static void argmax_bf16_range(const float *x, const uint16_t *W_bf16,
                               int in_dim, int start, int end,
                               int *best_out, float *best_val_out) {
    qwen_argmax_bf16_range_impl(x, W_bf16, in_dim, start, end, best_out, best_val_out);
}

typedef struct {
    const float *x;
    const uint16_t *W_bf16;
    int in_dim;
    int out_dim;
    int best_idx[QWEN_MAX_THREADS];
    float best_val[QWEN_MAX_THREADS];
} argmax_task_t;

static void argmax_worker(int tid, int n_threads, void *arg) {
    argmax_task_t *t = (argmax_task_t *)arg;
    int chunk = (t->out_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->out_dim) end = t->out_dim;
    if (start >= end) {
        t->best_val[tid] = -1e30f;
        t->best_idx[tid] = 0;
        return;
    }
    argmax_bf16_range(t->x, t->W_bf16, t->in_dim, start, end,
                      &t->best_idx[tid], &t->best_val[tid]);
}

int qwen_argmax_matvec_bf16(const float *x, const uint16_t *W_bf16,
                             int in_dim, int out_dim) {
    if (tp.n_threads <= 1) {
        int best;
        float best_val;
        argmax_bf16_range(x, W_bf16, in_dim, 0, out_dim, &best, &best_val);
        return best;
    }

    argmax_task_t task;
    task.x = x;
    task.W_bf16 = W_bf16;
    task.in_dim = in_dim;
    task.out_dim = out_dim;
    parallel_for(argmax_worker, &task);

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

/* Q8_0 argmax: find argmax(W_q8 @ x) using INT8 dot products */
static void argmax_q8_range(const block_q8_0 *x_q8, const block_q8_0 *W_q8,
                              int n_blocks, int start, int end,
                              int *best_out, float *best_val_out) {
    qwen_argmax_q8_range_impl(x_q8, W_q8, n_blocks, start, end, best_out, best_val_out);
}

typedef struct {
    const block_q8_0 *x_q8;
    const block_q8_0 *W_q8;
    int n_blocks;
    int out_dim;
    int best_idx[QWEN_MAX_THREADS];
    float best_val[QWEN_MAX_THREADS];
} argmax_q8_task_t;

static void argmax_q8_worker(int tid, int n_threads, void *arg) {
    argmax_q8_task_t *t = (argmax_q8_task_t *)arg;
    int chunk = (t->out_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->out_dim) end = t->out_dim;
    if (start >= end) {
        t->best_val[tid] = -1e30f;
        t->best_idx[tid] = 0;
        return;
    }
    argmax_q8_range(t->x_q8, t->W_q8, t->n_blocks, start, end,
                     &t->best_idx[tid], &t->best_val[tid]);
}

int qwen_argmax_matvec_q8(const float *x, const block_q8_0 *W_q8,
                            int in_dim, int out_dim) {
    int n_blocks = in_dim / QK8_0;

    /* Stack-allocate x_q8: max 64 blocks = 2048 elements, ~2.3KB on stack */
    block_q8_0 x_q8[64];
    quantize_f32_to_q8_0(x, x_q8, in_dim);

    int best;
    float best_val;

    if (tp.n_threads <= 1) {
        argmax_q8_range(x_q8, W_q8, n_blocks, 0, out_dim, &best, &best_val);
    } else {
        argmax_q8_task_t task;
        task.x_q8 = x_q8;
        task.W_q8 = W_q8;
        task.n_blocks = n_blocks;
        task.out_dim = out_dim;
        parallel_for(argmax_q8_worker, &task);

        best = task.best_idx[0];
        best_val = task.best_val[0];
        for (int i = 1; i < tp.n_threads; i++) {
            if (task.best_val[i] > best_val) {
                best_val = task.best_val[i];
                best = task.best_idx[i];
            }
        }
    }

    return best;
}

void qwen_matmul_t_bf16(float *C, const float *A, const uint16_t *B_bf16,
                         int M, int K, int N) {
    if (M > 1) {
        bf16_gemm_batched(C, A, B_bf16, NULL, M, K, N);
    } else {
        bf16_matvec_threaded(C, A, B_bf16, NULL, K, N);
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
        parallel_for(q8_gemm_worker, &task);
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
        parallel_for(q8_matvec_worker, &task);
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
    parallel_for(q8_qkv_matvec_worker, &task);
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
        parallel_for(q8_gemm_worker, &task);
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
    parallel_for(conv2d_gemm_worker, &task);
    free(cols);
#endif
}

/* ========================================================================
 * Normalization
 * ======================================================================== */

void qwen_layer_norm(float *out, const float *x, const float *weight, const float *bias,
                     int seq_len, int hidden, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

        /* Compute mean */
#if defined(__AVX512F__)
        __m512 sumv = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= hidden; i += 16) {
            sumv = _mm512_add_ps(sumv, _mm512_loadu_ps(x_row + i));
        }
        float mean = _mm512_reduce_add_ps(sumv);
        for (; i < hidden; i++) mean += x_row[i];
#elif defined(__AVX2__)
        __m256 sumv = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= hidden; i += 8) {
            sumv = _mm256_add_ps(sumv, _mm256_loadu_ps(x_row + i));
        }
        __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sumv), _mm256_extractf128_ps(sumv, 1));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float mean = _mm_cvtss_f32(sum128);
        for (; i < hidden; i++) mean += x_row[i];
#elif defined(__ARM_NEON)
        float32x4_t sumv0 = vdupq_n_f32(0.0f);
        float32x4_t sumv1 = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 8 <= hidden; i += 8) {
            sumv0 = vaddq_f32(sumv0, vld1q_f32(x_row + i));
            sumv1 = vaddq_f32(sumv1, vld1q_f32(x_row + i + 4));
        }
        float mean = vaddvq_f32(vaddq_f32(sumv0, sumv1));
        for (; i < hidden; i++) mean += x_row[i];
#else
        float mean = 0.0f;
        for (int i = 0; i < hidden; i++) mean += x_row[i];
#endif
        mean /= hidden;

        /* Compute variance */
#if defined(__AVX512F__) && defined(__FMA__)
        __m512 meanv = _mm512_set1_ps(mean);
        __m512 accv = _mm512_setzero_ps();
        int j = 0;
        for (; j + 16 <= hidden; j += 16) {
            __m512 v = _mm512_sub_ps(_mm512_loadu_ps(x_row + j), meanv);
            accv = _mm512_fmadd_ps(v, v, accv);
        }
        float var = _mm512_reduce_add_ps(accv);
        for (; j < hidden; j++) {
            float d = x_row[j] - mean;
            var += d * d;
        }
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 meanv = _mm256_set1_ps(mean);
        __m256 accv = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= hidden; j += 8) {
            __m256 v = _mm256_sub_ps(_mm256_loadu_ps(x_row + j), meanv);
            accv = _mm256_fmadd_ps(v, v, accv);
        }
        __m128 acc128 = _mm_add_ps(_mm256_castps256_ps128(accv), _mm256_extractf128_ps(accv, 1));
        acc128 = _mm_hadd_ps(acc128, acc128);
        acc128 = _mm_hadd_ps(acc128, acc128);
        float var = _mm_cvtss_f32(acc128);
        for (; j < hidden; j++) {
            float d = x_row[j] - mean;
            var += d * d;
        }
#elif defined(__ARM_NEON)
        float32x4_t meanv = vdupq_n_f32(mean);
        float32x4_t accv0 = vdupq_n_f32(0.0f);
        float32x4_t accv1 = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j + 8 <= hidden; j += 8) {
            float32x4_t d0 = vsubq_f32(vld1q_f32(x_row + j), meanv);
            float32x4_t d1 = vsubq_f32(vld1q_f32(x_row + j + 4), meanv);
            accv0 = vfmaq_f32(accv0, d0, d0);
            accv1 = vfmaq_f32(accv1, d1, d1);
        }
        float var = vaddvq_f32(vaddq_f32(accv0, accv1));
        for (; j < hidden; j++) {
            float d = x_row[j] - mean;
            var += d * d;
        }
#else
        float var = 0.0f;
        for (int i = 0; i < hidden; i++) {
            float d = x_row[i] - mean;
            var += d * d;
        }
#endif
        var /= hidden;

        float inv_std = 1.0f / sqrtf(var + eps);
#if defined(__AVX512F__) && defined(__FMA__)
        __m512 meanv2 = _mm512_set1_ps(mean);
        __m512 invv = _mm512_set1_ps(inv_std);
        int k = 0;
        for (; k + 16 <= hidden; k += 16) {
            __m512 vx = _mm512_sub_ps(_mm512_loadu_ps(x_row + k), meanv2);
            __m512 vw = _mm512_loadu_ps(weight + k);
            __m512 vb = _mm512_loadu_ps(bias + k);
            __m512 v = _mm512_mul_ps(_mm512_mul_ps(vx, invv), vw);
            v = _mm512_add_ps(v, vb);
            _mm512_storeu_ps(out_row + k, v);
        }
        for (; k < hidden; k++) {
            out_row[k] = (x_row[k] - mean) * inv_std * weight[k] + bias[k];
        }
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 meanv2 = _mm256_set1_ps(mean);
        __m256 invv = _mm256_set1_ps(inv_std);
        int k = 0;
        for (; k + 8 <= hidden; k += 8) {
            __m256 vx = _mm256_sub_ps(_mm256_loadu_ps(x_row + k), meanv2);
            __m256 vw = _mm256_loadu_ps(weight + k);
            __m256 vb = _mm256_loadu_ps(bias + k);
            __m256 v = _mm256_mul_ps(_mm256_mul_ps(vx, invv), vw);
            v = _mm256_add_ps(v, vb);
            _mm256_storeu_ps(out_row + k, v);
        }
        for (; k < hidden; k++) {
            out_row[k] = (x_row[k] - mean) * inv_std * weight[k] + bias[k];
        }
#elif defined(__ARM_NEON)
        float32x4_t meanv2 = vdupq_n_f32(mean);
        float32x4_t invv = vdupq_n_f32(inv_std);
        int k = 0;
        for (; k + 8 <= hidden; k += 8) {
            float32x4_t vx0 = vsubq_f32(vld1q_f32(x_row + k), meanv2);
            float32x4_t vw0 = vld1q_f32(weight + k);
            float32x4_t vb0 = vld1q_f32(bias + k);
            float32x4_t vx1 = vsubq_f32(vld1q_f32(x_row + k + 4), meanv2);
            float32x4_t vw1 = vld1q_f32(weight + k + 4);
            float32x4_t vb1 = vld1q_f32(bias + k + 4);
            vst1q_f32(out_row + k, vaddq_f32(vmulq_f32(vmulq_f32(vx0, invv), vw0), vb0));
            vst1q_f32(out_row + k + 4, vaddq_f32(vmulq_f32(vmulq_f32(vx1, invv), vw1), vb1));
        }
        for (; k < hidden; k++) {
            out_row[k] = (x_row[k] - mean) * inv_std * weight[k] + bias[k];
        }
#else
        for (int i = 0; i < hidden; i++) {
            out_row[i] = (x_row[i] - mean) * inv_std * weight[i] + bias[i];
        }
#endif
    }
}

void qwen_rms_norm(float *out, const float *x, const float *weight,
                   int seq_len, int hidden, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

#if defined(__AVX512F__) && defined(__FMA__)
        __m512 accv = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= hidden; i += 16) {
            __m512 v = _mm512_loadu_ps(x_row + i);
            accv = _mm512_fmadd_ps(v, v, accv);
        }
        float sum_sq = _mm512_reduce_add_ps(accv);
        for (; i < hidden; i++) sum_sq += x_row[i] * x_row[i];
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 accv = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= hidden; i += 8) {
            __m256 v = _mm256_loadu_ps(x_row + i);
            accv = _mm256_fmadd_ps(v, v, accv);
        }
        __m128 acc128 = _mm_add_ps(_mm256_castps256_ps128(accv), _mm256_extractf128_ps(accv, 1));
        acc128 = _mm_hadd_ps(acc128, acc128);
        acc128 = _mm_hadd_ps(acc128, acc128);
        float sum_sq = _mm_cvtss_f32(acc128);
        for (; i < hidden; i++) sum_sq += x_row[i] * x_row[i];
#elif defined(__ARM_NEON)
        float32x4_t accv0 = vdupq_n_f32(0.0f);
        float32x4_t accv1 = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 8 <= hidden; i += 8) {
            float32x4_t v0 = vld1q_f32(x_row + i);
            float32x4_t v1 = vld1q_f32(x_row + i + 4);
            accv0 = vfmaq_f32(accv0, v0, v0);
            accv1 = vfmaq_f32(accv1, v1, v1);
        }
        float sum_sq = vaddvq_f32(vaddq_f32(accv0, accv1));
        for (; i < hidden; i++) sum_sq += x_row[i] * x_row[i];
#else
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden; i++) {
            sum_sq += x_row[i] * x_row[i];
        }
#endif
        float rms_inv = 1.0f / sqrtf(sum_sq / hidden + eps);

#if defined(__AVX512F__)
        __m512 scale = _mm512_set1_ps(rms_inv);
        int j = 0;
        for (; j + 16 <= hidden; j += 16) {
            __m512 vx = _mm512_loadu_ps(x_row + j);
            __m512 vw = _mm512_loadu_ps(weight + j);
            _mm512_storeu_ps(out_row + j, _mm512_mul_ps(_mm512_mul_ps(vx, vw), scale));
        }
        for (; j < hidden; j++) out_row[j] = x_row[j] * rms_inv * weight[j];
#elif defined(__AVX2__)
        __m256 scale = _mm256_set1_ps(rms_inv);
        int j = 0;
        for (; j + 8 <= hidden; j += 8) {
            __m256 vx = _mm256_loadu_ps(x_row + j);
            __m256 vw = _mm256_loadu_ps(weight + j);
            _mm256_storeu_ps(out_row + j, _mm256_mul_ps(_mm256_mul_ps(vx, vw), scale));
        }
        for (; j < hidden; j++) out_row[j] = x_row[j] * rms_inv * weight[j];
#elif defined(__ARM_NEON)
        float32x4_t scalev = vdupq_n_f32(rms_inv);
        int j = 0;
        for (; j + 8 <= hidden; j += 8) {
            float32x4_t vx0 = vld1q_f32(x_row + j);
            float32x4_t vw0 = vld1q_f32(weight + j);
            float32x4_t vx1 = vld1q_f32(x_row + j + 4);
            float32x4_t vw1 = vld1q_f32(weight + j + 4);
            vst1q_f32(out_row + j, vmulq_f32(vmulq_f32(vx0, vw0), scalev));
            vst1q_f32(out_row + j + 4, vmulq_f32(vmulq_f32(vx1, vw1), scalev));
        }
        for (; j < hidden; j++) out_row[j] = x_row[j] * rms_inv * weight[j];
#else
        for (int i = 0; i < hidden; i++) {
            out_row[i] = x_row[i] * rms_inv * weight[i];
        }
#endif
    }
}

void qwen_rms_norm_per_head(float *x, const float *weight,
                             int seq_len, int n_heads, int head_dim, float eps) {
    /* x is [seq, n_heads * head_dim] - normalize each [head_dim] segment */
    int hidden = n_heads * head_dim;
    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < n_heads; h++) {
            float *vec = x + s * hidden + h * head_dim;

#if defined(__AVX512F__) && defined(__FMA__)
            __m512 accv = _mm512_setzero_ps();
            int d = 0;
            for (; d + 16 <= head_dim; d += 16) {
                __m512 v = _mm512_loadu_ps(vec + d);
                accv = _mm512_fmadd_ps(v, v, accv);
            }
            float sum_sq = _mm512_reduce_add_ps(accv);
            for (; d < head_dim; d++) sum_sq += vec[d] * vec[d];
#elif defined(__AVX2__) && defined(__FMA__)
            __m256 accv = _mm256_setzero_ps();
            int d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 v = _mm256_loadu_ps(vec + d);
                accv = _mm256_fmadd_ps(v, v, accv);
            }
            __m128 acc128 = _mm_add_ps(_mm256_castps256_ps128(accv), _mm256_extractf128_ps(accv, 1));
            acc128 = _mm_hadd_ps(acc128, acc128);
            acc128 = _mm_hadd_ps(acc128, acc128);
            float sum_sq = _mm_cvtss_f32(acc128);
            for (; d < head_dim; d++) sum_sq += vec[d] * vec[d];
#elif defined(__ARM_NEON)
            float32x4_t accv0 = vdupq_n_f32(0.0f);
            float32x4_t accv1 = vdupq_n_f32(0.0f);
            int d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                float32x4_t v0 = vld1q_f32(vec + d);
                float32x4_t v1 = vld1q_f32(vec + d + 4);
                accv0 = vfmaq_f32(accv0, v0, v0);
                accv1 = vfmaq_f32(accv1, v1, v1);
            }
            float sum_sq = vaddvq_f32(vaddq_f32(accv0, accv1));
            for (; d < head_dim; d++) sum_sq += vec[d] * vec[d];
#else
            float sum_sq = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum_sq += vec[d] * vec[d];
            }
#endif
            float rms_inv = 1.0f / sqrtf(sum_sq / head_dim + eps);

#if defined(__AVX512F__)
            __m512 scale = _mm512_set1_ps(rms_inv);
            int j = 0;
            for (; j + 16 <= head_dim; j += 16) {
                __m512 v = _mm512_loadu_ps(vec + j);
                __m512 w = _mm512_loadu_ps(weight + j);
                _mm512_storeu_ps(vec + j, _mm512_mul_ps(_mm512_mul_ps(v, w), scale));
            }
            for (; j < head_dim; j++) vec[j] = vec[j] * rms_inv * weight[j];
#elif defined(__AVX2__)
            __m256 scale = _mm256_set1_ps(rms_inv);
            int j = 0;
            for (; j + 8 <= head_dim; j += 8) {
                __m256 v = _mm256_loadu_ps(vec + j);
                __m256 w = _mm256_loadu_ps(weight + j);
                _mm256_storeu_ps(vec + j, _mm256_mul_ps(_mm256_mul_ps(v, w), scale));
            }
            for (; j < head_dim; j++) vec[j] = vec[j] * rms_inv * weight[j];
#elif defined(__ARM_NEON)
            float32x4_t scalev = vdupq_n_f32(rms_inv);
            int j = 0;
            for (; j + 8 <= head_dim; j += 8) {
                float32x4_t v0 = vld1q_f32(vec + j);
                float32x4_t w0 = vld1q_f32(weight + j);
                float32x4_t v1 = vld1q_f32(vec + j + 4);
                float32x4_t w1 = vld1q_f32(weight + j + 4);
                vst1q_f32(vec + j, vmulq_f32(vmulq_f32(v0, w0), scalev));
                vst1q_f32(vec + j + 4, vmulq_f32(vmulq_f32(v1, w1), scalev));
            }
            for (; j < head_dim; j++) vec[j] = vec[j] * rms_inv * weight[j];
#else
            for (int d = 0; d < head_dim; d++) {
                vec[d] = vec[d] * rms_inv * weight[d];
            }
#endif
        }
    }
}

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

#ifdef __ARM_NEON
/* NEON fast expf: 7th-order polynomial, max error ~1e-5 */
static inline float32x4_t neon_expf(float32x4_t x) {
    /* Clamp to [-88, 88] to avoid overflow */
    x = vmaxq_f32(x, vdupq_n_f32(-88.0f));
    x = vminq_f32(x, vdupq_n_f32(88.0f));
    /* exp(x) = 2^(x / ln2) = 2^(n + f) where n = round(x/ln2), f = fractional */
    float32x4_t log2e = vdupq_n_f32(1.44269504089f);
    float32x4_t t = vmulq_f32(x, log2e);
    float32x4_t n = vrndnq_f32(t);  /* round to nearest int */
    float32x4_t f = vsubq_f32(t, n);  /* fractional part in [-0.5, 0.5] */
    /* 2^f ≈ polynomial in f (minimax on [-0.5, 0.5]) */
    float32x4_t p = vdupq_n_f32(1.535336188e-4f);
    p = vfmaq_f32(vdupq_n_f32(1.339887440e-3f), p, f);
    p = vfmaq_f32(vdupq_n_f32(9.618437357e-3f), p, f);
    p = vfmaq_f32(vdupq_n_f32(5.550332471e-2f), p, f);
    p = vfmaq_f32(vdupq_n_f32(2.402264791e-1f), p, f);
    p = vfmaq_f32(vdupq_n_f32(6.931472028e-1f), p, f);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, f);
    /* Scale by 2^n: reinterpret n as exponent bits */
    int32x4_t ni = vcvtq_s32_f32(n);
    int32x4_t exp_bits = vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(exp_bits));
}

static inline float32x4_t neon_tanhf(float32x4_t x) {
    /* tanh(x) = 1 - 2 / (1 + exp(2x)) */
    float32x4_t two_x = vaddq_f32(x, x);
    float32x4_t e2x = neon_expf(two_x);
    float32x4_t one = vdupq_n_f32(1.0f);
    return vsubq_f32(one, vdivq_f32(vdupq_n_f32(2.0f), vaddq_f32(one, e2x)));
}
#endif /* __ARM_NEON */

void qwen_silu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float val = x[i];
        x[i] = val / (1.0f + expf(-val));
    }
}

void qwen_gelu(float *x, int n) {
#ifdef __ARM_NEON
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t coeff = vdupq_n_f32(0.7978845608028654f);
    const float32x4_t c3 = vdupq_n_f32(0.044715f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t v3 = vmulq_f32(vmulq_f32(v, v), v);
        float32x4_t inner = vmulq_f32(coeff, vfmaq_f32(v, c3, v3));
        float32x4_t t = neon_tanhf(inner);
        vst1q_f32(x + i, vmulq_f32(half, vmulq_f32(v, vaddq_f32(one, t))));
    }
    for (; i < n; i++) {
        float val = x[i];
        float x3 = val * val * val;
        float inner = 0.7978845608028654f * (val + 0.044715f * x3);
        x[i] = 0.5f * val * (1.0f + tanhf(inner));
    }
#else
    for (int i = 0; i < n; i++) {
        float val = x[i];
        float x3 = val * val * val;
        float inner = 0.7978845608028654f * (val + 0.044715f * x3);
        x[i] = 0.5f * val * (1.0f + tanhf(inner));
    }
#endif
}

typedef struct {
    float *out;
    const float *gate_up;
    int seq_len;
    int intermediate;
} swiglu_task_t;

static void swiglu_worker(int tid, int n_threads, void *arg) {
    swiglu_task_t *t = (swiglu_task_t *)arg;
    int chunk = (t->seq_len + n_threads - 1) / n_threads;
    int s0 = tid * chunk;
    int s1 = s0 + chunk;
    if (s1 > t->seq_len) s1 = t->seq_len;
    if (s0 >= s1) return;

    int inter = t->intermediate;
    int alias_inplace = (t->out == t->gate_up);
    for (int s = s0; s < s1; s++) {
        const float *gu = t->gate_up + (size_t)s * 2 * inter;
        float *o = t->out + (size_t)s * inter;
        if (!alias_inplace) {
#if defined(__APPLE__) && defined(USE_BLAS)
            /* Fast path for prefill: vectorized exp(-g) using Accelerate/vForce. */
            for (int j = 0; j < inter; j++) o[j] = -gu[2 * j];
            int n = inter;
            vvexpf(o, o, &n);
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                o[j] = (g / (1.0f + o[j])) * u;
            }
#elif defined(__ARM_NEON)
            /* NEON SiLU: process 4 gate-up pairs at a time */
            int j = 0;
            float32x4_t one = vdupq_n_f32(1.0f);
            for (; j + 3 < inter; j += 4) {
                float32x4x2_t gu4 = vld2q_f32(gu + 2 * j);
                float32x4_t g = gu4.val[0];
                float32x4_t u = gu4.val[1];
                float32x4_t neg_g = vnegq_f32(g);
                float32x4_t silu = vdivq_f32(g, vaddq_f32(one, neon_expf(neg_g)));
                vst1q_f32(o + j, vmulq_f32(silu, u));
            }
            for (; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                g = g / (1.0f + expf(-g));
                o[j] = g * u;
            }
#else
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                g = g / (1.0f + expf(-g)); /* SiLU */
                o[j] = g * u;
            }
#endif
        } else {
            /* In-place mode (decode seq=1): gate_up is interleaved [g0,u0,g1,u1,...],
             * out writes to front half. out[j] reads from 2j,2j+1 (2j >= j for j>=1),
             * so forward order is safe. */
#ifdef __ARM_NEON
            int j = 0;
            float32x4_t one = vdupq_n_f32(1.0f);
            for (; j + 3 < inter; j += 4) {
                float32x4x2_t gu4 = vld2q_f32(gu + 2 * j);
                float32x4_t g = gu4.val[0];
                float32x4_t u = gu4.val[1];
                float32x4_t neg_g = vnegq_f32(g);
                float32x4_t silu = vdivq_f32(g, vaddq_f32(one, neon_expf(neg_g)));
                vst1q_f32(o + j, vmulq_f32(silu, u));
            }
            for (; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                g = g / (1.0f + expf(-g));
                o[j] = g * u;
            }
#else
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                g = g / (1.0f + expf(-g)); /* SiLU */
                o[j] = g * u;
            }
#endif
        }
    }
}

void qwen_swiglu_multiply(float *out, const float *gate_up, int seq_len, int intermediate) {
    swiglu_task_t task = {
        .out = out,
        .gate_up = gate_up,
        .seq_len = seq_len,
        .intermediate = intermediate
    };

    if (tp.n_threads > 1 && seq_len >= 2 && intermediate >= 256) {
        parallel_for(swiglu_worker, &task);
    } else {
        swiglu_worker(0, 1, &task);
    }
}

void qwen_softmax(float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;
        float max_val = row[0];
        for (int c = 1; c < cols; c++) {
            if (row[c] > max_val) max_val = row[c];
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - max_val);
            sum += row[c];
        }
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) {
            row[c] *= inv_sum;
        }
    }
}

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

static inline float qwen_dot_f32(const float *a, const float *b, int n) {
    return qwen_dot_f32_impl(a, b, n);
}

/* dst = dst * scale */
static inline void qwen_vec_scale_inplace(float *dst, float scale, int n) {
    qwen_vec_scale_inplace_impl(dst, scale, n);
}

/* dst += alpha * src */
static inline void qwen_vec_axpy_inplace(float *dst, const float *src, float alpha, int n) {
    qwen_vec_axpy_inplace_impl(dst, src, alpha, n);
}

/* dst = dst * correction + src */
static inline void qwen_vec_scale_add(float *dst, const float *src, float correction, int n) {
    qwen_vec_scale_add_impl(dst, src, correction, n);
}

static void qwen_bidirectional_attention_heads(float *out, const float *Q, const float *K,
                                                const float *V, int n_heads, int head_dim,
                                                float scale, const int *window_starts,
                                                int n_windows, int head_start, int head_end) {
    int hidden = n_heads * head_dim;

    for (int h = head_start; h < head_end; h++) {
        for (int w = 0; w < n_windows; w++) {
            int ws = window_starts[w];
            int we = window_starts[w + 1];

            for (int i = ws; i < we; i++) {
                const float *q_row = Q + i * hidden + h * head_dim;
                float *o_row = out + i * hidden + h * head_dim;

                /* Online softmax */
                float max_score = -1e30f;
                float sum_exp = 0.0f;
                for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

                for (int j = ws; j < we; j++) {
                    const float *k_row = K + j * hidden + h * head_dim;
                    const float *v_row = V + j * hidden + h * head_dim;

                    float score = qwen_dot_f32(q_row, k_row, head_dim) * scale;

                    if (score > max_score) {
                        float correction = expf(max_score - score);
                        sum_exp = sum_exp * correction + 1.0f;
                        qwen_vec_scale_add(o_row, v_row, correction, head_dim);
                        max_score = score;
                    } else {
                        float wt = expf(score - max_score);
                        sum_exp += wt;
                        qwen_vec_axpy_inplace(o_row, v_row, wt, head_dim);
                    }
                }

                if (sum_exp > 0.0f) {
                    float inv_sum = 1.0f / sum_exp;
                    qwen_vec_scale_inplace(o_row, inv_sum, head_dim);
                }
            }
        }
    }
}

typedef struct {
    float *out;
    const float *Q;
    const float *K;
    const float *V;
    int n_heads;
    int head_dim;
    float scale;
    const int *window_starts;
    int n_windows;
} bidir_attn_task_t;

static void bidir_attn_worker(int tid, int n_threads, void *arg) {
    bidir_attn_task_t *t = (bidir_attn_task_t *)arg;
    int chunk = (t->n_heads + n_threads - 1) / n_threads;
    int h0 = tid * chunk;
    int h1 = h0 + chunk;
    if (h1 > t->n_heads) h1 = t->n_heads;
    if (h0 >= h1) return;

    qwen_bidirectional_attention_heads(t->out, t->Q, t->K, t->V,
                                        t->n_heads, t->head_dim, t->scale,
                                        t->window_starts, t->n_windows, h0, h1);
}

void qwen_bidirectional_attention(float *out, const float *Q, const float *K,
                                   const float *V, int seq __attribute__((unused)),
                                   int n_heads, int head_dim, float scale,
                                   const int *window_starts, int n_windows) {
    if (tp.n_threads > 1 && n_heads >= 2) {
        bidir_attn_task_t task = {
            .out = out, .Q = Q, .K = K, .V = V,
            .n_heads = n_heads, .head_dim = head_dim, .scale = scale,
            .window_starts = window_starts, .n_windows = n_windows
        };
        parallel_for(bidir_attn_worker, &task);
        return;
    }

    qwen_bidirectional_attention_heads(out, Q, K, V, n_heads, head_dim, scale,
                                        window_starts, n_windows, 0, n_heads);
}

static void qwen_causal_attention_heads(float *out, const float *Q, const float *K,
                                        const float *V, int seq_q, int seq_k,
                                        int n_heads, int n_kv_heads, int head_dim,
                                        float scale, int q_offset,
                                        int head_start, int head_end) {
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    for (int h = head_start; h < head_end; h++) {
        int kv_h = h / heads_per_kv;

        for (int i = 0; i < seq_q; i++) {
            const float *q_row = Q + i * q_hidden + h * head_dim;
            float *o_row = out + i * q_hidden + h * head_dim;
            int global_pos = q_offset + i;
            int k_end = global_pos + 1;
            if (k_end > seq_k) k_end = seq_k;

            float max_score = -1e30f;
            float sum_exp = 0.0f;
            for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

            for (int j = 0; j < k_end; j++) {
                const float *k_row = K + j * kv_hidden + kv_h * head_dim;
                const float *v_row = V + j * kv_hidden + kv_h * head_dim;

                float score = qwen_dot_f32(q_row, k_row, head_dim) * scale;

                if (score > max_score) {
                    float correction = expf(max_score - score);
                    sum_exp = sum_exp * correction + 1.0f;
                    qwen_vec_scale_add(o_row, v_row, correction, head_dim);
                    max_score = score;
                } else {
                    float wt = expf(score - max_score);
                    sum_exp += wt;
                    qwen_vec_axpy_inplace(o_row, v_row, wt, head_dim);
                }
            }

            if (sum_exp > 0.0f) {
                float inv_sum = 1.0f / sum_exp;
                qwen_vec_scale_inplace(o_row, inv_sum, head_dim);
            }
        }
    }
}

typedef struct {
    float *out;
    const float *Q;
    const float *K;
    const float *V;
    int seq_q, seq_k;
    int n_heads, n_kv_heads;
    int head_dim;
    float scale;
    int q_offset;
} causal_attn_task_t;

static void causal_attn_worker(int tid, int n_threads, void *arg) {
    causal_attn_task_t *t = (causal_attn_task_t *)arg;
    int chunk = (t->n_heads + n_threads - 1) / n_threads;
    int h0 = tid * chunk;
    int h1 = h0 + chunk;
    if (h1 > t->n_heads) h1 = t->n_heads;
    if (h0 >= h1) return;

    qwen_causal_attention_heads(t->out, t->Q, t->K, t->V,
                                t->seq_q, t->seq_k, t->n_heads, t->n_kv_heads,
                                t->head_dim, t->scale, t->q_offset, h0, h1);
}

void qwen_causal_attention(float *out, const float *Q, const float *K, const float *V,
                            int seq_q, int seq_k, int n_heads, int n_kv_heads,
                            int head_dim, float scale, int q_offset) {
    if (tp.n_threads > 1 && n_heads >= 2 && (seq_q >= 2 || seq_k >= 128)) {
        causal_attn_task_t task = {
            .out = out, .Q = Q, .K = K, .V = V,
            .seq_q = seq_q, .seq_k = seq_k,
            .n_heads = n_heads, .n_kv_heads = n_kv_heads,
            .head_dim = head_dim, .scale = scale, .q_offset = q_offset
        };
        parallel_for(causal_attn_worker, &task);
        return;
    }

    qwen_causal_attention_heads(out, Q, K, V,
                                seq_q, seq_k, n_heads, n_kv_heads,
                                head_dim, scale, q_offset, 0, n_heads);
}

/* ========================================================================
 * Position Embeddings
 * ======================================================================== */

void qwen_sinusoidal_pe(float *pe, int n_pos, int d_model) {
    int half = d_model / 2;
    float log_timescale = logf(10000.0f) / (float)(half - 1);

    for (int p = 0; p < n_pos; p++) {
        float *row = pe + p * d_model;
        for (int d = 0; d < half; d++) {
            float inv_timescale = expf(-(float)d * log_timescale);
            float angle = (float)p * inv_timescale;
            row[d] = sinf(angle);          /* first half: sin */
            row[half + d] = cosf(angle);   /* second half: cos */
        }
    }
}

void qwen_compute_rope_neox(float *cos_out, float *sin_out, const int *positions,
                              int seq, int head_dim, float theta) {
    int half = head_dim / 2;

    for (int s = 0; s < seq; s++) {
        float pos = (float)positions[s];
        for (int d = 0; d < half; d++) {
            float freq = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
            float angle = pos * freq;
            float c = cosf(angle);
            float sn = sinf(angle);
            /* Duplicate for full head_dim */
            cos_out[s * head_dim + d] = c;
            cos_out[s * head_dim + half + d] = c;
            sin_out[s * head_dim + d] = sn;
            sin_out[s * head_dim + half + d] = sn;
        }
    }
}

void qwen_apply_rope_neox(float *x, const float *cos_vals, const float *sin_vals,
                            int seq, int n_heads, int head_dim) {
    /*
     * NeoX split-half style:
     *   x1 = x[..., :half], x2 = x[..., half:]
     *   rotated = cat(-x2, x1)
     *   result = x * cos + rotated * sin
     */
    int half = head_dim / 2;
    int hidden = n_heads * head_dim;

    for (int s = 0; s < seq; s++) {
        const float *c = cos_vals + s * head_dim;
        const float *sn = sin_vals + s * head_dim;

        for (int h = 0; h < n_heads; h++) {
            float *vec = x + s * hidden + h * head_dim;

#if defined(__AVX512F__) && defined(__FMA__)
            int d = 0;
            for (; d + 16 <= half; d += 16) {
                __m512 x1 = _mm512_loadu_ps(vec + d);
                __m512 x2 = _mm512_loadu_ps(vec + half + d);
                /* RoPE cache duplicates cos/sin across halves. */
                __m512 cc = _mm512_loadu_ps(c + d);
                __m512 ss = _mm512_loadu_ps(sn + d);
                __m512 new1 = _mm512_fmsub_ps(x1, cc, _mm512_mul_ps(x2, ss));
                __m512 new2 = _mm512_fmadd_ps(x2, cc, _mm512_mul_ps(x1, ss));
                _mm512_storeu_ps(vec + d, new1);
                _mm512_storeu_ps(vec + half + d, new2);
            }
            for (; d < half; d++) {
                float x1 = vec[d];
                float x2 = vec[half + d];
                vec[d]        = x1 * c[d]        + (-x2) * sn[d];
                vec[half + d] = x2 * c[half + d] + x1 * sn[half + d];
            }
#elif defined(__AVX2__) && defined(__FMA__)
            int d = 0;
            for (; d + 8 <= half; d += 8) {
                __m256 x1 = _mm256_loadu_ps(vec + d);
                __m256 x2 = _mm256_loadu_ps(vec + half + d);
                __m256 cc = _mm256_loadu_ps(c + d);
                __m256 ss = _mm256_loadu_ps(sn + d);
                __m256 new1 = _mm256_fmsub_ps(x1, cc, _mm256_mul_ps(x2, ss));
                __m256 new2 = _mm256_fmadd_ps(x2, cc, _mm256_mul_ps(x1, ss));
                _mm256_storeu_ps(vec + d, new1);
                _mm256_storeu_ps(vec + half + d, new2);
            }
            for (; d < half; d++) {
                float x1 = vec[d];
                float x2 = vec[half + d];
                vec[d]        = x1 * c[d]        + (-x2) * sn[d];
                vec[half + d] = x2 * c[half + d] + x1 * sn[half + d];
            }
#else
            for (int d = 0; d < half; d++) {
                float x1 = vec[d];           /* first half */
                float x2 = vec[half + d];    /* second half */
                vec[d]        = x1 * c[d]        + (-x2) * sn[d];
                vec[half + d] = x2 * c[half + d] + x1 * sn[half + d];
            }
#endif
        }
    }
}
