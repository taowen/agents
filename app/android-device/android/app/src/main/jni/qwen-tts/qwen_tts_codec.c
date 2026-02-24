/*
 * qwen_tts_codec.c - Codec Decoder (Speech Tokenizer Decoder)
 *
 * Converts codec tokens to waveform:
 *   1. SplitResidualVectorQuantizer: dequantize tokens → continuous embeddings
 *   2. Pre-conv: CausalConv1d (codebook_dim → latent_dim, k=3)
 *   3. Transformer: 8-layer sliding-window transformer (latent → hidden → latent)
 *   4. Upsample: 2× (TransConv + ConvNeXt) stages
 *   5. Vocoder (BigVGAN): initial conv → 4 blocks × (SnakeBeta + TransConv + 3×ResUnit) → final conv
 *   6. Clamp to [-1, 1]
 *
 * Total upsampling: 2 × 2 × 8 × 5 × 4 × 3 = 1920×
 * At 12.5 Hz codec rate, produces 24000 Hz audio.
 */

#include "qwen_tts.h"
#include "qwen_tts_internal.h"
#include "qwen_tts_kernels.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

extern int qwen_tts_verbose;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static int codec_decoder_weights_ready(const qwen_tts_ctx_t *ctx) {
    const qwen_tts_config_t *cfg = &ctx->config;
    const qwen_tts_codec_decoder_t *codec = &ctx->codec;

    if (!codec->rvq.semantic_codebooks[0].cluster_usage ||
        !codec->rvq.semantic_codebooks[0].embedding_sum ||
        !codec->rvq.semantic_output_proj ||
        !codec->rvq.acoustic_output_proj ||
        !codec->pre_conv_weight ||
        !codec->transformer_input_proj_weight ||
        !codec->transformer_output_proj_weight ||
        !codec->vocoder_pre_conv_weight ||
        !codec->vocoder_final_conv_weight) {
        return 0;
    }

    for (int q = 0; q < cfg->codec_num_quantizers - 1; q++) {
        if (!codec->rvq.acoustic_codebooks[q].cluster_usage ||
            !codec->rvq.acoustic_codebooks[q].embedding_sum) {
            return 0;
        }
    }

    for (int i = 0; i < cfg->codec_layers; i++) {
        const qwen_tts_codec_transformer_layer_t *l = &codec->transformer_layers[i];
        if (!l->input_norm || !l->post_attn_norm || !l->wqkv_q8 ||
            !l->wo_q8 || !l->gate_up_q8 || !l->down_q8) {
            return 0;
        }
    }

    for (int s = 0; s < 2; s++) {
        const qwen_tts_convnext_block_t *cn = &codec->upsample_convnext[s];
        if (!codec->upsample_transconv_weight[s] || !codec->upsample_transconv_bias[s] ||
            !cn->dwconv_weight || !cn->norm_weight || !cn->norm_bias ||
            !cn->pwconv1_weight || !cn->pwconv1_bias ||
            !cn->pwconv2_weight || !cn->pwconv2_bias || !cn->gamma) {
            return 0;
        }
    }

    for (int b = 0; b < 4; b++) {
        const qwen_tts_vocoder_block_t *vb = &codec->vocoder_blocks[b];
        if (!vb->act_alpha || !vb->act_beta || !vb->transconv_weight || !vb->transconv_bias) {
            return 0;
        }
        for (int r = 0; r < 3; r++) {
            const qwen_tts_vocoder_resunit_t *ru = &vb->resunits[r];
            if (!ru->act1_alpha || !ru->act1_beta || !ru->conv1_weight || !ru->conv1_bias ||
                !ru->act2_alpha || !ru->act2_beta || !ru->conv2_weight || !ru->conv2_bias) {
                return 0;
            }
        }
    }

    return 1;
}

static inline float codec_dot(const float *a, const float *b, int n) {
#ifdef USE_BLAS
    return cblas_sdot(n, a, 1, b, 1);
#elif defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    acc0 = vaddq_f32(acc0, acc1);
    float sum = vaddvq_f32(acc0);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#else
    return kernel_dot(a, b, n);
#endif
}

static inline void codec_axpy(int n, float alpha, const float *x, float *y) {
#ifdef USE_BLAS
    cblas_saxpy(n, alpha, x, 1, y, 1);
#elif defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t va = vdupq_n_f32(alpha);
    int i = 0;
    for (; i + 3 < n; i += 4)
        vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), va, vld1q_f32(x + i)));
    for (; i < n; i++) y[i] += alpha * x[i];
#else
    for (int i = 0; i < n; i++) y[i] += alpha * x[i];
#endif
}

/* ========================================================================
 * RVQ Dequantization
 * ======================================================================== */

/*
 * Dequantize codes[time_steps][num_quantizers] into continuous embeddings.
 *
 * SplitResidualVectorQuantizer layout:
 *   semantic: 1 codebook (quantizer 0)
 *   acoustic: 15 codebooks (quantizers 1-15)
 *
 * For each VQ layer:
 *   embedding = embedding_sum / cluster_usage  (EuclideanCodebook)
 *   quantized = F.embedding(codes, embedding)
 *   quantized = project_out(quantized)  -- Linear if codebook_dim != dim, else Identity
 *   quantized = quantized.transpose(1, 2)
 *
 * Then: quantized = sum of all VQ layers → output_proj(Conv1d k=1)
 *
 * Output: [codebook_dim, time_steps]
 */

static float *codec_rvq_dequantize(qwen_tts_ctx_t *ctx, const int *codes,
                                     int time_steps, int num_quantizers) {
    qwen_tts_config_t *cfg = &ctx->config;
    qwen_tts_rvq_t *rvq = &ctx->codec.rvq;
    int codebook_size = cfg->codec_codebook_size;

    /* SplitResidualVectorQuantizer splits input into two halves:
     *   half_latent = latent_dim / 2 = 512
     *   vq_dim = codebook_dim / 2 = 256
     *   Each branch: input_proj(half_latent→vq_dim) → VQ(vq_dim) → output_proj(vq_dim→half_latent)
     *   Then concatenate semantic(512) + acoustic(512) = latent_dim(1024)
     */

    int latent_dim = cfg->codec_latent;           /* 1024 */
    int half_latent = latent_dim / 2;              /* 512 */
    int vq_dim = cfg->codec_codebook_dim / 2;     /* 256 */

    /* Allocate VQ-domain partial sums */
    float *semantic_sum = (float *)calloc((size_t)vq_dim * time_steps, sizeof(float));
    float *acoustic_sum = (float *)calloc((size_t)vq_dim * time_steps, sizeof(float));

    /* Process semantic codebook (quantizer 0) */
    {
        qwen_tts_codebook_t *cb = &rvq->semantic_codebooks[0];
        const float *embeddings = cb->embeddings;
        float *tmp_embeddings = NULL;
        if (!embeddings) {
            tmp_embeddings = (float *)malloc((size_t)codebook_size * vq_dim * sizeof(float));
            embeddings = tmp_embeddings;
            for (int c = 0; c < codebook_size; c++) {
                float usage = cb->cluster_usage[c];
                if (usage < 1e-5f) usage = 1e-5f;
                float inv_usage = 1.0f / usage;
                for (int d = 0; d < vq_dim; d++) {
                    tmp_embeddings[c * vq_dim + d] = cb->embedding_sum[c * vq_dim + d] * inv_usage;
                }
            }
        }

        /* Look up codes and sum */
        for (int t = 0; t < time_steps; t++) {
            int code = codes[t * num_quantizers + 0];
            if (code < 0) code = 0;
            if (code >= codebook_size) code = 0;
            for (int d = 0; d < vq_dim; d++) {
                semantic_sum[d * time_steps + t] += embeddings[code * vq_dim + d];
            }
        }
        free(tmp_embeddings);
    }

    /* Apply semantic output_proj: Conv1d(vq_dim, half_latent, 1, bias=False) */
    float *semantic_out = (float *)calloc((size_t)half_latent * time_steps, sizeof(float));
    if (rvq->semantic_output_proj) {
        /* output_proj is [half_latent, vq_dim, 1] pointwise conv = matmul per timestep */
        for (int t = 0; t < time_steps; t++) {
            for (int od = 0; od < half_latent; od++) {
                float sum = 0;
                for (int id = 0; id < vq_dim; id++) {
                    sum += rvq->semantic_output_proj[od * vq_dim + id] * semantic_sum[id * time_steps + t];
                }
                semantic_out[od * time_steps + t] = sum;
            }
        }
    } else {
        if (half_latent != vq_dim) {
            fprintf(stderr, "Error: missing semantic output projection for RVQ dequantization\n");
            free(semantic_sum); free(acoustic_sum);
            free(semantic_out);
            return NULL;
        }
        memcpy(semantic_out, semantic_sum, (size_t)half_latent * time_steps * sizeof(float));
    }

    /* Process acoustic codebooks (quantizers 1..num_quantizers-1) */
    for (int q = 1; q < num_quantizers; q++) {
        qwen_tts_codebook_t *cb = &rvq->acoustic_codebooks[q - 1];
        const float *embeddings = cb->embeddings;
        float *tmp_embeddings = NULL;
        if (!embeddings) {
            tmp_embeddings = (float *)malloc((size_t)codebook_size * vq_dim * sizeof(float));
            embeddings = tmp_embeddings;
            for (int c = 0; c < codebook_size; c++) {
                float usage = cb->cluster_usage[c];
                if (usage < 1e-5f) usage = 1e-5f;
                float inv_usage = 1.0f / usage;
                for (int d = 0; d < vq_dim; d++) {
                    tmp_embeddings[c * vq_dim + d] = cb->embedding_sum[c * vq_dim + d] * inv_usage;
                }
            }
        }
        for (int t = 0; t < time_steps; t++) {
            int code = codes[t * num_quantizers + q];
            if (code < 0) code = 0;
            if (code >= codebook_size) code = 0;
            for (int d = 0; d < vq_dim; d++) {
                acoustic_sum[d * time_steps + t] += embeddings[code * vq_dim + d];
            }
        }
        free(tmp_embeddings);
    }

    /* Apply acoustic output_proj */
    float *acoustic_out = (float *)calloc((size_t)half_latent * time_steps, sizeof(float));
    if (rvq->acoustic_output_proj) {
        for (int t = 0; t < time_steps; t++) {
            for (int od = 0; od < half_latent; od++) {
                float sum = 0;
                for (int id = 0; id < vq_dim; id++) {
                    sum += rvq->acoustic_output_proj[od * vq_dim + id] * acoustic_sum[id * time_steps + t];
                }
                acoustic_out[od * time_steps + t] = sum;
            }
        }
    } else {
        if (half_latent != vq_dim) {
            fprintf(stderr, "Error: missing acoustic output projection for RVQ dequantization\n");
            free(semantic_sum); free(acoustic_sum);
            free(semantic_out); free(acoustic_out);
            return NULL;
        }
        memcpy(acoustic_out, acoustic_sum, (size_t)half_latent * time_steps * sizeof(float));
    }

    /* Sum semantic + acoustic → [codebook_dim=512, time_steps] */
    float *output = (float *)malloc((size_t)half_latent * time_steps * sizeof(float));
    for (size_t i = 0; i < (size_t)half_latent * time_steps; i++) {
        output[i] = semantic_out[i] + acoustic_out[i];
    }

    free(semantic_sum); free(acoustic_sum);
    free(semantic_out); free(acoustic_out);

    return output; /* [codebook_dim, time_steps] in channels-first format */
}

/* ========================================================================
 * Codec Transformer (8 layers, sliding window attention, LayerScale)
 * ======================================================================== */

static void codec_transformer_forward(qwen_tts_ctx_t *ctx, float *hidden,
                                        int seq_len) {
    qwen_tts_config_t *cfg = &ctx->config;
    int codec_hidden = cfg->codec_hidden;
    int latent = cfg->codec_latent;
    int layers = cfg->codec_layers;
    int heads = cfg->codec_heads;
    int kv_heads = cfg->codec_kv_heads;
    int head_dim = codec_hidden / heads;
    int kv_dim = kv_heads * head_dim;
    int intermediate = cfg->codec_intermediate;
    int sliding_window = cfg->codec_sliding_window;
    int groups_per_head = heads / kv_heads;
    float eps = cfg->codec_rms_norm_eps;

    /* hidden comes as [seq_len, latent] format (already transposed from channels-first) */

    /* Input projection: latent_dim -> codec_hidden (batch GEMM) */
    float *x = (float *)malloc((size_t)seq_len * codec_hidden * sizeof(float));
    kernel_matmul_f32(
        x,
        hidden,
        ctx->codec.transformer_input_proj_weight,
        seq_len,
        codec_hidden,
        latent
    );
    if (ctx->codec.transformer_input_proj_bias) {
        for (int t = 0; t < seq_len; t++) {
            kernel_add_inplace(
                x + t * codec_hidden,
                ctx->codec.transformer_input_proj_bias,
                codec_hidden
            );
        }
    }

    /* Compute RoPE cache */
    float *rope_cos = (float *)malloc((size_t)seq_len * head_dim * sizeof(float));
    float *rope_sin = (float *)malloc((size_t)seq_len * head_dim * sizeof(float));
    {
        int half = head_dim / 2;
        float theta = 10000.0f;
        for (int pos = 0; pos < seq_len; pos++) {
            for (int i = 0; i < half; i++) {
                float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
                float angle = (float)pos * freq;
                rope_cos[pos * head_dim + i] = cosf(angle);
                rope_cos[pos * head_dim + i + half] = cosf(angle);
                rope_sin[pos * head_dim + i] = sinf(angle);
                rope_sin[pos * head_dim + i + half] = sinf(angle);
            }
        }
    }

    /* Scratch buffers */
    float *x_norm = (float *)malloc((size_t)seq_len * codec_hidden * sizeof(float));
    float *q_all = (float *)malloc((size_t)seq_len * heads * head_dim * sizeof(float));
    float *k_all = (float *)malloc((size_t)seq_len * kv_dim * sizeof(float));
    float *v_all = (float *)malloc((size_t)seq_len * kv_dim * sizeof(float));
    float *attn_out = (float *)malloc((size_t)seq_len * heads * head_dim * sizeof(float));
    float *attn_scores = (float *)malloc((size_t)seq_len * sizeof(float));
    float *gate_all = (float *)malloc((size_t)seq_len * intermediate * sizeof(float));

    for (int layer = 0; layer < layers; layer++) {
        qwen_tts_codec_transformer_layer_t *l = &ctx->codec.transformer_layers[layer];

        /* 1. Input RMSNorm */
        for (int t = 0; t < seq_len; t++)
            kernel_rms_norm(x_norm + t * codec_hidden, x + t * codec_hidden,
                           l->input_norm, codec_hidden, eps);

        /* 2. Q, K, V projections */
        {
            int q_dim = heads * head_dim;
            int total_rows = q_dim + kv_dim + kv_dim;
            float *qkv_tmp = (float *)malloc(total_rows * sizeof(float));
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            if (l->wqkv_f16) {
                for (int t = 0; t < seq_len; t++) {
                    kernel_matvec_f16w(qkv_tmp, l->wqkv_f16, x_norm + t * codec_hidden, total_rows, codec_hidden);
                    memcpy(q_all + (size_t)t * q_dim, qkv_tmp, q_dim * sizeof(float));
                    memcpy(k_all + (size_t)t * kv_dim, qkv_tmp + q_dim, kv_dim * sizeof(float));
                    memcpy(v_all + (size_t)t * kv_dim, qkv_tmp + q_dim + kv_dim, kv_dim * sizeof(float));
                }
            } else
#endif
            {
                int n_blocks = codec_hidden / QK8_0;
                for (int t = 0; t < seq_len; t++) {
                    block_q8_0 xn_q8[n_blocks];
                    kernel_quantize_x_q8(x_norm + t * codec_hidden, codec_hidden, xn_q8);
                    kernel_matvec_q8(qkv_tmp, l->wqkv_q8, xn_q8, total_rows, n_blocks);
                    memcpy(q_all + (size_t)t * q_dim, qkv_tmp, q_dim * sizeof(float));
                    memcpy(k_all + (size_t)t * kv_dim, qkv_tmp + q_dim, kv_dim * sizeof(float));
                    memcpy(v_all + (size_t)t * kv_dim, qkv_tmp + q_dim + kv_dim, kv_dim * sizeof(float));
                }
            }
            free(qkv_tmp);
        }

        /* 3. RoPE (standard, not M-RoPE. No QK-Norm for codec decoder) */
        for (int t = 0; t < seq_len; t++) {
            kernel_rope_apply(q_all + t * heads * head_dim, NULL,
                             rope_cos + t * head_dim, rope_sin + t * head_dim,
                             heads, head_dim);
            kernel_rope_apply(k_all + t * kv_dim, NULL,
                             rope_cos + t * head_dim, rope_sin + t * head_dim,
                             kv_heads, head_dim);
        }

        /* 4. Sliding-window causal attention */
        float scale = 1.0f / sqrtf((float)head_dim);
        memset(attn_out, 0, (size_t)seq_len * heads * head_dim * sizeof(float));

        for (int h = 0; h < heads; h++) {
            int kv_h = h / groups_per_head;
            for (int qi = 0; qi < seq_len; qi++) {
                float *qh = q_all + qi * heads * head_dim + h * head_dim;
                int start = qi - sliding_window + 1;
                if (start < 0) start = 0;

                /* Compute attention scores within window */
                int wlen = qi - start + 1;
                for (int i = 0; i < wlen; i++) {
                    int ki = start + i;
                    float *kh = k_all + ki * kv_dim + kv_h * head_dim;
                    attn_scores[i] = codec_dot(qh, kh, head_dim) * scale;
                }
                kernel_softmax(attn_scores, wlen);

                float *oh = attn_out + qi * heads * head_dim + h * head_dim;
                for (int i = 0; i < wlen; i++) {
                    int ki = start + i;
                    float w = attn_scores[i];
                    float *vh = v_all + ki * kv_dim + kv_h * head_dim;
                    codec_axpy(head_dim, w, vh, oh);
                }
            }
        }

        /* 5. Output projection + LayerScale + residual */
        {
            int q_dim = heads * head_dim;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            if (l->wo_f16) {
                for (int t = 0; t < seq_len; t++)
                    kernel_matvec_f16w(x_norm + t * codec_hidden, l->wo_f16,
                                       attn_out + (size_t)t * q_dim, codec_hidden, q_dim);
            } else
#endif
            {
                int n_blocks = q_dim / QK8_0;
                for (int t = 0; t < seq_len; t++) {
                    block_q8_0 attn_q8[n_blocks];
                    kernel_quantize_x_q8(attn_out + (size_t)t * q_dim, q_dim, attn_q8);
                    kernel_matvec_q8(x_norm + t * codec_hidden, l->wo_q8, attn_q8, codec_hidden, n_blocks);
                }
            }
        }
        for (int t = 0; t < seq_len; t++) {
            if (l->attn_layer_scale)
                kernel_mul_inplace(x_norm + t * codec_hidden, l->attn_layer_scale, codec_hidden);
            kernel_add_inplace(x + t * codec_hidden, x_norm + t * codec_hidden, codec_hidden);
        }

        /* 6. Post-attention norm + SwiGLU MLP + LayerScale */
        for (int t = 0; t < seq_len; t++)
            kernel_rms_norm(x_norm + t * codec_hidden, x + t * codec_hidden,
                           l->post_attn_norm, codec_hidden, eps);

        /* SwiGLU MLP + down projection */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        if (l->gate_up_f16 && l->down_f16) {
            float *gu_tmp = (float *)malloc(2 * intermediate * sizeof(float));
            for (int t = 0; t < seq_len; t++) {
                kernel_matvec_f16w(gu_tmp, l->gate_up_f16, x_norm + t * codec_hidden,
                                   2 * intermediate, codec_hidden);
                float *g_out = gate_all + (size_t)t * intermediate;
                for (int i = 0; i < intermediate; i++) {
                    float g = gu_tmp[i];
                    g_out[i] = (g / (1.0f + expf(-g))) * gu_tmp[intermediate + i];
                }
                kernel_matvec_f16w(x_norm + t * codec_hidden, l->down_f16,
                                   g_out, codec_hidden, intermediate);
            }
            free(gu_tmp);
        } else
#endif
        if (l->gate_up_f32 && l->down_f32) {
            float *gu_tmp = (float *)malloc(2 * intermediate * sizeof(float));
            for (int t = 0; t < seq_len; t++) {
                kernel_matvec_f32(gu_tmp, l->gate_up_f32, x_norm + t * codec_hidden,
                                  2 * intermediate, codec_hidden);
                float *g_out = gate_all + (size_t)t * intermediate;
                for (int i = 0; i < intermediate; i++) {
                    float g = gu_tmp[i];
                    g_out[i] = (g / (1.0f + expf(-g))) * gu_tmp[intermediate + i];
                }
                kernel_matvec_f32(x_norm + t * codec_hidden, l->down_f32,
                                  g_out, codec_hidden, intermediate);
            }
            free(gu_tmp);
        } else {
            int n_blocks_h = codec_hidden / QK8_0;
            int n_blocks_i = intermediate / QK8_0;
            for (int t = 0; t < seq_len; t++) {
                block_q8_0 xn_q8[n_blocks_h];
                kernel_quantize_x_q8(x_norm + t * codec_hidden, codec_hidden, xn_q8);
                kernel_swiglu_matvec_q8(gate_all + (size_t)t * intermediate,
                                         l->gate_up_q8, xn_q8, intermediate, n_blocks_h);
                block_q8_0 gate_q8[n_blocks_i];
                kernel_quantize_x_q8(gate_all + (size_t)t * intermediate, intermediate, gate_q8);
                kernel_matvec_q8(x_norm + t * codec_hidden, l->down_q8, gate_q8, codec_hidden, n_blocks_i);
            }
        }

        for (int t = 0; t < seq_len; t++) {
            /* LayerScale */
            if (l->mlp_layer_scale)
                kernel_mul_inplace(x_norm + t * codec_hidden, l->mlp_layer_scale, codec_hidden);
            kernel_add_inplace(x + t * codec_hidden, x_norm + t * codec_hidden, codec_hidden);
        }
    }

    /* Final norm */
    if (ctx->codec.transformer_norm) {
        for (int t = 0; t < seq_len; t++)
            kernel_rms_norm_inplace(x + t * codec_hidden, ctx->codec.transformer_norm, codec_hidden, eps);
    }

    /* Output projection: codec_hidden -> latent_dim (batch GEMM) */
    kernel_matmul_f32(
        hidden,
        x,
        ctx->codec.transformer_output_proj_weight,
        seq_len,
        latent,
        codec_hidden
    );
    if (ctx->codec.transformer_output_proj_bias) {
        for (int t = 0; t < seq_len; t++) {
            kernel_add_inplace(
                hidden + t * latent,
                ctx->codec.transformer_output_proj_bias,
                latent
            );
        }
    }

    free(x); free(x_norm); free(q_all); free(k_all); free(v_all);
    free(attn_out); free(attn_scores); free(gate_all);
    free(rope_cos); free(rope_sin);
}

/* ========================================================================
 * ConvNeXt upsampling block
 * ======================================================================== */

static void codec_convnext_forward(qwen_tts_convnext_block_t *block,
                                     float *hidden, int dim, int *length) {
    int len = *length;

    /* Residual */
    float *residual = (float *)malloc((size_t)dim * len * sizeof(float));
    memcpy(residual, hidden, (size_t)dim * len * sizeof(float));

    /* Depthwise causal conv (k=7, groups=dim) */
    float *conv_out = (float *)malloc((size_t)dim * len * sizeof(float));
    kernel_causal_conv1d(conv_out, hidden, block->dwconv_weight, block->dwconv_bias,
                         dim, dim, 7, len, 1, dim);

    /* permute to [len, dim] for LayerNorm and pointwise ops */
    float *x_ld = (float *)malloc((size_t)len * dim * sizeof(float));
    for (int c = 0; c < dim; c++)
        for (int t = 0; t < len; t++)
            x_ld[t * dim + c] = conv_out[c * len + t];

    /* LayerNorm */
    for (int t = 0; t < len; t++)
        kernel_layer_norm(x_ld + t * dim, x_ld + t * dim, block->norm_weight, block->norm_bias, dim, 1e-6f);

    /* pwconv1: [dim] → [4*dim] */
    int dim4 = 4 * dim;
    float *pw1 = (float *)malloc((size_t)len * dim4 * sizeof(float));
    for (int t = 0; t < len; t++) {
        kernel_matvec_f32(pw1 + t * dim4, block->pwconv1_weight, x_ld + t * dim, dim4, dim);
        if (block->pwconv1_bias)
            kernel_add_inplace(pw1 + t * dim4, block->pwconv1_bias, dim4);
    }

    /* GELU */
    kernel_gelu_inplace(pw1, len * dim4);

    /* pwconv2: [4*dim] → [dim] */
    for (int t = 0; t < len; t++) {
        kernel_matvec_f32(x_ld + t * dim, block->pwconv2_weight, pw1 + t * dim4, dim, dim4);
        if (block->pwconv2_bias)
            kernel_add_inplace(x_ld + t * dim, block->pwconv2_bias, dim);
    }

    /* Apply gamma (learnable residual scale) */
    for (int t = 0; t < len; t++)
        kernel_mul_inplace(x_ld + t * dim, block->gamma, dim);

    /* permute back to [dim, len] */
    for (int c = 0; c < dim; c++)
        for (int t = 0; t < len; t++)
            hidden[c * len + t] = x_ld[t * dim + c];

    /* Skip connection */
    kernel_add_inplace(hidden, residual, dim * len);

    free(residual); free(conv_out); free(x_ld); free(pw1);
}

/* ========================================================================
 * Vocoder residual unit
 * ======================================================================== */

typedef struct {
    float *residual;
    float *conv1_out;
    size_t cap;
} vocoder_resunit_scratch_t;

static int ensure_vocoder_resunit_scratch(vocoder_resunit_scratch_t *s, size_t n) {
    if (s->cap >= n) return 0;

    float *p = (float *)realloc(s->residual, n * sizeof(float));
    if (!p) return -1;
    s->residual = p;

    p = (float *)realloc(s->conv1_out, n * sizeof(float));
    if (!p) return -1;
    s->conv1_out = p;

    s->cap = n;
    return 0;
}

static int vocoder_resunit_forward(qwen_tts_vocoder_resunit_t *unit,
                                      float *hidden, int dim, int length, int dilation,
                                      vocoder_resunit_scratch_t *scratch,
                                      double *out_snake_ms, double *out_conv7_ms,
                                      double *out_conv1_ms, double *out_resadd_ms) {
    size_t n = (size_t)dim * length;
    if (ensure_vocoder_resunit_scratch(scratch, n) != 0) return -1;
    float *residual = scratch->residual;
    float *conv1_out = scratch->conv1_out;
    memcpy(residual, hidden, n * sizeof(float));

    double t0, t1;

    /* SnakeBeta activation 1 (in-place) */
    t0 = now_ms();
    kernel_snake_beta(hidden, hidden, unit->act1_alpha, unit->act1_beta, dim, length);
    t1 = now_ms();
    if (out_snake_ms) *out_snake_ms += t1 - t0;

    /* Causal conv1 (k=7, dilation) */
    t0 = now_ms();
    kernel_causal_conv1d(conv1_out, hidden, unit->conv1_weight, unit->conv1_bias,
                         dim, dim, 7, length, dilation, 1);
    t1 = now_ms();
    if (out_conv7_ms) *out_conv7_ms += t1 - t0;

    /* SnakeBeta activation 2 (in-place) */
    t0 = now_ms();
    kernel_snake_beta(conv1_out, conv1_out, unit->act2_alpha, unit->act2_beta, dim, length);
    t1 = now_ms();
    if (out_snake_ms) *out_snake_ms += t1 - t0;

    /* Causal conv2 (k=1, dilation=1) */
    t0 = now_ms();
    kernel_causal_conv1d(hidden, conv1_out, unit->conv2_weight, unit->conv2_bias,
                         dim, dim, 1, length, 1, 1);
    t1 = now_ms();
    if (out_conv1_ms) *out_conv1_ms += t1 - t0;

    /* Skip connection */
    t0 = now_ms();
    kernel_add_inplace(hidden, residual, dim * length);
    t1 = now_ms();
    if (out_resadd_ms) *out_resadd_ms += t1 - t0;
    return 0;
}

/* ========================================================================
 * Full codec decode pipeline
 * ======================================================================== */

float *qwen_tts_codec_decode(qwen_tts_ctx_t *ctx, const int *codes,
                              int time_steps, int *out_samples) {
    if (!ctx || !codes || !out_samples || time_steps <= 0) {
        if (out_samples) *out_samples = 0;
        return NULL;
    }
    if (!codec_decoder_weights_ready(ctx)) {
        fprintf(stderr, "Error: codec decoder is not fully loaded; cannot decode audio\n");
        *out_samples = 0;
        return NULL;
    }

    qwen_tts_config_t *cfg = &ctx->config;
    int num_quantizers = cfg->codec_num_quantizers;
    int latent_dim = cfg->codec_latent;
    int codebook_dim = cfg->codec_codebook_dim;

    if (qwen_tts_verbose >= 1)
        fprintf(stderr, "Codec decode: %d timesteps, %d quantizers\n", time_steps, num_quantizers);

    double stage_t0 = now_ms();
    double stage_rvq_ms = 0.0;
    double stage_preconv_ms = 0.0;
    double stage_transformer_ms = 0.0;
    double stage_upsample_ms = 0.0;
    double stage_vocoder_ms = 0.0;

    /* 1. RVQ Dequantize → [codebook_dim, time_steps] */
    float *hidden = codec_rvq_dequantize(ctx, codes, time_steps, num_quantizers);
    if (!hidden) {
        *out_samples = 0;
        return NULL;
    }
    stage_rvq_ms = now_ms() - stage_t0;

    /* 2. Pre-conv: CausalConv1d(codebook_dim=512, latent_dim=1024, k=3) */
    stage_t0 = now_ms();
    float *pre_conv_out = (float *)malloc((size_t)latent_dim * time_steps * sizeof(float));
    kernel_causal_conv1d(pre_conv_out, hidden, ctx->codec.pre_conv_weight,
                         ctx->codec.pre_conv_bias, codebook_dim, latent_dim,
                         3, time_steps, 1, 1);
    free(hidden);
    stage_preconv_ms = now_ms() - stage_t0;

    /* 3. Transpose to [time_steps, latent_dim] for transformer */
    float *hidden_seq = (float *)malloc((size_t)time_steps * latent_dim * sizeof(float));
    for (int c = 0; c < latent_dim; c++)
        for (int t = 0; t < time_steps; t++)
            hidden_seq[t * latent_dim + c] = pre_conv_out[c * time_steps + t];
    free(pre_conv_out);

    /* 4. Transformer forward pass */
    stage_t0 = now_ms();
    codec_transformer_forward(ctx, hidden_seq, time_steps);
    stage_transformer_ms = now_ms() - stage_t0;

    /* 5. Transpose back to [latent_dim, time_steps] */
    hidden = (float *)malloc((size_t)latent_dim * time_steps * sizeof(float));
    for (int c = 0; c < latent_dim; c++)
        for (int t = 0; t < time_steps; t++)
            hidden[c * time_steps + t] = hidden_seq[t * latent_dim + c];
    free(hidden_seq);

    /* 6. Upsample stages (2× TransConv + ConvNeXt) */
    stage_t0 = now_ms();
    int current_len = time_steps;
    for (int stage = 0; stage < 2; stage++) {
        int factor = cfg->codec_upsampling_ratios[stage];
        int new_len;

        /* TransposedConv1d upsample */
        float *up_out = (float *)malloc((size_t)latent_dim * (current_len * factor + factor) * sizeof(float));
        kernel_transposed_conv1d(up_out, hidden, ctx->codec.upsample_transconv_weight[stage],
                                  ctx->codec.upsample_transconv_bias[stage],
                                  latent_dim, latent_dim, factor, factor, current_len, &new_len);
        free(hidden);
        hidden = (float *)realloc(up_out, (size_t)latent_dim * new_len * sizeof(float));
        current_len = new_len;

        /* ConvNeXt block */
        codec_convnext_forward(&ctx->codec.upsample_convnext[stage], hidden, latent_dim, &current_len);
    }
    stage_upsample_ms = now_ms() - stage_t0;

    /* 7. Vocoder */
    stage_t0 = now_ms();

    /* 7a. Pre-allocate vocoder buffers: compute max size across all blocks */
    int decoder_dim = cfg->codec_decoder_dim;
    size_t voc_max_buf = (size_t)decoder_dim * current_len;
    {
        int sim_dim = decoder_dim;
        int sim_len = current_len;
        for (int b = 0; b < 4; b++) {
            int od = sim_dim / 2;
            int rate = cfg->codec_upsample_rates[b];
            int k = 2 * rate;
            int raw = (sim_len - 1) * rate + k;
            int nlen = raw - (k - rate);
            if (nlen < 0) nlen = 0;
            size_t need = (size_t)od * ((size_t)sim_len * rate + k);
            if (need > voc_max_buf) voc_max_buf = need;
            need = (size_t)od * nlen;
            if (need > voc_max_buf) voc_max_buf = need;
            sim_dim = od;
            sim_len = nlen;
        }
    }

    int upsample_rates[4] = {
        cfg->codec_upsample_rates[0],
        cfg->codec_upsample_rates[1],
        cfg->codec_upsample_rates[2],
        cfg->codec_upsample_rates[3],
    };
    int current_dim = decoder_dim;

    double voc_total_snake_ms = 0.0, voc_total_transconv_ms = 0.0;
    double voc_total_conv7_ms = 0.0, voc_total_conv1_ms = 0.0, voc_total_resadd_ms = 0.0;

    float *wav;

    /* F32 vocoder pipeline */
    {
        hidden = (float *)realloc(hidden, voc_max_buf * sizeof(float));
        float *voc_buf_b = (float *)malloc(voc_max_buf * sizeof(float));

        kernel_causal_conv1d(voc_buf_b, hidden, ctx->codec.vocoder_pre_conv_weight,
                             ctx->codec.vocoder_pre_conv_bias,
                             latent_dim, decoder_dim, 7, current_len, 1, 1);
        float *voc = voc_buf_b;
        float *voc_alt = hidden;

        vocoder_resunit_scratch_t ru_scratch = {0};

        for (int block = 0; block < 4; block++) {
            int in_dim = current_dim;
            int out_dim = in_dim / 2;
            int rate = upsample_rates[block];
            qwen_tts_vocoder_block_t *vb = &ctx->codec.vocoder_blocks[block];

            double blk_snake_ms = 0.0, blk_transconv_ms = 0.0;
            double blk_conv7_ms = 0.0, blk_conv1_ms = 0.0, blk_resadd_ms = 0.0;
            double t0;

            t0 = now_ms();
            kernel_snake_beta(voc, voc, vb->act_alpha, vb->act_beta, in_dim, current_len);
            blk_snake_ms += now_ms() - t0;

            int new_len;
            int ks = 2 * rate;
            t0 = now_ms();
            kernel_transposed_conv1d(voc_alt, voc, vb->transconv_weight, vb->transconv_bias,
                                      in_dim, out_dim, ks, rate, current_len, &new_len);
            blk_transconv_ms += now_ms() - t0;

            { float *tmp = voc; voc = voc_alt; voc_alt = tmp; }
            current_len = new_len;
            current_dim = out_dim;

            int dilations[3] = {1, 3, 9};
            for (int ru = 0; ru < 3; ru++) {
                if (vocoder_resunit_forward(&vb->resunits[ru], voc, current_dim, current_len,
                                            dilations[ru], &ru_scratch,
                                            &blk_snake_ms, &blk_conv7_ms,
                                            &blk_conv1_ms, &blk_resadd_ms) != 0) {
                    free(voc); free(voc_alt);
                    free(ru_scratch.residual); free(ru_scratch.conv1_out);
                    *out_samples = 0;
                    return NULL;
                }
            }

            if (qwen_tts_verbose >= 1) {
                fprintf(stderr, "  Vocoder block %d [%d->%d, len %d]: snake=%.1f transconv=%.1f conv7=%.1f conv1=%.1f resadd=%.1f ms\n",
                        block, in_dim, out_dim, current_len,
                        blk_snake_ms, blk_transconv_ms, blk_conv7_ms, blk_conv1_ms, blk_resadd_ms);
            }
            voc_total_snake_ms += blk_snake_ms;
            voc_total_transconv_ms += blk_transconv_ms;
            voc_total_conv7_ms += blk_conv7_ms;
            voc_total_conv1_ms += blk_conv1_ms;
            voc_total_resadd_ms += blk_resadd_ms;
        }

        kernel_snake_beta(voc, voc, ctx->codec.vocoder_final_act_alpha,
                          ctx->codec.vocoder_final_act_beta, current_dim, current_len);

        wav = (float *)malloc((size_t)current_len * sizeof(float));
        kernel_causal_conv1d(wav, voc, ctx->codec.vocoder_final_conv_weight,
                             ctx->codec.vocoder_final_conv_bias,
                             current_dim, 1, 7, current_len, 1, 1);
        free(voc); free(voc_alt);
        free(ru_scratch.residual); free(ru_scratch.conv1_out);
    }

    /* 8. Clamp to [-1, 1] */
    kernel_clamp(wav, current_len, -1.0f, 1.0f);
    stage_vocoder_ms = now_ms() - stage_t0;

    *out_samples = current_len;

    if (qwen_tts_verbose >= 1)
        fprintf(stderr, "Codec decode complete: %d samples (%.2f seconds)\n",
                current_len, (float)current_len / QWEN_TTS_SAMPLE_RATE);
    if (qwen_tts_verbose >= 1) {
        fprintf(stderr, "Codec stages (ms): rvq=%.1f preconv=%.1f transformer=%.1f upsample=%.1f vocoder=%.1f\n",
                stage_rvq_ms, stage_preconv_ms, stage_transformer_ms, stage_upsample_ms, stage_vocoder_ms);
        fprintf(stderr, "Vocoder totals (ms): snake=%.1f transconv=%.1f conv7=%.1f conv1=%.1f resadd=%.1f\n",
                voc_total_snake_ms, voc_total_transconv_ms, voc_total_conv7_ms, voc_total_conv1_ms, voc_total_resadd_ms);
    }

    return wav;
}

/* ========================================================================
 * Incremental Codec Decode
 *
 * Process one codec token at a time, maintaining causal conv states,
 * transformer KV cache, and transposed conv overlap buffers.
 * Each token produces exactly 1920 PCM samples (80ms at 24kHz).
 * ======================================================================== */

/* --------------------------------------------------------------------
 * Incremental CausalConv1d
 *
 * Strategy: prepend state + call batch kernel + extract last N_new outputs.
 * The batch kernel adds (K-1)*D zeros on the left internally, so:
 *   effective input = [zeros, state, new_input]
 *   output positions [state_len .. state_len+N_new-1] are what we need.
 * -------------------------------------------------------------------- */

static void codec_causal_conv_incremental(
    float *out,          /* [out_ch, N_new] */
    const float *input,  /* [in_ch, N_new] */
    float *state,        /* [in_ch, state_len] - updated in-place */
    const float *weight, const float *bias,
    int in_ch, int out_ch, int K, int dilation, int groups, int N_new)
{
    int state_len = (K - 1) * dilation;
    int combined_len = state_len + N_new;

    /* 1. Concatenate: combined = [state, new_input] per channel */
    float *combined = (float *)malloc((size_t)in_ch * combined_len * sizeof(float));
    for (int c = 0; c < in_ch; c++) {
        memcpy(&combined[c * combined_len],
               &state[c * state_len],
               state_len * sizeof(float));
        memcpy(&combined[c * combined_len + state_len],
               &input[c * N_new],
               N_new * sizeof(float));
    }

    /* 2. Run existing batch kernel on the combined input */
    float *full_out = (float *)malloc((size_t)out_ch * combined_len * sizeof(float));
    kernel_causal_conv1d(full_out, combined, weight, bias,
                         in_ch, out_ch, K, combined_len, dilation, groups);

    /* 3. Extract last N_new output positions (skip the first state_len garbage) */
    for (int c = 0; c < out_ch; c++)
        memcpy(&out[c * N_new],
               &full_out[c * combined_len + state_len],
               N_new * sizeof(float));

    /* 4. Update state = last state_len positions of combined input */
    for (int c = 0; c < in_ch; c++)
        memcpy(&state[c * state_len],
               &combined[c * combined_len + N_new],
               state_len * sizeof(float));

    free(combined);
    free(full_out);
}

/* --------------------------------------------------------------------
 * Incremental TransposedConv1d (with overlap-add)
 *
 * For vocoder upsampling where K = 2 * stride, adjacent inputs
 * overlap by (K - stride) positions. We maintain an overlap buffer.
 * -------------------------------------------------------------------- */

static void codec_transconv_incremental(
    float *out,           /* [out_ch, N_new * stride] */
    const float *input,   /* [in_ch, N_new] */
    float *overlap,       /* [out_ch, overlap_len] - updated in-place */
    const float *weight,  /* [in_ch, out_ch, K] */
    const float *bias,
    int in_ch, int out_ch, int K, int stride, int N_new)
{
    int overlap_len = K - stride;
    int emit_len = N_new * stride;
    int raw_len = (N_new > 0) ? (N_new - 1) * stride + K : 0;

    if (raw_len <= 0) return;

    /* 1. Compute raw transposed conv output (no trim) */
    float *raw = (float *)calloc((size_t)out_ch * raw_len, sizeof(float));
    for (int ic = 0; ic < in_ch; ic++) {
        for (int t = 0; t < N_new; t++) {
            float val = input[ic * N_new + t];
            int base = t * stride;
            for (int oc = 0; oc < out_ch; oc++) {
                const float *w = weight + (size_t)ic * out_ch * K + (size_t)oc * K;
                float *r = raw + (size_t)oc * raw_len + base;
                for (int k = 0; k < K; k++) {
                    r[k] += val * w[k];
                }
            }
        }
    }

    /* 2. Add overlap from previous call to the first overlap_len positions */
    for (int c = 0; c < out_ch; c++)
        for (int i = 0; i < overlap_len && i < raw_len; i++)
            raw[c * raw_len + i] += overlap[c * overlap_len + i];

    /* 3. Add bias to first emit_len positions */
    if (bias) {
        for (int c = 0; c < out_ch; c++) {
            float b = bias[c];
            for (int t = 0; t < emit_len && t < raw_len; t++)
                raw[c * raw_len + t] += b;
        }
    }

    /* 4. Output first emit_len positions */
    for (int c = 0; c < out_ch; c++)
        memcpy(&out[c * emit_len],
               &raw[c * raw_len],
               emit_len * sizeof(float));

    /* 5. Save new overlap (tail positions, without bias added) */
    for (int c = 0; c < out_ch; c++) {
        for (int i = 0; i < overlap_len; i++) {
            if (emit_len + i < raw_len) {
                float v = raw[c * raw_len + emit_len + i];
                /* Subtract bias that was added in step 3 (if it reached this position) */
                if (bias && emit_len + i < emit_len) v -= bias[c];
                overlap[c * overlap_len + i] = v;
            } else {
                overlap[c * overlap_len + i] = 0.0f;
            }
        }
    }

    free(raw);
}

/* --------------------------------------------------------------------
 * Codec Transformer single-token forward pass
 *
 * Processes one position through 8 transformer layers using KV cache.
 * Standard RoPE (not M-RoPE), LayerScale, sliding window attention.
 * -------------------------------------------------------------------- */

static void codec_transformer_step(qwen_tts_ctx_t *ctx,
                                     float *hidden_io, /* [latent] in/out */
                                     int pos)
{
    qwen_tts_config_t *cfg = &ctx->config;
    int codec_hidden = cfg->codec_hidden;
    int latent = cfg->codec_latent;
    int layers = cfg->codec_layers;
    int heads = cfg->codec_heads;
    int kv_heads = cfg->codec_kv_heads;
    int head_dim = codec_hidden / heads;
    int kv_dim = kv_heads * head_dim;
    int intermediate = cfg->codec_intermediate;
    int sliding_window = cfg->codec_sliding_window;
    int groups_per_head = heads / kv_heads;
    float eps = cfg->codec_rms_norm_eps;

    /* Ensure KV cache is large enough */
    int needed = pos + 1;
    if (needed > ctx->codec_kv_max) {
        int new_max = needed + 256;
        size_t kv_size = (size_t)layers * new_max * kv_dim * sizeof(float);
        ctx->codec_kv_k = (float *)realloc(ctx->codec_kv_k, kv_size);
        ctx->codec_kv_v = (float *)realloc(ctx->codec_kv_v, kv_size);
        ctx->codec_kv_max = new_max;
    }

    /* Input projection: latent → codec_hidden */
    float *x = (float *)malloc(codec_hidden * sizeof(float));
    kernel_matvec_f32(x, ctx->codec.transformer_input_proj_weight,
                       hidden_io, codec_hidden, latent);
    if (ctx->codec.transformer_input_proj_bias)
        kernel_add_inplace(x, ctx->codec.transformer_input_proj_bias, codec_hidden);

    /* Compute RoPE for this position */
    float rope_cos[512], rope_sin[512]; /* max head_dim */
    {
        int half = head_dim / 2;
        float theta = 10000.0f;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
            float angle = (float)pos * freq;
            rope_cos[i] = cosf(angle);
            rope_cos[i + half] = cosf(angle);
            rope_sin[i] = sinf(angle);
            rope_sin[i + half] = sinf(angle);
        }
    }

    /* Scratch buffers */
    float *x_norm = (float *)malloc(codec_hidden * sizeof(float));
    int q_dim = heads * head_dim;
    float *q_buf = (float *)malloc(q_dim * sizeof(float));
    float *k_buf = (float *)malloc(kv_dim * sizeof(float));
    float *v_buf = (float *)malloc(kv_dim * sizeof(float));
    float *attn_out = (float *)malloc(q_dim * sizeof(float));
    int max_wlen = (pos + 1 < sliding_window) ? pos + 1 : sliding_window;
    float *scores = (float *)malloc(max_wlen * sizeof(float));
    float *gate_buf = (float *)malloc(intermediate * sizeof(float));

    for (int layer = 0; layer < layers; layer++) {
        qwen_tts_codec_transformer_layer_t *l = &ctx->codec.transformer_layers[layer];
        size_t kv_stride = (size_t)ctx->codec_kv_max * kv_dim;

        /* 1. Input RMSNorm */
        kernel_rms_norm(x_norm, x, l->input_norm, codec_hidden, eps);

        /* 2. QKV projections */
        {
            int total_rows = q_dim + kv_dim + kv_dim;
            float *qkv_tmp = (float *)malloc(total_rows * sizeof(float));
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            if (l->wqkv_f16) {
                kernel_matvec_f16w(qkv_tmp, l->wqkv_f16, x_norm, total_rows, codec_hidden);
            } else
#endif
            {
                int n_blocks = codec_hidden / QK8_0;
                block_q8_0 xn_q8[n_blocks];
                kernel_quantize_x_q8(x_norm, codec_hidden, xn_q8);
                kernel_matvec_q8(qkv_tmp, l->wqkv_q8, xn_q8, total_rows, n_blocks);
            }
            memcpy(q_buf, qkv_tmp, q_dim * sizeof(float));
            memcpy(k_buf, qkv_tmp + q_dim, kv_dim * sizeof(float));
            memcpy(v_buf, qkv_tmp + q_dim + kv_dim, kv_dim * sizeof(float));
            free(qkv_tmp);
        }

        /* 3. Standard RoPE (no QK-Norm for codec) */
        kernel_rope_apply(q_buf, NULL, rope_cos, rope_sin, heads, head_dim);
        kernel_rope_apply(k_buf, NULL, rope_cos, rope_sin, kv_heads, head_dim);

        /* 4. Store K, V in cache */
        memcpy(ctx->codec_kv_k + layer * kv_stride + (size_t)pos * kv_dim,
               k_buf, kv_dim * sizeof(float));
        memcpy(ctx->codec_kv_v + layer * kv_stride + (size_t)pos * kv_dim,
               v_buf, kv_dim * sizeof(float));

        /* 5. Sliding-window causal attention (single query) */
        float scale = 1.0f / sqrtf((float)head_dim);
        int start = pos - sliding_window + 1;
        if (start < 0) start = 0;
        int wlen = pos - start + 1;

        for (int h = 0; h < heads; h++) {
            int kv_h = h / groups_per_head;
            float *qh = q_buf + h * head_dim;
            float *oh = attn_out + h * head_dim;
            memset(oh, 0, head_dim * sizeof(float));

            for (int i = 0; i < wlen; i++) {
                int ki = start + i;
                float *kh = ctx->codec_kv_k + layer * kv_stride + (size_t)ki * kv_dim + kv_h * head_dim;
                scores[i] = codec_dot(qh, kh, head_dim) * scale;
            }
            kernel_softmax(scores, wlen);

            for (int i = 0; i < wlen; i++) {
                int ki = start + i;
                float *vh = ctx->codec_kv_v + layer * kv_stride + (size_t)ki * kv_dim + kv_h * head_dim;
                codec_axpy(head_dim, scores[i], vh, oh);
            }
        }

        /* 6. Output projection + LayerScale + residual */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        if (l->wo_f16) {
            kernel_matvec_f16w(x_norm, l->wo_f16, attn_out, codec_hidden, q_dim);
        } else
#endif
        {
            int n_blocks = q_dim / QK8_0;
            block_q8_0 attn_q8[n_blocks];
            kernel_quantize_x_q8(attn_out, q_dim, attn_q8);
            kernel_matvec_q8(x_norm, l->wo_q8, attn_q8, codec_hidden, n_blocks);
        }
        if (l->attn_layer_scale)
            kernel_mul_inplace(x_norm, l->attn_layer_scale, codec_hidden);
        kernel_add_inplace(x, x_norm, codec_hidden);

        /* 7. Post-attention norm + SwiGLU MLP + LayerScale */
        kernel_rms_norm(x_norm, x, l->post_attn_norm, codec_hidden, eps);

        /* SwiGLU MLP + down projection */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        if (l->gate_up_f16 && l->down_f16) {
            float *gu_tmp = (float *)malloc(2 * intermediate * sizeof(float));
            kernel_matvec_f16w(gu_tmp, l->gate_up_f16, x_norm, 2 * intermediate, codec_hidden);
            for (int i = 0; i < intermediate; i++) {
                float g = gu_tmp[i];
                gate_buf[i] = (g / (1.0f + expf(-g))) * gu_tmp[intermediate + i];
            }
            free(gu_tmp);
            kernel_matvec_f16w(x_norm, l->down_f16, gate_buf, codec_hidden, intermediate);
        } else
#endif
        if (l->gate_up_f32 && l->down_f32) {
            float *gu_tmp = (float *)malloc(2 * intermediate * sizeof(float));
            kernel_matvec_f32(gu_tmp, l->gate_up_f32, x_norm, 2 * intermediate, codec_hidden);
            for (int i = 0; i < intermediate; i++) {
                float g = gu_tmp[i];
                gate_buf[i] = (g / (1.0f + expf(-g))) * gu_tmp[intermediate + i];
            }
            free(gu_tmp);
            kernel_matvec_f32(x_norm, l->down_f32, gate_buf, codec_hidden, intermediate);
        } else {
            {
                int n_blocks = codec_hidden / QK8_0;
                block_q8_0 xn_q8[n_blocks];
                kernel_quantize_x_q8(x_norm, codec_hidden, xn_q8);
                kernel_swiglu_matvec_q8(gate_buf, l->gate_up_q8, xn_q8, intermediate, n_blocks);
            }
            {
                int n_blocks = intermediate / QK8_0;
                block_q8_0 gate_q8[n_blocks];
                kernel_quantize_x_q8(gate_buf, intermediate, gate_q8);
                kernel_matvec_q8(x_norm, l->down_q8, gate_q8, codec_hidden, n_blocks);
            }
        }
        if (l->mlp_layer_scale)
            kernel_mul_inplace(x_norm, l->mlp_layer_scale, codec_hidden);
        kernel_add_inplace(x, x_norm, codec_hidden);
    }

    /* Final norm */
    if (ctx->codec.transformer_norm)
        kernel_rms_norm_inplace(x, ctx->codec.transformer_norm, codec_hidden, eps);

    /* Output projection: codec_hidden → latent */
    kernel_matvec_f32(hidden_io, ctx->codec.transformer_output_proj_weight,
                       x, latent, codec_hidden);
    if (ctx->codec.transformer_output_proj_bias)
        kernel_add_inplace(hidden_io, ctx->codec.transformer_output_proj_bias, latent);

    free(x); free(x_norm); free(q_buf); free(k_buf); free(v_buf);
    free(attn_out); free(scores); free(gate_buf);
}

/* --------------------------------------------------------------------
 * Incremental ConvNeXt block
 *
 * Same as batch but uses incremental dwconv and operates on N_new positions.
 * -------------------------------------------------------------------- */

static void codec_convnext_incremental(qwen_tts_convnext_block_t *block,
                                         float *hidden, /* [dim, N_new] in/out */
                                         float *dwconv_state, /* [dim, 6] */
                                         int dim, int N_new)
{
    /* Residual */
    size_t n = (size_t)dim * N_new;
    float *residual = (float *)malloc(n * sizeof(float));
    memcpy(residual, hidden, n * sizeof(float));

    /* Depthwise causal conv (k=7, groups=dim) - incremental */
    float *conv_out = (float *)malloc(n * sizeof(float));
    codec_causal_conv_incremental(conv_out, hidden, dwconv_state,
                                   block->dwconv_weight, block->dwconv_bias,
                                   dim, dim, 7, 1, dim, N_new);

    /* Permute to [N_new, dim] for pointwise ops */
    float *x_ld = (float *)malloc((size_t)N_new * dim * sizeof(float));
    for (int c = 0; c < dim; c++)
        for (int t = 0; t < N_new; t++)
            x_ld[t * dim + c] = conv_out[c * N_new + t];

    /* LayerNorm */
    for (int t = 0; t < N_new; t++)
        kernel_layer_norm(x_ld + t * dim, x_ld + t * dim,
                          block->norm_weight, block->norm_bias, dim, 1e-6f);

    /* pwconv1: [dim] → [4*dim] */
    int dim4 = 4 * dim;
    float *pw1 = (float *)malloc((size_t)N_new * dim4 * sizeof(float));
    for (int t = 0; t < N_new; t++) {
        kernel_matvec_f32(pw1 + t * dim4, block->pwconv1_weight, x_ld + t * dim, dim4, dim);
        if (block->pwconv1_bias)
            kernel_add_inplace(pw1 + t * dim4, block->pwconv1_bias, dim4);
    }

    /* GELU */
    kernel_gelu_inplace(pw1, N_new * dim4);

    /* pwconv2: [4*dim] → [dim] */
    for (int t = 0; t < N_new; t++) {
        kernel_matvec_f32(x_ld + t * dim, block->pwconv2_weight, pw1 + t * dim4, dim, dim4);
        if (block->pwconv2_bias)
            kernel_add_inplace(x_ld + t * dim, block->pwconv2_bias, dim);
    }

    /* Gamma */
    for (int t = 0; t < N_new; t++)
        kernel_mul_inplace(x_ld + t * dim, block->gamma, dim);

    /* Permute back to [dim, N_new] */
    for (int c = 0; c < dim; c++)
        for (int t = 0; t < N_new; t++)
            hidden[c * N_new + t] = x_ld[t * dim + c];

    /* Skip connection */
    kernel_add_inplace(hidden, residual, dim * N_new);

    free(residual); free(conv_out); free(x_ld); free(pw1);
}

/* --------------------------------------------------------------------
 * Incremental vocoder ResUnit
 * -------------------------------------------------------------------- */

static void vocoder_resunit_incremental(
    qwen_tts_vocoder_resunit_t *unit,
    float *hidden,       /* [dim, N_new] in/out */
    float *conv1_state,  /* [dim, (K-1)*dilation] */
    int dim, int N_new, int dilation)
{
    size_t n = (size_t)dim * N_new;
    float *residual = (float *)malloc(n * sizeof(float));
    memcpy(residual, hidden, n * sizeof(float));

    /* SnakeBeta 1 */
    kernel_snake_beta(hidden, hidden, unit->act1_alpha, unit->act1_beta, dim, N_new);

    /* Conv1 (k=7, dilation) - incremental */
    float *conv1_out = (float *)malloc(n * sizeof(float));
    codec_causal_conv_incremental(conv1_out, hidden, conv1_state,
                                   unit->conv1_weight, unit->conv1_bias,
                                   dim, dim, 7, dilation, 1, N_new);

    /* SnakeBeta 2 */
    kernel_snake_beta(conv1_out, conv1_out, unit->act2_alpha, unit->act2_beta, dim, N_new);

    /* Conv2 (k=1, no state needed) */
    kernel_causal_conv1d(hidden, conv1_out, unit->conv2_weight, unit->conv2_bias,
                         dim, dim, 1, N_new, 1, 1);

    /* Skip connection */
    kernel_add_inplace(hidden, residual, dim * N_new);

    free(residual); free(conv1_out);
}

/* --------------------------------------------------------------------
 * RVQ dequantize single timestep
 * -------------------------------------------------------------------- */

static void codec_rvq_dequantize_step(qwen_tts_ctx_t *ctx, const int *codes,
                                        int num_quantizers, float *out)
{
    /* out: [half_latent = 512] (semantic + acoustic summed, projected) */
    qwen_tts_config_t *cfg = &ctx->config;
    qwen_tts_rvq_t *rvq = &ctx->codec.rvq;
    int codebook_size = cfg->codec_codebook_size;
    int half_latent = cfg->codec_latent / 2;  /* 512 */
    int vq_dim = cfg->codec_codebook_dim / 2; /* 256 */

    float *semantic_sum = (float *)calloc(vq_dim, sizeof(float));
    float *acoustic_sum = (float *)calloc(vq_dim, sizeof(float));

    /* Semantic codebook (quantizer 0) */
    {
        qwen_tts_codebook_t *cb = &rvq->semantic_codebooks[0];
        int code = codes[0];
        if (code < 0 || code >= codebook_size) code = 0;
        if (cb->embeddings) {
            for (int d = 0; d < vq_dim; d++)
                semantic_sum[d] += cb->embeddings[code * vq_dim + d];
        } else {
            float usage = cb->cluster_usage[code];
            if (usage < 1e-5f) usage = 1e-5f;
            float inv_usage = 1.0f / usage;
            for (int d = 0; d < vq_dim; d++)
                semantic_sum[d] += cb->embedding_sum[code * vq_dim + d] * inv_usage;
        }
    }

    /* Acoustic codebooks (quantizers 1..N-1) */
    for (int q = 1; q < num_quantizers; q++) {
        qwen_tts_codebook_t *cb = &rvq->acoustic_codebooks[q - 1];
        int code = codes[q];
        if (code < 0 || code >= codebook_size) code = 0;
        if (cb->embeddings) {
            for (int d = 0; d < vq_dim; d++)
                acoustic_sum[d] += cb->embeddings[code * vq_dim + d];
        } else {
            float usage = cb->cluster_usage[code];
            if (usage < 1e-5f) usage = 1e-5f;
            float inv_usage = 1.0f / usage;
            for (int d = 0; d < vq_dim; d++)
                acoustic_sum[d] += cb->embedding_sum[code * vq_dim + d] * inv_usage;
        }
    }

    /* Apply output projections and sum */
    /* Output is [half_latent] = semantic_proj(semantic_sum) + acoustic_proj(acoustic_sum) */
    for (int od = 0; od < half_latent; od++) {
        float sem = 0, aco = 0;
        if (rvq->semantic_output_proj) {
            for (int id = 0; id < vq_dim; id++)
                sem += rvq->semantic_output_proj[od * vq_dim + id] * semantic_sum[id];
        } else {
            sem = (od < vq_dim) ? semantic_sum[od] : 0.0f;
        }
        if (rvq->acoustic_output_proj) {
            for (int id = 0; id < vq_dim; id++)
                aco += rvq->acoustic_output_proj[od * vq_dim + id] * acoustic_sum[id];
        } else {
            aco = (od < vq_dim) ? acoustic_sum[od] : 0.0f;
        }
        out[od] = sem + aco;
    }

    free(semantic_sum);
    free(acoustic_sum);
}

/* ========================================================================
 * Stream init / free
 * ======================================================================== */

qwen_tts_codec_stream_state_t *qwen_tts_codec_stream_init(qwen_tts_ctx_t *ctx)
{
    qwen_tts_config_t *cfg = &ctx->config;
    int latent = cfg->codec_latent;         /* 1024 */
    int half_latent = latent / 2;           /* 512 */
    int decoder_dim = cfg->codec_decoder_dim; /* 1536 */

    qwen_tts_codec_stream_state_t *s = (qwen_tts_codec_stream_state_t *)
        calloc(1, sizeof(qwen_tts_codec_stream_state_t));
    if (!s) return NULL;

    /* Pre-conv: CausalConv1d(512→1024, k=3), state_len = 2 */
    s->pre_conv_state = (float *)calloc((size_t)half_latent * 2, sizeof(float));

    /* Upsample ConvNeXt dwconv states: k=7, state_len=6 */
    for (int i = 0; i < 2; i++)
        s->upsample_cn_state[i] = (float *)calloc((size_t)latent * 6, sizeof(float));

    /* Vocoder pre-conv: CausalConv1d(1024→1536, k=7), state_len=6 */
    int dim = decoder_dim;
    s->voc_preconv_state = (float *)calloc((size_t)latent * 6, sizeof(float));

    for (int b = 0; b < 4; b++) {
        int out_dim = dim / 2;
        int rate = cfg->codec_upsample_rates[b];
        int K = 2 * rate;
        int overlap_len = K - rate;

        s->voc_blocks[b].transconv_overlap = (float *)calloc(
            (size_t)out_dim * overlap_len, sizeof(float));

        int dilations[3] = {1, 3, 9};
        for (int r = 0; r < 3; r++) {
            int state_len = (7 - 1) * dilations[r];
            s->voc_blocks[b].ru_conv1_state[r] = (float *)calloc(
                (size_t)out_dim * state_len, sizeof(float));
        }

        dim = out_dim;
    }

    s->final_conv_state = (float *)calloc((size_t)dim * 6, sizeof(float));

    s->n_processed = 0;
    s->transformer_pos = 0;

    /* Reset codec KV cache */
    ctx->codec_kv_len = 0;

    return s;
}

void qwen_tts_codec_stream_free(qwen_tts_codec_stream_state_t *state)
{
    if (!state) return;

    free(state->pre_conv_state);
    for (int i = 0; i < 2; i++)
        free(state->upsample_cn_state[i]);
    free(state->voc_preconv_state);

    for (int b = 0; b < 4; b++) {
        free(state->voc_blocks[b].transconv_overlap);
        for (int r = 0; r < 3; r++)
            free(state->voc_blocks[b].ru_conv1_state[r]);
    }

    free(state->final_conv_state);
    free(state);
}

/* ========================================================================
 * Decode single token
 * ======================================================================== */

float *qwen_tts_codec_decode_step(
    qwen_tts_ctx_t *ctx,
    qwen_tts_codec_stream_state_t *state,
    const int *codes,
    int *out_samples)
{
    if (!ctx || !state || !codes || !out_samples) {
        if (out_samples) *out_samples = 0;
        return NULL;
    }

    qwen_tts_config_t *cfg = &ctx->config;
    int num_quantizers = cfg->codec_num_quantizers;
    int latent = cfg->codec_latent;       /* 1024 */
    int half_latent = latent / 2;         /* 512 */
    int decoder_dim = cfg->codec_decoder_dim; /* 1536 */

    double step_t0 = now_ms();
    double t0, t1;
    double ms_rvq = 0, ms_preconv = 0, ms_transformer = 0, ms_upsample = 0;
    double ms_voc_preconv = 0, ms_voc_blocks[4] = {0}, ms_final = 0;

    /* 1. RVQ dequantize: 1 token → [half_latent=512, 1] (channels-first) */
    t0 = now_ms();
    float *rvq_out = (float *)malloc(half_latent * sizeof(float));
    codec_rvq_dequantize_step(ctx, codes, num_quantizers, rvq_out);
    ms_rvq = now_ms() - t0;

    /* 2. Pre-conv: CausalConv1d(512→1024, k=3, N_new=1) → [1024, 1] */
    t0 = now_ms();
    float *preconv_out = (float *)malloc(latent * sizeof(float));
    codec_causal_conv_incremental(preconv_out, rvq_out, state->pre_conv_state,
                                   ctx->codec.pre_conv_weight,
                                   ctx->codec.pre_conv_bias,
                                   half_latent, latent, 3, 1, 1, 1);
    free(rvq_out);
    ms_preconv = now_ms() - t0;

    /* 3. Transformer: single token → [latent, 1] */
    t0 = now_ms();
    codec_transformer_step(ctx, preconv_out, state->transformer_pos);
    state->transformer_pos++;
    ms_transformer = now_ms() - t0;

    /* preconv_out now holds transformer output [latent, 1] */

    /* 4-5. Upsample stages (2×): TransConv(k=stride=2, no overlap) + ConvNeXt */
    t0 = now_ms();
    int cur_len = 1;
    float *hidden = preconv_out; /* [latent, cur_len] */

    for (int stage = 0; stage < 2; stage++) {
        int factor = cfg->codec_upsampling_ratios[stage]; /* 2 */
        int new_len;

        /* TransposedConv1d (k=stride=factor, so no overlap) */
        float *up_out = (float *)malloc((size_t)latent * cur_len * factor * 2 * sizeof(float));
        kernel_transposed_conv1d(up_out, hidden,
                                  ctx->codec.upsample_transconv_weight[stage],
                                  ctx->codec.upsample_transconv_bias[stage],
                                  latent, latent, factor, factor, cur_len, &new_len);
        free(hidden);
        hidden = up_out;
        cur_len = new_len;

        /* ConvNeXt (incremental dwconv) */
        codec_convnext_incremental(&ctx->codec.upsample_convnext[stage],
                                    hidden, state->upsample_cn_state[stage],
                                    latent, cur_len);
    }
    ms_upsample = now_ms() - t0;
    /* After 2 upsample stages: cur_len = 1*2*2 = 4, hidden = [1024, 4] */

    /* 6-8. Vocoder: pre-conv → 4 blocks → final conv */
    int current_dim = decoder_dim;
    float *wav;

    /* 6. Vocoder pre-conv */
    t0 = now_ms();
    float *voc_pre = (float *)malloc((size_t)decoder_dim * cur_len * sizeof(float));
    codec_causal_conv_incremental(voc_pre, hidden, state->voc_preconv_state,
                                   ctx->codec.vocoder_pre_conv_weight,
                                   ctx->codec.vocoder_pre_conv_bias,
                                   latent, decoder_dim, 7, 1, 1, cur_len);
    free(hidden);
    ms_voc_preconv = now_ms() - t0;

    hidden = voc_pre;

    /* 7. Vocoder blocks */
    for (int block = 0; block < 4; block++) {
        t0 = now_ms();
        int in_dim = current_dim;
        int out_dim = in_dim / 2;
        int rate = cfg->codec_upsample_rates[block];
        int K = 2 * rate;
        qwen_tts_vocoder_block_t *vb = &ctx->codec.vocoder_blocks[block];

        kernel_snake_beta(hidden, hidden, vb->act_alpha, vb->act_beta, in_dim, cur_len);

        int emit_len = cur_len * rate;
        float *tc_out = (float *)malloc((size_t)out_dim * emit_len * sizeof(float));
        codec_transconv_incremental(tc_out, hidden,
                                     state->voc_blocks[block].transconv_overlap,
                                     vb->transconv_weight, vb->transconv_bias,
                                     in_dim, out_dim, K, rate, cur_len);
        free(hidden);
        hidden = tc_out;
        cur_len = emit_len;
        current_dim = out_dim;

        int dilations[3] = {1, 3, 9};
        for (int ru = 0; ru < 3; ru++) {
            vocoder_resunit_incremental(&vb->resunits[ru],
                                         hidden,
                                         state->voc_blocks[block].ru_conv1_state[ru],
                                         current_dim, cur_len, dilations[ru]);
        }
        ms_voc_blocks[block] = now_ms() - t0;
    }

    /* 8. Final */
    t0 = now_ms();
    kernel_snake_beta(hidden, hidden,
                      ctx->codec.vocoder_final_act_alpha,
                      ctx->codec.vocoder_final_act_beta,
                      current_dim, cur_len);

    wav = (float *)malloc(cur_len * sizeof(float));
    codec_causal_conv_incremental(wav, hidden, state->final_conv_state,
                                   ctx->codec.vocoder_final_conv_weight,
                                   ctx->codec.vocoder_final_conv_bias,
                                   current_dim, 1, 7, 1, 1, cur_len);
    free(hidden);

    /* 9. Clamp */
    kernel_clamp(wav, cur_len, -1.0f, 1.0f);
    ms_final = now_ms() - t0;

    state->n_processed++;
    *out_samples = cur_len;

    {
        double elapsed = now_ms() - step_t0;
        double ms_vocoder = ms_voc_preconv + ms_voc_blocks[0] + ms_voc_blocks[1]
                          + ms_voc_blocks[2] + ms_voc_blocks[3] + ms_final;
        if (qwen_tts_verbose >= 1) {
            fprintf(stderr, "  decode_step[%d]: %.0fms (rvq=%.0f pre=%.0f tf=%.0f up=%.0f voc=%.0f [pre=%.0f b0=%.0f b1=%.0f b2=%.0f b3=%.0f fin=%.0f])\n",
                    state->n_processed, elapsed,
                    ms_rvq, ms_preconv, ms_transformer, ms_upsample,
                    ms_vocoder, ms_voc_preconv,
                    ms_voc_blocks[0], ms_voc_blocks[1], ms_voc_blocks[2], ms_voc_blocks[3],
                    ms_final);
        }
    }

    return wav;
}

/* ========================================================================
 * Verify incremental vs batch decode
 * ======================================================================== */

int qwen_tts_codec_verify_incremental(qwen_tts_ctx_t *ctx,
                                        const int *all_codes,
                                        int n_tokens)
{
    if (!ctx || !all_codes || n_tokens <= 0) return -1;

    qwen_tts_config_t *cfg = &ctx->config;
    int num_groups = cfg->codec_num_quantizers;

    fprintf(stderr, "Verify incremental: %d tokens, %d quantizers\n", n_tokens, num_groups);

    /* 1. Batch decode */
    int batch_len = 0;
    float *batch_audio = qwen_tts_codec_decode(ctx, all_codes, n_tokens, &batch_len);
    if (!batch_audio || batch_len <= 0) {
        fprintf(stderr, "  Batch decode failed\n");
        return -1;
    }
    fprintf(stderr, "  Batch decode: %d samples\n", batch_len);

    /* 2. Incremental decode */
    qwen_tts_codec_stream_state_t *state = qwen_tts_codec_stream_init(ctx);
    if (!state) {
        free(batch_audio);
        fprintf(stderr, "  Stream init failed\n");
        return -1;
    }

    int inc_total = 0;
    float *inc_audio = (float *)malloc(batch_len * sizeof(float));

    for (int t = 0; t < n_tokens; t++) {
        int n_samp = 0;
        float *chunk = qwen_tts_codec_decode_step(ctx, state,
                           all_codes + t * num_groups, &n_samp);
        if (chunk && n_samp > 0) {
            if (inc_total + n_samp <= batch_len)
                memcpy(inc_audio + inc_total, chunk, n_samp * sizeof(float));
            inc_total += n_samp;
            free(chunk);
        }
    }
    qwen_tts_codec_stream_free(state);

    fprintf(stderr, "  Incremental decode: %d samples\n", inc_total);

    /* 3. Compare */
    int compare_len = batch_len < inc_total ? batch_len : inc_total;
    float max_diff = 0.0f;
    double sum_diff = 0.0;
    for (int i = 0; i < compare_len; i++) {
        float d = batch_audio[i] - inc_audio[i];
        if (d < 0) d = -d;
        if (d > max_diff) max_diff = d;
        sum_diff += d;
    }

    float mean_diff = (compare_len > 0) ? (float)(sum_diff / compare_len) : 0.0f;
    fprintf(stderr, "  Comparison: max_diff=%.6f mean_diff=%.6f length_match=%s\n",
            max_diff, mean_diff,
            (batch_len == inc_total) ? "yes" : "no");

    int pass = (max_diff < 1e-4f && batch_len == inc_total);
    fprintf(stderr, "  Result: %s\n", pass ? "PASS" : "FAIL");

    free(batch_audio);
    free(inc_audio);

    return pass ? 0 : 1;
}
