/*
 * qwen_tts_jni.c - JNI wrapper for the Qwen3-TTS C inference engine.
 *
 * Exposes native methods called from HermesRuntime.java:
 *   - nativeTtsLoadModel(String modelDir) -> boolean
 *   - nativeTtsGenerate(String tokenIds, String speaker, String language) -> short[]
 *   - nativeTtsGenerateStream(String tokenIds, String speaker, String language,
 *                             int chunkSize, TtsStreamCallback callback) -> void
 *   - nativeTtsIsLoaded() -> boolean
 *   - nativeTtsFree() -> void
 */

#include <jni.h>
#include <android/log.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

#include "qwen_tts.h"

#define TAG "QwenTTS_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

/* Redirect stderr to logcat so C-level fprintf(stderr, ...) is visible */
static int s_stderr_redirected = 0;

static void *stderr_reader_thread(void *arg) {
    int fd = (int)(intptr_t)arg;
    char buf[512];
    ssize_t n;
    while ((n = read(fd, buf, sizeof(buf) - 1)) > 0) {
        buf[n] = '\0';
        /* Remove trailing newlines */
        while (n > 0 && (buf[n-1] == '\n' || buf[n-1] == '\r')) buf[--n] = '\0';
        if (n > 0) __android_log_print(ANDROID_LOG_INFO, "QwenTTS", "%s", buf);
    }
    close(fd);
    return NULL;
}

static void redirect_stderr_to_logcat(void) {
    if (s_stderr_redirected) return;
    s_stderr_redirected = 1;
    int pipefd[2];
    if (pipe(pipefd) == -1) return;
    dup2(pipefd[1], STDERR_FILENO);
    close(pipefd[1]);
    pthread_t tid;
    pthread_create(&tid, NULL, stderr_reader_thread, (void *)(intptr_t)pipefd[0]);
    pthread_detach(tid);
}

/* Global TTS context - lazy-loaded, kept in memory */
static qwen_tts_ctx_t *g_tts_ctx = NULL;

/* Progress callback that logs to Android */
static void jni_progress_cb(int step, int total, void *userdata) {
    if (step % 20 == 0) {
        LOGI("TTS generate: step %d / %d", step, total);
    }
}

JNIEXPORT jboolean JNICALL
Java_ai_connct_1screen_rn_HermesRuntime_nativeTtsLoadModel(JNIEnv *env, jclass cls, jstring jModelDir) {
    if (g_tts_ctx != NULL) {
        LOGI("TTS model already loaded");
        return JNI_TRUE;
    }

    const char *model_dir = (*env)->GetStringUTFChars(env, jModelDir, NULL);
    if (!model_dir) {
        LOGE("Failed to get model dir string");
        return JNI_FALSE;
    }

    LOGI("Loading TTS model from: %s", model_dir);
    redirect_stderr_to_logcat();
    qwen_tts_verbose = 1;

    /* Set writable cache directory for quantized weight cache */
    qwen_tts_cache_dir_override = "/data/data/ai.connct_screen.rn/cache";

    g_tts_ctx = qwen_tts_load(model_dir);
    (*env)->ReleaseStringUTFChars(env, jModelDir, model_dir);

    if (!g_tts_ctx) {
        LOGE("Failed to load TTS model");
        return JNI_FALSE;
    }

    qwen_tts_set_progress_callback(g_tts_ctx, jni_progress_cb, NULL);
    LOGI("TTS model loaded successfully");
    return JNI_TRUE;
}

JNIEXPORT jshortArray JNICALL
Java_ai_connct_1screen_rn_HermesRuntime_nativeTtsGenerate(JNIEnv *env, jclass cls,
                                                           jstring jTokenIds,
                                                           jstring jSpeaker,
                                                           jstring jLanguage) {
    if (!g_tts_ctx) {
        LOGE("TTS model not loaded");
        return NULL;
    }

    const char *token_ids = (*env)->GetStringUTFChars(env, jTokenIds, NULL);
    const char *speaker = jSpeaker ? (*env)->GetStringUTFChars(env, jSpeaker, NULL) : NULL;
    const char *language = jLanguage ? (*env)->GetStringUTFChars(env, jLanguage, NULL) : NULL;

    LOGI("TTS generate: tokens=%s speaker=%s language=%s",
         token_ids, speaker ? speaker : "(null)", language ? language : "(null)");

    int out_samples = 0;
    float *pcm_float = qwen_tts_generate(g_tts_ctx, token_ids, speaker, language, &out_samples);

    (*env)->ReleaseStringUTFChars(env, jTokenIds, token_ids);
    if (speaker) (*env)->ReleaseStringUTFChars(env, jSpeaker, speaker);
    if (language) (*env)->ReleaseStringUTFChars(env, jLanguage, language);

    if (!pcm_float || out_samples <= 0) {
        LOGE("TTS generate returned no audio");
        if (pcm_float) free(pcm_float);
        return NULL;
    }

    LOGI("TTS generated %d samples (%.2f seconds)", out_samples,
         (float)out_samples / QWEN_TTS_SAMPLE_RATE);

    /* Convert float32 PCM [-1,1] to int16 PCM */
    jshortArray result = (*env)->NewShortArray(env, out_samples);
    if (!result) {
        LOGE("Failed to allocate short array for %d samples", out_samples);
        free(pcm_float);
        return NULL;
    }

    jshort *shorts = (*env)->GetShortArrayElements(env, result, NULL);
    for (int i = 0; i < out_samples; i++) {
        float sample = pcm_float[i];
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        shorts[i] = (jshort)(sample * 32767.0f);
    }
    (*env)->ReleaseShortArrayElements(env, result, shorts, 0);

    free(pcm_float);
    return result;
}

/* ========================================================================
 * Streaming TTS: C audio callback → Java TtsStreamCallback bridge
 * ======================================================================== */

typedef struct {
    JavaVM *jvm;
    jobject callback_ref;   /* global ref to TtsStreamCallback */
    jmethodID onAudioChunk;
    jmethodID onComplete;
    jmethodID onError;
    int total_samples;
    long start_ms;
} stream_cb_data_t;

static long now_ms_jni(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)(ts.tv_sec * 1000L + ts.tv_nsec / 1000000L);
}

/* C audio callback: convert float32→int16 and call Java onAudioChunk */
static int jni_audio_cb(const float *samples, int n_samples, void *userdata) {
    stream_cb_data_t *data = (stream_cb_data_t *)userdata;
    JNIEnv *env = NULL;
    int attached = 0;

    if ((*data->jvm)->GetEnv(data->jvm, (void **)&env, JNI_VERSION_1_6) != JNI_OK) {
        if ((*data->jvm)->AttachCurrentThread(data->jvm, &env, NULL) != JNI_OK) {
            LOGE("Failed to attach thread for audio callback");
            return -1;
        }
        attached = 1;
    }

    /* Convert float32 → int16 */
    jshortArray arr = (*env)->NewShortArray(env, n_samples);
    if (arr) {
        jshort *shorts = (*env)->GetShortArrayElements(env, arr, NULL);
        for (int i = 0; i < n_samples; i++) {
            float s = samples[i];
            if (s > 1.0f) s = 1.0f;
            if (s < -1.0f) s = -1.0f;
            shorts[i] = (jshort)(s * 32767.0f);
        }
        (*env)->ReleaseShortArrayElements(env, arr, shorts, 0);

        (*env)->CallVoidMethod(env, data->callback_ref, data->onAudioChunk, arr, n_samples);
        (*env)->DeleteLocalRef(env, arr);
    }

    data->total_samples += n_samples;

    if (attached) {
        (*data->jvm)->DetachCurrentThread(data->jvm);
    }
    return 0;
}

JNIEXPORT void JNICALL
Java_ai_connct_1screen_rn_HermesRuntime_nativeTtsGenerateStream(
    JNIEnv *env, jclass cls,
    jstring jTokenIds, jstring jSpeaker, jstring jLanguage,
    jint chunkSize, jobject jCallback)
{
    if (!g_tts_ctx) {
        LOGE("TTS model not loaded");
        /* Call onError */
        jclass cbClass = (*env)->GetObjectClass(env, jCallback);
        jmethodID onError = (*env)->GetMethodID(env, cbClass, "onError", "(Ljava/lang/String;)V");
        if (onError) {
            jstring msg = (*env)->NewStringUTF(env, "TTS model not loaded");
            (*env)->CallVoidMethod(env, jCallback, onError, msg);
        }
        return;
    }

    const char *token_ids = (*env)->GetStringUTFChars(env, jTokenIds, NULL);
    const char *speaker = jSpeaker ? (*env)->GetStringUTFChars(env, jSpeaker, NULL) : NULL;
    const char *language = jLanguage ? (*env)->GetStringUTFChars(env, jLanguage, NULL) : NULL;

    LOGI("TTS stream generate: tokens=%s speaker=%s language=%s chunk_size=%d",
         token_ids, speaker ? speaker : "(null)", language ? language : "(null)", chunkSize);

    /* Set up callback bridge */
    jclass cbClass = (*env)->GetObjectClass(env, jCallback);
    stream_cb_data_t cb_data;
    (*env)->GetJavaVM(env, &cb_data.jvm);
    cb_data.callback_ref = (*env)->NewGlobalRef(env, jCallback);
    cb_data.onAudioChunk = (*env)->GetMethodID(env, cbClass, "onAudioChunk", "([SI)V");
    cb_data.onComplete = (*env)->GetMethodID(env, cbClass, "onComplete", "(IJ)V");
    cb_data.onError = (*env)->GetMethodID(env, cbClass, "onError", "(Ljava/lang/String;)V");
    cb_data.total_samples = 0;
    cb_data.start_ms = now_ms_jni();

    int ret = qwen_tts_generate_stream(g_tts_ctx, token_ids, speaker, language,
                                        (int)chunkSize, jni_audio_cb, &cb_data);

    (*env)->ReleaseStringUTFChars(env, jTokenIds, token_ids);
    if (speaker) (*env)->ReleaseStringUTFChars(env, jSpeaker, speaker);
    if (language) (*env)->ReleaseStringUTFChars(env, jLanguage, language);

    long elapsed_ms = now_ms_jni() - cb_data.start_ms;

    if (ret == 0) {
        LOGI("TTS stream complete: %d samples in %ldms", cb_data.total_samples, elapsed_ms);
        (*env)->CallVoidMethod(env, cb_data.callback_ref, cb_data.onComplete,
                               cb_data.total_samples, (jlong)elapsed_ms);
    } else {
        const char *err = (ret == 1) ? "Generation aborted by callback" : "Generation failed";
        LOGE("TTS stream error: %s (ret=%d)", err, ret);
        jstring msg = (*env)->NewStringUTF(env, err);
        (*env)->CallVoidMethod(env, cb_data.callback_ref, cb_data.onError, msg);
    }

    (*env)->DeleteGlobalRef(env, cb_data.callback_ref);
}

/* Collector callback for verify: accumulates PCM samples */
typedef struct {
    float *samples;
    int n_samples;
    int capacity;
} verify_collector_t;

static int verify_audio_cb(const float *samples, int n_samples, void *userdata) {
    verify_collector_t *col = (verify_collector_t *)userdata;
    if (col->n_samples + n_samples > col->capacity) {
        int new_cap = (col->n_samples + n_samples) * 2;
        col->samples = (float *)realloc(col->samples, new_cap * sizeof(float));
        col->capacity = new_cap;
    }
    memcpy(col->samples + col->n_samples, samples, n_samples * sizeof(float));
    col->n_samples += n_samples;
    return 0;
}

JNIEXPORT jint JNICALL
Java_ai_connct_1screen_rn_HermesRuntime_nativeTtsVerifyIncremental(
    JNIEnv *env, jclass cls,
    jstring jTokenIds, jstring jSpeaker, jstring jLanguage)
{
    if (!g_tts_ctx) {
        LOGE("TTS model not loaded");
        return -1;
    }

    const char *token_ids = (*env)->GetStringUTFChars(env, jTokenIds, NULL);
    const char *speaker = jSpeaker ? (*env)->GetStringUTFChars(env, jSpeaker, NULL) : NULL;
    const char *language = jLanguage ? (*env)->GetStringUTFChars(env, jLanguage, NULL) : NULL;

    LOGI("TTS verify: comparing batch vs incremental decode");

    /* Use a fixed seed so both runs produce the same codec tokens */
    g_tts_ctx->sample_seed = 42;

    /* Step 1: Batch generate (chunk_size=0) */
    LOGI("TTS verify: running batch generate...");
    verify_collector_t batch_col = {NULL, 0, 0};
    batch_col.capacity = 48000;
    batch_col.samples = (float *)malloc(batch_col.capacity * sizeof(float));

    int ret1 = qwen_tts_generate_stream(g_tts_ctx, token_ids, speaker, language,
                                          0, verify_audio_cb, &batch_col);
    LOGI("TTS verify: batch returned %d, %d samples", ret1, batch_col.n_samples);

    /* Step 2: Incremental generate (chunk_size=1) with same seed */
    g_tts_ctx->sample_seed = 42;

    LOGI("TTS verify: running incremental generate...");
    verify_collector_t incr_col = {NULL, 0, 0};
    incr_col.capacity = 48000;
    incr_col.samples = (float *)malloc(incr_col.capacity * sizeof(float));

    int ret2 = qwen_tts_generate_stream(g_tts_ctx, token_ids, speaker, language,
                                          1, verify_audio_cb, &incr_col);
    LOGI("TTS verify: incremental returned %d, %d samples", ret2, incr_col.n_samples);

    /* Step 3: Compare */
    int compare_len = batch_col.n_samples < incr_col.n_samples ?
                      batch_col.n_samples : incr_col.n_samples;
    float max_diff = 0.0f;
    double sum_diff = 0.0;
    for (int i = 0; i < compare_len; i++) {
        float d = batch_col.samples[i] - incr_col.samples[i];
        if (d < 0) d = -d;
        if (d > max_diff) max_diff = d;
        sum_diff += d;
    }
    float mean_diff = (compare_len > 0) ? (float)(sum_diff / compare_len) : 0.0f;

    int length_match = (batch_col.n_samples == incr_col.n_samples);
    int pass = (max_diff < 1e-3f && length_match);

    LOGI("TTS verify: batch=%d incr=%d samples, max_diff=%.6f mean_diff=%.6f length_match=%d => %s",
         batch_col.n_samples, incr_col.n_samples, max_diff, mean_diff, length_match,
         pass ? "PASS" : "FAIL");

    free(batch_col.samples);
    free(incr_col.samples);

    (*env)->ReleaseStringUTFChars(env, jTokenIds, token_ids);
    if (speaker) (*env)->ReleaseStringUTFChars(env, jSpeaker, speaker);
    if (language) (*env)->ReleaseStringUTFChars(env, jLanguage, language);

    return pass ? 0 : 1;
}

JNIEXPORT jboolean JNICALL
Java_ai_connct_1screen_rn_HermesRuntime_nativeTtsIsLoaded(JNIEnv *env, jclass cls) {
    return g_tts_ctx != NULL ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_ai_connct_1screen_rn_HermesRuntime_nativeTtsFree(JNIEnv *env, jclass cls) {
    if (g_tts_ctx) {
        LOGI("Freeing TTS model");
        qwen_tts_free(g_tts_ctx);
        g_tts_ctx = NULL;
    }
}
