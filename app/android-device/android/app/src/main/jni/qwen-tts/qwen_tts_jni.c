/*
 * qwen_tts_jni.c - JNI wrapper for the Qwen3-TTS C inference engine.
 *
 * Exposes native methods called from HermesRuntime.java:
 *   - nativeTtsLoadModel(String modelDir) -> boolean
 *   - nativeTtsGenerate(String tokenIds, String speaker, String language) -> short[]
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
