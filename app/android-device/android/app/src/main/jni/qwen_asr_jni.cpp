/**
 * qwen_asr_jni.cpp
 *
 * JNI bridge between VoiceService.java and the qwen-asr C library.
 * Manages model lifecycle, audio push, and ASR inference thread.
 */

#include <jni.h>
#include <android/log.h>
#include <pthread.h>
#include <cstring>
#include <cstdlib>
#include <unistd.h>

extern "C" {
#include "qwen_asr.h"
#include "qwen_asr_audio.h"
#include "qwen_asr_kernels.h"
}

#define TAG "QwenASR_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ---------------------------------------------------------------------------
// Redirect stderr to logcat so C library fprintf(stderr,...) becomes visible
// ---------------------------------------------------------------------------
static int g_stderr_pipe[2] = {-1, -1};
static pthread_t g_stderr_thread;

static void *stderr_reader_thread(void *) {
    char buf[512];
    ssize_t n;
    while ((n = read(g_stderr_pipe[0], buf, sizeof(buf) - 1)) > 0) {
        buf[n] = '\0';
        // Trim trailing newline
        while (n > 0 && (buf[n-1] == '\n' || buf[n-1] == '\r')) buf[--n] = '\0';
        if (n > 0) __android_log_print(ANDROID_LOG_INFO, "QwenASR", "%s", buf);
    }
    return nullptr;
}

static void setup_stderr_redirect() {
    if (g_stderr_pipe[0] >= 0) return; // already set up
    if (pipe(g_stderr_pipe) == -1) return;
    dup2(g_stderr_pipe[1], STDERR_FILENO);
    close(g_stderr_pipe[1]);
    g_stderr_pipe[1] = -1;
    pthread_create(&g_stderr_thread, nullptr, stderr_reader_thread, nullptr);
}

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

static JavaVM *g_jvm = nullptr;
static qwen_ctx_t *g_ctx = nullptr;
static qwen_live_audio_t *g_live = nullptr;
static pthread_t g_asr_thread;
static bool g_asr_running = false;

// Cached JNI references for token callback
static jclass g_voice_service_class = nullptr;
static jmethodID g_on_native_token = nullptr;

// ---------------------------------------------------------------------------
// JNI_OnLoad
// ---------------------------------------------------------------------------

extern "C" JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void * /*reserved*/) {
    g_jvm = vm;
    return JNI_VERSION_1_6;
}

// ---------------------------------------------------------------------------
// Token callback: called from the ASR inference thread
// ---------------------------------------------------------------------------

static void token_callback(const char *piece, void * /*userdata*/) {
    if (!g_jvm || !piece) return;

    JNIEnv *env = nullptr;
    bool attached = false;
    int status = g_jvm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6);
    if (status == JNI_EDETACHED) {
        if (g_jvm->AttachCurrentThread(&env, nullptr) != 0) {
            LOGE("token_callback: failed to attach thread");
            return;
        }
        attached = true;
    }

    if (env && g_voice_service_class && g_on_native_token) {
        jstring jpiece = env->NewStringUTF(piece);
        if (jpiece) {
            env->CallStaticVoidMethod(g_voice_service_class, g_on_native_token, jpiece);
            env->DeleteLocalRef(jpiece);
        }
    }

    if (attached) {
        g_jvm->DetachCurrentThread();
    }
}

// ---------------------------------------------------------------------------
// ASR inference thread
// ---------------------------------------------------------------------------

static void *asr_thread_func(void * /*arg*/) {
    if (!g_ctx || !g_live) return nullptr;

    LOGI("ASR inference thread started");
    char *text = qwen_transcribe_stream_live(g_ctx, g_live);
    if (text) {
        LOGI("ASR final text: %s", text);
        free(text);
    }
    LOGI("ASR inference thread ended");
    return nullptr;
}

// ---------------------------------------------------------------------------
// JNI exports: ai.connct_screen.rn.VoiceService
// ---------------------------------------------------------------------------

extern "C" {

JNIEXPORT void JNICALL
Java_ai_connct_1screen_rn_VoiceService_nativeSetCacheDir(
        JNIEnv *env, jclass clazz, jstring cacheDir) {
    const char *dir = env->GetStringUTFChars(cacheDir, nullptr);
    if (dir) {
        qwen_set_cache_dir(dir);
        LOGI("Cache dir set to: %s", dir);
        env->ReleaseStringUTFChars(cacheDir, dir);
    }
}

JNIEXPORT jboolean JNICALL
Java_ai_connct_1screen_rn_VoiceService_nativeLoadModel(
        JNIEnv *env, jclass clazz, jstring modelDir, jint nThreads) {

    // Cache JNI references for token callback
    g_voice_service_class = (jclass)env->NewGlobalRef(clazz);
    g_on_native_token = env->GetStaticMethodID(clazz, "onNativeToken", "(Ljava/lang/String;)V");
    if (!g_on_native_token) {
        LOGE("nativeLoadModel: cannot find onNativeToken method");
        return JNI_FALSE;
    }

    const char *dir = env->GetStringUTFChars(modelDir, nullptr);
    if (!dir) return JNI_FALSE;

    LOGI("Loading model from: %s", dir);
    g_ctx = qwen_load(dir);
    env->ReleaseStringUTFChars(modelDir, dir);

    if (!g_ctx) {
        LOGE("nativeLoadModel: qwen_load failed");
        return JNI_FALSE;
    }

    // Set thread count (default to 4 if not specified)
    int threads = nThreads > 0 ? nThreads : 4;
    qwen_set_threads(threads);
    LOGI("Set thread count to %d", threads);

    // Configure for live streaming
    g_ctx->stream_chunk_sec = 2.0f;
    g_ctx->stream_rollback = 5;
    g_ctx->stream_unfixed_chunks = 2;
    g_ctx->stream_max_new_tokens = 32;

    // Set token callback
    qwen_set_token_callback(g_ctx, token_callback, nullptr);

    LOGI("Model loaded successfully");
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_ai_connct_1screen_rn_VoiceService_nativeStartAsr(JNIEnv *env, jclass /*clazz*/) {
    if (!g_ctx) {
        LOGE("nativeStartAsr: model not loaded");
        return JNI_FALSE;
    }
    if (g_asr_running) {
        LOGE("nativeStartAsr: already running");
        return JNI_FALSE;
    }

    // Reset KV cache for fresh start
    g_ctx->kv_cache_len = 0;

    g_live = qwen_live_audio_create();
    if (!g_live) {
        LOGE("nativeStartAsr: failed to create live audio");
        return JNI_FALSE;
    }

    g_asr_running = true;
    if (pthread_create(&g_asr_thread, nullptr, asr_thread_func, nullptr) != 0) {
        LOGE("nativeStartAsr: failed to create ASR thread");
        qwen_live_audio_free(g_live);
        g_live = nullptr;
        g_asr_running = false;
        return JNI_FALSE;
    }

    LOGI("ASR started");
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_ai_connct_1screen_rn_VoiceService_nativePushAudio(
        JNIEnv *env, jclass /*clazz*/, jshortArray samples, jint length) {
    if (!g_live || !samples) return;

    jshort *buf = env->GetShortArrayElements(samples, nullptr);
    if (!buf) return;

    qwen_live_audio_push_s16(g_live, buf, length);
    env->ReleaseShortArrayElements(samples, buf, JNI_ABORT);
}

JNIEXPORT void JNICALL
Java_ai_connct_1screen_rn_VoiceService_nativeStopAsr(JNIEnv *env, jclass /*clazz*/) {
    if (!g_asr_running || !g_live) return;

    LOGI("Stopping ASR...");
    qwen_live_audio_signal_eof(g_live);
    pthread_join(g_asr_thread, nullptr);

    qwen_live_audio_free(g_live);
    g_live = nullptr;
    g_asr_running = false;
    LOGI("ASR stopped");
}

JNIEXPORT void JNICALL
Java_ai_connct_1screen_rn_VoiceService_nativeResetAsr(JNIEnv *env, jclass /*clazz*/) {
    if (!g_ctx) return;

    // Stop current ASR if running
    if (g_asr_running && g_live) {
        qwen_live_audio_signal_eof(g_live);
        pthread_join(g_asr_thread, nullptr);
        qwen_live_audio_free(g_live);
        g_live = nullptr;
        g_asr_running = false;
    }

    // Reset KV cache
    g_ctx->kv_cache_len = 0;

    // Create fresh live audio and restart ASR thread
    g_live = qwen_live_audio_create();
    if (!g_live) {
        LOGE("nativeResetAsr: failed to create live audio");
        return;
    }

    g_asr_running = true;
    if (pthread_create(&g_asr_thread, nullptr, asr_thread_func, nullptr) != 0) {
        LOGE("nativeResetAsr: failed to create ASR thread");
        qwen_live_audio_free(g_live);
        g_live = nullptr;
        g_asr_running = false;
    }

    LOGI("ASR reset and restarted");
}

JNIEXPORT void JNICALL
Java_ai_connct_1screen_rn_VoiceService_nativeFreeModel(JNIEnv *env, jclass /*clazz*/) {
    // Stop ASR first
    if (g_asr_running && g_live) {
        qwen_live_audio_signal_eof(g_live);
        pthread_join(g_asr_thread, nullptr);
        qwen_live_audio_free(g_live);
        g_live = nullptr;
        g_asr_running = false;
    }

    if (g_ctx) {
        qwen_free(g_ctx);
        g_ctx = nullptr;
    }

    if (g_voice_service_class) {
        env->DeleteGlobalRef(g_voice_service_class);
        g_voice_service_class = nullptr;
    }
    g_on_native_token = nullptr;

    LOGI("Model freed");
}

JNIEXPORT void JNICALL
Java_ai_connct_1screen_rn_VoiceService_nativeTestWav(
        JNIEnv *env, jclass clazz, jstring wavPath) {

    const char *path = env->GetStringUTFChars(wavPath, nullptr);
    if (!path) {
        LOGE("nativeTestWav: null path");
        return;
    }

    if (!g_ctx) {
        LOGE("nativeTestWav: model not loaded");
        env->ReleaseStringUTFChars(wavPath, path);
        return;
    }

    // Cache JNI refs if not yet done (in case test_wav is called before normal start)
    if (!g_voice_service_class) {
        g_voice_service_class = (jclass)env->NewGlobalRef(clazz);
        g_on_native_token = env->GetStaticMethodID(clazz, "onNativeToken", "(Ljava/lang/String;)V");
    }

    // Enable verbose timing for per-phase profiling
    qwen_verbose = 3;
    setup_stderr_redirect();

    LOGI("nativeTestWav: loading %s", path);
    int n_samples = 0;
    float *samples = qwen_load_wav(path, &n_samples);
    env->ReleaseStringUTFChars(wavPath, path);

    if (!samples || n_samples <= 0) {
        LOGE("nativeTestWav: failed to load WAV");
        return;
    }
    LOGI("nativeTestWav: loaded %d samples (%.2f sec)", n_samples, (float)n_samples / 16000.0f);

    // Reset KV cache for clean inference
    g_ctx->kv_cache_len = 0;

    // Use batch mode for WAV files — single encoder + prefill + decode pass,
    // much faster than streaming which re-prefills O(N²) per chunk.
    LOGI("nativeTestWav: starting batch transcription...");
    char *text = qwen_transcribe_audio(g_ctx, samples, n_samples);
    free(samples);

    if (text) {
        LOGI("nativeTestWav: result = %s", text);
        free(text);
    } else {
        LOGI("nativeTestWav: no text returned");
    }

    LOGI("nativeTestWav: done");
}

JNIEXPORT void JNICALL
Java_ai_connct_1screen_rn_VoiceService_nativeTestWavStream(
        JNIEnv *env, jclass clazz, jstring wavPath) {

    const char *path = env->GetStringUTFChars(wavPath, nullptr);
    if (!path) {
        LOGE("nativeTestWavStream: null path");
        return;
    }

    if (!g_ctx) {
        LOGE("nativeTestWavStream: model not loaded");
        env->ReleaseStringUTFChars(wavPath, path);
        return;
    }

    // Cache JNI refs if not yet done
    if (!g_voice_service_class) {
        g_voice_service_class = (jclass)env->NewGlobalRef(clazz);
        g_on_native_token = env->GetStaticMethodID(clazz, "onNativeToken", "(Ljava/lang/String;)V");
    }

    qwen_verbose = 3;
    setup_stderr_redirect();

    LOGI("nativeTestWavStream: loading %s", path);
    int n_samples = 0;
    float *samples = qwen_load_wav(path, &n_samples);
    env->ReleaseStringUTFChars(wavPath, path);

    if (!samples || n_samples <= 0) {
        LOGE("nativeTestWavStream: failed to load WAV");
        return;
    }
    LOGI("nativeTestWavStream: loaded %d samples (%.2f sec)", n_samples, (float)n_samples / 16000.0f);

    // Reset KV cache for clean inference
    g_ctx->kv_cache_len = 0;

    // Use streaming path: create live audio, push all samples, signal EOF, transcribe
    qwen_live_audio_t *live = qwen_live_audio_create();
    if (!live) {
        LOGE("nativeTestWavStream: failed to create live audio");
        free(samples);
        return;
    }

    qwen_live_audio_push(live, samples, n_samples);
    qwen_live_audio_signal_eof(live);

    LOGI("nativeTestWavStream: starting streaming transcription...");
    char *text = qwen_transcribe_stream_live(g_ctx, live);

    if (text) {
        LOGI("nativeTestWavStream: result = %s", text);
        free(text);
    } else {
        LOGI("nativeTestWavStream: no text returned");
    }

    LOGI("nativeTestWavStream: done");
    qwen_live_audio_free(live);
    free(samples);
}

} // extern "C"
