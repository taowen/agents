/**
 * hermes_runtime.h
 *
 * Shared types and helpers for the multi-agent Hermes runtime.
 * Each agent type (app, browser, ...) gets its own Hermes runtime
 * with a shared set of common tools plus agent-specific tools.
 */

#pragma once

#include <jni.h>
#include <android/log.h>
#include <hermes/hermes.h>
#include <jsi/jsi.h>
#include <string>
#include <memory>
#include <unordered_map>

#define LOG_TAG "HermesRuntime"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Cache for HermesRuntime.java shared static methods.
struct CommonJniCache {
    jclass clazz;          // HermesRuntime.java
    jmethodID httpPost;
    jmethodID appendLog;
    jmethodID updateStatus;
    jmethodID askUser;
    jmethodID hideOverlay;
    jmethodID sleepMs;
    jmethodID speak;
};

// One runtime per agent type.
struct RuntimeEntry {
    std::unique_ptr<facebook::hermes::HermesRuntime> runtime;
    std::string agentType;
};

// Global state — defined in hermes_runtime.cpp
extern JavaVM* g_jvm;
extern CommonJniCache g_common_cache;
extern std::unordered_map<std::string, RuntimeEntry> g_runtimes;

// Get JNIEnv for the current thread, attaching if necessary.
JNIEnv* getEnv();

// Convert jstring → std::string (handles null).
std::string jstringToStd(JNIEnv* env, jstring js);

// Register the common host functions (http_post, log, sleep, update_status, ask_user, hide_overlay).
void registerCommonTools(facebook::jsi::Runtime& rt);
