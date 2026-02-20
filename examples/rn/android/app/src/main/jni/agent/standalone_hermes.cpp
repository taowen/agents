/**
 * standalone_hermes.cpp
 *
 * Creates an independent Hermes JS runtime inside the AccessibilityService
 * process. Host functions are registered on globalThis so the agent JS code
 * can call native Android operations (screen reading, clicking, scrolling, etc.)
 * and HTTP requests without going through React Native's bridge.
 *
 * Each host function calls back into Java via JNI static methods on
 * ai.connct_screen.rn.HermesAgentRunner.
 */

#include <jni.h>
#include <android/log.h>
#include <hermes/hermes.h>
#include <jsi/jsi.h>
#include <string>
#include <memory>

#define LOG_TAG "HermesAgent"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace facebook::jsi;
using namespace facebook::hermes;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static JavaVM* g_jvm = nullptr;

// Get JNIEnv for the current thread, attaching if necessary.
static JNIEnv* getEnv() {
    JNIEnv* env = nullptr;
    if (g_jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) == JNI_EDETACHED) {
        g_jvm->AttachCurrentThread(&env, nullptr);
    }
    return env;
}

// Convert jsi::String → std::string
static std::string toStdString(Runtime& rt, const Value& val) {
    return val.asString(rt).utf8(rt);
}

// Convert jstring → std::string (handles null)
static std::string jstringToStd(JNIEnv* env, jstring js) {
    if (!js) return "";
    const char* chars = env->GetStringUTFChars(js, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(js, chars);
    return result;
}

// Cache for HermesAgentRunner class and its static methods.
// Resolved once per runtime creation to avoid repeated lookups.
struct JniCache {
    jclass clazz;
    jmethodID getScreen;
    jmethodID takeScreenshot;
    jmethodID clickByText;
    jmethodID clickByDesc;
    jmethodID clickByCoords;
    jmethodID longClickByText;
    jmethodID longClickByDesc;
    jmethodID longClickByCoords;
    jmethodID scrollScreen;
    jmethodID scrollElement;
    jmethodID typeText;
    jmethodID pressHome;
    jmethodID pressBack;
    jmethodID pressRecents;
    jmethodID showNotifications;
    jmethodID launchApp;
    jmethodID listApps;
    jmethodID sleepMs;
    jmethodID httpPost;
    jmethodID appendLog;
};

static JniCache g_cache = {};

static void resolveJniCache(JNIEnv* env) {
    jclass cls = env->FindClass("ai/connct_screen/rn/HermesAgentRunner");
    g_cache.clazz            = (jclass)env->NewGlobalRef(cls);
    g_cache.getScreen        = env->GetStaticMethodID(cls, "nativeGetScreen",        "()Ljava/lang/String;");
    g_cache.takeScreenshot   = env->GetStaticMethodID(cls, "nativeTakeScreenshot",  "()Ljava/lang/String;");
    g_cache.clickByText      = env->GetStaticMethodID(cls, "nativeClickByText",      "(Ljava/lang/String;)Z");
    g_cache.clickByDesc      = env->GetStaticMethodID(cls, "nativeClickByDesc",      "(Ljava/lang/String;)Z");
    g_cache.clickByCoords    = env->GetStaticMethodID(cls, "nativeClickByCoords",    "(II)Z");
    g_cache.longClickByText  = env->GetStaticMethodID(cls, "nativeLongClickByText",  "(Ljava/lang/String;)Z");
    g_cache.longClickByDesc  = env->GetStaticMethodID(cls, "nativeLongClickByDesc",  "(Ljava/lang/String;)Z");
    g_cache.longClickByCoords = env->GetStaticMethodID(cls, "nativeLongClickByCoords","(II)Z");
    g_cache.scrollScreen     = env->GetStaticMethodID(cls, "nativeScrollScreen",     "(Ljava/lang/String;)Z");
    g_cache.scrollElement    = env->GetStaticMethodID(cls, "nativeScrollElement",    "(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;");
    g_cache.typeText         = env->GetStaticMethodID(cls, "nativeTypeText",         "(Ljava/lang/String;)Z");
    g_cache.pressHome        = env->GetStaticMethodID(cls, "nativePressHome",        "()Z");
    g_cache.pressBack        = env->GetStaticMethodID(cls, "nativePressBack",        "()Z");
    g_cache.pressRecents     = env->GetStaticMethodID(cls, "nativePressRecents",     "()Z");
    g_cache.showNotifications= env->GetStaticMethodID(cls, "nativeShowNotifications","()Z");
    g_cache.launchApp        = env->GetStaticMethodID(cls, "nativeLaunchApp",        "(Ljava/lang/String;)Ljava/lang/String;");
    g_cache.listApps         = env->GetStaticMethodID(cls, "nativeListApps",         "()Ljava/lang/String;");
    g_cache.sleepMs          = env->GetStaticMethodID(cls, "nativeSleepMs",          "(J)V");
    g_cache.httpPost         = env->GetStaticMethodID(cls, "nativeHttpPost",         "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;");
    g_cache.appendLog        = env->GetStaticMethodID(cls, "nativeAppendLog",        "(Ljava/lang/String;)V");
    env->DeleteLocalRef(cls);
}

// ---------------------------------------------------------------------------
// Host Function factories
// ---------------------------------------------------------------------------

// Macro to reduce boilerplate for simple void->string calls
#define REGISTER_SIMPLE_STRING_FN(rt, name, jniMethod) \
    rt.global().setProperty(rt, name, \
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, name), 0, \
            [](Runtime& rt, const Value&, const Value*, size_t) -> Value { \
                JNIEnv* env = getEnv(); \
                jstring result = (jstring)env->CallStaticObjectMethod(g_cache.clazz, g_cache.jniMethod); \
                std::string str = jstringToStd(env, result); \
                if (result) env->DeleteLocalRef(result); \
                return String::createFromUtf8(rt, str); \
            }))

#define REGISTER_SIMPLE_BOOL_FN(rt, name, jniMethod) \
    rt.global().setProperty(rt, name, \
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, name), 0, \
            [](Runtime& rt, const Value&, const Value*, size_t) -> Value { \
                JNIEnv* env = getEnv(); \
                jboolean result = env->CallStaticBooleanMethod(g_cache.clazz, g_cache.jniMethod); \
                return Value((bool)result); \
            }))

static void registerHostFunctions(Runtime& rt) {
    // get_screen() -> string
    REGISTER_SIMPLE_STRING_FN(rt, "get_screen", getScreen);

    // take_screenshot() -> string (base64 JPEG or error)
    REGISTER_SIMPLE_STRING_FN(rt, "take_screenshot", takeScreenshot);

    // click(target) -> bool
    // Supports: click("text"), click({desc:"..."}), click({x:N, y:N})
    rt.global().setProperty(rt, "click",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "click"), 1,
            [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
                if (count < 1) return Value(false);
                JNIEnv* env = getEnv();
                if (args[0].isString()) {
                    std::string text = args[0].asString(rt).utf8(rt);
                    jstring jtext = env->NewStringUTF(text.c_str());
                    jboolean res = env->CallStaticBooleanMethod(g_cache.clazz, g_cache.clickByText, jtext);
                    env->DeleteLocalRef(jtext);
                    return Value((bool)res);
                }
                if (args[0].isObject()) {
                    Object obj = args[0].asObject(rt);
                    if (obj.hasProperty(rt, "desc")) {
                        std::string desc = obj.getProperty(rt, "desc").asString(rt).utf8(rt);
                        jstring jdesc = env->NewStringUTF(desc.c_str());
                        jboolean res = env->CallStaticBooleanMethod(g_cache.clazz, g_cache.clickByDesc, jdesc);
                        env->DeleteLocalRef(jdesc);
                        return Value((bool)res);
                    }
                    if (obj.hasProperty(rt, "x") && obj.hasProperty(rt, "y")) {
                        int x = (int)obj.getProperty(rt, "x").asNumber();
                        int y = (int)obj.getProperty(rt, "y").asNumber();
                        jboolean res = env->CallStaticBooleanMethod(g_cache.clazz, g_cache.clickByCoords, x, y);
                        return Value((bool)res);
                    }
                }
                return Value(false);
            }));

    // long_click(target) -> bool  (same argument pattern as click)
    rt.global().setProperty(rt, "long_click",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "long_click"), 1,
            [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
                if (count < 1) return Value(false);
                JNIEnv* env = getEnv();
                if (args[0].isString()) {
                    std::string text = args[0].asString(rt).utf8(rt);
                    jstring jtext = env->NewStringUTF(text.c_str());
                    jboolean res = env->CallStaticBooleanMethod(g_cache.clazz, g_cache.longClickByText, jtext);
                    env->DeleteLocalRef(jtext);
                    return Value((bool)res);
                }
                if (args[0].isObject()) {
                    Object obj = args[0].asObject(rt);
                    if (obj.hasProperty(rt, "desc")) {
                        std::string desc = obj.getProperty(rt, "desc").asString(rt).utf8(rt);
                        jstring jdesc = env->NewStringUTF(desc.c_str());
                        jboolean res = env->CallStaticBooleanMethod(g_cache.clazz, g_cache.longClickByDesc, jdesc);
                        env->DeleteLocalRef(jdesc);
                        return Value((bool)res);
                    }
                    if (obj.hasProperty(rt, "x") && obj.hasProperty(rt, "y")) {
                        int x = (int)obj.getProperty(rt, "x").asNumber();
                        int y = (int)obj.getProperty(rt, "y").asNumber();
                        jboolean res = env->CallStaticBooleanMethod(g_cache.clazz, g_cache.longClickByCoords, x, y);
                        return Value((bool)res);
                    }
                }
                return Value(false);
            }));

    // scroll(direction) -> bool
    rt.global().setProperty(rt, "scroll",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "scroll"), 1,
            [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
                if (count < 1) return Value(false);
                JNIEnv* env = getEnv();
                std::string dir = args[0].asString(rt).utf8(rt);
                jstring jdir = env->NewStringUTF(dir.c_str());
                jboolean res = env->CallStaticBooleanMethod(g_cache.clazz, g_cache.scrollScreen, jdir);
                env->DeleteLocalRef(jdir);
                return Value((bool)res);
            }));

    // scroll_element(text, direction) -> string
    rt.global().setProperty(rt, "scroll_element",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "scroll_element"), 2,
            [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
                if (count < 2) return String::createFromUtf8(rt, "Error: need text and direction");
                JNIEnv* env = getEnv();
                std::string text = args[0].asString(rt).utf8(rt);
                std::string dir = args[1].asString(rt).utf8(rt);
                jstring jtext = env->NewStringUTF(text.c_str());
                jstring jdir = env->NewStringUTF(dir.c_str());
                jstring result = (jstring)env->CallStaticObjectMethod(g_cache.clazz, g_cache.scrollElement, jtext, jdir);
                std::string str = jstringToStd(env, result);
                env->DeleteLocalRef(jtext);
                env->DeleteLocalRef(jdir);
                if (result) env->DeleteLocalRef(result);
                return String::createFromUtf8(rt, str);
            }));

    // type_text(text) -> bool
    rt.global().setProperty(rt, "type_text",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "type_text"), 1,
            [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
                if (count < 1) return Value(false);
                JNIEnv* env = getEnv();
                std::string text = args[0].asString(rt).utf8(rt);
                jstring jtext = env->NewStringUTF(text.c_str());
                jboolean res = env->CallStaticBooleanMethod(g_cache.clazz, g_cache.typeText, jtext);
                env->DeleteLocalRef(jtext);
                return Value((bool)res);
            }));

    // press_home() -> bool
    REGISTER_SIMPLE_BOOL_FN(rt, "press_home", pressHome);

    // press_back() -> bool
    REGISTER_SIMPLE_BOOL_FN(rt, "press_back", pressBack);

    // press_recents() -> bool
    REGISTER_SIMPLE_BOOL_FN(rt, "press_recents", pressRecents);

    // show_notifications() -> bool
    REGISTER_SIMPLE_BOOL_FN(rt, "show_notifications", showNotifications);

    // launch_app(name) -> string
    rt.global().setProperty(rt, "launch_app",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "launch_app"), 1,
            [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
                if (count < 1) return String::createFromUtf8(rt, "Error: no app name");
                JNIEnv* env = getEnv();
                std::string name = args[0].asString(rt).utf8(rt);
                jstring jname = env->NewStringUTF(name.c_str());
                jstring result = (jstring)env->CallStaticObjectMethod(g_cache.clazz, g_cache.launchApp, jname);
                std::string str = jstringToStd(env, result);
                env->DeleteLocalRef(jname);
                if (result) env->DeleteLocalRef(result);
                return String::createFromUtf8(rt, str);
            }));

    // list_apps() -> string
    REGISTER_SIMPLE_STRING_FN(rt, "list_apps", listApps);

    // sleep(ms) -> undefined
    rt.global().setProperty(rt, "sleep",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "sleep"), 1,
            [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
                if (count < 1) return Value::undefined();
                JNIEnv* env = getEnv();
                jlong ms = (jlong)args[0].asNumber();
                env->CallStaticVoidMethod(g_cache.clazz, g_cache.sleepMs, ms);
                return Value::undefined();
            }));

    // log(msg) -> undefined
    rt.global().setProperty(rt, "log",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "log"), 1,
            [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
                if (count < 1) return Value::undefined();
                JNIEnv* env = getEnv();
                std::string msg = args[0].asString(rt).utf8(rt);
                LOGI("[JS] %s", msg.c_str());
                jstring jmsg = env->NewStringUTF(msg.c_str());
                env->CallStaticVoidMethod(g_cache.clazz, g_cache.appendLog, jmsg);
                env->DeleteLocalRef(jmsg);
                return Value::undefined();
            }));

    // http_post(url, headersJson, body) -> string
    rt.global().setProperty(rt, "http_post",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "http_post"), 3,
            [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
                if (count < 3) return String::createFromUtf8(rt, "{\"error\":\"need url, headers, body\"}");
                JNIEnv* env = getEnv();
                std::string url = args[0].asString(rt).utf8(rt);
                std::string headers = args[1].asString(rt).utf8(rt);
                std::string body = args[2].asString(rt).utf8(rt);
                jstring jurl = env->NewStringUTF(url.c_str());
                jstring jheaders = env->NewStringUTF(headers.c_str());
                jstring jbody = env->NewStringUTF(body.c_str());
                jstring result = (jstring)env->CallStaticObjectMethod(
                    g_cache.clazz, g_cache.httpPost, jurl, jheaders, jbody);
                std::string str = jstringToStd(env, result);
                env->DeleteLocalRef(jurl);
                env->DeleteLocalRef(jheaders);
                env->DeleteLocalRef(jbody);
                if (result) env->DeleteLocalRef(result);
                return String::createFromUtf8(rt, str);
            }));
}

// ---------------------------------------------------------------------------
// JNI exports (called from HermesAgentRunner.java)
// ---------------------------------------------------------------------------

// We store a single runtime pointer. For simplicity only one agent runs at a time.
static std::unique_ptr<HermesRuntime> g_runtime;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
    g_jvm = vm;
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL
Java_ai_connct_1screen_rn_HermesAgentRunner_nativeCreateRuntime(JNIEnv* env, jclass) {
    LOGI("Creating standalone Hermes runtime");
    resolveJniCache(env);
    g_runtime = makeHermesRuntime();
    registerHostFunctions(*g_runtime);
    LOGI("Hermes runtime created and host functions registered");
}

JNIEXPORT jstring JNICALL
Java_ai_connct_1screen_rn_HermesAgentRunner_nativeEvaluateJS(JNIEnv* env, jclass, jstring jsCode, jstring sourceURL) {
    if (!g_runtime) {
        return env->NewStringUTF("{\"error\":\"Runtime not created\"}");
    }
    std::string code = jstringToStd(env, jsCode);
    std::string url = jstringToStd(env, sourceURL);

    try {
        auto buffer = std::make_shared<StringBuffer>(code);
        Value result = g_runtime->evaluateJavaScript(buffer, url);

        if (result.isString()) {
            std::string str = result.asString(*g_runtime).utf8(*g_runtime);
            return env->NewStringUTF(str.c_str());
        } else if (result.isNumber()) {
            std::string str = std::to_string(result.asNumber());
            return env->NewStringUTF(str.c_str());
        } else if (result.isBool()) {
            return env->NewStringUTF(result.getBool() ? "true" : "false");
        } else if (result.isUndefined()) {
            return env->NewStringUTF("undefined");
        } else if (result.isNull()) {
            return env->NewStringUTF("null");
        } else {
            // For objects, try JSON.stringify
            try {
                auto json = g_runtime->global()
                    .getPropertyAsObject(*g_runtime, "JSON")
                    .getPropertyAsFunction(*g_runtime, "stringify")
                    .call(*g_runtime, result);
                if (json.isString()) {
                    std::string str = json.asString(*g_runtime).utf8(*g_runtime);
                    return env->NewStringUTF(str.c_str());
                }
            } catch (...) {}
            return env->NewStringUTF("[object]");
        }
    } catch (const JSError& e) {
        std::string error = std::string("[JS Error] ") + e.what();
        LOGE("%s", error.c_str());
        return env->NewStringUTF(error.c_str());
    } catch (const std::exception& e) {
        std::string error = std::string("[Native Error] ") + e.what();
        LOGE("%s", error.c_str());
        return env->NewStringUTF(error.c_str());
    }
}

JNIEXPORT void JNICALL
Java_ai_connct_1screen_rn_HermesAgentRunner_nativeDestroyRuntime(JNIEnv* env, jclass) {
    LOGI("Destroying standalone Hermes runtime");
    g_runtime.reset();
    if (g_cache.clazz) {
        env->DeleteGlobalRef(g_cache.clazz);
        g_cache = {};
    }
}

} // extern "C"
