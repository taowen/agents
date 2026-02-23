/**
 * hermes_runtime.cpp
 *
 * Multi-agent Hermes runtime manager. Maintains a map of named runtimes
 * (one per agent type), registers shared "common" host functions, and
 * delegates agent-specific tool registration to separate translation units.
 *
 * JNI exports are called from ai.connct_screen.rn.HermesRuntime.
 */

#include "hermes_runtime.h"
#include "tools_app.h"
#include "tools_browser.h"

using namespace facebook::jsi;
using namespace facebook::hermes;

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

JavaVM* g_jvm = nullptr;
CommonJniCache g_common_cache = {};
std::unordered_map<std::string, RuntimeEntry> g_runtimes;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

JNIEnv* getEnv() {
    JNIEnv* env = nullptr;
    if (g_jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) == JNI_EDETACHED) {
        g_jvm->AttachCurrentThread(&env, nullptr);
    }
    return env;
}

std::string jstringToStd(JNIEnv* env, jstring js) {
    if (!js) return "";
    const char* chars = env->GetStringUTFChars(js, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(js, chars);
    return result;
}

static void resolveCommonJniCache(JNIEnv* env) {
    jclass cls = env->FindClass("ai/connct_screen/rn/HermesRuntime");
    g_common_cache.clazz        = (jclass)env->NewGlobalRef(cls);
    g_common_cache.httpPost     = env->GetStaticMethodID(cls, "nativeHttpPost",     "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;");
    g_common_cache.appendLog    = env->GetStaticMethodID(cls, "nativeAppendLog",    "(Ljava/lang/String;)V");
    g_common_cache.updateStatus = env->GetStaticMethodID(cls, "nativeUpdateStatus", "(Ljava/lang/String;)V");
    g_common_cache.askUser      = env->GetStaticMethodID(cls, "nativeAskUser",      "(Ljava/lang/String;)Ljava/lang/String;");
    g_common_cache.hideOverlay  = env->GetStaticMethodID(cls, "nativeHideOverlay",  "()V");
    g_common_cache.sleepMs      = env->GetStaticMethodID(cls, "nativeSleepMs",      "(J)V");
    g_common_cache.speak        = env->GetStaticMethodID(cls, "nativeSpeak",        "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;");
    env->DeleteLocalRef(cls);
}

// ---------------------------------------------------------------------------
// Common host functions (shared by all agent types)
// ---------------------------------------------------------------------------

void registerCommonTools(Runtime& rt) {
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
                    g_common_cache.clazz, g_common_cache.httpPost, jurl, jheaders, jbody);
                std::string str = jstringToStd(env, result);
                env->DeleteLocalRef(jurl);
                env->DeleteLocalRef(jheaders);
                env->DeleteLocalRef(jbody);
                if (result) env->DeleteLocalRef(result);
                return String::createFromUtf8(rt, str);
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
                env->CallStaticVoidMethod(g_common_cache.clazz, g_common_cache.appendLog, jmsg);
                env->DeleteLocalRef(jmsg);
                return Value::undefined();
            }));

    // sleep(ms) -> undefined
    rt.global().setProperty(rt, "sleep",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "sleep"), 1,
            [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
                if (count < 1) return Value::undefined();
                JNIEnv* env = getEnv();
                jlong ms = (jlong)args[0].asNumber();
                env->CallStaticVoidMethod(g_common_cache.clazz, g_common_cache.sleepMs, ms);
                return Value::undefined();
            }));

    // update_status(text) -> undefined
    rt.global().setProperty(rt, "update_status",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "update_status"), 1,
            [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
                if (count < 1) return Value::undefined();
                JNIEnv* env = getEnv();
                std::string text = args[0].asString(rt).utf8(rt);
                jstring jtext = env->NewStringUTF(text.c_str());
                env->CallStaticVoidMethod(g_common_cache.clazz, g_common_cache.updateStatus, jtext);
                env->DeleteLocalRef(jtext);
                return Value::undefined();
            }));

    // ask_user(question) -> string
    rt.global().setProperty(rt, "ask_user",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "ask_user"), 1,
            [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
                if (count < 1) return String::createFromUtf8(rt, "abandoned");
                JNIEnv* env = getEnv();
                std::string question = args[0].asString(rt).utf8(rt);
                jstring jquestion = env->NewStringUTF(question.c_str());
                jstring result = (jstring)env->CallStaticObjectMethod(g_common_cache.clazz, g_common_cache.askUser, jquestion);
                std::string str = jstringToStd(env, result);
                env->DeleteLocalRef(jquestion);
                if (result) env->DeleteLocalRef(result);
                return String::createFromUtf8(rt, str);
            }));

    // hide_overlay() -> undefined
    rt.global().setProperty(rt, "hide_overlay",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "hide_overlay"), 0,
            [](Runtime& rt, const Value&, const Value*, size_t) -> Value {
                JNIEnv* env = getEnv();
                env->CallStaticVoidMethod(g_common_cache.clazz, g_common_cache.hideOverlay);
                return Value::undefined();
            }));

    // speak(text, speaker?, language?) -> bool
    rt.global().setProperty(rt, "speak",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "speak"), 3,
            [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
                if (count < 1) return Value(false);
                JNIEnv* env = getEnv();
                std::string text = args[0].asString(rt).utf8(rt);
                jstring jtext = env->NewStringUTF(text.c_str());
                jstring jspeaker = nullptr;
                jstring jlanguage = nullptr;
                if (count >= 2 && !args[1].isUndefined() && !args[1].isNull()) {
                    std::string speaker = args[1].asString(rt).utf8(rt);
                    jspeaker = env->NewStringUTF(speaker.c_str());
                }
                if (count >= 3 && !args[2].isUndefined() && !args[2].isNull()) {
                    std::string language = args[2].asString(rt).utf8(rt);
                    jlanguage = env->NewStringUTF(language.c_str());
                }
                jstring result = (jstring)env->CallStaticObjectMethod(
                    g_common_cache.clazz, g_common_cache.speak, jtext, jspeaker, jlanguage);
                std::string str = jstringToStd(env, result);
                env->DeleteLocalRef(jtext);
                if (jspeaker) env->DeleteLocalRef(jspeaker);
                if (jlanguage) env->DeleteLocalRef(jlanguage);
                if (result) env->DeleteLocalRef(result);
                return Value(str == "true");
            }));
}

// ---------------------------------------------------------------------------
// JNI exports (called from HermesRuntime.java)
// ---------------------------------------------------------------------------

static bool g_caches_resolved = false;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
    g_jvm = vm;
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL
Java_ai_connct_1screen_rn_HermesRuntime_nativeCreateRuntime(JNIEnv* env, jclass, jstring jAgentType) {
    std::string agentType = jstringToStd(env, jAgentType);
    LOGI("Creating Hermes runtime for agent type: %s", agentType.c_str());

    // Resolve JNI caches once
    if (!g_caches_resolved) {
        resolveCommonJniCache(env);
        resolveAppJniCache(env);
        resolveBrowserJniCache(env);
        g_caches_resolved = true;
    }

    // Destroy existing runtime for this agent type if any
    auto it = g_runtimes.find(agentType);
    if (it != g_runtimes.end()) {
        LOGI("Destroying existing runtime for agent type: %s", agentType.c_str());
        g_runtimes.erase(it);
    }

    auto runtime = makeHermesRuntime();
    Runtime& rt = *runtime;

    // Register common tools
    registerCommonTools(rt);

    // Register agent-specific tools
    if (agentType == "app") {
        registerAppTools(rt);
    } else if (agentType == "browser") {
        registerBrowserTools(rt);
    }

    RuntimeEntry entry;
    entry.runtime = std::move(runtime);
    entry.agentType = agentType;
    g_runtimes[agentType] = std::move(entry);

    LOGI("Hermes runtime created for agent type: %s", agentType.c_str());
}

JNIEXPORT jstring JNICALL
Java_ai_connct_1screen_rn_HermesRuntime_nativeEvaluateJS(JNIEnv* env, jclass, jstring jAgentType, jstring jsCode, jstring sourceURL) {
    std::string agentType = jstringToStd(env, jAgentType);
    auto it = g_runtimes.find(agentType);
    if (it == g_runtimes.end() || !it->second.runtime) {
        return env->NewStringUTF("{\"error\":\"Runtime not created\"}");
    }

    auto& runtime = it->second.runtime;
    std::string code = jstringToStd(env, jsCode);
    std::string url = jstringToStd(env, sourceURL);

    try {
        auto buffer = std::make_shared<StringBuffer>(code);
        Value result = runtime->evaluateJavaScript(buffer, url);

        if (result.isString()) {
            std::string str = result.asString(*runtime).utf8(*runtime);
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
                auto json = runtime->global()
                    .getPropertyAsObject(*runtime, "JSON")
                    .getPropertyAsFunction(*runtime, "stringify")
                    .call(*runtime, result);
                if (json.isString()) {
                    std::string str = json.asString(*runtime).utf8(*runtime);
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
Java_ai_connct_1screen_rn_HermesRuntime_nativeDestroyRuntime(JNIEnv* env, jclass, jstring jAgentType) {
    std::string agentType = jstringToStd(env, jAgentType);
    LOGI("Destroying Hermes runtime for agent type: %s", agentType.c_str());
    g_runtimes.erase(agentType);
}

} // extern "C"
