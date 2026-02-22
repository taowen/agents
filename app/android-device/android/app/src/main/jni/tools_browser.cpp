/**
 * tools_browser.cpp
 *
 * Host function registration for the "browser" agent type.
 * These functions provide WebView-based web automation capabilities:
 * DOM reading, clicking, typing, navigation, screenshots, etc.
 *
 * JNI callbacks go to ai.connct_screen.rn.BrowserToolsHost.
 */

#include "tools_browser.h"
#include "hermes_runtime.h"

using namespace facebook::jsi;

// ---------------------------------------------------------------------------
// Browser JNI cache (BrowserToolsHost.java)
// ---------------------------------------------------------------------------

static BrowserJniCache g_browser_cache = {};

void resolveBrowserJniCache(JNIEnv* env) {
    jclass cls = env->FindClass("ai/connct_screen/rn/BrowserToolsHost");
    g_browser_cache.clazz          = (jclass)env->NewGlobalRef(cls);
    g_browser_cache.getPage        = env->GetStaticMethodID(cls, "nativeGetPage",        "()Ljava/lang/String;");
    g_browser_cache.clickElement   = env->GetStaticMethodID(cls, "nativeClickElement",   "(I)Z");
    g_browser_cache.typeText       = env->GetStaticMethodID(cls, "nativeTypeText",       "(ILjava/lang/String;)Z");
    g_browser_cache.gotoUrl        = env->GetStaticMethodID(cls, "nativeGotoUrl",        "(Ljava/lang/String;)Z");
    g_browser_cache.scrollPage     = env->GetStaticMethodID(cls, "nativeScrollPage",     "(Ljava/lang/String;)Z");
    g_browser_cache.goBack         = env->GetStaticMethodID(cls, "nativeGoBack",         "()Z");
    g_browser_cache.takeScreenshot = env->GetStaticMethodID(cls, "nativeTakeScreenshot", "()Ljava/lang/String;");
    g_browser_cache.switchUa       = env->GetStaticMethodID(cls, "nativeSwitchUa",       "(Ljava/lang/String;)Ljava/lang/String;");
    g_browser_cache.setViewport    = env->GetStaticMethodID(cls, "nativeSetViewport",    "(II)Ljava/lang/String;");
    env->DeleteLocalRef(cls);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void registerBrowserTools(Runtime& rt) {
    // get_page() -> string (DOM tree with interactive element IDs)
    rt.global().setProperty(rt, "get_page",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "get_page"), 0,
            [](Runtime& rt, const Value&, const Value*, size_t) -> Value {
                JNIEnv* env = getEnv();
                jstring result = (jstring)env->CallStaticObjectMethod(g_browser_cache.clazz, g_browser_cache.getPage);
                std::string str = jstringToStd(env, result);
                if (result) env->DeleteLocalRef(result);
                return String::createFromUtf8(rt, str);
            }));

    // click_element(id) -> bool
    rt.global().setProperty(rt, "click_element",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "click_element"), 1,
            [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
                if (count < 1) return Value(false);
                JNIEnv* env = getEnv();
                int id = (int)args[0].asNumber();
                jboolean result = env->CallStaticBooleanMethod(g_browser_cache.clazz, g_browser_cache.clickElement, id);
                return Value((bool)result);
            }));

    // type_text(id, text) -> bool
    rt.global().setProperty(rt, "type_text",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "type_text"), 2,
            [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
                if (count < 2) return Value(false);
                JNIEnv* env = getEnv();
                int id = (int)args[0].asNumber();
                std::string text = args[1].asString(rt).utf8(rt);
                jstring jtext = env->NewStringUTF(text.c_str());
                jboolean result = env->CallStaticBooleanMethod(g_browser_cache.clazz, g_browser_cache.typeText, id, jtext);
                env->DeleteLocalRef(jtext);
                return Value((bool)result);
            }));

    // goto_url(url) -> bool
    rt.global().setProperty(rt, "goto_url",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "goto_url"), 1,
            [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
                if (count < 1) return Value(false);
                JNIEnv* env = getEnv();
                std::string url = args[0].asString(rt).utf8(rt);
                jstring jurl = env->NewStringUTF(url.c_str());
                jboolean result = env->CallStaticBooleanMethod(g_browser_cache.clazz, g_browser_cache.gotoUrl, jurl);
                env->DeleteLocalRef(jurl);
                return Value((bool)result);
            }));

    // scroll_page(direction) -> bool
    rt.global().setProperty(rt, "scroll_page",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "scroll_page"), 1,
            [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
                if (count < 1) return Value(false);
                JNIEnv* env = getEnv();
                std::string dir = args[0].asString(rt).utf8(rt);
                jstring jdir = env->NewStringUTF(dir.c_str());
                jboolean result = env->CallStaticBooleanMethod(g_browser_cache.clazz, g_browser_cache.scrollPage, jdir);
                env->DeleteLocalRef(jdir);
                return Value((bool)result);
            }));

    // go_back() -> bool
    rt.global().setProperty(rt, "go_back",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "go_back"), 0,
            [](Runtime& rt, const Value&, const Value*, size_t) -> Value {
                JNIEnv* env = getEnv();
                jboolean result = env->CallStaticBooleanMethod(g_browser_cache.clazz, g_browser_cache.goBack);
                return Value((bool)result);
            }));

    // take_screenshot() -> string (base64 JPEG)
    rt.global().setProperty(rt, "take_screenshot",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "take_screenshot"), 0,
            [](Runtime& rt, const Value&, const Value*, size_t) -> Value {
                JNIEnv* env = getEnv();
                jstring result = (jstring)env->CallStaticObjectMethod(g_browser_cache.clazz, g_browser_cache.takeScreenshot);
                std::string str = jstringToStd(env, result);
                if (result) env->DeleteLocalRef(result);
                return String::createFromUtf8(rt, str);
            }));

    // switch_ua(mode) -> string
    rt.global().setProperty(rt, "switch_ua",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "switch_ua"), 1,
            [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
                if (count < 1) return String::createFromUtf8(rt, "Error: no mode");
                JNIEnv* env = getEnv();
                std::string mode = args[0].asString(rt).utf8(rt);
                jstring jmode = env->NewStringUTF(mode.c_str());
                jstring result = (jstring)env->CallStaticObjectMethod(g_browser_cache.clazz, g_browser_cache.switchUa, jmode);
                std::string str = jstringToStd(env, result);
                env->DeleteLocalRef(jmode);
                if (result) env->DeleteLocalRef(result);
                return String::createFromUtf8(rt, str);
            }));

    // set_viewport(width, height) -> string
    rt.global().setProperty(rt, "set_viewport",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "set_viewport"), 2,
            [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
                if (count < 2) return String::createFromUtf8(rt, "Error: need width, height");
                JNIEnv* env = getEnv();
                int w = (int)args[0].asNumber();
                int h = (int)args[1].asNumber();
                jstring result = (jstring)env->CallStaticObjectMethod(g_browser_cache.clazz, g_browser_cache.setViewport, w, h);
                std::string str = jstringToStd(env, result);
                if (result) env->DeleteLocalRef(result);
                return String::createFromUtf8(rt, str);
            }));
}
