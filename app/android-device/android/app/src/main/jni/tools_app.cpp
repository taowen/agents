/**
 * tools_app.cpp
 *
 * Host function registration for the "app" agent type.
 * These functions provide Android accessibility / app automation
 * capabilities: screen reading, clicking, scrolling, typing, etc.
 *
 * JNI callbacks go to ai.connct_screen.rn.AppToolsHost.
 */

#include "tools_app.h"
#include "hermes_runtime.h"

using namespace facebook::jsi;

// ---------------------------------------------------------------------------
// App JNI cache (AppToolsHost.java)
// ---------------------------------------------------------------------------

static AppJniCache g_app_cache = {};

void resolveAppJniCache(JNIEnv* env) {
    jclass cls = env->FindClass("ai/connct_screen/rn/AppToolsHost");
    g_app_cache.clazz            = (jclass)env->NewGlobalRef(cls);
    g_app_cache.getScreen        = env->GetStaticMethodID(cls, "nativeGetScreen",        "()Ljava/lang/String;");
    g_app_cache.takeScreenshot   = env->GetStaticMethodID(cls, "nativeTakeScreenshot",   "()Ljava/lang/String;");
    g_app_cache.clickByText      = env->GetStaticMethodID(cls, "nativeClickByText",      "(Ljava/lang/String;)Z");
    g_app_cache.clickByDesc      = env->GetStaticMethodID(cls, "nativeClickByDesc",      "(Ljava/lang/String;)Z");
    g_app_cache.clickByCoords    = env->GetStaticMethodID(cls, "nativeClickByCoords",    "(II)Z");
    g_app_cache.longClickByText  = env->GetStaticMethodID(cls, "nativeLongClickByText",  "(Ljava/lang/String;)Z");
    g_app_cache.longClickByDesc  = env->GetStaticMethodID(cls, "nativeLongClickByDesc",  "(Ljava/lang/String;)Z");
    g_app_cache.longClickByCoords = env->GetStaticMethodID(cls, "nativeLongClickByCoords","(II)Z");
    g_app_cache.scrollScreen     = env->GetStaticMethodID(cls, "nativeScrollScreen",     "(Ljava/lang/String;)Z");
    g_app_cache.scrollElement    = env->GetStaticMethodID(cls, "nativeScrollElement",    "(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;");
    g_app_cache.typeText         = env->GetStaticMethodID(cls, "nativeTypeText",         "(Ljava/lang/String;)Z");
    g_app_cache.pressHome        = env->GetStaticMethodID(cls, "nativePressHome",        "()Z");
    g_app_cache.pressBack        = env->GetStaticMethodID(cls, "nativePressBack",        "()Z");
    g_app_cache.pressRecents     = env->GetStaticMethodID(cls, "nativePressRecents",     "()Z");
    g_app_cache.showNotifications= env->GetStaticMethodID(cls, "nativeShowNotifications","()Z");
    g_app_cache.launchApp        = env->GetStaticMethodID(cls, "nativeLaunchApp",        "(Ljava/lang/String;)Ljava/lang/String;");
    g_app_cache.listApps         = env->GetStaticMethodID(cls, "nativeListApps",         "()Ljava/lang/String;");
    env->DeleteLocalRef(cls);
}

// ---------------------------------------------------------------------------
// Macros for simple host functions
// ---------------------------------------------------------------------------

#define REGISTER_APP_STRING_FN(rt, name, jniMethod) \
    rt.global().setProperty(rt, name, \
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, name), 0, \
            [](Runtime& rt, const Value&, const Value*, size_t) -> Value { \
                JNIEnv* env = getEnv(); \
                jstring result = (jstring)env->CallStaticObjectMethod(g_app_cache.clazz, g_app_cache.jniMethod); \
                std::string str = jstringToStd(env, result); \
                if (result) env->DeleteLocalRef(result); \
                return String::createFromUtf8(rt, str); \
            }))

#define REGISTER_APP_BOOL_FN(rt, name, jniMethod) \
    rt.global().setProperty(rt, name, \
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, name), 0, \
            [](Runtime& rt, const Value&, const Value*, size_t) -> Value { \
                JNIEnv* env = getEnv(); \
                jboolean result = env->CallStaticBooleanMethod(g_app_cache.clazz, g_app_cache.jniMethod); \
                return Value((bool)result); \
            }))

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void registerAppTools(Runtime& rt) {
    // get_screen() -> string
    REGISTER_APP_STRING_FN(rt, "get_screen", getScreen);

    // take_screenshot() -> string (base64 JPEG or error)
    REGISTER_APP_STRING_FN(rt, "take_screenshot", takeScreenshot);

    // click(target) -> bool
    rt.global().setProperty(rt, "click",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "click"), 1,
            [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
                if (count < 1) return Value(false);
                JNIEnv* env = getEnv();
                if (args[0].isString()) {
                    std::string text = args[0].asString(rt).utf8(rt);
                    jstring jtext = env->NewStringUTF(text.c_str());
                    jboolean res = env->CallStaticBooleanMethod(g_app_cache.clazz, g_app_cache.clickByText, jtext);
                    env->DeleteLocalRef(jtext);
                    return Value((bool)res);
                }
                if (args[0].isObject()) {
                    Object obj = args[0].asObject(rt);
                    if (obj.hasProperty(rt, "desc")) {
                        std::string desc = obj.getProperty(rt, "desc").asString(rt).utf8(rt);
                        jstring jdesc = env->NewStringUTF(desc.c_str());
                        jboolean res = env->CallStaticBooleanMethod(g_app_cache.clazz, g_app_cache.clickByDesc, jdesc);
                        env->DeleteLocalRef(jdesc);
                        return Value((bool)res);
                    }
                    if (obj.hasProperty(rt, "x") && obj.hasProperty(rt, "y")) {
                        int x = (int)obj.getProperty(rt, "x").asNumber();
                        int y = (int)obj.getProperty(rt, "y").asNumber();
                        jboolean res = env->CallStaticBooleanMethod(g_app_cache.clazz, g_app_cache.clickByCoords, x, y);
                        return Value((bool)res);
                    }
                }
                return Value(false);
            }));

    // long_click(target) -> bool
    rt.global().setProperty(rt, "long_click",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "long_click"), 1,
            [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
                if (count < 1) return Value(false);
                JNIEnv* env = getEnv();
                if (args[0].isString()) {
                    std::string text = args[0].asString(rt).utf8(rt);
                    jstring jtext = env->NewStringUTF(text.c_str());
                    jboolean res = env->CallStaticBooleanMethod(g_app_cache.clazz, g_app_cache.longClickByText, jtext);
                    env->DeleteLocalRef(jtext);
                    return Value((bool)res);
                }
                if (args[0].isObject()) {
                    Object obj = args[0].asObject(rt);
                    if (obj.hasProperty(rt, "desc")) {
                        std::string desc = obj.getProperty(rt, "desc").asString(rt).utf8(rt);
                        jstring jdesc = env->NewStringUTF(desc.c_str());
                        jboolean res = env->CallStaticBooleanMethod(g_app_cache.clazz, g_app_cache.longClickByDesc, jdesc);
                        env->DeleteLocalRef(jdesc);
                        return Value((bool)res);
                    }
                    if (obj.hasProperty(rt, "x") && obj.hasProperty(rt, "y")) {
                        int x = (int)obj.getProperty(rt, "x").asNumber();
                        int y = (int)obj.getProperty(rt, "y").asNumber();
                        jboolean res = env->CallStaticBooleanMethod(g_app_cache.clazz, g_app_cache.longClickByCoords, x, y);
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
                jboolean res = env->CallStaticBooleanMethod(g_app_cache.clazz, g_app_cache.scrollScreen, jdir);
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
                jstring result = (jstring)env->CallStaticObjectMethod(g_app_cache.clazz, g_app_cache.scrollElement, jtext, jdir);
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
                jboolean res = env->CallStaticBooleanMethod(g_app_cache.clazz, g_app_cache.typeText, jtext);
                env->DeleteLocalRef(jtext);
                return Value((bool)res);
            }));

    // press_home() -> bool
    REGISTER_APP_BOOL_FN(rt, "press_home", pressHome);

    // press_back() -> bool
    REGISTER_APP_BOOL_FN(rt, "press_back", pressBack);

    // press_recents() -> bool
    REGISTER_APP_BOOL_FN(rt, "press_recents", pressRecents);

    // show_notifications() -> bool
    REGISTER_APP_BOOL_FN(rt, "show_notifications", showNotifications);

    // launch_app(name) -> string
    rt.global().setProperty(rt, "launch_app",
        Function::createFromHostFunction(rt, PropNameID::forAscii(rt, "launch_app"), 1,
            [](Runtime& rt, const Value&, const Value* args, size_t count) -> Value {
                if (count < 1) return String::createFromUtf8(rt, "Error: no app name");
                JNIEnv* env = getEnv();
                std::string name = args[0].asString(rt).utf8(rt);
                jstring jname = env->NewStringUTF(name.c_str());
                jstring result = (jstring)env->CallStaticObjectMethod(g_app_cache.clazz, g_app_cache.launchApp, jname);
                std::string str = jstringToStd(env, result);
                env->DeleteLocalRef(jname);
                if (result) env->DeleteLocalRef(result);
                return String::createFromUtf8(rt, str);
            }));

    // list_apps() -> string
    REGISTER_APP_STRING_FN(rt, "list_apps", listApps);
}
