/**
 * tools_browser.h
 *
 * Declares JNI cache resolution and host function registration
 * for the "browser" agent type (WebView-based web automation).
 */

#pragma once

#include <jni.h>
#include <jsi/jsi.h>

// Cache for BrowserToolsHost.java static methods.
struct BrowserJniCache {
    jclass clazz;              // BrowserToolsHost.java
    jmethodID getPage;
    jmethodID clickElement;
    jmethodID typeText;
    jmethodID gotoUrl;
    jmethodID scrollPage;
    jmethodID goBack;
    jmethodID takeScreenshot;
    jmethodID switchUa;
    jmethodID setViewport;
};

// Resolve JNI method IDs for BrowserToolsHost. Call once from the main thread.
void resolveBrowserJniCache(JNIEnv* env);

// Register browser-automation host functions on the given JS runtime.
void registerBrowserTools(facebook::jsi::Runtime& rt);
