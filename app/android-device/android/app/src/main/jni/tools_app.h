/**
 * tools_app.h
 *
 * Declares JNI cache resolution and host function registration
 * for the "app" agent type (Android accessibility / app automation).
 */

#pragma once

#include <jni.h>
#include <jsi/jsi.h>

// Cache for AppToolsHost.java static methods.
struct AppJniCache {
    jclass clazz;              // AppToolsHost.java
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
};

// Resolve JNI method IDs for AppToolsHost. Call once from the main thread.
void resolveAppJniCache(JNIEnv* env);

// Register app-automation host functions on the given JS runtime.
void registerAppTools(facebook::jsi::Runtime& rt);
