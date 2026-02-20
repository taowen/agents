cmake_minimum_required(VERSION 3.13)
project(appmodules)

# Include the standard React Native application build setup.
# This creates the "appmodules" shared library with RN's OnLoad.cpp (or our custom one).
include(${REACT_ANDROID_DIR}/cmake-utils/ReactNative-application.cmake)

# --- Standalone Hermes Agent library ---
# A separate shared library that embeds an independent Hermes runtime
# for running the agent loop inside the AccessibilityService process,
# free from Android's background-app JS thread restrictions.

add_library(hermesagent SHARED agent/standalone_hermes.cpp)

find_package(hermes-engine REQUIRED CONFIG)

target_link_libraries(hermesagent
    hermes-engine::hermesvm   # Hermes runtime (makeHermesRuntime)
    ReactAndroid::jsi         # JSI API (jsi::Runtime, HostFunction, etc.)
    log                       # Android __android_log_print
)
