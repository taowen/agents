cmake_minimum_required(VERSION 3.13)
project(hermesagent)

# --- Standalone Hermes Agent library ---
# A separate shared library that embeds an independent Hermes runtime
# for running the agent loop inside the AccessibilityService process,
# free from Android's background-app JS thread restrictions.

# Build libjsi.so from source (replaces the copy from react-android).
# The prebuilt libhermesvm.so dynamically links against libjsi.so at runtime.
set(JSI_DIR "${CMAKE_SOURCE_DIR}/jsi")

add_library(jsi SHARED ${JSI_DIR}/jsi/jsi.cpp)
target_include_directories(jsi PUBLIC ${JSI_DIR})

add_library(hermesagent SHARED agent/standalone_hermes.cpp)

find_package(hermes-engine REQUIRED CONFIG)

target_link_libraries(hermesagent
    hermes-engine::hermesvm   # Hermes runtime (makeHermesRuntime)
    jsi                       # JSI API (jsi::Runtime, HostFunction, etc.)
    log                       # Android __android_log_print
)
