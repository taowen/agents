cmake_minimum_required(VERSION 3.13)
project(hermesruntime)

# --- Multi-agent Hermes Runtime library ---
# A shared library that embeds independent Hermes runtimes (one per agent type)
# inside the AccessibilityService process, free from Android's background-app
# JS thread restrictions.

# Build libjsi.so from source (replaces the copy from react-android).
# The prebuilt libhermesvm.so dynamically links against libjsi.so at runtime.
set(JSI_DIR "${CMAKE_SOURCE_DIR}/jsi")

add_library(jsi SHARED ${JSI_DIR}/jsi/jsi.cpp)
target_include_directories(jsi PUBLIC ${JSI_DIR})

add_library(hermesruntime SHARED
    hermes_runtime.cpp
    tools_app.cpp
    tools_browser.cpp
)

find_package(hermes-engine REQUIRED CONFIG)

target_link_libraries(hermesruntime
    hermes-engine::hermesvm   # Hermes runtime (makeHermesRuntime)
    jsi                       # JSI API (jsi::Runtime, HostFunction, etc.)
    log                       # Android __android_log_print
)
