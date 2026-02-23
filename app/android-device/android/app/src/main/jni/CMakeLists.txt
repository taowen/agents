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

# --- Qwen ASR JNI library ---
set(QWEN_ASR_DIR "${CMAKE_SOURCE_DIR}/qwen-asr")
add_subdirectory(${QWEN_ASR_DIR} ${CMAKE_CURRENT_BINARY_DIR}/qwen_asr)

add_library(qwenasr_jni SHARED qwen_asr_jni.cpp)
target_compile_options(qwenasr_jni PRIVATE -O3 -flto)
target_link_options(qwenasr_jni PRIVATE -flto)
target_link_libraries(qwenasr_jni qwen_asr_static log)

# --- Qwen TTS JNI library ---
# Pure C inference engine for Qwen3-TTS (0.6B CustomVoice).
# No BLAS dependency; uses OpenMP for threading and NEON on ARM.
set(QWEN_TTS_DIR "${CMAKE_SOURCE_DIR}/qwen-tts")

add_library(qwentts SHARED
    ${QWEN_TTS_DIR}/qwen_tts.c
    ${QWEN_TTS_DIR}/qwen_tts_kernels.c
    ${QWEN_TTS_DIR}/qwen_tts_talker.c
    ${QWEN_TTS_DIR}/qwen_tts_codec.c
    ${QWEN_TTS_DIR}/qwen_tts_safetensors.c
    ${QWEN_TTS_DIR}/qwen_tts_audio.c
    ${QWEN_TTS_DIR}/qwen_tts_quant.c
    ${QWEN_TTS_DIR}/qwen_tts_jni.c
)

target_include_directories(qwentts PRIVATE ${QWEN_TTS_DIR})
target_compile_options(qwentts PRIVATE -DUSE_OPENMP -fopenmp -O3 -ffast-math -march=armv8.2-a+dotprod)
target_link_libraries(qwentts log m OpenMP::OpenMP_C)

find_package(OpenMP REQUIRED)
