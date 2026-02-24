cmake_minimum_required(VERSION 3.13)
project(qwen_tts C)

set(QWEN_TTS_SOURCES
    qwen_tts.c
    qwen_tts_quant.c
    qwen_tts_kernels.c
    qwen_tts_kernels_neon.c
    qwen_tts_kernels_ops.c
    qwen_tts_talker.c
    qwen_tts_codec.c
    qwen_tts_safetensors.c
    qwen_tts_audio.c
)

add_library(qwen_tts_static STATIC ${QWEN_TTS_SOURCES})
target_include_directories(qwen_tts_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(qwen_tts_static PRIVATE -O3 -ffast-math -flto)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(qwen_tts_static PRIVATE -march=armv8.2-a+dotprod+fp16)
endif()

find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_compile_definitions(qwen_tts_static PRIVATE USE_OPENMP)
    target_link_libraries(qwen_tts_static PRIVATE OpenMP::OpenMP_C)
endif()

target_link_libraries(qwen_tts_static PRIVATE m)
