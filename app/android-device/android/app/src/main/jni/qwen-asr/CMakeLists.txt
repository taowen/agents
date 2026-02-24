cmake_minimum_required(VERSION 3.13)
project(qwen_asr C)

set(QWEN_ASR_SOURCES
    qwen_asr.c
    qwen_asr_kernels.c
    qwen_asr_kernels_generic.c
    qwen_asr_kernels_neon.c
    qwen_asr_kernels_avx.c
    qwen_asr_quant.c
    qwen_asr_audio.c
    qwen_asr_encoder.c
    qwen_asr_decoder.c
    qwen_asr_tokenizer.c
    qwen_asr_gguf.c
)

add_library(qwen_asr_static STATIC ${QWEN_ASR_SOURCES})
target_include_directories(qwen_asr_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(qwen_asr_static PRIVATE -O3 -ffast-math -flto)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(qwen_asr_static PRIVATE -march=armv8.2-a+dotprod)
endif()
target_link_libraries(qwen_asr_static PRIVATE m)
