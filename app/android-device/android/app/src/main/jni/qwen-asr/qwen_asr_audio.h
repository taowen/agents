/*
 * qwen_asr_audio.h - WAV loading and mel spectrogram computation
 */

#ifndef QWEN_ASR_AUDIO_H
#define QWEN_ASR_AUDIO_H

#include <stddef.h>
#include <stdint.h>
#include "qwen_asr.h"

/* Load a WAV file, returns mono float32 samples in [-1,1] at 16kHz.
 * Handles: 16-bit PCM, mono or stereo (mixed to mono).
 * Resamples to 16kHz if needed.
 * Returns NULL on error. Caller must free returned buffer. */
float *qwen_load_wav(const char *path, int *out_n_samples);

/* Parse a WAV file from a memory buffer. Caller must free returned buffer. */
float *qwen_parse_wav_buffer(const uint8_t *data, size_t size, int *out_n_samples);

/* Read audio from stdin (auto-detect WAV or raw s16le 16kHz mono).
 * Returns NULL on error. Caller must free returned buffer. */
float *qwen_read_pcm_stdin(int *out_n_samples);

/* Compute log-mel spectrogram from audio samples.
 * Uses dynamic maximum for clamping (unlike Voxtral's fixed 1.5).
 * samples: mono float32 at 16kHz
 * n_samples: number of samples
 * out_frames: set to number of mel frames produced
 * preset_global_max: if non-NULL and *preset_global_max > -1e20f, use as global_max
 *   (skip max search, still clamp). Otherwise compute normally and write back if non-NULL.
 * Returns: [128, n_frames] mel spectrogram (caller must free)
 * Note: Returns in [mel_bins, frames] layout for Conv2D compatibility. */
float *qwen_mel_spectrogram(const float *samples, int n_samples, int *out_frames,
                            float *preset_global_max);

/* Start a reader thread that incrementally fills a live audio buffer from stdin.
 * Detects WAV vs raw s16le. For WAV, requires 16kHz sample rate.
 * Returns NULL on error. Caller must call qwen_live_audio_free() when done. */
qwen_live_audio_t *qwen_live_audio_start_stdin(void);

/* Create a live audio context without starting a reader thread.
 * Audio data is pushed externally via qwen_live_audio_push() or _push_s16().
 * Caller must call qwen_live_audio_free() when done. */
qwen_live_audio_t *qwen_live_audio_create(void);

/* Push float32 samples (mono, 16kHz) into the live audio buffer. Thread-safe. */
void qwen_live_audio_push(qwen_live_audio_t *la, const float *samples, int n_samples);

/* Push int16 samples (mono, 16kHz) into the live audio buffer.
 * Converts to float32 internally. Thread-safe. */
void qwen_live_audio_push_s16(qwen_live_audio_t *la, const int16_t *samples, int n_samples);

/* Signal end-of-stream. The inference loop will finish processing remaining data. */
void qwen_live_audio_signal_eof(qwen_live_audio_t *la);

/* Reset the live audio buffer for reuse (clears samples, offset, and eof flag). */
void qwen_live_audio_reset(qwen_live_audio_t *la);

/* Join reader thread and free all resources. */
void qwen_live_audio_free(qwen_live_audio_t *la);

#endif /* QWEN_ASR_AUDIO_H */
