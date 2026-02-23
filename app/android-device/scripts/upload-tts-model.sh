#!/bin/bash
# Upload Qwen3-TTS model files to R2 public bucket.
# Usage: bash scripts/upload-tts-model.sh
#
# Requires: wrangler CLI authenticated with Cloudflare account.
# Model files are read from the local Qwen3-TTS-C model directory.

set -euo pipefail

BUCKET="ai-chat-public"
PREFIX="qwen3-tts-0.6b"
MODEL_DIR="/home/taowen/Qwen3-TTS-C/Qwen3-TTS-12Hz-0.6B-CustomVoice"

upload() {
    local src="$1"
    local dest="$2"
    local content_type="${3:-application/octet-stream}"
    echo "Uploading $dest ..."
    npx wrangler r2 object put "${BUCKET}/${PREFIX}/${dest}" \
        --file="$src" \
        --content-type="$content_type" \
        --remote
}

echo "Uploading Qwen3-TTS model to R2 bucket: ${BUCKET}/${PREFIX}/"

upload "$MODEL_DIR/config.json"                        "config.json"                        "application/json"
upload "$MODEL_DIR/vocab.json"                         "vocab.json"                         "application/json"
upload "$MODEL_DIR/merges.txt"                         "merges.txt"                         "text/plain"
upload "$MODEL_DIR/model.safetensors"                  "model.safetensors"                  "application/octet-stream"
upload "$MODEL_DIR/speech_tokenizer/config.json"       "speech_tokenizer/config.json"       "application/json"
upload "$MODEL_DIR/speech_tokenizer/model.safetensors" "speech_tokenizer/model.safetensors" "application/octet-stream"

echo "Done! All model files uploaded to ${BUCKET}/${PREFIX}/"
