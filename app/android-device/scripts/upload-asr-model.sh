#!/bin/bash
# Upload Qwen3-ASR-0.6B pre-quantized model files to R2 public bucket.
# Usage: bash scripts/upload-asr-model.sh [model_dir]
#
# Requires: wrangler CLI authenticated with Cloudflare account.
# Default model_dir is the current directory.

set -euo pipefail

BUCKET="ai-chat-public"
PREFIX="qwen3-asr-0.6b"
MODEL_DIR="${1:-.}"

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

echo "Uploading Qwen3-ASR model to R2 bucket: ${BUCKET}/${PREFIX}/"

upload "$MODEL_DIR/vocab.json"    "vocab.json"    "application/json"
upload "$MODEL_DIR/model.qmodel"  "model.qmodel"  "application/octet-stream"

echo "Done! All model files uploaded to ${BUCKET}/${PREFIX}/"
