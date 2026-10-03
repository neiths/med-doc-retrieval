#!/usr/bin/env bash
# Sync processed Parquet corpus to Hugging Face Storage Bucket
set -e

BUCKET_URI="hf://buckets/nieths/ViBioMIR/corpus"
SOURCE_DIR="data/processed/parquet_corpus"

echo "============================================================"
echo "Hugging Face Storage Bucket Sync"
echo "Source:      ${SOURCE_DIR}"
echo "Destination: ${BUCKET_URI}"
echo "============================================================"

# Ensure hf CLI is in PATH
export PATH="${HOME}/.local/bin:${PATH}"

if ! command -v hf &> /dev/null; then
    echo "Installing 'hf' CLI tool via uv..."
    uv tool install hf
fi

# Check for HF token
if [ -z "${HF_TOKEN}" ] && [ -f ".env" ]; then
    HF_TOKEN=$(grep -E '^HF_TOKEN=' .env | cut -d '=' -f2- | tr -d ' "\r')
    export HF_TOKEN
fi

if [ -z "${HF_TOKEN}" ]; then
    echo "Notice: HF_TOKEN is not set in environment or .env."
    echo "If sync fails with authentication error, run 'hf auth login' or export HF_TOKEN=hf_..."
fi

hf sync "${SOURCE_DIR}" "${BUCKET_URI}" "$@"
echo "Sync process finished successfully!"
