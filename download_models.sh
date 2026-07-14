#!/bin/bash
set -e

# Create target directories
mkdir -p models

download_if_missing() {
    local filepath="$1"
    local url="$2"
    if [ -f "$filepath" ]; then
        echo "[Download] File $filepath already exists, skipping."
    else
        echo "[Download] Fetching $filepath from $url..."
        curl -L -o "$filepath" "$url"
    fi
}

download_if_missing "models/whisper-base.gguf" "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
download_if_missing "models/wavtokenizer-large-Q4_K_M.gguf" "https://huggingface.co/koboldcpp/tts/resolve/main/WavTokenizer-Large-75-Q4_0.gguf"
download_if_missing "models/OuteTTS-0.2-500M-Q4_K_M.gguf" "https://huggingface.co/OuteAI/OuteTTS-0.2-500M-GGUF/resolve/main/OuteTTS-0.2-500M-Q4_K_M.gguf"
download_if_missing "models/qwen2-vl-2b-it-Q4_K_M.gguf" "https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf"
download_if_missing "models/qwen2-vl-2b-it-mmproj.gguf" "https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-2B-Instruct-f16.gguf"
download_if_missing "models/jina-embeddings-v3-Q4_K_M.gguf" "https://huggingface.co/second-state/jina-embeddings-v3-GGUF/resolve/main/jina-embeddings-v3-Q4_K_M.gguf"

echo "[Download] All models checked and prepared!"
