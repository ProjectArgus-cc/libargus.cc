#!/bin/bash
set -e

# Create target directories
mkdir -p models

echo "[Download] Fetching Whisper Base ASR model..."
curl -L -C - -o models/whisper-base.gguf "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"

echo "[Download] Fetching WavTokenizer TTS Vocoder model..."
curl -L -C - -o models/wavtokenizer-large-Q4_K_M.gguf "https://huggingface.co/koboldcpp/tts/resolve/main/WavTokenizer-Large-75-Q4_0.gguf"

echo "[Download] Fetching OuteTTS 500M TTS Base model..."
curl -L -C - -o models/OuteTTS-0.2-500M-Q4_K_M.gguf "https://huggingface.co/OuteAI/OuteTTS-0.2-500M-GGUF/resolve/main/OuteTTS-0.2-500M-Q4_K_M.gguf"

echo "[Download] Fetching Qwen2-VL 2B VLM Base model..."
curl -L -C - -o models/qwen2-vl-2b-it-Q4_K_M.gguf "https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf"

echo "[Download] Fetching Qwen2-VL 2B VLM Projector..."
curl -L -C - -o models/qwen2-vl-2b-it-mmproj.gguf "https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-2B-Instruct-f16.gguf"

echo "[Download] All models downloaded successfully!"
