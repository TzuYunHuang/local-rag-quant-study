#!/bin/bash
# Start llama-server with the fixed settings used in the experiment.
# Usage: bash scripts/serve.sh <tag>   (tag: f16 | q8_0 | q4_k_m | q2_k | 7b_q4_k_m)
set -e
: "${LLAMA_CPP_DIR:?Set LLAMA_CPP_DIR to your llama.cpp checkout}"
: "${MODEL_DIR:?Set MODEL_DIR to the folder with the GGUF files}"
BIN="$LLAMA_CPP_DIR/build/bin/llama-server"
M="$MODEL_DIR"
case "$1" in
  f16)       file=qwen2.5-3b-instruct-fp16-00001-of-00002.gguf ;;
  q8_0)      file=qwen2.5-3b-instruct-q8_0.gguf ;;
  q4_k_m)    file=qwen2.5-3b-instruct-q4_k_m.gguf ;;
  q2_k)      file=qwen2.5-3b-instruct-q2_k.gguf ;;
  7b_q4_k_m) file=qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf ;;
  *) echo "usage: bash scripts/serve.sh <f16|q8_0|q4_k_m|q2_k|7b_q4_k_m>"; exit 1 ;;
esac
exec "$BIN" -m "$M/$file" -c 4096 -ngl 99 --port 8080
