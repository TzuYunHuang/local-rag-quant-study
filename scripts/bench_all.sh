#!/bin/bash
BIN=~/ai_learning/llama/llama.cpp/build/bin/llama-bench
M=~/ai_learning/llama/models
mkdir -p bench
for spec in \
  "f16:qwen2.5-3b-instruct-fp16-00001-of-00002.gguf" \
  "q8_0:qwen2.5-3b-instruct-q8_0.gguf" \
  "q4_k_m:qwen2.5-3b-instruct-q4_k_m.gguf" \
  "q2_k:qwen2.5-3b-instruct-q2_k.gguf"; do
  tag=${spec%%:*}; file=${spec#*:}
  echo "== $tag =="
  /usr/bin/time -l "$BIN" -m "$M/$file" -ngl 99 -p 512 -n 128 -r 3 -o md > bench/$tag.md 2> bench/$tag.time
  sleep 60
done
echo "完成"
