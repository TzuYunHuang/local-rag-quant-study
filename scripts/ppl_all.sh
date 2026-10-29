#!/bin/bash
BIN="${LLAMA_CPP_DIR:-$HOME/ai_learning/llama/llama.cpp}/build/bin/llama-perplexity"
M="${MODEL_DIR:-$HOME/ai_learning/llama/models}"
TXT="${WIKITEXT:-ppl/wikitext-2-raw/wiki.test.raw}"   # run from the repo root
CHUNKS=""   # 空白＝跑完整份；小測試時由外面傳入 --chunks 2
for spec in \
  "f16:qwen2.5-3b-instruct-fp16-00001-of-00002.gguf" \
  "q8_0:qwen2.5-3b-instruct-q8_0.gguf" \
  "q4_k_m:qwen2.5-3b-instruct-q4_k_m.gguf" \
  "q2_k:qwen2.5-3b-instruct-q2_k.gguf"; do
  tag=${spec%%:*}; file=${spec#*:}
  echo "== $tag =="
  "$BIN" -m "$M/$file" -f "$TXT" -c 512 -ngl 99 $1 $2 > ppl/$tag.log 2>&1
  tail -n 3 ppl/$tag.log
  sleep 60
done
echo "完成"
