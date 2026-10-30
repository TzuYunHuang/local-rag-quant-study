#!/bin/bash
: "${LLAMA_CPP_DIR:?Set LLAMA_CPP_DIR to your llama.cpp checkout}"
: "${MODEL_DIR:?Set MODEL_DIR to the folder with the GGUF files}"
BIN="$LLAMA_CPP_DIR/build/bin/llama-perplexity"
M="$MODEL_DIR"
TXT="${WIKITEXT:-ppl/wikitext-2-raw/wiki.test.raw}"   # run from the repo root
CHUNKS=""   # empty = run the full set; for a quick test pass --chunks 2 from outside
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
echo "done"
