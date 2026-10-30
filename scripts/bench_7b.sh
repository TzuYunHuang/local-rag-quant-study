#!/bin/bash
# D11: controlled speed and memory measurement for 7B Q4_K_M, same method as bench_all.sh
: "${LLAMA_CPP_DIR:?Set LLAMA_CPP_DIR to your llama.cpp checkout}"
: "${MODEL_DIR:?Set MODEL_DIR to the folder with the GGUF files}"
BIN="$LLAMA_CPP_DIR/build/bin/llama-bench"
M="$MODEL_DIR"
mkdir -p bench
tag=7b_q4_k_m; file=qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
echo "== $tag =="
/usr/bin/time -l "$BIN" -m "$M/$file" -ngl 99 -p 512 -n 128 -r 3 -o md > bench/$tag.md 2> bench/$tag.time
cat bench/$tag.md
grep "maximum resident set size" bench/$tag.time
echo "done"
