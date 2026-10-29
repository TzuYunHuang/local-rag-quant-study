#!/bin/bash
# D11：7B Q4_K_M 的受控速度與記憶體量測，方法與 bench_all.sh 完全相同
BIN=~/ai_learning/llama/llama.cpp/build/bin/llama-bench
M=~/ai_learning/llama/models
mkdir -p bench
tag=7b_q4_k_m; file=qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
echo "== $tag =="
/usr/bin/time -l "$BIN" -m "$M/$file" -ngl 99 -p 512 -n 128 -r 3 -o md > bench/$tag.md 2> bench/$tag.time
cat bench/$tag.md
grep "maximum resident set size" bench/$tag.time
echo "完成"
