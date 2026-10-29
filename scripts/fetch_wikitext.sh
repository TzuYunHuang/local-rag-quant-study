#!/bin/bash
# Download WikiText-2 (raw) used for the perplexity runs. Run from the repo root.
set -e
mkdir -p ppl
if [ ! -f ppl/wikitext-2-raw/wiki.test.raw ]; then
  curl -L -o ppl/wikitext-2-raw-v1.zip \
    https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
  unzip -o ppl/wikitext-2-raw-v1.zip -d ppl
fi
ls -l ppl/wikitext-2-raw/wiki.test.raw
