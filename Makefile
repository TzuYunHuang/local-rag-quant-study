# Reproduction entry points. Run from the repo root.
# Paths can be overridden:  make bench LLAMA_CPP_DIR=/path/to/llama.cpp MODEL_DIR=/path/to/models
LLAMA_CPP_DIR ?= $(HOME)/ai_learning/llama/llama.cpp
MODEL_DIR     ?= $(HOME)/ai_learning/llama/models
export LLAMA_CPP_DIR MODEL_DIR
PY  ?= python
TAG ?= q4_k_m

.PHONY: help chunks index serve eval bench bench-7b wikitext ppl report

help:
	@echo "make chunks            re-chunk corpus/ -> chunks.jsonl"
	@echo "make index             embed chunks, build index.faiss"
	@echo "make serve TAG=q4_k_m  start llama-server (leave running in its own terminal)"
	@echo "make eval TAG=q4_k_m   run 40 retrieved + 34 oracle questions (server must be running)"
	@echo "make bench             llama-bench for the four 3B versions"
	@echo "make bench-7b          llama-bench for 7B Q4_K_M"
	@echo "make wikitext          download WikiText-2"
	@echo "make ppl               perplexity for the four 3B versions"
	@echo "make report            summary table and plots from the committed scores"

chunks:
	$(PY) scripts/chunk.py

index:
	$(PY) scripts/build_index.py

serve:
	bash scripts/serve.sh $(TAG)

eval:
	$(PY) scripts/run_eval.py $(TAG)

bench:
	bash scripts/bench_all.sh

bench-7b:
	bash scripts/bench_7b.sh

wikitext:
	bash scripts/fetch_wikitext.sh

ppl: wikitext
	bash scripts/ppl_all.sh

report:
	$(PY) scripts/score_report.py
	$(PY) scripts/score_report_7b.py
	$(PY) scripts/summarize.py
	$(PY) scripts/plot_knee.py
	$(PY) scripts/plot_7b.py
