import json, sys, argparse, pathlib, statistics
from rag import build_messages, generate, REFUSAL, _chunks
from retrieve import retrieve

ROOT = pathlib.Path(__file__).resolve().parent.parent
TAGS = {"f16": "fp16", "q8_0": "q8_0", "q4_k_m": "q4_k_m", "q2_k": "q2_k",
        "7b_q4_k_m": "7b-instruct-q4_k_m"}  # tag -> keyword in the model file name (the 7B tag uses a keyword containing 7b to avoid confusion with 3B)

ap = argparse.ArgumentParser()
ap.add_argument("tag", choices=TAGS)
ap.add_argument("--modes", default="retrieved,oracle")
ap.add_argument("--limit", type=int, default=0, help="only run the first N questions (for testing)")
args = ap.parse_args()

all_qs = [json.loads(l) for l in open(ROOT / "eval/eval_40.jsonl", encoding="utf-8")]
qs = all_qs[:args.limit] if args.limit else all_qs

# 1. Frozen retrieval results: created on the first run, shared by all versions afterwards
ctx_path = ROOT / "eval/contexts_retrieved.json"
if ctx_path.exists():
    frozen = json.load(open(ctx_path, encoding="utf-8"))
else:
    frozen = {q["qid"]: [h["id"] for h in retrieve(q["question"], k=4)] for q in all_qs}
    json.dump(frozen, open(ctx_path, "w", encoding="utf-8"), indent=1)
    print("Created frozen retrieval results:", ctx_path.name)

out_dir = ROOT / "outputs"
out_dir.mkdir(exist_ok=True)

for mode in args.modes.split(","):
    path = out_dir / f"{args.tag}__{mode}.jsonl"
    done = {json.loads(l)["qid"] for l in open(path, encoding="utf-8")} if path.exists() else set()
    todo = [q for q in qs if q["qid"] not in done and (mode == "retrieved" or q["gold_chunks"])]
    print(f"[{args.tag}/{mode}] done {len(done)}, to run {len(todo)}")
    with open(path, "a", encoding="utf-8") as f:
        for i, q in enumerate(todo, 1):
            ids = frozen[q["qid"]] if mode == "retrieved" else q["gold_chunks"]
            r = generate(build_messages(q["question"], [_chunks[c] for c in ids]))
            if TAGS[args.tag] not in r["model"].lower():
                sys.exit(f"The server loaded {r['model']}, which does not match the requested {args.tag}; stopping.")
            rec = {"qid": q["qid"], "type": q["type"], "mode": mode, "model_tag": args.tag,
                   "context_ids": ids, "answer": r["answer"], "finish": r["finish"],
                   "refused": REFUSAL.lower() in r["answer"].lower(),
                   "prompt_tokens": r["usage"]["prompt_tokens"], "gen_tokens": r["usage"]["completion_tokens"],
                   "prompt_ms": r["timings"]["prompt_ms"], "predicted_ms": r["timings"]["predicted_ms"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(f"  {i}/{len(todo)} {q['qid']} {'REFUSED' if rec['refused'] else '':8} {rec['gen_tokens']} tok", flush=True)
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    ans = [r for r in rows if r["type"] != "unanswerable"]
    un = [r for r in rows if r["type"] == "unanswerable"]
    print(f"== {args.tag}/{mode} summary ==  answerable refused {sum(r['refused'] for r in ans)}/{len(ans)}  "
          f"unanswerable refused {sum(r['refused'] for r in un)}/{len(un)}  "
          f"truncated {sum(r['finish'] == 'length' for r in rows)}  "
          f"mean generated {statistics.mean(r['gen_tokens'] for r in rows):.0f} tokens")
