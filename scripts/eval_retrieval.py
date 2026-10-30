import json, pathlib, statistics, collections
from retrieve import retrieve   # same folder as scripts/

ROOT = pathlib.Path(__file__).resolve().parent.parent
qs = [json.loads(l) for l in open(ROOT / "eval/eval_40.jsonl", encoding="utf-8")]
KS = (1, 4, 8)

results = []
for q in qs:
    hits = retrieve(q["question"], k=8)
    ids = [h["id"] for h in hits]
    gold = q["gold_chunks"]
    ranks = [ids.index(g) + 1 if g in ids else None for g in gold]
    results.append({
        "qid": q["qid"], "type": q["type"], "gold": gold, "ranks": ranks,
        "top_ids": ids[:4], "top1_score": hits[0]["score"],
        "any": {k: any(r and r <= k for r in ranks) for k in KS},
        "all": {k: bool(gold) and all(r and r <= k for r in ranks) for k in KS},
    })

with open(ROOT / "eval/retrieval_results.jsonl", "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

ans = [r for r in results if r["type"] != "unanswerable"]
pct = lambda xs: f"{100*sum(xs)/len(xs):.0f}%"
print(f"=== Answerable questions ({len(ans)}) ===")
for k in KS:
    print(f"Top-{k}: any hit {pct([r['any'][k] for r in ans])}   all hit {pct([r['all'][k] for r in ans])}")
print("\n=== By question type (Top-4) ===")
for t in ("fact", "compare", "number"):
    g = [r for r in ans if r["type"] == t]
    print(f"{t:8} n={len(g):2}  any {pct([r['any'][4] for r in g])}  all {pct([r['all'][4] for r in g])}")

print("\n=== Questions not fully hit in the top 4 ===")
for r in ans:
    if not r["any"][4]:
        print(f"{r['qid']} gold={r['gold']} ranks={r['ranks']} top4={r['top_ids']}")
    elif not r["all"][4]:
        print(f"{r['qid']} (partial hit only) gold={r['gold']} ranks={r['ranks']}")

s_ans = [r["top1_score"] for r in ans]
s_un = [r["top1_score"] for r in results if r["type"] == "unanswerable"]
print("\n=== Top-1 similarity ===")
print(f"answerable    median {statistics.median(s_ans):.3f}  min {min(s_ans):.3f}  max {max(s_ans):.3f}")
print(f"unanswerable  median {statistics.median(s_un):.3f}  min {min(s_un):.3f}  max {max(s_un):.3f}")
for r in results:
    if r["type"] == "unanswerable":
        print(f"  {r['qid']}  {r['top1_score']:.3f}  {r['top_ids'][0]}")
