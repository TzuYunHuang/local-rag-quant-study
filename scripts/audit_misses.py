import json, re, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
rows = {r["id"]: r for r in map(json.loads, open(ROOT / "chunks.jsonl", encoding="utf-8"))}
qs = {q["qid"]: q for q in map(json.loads, open(ROOT / "eval/eval_40.jsonl", encoding="utf-8"))}
res = [json.loads(l) for l in open(ROOT / "eval/retrieval_results.jsonl", encoding="utf-8")]

STOP = set("the a an of to in and or is are for with that this it as by on be can from not which what how does do".split())
words = lambda s: {w for w in re.findall(r"[a-z0-9_.]+", s.lower()) if len(w) > 2 and w not in STOP}

def best_line(text, gold):
    g, best = words(gold), ("", 0.0)
    for line in re.split(r"\n+|(?<=[.!?])\s+", text):
        sc = len(words(line) & g) / max(1, len(g))
        if sc > best[1]: best = (line.strip(), sc)
    return best

for r in res:
    if r["type"] == "unanswerable" or r["all"]["4"]:
        continue
    q = qs[r["qid"]]
    print("=" * 90)
    print(f"{q['qid']} [{q['type']}] {q['question']}")
    print(f"  標準答案：{q['gold_answer'][:170]}")
    for cid in r["gold"]:
        line, sc = best_line(rows[cid]["text"], q["gold_answer"])
        print(f"  GOLD {cid}  ({sc:.2f})  {line[:190]!r}")
    for cid in r["top_ids"]:
        line, sc = best_line(rows[cid]["text"], q["gold_answer"])
        print(f"  top4 {cid}  ({sc:.2f})  {line[:190]!r}")
