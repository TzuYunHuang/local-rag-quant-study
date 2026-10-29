import json, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
mode, qids = sys.argv[1], sys.argv[2:]
TAGS = ["f16", "q8_0", "q4_k_m", "q2_k"]
data = {}
for t in TAGS:
    p = ROOT / "outputs" / f"{t}__{mode}.jsonl"
    if p.exists():
        data[t] = {r["qid"]: r for r in map(json.loads, open(p, encoding="utf-8"))}
qs = {q["qid"]: q for q in map(json.loads, open(ROOT / "eval/eval_40.jsonl", encoding="utf-8"))}
for qid in qids:
    q = qs[qid]
    print("=" * 80)
    print(f"{qid} [{q['type']}] {q['question']}")
    print(f"GOLD: {q['gold_answer']}")
    for t in TAGS:
        r = data.get(t, {}).get(qid)
        if r:
            print(f"--- {t}{' (REFUSED)' if r['refused'] else ''} [{r['gen_tokens']} tok]")
            print(r["answer"][:500])
