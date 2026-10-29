import json, random, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
rd = lambda p: [json.loads(l) for l in open(ROOT/p, encoding="utf-8") if l.strip()]
chunks = {c["id"]: c for c in rd("chunks.jsonl")}
ev = {e["qid"]: e for e in rd("eval/eval_40.jsonl")}
ctx = json.load(open(ROOT/"eval/contexts_retrieved.json"))
tags = ["f16", "q8_0", "q4_k_m", "q2_k"]
out = {(t, m): {r["qid"]: r for r in rd(f"outputs/{t}__{m}.jsonl")} for t in tags for m in ("retrieved", "oracle")}
rng = random.Random(20261028)
key = {}
lines_by_q = {}
for qid in sorted(ev):
    e = ev[qid]
    L = [f"##### {qid} [{e['type']}]", f"QUESTION: {e['question']}", f"GOLD: {e['gold_answer']}", ""]
    shown = set()
    L.append("--- RETRIEVED CONTEXT (what the model saw in condition R) ---")
    for i, cid in enumerate(ctx[qid], 1):
        c = chunks[cid]; shown.add(cid)
        L.append(f"[{i}] {cid} | {c['heading']}\n{c['text']}\n")
    for mode, pre in (("retrieved", "R"), ("oracle", "O")):
        if mode == "oracle" and qid in out[("f16", "oracle")]:
            ids = out[("f16", "oracle")][qid]["context_ids"]
            extra = [i for i in ids if i not in shown]
            L.append(f"--- ORACLE CONTEXT (condition O) = {ids}; " + ("already shown above" if not extra else "extra chunks below") + " ---")
            for cid in extra:
                c = chunks[cid]; L.append(f"{cid} | {c['heading']}\n{c['text']}\n")
        avail = [t for t in tags if qid in out[(t, mode)]]
        if not avail: continue
        order = avail[:]; rng.shuffle(order)
        L.append(f"--- ANSWERS, condition {pre} ---")
        for j, t in enumerate(order):
            lab = f"{pre}{'ABCD'[j]}"
            key[f"{qid}:{lab}"] = t
            a = out[(t, mode)][qid]
            fin = "" if a["finish"] == "stop" else f" [TRUNCATED finish={a['finish']}]"
            L.append(f"{lab}{fin}: {a['answer']}")
    lines_by_q[qid] = "\n".join(L)
json.dump(key, open(ROOT/"review/key_DO_NOT_READ.json", "w"), indent=1)
qs = sorted(lines_by_q)
for b in range(0, 40, 5):
    (ROOT/f"review/batch_{b//5+1}.txt").write_text("\n\n".join(lines_by_q[q] for q in qs[b:b+5]), encoding="utf-8")
print("ok", len(key))
