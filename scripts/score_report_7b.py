"""Unblind the 7B-vs-Q8_0 blind scoring and print the comparison.
Input: review/scores7b_blind.txt, review/key7b_blind_labels.json
Output: eval/scores_7b.jsonl + summary tables."""
import json, pathlib, statistics as st
ROOT = pathlib.Path(__file__).resolve().parent.parent
key = json.load(open(ROOT/"review/key7b_blind_labels.json"))
ev = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT/"eval/eval_40.jsonl", encoding="utf-8")}
rows = []
for line in open(ROOT/"review/scores7b_blind.txt", encoding="utf-8"):
    p = line.rstrip("\n").split(" ", 4)
    if len(p) < 4: continue
    qid, l, c, g = p[0], p[1], int(p[2]), int(p[3])
    rows.append(dict(qid=qid, label=l, model=key[f"{qid}:{l}"], mode="retrieved" if l[0]=="R" else "oracle",
                     type=ev[qid]["type"], correct=c, grounded=g, note=p[4] if len(p)>4 else ""))
assert len(rows) == 148 and len({(r["qid"], r["label"]) for r in rows}) == 148, len(rows)
with open(ROOT/"eval/scores_7b.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
M = ["7b_q4_k_m", "q8_0"]
sel = lambda m, mode, pred=lambda r: True: [r for r in rows if r["model"]==m and r["mode"]==mode and pred(r)]
ans = lambda r: r["type"] != "unanswerable"
def line(name, f): print(name, "|", " | ".join(f(m) for m in M))
print("model |", " | ".join(M))
line("retrieved answerable mean(34)", lambda m: f"{st.mean(r['correct'] for r in sel(m,'retrieved',ans)):.2f}")
line("oracle answerable mean(34)", lambda m: f"{st.mean(r['correct'] for r in sel(m,'oracle',ans)):.2f}")
line("unanswerable mean(6)", lambda m: f"{st.mean(r['correct'] for r in sel(m,'retrieved',lambda r: not ans(r))):.2f}")
for mode in ("retrieved","oracle"):
    line(f"{mode} dist 0/1/2", lambda m: "/".join(str(sum(1 for r in sel(m,mode,ans) if r['correct']==s)) for s in (0,1,2)))
line("grounded retrieved(40)", lambda m: f"{sum(r['grounded'] for r in sel(m,'retrieved'))}")
line("grounded oracle(34)", lambda m: f"{sum(r['grounded'] for r in sel(m,'oracle'))}")
for t in ("fact","compare","number"):
    line(f"retrieved {t}", lambda m: f"{st.mean(r['correct'] for r in sel(m,'retrieved',lambda r: r['type']==t)):.2f}")
    line(f"oracle {t}", lambda m: f"{st.mean(r['correct'] for r in sel(m,'oracle',lambda r: r['type']==t)):.2f}")
# anchor consistency vs earlier D9 q8_0 scores
old = {}
for l in open(ROOT/"eval/scores_claude.jsonl", encoding="utf-8"):
    d = json.loads(l)
    if d["model"]=="q8_0": old[(d["qid"], d["mode"])] = d
diff = [(r["qid"], r["mode"], old[(r["qid"],r["mode"])]["correct"], r["correct"]) for r in sel("q8_0","retrieved")+sel("q8_0","oracle") if (r["qid"],r["mode"]) in old and old[(r["qid"],r["mode"])]["correct"]!=r["correct"]]
n = len(sel("q8_0","retrieved"))+len(sel("q8_0","oracle"))
print(f"\nQ8_0 anchor: correctness differs from D9 in {len(diff)}/{n}:", diff)
