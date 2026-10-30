"""Unblind and produce the score report.
Input: review/scores_blind.txt (blind scores), review/key_blind_labels.json (label -> version).
Output: eval/scores_claude.jsonl (per-answer scores) and per-version summary tables."""
import json, pathlib, collections, statistics as st
ROOT = pathlib.Path(__file__).resolve().parent.parent
key = json.load(open(ROOT/"review/key_blind_labels.json"))
ev = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT/"eval/eval_40.jsonl", encoding="utf-8")}
lab = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT/"eval/retrieval_labeled.jsonl", encoding="utf-8")}
rows = []
for line in open(ROOT/"review/scores_blind.txt", encoding="utf-8"):
    p = line.rstrip("\n").split(" ", 4)
    if len(p) < 4: continue
    qid, l, c, g = p[0], p[1], int(p[2]), int(p[3]); note = p[4] if len(p) > 4 else ""
    k = f"{qid}:{l}"
    rows.append(dict(qid=qid, label=l, model=key[k], mode="retrieved" if l[0]=="R" else "oracle",
                     type=ev[qid]["type"], correct=c, grounded=g, note=note,
                     ctx_label=lab[qid].get("context_label")))
assert len(rows) == 296 and len({(r["qid"], r["label"]) for r in rows}) == 296, len(rows)
with open(ROOT/"eval/scores_claude.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
M = ["f16", "q8_0", "q4_k_m", "q2_k"]
def sel(model, mode, pred=lambda r: True): return [r for r in rows if r["model"]==model and r["mode"]==mode and pred(r)]
def line(name, f):
    out = [name]
    for m in M: out.append(f(m))
    print(" | ".join(out))
ans = lambda r: r["type"] != "unanswerable"
print("== Mean correctness (0-2) ==")
print("condition | " + " | ".join(M))
line("retrieved answerable (34)", lambda m: f"{st.mean(r['correct'] for r in sel(m,'retrieved',ans)):.2f}")
line("oracle answerable (34)",   lambda m: f"{st.mean(r['correct'] for r in sel(m,'oracle',ans)):.2f}")
line("unanswerable (6; 2 = correct refusal)", lambda m: f"{st.mean(r['correct'] for r in sel(m,'retrieved',lambda r: not ans(r))):.2f}")
print("\n== Score distribution 0/1/2 (count) ==")
for mode in ("retrieved","oracle"):
    line(mode+" answerable", lambda m: "/".join(str(sum(1 for r in sel(m,mode,ans) if r['correct']==s)) for s in (0,1,2)))
line("unanswerable", lambda m: "/".join(str(sum(1 for r in sel(m,'retrieved',lambda r: not ans(r)) if r['correct']==s)) for s in (0,1,2)))
print("\n== Grounded share (grounded=1) ==")
line("retrieved all (40)", lambda m: f"{sum(r['grounded'] for r in sel(m,'retrieved'))}/40")
line("oracle(34)", lambda m: f"{sum(r['grounded'] for r in sel(m,'oracle'))}/34")
print("\n== retrieved answerable questions, by whether the top-4 chunks contain the answer (mean correctness) ==")
for L in ("yes","partial","no"):
    line(f"ctx={L}", lambda m: (lambda xs: f"{st.mean(r['correct'] for r in xs):.2f} (n={len(xs)})")(sel(m,'retrieved',lambda r: ans(r) and r['ctx_label']==L)))
print("\n== By question type (retrieved mean correctness) ==")
for t in ("fact","compare","number"):
    line(t, lambda m: f"{st.mean(r['correct'] for r in sel(m,'retrieved',lambda r: r['type']==t)):.2f}")
