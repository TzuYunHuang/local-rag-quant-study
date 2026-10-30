"""解盲並產出評分報告。
輸入：review/scores_blind.txt（盲評結果）、review/key_blind_labels.json（標籤→版本）。
輸出：eval/scores_claude.jsonl（逐答案分數）與各版本統計表。"""
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
print("== 平均正確性（0-2） ==")
print("條件 | " + " | ".join(M))
line("retrieved 有答案題(34)", lambda m: f"{st.mean(r['correct'] for r in sel(m,'retrieved',ans)):.2f}")
line("oracle 有答案題(34)",   lambda m: f"{st.mean(r['correct'] for r in sel(m,'oracle',ans)):.2f}")
line("無答案題(6，2=正確拒答)", lambda m: f"{st.mean(r['correct'] for r in sel(m,'retrieved',lambda r: not ans(r))):.2f}")
print("\n== 分數分布 0/1/2（題數） ==")
for mode in ("retrieved","oracle"):
    line(mode+" 有答案題", lambda m: "/".join(str(sum(1 for r in sel(m,mode,ans) if r['correct']==s)) for s in (0,1,2)))
line("無答案題", lambda m: "/".join(str(sum(1 for r in sel(m,'retrieved',lambda r: not ans(r)) if r['correct']==s)) for s in (0,1,2)))
print("\n== 有依據比例（grounded=1） ==")
line("retrieved 全部(40)", lambda m: f"{sum(r['grounded'] for r in sel(m,'retrieved'))}/40")
line("oracle(34)", lambda m: f"{sum(r['grounded'] for r in sel(m,'oracle'))}/34")
print("\n== retrieved 有答案題，依前4塊是否含答案（平均正確性） ==")
for L in ("yes","partial","no"):
    line(f"ctx={L}", lambda m: (lambda xs: f"{st.mean(r['correct'] for r in xs):.2f} (n={len(xs)})")(sel(m,'retrieved',lambda r: ans(r) and r['ctx_label']==L)))
print("\n== 依題型（retrieved 平均正確性） ==")
for t in ("fact","compare","number"):
    line(t, lambda m: f"{st.mean(r['correct'] for r in sel(m,'retrieved',lambda r: r['type']==t)):.2f}")
