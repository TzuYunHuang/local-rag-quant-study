"""Stratified sample of 30 answers by Claude's score (10 each of 0, 1, 2) for the user to review; version names and Claude's scores are hidden."""
import json, random, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
rd = lambda p: [json.loads(l) for l in open(ROOT/p, encoding="utf-8") if l.strip()]
rows = rd("eval/scores_claude.jsonl")
ev = {e["qid"]: e for e in rd("eval/eval_40.jsonl")}
out = {(m, mode): {r["qid"]: r for r in rd(f"outputs/{m}__{mode}.jsonl")} for m in ("f16","q8_0","q4_k_m","q2_k") for mode in ("retrieved","oracle")}
rng = random.Random(7)
pick = []
for s in (0, 1, 2):
    pool = [r for r in rows if r["correct"] == s and (s != 2 or r["type"] != "unanswerable")]
    rng.shuffle(pool)
    seen, chosen = set(), []
    for r in pool:                                   # avoid repeating a question where possible, to spread out the questions
        if r["qid"] in seen: continue
        seen.add(r["qid"]); chosen.append(r)
        if len(chosen) == 10: break
    pick += chosen
rng.shuffle(pick)
md = ["# D9 review: score each answer 0 / 1 / 2", "",
      "- **2**: all key facts match the gold answer, no wrong content",
      "- **1**: partly correct (missing a key point, or a minor error)",
      "- **0**: wrong, off-topic, truncated, or refused although the question is answerable",
      "- Unanswerable (gold answer is Not stated): correct refusal = 2; refusal with an invented abbreviation expansion = 1; answering = 0", "",
      "Reply format example: `1=2, 2=0, 3=1, ...` (30 items)", ""]
key = {}
for i, r in enumerate(pick, 1):
    e = ev[r["qid"]]; o = out[(r["model"], r["mode"])][r["qid"]]
    flag = "(this answer was cut off at 256 tokens)" if o["finish"] != "stop" else ""
    md += [f"## {i}. [{e['type']}] {e['question']}", f"**Gold answer**: {e['gold_answer']}", "", f"**Model answer**{flag}:", "",
           *["> " + ln for ln in o["answer"].splitlines()], ""]
    key[i] = dict(qid=r["qid"], label=r["label"], model=r["model"], mode=r["mode"], claude=r["correct"])
(ROOT/"review/for_user_30.md").write_text("\n".join(md), encoding="utf-8")
json.dump(key, open(ROOT/"review/for_user_30_key.json", "w"), indent=1)
print("ok", len(pick), {s: sum(1 for k in key.values() if k["claude"]==s) for s in (0,1,2)})
