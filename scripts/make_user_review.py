"""依 Claude 的分數分層抽樣 30 題給使用者覆核（0、1、2 分各 10 題），隱藏版本名稱與 Claude 的分數。"""
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
    for r in pool:                                   # 同一題盡量不重複，讓題目分散
        if r["qid"] in seen: continue
        seen.add(r["qid"]); chosen.append(r)
        if len(chosen) == 10: break
    pick += chosen
rng.shuffle(pick)
md = ["# D9 覆核：請對每個答案打 0／1／2 分", "",
      "- **2**：與標準答案的關鍵事實都一致，沒有錯誤內容",
      "- **1**：對了一部分（缺關鍵點，或夾帶小錯）",
      "- **0**：錯誤、離題、答案被截斷，或有答案題卻拒答",
      "- 無答案題（標準答案為 Not stated）：正確拒答＝2；拒答但編造縮寫全名＝1；直接作答＝0", "",
      "回覆格式範例：`1=2, 2=0, 3=1, ...`（30 題）", ""]
key = {}
for i, r in enumerate(pick, 1):
    e = ev[r["qid"]]; o = out[(r["model"], r["mode"])][r["qid"]]
    flag = "（此答案被 256 tokens 截斷）" if o["finish"] != "stop" else ""
    md += [f"## {i}. [{e['type']}] {e['question']}", f"**標準答案**：{e['gold_answer']}", "", f"**模型答案**{flag}：", "",
           *["> " + ln for ln in o["answer"].splitlines()], ""]
    key[i] = dict(qid=r["qid"], label=r["label"], model=r["model"], mode=r["mode"], claude=r["correct"])
(ROOT/"review/for_user_30.md").write_text("\n".join(md), encoding="utf-8")
json.dump(key, open(ROOT/"review/for_user_30_key.json", "w"), indent=1)
print("ok", len(pick), {s: sum(1 for k in key.values() if k["claude"]==s) for s in (0,1,2)})
