import json, pathlib, collections
ROOT = pathlib.Path(__file__).resolve().parent.parent
res = [json.loads(l) for l in open(ROOT / "eval/retrieval_results.jsonl", encoding="utf-8")]

MANUAL = {  # 嚴格未全中的 14 題，人工逐題判定（yes／partial／no）
    "Q03": "no", "Q06": "partial", "Q07": "yes", "Q08": "no", "Q09": "yes",
    "Q12": "partial", "Q13": "partial", "Q15": "partial", "Q18": "no", "Q22": "no",
    "Q29": "partial", "Q30": "no", "Q31": "partial", "Q32": "partial",
}
need = {r["qid"] for r in res if r["type"] != "unanswerable" and not r["all"]["4"]}
assert need == set(MANUAL), f"標記與未全中題不一致：{need ^ set(MANUAL)}"

for r in res:
    r["context_label"] = ("n/a" if r["type"] == "unanswerable"
                          else "yes" if r["all"]["4"] else MANUAL[r["qid"]])
with open(ROOT / "eval/retrieval_labeled.jsonl", "w", encoding="utf-8") as f:
    for r in res: f.write(json.dumps(r, ensure_ascii=False) + "\n")

ans = [r for r in res if r["type"] != "unanswerable"]
c = collections.Counter(r["context_label"] for r in ans)
print(f"有答案 {len(ans)} 題：含答案 {c['yes']}（{100*c['yes']/len(ans):.0f}%）、"
      f"部分 {c['partial']}（{100*c['partial']/len(ans):.0f}%）、沒有 {c['no']}（{100*c['no']/len(ans):.0f}%）")
for t in ("fact", "compare", "number"):
    cc = collections.Counter(r["context_label"] for r in ans if r["type"] == t)
    print(f"  {t:8} yes {cc['yes']}  partial {cc['partial']}  no {cc['no']}")
