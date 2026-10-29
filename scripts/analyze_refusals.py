import json, re, pathlib, collections
ROOT = pathlib.Path(__file__).resolve().parent.parent
REFUSAL = "I cannot find this in the provided documents."
BROAD = re.compile(r"(?:cannot|can't|could not|couldn't|unable to|not able to)\s+find"
                   r"|(?:not|n't)\s+(?:stated|mentioned|specified|covered|available)\b", re.I)
TAGS = ["f16", "q8_0", "q4_k_m", "q2_k"]
label = {r["qid"]: r["context_label"]
         for r in map(json.loads, open(ROOT / "eval/retrieval_labeled.jsonl", encoding="utf-8"))}

def style(ans):
    a = ans.strip()
    if a == REFUSAL: return "exact"
    if REFUSAL.lower() in a.lower(): return "verbatim+extra"
    if BROAD.search(a): return "paraphrase?"
    return "answered"

rows = {}
for t in TAGS:
    for m in ("retrieved", "oracle"):
        p = ROOT / "outputs" / f"{t}__{m}.jsonl"
        if p.exists():
            rows[(t, m)] = [json.loads(l) for l in open(p, encoding="utf-8")]

print("== 拒答方式（exact 整句／+extra 含整句另加文字／改寫? 疑似改寫／作答）==")
print(f"{'版本':8}{'條件':11}{'題組':7}{'exact':>7}{'+extra':>8}{'改寫?':>7}{'作答':>6}")
for (t, m), rs in rows.items():
    for grp, pick in (("有答案", lambda r: r["type"] != "unanswerable"),
                      ("無答案", lambda r: r["type"] == "unanswerable")):
        sub = [r for r in rs if pick(r)]
        if not sub: continue
        c = collections.Counter(style(r["answer"]) for r in sub)
        print(f"{t:8}{m:11}{grp:7}{c['exact']:>7}{c['verbatim+extra']:>8}{c['paraphrase?']:>7}{c['answered']:>6}")

print("\n== retrieved 條件、有答案題，依前 4 塊標記拆開（拒答含改寫）==")
for t in TAGS:
    rs = rows.get((t, "retrieved"))
    if not rs: continue
    for lab in ("yes", "partial", "no"):
        sub = [r for r in rs if r["type"] != "unanswerable" and label[r["qid"]] == lab]
        print(f"{t:8}{lab:9}題數 {len(sub):2}  拒答 {sum(style(r['answer']) != 'answered' for r in sub)}")

print("\n== 需人工確認：含整句另加文字，或疑似改寫 ==")
for (t, m), rs in rows.items():
    for r in rs:
        s = style(r["answer"])
        if s in ("verbatim+extra", "paraphrase?"):
            print(f"[{t}/{m}] {r['qid']} {s}: {r['answer'][:170]!r}")
