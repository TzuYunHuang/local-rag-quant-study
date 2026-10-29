"""把每題的證據片段對應到 chunks.jsonl 裡的 chunk，寫出 gold_chunks，並報告對不上的題目。
用法：python scripts/check_evalset.py eval/eval_d4.jsonl"""
import json, re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
norm = lambda s: re.sub(r"\s+", " ", s)
rows = [json.loads(l) for l in open(ROOT / "chunks.jsonl", encoding="utf-8")]
texts = [(r["id"], norm(r["text"])) for r in rows]

path = pathlib.Path(sys.argv[1])
qs = [json.loads(l) for l in open(path, encoding="utf-8")]
problems = 0
for q in qs:
    gold, per_ev = [], {}
    for ev in q["evidence"]:
        hits = [cid for cid, t in texts if norm(ev) in t]
        per_ev[ev] = hits
        gold += [h for h in hits if h not in gold]
    q["gold_chunks"] = gold
    flag = ""
    if any(not h for h in per_ev.values()):
        flag = "  <-- 有證據片段找不到"; problems += 1
    elif any(len(h) > 3 for h in per_ev.values()):
        flag = "  <-- 片段太常見，請縮小範圍"; problems += 1
    print(f"{q['qid']} [{q['type']:7}] {len(gold)} chunk(s): {gold}{flag}")

out = path.with_name(path.stem + "_resolved.jsonl")
with open(out, "w", encoding="utf-8") as f:
    for q in qs:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")
print(f"\n{len(qs)} 題，有問題 {problems} 題；已寫出 {out.name}")
