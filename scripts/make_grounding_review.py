"""挑 10 個答案請使用者獨立判斷「有沒有依據」（答案的事實是否都能在當題上下文找到）。
5 個 Claude 判無依據、5 個判有依據（含邊界案例），隱藏版本名稱與 Claude 的判斷。"""
import json, random, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
rd = lambda p: [json.loads(l) for l in open(ROOT/p, encoding="utf-8") if l.strip()]
chunks = {c["id"]: c for c in rd("chunks.jsonl")}
ev = {e["qid"]: e for e in rd("eval/eval_40.jsonl")}
ctx = json.load(open(ROOT/"eval/contexts_retrieved.json"))
sc = {(r["qid"], r["label"]): r for r in rd("eval/scores_claude.jsonl")}
SEL = [("Q05","OC"),("Q21","OB"),("Q12","OB"),("Q10","OC"),("Q05","RA"),   # Claude 判 0
       ("Q07","OB"),("Q01","OB"),("Q31","OB"),("Q03","RB"),("Q33","OB")]   # Claude 判 1
rng = random.Random(11); rng.shuffle(SEL)
out, key = [], {}
for i, (q, l) in enumerate(SEL, 1):
    r = sc[(q, l)]
    a = [x for x in rd(f"outputs/{r['model']}__{r['mode']}.jsonl") if x["qid"] == q][0]
    cids = a["context_ids"]
    out.append(dict(n=i, qid=q, label=l, type=ev[q]["type"], question=ev[q]["question"], gold=ev[q]["gold_answer"],
                    answer=a["answer"], mode=r["mode"], context=[(c, chunks[c]["heading"], chunks[c]["text"]) for c in cids]))
    key[i] = dict(qid=q, label=l, model=r["model"], mode=r["mode"], claude_grounded=r["grounded"])
json.dump(key, open(ROOT/"review/grounding10_key.json", "w"), indent=1)
json.dump(out, open(ROOT/"review/grounding10_dump.json", "w"), ensure_ascii=False, indent=1)
for o in out:
    print(f"### {o['n']} [{o['type']}] {o['mode']} {o['qid']}\nQ: {o['question']}\nGOLD: {o['gold']}\nANS: {o['answer']}\nCTX_IDS: {[c[0] for c in o['context']]}  chars={sum(len(c[2]) for c in o['context'])}\n")
