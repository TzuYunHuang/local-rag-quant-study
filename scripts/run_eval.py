import json, sys, argparse, pathlib, statistics
from rag import build_messages, generate, REFUSAL, _chunks
from retrieve import retrieve

ROOT = pathlib.Path(__file__).resolve().parent.parent
TAGS = {"f16": "fp16", "q8_0": "q8_0", "q4_k_m": "q4_k_m", "q2_k": "q2_k",
        "7b_q4_k_m": "7b-instruct-q4_k_m"}  # 版本 -> 模型檔名關鍵字（7b 組用含 7b 的關鍵字，避免和 3B 同名混淆）

ap = argparse.ArgumentParser()
ap.add_argument("tag", choices=TAGS)
ap.add_argument("--modes", default="retrieved,oracle")
ap.add_argument("--limit", type=int, default=0, help="只跑前 N 題（測試用）")
args = ap.parse_args()

all_qs = [json.loads(l) for l in open(ROOT / "eval/eval_40.jsonl", encoding="utf-8")]
qs = all_qs[:args.limit] if args.limit else all_qs

# 1. 固定檢索結果：第一次建立，之後所有版本共用
ctx_path = ROOT / "eval/contexts_retrieved.json"
if ctx_path.exists():
    frozen = json.load(open(ctx_path, encoding="utf-8"))
else:
    frozen = {q["qid"]: [h["id"] for h in retrieve(q["question"], k=4)] for q in all_qs}
    json.dump(frozen, open(ctx_path, "w", encoding="utf-8"), indent=1)
    print("已建立固定檢索結果：", ctx_path.name)

out_dir = ROOT / "outputs"
out_dir.mkdir(exist_ok=True)

for mode in args.modes.split(","):
    path = out_dir / f"{args.tag}__{mode}.jsonl"
    done = {json.loads(l)["qid"] for l in open(path, encoding="utf-8")} if path.exists() else set()
    todo = [q for q in qs if q["qid"] not in done and (mode == "retrieved" or q["gold_chunks"])]
    print(f"[{args.tag}/{mode}] 已完成 {len(done)} 題，待跑 {len(todo)} 題")
    with open(path, "a", encoding="utf-8") as f:
        for i, q in enumerate(todo, 1):
            ids = frozen[q["qid"]] if mode == "retrieved" else q["gold_chunks"]
            r = generate(build_messages(q["question"], [_chunks[c] for c in ids]))
            if TAGS[args.tag] not in r["model"].lower():
                sys.exit(f"伺服器載入的是 {r['model']}，與指定的 {args.tag} 不符，已停止。")
            rec = {"qid": q["qid"], "type": q["type"], "mode": mode, "model_tag": args.tag,
                   "context_ids": ids, "answer": r["answer"], "finish": r["finish"],
                   "refused": REFUSAL.lower() in r["answer"].lower(),
                   "prompt_tokens": r["usage"]["prompt_tokens"], "gen_tokens": r["usage"]["completion_tokens"],
                   "prompt_ms": r["timings"]["prompt_ms"], "predicted_ms": r["timings"]["predicted_ms"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(f"  {i}/{len(todo)} {q['qid']} {'REFUSED' if rec['refused'] else '':8} {rec['gen_tokens']} tok", flush=True)
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    ans = [r for r in rows if r["type"] != "unanswerable"]
    un = [r for r in rows if r["type"] == "unanswerable"]
    print(f"== {args.tag}/{mode} 摘要 ==  有答案題拒答 {sum(r['refused'] for r in ans)}/{len(ans)}  "
          f"無答案題拒答 {sum(r['refused'] for r in un)}/{len(un)}  "
          f"被截斷 {sum(r['finish'] == 'length' for r in rows)}  "
          f"平均生成 {statistics.mean(r['gen_tokens'] for r in rows):.0f} tokens")
