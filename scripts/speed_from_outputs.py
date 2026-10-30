import json, statistics, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
print(f"{'model':8}{'mode':11}{'n':>4}{'prompt tok':>14}{'TTFT ms':>14}{'prefill/s':>11}{'decode/s':>11}")
for t in ["f16", "q8_0", "q4_k_m", "q2_k"]:
    for m in ("retrieved", "oracle"):
        p = ROOT / "outputs" / f"{t}__{m}.jsonl"
        if not p.exists(): continue
        rs = [json.loads(l) for l in open(p, encoding="utf-8")]
        pre = sum(r["prompt_tokens"] for r in rs) / sum(r["prompt_ms"] for r in rs) * 1000
        dec = sum(r["gen_tokens"] for r in rs) / sum(r["predicted_ms"] for r in rs) * 1000
        print(f"{t:8}{m:11}{len(rs):>4}{statistics.median(r['prompt_tokens'] for r in rs):>14.0f}"
              f"{statistics.median(r['prompt_ms'] for r in rs):>14.0f}{pre:>11.0f}{dec:>11.1f}")
