"""Plot 7B Q4_K_M vs 3B Q8_0 vs 3B F16 (speed, memory, answer quality)."""
import pathlib, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parent.parent
import json, re, statistics as st
names = ["3B F16", "3B Q8_0", "7B Q4_K_M"]
tags = ["f16", "q8_0", "7b_q4_k_m"]
cols = ["#2a78d6", "#eb6834", "#1baf7a"]

def bench(tag):  # pp512 / tg128 (tok/s) from llama-bench markdown, RSS (GiB) from /usr/bin/time -l
    t = (ROOT / f"bench/{tag}.md").read_text()
    pp = float(re.search(r"pp512\s*\|\s*([\d.]+)", t).group(1))
    tg = float(re.search(r"tg128\s*\|\s*([\d.]+)", t).group(1))
    rss = int(re.search(r"(\d+)\s+maximum resident set size", (ROOT / f"bench/{tag}.time").read_text()).group(1)) / 2**30
    return pp, tg, rss
pp, tg, rss = (list(v) for v in zip(*[bench(t) for t in tags]))

def load(path): return [json.loads(l) for l in open(ROOT / path, encoding="utf-8") if l.strip()]
# Quality: mean correctness (0-2) over the 34 answerable questions.
# 7B and 3B Q8_0 come from the same blind batch (eval/scores_7b.jsonl); 3B F16 from the earlier round (eval/scores_claude.jsonl).
old, new = load("eval/scores_claude.jsonl"), load("eval/scores_7b.jsonl")
def q(rows, model, mode):
    return st.mean(r["correct"] for r in rows if r["model"] == model and r["mode"] == mode and r["type"] != "unanswerable")
src = [(old, "f16"), (new, "q8_0"), (new, "7b_q4_k_m")]
q_ret = [q(r, m, "retrieved") for r, m in src]; q_ora = [q(r, m, "oracle") for r, m in src]
S = "#fcfcfb"; T = "#0b0b0b"; T2 = "#52514e"
fig, ax = plt.subplots(1, 4, figsize=(14, 4.2), facecolor=S)
panels = [(pp, "Prefill speed, pp512 (tok/s)", "{:.0f}"), (tg, "Generation speed, tg128 (tok/s)", "{:.1f}"),
          (rss, "Resident memory (GiB)", "{:.2f}"), (None, "Answer quality (mean 0-2, n=34)", "{:.2f}")]
for a, (v, title, fmt) in zip(ax, panels):
    a.set_facecolor(S); a.set_title(title, color=T, fontsize=10, loc="left")
    if v:
        b = a.bar(names, v, color=cols, width=0.6)
        for r, y in zip(b, v): a.text(r.get_x() + r.get_width()/2, y, fmt.format(y), ha="center", va="bottom", fontsize=9, color=T2)
    else:
        w = 0.36
        for i, (lab, vals, al) in enumerate([("retrieved", q_ret, 0.55), ("oracle", q_ora, 1.0)]):
            xs = [j + (i - 0.5) * w for j in range(3)]
            b = a.bar(xs, vals, width=w, color=cols, alpha=al, label=lab)
            for x, y in zip(xs, vals): a.text(x, y, fmt.format(y), ha="center", va="bottom", fontsize=8, color=T2)
        a.set_xticks(range(3)); a.set_xticklabels(names); a.set_ylim(0, 2.1)
        a.text(0.02, 0.97, "light = retrieved, solid = oracle", transform=a.transAxes, fontsize=8, color=T2, va="top")
    a.tick_params(colors=T2, labelsize=8)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color("#c9c8c3")
fig.suptitle("7B Q4_K_M vs 3B Q8_0 vs 3B F16 on a MacBook Air M2 (16 GB)", color=T, fontsize=13, x=0.01, ha="left")
fig.text(0.01, 0.01, "Quality: 7B and 3B Q8_0 scored in the same blind batch (n=34 answerable questions); 3B F16 from the earlier scoring round.", fontsize=8, color=T2)
fig.tight_layout(rect=(0, 0.04, 1, 0.93)); fig.savefig(ROOT/"analysis/compare_7b.png", dpi=160, facecolor=S)
