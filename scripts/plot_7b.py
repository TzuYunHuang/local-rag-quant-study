"""Plot 7B Q4_K_M vs 3B Q8_0 vs 3B F16 (speed, memory, answer quality)."""
import pathlib, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parent.parent
# bench numbers: analysis/summary.csv (3B) and bench/7b_q4_k_m.md (7B); quality: D9 / D11 blind scoring (answerable questions, n=34)
names = ["3B F16", "3B Q8_0", "7B Q4_K_M"]
cols = ["#2a78d6", "#eb6834", "#1baf7a"]
pp = [495.08, 478.11, 186.97]; tg = [13.82, 26.00, 19.22]; rss = [6.41, 3.21, 4.55]
q_ret = [1.32, 1.29, 1.35]; q_ora = [1.65, 1.62, 1.82]
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
