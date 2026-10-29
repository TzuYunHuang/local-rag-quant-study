"""把 bench / eval / ppl 的數據合成一張總表 analysis/summary.csv（之後所有圖都從這個檔來）。"""
import json, re, csv, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
TAGS = ["f16", "q8_0", "q4_k_m", "q2_k"]

def bench(tag):                      # 從 llama-bench 的 markdown 表抓 pp512 / tg128 的 t/s
    t = (ROOT/f"bench/{tag}.md").read_text()
    pp = re.search(r"pp512\s*\|\s*([\d.]+)", t).group(1)
    tg = re.search(r"tg128\s*\|\s*([\d.]+)", t).group(1)
    return float(pp), float(tg)

def rss_gib(tag):                    # /usr/bin/time -l 的 maximum resident set size（bytes）
    t = (ROOT/f"bench/{tag}.time").read_text()
    return int(re.search(r"(\d+)\s+maximum resident set size", t).group(1)) / 2**30

def ppl(tag):                        # 取 Final estimate 的 PPL 與誤差
    t = (ROOT/f"ppl/{tag}.log").read_text()
    m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", t)
    return float(m.group(1)), float(m.group(2))

rows = [json.loads(l) for l in open(ROOT/"eval/scores_claude.jsonl", encoding="utf-8") if l.strip()]
def mean(xs): return sum(xs) / len(xs)

out = []
for tag in TAGS:
    r = [x for x in rows if x["model"] == tag and x["mode"] == "retrieved"]
    o = [x for x in rows if x["model"] == tag and x["mode"] == "oracle"]
    pp, tg = bench(tag); p, pe = ppl(tag)
    out.append(dict(model=tag, pp512=pp, tg128=tg, rss_gib=round(rss_gib(tag), 2),
        correct_retrieved=round(mean([x["correct"] for x in r]), 2), n_retrieved=len(r),
        correct_oracle=round(mean([x["correct"] for x in o]), 2), n_oracle=len(o),
        grounded_retrieved=sum(x["grounded"] for x in r), ppl=p, ppl_err=pe))
with open(ROOT/"analysis/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
for o in out: print(o)
