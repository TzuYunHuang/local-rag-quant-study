"""比對使用者覆核與 Claude 評分：完全一致率、差一級比例、偏鬆／偏嚴方向、加權 kappa。"""
import json, pathlib, collections
ROOT = pathlib.Path(__file__).resolve().parent.parent
key = json.load(open(ROOT/"review/for_user_30_key.json"))
user = {int(l.split("=")[0]): int(l.split("=")[1]) for l in open(ROOT/"review/user_scores.txt") if "=" in l}
rows = [(i, key[str(i)], user[i]) for i in sorted(user)]
exact = sum(1 for i,k,u in rows if k["claude"] == u)
print(f"完全一致 {exact}/30 = {exact/30:.0%}")
diff = collections.Counter(u - k["claude"] for i,k,u in rows)
print("使用者分數 − Claude 分數 分布：", dict(sorted(diff.items())))
print("Claude 較鬆（Claude>使用者）:", sum(1 for i,k,u in rows if k["claude"]>u), " Claude 較嚴（Claude<使用者）:", sum(1 for i,k,u in rows if k["claude"]<u))
print("\n混淆表（列＝Claude，欄＝使用者）")
for c in (0,1,2): print(c, [sum(1 for i,k,u in rows if k['claude']==c and u==v) for v in (0,1,2)])
# 二次加權 kappa
n=30; w=lambda a,b: ((a-b)/2)**2
O=sum(w(k["claude"],u) for i,k,u in rows)/n
pc=collections.Counter(k["claude"] for i,k,u in rows); pu=collections.Counter(u for i,k,u in rows)
E=sum(w(a,b)*pc[a]*pu[b] for a in (0,1,2) for b in (0,1,2))/n/n
print(f"\n加權 kappa = {1-O/E:.2f}")
print("\n不一致的題目：")
for i,k,u in rows:
    if k["claude"]!=u: print(f"  #{i} {k['qid']} {k['label']} 版本={k['model']} 條件={k['mode']}  Claude={k['claude']} 使用者={u}")
