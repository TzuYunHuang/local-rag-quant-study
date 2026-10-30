"""Compare the user's review with Claude's scores: exact agreement, off-by-one share, direction of bias (lenient/strict), weighted kappa."""
import json, pathlib, collections
ROOT = pathlib.Path(__file__).resolve().parent.parent
key = json.load(open(ROOT/"review/for_user_30_key.json"))
user = {int(l.split("=")[0]): int(l.split("=")[1]) for l in open(ROOT/"review/user_scores.txt") if "=" in l}
rows = [(i, key[str(i)], user[i]) for i in sorted(user)]
exact = sum(1 for i,k,u in rows if k["claude"] == u)
print(f"Exact agreement {exact}/30 = {exact/30:.0%}")
diff = collections.Counter(u - k["claude"] for i,k,u in rows)
print("User score - Claude score distribution:", dict(sorted(diff.items())))
print("Claude more lenient (Claude>user):", sum(1 for i,k,u in rows if k["claude"]>u), " Claude stricter (Claude<user):", sum(1 for i,k,u in rows if k["claude"]<u))
print("\nConfusion table (rows = Claude, columns = user)")
for c in (0,1,2): print(c, [sum(1 for i,k,u in rows if k['claude']==c and u==v) for v in (0,1,2)])
# quadratic weighted kappa
n=30; w=lambda a,b: ((a-b)/2)**2
O=sum(w(k["claude"],u) for i,k,u in rows)/n
pc=collections.Counter(k["claude"] for i,k,u in rows); pu=collections.Counter(u for i,k,u in rows)
E=sum(w(a,b)*pc[a]*pu[b] for a in (0,1,2) for b in (0,1,2))/n/n
print(f"\nWeighted kappa = {1-O/E:.2f}")
print("\nDisagreements:")
for i,k,u in rows:
    if k["claude"]!=u: print(f"  #{i} {k['qid']} {k['label']} model={k['model']} mode={k['mode']}  Claude={k['claude']} user={u}")
