import csv, pathlib, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Noto Sans CJK TC","PingFang TC","Heiti TC","Arial Unicode MS"]; plt.rcParams["axes.unicode_minus"]=False
rows=list(csv.DictReader(open(str(pathlib.Path(__file__).resolve().parent.parent/"analysis/summary.csv"))))
name={"f16":"F16","q8_0":"Q8_0","q4_k_m":"Q4_K_M","q2_k":"Q2_K"}
col={"f16":"#2a78d6","q8_0":"#eb6834","q4_k_m":"#1baf7a","q2_k":"#eda100"}
S="#fcfcfb"; T="#0b0b0b"; T2="#52514e"
fig,ax=plt.subplots(1,2,figsize=(11,4.6),facecolor=S)
panels=[("correct_retrieved","Answer quality (mean 0–2, retrieved, n=40)",(1.0,1.6)),
        ("grounded_retrieved","Grounded answers (retrieved, out of 40)",(10,36))]
for a,(k,yl,lim) in zip(ax,panels):
    a.set_facecolor(S)
    for r in rows:
        x=float(r["tg128"]); y=float(r[k]); m=r["model"]
        a.scatter(x,y,s=110,color=col[m],edgecolor=S,linewidth=2,zorder=3)
        a.annotate(f'{name[m]}\n{float(r["rss_gib"]):.2f} GiB',(x,y),xytext=(0,12),textcoords="offset points",ha="center",fontsize=9,color=T2)
    a.set_xlim(5,55); a.set_ylim(*lim)
    a.set_xlabel("Generation speed, tg128 (tok/s; further right = faster)",color=T2); a.set_ylabel(yl,color=T2)
    a.grid(color="#e6e5e1",linewidth=0.8,zorder=0); a.tick_params(colors=T2)
    for s in ("top","right"): a.spines[s].set_visible(False)
    for s in ("left","bottom"): a.spines[s].set_color("#c9c8c3")
fig.suptitle("Quantization: speed vs quality (label shows RSS memory)",color=T,fontsize=13,x=0.06,ha="left")
fig.tight_layout(rect=(0,0,1,0.94)); fig.savefig(str(pathlib.Path(__file__).resolve().parent.parent/"analysis/knee_speed_quality.png"),dpi=160,facecolor=S)
