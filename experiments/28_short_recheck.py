"""Re-check short_passage_engine on the CURRENT file (it was rewritten at
02:51, after experiment 25 cached its OOF). Same protocol, so the two runs are
comparable."""
import csv, json, sys, time, numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
sys.path.insert(0, ".")
from src.short_passage_engine import ShortPassageFallbackModel, entropy_regularized_short_gate
from src.test_time_calibration import sinkhorn_knopp_transport

V = ["claude-fable-5","claude-opus-4-6","claude-opus-4-8","claude-opus-5","claude-sonnet-5",
     "gemini-3.1-pro-preview","gemini-3.6-flash","gemini-3.7-flash","gemini-3.8-flash",
     "glm-5.3","glm-5.3-flash","gpt-5.5","gpt-5.6-luna","gpt-5.6-sol","gpt-5.6-terra",
     "gpt-6-astra","kimi-k3","qwen3.8-max"]
M = {m:i for i,m in enumerate(V)}
rows=list(csv.DictReader(open("data/train.csv")))
texts=[r["text"] or "" for r in rows]; y=np.array([M[r["model"]] for r in rows])
g=np.array([r["prompt_id"] for r in rows]); wc=np.array([len(t.split()) for t in texts]); n=len(rows)
def bacc(a,p): return float(balanced_accuracy_score(a, p.argmax(1)))
def sm(z):
    z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y,groups=g))
t=time.time(); p=np.zeros((n,18))
for k,(tr,va) in enumerate(folds):
    m=ShortPassageFallbackModel(class_order=V, random_state=42+k)
    m.fit([texts[i] for i in tr],[V[y[i]] for i in tr],wc[tr])
    p[va]=m.predict_proba([texts[i] for i in va])
    print(f"fold {k} {time.time()-t:.0f}s bacc={bacc(y[va],p[va]):.4f}")
np.savez_compressed("experiments/cache_short_current.npz", p_short=p)
print("OOF short-view bacc", round(bacc(y,p),4))
d=np.load("experiments/cache_oof.npz")
base = d["lp_dense"] if bacc(y,d["lp_dense"])>bacc(y,d["lp_ng"]) else d["lp_ng"]
pb=sm(base)
cl=[np.where(g==x)[0] for x in dict.fromkeys(g)]
for ix in cl: pb[ix]=sinkhorn_knopp_transport(sm(base[ix]),reg=0.10,alpha=0.5,max_iter=200)
bf=bacc(y,pb); print("final (oracle-cluster OT):",round(bf,4))
for lo,hi in [(0,60),(60,80),(80,120),(120,160),(160,300)]:
    s=(wc>=lo)&(wc<hi)
    print(f"  bin {lo}-{hi} n={s.sum():4d} base={bacc(y[s],pb[s]):.4f} short={bacc(y[s],p[s]):.4f}")
print("module gate delta:", round(bacc(y, entropy_regularized_short_gate(pb,p,wc))-bf,4))
for w in (0.05,0.1,0.2,0.35):
    print(f"  flat w={w}: {bacc(y,(1-w)*pb+w*p):.4f} ({bacc(y,(1-w)*pb+w*p)-bf:+.4f})")
