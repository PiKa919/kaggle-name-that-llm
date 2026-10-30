"""
Diagnose and fix the lab-conditional combination.

experiments/40 found the important half of the result and then broke the other half:

    flat        BA 0.3094   lab 0.6265   sibling|lab-right 0.4941
    two-stage   BA 0.1750   lab 0.2623   sibling|lab-right 0.6687

A dedicated within-lab sibling model lifts sibling discrimination from 0.494 to 0.669, which
is exactly the documented bottleneck (55-58% of errors are sibling errors). But the joint
combination collapsed lab accuracy from 0.627 to 0.262, so the two-stage total is worse.

Cause: raw LinearSVC decision values are not calibrated log-probabilities. Softmaxing a
2-member lab (zai) yields log-probs near log(1/2) while a 5-member lab (anthropic) yields near
log(1/5), and the *confidence* spread also differs per lab, so the joint is dominated by
whichever lab's sub-model happens to be sharpest rather than by P(lab | x).

Fix: keep the two stages separate and combine with one scalar,

    lp(model) = lp_lab[:, li] + alpha * lsm(sibling | lab)

sweeping alpha. alpha = 0 is "trust the lab model, order siblings by nothing"; alpha = 1 is the
full joint; alpha < 1 damps the uncalibrated sibling confidence. This also reports the
standalone accuracy of each stage so the failure is attributable.

Usage: python experiments/41_sibling_fix.py [--pool_rows 40000] [--folds 3]
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
L2I = {l: i for i, l in enumerate(LABELS)}
K = len(LABELS)


def lab_of(m):
    if m.startswith('gpt-'): return 'openai'
    if m.startswith('gemini-'): return 'google'
    if m.startswith('claude-'): return 'anthropic'
    if m.startswith('qwen'): return 'qwen'
    if m.startswith('glm-'): return 'zai'
    return 'moonshot'


LABS = sorted({lab_of(m) for m in LABELS})
LAB2I = {l: i for i, l in enumerate(LABS)}
MODEL_LAB = np.array([LAB2I[lab_of(m)] for m in LABELS])
LAB_MEMBERS = {LAB2I[l]: np.array([L2I[m] for m in LABELS if lab_of(m) == l]) for l in LABS}


def lsm(z):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = np.column_stack([-0.5 * z, 0.5 * z])
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool_rows', type=int, default=40_000)
    ap.add_argument('--folds', type=int, default=3)
    a = ap.parse_args()

    tr = pd.read_csv('data/train.csv')
    truth = tr.prompt_id.to_numpy()
    y = tr.model.map(L2I).to_numpy()

    pool = pd.read_parquet('/tmp/pool/ntl_open_pool.parquet',
                           columns=['text', 'model', 'flag_near_train_prompt',
                                    'flag_train_prompt_regen', 'flag_gemini37_routing_risk'])
    clean = pool[~pool.flag_near_train_prompt & ~pool.flag_train_prompt_regen
                 & ~pool.flag_gemini37_routing_risk & pool.model.isin(LABELS)].drop_duplicates('text')
    clean = clean.sample(min(len(clean), a.pool_rows), random_state=0)
    texts = clean.text.tolist() + tr.text.tolist()
    n_pool = len(clean)
    v = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                        max_features=150_000, dtype=np.float32)
    X = normalize(v.fit_transform(texts))
    Xp, Xt = X[:n_pool], X[n_pool:]
    py = clean.model.map(L2I).to_numpy()
    print(f'pool rows {n_pool}', flush=True)

    alphas = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
    res = []
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                 .split(tr, y, groups=truth))[:a.folds]
    for k, (itr, iva) in enumerate(folds):
        Xtr = vstack([Xp, Xt[itr]], format='csr')
        ytr = np.concatenate([py, y[itr]])
        wtr = np.concatenate([np.full(n_pool, 0.35, np.float32), np.ones(len(itr), np.float32)])
        yva = y[iva]

        flat = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight='balanced')
        flat.fit(Xtr, ytr, sample_weight=wtr)
        lp_flat = lsm(flat.decision_function(Xt[iva]))

        lab = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight='balanced')
        lab.fit(Xtr, MODEL_LAB[ytr], sample_weight=wtr)
        d_lab = lab.decision_function(Xt[iva])
        lp_lab = lsm(d_lab)
        lab_only_acc = float((MODEL_LAB[yva] == lp_lab.argmax(1)).mean())
        flat_lab_acc = float((MODEL_LAB[yva] == MODEL_LAB[lp_flat.argmax(1)]).mean())
        print(f'  fold {k}  lab accuracy: dedicated {lab_only_acc:.4f}  '
              f'derived-from-flat {flat_lab_acc:.4f}', flush=True)

        sib = {}
        for li in np.unique(MODEL_LAB[ytr]):
            members = LAB_MEMBERS[int(li)]
            sel = MODEL_LAB[ytr] == li
            if len(members) < 2 or len(np.unique(ytr[sel])) < 2:
                continue
            remap = {int(c): j for j, c in enumerate(members)}
            yy = np.array([remap[int(c)] for c in ytr[sel]])
            m = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight='balanced')
            m.fit(Xtr[sel], yy, sample_weight=wtr[sel])
            sib[int(li)] = (m, members)

        for alpha in alphas:
            lp = np.full((len(iva), K), -1e4, np.float32)
            for li in range(len(LABS)):
                members = LAB_MEMBERS[li]
                if li not in sib:
                    lp[:, members] = lp_lab[:, [li]]
                    continue
                m, members = sib[li]
                lp[:, members] = lp_lab[:, [li]] + alpha * lsm(m.decision_function(Xt[iva]))
            lp = lsm(lp)
            pred = lp.argmax(1)
            ba = balanced_accuracy_score(yva, pred)
            rl = MODEL_LAB[yva] == MODEL_LAB[pred]
            row = dict(fold=k, alpha=alpha, ba=float(ba),
                       lab_acc=float((MODEL_LAB[yva] == MODEL_LAB[pred]).mean()),
                       sib_acc=float((yva[rl] == pred[rl]).mean()) if rl.any() else 0.0)
            res.append(row)
            print(f'    alpha {alpha:.2f}  BA {ba:.4f}  lab {row["lab_acc"]:.4f}  '
                  f'sib {row["sib_acc"]:.4f}', flush=True)
        res.append(dict(fold=k, alpha=-1.0, ba=float(balanced_accuracy_score(yva, lp_flat.argmax(1))),
                        lab_acc=flat_lab_acc,
                        sib_acc=float((yva[(MODEL_LAB[yva] == MODEL_LAB[lp_flat.argmax(1)])] ==
                                       lp_flat.argmax(1)[(MODEL_LAB[yva] == MODEL_LAB[lp_flat.argmax(1)])]).mean())))

    df = pd.DataFrame(res)
    print('\n=== mean over folds ===')
    print(df.groupby('alpha')[['ba', 'lab_acc', 'sib_acc']].mean().round(4).to_string())
    print('\nalpha = -1 is the flat 18-way reference')
    json.dump(res, open('experiments/41_sibling_fix.json', 'w'), indent=1)
    print('written experiments/41_sibling_fix.json')


if __name__ == '__main__':
    main()
