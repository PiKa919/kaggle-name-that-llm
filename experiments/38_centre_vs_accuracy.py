"""
Centring vs assignment as a function of base accuracy, on real logits.

experiments/36 gave the assignment curve (real logits, recovered clusters @0.05):
    base 0.25 +0.092   base 0.45 +0.031   base 0.55 +0.020
    base 0.40 +0.059   base 0.50 +0.045   base 0.65 +0.004   base 0.70 -0.022
Assignment crosses zero at ~0.66, i.e. at the blend's accuracy. That matches the two
public reports (751802: Hungarian OOF +0.107 but LB 0.350 vs 0.353; 752357: flat to
negative), so assignment is not the v21 lever.

Centring is a different mechanism: it subtracts the prompt's shared direction instead of
forcing distinctness, so it cannot destroy a correct row by construction, and 750213's
simulation says it never loses when groups are true. experiments/33 measured it only at a
weak base. This script measures centring on the same injection sweep as experiments/36,
alongside assignment and the combination, so the v21 parameter choice is made at the
accuracy the blend actually has rather than at the member's.

Also sweeps the clustering cut, because the pipeline ships --adapt_thr 0.15 (ARI 0.867,
mean cluster size 7.2) while cut 0.05 recovers prompts almost perfectly (ARI 0.988,
mean size 17.4) - the v20 setting is the worst of the cuts measured.

Usage: python experiments/38_centre_vs_accuracy.py
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
L2I = {l: i for i, l in enumerate(LABELS)}
K = len(LABELS)


def capped_agglo_sim(S, cap=18, thr=0.15):
    n = len(S); A = S.astype(np.float32).copy(); np.fill_diagonal(A, -np.inf)
    size = np.ones(n, np.int32); alive = np.ones(n, bool); lab = np.arange(n)
    while True:
        ok = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        M = np.where(ok, A, -np.inf); k = int(np.argmax(M)); i, j = divmod(k, n)
        if not np.isfinite(M[i, j]) or M[i, j] < thr:
            break
        new = (size[i] * A[i] + size[j] * A[j]) / (size[i] + size[j])
        A[i] = new; A[:, i] = new; A[i, i] = -np.inf; A[j] = -np.inf; A[:, j] = -np.inf
        size[i] += size[j]; alive[j] = False; lab[lab == j] = i
    return lab


def cluster_sizes(labels):
    _, inv, cnt = np.unique(labels, return_inverse=True, return_counts=True)
    return cnt[inv]


def centre(lp, labels, w, mode='mean'):
    """Subtract w * the cluster's mean log-post from every member. mode='mean' is the
    pipeline's soft centring; mode='loo' is leave-one-out (cavity), which cannot drag a
    row toward its own evidence."""
    out = lp.copy()
    for g in np.unique(labels):
        r = np.flatnonzero(labels == g)
        if len(r) < 2:
            continue
        if mode == 'mean':
            out[r] -= w * lp[r].mean(0, keepdims=True)
        else:
            s = lp[r].sum(0, keepdims=True)
            out[r] -= w * (s - lp[r]) / (len(r) - 1)
    return out


def assign(lp, labels, cap=18):
    pred = lp.argmax(1).copy()
    for g in np.unique(labels):
        r = np.flatnonzero(labels == g)
        if len(r) < 2 or len(r) > cap:
            continue
        ri, ci = linear_sum_assignment(-lp[np.ix_(r, np.arange(K))])
        pred[r[ri]] = ci
    return pred


def main():
    rng = np.random.default_rng(42)
    tr = pd.read_csv('data/train.csv')
    truth = tr.prompt_id.to_numpy()
    y = tr.model.map(L2I).to_numpy()

    pool = pd.read_parquet('/tmp/pool/ntl_open_pool.parquet',
                           columns=['text', 'model', 'flag_near_train_prompt',
                                    'flag_train_prompt_regen', 'flag_gemini37_routing_risk'])
    clean = pool[~pool.flag_near_train_prompt & ~pool.flag_train_prompt_regen
                 & ~pool.flag_gemini37_routing_risk & pool.model.isin(LABELS)].drop_duplicates('text')
    clean = clean.sample(min(len(clean), 20000), random_state=0)
    texts = clean.text.tolist() + tr.text.tolist()
    n_pool = len(clean)
    v = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                        max_features=150_000, dtype=np.float32)
    X = normalize(v.fit_transform(texts))
    Xp, Xt = X[:n_pool], X[n_pool:]
    py = clean.model.map(L2I).to_numpy()

    vc = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05)
    Xc = normalize(vc.fit_transform(tr.text.tolist()))
    Sc = (Xc @ Xc.T).toarray()

    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                 .split(tr, y, groups=truth))[:3]
    res = []
    for k, (itr, iva) in enumerate(folds):
        svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight='balanced')
        svc.fit(vstack([Xp, Xt[itr]], format='csr'), np.concatenate([py, y[itr]]),
                sample_weight=np.concatenate([np.full(n_pool, 0.35, np.float32),
                                              np.ones(len(itr), np.float32)]))
        d = svc.decision_function(Xt[iva])
        lp = d - d.max(1, keepdims=True)
        lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
        yva, tva = y[iva], truth[iva]
        wrong = np.flatnonzero(lp.argmax(1) != yva)
        labs = {th: capped_agglo_sim(Sc[np.ix_(iva, iva)], 18, th) for th in (0.05, 0.10, 0.15)}
        print(f'fold {k}: ' + '  '.join(f'cut {th} ARI {adjusted_rand_score(tva, l):.3f} '
                                       f'size {cluster_sizes(l).mean():.1f}' for th, l in labs.items()),
              flush=True)

        for p in (0.0, 0.2, 0.4, 0.6, 0.75):
            n_inj = int(round(p * len(wrong)))
            inj = rng.choice(wrong, size=n_inj, replace=False) if n_inj else np.array([], int)

            def score(pred):
                pred = pred.copy(); pred[inj] = yva[inj]
                return balanced_accuracy_score(yva, pred)

            base = score(lp.argmax(1))
            for th, lab in labs.items():
                cells = [('centre mean w=0.25', centre(lp, lab, 0.25).argmax(1)),
                         ('centre mean w=1.0', centre(lp, lab, 1.0).argmax(1)),
                         ('centre loo w=0.25', centre(lp, lab, 0.25, 'loo').argmax(1)),
                         ('centre loo w=1.0', centre(lp, lab, 1.0, 'loo').argmax(1)),
                         ('assign', assign(lp, lab)),
                         ('centre loo w=0.25 + assign', assign(centre(lp, lab, 0.25, 'loo'), lab))]
                for tag, pred in cells:
                    ba = score(pred)
                    res.append(dict(fold=k, p=p, cut=th, tag=tag, base=float(base), ba=float(ba),
                                    gain=float(ba - base)))
            print(f'  p={p:.2f} base {base:.4f}', flush=True)

    df = pd.DataFrame(res)
    df['bin'] = (df.base * 20).round() / 20
    print('\n=== gain over base, mean over folds (rows = base accuracy bin) ===')
    for cut in sorted(df.cut.unique()):
        sub = df[df.cut == cut]
        piv = sub.pivot_table(index='bin', columns='tag', values='gain', aggfunc='mean').round(4)
        print(f'\n--- cut {cut} ---')
        print(piv.to_string())
    print('\n=== best tag per accuracy bin, averaged over cuts ===')
    print(df.pivot_table(index='bin', columns='tag', values='gain', aggfunc='mean').round(4).to_string())
    json.dump(res, open('experiments/38_centre_vs_accuracy.json', 'w'), indent=1)
    print('\nwritten experiments/38_centre_vs_accuracy.json')


if __name__ == '__main__':
    main()
