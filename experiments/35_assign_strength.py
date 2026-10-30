"""
Does prompt-cluster assignment still pay when the base model is *strong*?

experiments/33 measured, on a weak single member (raw BA 0.2891):
    oracle assignment           0.3502  (+0.061)
    recovered assignment @0.05  0.3428  (+0.054)   <- ARI 0.954
    recovered assignment @0.10  0.3298  (+0.041)
    recovered assignment @0.15  0.3076  (+0.019)   <- what v20 actually uses

The blend is not that member. It scores 0.661 on the LB, so its raw argmax is far
stronger and there are far fewer duplicate predictions inside a prompt. Assignment can
only fix *collisions*, so its gain must shrink as the base gets better. This script
measures how fast.

Emulating a stronger base without training anything: temperature-sharpen the log-posts
(lp * T, renormalised). That monotonically raises raw BA and monotonically reduces the
number of rows that share a prediction inside a cluster, i.e. exactly the failure mode
assignment repairs. If the gain survives at T where raw BA ~0.50-0.60, the lever is real
for the blend; if it dies, the lever is a weak-member artefact and v21 should not spend a
submission on it.

Also measures:
  * collision rate (rows whose argmax equals another row's argmax inside the same cluster)
  * assignment using all K columns (correct formulation) vs the top-s column preselect
    that experiments/33 used
  * assignment + light centring, and assignment restricted to thin clusters only
  * 3 folds for a mean +- std

Usage: python experiments/35_assign_strength.py
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
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


def assign(lp, labels, min_size=2, max_size=18, all_cols=True):
    """Distinct-model assignment inside each cluster. Rows in clusters outside
    [min_size, max_size] keep their argmax."""
    pred = lp.argmax(1).copy()
    sz = cluster_sizes(labels)
    for g in np.unique(labels):
        r = np.flatnonzero(labels == g)
        s = len(r)
        if s < min_size or s > max_size:
            continue
        if all_cols:
            cols = np.arange(K)                      # s x K -> s distinct columns, optimal
        else:
            cols = np.argsort(-lp[r].max(0))[:s]     # the experiments/33 heuristic
        cost = -lp[np.ix_(r, cols)]
        ri, ci = linear_sum_assignment(cost)
        pred[r[ri]] = cols[ci]
    return pred


def centre(lp, labels, w):
    out = lp.copy()
    for g in np.unique(labels):
        r = np.flatnonzero(labels == g)
        if len(r) >= 2:
            out[r] -= w * lp[r].mean(0, keepdims=True)
    return out


def collision_rate(pred, labels):
    """Fraction of rows whose prediction is shared with another row of the same cluster."""
    dup = np.zeros(len(pred), bool)
    for g in np.unique(labels):
        r = np.flatnonzero(labels == g)
        if len(r) < 2:
            continue
        _, inv, cnt = np.unique(pred[r], return_inverse=True, return_counts=True)
        dup[r[cnt[inv] > 1]] = True
    return float(dup.mean())


def main():
    tr = pd.read_csv('data/train.csv')
    truth = tr.prompt_id.to_numpy()
    y = tr.model.map(L2I).to_numpy()

    pool = pd.read_parquet('/tmp/pool/ntl_open_pool.parquet',
                           columns=['text', 'model', 'flag_near_train_prompt',
                                    'flag_train_prompt_regen', 'flag_gemini37_routing_risk'])
    clean = pool[~pool.flag_near_train_prompt & ~pool.flag_train_prompt_regen
                 & ~pool.flag_gemini37_routing_risk & pool.model.isin(LABELS)].drop_duplicates('text')
    clean = clean.sample(min(len(clean), 20000), random_state=0)
    print(f'pool rows {len(clean)}', flush=True)

    texts = clean.text.tolist() + tr.text.tolist()
    n_pool = len(clean)
    v = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                        max_features=150_000, dtype=np.float32)
    X = normalize(v.fit_transform(texts))
    Xp, Xt = X[:n_pool], X[n_pool:]
    py = clean.model.map(L2I).to_numpy()

    # prompt-recovery view, fit on the eval fold only (as the pipeline does on test)
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
        lp0 = d - d.max(1, keepdims=True)
        lp0 = lp0 - np.log(np.exp(lp0).sum(1, keepdims=True))
        yva, tva = y[iva], truth[iva]

        print(f'\n--- fold {k}  n={len(iva)} ---', flush=True)
        lab_o = capped_agglo_sim(Sc[np.ix_(iva, iva)], 18, 0.05)
        lab_t = tva  # oracle
        from sklearn.metrics import adjusted_rand_score
        print(f'  cluster: recovered ARI {adjusted_rand_score(tva, lab_o):.3f} '
              f'mean size {cluster_sizes(lab_o).mean():.2f}', flush=True)

        def rep(tag, pred, base=None):
            ba = balanced_accuracy_score(yva, pred)
            res.append(dict(fold=k, tag=tag, ba=float(ba), T=base))
            print(f'  {tag:52s} BA {ba:.4f}', flush=True)
            return ba

        for T in (1.0, 2.0, 3.0, 5.0):
            lpT = lp0 * T
            lpT = lpT - lpT.max(1, keepdims=True)
            lpT = lpT - np.log(np.exp(lpT).sum(1, keepdims=True))
            base = rep(f'T={T:g} raw argmax', lpT.argmax(1), T)
            print(f'      collision rate {collision_rate(lpT.argmax(1), lab_o):.3f} '
                  f'(oracle clusters {collision_rate(lpT.argmax(1), lab_t):.3f})', flush=True)
            a_o = rep(f'T={T:g} assign @0.05 all-cols', assign(lpT, lab_o), T)
            a_t = rep(f'T={T:g} assign oracle-clusters', assign(lpT, lab_t), T)
            rep(f'T={T:g} assign @0.05 top-s-cols', assign(lpT, lab_o, all_cols=False), T)
            rep(f'T={T:g} assign @0.05 thin(2-6) only', assign(lpT, lab_o, max_size=6), T)
            rep(f'T={T:g} centre w=0.25 @0.05', centre(lpT, lab_o, 0.25).argmax(1), T)
            rep(f'T={T:g} assign + centre w=0.25', assign(centre(lpT, lab_o, 0.25), lab_o), T)
            print(f'      -> recovered assignment keeps {100 * (a_o - base) / max(a_t - base, 1e-9):.0f}%'
                  f' of the oracle gain', flush=True)

    df = pd.DataFrame(res)
    print('\n=== mean over folds, by temperature ===', flush=True)
    piv = df.pivot_table(index='tag', values='ba', aggfunc=['mean', 'std']).round(4)
    print(piv.to_string(), flush=True)
    json.dump(res, open('experiments/35_assign_strength.json', 'w'), indent=1)
    print('written experiments/35_assign_strength.json')


if __name__ == '__main__':
    main()
