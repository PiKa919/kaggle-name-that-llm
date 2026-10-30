"""
How does prompt-cluster assignment scale with base accuracy?

experiments/33 + 35 established, on a weak member (raw BA 0.2723, 3 folds):
    oracle-cluster assignment   0.3562  (+0.084)
    recovered @0.05 assignment  0.3532  (+0.081)   94-100% of the oracle gain
    recovered @0.05 centring    0.2908  (+0.019)   light centring is the weak sibling
    thin-cluster-only (2-6)     0.2717  (-0.001)   so 752708's "LB follows the thin columns"
                                                   is about *their* clustering, not about
                                                   thin clusters being the target

Assignment repairs *collisions*: rows of one prompt that receive the same label. A 0.66
blend has a 34% error rate instead of 73%, so it collides far less and the gain must be
smaller. This script measures the shape of that decay, because that is what decides
whether shipping assignment is worth a submission.

Strength knob: inject the true label into a fraction `p` of the rows that the base gets
wrong. Base BA then moves continuously from 0.27 to ~0.75 with the pool member's own
error structure preserved. At each level we report
  * gain        balanced accuracy after assignment minus before
  * damage      fraction of rows the base got RIGHT that assignment moved to a wrong label
  * repair      fraction of rows the base got WRONG that assignment moved to the right label
so the net is auditable rather than a single number.

Usage: python experiments/36_assign_strength.py
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


def assign(lp, labels, cap=18):
    pred = lp.argmax(1).copy()
    sz = cluster_sizes(labels)
    for g in np.unique(labels):
        r = np.flatnonzero(labels == g)
        if len(r) < 2 or len(r) > cap:
            continue
        cost = -lp[np.ix_(r, np.arange(K))]
        ri, ci = linear_sum_assignment(cost)
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
        base_pred = lp.argmax(1)
        wrong = np.flatnonzero(base_pred != yva)

        labs = {th: capped_agglo_sim(Sc[np.ix_(iva, iva)], 18, th) for th in (0.02, 0.03, 0.05, 0.10)}
        for th, lab in labs.items():
            print(f'fold {k} cut {th}: ARI {adjusted_rand_score(tva, lab):.3f} '
                  f'mean size {cluster_sizes(lab).mean():.2f} '
                  f'purity {_purity(lab, tva):.3f}', flush=True)

        for p in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            n_inj = int(round(p * len(wrong)))
            inj = rng.choice(wrong, size=n_inj, replace=False) if n_inj else np.array([], int)
            pred = base_pred.copy()
            pred[inj] = yva[inj]
            base_ba = balanced_accuracy_score(yva, pred)
            for th, lab in labs.items():
                ap = assign(lp, lab)
                # apply the same injection to the assigned prediction so the comparison
                # is base-vs-assigned at identical injected strength
                ap = ap.copy()
                ap[inj] = yva[inj]
                ba = balanced_accuracy_score(yva, ap)
                right = pred == yva
                changed = ap != pred
                row = dict(fold=k, p=p, cut=th, base_ba=float(base_ba), assign_ba=float(ba),
                           gain=float(ba - base_ba),
                           damage=float((right & changed & (ap != yva)).sum() / max(right.sum(), 1)),
                           repair=float((~right & changed & (ap == yva)).sum() / max((~right).sum(), 1)),
                           n_changed=int(changed.sum()),
                           collision=float(_collision(pred, lab)))
                res.append(row)
                print(f'  p={p:.1f} cut={th:.2f}  base {base_ba:.4f} -> assign {ba:.4f} '
                      f'({row["gain"]:+.4f})  repair {row["repair"]:.3f} damage {row["damage"]:.3f} '
                      f'collide {row["collision"]:.3f}', flush=True)

    df = pd.DataFrame(res)
    print('\n=== gain by base accuracy and cut (mean over folds) ===')
    df['bin'] = (df.base_ba * 20).round() / 20
    piv = df.pivot_table(index='bin', columns='cut', values='gain', aggfunc='mean').round(4)
    piv['collide'] = df.groupby('bin').collision.mean().round(3)
    print(piv.to_string())
    json.dump(res, open('experiments/36_assign_strength.json', 'w'), indent=1)
    print('written experiments/36_assign_strength.json')


def _purity(lab, truth):
    df = pd.DataFrame({'c': lab, 't': truth})
    g = df.groupby('c')
    return float(((g.t.agg(lambda s: s.value_counts().iloc[0]) / g.size()) * g.size()).sum() / len(lab))


def _collision(pred, lab):
    dup = np.zeros(len(pred), bool)
    for g in np.unique(lab):
        r = np.flatnonzero(lab == g)
        if len(r) < 2:
            continue
        _, inv, cnt = np.unique(pred[r], return_inverse=True, return_counts=True)
        dup[r[cnt[inv] > 1]] = True
    return dup.mean()


if __name__ == '__main__':
    main()
