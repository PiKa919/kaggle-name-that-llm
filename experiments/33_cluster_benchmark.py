"""
Is test-time prompt recovery the big lever, or a dead end?

Why this matters
----------------
The hidden test is ~4,000 passages and the train set is 150 prompts x 18 answers.
4,000 / 18 = 222 prompts, so the test is probably *dense*: many answers per prompt.
If prompts could be recovered at test time, every prediction stops being an 18-way
problem and becomes a within-cluster assignment where each model answers at most once.

Public work reports Hungarian assignment as "flat to negative" on the LB (752357) and
752708 reports that the LB follows the *thin-cluster* columns, i.e. their clustering
produced 2-3 rows per cluster. Both point at the same thing: the assignment step is not
the problem, the clustering is.

This script measures exactly that, on train, where the true prompt ids are known:

  part 1  clustering quality (ARI, purity, size distribution) for several content
          representations and cuts, against the 150 true prompts;
  part 2  end-to-end balanced accuracy of a pool-trained member on held-out prompts,
          scored (a) raw, (b) with soft centring on recovered clusters,
          (c) with constrained assignment inside recovered clusters,
          and the same three with *true* prompt ids as the oracle ceiling.

Usage: python experiments/33_cluster_benchmark.py [--pool_rows 20000]
"""
import argparse
import json
import re
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
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


def content_matrix(texts, ngram=(1, 2), maxf=200_000, max_df=1.0):
    v = TfidfVectorizer(stop_words='english', ngram_range=ngram, sublinear_tf=True,
                        min_df=2, max_features=maxf, max_df=max_df, dtype=np.float32)
    return normalize(v.fit_transform(texts))


def capped_agglo_sim(S, cap=18, thr=0.15):
    """The pipeline's own grouping: average-linkage on a similarity matrix, never past `cap`,
    stop when the best average similarity falls below `thr` (verbatim from the 0.660 notebook)."""
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


def style_matrix(texts):
    v = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), lowercase=False, sublinear_tf=True,
                        min_df=2, max_features=200_000, dtype=np.float32)
    return normalize(v.fit_transform(texts))


def capped_agglo(D, thresh, cap=18):
    """Average-linkage clustering on a cosine distance matrix, with clusters split above `cap`."""
    lab = fcluster(linkage(squareform(D, checks=False), method='average'), thresh, criterion='distance')
    out = lab.copy()
    nxt = lab.max() + 1
    for g in np.unique(lab):
        idx = np.flatnonzero(lab == g)
        if len(idx) <= cap:
            continue
        # split the oversized cluster by re-running the same linkage on its own block
        sub = D[np.ix_(idx, idx)]
        sl = fcluster(linkage(squareform(sub, checks=False), method='average'), thresh, criterion='distance')
        for s in np.unique(sl):
            if s == sl.min():
                out[idx[sl == s]] = g
            else:
                out[idx[sl == s]] = nxt
                nxt += 1
    return out


def quality(labels, truth):
    """ARI, size-weighted purity, mean cluster size, fraction of rows in clusters of 2..18."""
    df = pd.DataFrame({'c': labels, 't': truth})
    g = df.groupby('c')
    pur = (g.t.agg(lambda s: s.value_counts().iloc[0]) / g.size()).to_numpy()
    size = g.size().to_numpy()
    sz = np.bincount(pd.factorize(labels)[0])
    return dict(ari=float(adjusted_rand_score(truth, labels)),
                purity=float((pur * size).sum() / size.sum()),
                n_clusters=int(len(np.unique(labels))),
                mean_size=float(sz.mean()),
                frac_2_18=float(sz[(sz >= 2) & (sz <= 18)].sum() / sz.sum()))


def cluster_rows(labels, rows):
    """Boolean mask: rows that sit in a cluster of size 2..18."""
    lab = labels[rows]
    _, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    ok = (cnt[inv] >= 2) & (cnt[inv] <= 18)
    return ok


def centre_lp(lp, labels, mask):
    out = lp.copy()
    lab = labels[mask]
    uniq, inv = np.unique(lab, return_inverse=True)
    for j in range(len(uniq)):
        r = np.flatnonzero(inv == j)
        if len(r) >= 2:
            out[np.flatnonzero(mask)[r]] -= lp[np.flatnonzero(mask)[r]].mean(0, keepdims=True)
    return out


def assign(lp, labels, mask):
    """Inside each cluster, assign distinct models by maximum total log-probability."""
    out = lp.argmax(1).copy()
    idx_all = np.flatnonzero(mask)
    lab = labels[idx_all]
    for g in np.unique(lab):
        r = idx_all[lab == g]
        if len(r) < 2:
            continue
        s = len(r)
        cols = np.argsort(-lp[r].max(0))[:s] if s < K else np.arange(K)
        cost = -lp[np.ix_(r, cols)]
        ri, ci = linear_sum_assignment(cost)
        out[r[ri]] = cols[ci]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool_rows', type=int, default=20_000)
    ap.add_argument('--folds', type=int, default=1)
    ap.add_argument('--out', default='experiments/33_cluster_benchmark.json')
    ap.add_argument('--skip1', action='store_true')
    ap.add_argument('--cuts', default='0.05,0.10,0.15')
    a = ap.parse_args()

    tr = pd.read_csv('data/train.csv')
    truth = tr.prompt_id.to_numpy()
    y = tr.model.map(L2I).to_numpy()
    print(f'train {tr.shape}, {len(np.unique(truth))} prompts', flush=True)

    # ---------------- part 1: clustering quality
    print('\n=== part 1: prompt recovery on train ===', flush=True)
    # the pipeline's view, verbatim
    Cp = content_matrix(tr.text.tolist(), ngram=(1, 1), maxf=200_000, max_df=0.05)
    C2 = content_matrix(tr.text.tolist(), ngram=(1, 2), maxf=200_000)
    S = style_matrix(tr.text.tolist())
    Sc, S2, Ss = (Cp @ Cp.T).toarray(), (C2 @ C2.T).toarray(), (S @ S.T).toarray()
    same = truth[:, None] == truth[None, :]
    iu = np.triu_indices(len(truth), 1)
    print(f"  cosine similarity: same prompt median {np.median(Sc[iu][same[iu]]):.3f}  "
          f"cross prompt median {np.median(Sc[iu][~same[iu]]):.3f}  "
          f"(unigram max_df .05 view)", flush=True)
    print(f"  cosine similarity: same prompt median {np.median(S2[iu][same[iu]]):.3f}  "
          f"cross prompt median {np.median(S2[iu][~same[iu]]):.3f}  (1-2 gram view)", flush=True)
    cuts = [float(c) for c in a.cuts.split(',')]
    rows = [] if not a.skip1 else [dict(rep='pipeline unigram', cut=c) for c in cuts]
    for name, Sx, cuts_ in ([] if a.skip1 else [('pipeline unigram', Sc, cuts),
                           ('word 1-2', S2, cuts), ('char 3-5', Ss, cuts),
                           ('mean(word,char)', (S2 + Ss) / 2, cuts)]):
        for th in cuts_:
            t0 = time.time()
            lab = capped_agglo_sim(Sx, 18, th)
            q = quality(lab, truth)
            q.update(rep=name, cut=th, sec=round(time.time() - t0, 1))
            rows.append(q)
            print(f"  {name:18s} sim>={th:.2f}  ARI {q['ari']:.3f}  purity {q['purity']:.3f}  "
                  f"clusters {q['n_clusters']:4d}  mean {q['mean_size']:5.2f}  in 2-18 {q['frac_2_18']:.2f}", flush=True)
    qdf = pd.DataFrame(rows)
    best = qdf.sort_values('ari', ascending=False).head(3) if 'ari' in qdf else qdf
    if 'ari' in qdf:
        print('\nbest cuts by ARI:\n', best[['rep', 'cut', 'ari', 'purity', 'n_clusters', 'mean_size', 'frac_2_18']].to_string(index=False), flush=True)

    # ---------------- part 2: does better clustering buy accuracy?
    print('\n=== part 2: end-to-end, held-out prompts ===', flush=True)
    pool = pd.read_parquet('/tmp/pool/ntl_open_pool.parquet',
                           columns=['text', 'model', 'n_words', 'flag_near_train_prompt',
                                    'flag_train_prompt_regen', 'flag_gemini37_routing_risk'])
    clean = pool[~pool.flag_near_train_prompt & ~pool.flag_train_prompt_regen
                 & ~pool.flag_gemini37_routing_risk & pool.model.isin(LABELS)]
    clean = clean.drop_duplicates('text')
    clean = clean.sample(min(len(clean), a.pool_rows), random_state=0)
    print(f'pool rows {len(clean)}', flush=True)

    texts = clean.text.tolist() + tr.text.tolist()
    n_pool = len(clean)
    X = content_matrix(texts, ngram=(1, 2), maxf=150_000)
    Xp, Xt = X[:n_pool], X[n_pool:]
    py = clean.model.map(L2I).to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(sgkf.split(tr, y, groups=truth))[:a.folds]
    res = []
    for k, (itr, iva) in enumerate(folds):
        svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight='balanced')
        from scipy.sparse import vstack
        svc.fit(vstack([Xp, Xt[itr]], format='csr'), np.concatenate([py, y[itr]]),
                sample_weight=np.concatenate([np.full(n_pool, 0.35, dtype=np.float32),
                                              np.ones(len(itr), dtype=np.float32)]))
        d = svc.decision_function(Xt[iva])
        lp = d - d.max(1, keepdims=True)
        lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
        yva, tva = y[iva], truth[iva]

        def report(tag, pred):
            ba = balanced_accuracy_score(yva, pred)
            print(f'  fold {k} {tag:46s} BA {ba:.4f}', flush=True)
            res.append(dict(fold=k, tag=tag, ba=float(ba)))

        report('raw argmax', lp.argmax(1))

        # true prompts = oracle
        m_true = cluster_rows(tva, np.arange(len(iva)))
        for w in (0.25, 0.5, 1.0):
            lpc = lp.copy(); lpc[m_true] = lp[m_true] - w * (lp[m_true] - centre_lp(lp, tva, m_true)[m_true])
            report(f'ORACLE soft centring w={w}', lpc.argmax(1))
        report('ORACLE constrained assignment', assign(lp, tva, m_true))

        # recovered prompts, at the best few cuts
        SRC = {'pipeline unigram': Sc, 'word 1-2': S2, 'char 3-5': Ss, 'mean(word,char)': (S2 + Ss) / 2}
        reps = (qdf.sort_values('ari', ascending=False).head(2)[['rep', 'cut']].to_numpy()
                if 'ari' in qdf else [('pipeline unigram', c) for c in cuts])
        for rep_name, cut in reps:
            Sv = SRC[rep_name][np.ix_(iva, iva)]
            lab = capped_agglo_sim(Sv, 18, float(cut))
            q = quality(lab, tva)
            m = cluster_rows(lab, np.arange(len(iva)))
            print(f"    recovered clusters ({rep_name}, cut {cut}): ARI {q['ari']:.3f} purity {q['purity']:.3f} "
                  f"mean size {q['mean_size']:.2f} in 2-18 {q['frac_2_18']:.2f}", flush=True)
            for w in (0.25, 0.5, 1.0):
                lpc = lp.copy(); lpc[m] = lp[m] - w * (lp[m] - centre_lp(lp, lab, m)[m])
                report(f'recovered soft centring w={w} (cut {cut})', lpc.argmax(1))
            report(f'recovered constrained assignment (cut {cut})', assign(lp, lab, m))

    df = pd.DataFrame(res)
    print('\n=== mean over folds ===')
    print(df.groupby('tag').ba.agg(['mean', 'std']).round(4).to_string())
    json.dump(dict(quality=rows, end_to_end=res), open(a.out, 'w'), indent=1)
    print('written', a.out)


if __name__ == '__main__':
    main()
