"""
Can a lab-conditional sibling model beat the flat 18-way classifier?

AGENTS.md §2 states the bottleneck: 55-58% of errors keep the lab right and pick the wrong
sibling (`glm-5.3` vs `glm-5.3-flash`, or among the 5 Claude / 5 GPT variants). A flat
18-way linear model spends its capacity separating labs, because lab separation is easy and
the lab direction dominates the TF-IDF geometry. The sibling sub-problems are then whatever
is left over.

This measures the obvious remedy on identical features, identical pool, identical grouped
folds, so the only difference is the decision structure:

  flat        LinearSVC over 18 classes
  two-stage   P(model) = P(lab | x) * P(sibling | lab, x), the second stage trained only on
              that lab's rows and re-balanced over just that lab's classes
  residual    flat, then a per-lab re-scoring of the flat decision values using a sibling-only
              model's relative log-odds (keeps the flat lab decision, replaces the within-lab
              ordering)

Reported as balanced accuracy over all 18 classes plus the decomposition that matters:
lab accuracy (argmax over the 6 labs) and sibling accuracy given the lab is right.

Usage: python experiments/40_sibling_model.py [--pool_rows 40000] [--folds 2]
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
    if z.ndim == 1:                       # binary LinearSVC returns (n,)
        z = np.column_stack([-0.5 * z, 0.5 * z])
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def decompose(y_true, y_pred):
    """balanced accuracy, lab accuracy, and sibling accuracy given the lab is right."""
    ba = balanced_accuracy_score(y_true, y_pred)
    lab_acc = float((MODEL_LAB[y_true] == MODEL_LAB[y_pred]).mean())
    right_lab = MODEL_LAB[y_true] == MODEL_LAB[y_pred]
    sib = float((y_true[right_lab] == y_pred[right_lab]).mean()) if right_lab.any() else 0.0
    return ba, lab_acc, sib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool_rows', type=int, default=40_000)
    ap.add_argument('--folds', type=int, default=2)
    a = ap.parse_args()

    tr = pd.read_csv('data/train.csv')
    truth = tr.prompt_id.to_numpy()
    y = tr.model.map(L2I).to_numpy()

    pool = pd.read_parquet('/tmp/pool/ntl_open_pool.parquet',
                           columns=['text', 'model', 'prompt_group', 'flag_near_train_prompt',
                                    'flag_train_prompt_regen', 'flag_gemini37_routing_risk'])
    clean = pool[~pool.flag_near_train_prompt & ~pool.flag_train_prompt_regen
                 & ~pool.flag_gemini37_routing_risk & pool.model.isin(LABELS)].drop_duplicates('text')
    clean = clean.sample(min(len(clean), a.pool_rows), random_state=0)
    print(f'pool rows {len(clean)}', flush=True)

    texts = clean.text.tolist() + tr.text.tolist()
    n_pool = len(clean)
    v = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                        max_features=150_000, dtype=np.float32)
    X = normalize(v.fit_transform(texts))
    Xp, Xt = X[:n_pool], X[n_pool:]
    py = clean.model.map(L2I).to_numpy()
    plab = MODEL_LAB[py]

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
        dflat = flat.decision_function(Xt[iva])
        lp_flat = lsm(dflat)

        lab = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight='balanced')
        lab.fit(Xtr, MODEL_LAB[ytr], sample_weight=wtr)
        lp_lab = lsm(lab.decision_function(Xt[iva]))          # (n, 6)

        # sibling stage: one model per lab, trained on that lab's rows only
        sib_models = {}
        for li in np.unique(MODEL_LAB[ytr]):
            sel = MODEL_LAB[ytr] == li
            members = LAB_MEMBERS[li]
            if len(members) < 2 or len(np.unique(ytr[sel])) < 2:
                continue
            remap = {int(c): j for j, c in enumerate(members)}
            yy = np.array([remap[int(c)] for c in ytr[sel]])
            m = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight='balanced')
            m.fit(Xtr[sel], yy, sample_weight=wtr[sel])
            sib_models[int(li)] = (m, members)

        # combine
        lp_two = np.full((len(iva), K), -1e4, np.float32)
        for li in range(len(LABS)):
            members = LAB_MEMBERS[li]
            if li not in sib_models:                     # single-member lab: nothing to split
                lp_two[:, members] = lp_lab[:, [li]]
                continue
            m, members = sib_models[li]
            lp_two[:, members] = lp_lab[:, [li]] + lsm(m.decision_function(Xt[iva]))
        lp_two = lsm(lp_two)

        # residual: keep the flat decision, replace the within-lab ordering with the sibling model
        lp_res = lp_flat.copy()
        for li, (m, members) in sib_models.items():
            s = lsm(m.decision_function(Xt[iva]))
            s = s - s.mean(1, keepdims=True)                 # relative log-odds only
            lp_res[:, members] = lp_flat[:, members] - lp_flat[:, members].mean(1, keepdims=True) + s
        lp_res = lsm(lp_res)

        for tag, lp in [('flat', lp_flat), ('two-stage', lp_two), ('residual', lp_res)]:
            ba, la, sa = decompose(yva, lp.argmax(1))
            res.append(dict(fold=k, tag=tag, ba=float(ba), lab_acc=la, sib_acc=sa))
            print(f'  fold {k} {tag:10s} BA {ba:.4f}  lab {la:.4f}  sibling|lab-right {sa:.4f}', flush=True)

    df = pd.DataFrame(res)
    print('\n=== mean over folds ===')
    print(df.groupby('tag')[['ba', 'lab_acc', 'sib_acc']].mean().round(4).to_string())
    json.dump(res, open('experiments/40_sibling_model.json', 'w'), indent=1)
    print('written experiments/40_sibling_model.json')


if __name__ == '__main__':
    main()
