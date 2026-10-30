"""
Does sibling-concentrated error rescue cluster assignment at blend accuracy?

experiments/36 measured assignment gain against base accuracy using a pool member's own
errors, and found a clean decay that crosses zero at base BA ~0.66:

    base 0.25  +0.092      base 0.50  +0.045      base 0.65  +0.004
    base 0.40  +0.059      base 0.55  +0.020      base 0.70  -0.022

That decay is driven by collisions (0.72 -> 0.33 as accuracy rises), and assignment can
only repair collisions. But the pool member's errors are spread over all 18 classes, while
a real blend's errors are concentrated on *siblings*: a glm-5.3 passage predicted
glm-5.3-flash. Sibling errors collide more often per unit error rate, because the sibling
that steals one row is also the sibling that steals its partner's row. If that is what the
blend does, assignment might still pay at 0.66.

This script settles it by simulating predictions directly, so accuracy and error
*structure* are independent knobs:

    q = 0.0   errors uniform over the 17 wrong classes
    q = 0.5   50% of errors go to a same-lab sibling, 50% uniform
    q = 1.0   every error goes to a same-lab sibling

at error rates that bracket the blend (0.25 -> 0.50), on the real recovered prompt
clusters from train. Reports gain, repair and damage for each cell.

Usage: python experiments/37_sibling_errors.py
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import normalize

LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
L2I = {l: i for i, l in enumerate(LABELS)}
K = len(LABELS)

LAB_OF = {}
for m in LABELS:
    if m.startswith('gpt-'): LAB_OF[m] = 'openai'
    elif m.startswith('gemini-'): LAB_OF[m] = 'google'
    elif m.startswith('claude-'): LAB_OF[m] = 'anthropic'
    elif m.startswith('qwen'): LAB_OF[m] = 'qwen'
    elif m.startswith('glm-'): LAB_OF[m] = 'zai'
    else: LAB_OF[m] = 'moonshot'
SIB = {L2I[m]: np.array([L2I[o] for o in LABELS if LAB_OF[o] == LAB_OF[m] and o != m])
       for m in LABELS}


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


def assign_from_pred(pred, lab):
    """Assignment with only the predicted labels as evidence (no logits), the way a blend
    with sharp posteriors would behave: build a one-hot score matrix."""
    lp = np.full((len(pred), K), -6.0, np.float32)
    lp[np.arange(len(pred)), pred] = 0.0
    out = pred.copy()
    for g in np.unique(lab):
        r = np.flatnonzero(lab == g)
        if len(r) < 2 or len(r) > 18:
            continue
        ri, ci = linear_sum_assignment(-lp[np.ix_(r, np.arange(K))])
        out[r[ri]] = ci
    return out


def simulate(y, err_rate, q, rng):
    """Flip `err_rate` of rows; each flipped row goes to a same-lab sibling with prob q."""
    pred = y.copy()
    flip = rng.random(len(y)) < err_rate
    for i in np.flatnonzero(flip):
        if q > 0 and len(SIB[y[i]]) and rng.random() < q:
            pred[i] = rng.choice(SIB[y[i]])
        else:
            pred[i] = rng.choice([c for c in range(K) if c != y[i]])
    return pred


def collision(pred, lab):
    dup = np.zeros(len(pred), bool)
    for g in np.unique(lab):
        r = np.flatnonzero(lab == g)
        if len(r) < 2:
            continue
        _, inv, cnt = np.unique(pred[r], return_inverse=True, return_counts=True)
        dup[r[cnt[inv] > 1]] = True
    return dup.mean()


def main():
    rng = np.random.default_rng(42)
    tr = pd.read_csv('data/train.csv')
    truth = tr.prompt_id.to_numpy()
    y = tr.model.map(L2I).to_numpy()

    vc = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05)
    Xc = normalize(vc.fit_transform(tr.text.tolist()))
    Sc = (Xc @ Xc.T).toarray()
    lab = capped_agglo_sim(Sc, 18, 0.05)
    print(f'clusters: {len(np.unique(lab))}  mean size {cluster_sizes(lab).mean():.2f}', flush=True)

    res = []
    for err in (0.25, 0.30, 0.34, 0.40, 0.45, 0.50):
        for q in (0.0, 0.5, 1.0):
            for rep in range(8):
                pred = simulate(y, err, q, rng)
                base = balanced_accuracy_score(y, pred)
                ap = assign_from_pred(pred, lab)
                ba = balanced_accuracy_score(y, ap)
                right = pred == y
                ch = ap != pred
                res.append(dict(err=err, q=q, rep=rep, base=float(base), assign=float(ba),
                                gain=float(ba - base),
                                repair=float((~right & ch & (ap == y)).sum() / max((~right).sum(), 1)),
                                damage=float((right & ch & (ap != y)).sum() / max(right.sum(), 1)),
                                collide=float(collision(pred, lab)),
                                collide_assigned=float(collision(ap, lab))))
    df = pd.DataFrame(res)
    print('\n=== assignment gain: error rate x sibling concentration ===')
    print(df.pivot_table(index='base', columns='q', values='gain', aggfunc='mean')
            .round(4).to_string())
    print('\n=== collision rate: before -> after assignment ===')
    print(df.groupby(['err', 'q'])[['collide', 'collide_assigned']].mean().round(3).to_string())
    print('\n=== repair vs damage at the blend operating point ===')
    print(df.groupby(['err', 'q'])[['base', 'gain', 'repair', 'damage']].mean().round(4).to_string())
    json.dump(res, open('experiments/37_sibling_errors.json', 'w'), indent=1)
    print('written experiments/37_sibling_errors.json')


if __name__ == '__main__':
    main()
