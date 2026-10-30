"""
Decompose the fitted-weight gain: is it reallocation, or is it more centring?

experiments/45 on the harness dump gives +0.036 out-of-fold BA (0.7286 -> 0.7648), which is the
largest lever measured in this project. But the fitted vector also raises the *effective centring
weight* of every member from v20's 0.259 to 0.43-0.68:

    effective centring = twin_weight / (plain_weight + twin_weight)

and experiments/38 measured, on real train logits with 3 grouped folds, that at blend accuracy
(0.70-0.80) centring at w=1.0 costs 0.010-0.030 while w=0.25 is worth +0.003. If the +0.036 is
mostly "more centring", it is an artefact of this eval set's density (deliberately 8-18 answers
per prompt) and will not transfer to a sparser hidden test.

Four fits, same folds, same data, different constraint sets, so the gain is attributable:

  A  all components free                      the experiments/45 number
  B  plain only (no twins, no gauss)          pure reallocation of the flat members
  C  twins locked to v20's ratio, plain free  reallocation while centring stays at 0.259
  D  plain free + centring ratio swept        where the centring optimum actually is here

If C captures most of A's gain, the finding is "n14b/q25_14/deb are badly over-weighted and
ll8v5 is badly under-weighted", which is a structural claim about the members and should
transfer. If C captures little, the finding is about this eval set.

Usage: python experiments/46_weight_decomposition.py --dump experiments/oof/dump_dense
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax
from sklearn.metrics import balanced_accuracy_score

LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])


def blend(comps, w):
    tot = sum(wi * c for wi, c in zip(w, comps))
    return log_softmax(tot / max(np.sum(np.abs(w) > 0), 1), axis=1)


def fit(comps, y, l2=1e-3, restarts=4):
    K = len(comps)

    def obj(theta):
        lp = blend(comps, np.exp(theta))
        return -lp[np.arange(len(y)), y].mean() + l2 * float(np.sum(theta ** 2))

    best, bt = np.inf, None
    for r in range(restarts):
        t0 = np.zeros(K) if r == 0 else np.random.default_rng(r).normal(0, 0.7, K)
        res = minimize(obj, t0, method='L-BFGS-B', options=dict(maxiter=800))
        if res.fun < best:
            best, bt = res.fun, res.x
    return np.exp(bt)


def fold_assign(prompt, k, seed=42):
    groups = np.unique(prompt)
    gf = {g: i % k for i, g in enumerate(np.random.default_rng(seed).permutation(groups))}
    return np.array([gf[p] for p in prompt])


def oof(comps, y, fold, k):
    pred = np.zeros(len(y), int)
    for f in range(k):
        tr, va = fold != f, fold == f
        w = fit([c[tr] for c in comps], y[tr])
        pred[va] = blend([c[va] for c in comps], w).argmax(1)
    return balanced_accuracy_score(y, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default='experiments/oof/dump_dense')
    ap.add_argument('--folds', type=int, default=3)
    a = ap.parse_args()

    z = np.load(os.path.join(a.dump, 'member_lp.npz'), allow_pickle=True)
    W = json.load(open(os.path.join(a.dump, 'weights.json')))['weights']
    y, prompt = z['y'], z['prompt']
    fold = fold_assign(prompt, a.folds)

    keys = [k[4:] for k in z.files if k.startswith('lp__')]
    comp, cur = {}, {}
    for k in keys:
        v = z[f'lp__{k}']
        w = float(W.get(k, 1.0))
        if not np.abs(v).any() or abs(w) < 1e-9:
            continue
        comp[k] = v / w
        cur[k] = w

    plain = [k for k in comp if not k.endswith(('_twin', '_gauss'))]
    twin_of = {k: k + '_twin' for k in plain if k + '_twin' in comp}
    gauss = [k for k in comp if k.endswith('_gauss')]

    def eff(k):
        if k not in twin_of:
            return 0.0
        t = twin_of[k]
        return cur[t] / (cur[k] + cur[t])

    print(f'{len(y)} rows, {len(comp)} components, folds {a.folds}')
    print('\n=== v20 effective centring weights ===')
    for k in plain:
        print(f'  {k:10s} {eff(k):.3f}')

    base = blend([comp[k] for k in comp], [cur[k] for k in comp])
    print(f'\nv20 weights, out-of-fold-equivalent BA {balanced_accuracy_score(y, base.argmax(1)):.4f}')

    res = {}

    # A: everything free
    ks = list(comp)
    res['A all free'] = oof([comp[k] for k in ks], y, fold, a.folds)

    # B: plain only
    ks = plain
    res['B plain only'] = oof([comp[k] for k in ks], y, fold, a.folds)

    # C: plain free, twins locked at v20's ratio, gauss locked at v20's plain:gauss ratio
    #    model each member as a single component = plain + r*twin at its v20 ratio, then fit only
    #    the per-member scalar. This keeps centring exactly where v20 put it.
    locked, lnames = [], []
    for k in plain:
        v = comp[k].copy()
        if k in twin_of:
            v = v + eff(k) * comp[twin_of[k]]
        locked.append(v)
        lnames.append(k)
    res['C plain free, centring locked'] = oof(locked, y, fold, a.folds)

    # D: plain free + one shared centring multiplier on top of v20's ratio
    for mult in (0.0, 0.5, 1.0, 2.0, 3.0):
        comps, names = [], []
        for k in plain:
            v = comp[k].copy()
            if k in twin_of:
                r = eff(k) * mult
                v = (v + r * comp[twin_of[k]]) / (1 + r)
            comps.append(v)
            names.append(k)
        # free per-member scale, centring ratio fixed at v20*mult
        def obj(theta):
            lp = blend(comps, np.exp(theta))
            return -lp[np.arange(len(y)), y].mean()
        pred = np.zeros(len(y), int)
        for f in range(a.folds):
            tr, va = fold != f, fold == f
            w = fit([c[tr] for c in comps], y[tr])
            pred[va] = blend([c[va] for c in comps], w).argmax(1)
        res[f'D centring x{mult:g}'] = balanced_accuracy_score(y, pred)

    print('\n=== out-of-fold balanced accuracy ===')
    for k, v in sorted(res.items(), key=lambda t: -t[1]):
        print(f'  {k:34s} {v:.4f}')

    json.dump({k: float(v) for k, v in res.items()},
              open(os.path.join(a.dump, 'decomposition.json'), 'w'), indent=1)
    print('\nwritten', os.path.join(a.dump, 'decomposition.json'))


if __name__ == '__main__':
    main()
