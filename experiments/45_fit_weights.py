"""
Fit blend weights on the harness dump, offline, on CPU.

Consumes `/kaggle/working/dump_*/member_lp.npz` produced by `notebooks/oof/`. Every `LP[key]` in
that dump is a per-member log-prob matrix with its blend weight already baked in, so the fitter
first divides the weight back out (using `weights.json`), then refits.

What this replaces: the blend weights in the 0.660 pipeline are inherited constants. Nobody has
ever fitted them, because no honest OOF existed for any frozen member. This makes them a fitting
problem.

Method:
  * objective  multiclass log-loss on a linear blend of unweighted member log-probs, softplus-
               parameterised so weights stay positive, L-BFGS
  * evaluation balanced accuracy, which is the competition metric
  * honesty    prompt-grouped 3-fold CV over the eval set: weights are fit on two folds and scored
               on the third, so the reported gain is out-of-fold. A fitted weight vector that only
               wins in-sample is worthless.
  * Sinkhorn   re-applied exactly as infer.py does (shrink toward the train prior), because a
               weight change shifts the implied test marginal.

Usage:
  python experiments/45_fit_weights.py --dump dump_dense/dump_dense
  python experiments/45_fit_weights.py --dump dump_nat/dump_nat
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax
from sklearn.metrics import balanced_accuracy_score

LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])


def sinkhorn(lp, y, strength, target_mix):
    """infer.py's damped marginal balancing, reproduced so fitted weights are scored the same way."""
    if strength <= 0:
        return lp
    P = np.exp(lp - lp.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    pred_marg = np.clip(P.mean(0), 1e-6, None); pred_marg /= pred_marg.sum()
    prior = np.bincount(y, minlength=18).astype(np.float64) + 1.0; prior /= prior.sum()
    target = (1 - target_mix) * pred_marg + target_mix * prior
    target = np.clip(target, 1e-6, None); target /= target.sum()
    Q = P.copy()
    for _ in range(100):
        Q *= (len(Q) * target[None, :]) / np.clip(Q.sum(0, keepdims=True), 1e-12, None)
        Q /= np.clip(Q.sum(1, keepdims=True), 1e-12, None)
    return (1 - strength) * np.log(P + 1e-12) + strength * np.log(Q + 1e-12)


def blend(comps, w):
    tot = sum(wi * c for wi, c in zip(w, comps))
    return log_softmax(tot / max(np.sum(np.abs(w) > 0), 1), axis=1)


def fit_weights(comps, y, l2=1e-3, restarts=3):
    """Softplus-parameterised positive weights minimising multiclass log-loss."""
    K = len(comps)

    def obj(theta):
        w = np.exp(theta)                       # positive, scale-free (blend is normalised)
        lp = blend(comps, w)
        ll = -lp[np.arange(len(y)), y].mean()
        return ll + l2 * float(np.sum(theta ** 2))

    best, best_theta = np.inf, None
    for r in range(restarts):
        theta0 = np.log(np.ones(K)) + (0.0 if r == 0 else np.random.default_rng(r).normal(0, 0.5, K))
        res = minimize(obj, theta0, method='L-BFGS-B', options=dict(maxiter=500))
        if res.fun < best:
            best, best_theta = res.fun, res.x
    return np.exp(best_theta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default='dump_dense/dump_dense')
    ap.add_argument('--eval_parquet', default='')
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--sinkhorn', type=float, default=0.45)
    ap.add_argument('--target_mix', type=float, default=0.35)
    ap.add_argument('--conf_thr', type=float, default=0.95,
                    help='drop eval rows where any member is more confident than this (contamination filter)')
    a = ap.parse_args()

    npz_path = os.path.join(a.dump, 'member_lp.npz')
    if not os.path.exists(npz_path):
        raise SystemExit(f'no dump at {npz_path} — the harness kernel has not finished')
    z = np.load(npz_path, allow_pickle=True)
    meta = json.load(open(os.path.join(a.dump, 'weights.json')))
    W = meta['weights']

    keys = [k[4:] for k in z.files if k.startswith('lp__')]
    y = z['y']
    prompt = z['prompt']
    Gte = z['Gte']
    print(f'{len(y)} rows, {len(np.unique(prompt))} prompts, {len(keys)} members, '
          f'{int((y >= 0).sum())} labelled')
    if (y < 0).all():
        raise SystemExit('dump has no labels (not a sim_pool run)')

    # ---------------- contamination check
    print('\n=== per-member balanced accuracy on the held-out pool ===')
    print('(a member far above the blend\'s 0.661 LB is in-sample and must be dropped)')
    rows = []
    for k in keys:
        v = z[f'lp__{k}']
        if not np.abs(v).any():
            print(f'  {k:14s} all-zero (skipped in this run)'); continue
        ba = balanced_accuracy_score(y, v.argmax(1))
        rows.append((k, ba))
    for k, ba in sorted(rows, key=lambda t: -t[1]):
        print(f'  {k:14s} {ba:.4f}')

    # ---------------- contamination filter
    # The adapters' training provenance is not published, so overlap with the eval pool cannot be
    # excluded by source. A member that memorised a row is near-certain on it, and a genuine
    # 0.66-level model is not: drop rows where any member exceeds `conf_thr` top-1 probability.
    keep = np.ones(len(y), bool)
    if a.conf_thr < 1.0:
        worst = np.zeros(len(y))
        for k in keys:
            v = z[f'lp__{k}']
            if not np.abs(v).any():
                continue
            p1 = np.exp(v - v.max(1, keepdims=True)); p1 = p1 / p1.sum(1, keepdims=True)
            worst = np.maximum(worst, p1.max(1))
        keep = worst <= a.conf_thr
        print(f'\ncontamination filter: dropping {int((~keep).sum())}/{len(y)} rows where a member is '
              f'>{a.conf_thr:.2f} confident; keeping {int(keep.sum())}')
        y_full, prompt_full = y, prompt
        y = y[keep]
        prompt = prompt[keep]
        Gte = Gte[keep]
        print(f'  class coverage after filter: {len(np.unique(y))}/18 classes, '
              f'{int(np.bincount(y, minlength=18).min())}-{int(np.bincount(y, minlength=18).max())} per class')

    comps, names, cur_w = [], [], []
    for k in keys:
        v = z[f'lp__{k}']
        w = float(W.get(k, 1.0))
        if not np.abs(v).any() or abs(w) < 1e-9:
            continue
        comps.append((v / w)[keep])             # unweight the member back to a pure log-prob
        names.append(k)
        cur_w.append(w)
    cur_w = np.array(cur_w)
    print(f'\n{len(comps)} usable components; current weights ' +
          ', '.join(f'{n}={w:g}' for n, w in zip(names, cur_w)))

    base_lp = blend(comps, cur_w)
    base_ba = balanced_accuracy_score(y, sinkhorn(base_lp, y, a.sinkhorn, a.target_mix).argmax(1))
    base_ba_nosk = balanced_accuracy_score(y, base_lp.argmax(1))
    print(f'\ncurrent weights: BA {base_ba:.4f} (sinkhorn {a.sinkhorn}) | {base_ba_nosk:.4f} (no sinkhorn)')

    # ---------------- grouped CV so the fitted gain is out-of-fold
    groups = np.unique(prompt)
    rng = np.random.default_rng(42)
    gf = {g: i % a.folds for i, g in enumerate(rng.permutation(groups))}
    fold = np.array([gf[p] for p in prompt])

    oof_fit = np.zeros(len(y), int)
    oof_cur = np.zeros(len(y), int)
    for f in range(a.folds):
        tr, va = fold != f, fold == f
        w = fit_weights([c[tr] for c in comps], y[tr])
        oof_fit[va] = blend([c[va] for c in comps], w).argmax(1)
        oof_cur[va] = base_lp[va].argmax(1)
    fit_ba = balanced_accuracy_score(y, oof_fit)
    print(f'fitted weights (out-of-fold, no sinkhorn): BA {fit_ba:.4f}')
    print(f'current weights (same rows):               BA {balanced_accuracy_score(y, oof_cur):.4f}')

    w_full = fit_weights(comps, y)
    print('\nfull-data fitted weights:')
    for n, w0, w1 in sorted(zip(names, cur_w, w_full), key=lambda t: -t[2]):
        print(f'  {n:14s} current {w0:7.3f} -> fitted {w1:7.3f}   ({w1 / max(w0, 1e-9):5.2f}x)')
    print(f'\nfull-data fit in-sample BA {balanced_accuracy_score(y, blend(comps, w_full).argmax(1)):.4f} '
          f'(optimistic; use the out-of-fold number above)')

    json.dump({'names': names, 'current': cur_w.tolist(), 'fitted': w_full.tolist(),
               'base_ba_sinkhorn': float(base_ba), 'base_ba': float(base_ba_nosk),
               'fitted_oof_ba': float(fit_ba),
               'current_oof_ba': float(balanced_accuracy_score(y, oof_cur))},
              open(os.path.join(a.dump, 'fit.json'), 'w'), indent=1)
    print('written', os.path.join(a.dump, 'fit.json'))


if __name__ == '__main__':
    main()
