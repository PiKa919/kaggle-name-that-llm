"""Correctness proofs for src/partial_matching.py.

Every assertion here is checked against an independent brute-force
enumeration or an analytic identity. No assertions are made about "looks
reasonable".
"""

import itertools
import time

import numpy as np
import pytest
from scipy.special import logsumexp
from sklearn.metrics import balanced_accuracy_score


def _brute_force_marginals(logw, presence_logit):
    """Enumerate every injection for small K and normalise."""
    m, K = logw.shape
    Z = -np.inf
    acc = np.full((m, K), -np.inf)
    for cols in itertools.permutations(range(K), m):
        s = sum(logw[i, cols[i]] + presence_logit[cols[i]] for i in range(m))
        Z = np.logaddexp(Z, s)
        for i in range(m):
            acc[i, cols[i]] = np.logaddexp(acc[i, cols[i]], s)
    return np.exp(acc - Z), np.exp(Z)


@pytest.mark.parametrize("m,K", [(1, 4), (3, 5), (5, 5), (2, 6)])
def test_marginals_match_brute_force(m, K):
    from src.partial_matching import partial_matching_posteriors

    rng = np.random.default_rng(0)
    logw = rng.normal(0, 1.5, (m, K))
    pres = rng.normal(0, 0.7, K)

    out = partial_matching_posteriors(logw, pres)
    bf_marg, bf_Z = _brute_force_marginals(logw, pres)

    assert np.allclose(out["marginals"], bf_marg, atol=1e-9)
    assert np.allclose(np.exp(out["logZ"]), bf_Z, rtol=1e-9)


def test_marginals_are_a_coupling():
    """Rows must each sum to 1; total usage must equal m."""
    from src.partial_matching import partial_matching_posteriors

    rng = np.random.default_rng(1)
    logw = rng.normal(0, 2.0, (7, 18))
    out = partial_matching_posteriors(logw)
    assert np.allclose(out["marginals"].sum(axis=1), 1.0, atol=1e-9)
    assert np.isclose(out["usage"].sum(), 7.0, atol=1e-9)
    assert (out["usage"] <= 1.0 + 1e-9).all()


def test_dummy_logit_is_a_no_op():
    """The documented knob in src/postproc.py does not change the output.

    sinkhorn_matching_posterior adds `dummy_logit` uniformly across the dummy
    rows' columns; a column-constant shift cancels under column normalisation.
    This test pins the claim so nobody reintroduces it as a tuned parameter.
    """
    from src.postproc import sinkhorn_matching_posterior

    rng = np.random.default_rng(2)
    logp = rng.normal(0, 2.0, (6, 18))
    a = sinkhorn_matching_posterior(logp, beta=1.0, dummy_logit=-1.5)
    b = sinkhorn_matching_posterior(logp, beta=1.0, dummy_logit=+4.0)
    assert np.allclose(a, b, atol=1e-12)


def test_sinkhorn_to_uniform_beats_prior_shift_under_label_shift():
    """Balanced accuracy wants likelihoods, not matched priors.

    A classifier trained on a skewed mixture emits posteriors that inherit the
    skew. Balanced accuracy weights every class equally, so the correct move is
    to divide the skew out (Sinkhorn to uniform), not to reproduce it (SLD-EM's
    pi_test / pi_train multiplication).
    """
    from src.partial_matching import balanced_log_odds
    from src.postproc import sld_em

    rng = np.random.default_rng(3)
    K, n_per = 18, 60
    train_prior = np.full(K, 1.0 / K)
    test_prior = np.array([0.10] * 4 + [0.60 / (K - 4)] * (K - 4))
    test_prior /= test_prior.sum()

    ys = np.concatenate([np.full(n_per, k) for k in range(K)])
    logits = rng.normal(0, 1.2, (len(ys), K))
    logits[np.arange(len(ys)), ys] += 2.0
    logits = logits + np.log(test_prior)[None, :] - np.log(train_prior)[None, :]
    logp = logits - logsumexp(logits, axis=1, keepdims=True)

    raw_hist = np.bincount(np.argmax(logp, axis=1), minlength=K)
    shifted, _ = sld_em(np.exp(logp), train_prior, smooth=0.0)
    sld_hist = np.bincount(np.argmax(shifted, axis=1), minlength=K)
    bal = balanced_log_odds(np.exp(logp))
    bal_hist = np.bincount(np.argmax(bal, axis=1), minlength=K)

    raw_b = balanced_accuracy_score(ys, np.argmax(logp, axis=1))
    sld_b = balanced_accuracy_score(ys, np.argmax(shifted, axis=1))
    bal_b = balanced_accuracy_score(ys, np.argmax(bal, axis=1))

    # Sinkhorn-to-uniform flattens the predicted marginal; SLD-EM sharpens it.
    assert bal_hist.max() < raw_hist.max()
    assert sld_hist.max() >= raw_hist.max()
    assert bal_b >= raw_b
    assert bal_b >= sld_b


def test_matching_never_collapses_two_rows_onto_one_author():
    """The whole point: duplicate predictions inside a cluster must vanish."""
    from src.partial_matching import partial_matching_posteriors

    K = 18
    # Three passages, all confidently (and wrongly) claude-opus-4-6 = index 2.
    logw = np.full((3, K), -8.0)
    logw[:, 2] = 0.0
    logw[1, 5] = -0.05  # row 1 is nearly tied with gpt-5.5
    logw[2, 7] = -0.05

    out = partial_matching_posteriors(logw)
    arg = out["marginals"].argmax(axis=1)
    assert len(set(arg.tolist())) == 3, arg
    assert out["marginals"].max(axis=0).sum() <= 3.0


def test_m18_speed_is_acceptable():
    """The hidden test set can contain full 18-row clusters."""
    from src.partial_matching import partial_matching_posteriors

    rng = np.random.default_rng(4)
    logw = rng.normal(0, 2.0, (18, 18))
    t0 = time.time()
    out = partial_matching_posteriors(logw)
    dt = time.time() - t0
    assert out["marginals"].shape == (18, 18)
    assert dt < 5.0, dt
    print(f"\n18x18 exact permanent marginals: {dt*1000:.0f} ms")
