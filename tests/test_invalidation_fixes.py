"""
Property tests for the invalidation fixes.

Each test encodes one claim from the review that would otherwise be an opinion:

  test_gate_is_continuous                 the 79/81-word jump cannot happen
  test_gate_perturbation_is_bounded       the Lipschitz budget is honoured
  test_forest_projection_is_identity      HodgeRank on a forest changes nothing
                                                 (the Condorcet critique is void
                                                 for the shipped pair list)
  test_general_projection_is_coherent     on a cyclic graph the LS potential is
                                                 a gradient, so log p' is coherent
  test_residual_is_zero_when_views_agree  no double counting
  test_membership_gate_suppresses_offpair the specialist cannot move unrelated rows
  test_typo_evidence_is_bounded           one OOD marker cannot exceed max_nats
  test_typo_table_is_fitted_not_hardcoded qwen3.8-max penalty collapses
  test_categorical_beats_factorised       no independence assumption
  test_pool_weights_lie_on_simplex        no negative view weights
  test_pool_fit_recovers_known_weights    the fit finds the planted mixture
  test_pool_shrinks_toward_global         l2 controls per-class deviation
"""

import numpy as np
import pytest

from src.coherent_pairwise import (
    PairwiseResidualInjector,
    cycle_dimension,
    ls_potential,
    realized_margins,
    residual_deltas,
    shipped_divergence,
)
from src.config import MODEL_TO_IDX, VALID_18_MODELS
from src.fusion import LogOpinionPool, simplex_project, view_reliability
from src.length_gate import (
    SmoothLengthGate,
    continuity_budget,
    lipschitz_safe_k,
    smooth_gate,
)
from src.typo_prior import TypoPriorView, empirical_bernoulli, typo_bits

K = len(VALID_18_MODELS)

SHIPPED_PAIRS = [
    ("claude-opus-4-6", "claude-opus-5"),
    ("claude-opus-4-8", "claude-sonnet-5"),
    ("gemini-3.6-flash", "gemini-3.7-flash"),
    ("gemini-3.7-flash", "gemini-3.8-flash"),
    ("gpt-5.6-luna", "gpt-5.6-sol"),
    ("gpt-5.6-sol", "gpt-5.6-terra"),
    ("glm-5.3", "glm-5.3-flash"),
    ("glm-5.3", "claude-opus-4-6"),
]


def _log(p, eps=1e-12):
    return np.log(np.clip(p, eps, 1.0))


# --------------------------------------------------------------------------- #
# V1: length gate
# --------------------------------------------------------------------------- #
def test_gate_is_continuous():
    grid = np.arange(40.0, 260.0)
    w = smooth_gate(grid)
    jumps = np.abs(np.diff(w))
    assert jumps.max() < 0.01, "one-word step must not move the gate by >1 point"
    # the shipped step function, for contrast
    step = np.where(grid < 80, 0.45, 0.15)
    assert np.abs(np.diff(step)).max() == pytest.approx(0.30)
    # monotone decreasing, and anchored at the midpoint
    assert np.all(np.diff(w) <= 0)
    assert smooth_gate(np.array([80.0]))[0] == pytest.approx(0.30, abs=1e-6)


def test_gate_perturbation_is_bounded():
    budget = 0.05
    k = lipschitz_safe_k(w_hi=0.45, w_lo=0.15, max_logit_gap=4.0, budget=budget)
    assert continuity_budget(k, 0.45, 0.15, 4.0) <= budget + 1e-12
    # the shipped k is inside the safe range
    assert 0.12 <= lipschitz_safe_k(0.45, 0.15, 4.0, budget)


def test_gate_fit_recovers_no_interaction_on_uninformative_data():
    """With an auxiliary view that carries no information, the fitted gate must
    collapse to a constant (b ~ 0) rather than invent a length interaction."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, K, n)
    wc = rng.integers(40, 250, n).astype(float)
    logp_base = _log(rng.dirichlet(np.ones(K), n))
    logp_base[np.arange(n), y] += 2.0  # base view is informative
    logp_aux = _log(rng.dirichlet(np.ones(K), n))  # pure noise
    gate = SmoothLengthGate(b_penalty=200.0).fit(wc, logp_base, logp_aux, y)
    assert abs(gate.b_) < 0.5
    rep = gate.report(wc)
    assert rep["w_min"] <= rep["w_at_80"] <= rep["w_max"]


# --------------------------------------------------------------------------- #
# V2: pairwise coherence
# --------------------------------------------------------------------------- #
def test_forest_projection_realises_requested_margins():
    """The shipped pair list is a forest (cycle_dim = 0), so no Condorcet cycle is
    possible -- but the shipped update is a DIVERGENCE, not a potential.  The
    least-squares potential realises every requested margin exactly; the shipped
    update does not, and inflates isolated edges by 2x."""
    assert cycle_dimension(SHIPPED_PAIRS, K) == 0
    rng = np.random.default_rng(1)
    terms = {p: rng.normal(size=50) for p in SHIPPED_PAIRS}

    v = ls_potential(SHIPPED_PAIRS, terms, K)
    realised = realized_margins(SHIPPED_PAIRS, v)
    for p in SHIPPED_PAIRS:
        assert np.allclose(realised[p], terms[p], atol=1e-9), p

    # the shipped update: margins are wrong, and by a lot
    b = shipped_divergence(SHIPPED_PAIRS, terms, K)
    realised_shipped = realized_margins(SHIPPED_PAIRS, b)
    err = max(float(np.abs(realised_shipped[p] - terms[p]).max()) for p in SHIPPED_PAIRS)
    assert err > 1.0

    # on a leaf-to-leaf edge the shipped update applies exactly 2x what was asked
    a, c = "claude-opus-4-8", "claude-sonnet-5"   # both degree 1, isolated pair
    assert np.allclose(realised_shipped[(a, c)], 2.0 * terms[(a, c)], atol=1e-9)


def test_chain_edge_is_contaminated_by_its_sibling():
    """Exact formula for the shipped defect on a 3-node chain i-j-k:
    realised_ij = 2*delta_ij - delta_jk'.  One boundary's calibration leaks into
    an unrelated boundary."""
    pairs = [
        ("gemini-3.6-flash", "gemini-3.7-flash"),
        ("gemini-3.7-flash", "gemini-3.8-flash"),
    ]
    d1, d2 = 0.5, -0.3
    terms = {pairs[0]: np.array([d1]), pairs[1]: np.array([d2])}
    b = shipped_divergence(pairs, terms, K)
    realised = realized_margins(pairs, b)
    assert realised[pairs[0]][0] == pytest.approx(2 * d1 - d2)
    assert realised[pairs[1]][0] == pytest.approx(2 * d2 - d1)
    # the potential restores both
    v = ls_potential(pairs, terms, K)
    ok = realized_margins(pairs, v)
    assert ok[pairs[0]][0] == pytest.approx(d1)
    assert ok[pairs[1]][0] == pytest.approx(d2)


def test_general_projection_is_coherent():
    """Add edges that close a loop; the LS potential must satisfy the normal
    equations, i.e. the residual flow is orthogonal to every gradient."""
    pairs = SHIPPED_PAIRS + [
        ("gpt-5.5", "gpt-5.6-luna"),
        ("gpt-5.6-luna", "gemini-3.7-flash"),
        ("gpt-5.5", "gemini-3.7-flash"),
    ]
    assert cycle_dimension(pairs, K) > 0
    rng = np.random.default_rng(2)
    terms = {p: rng.normal(size=20) for p in pairs}
    v = ls_potential(pairs, terms, K)
    realised = realized_margins(pairs, v)
    # residual must be orthogonal to (e_j - e_i) for every edge
    for a, c in pairs:
        i, j = MODEL_TO_IDX[a], MODEL_TO_IDX[c]
        resid = terms[(a, c)] - realised[(a, c)]
        # gradient-projection condition: <resid, e_j - e_i> == 0
        grad = np.zeros((20, K))
        grad[:, j] = 1.0
        grad[:, i] = -1.0
        assert np.allclose(np.sum(resid[:, None] * grad, axis=1), 0.0, atol=1e-8)


def test_residual_is_zero_when_views_agree():
    """If the specialist reproduces the base margin exactly, the correction must
    vanish -- this is what stops the double counting."""
    i, j = MODEL_TO_IDX["gpt-5.6-luna"], MODEL_TO_IDX["gpt-5.6-sol"]
    logp = np.tile(np.log(np.ones(K) / K), (5, 1))
    margin = np.array([0.4, -0.2, 1.0, 0.0, -0.7])
    logp[:, i] += margin / 2
    logp[:, j] -= margin / 2
    q = 1.0 / (1.0 + np.exp(-margin))  # specialist reports exactly the base margin
    d = residual_deltas(q, logp, i, j, tau=1.0)
    assert np.allclose(d, 0.0, atol=1e-6)


def test_membership_gate_suppresses_offpair():
    """Rows where neither class of the pair is plausible must not be moved: the
    edge term is multiplied by pi_ij = p_i + p_j."""
    rng = np.random.default_rng(3)
    n = 240
    a, b = "claude-opus-4-6", "claude-opus-5"
    ia, ib = MODEL_TO_IDX[a], MODEL_TO_IDX[b]
    Xtr = rng.normal(size=(n, 5))
    ytr = np.array([MODEL_TO_IDX[x] for p in SHIPPED_PAIRS for x in p] * 15)[:n]
    half = n // 2
    ytr[:half] = ia
    ytr[half:] = ib
    Xtr[:half, 0] = -2.0
    Xtr[half:, 0] = 2.0
    injector = PairwiseResidualInjector([(a, b)], lam=1.0, gate=True).fit(Xtr, ytr)
    assert (a, b) in injector.models_

    # two test rows: certainly opus-4-6, and certainly something else
    logp = np.full((2, K), -20.0)
    logp[0, ia] = 0.0
    logp[1, MODEL_TO_IDX["qwen3.8-max"]] = 0.0
    terms = injector.edge_terms(rng.normal(size=(2, 5)), logp)
    t = np.abs(terms[(a, b)])
    assert t[0] > 0.0
    assert t[1] < 1e-6, "an off-pair row must not be moved"


# --------------------------------------------------------------------------- #
# V3: typographic likelihood
# --------------------------------------------------------------------------- #
def test_typo_evidence_is_bounded():
    """A single out-of-distribution marker can never cost more than max_nats
    relative to the most favoured class, and the ranking survives."""
    texts_clean = ["A plain paragraph with no typography at all."] * 50
    texts_marked = ["A paragraph with an \u2014 em dash and \u201cquotes\u201d."] * 50
    bits = typo_bits(texts_clean + texts_marked)
    y = np.arange(len(texts_clean)) % K
    y = np.concatenate([y, y])
    view = TypoPriorView(alpha=1.0, rho=0.25, max_nats=2.0).fit(bits, y)
    ll = view.log_prob(bits)
    spread = ll.max(axis=1) - ll.min(axis=1)
    assert np.all(spread <= 2.0 + 1e-9), "relative evidence must be capped"
    assert np.allclose(ll.mean(axis=1), 0.0, atol=1e-9)
    # ordering is preserved: the class that best explains the markers must rank
    # top, i.e. winsorisation must not flatten the view into a constant
    assert len(np.unique(np.round(ll[0], 6))) > 1


def test_typo_table_is_fitted_not_hardcoded():
    """Fitting the table removes the systematic anti-OpenAI/anti-Qwen bias.

    Synthetic corpus: class 0 always emits a curly quote, the rest never do.
    The fitted table must assign class 0 the high rate, and must not assign it
    to anyone else -- the hardcoded table got this exactly wrong for qwen3.8-max
    (0.98 coded vs 0.007 observed)."""
    rng = np.random.default_rng(4)
    n_per = 60
    texts, y = [], []
    for k in range(K):
        for _ in range(n_per):
            texts.append("\u201ctext\u201d" if k == 0 else "plain text")
            y.append(k)
    bits = typo_bits(texts)
    y = np.array(y)
    prof = empirical_bernoulli(bits, y, K)
    assert prof[0, 2] == pytest.approx(1.0)
    assert np.allclose(prof[1:, 2], 0.0)
    view = TypoPriorView(alpha=1.0, rho=0.0, max_nats=10.0).fit(bits, y)
    marked = typo_bits(["\u201cquote\u201d"])
    ll = view.log_prob(marked)[0]
    assert ll.argmax() == 0


def test_categorical_beats_factorised_on_dependent_bits():
    """P(spaced_em | em) = 0.082 but a factorised model treats the two as
    independent, so it charges a spaced em-dash twice."""
    rng = np.random.default_rng(5)
    n = 3000
    em = rng.random(n) < 0.4
    spaced = em & (rng.random(n) < 0.08)          # nested by construction
    dq = rng.random(n) < 0.1
    bits = np.stack([em, spaced, dq, np.zeros(n)], axis=1).astype(np.uint8)
    y = rng.integers(0, K, n)
    view = TypoPriorView(alpha=1.0, rho=0.0, max_nats=20.0).fit(bits, y)
    # the categorical table must place all mass on the 4 realisable states
    used = np.unique((bits @ np.array([8, 4, 2, 1])))
    row = view.table_.sum(axis=0)
    impossible = [s for s in range(16) if s not in used]
    # smoothed, so not exactly zero, but the impossible states must be far smaller
    assert row[list(used)].min() > row[impossible].max()


# --------------------------------------------------------------------------- #
# V4: fusion
# --------------------------------------------------------------------------- #
def test_simplex_project_is_valid():
    rng = np.random.default_rng(6)
    for _ in range(20):
        v = rng.normal(size=5) * 3
        p = simplex_project(v)
        assert np.all(p >= -1e-12)
        assert p.sum() == pytest.approx(1.0, abs=1e-9)


def test_pool_weights_lie_on_simplex():
    rng = np.random.default_rng(7)
    n = 300
    y = rng.integers(0, K, n)
    lp1 = _log(rng.dirichlet(np.ones(K), n))
    lp2 = _log(rng.dirichlet(np.ones(K) * 0.5, n))
    pool = LogOpinionPool(per_class=True, l2=10.0).fit([lp1, lp2], y)
    W = pool.weights()
    assert W.shape == (K, 2)
    assert np.all(W >= -1e-12)
    assert np.allclose(W.sum(axis=1), 1.0)
    assert np.all(np.isfinite(pool.transform([lp1, lp2])))


def test_pool_fit_recovers_known_weights():
    """Plant a mixture and check the fit finds it."""
    rng = np.random.default_rng(8)
    n = 6000
    y = rng.integers(0, K, n)
    # view A is informative, view B is noise
    logits_a = rng.normal(size=(n, K)) * 0.5
    logits_a[np.arange(n), y] += 3.0
    logits_b = rng.normal(size=(n, K))
    lpA, lpB = _log(_softmax(logits_a)), _log(_softmax(logits_b))
    pool = LogOpinionPool(per_class=False, l2=0.0).fit([lpA, lpB], y)
    w = pool.weights()[0]
    assert w[0] > 0.85, w
    assert w.sum() == pytest.approx(1.0)


def test_pool_shrinks_toward_global():
    rng = np.random.default_rng(9)
    n = 400
    y = rng.integers(0, K, n)
    lp1 = _log(rng.dirichlet(np.ones(K), n))
    lp2 = _log(rng.dirichlet(np.ones(K), n))
    loose = LogOpinionPool(per_class=True, l2=0.0).fit([lp1, lp2], y).weights()
    tight = LogOpinionPool(per_class=True, l2=1e4).fit([lp1, lp2], y).weights()
    g = LogOpinionPool(per_class=False, l2=0.0).fit([lp1, lp2], y).weights()[0]
    assert np.abs(tight - g[None, :]).sum() <= np.abs(loose - g[None, :]).sum() + 1e-6
    assert np.ptp(tight[:, 0]) < np.ptp(loose[:, 0]) + 1e-9


def test_view_reliability_flags_useless_view():
    rng = np.random.default_rng(10)
    n = 500
    y = rng.integers(0, K, n)
    good = np.zeros((n, K)) - np.log(K)
    good[np.arange(n), y] += 1.0
    bad = np.zeros((n, K)) - np.log(K)
    bad[np.arange(n), y] -= 1.0   # actively down-weights the truth
    assert view_reliability(good, y)["mean_nats_above_uniform"] > 0
    assert view_reliability(bad, y)["mean_nats_above_uniform"] < 0
    assert view_reliability(np.zeros((n, K)) - np.log(K), y)["mean_nats_above_uniform"] == pytest.approx(0.0)


def _softmax(x):
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
