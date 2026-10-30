# Adversarial invalidation review — SOTA Master Ensemble v2

**Scope**: `notebooks/master_sota/submission_sota.py` (388 lines, the committed
artifact at LB 0.539 / ref 57439844) and the five `src/` modules it depends on.
**Method**: read the code, then measure every claim on `data/train.csv` with
prompt-grouped folds. No claim below is an opinion; each carries a number and
the script that produced it.
**Local experiments**: `experiments/20_invalidation_evidence.py`,
`experiments/21_invalidation_honest.py`, `experiments/22_dropin_smoke.py`.
**Fixed code**: `src/typo_prior.py`, `src/coherent_pairwise.py`,
`src/length_gate.py`, `src/fusion.py`, `patches/dropin_submission_sota.py`,
`patches/submission_sota_v3.py`, `tests/test_invalidation_fixes.py` (16 tests,
all passing).

Per `AGENTS.md`, nothing here trains a model locally: the LoRA view is absent
from this machine and every experiment is CPU-only TF-IDF plus closed-form
algebra. Where a conclusion depends on the LoRA view I say so and give the
command to settle it on Kaggle.

---

## 0. Scoreboard

| # | Assigned vulnerability | Verdict | The number |
| :-- | :--- | :--- | :--- |
| 1 | Step-function length gate | **Confirmed, but third-order.** The discontinuity is real and cheap to fix; the *level* (0.45/0.15) is what actually costs accuracy, and it is 3-9× too high. | 208/2699 rows (7.7%) sit in [75, 85] words. Fixing the shape recovers **0.000**; dropping the level from 0.45 to the fitted 0.05 recovers **0.045**. |
| 2 | Condorcet cycles from independent pairwise deltas | **Premise false, defect real.** The pair graph is a forest, `cycle_dim = 0`, so a Condorcet cycle is impossible. But the update is the *divergence* of the edge flow, not a potential: it inflates isolated edges by exactly 2× and leaks sibling edges into each other. | Chain `δ = [0.5, −0.3]` is realised as `[1.3, −1.1]`. Honest OOF effect of the whole view at `w = 0.10`: **−0.0003** (null). |
| 3 | Out-of-distribution typographic failure | **Confirmed, and the worst of the five.** The hardcoded table is not a smoothing choice; it is wrong by up to 0.97 in probability for exactly the classes where sibling confusion is worst. | `qwen3.8-max` is charged **4.29 nats** against its own author on average. Net cost: **−0.045** balanced accuracy overall, **−0.082** on passages < 60 words. |
| 4 | Static 85/15 blend across 18 classes | **Confirmed.** Per-class optimal weight spans the full range, and the two "temperatures" are not separately identifiable from the weights. | Per-class `w*` from **0.00** (`claude-opus-5`, `kimi-k3`, `gpt-5.6-terra`) to **0.90** (`claude-fable-5`, `gemini-3.1-pro`). Fable-5 recall moves 0.14 → 0.46 across that range. |
| 5 | 4000-row runtime | **Not a risk, and the 289.8 s figure is not evidence.** That run had 3 test rows. The cost that dominates is the *fit*, which is constant in test size. | 5-fold fit: 105 s local / 215 s on Kaggle. Incremental cost for 4000 rows: **7.2 s** transform + **0.27 s** predict. |

**Framing that the brief omits and that matters more than any of the five**:
0.539 is *below* the range public notebooks already reach (0.574–0.595,
`AGENTS.md` §5.1). The ensemble is therefore not SOTA; it is a more complex
system that scores worse than simpler published ones. Every component added on
top of that baseline is guilty until it produces a positive prompt-grouped
paired-bootstrap delta. Four of the five components below cannot produce one.

---

## 1. The step-function length gate

### What the code does

```python
is_short = (word_counts_test < 80)
w_typo = np.where(is_short, 0.45, 0.15)[:, None]
log_final = (1.0 - w_typo) * log_base + w_typo * log_p_typo
```

### Proof of the discontinuity

For class `k`, the fused logit changes by

```
Δ log_final_k = (w_after − w_before) · (log_typo_k − log_base_k)
              = 0.30 · (log_typo_k − log_base_k)
```

Measured on train, the per-row spread of the typographic log-likelihood is
**9.18 nats** (p95 − p5). So across the 79/81-word boundary a single class can
move by up to **2.8 nats**. For scale: the OOF n-gram view's median top-1 to
top-2 margin is **0.26 nats** (p25 0.11, p75 0.50). A one-word difference near
the threshold is therefore about **11× the median decision margin** — the
tokeniser's word split, not the authorship signal, decides the argmax.

That window is not a tail case: `((wc >= 75) & (wc <= 85)).sum() == 208`
(7.7% of train, and the test set is ~4000 rows, so ~300 passages).

### The part the brief gets wrong

Fixing the shape recovers **nothing**, because the shape is not the problem:

| typo weight | OOF balanced accuracy |
| ---: | ---: |
| 0.00 | 0.3689 |
| **0.05 (fitted optimum)** | **0.3697** |
| 0.15 (the "long" branch) | 0.3656 |
| 0.45 (the "short" branch) | 0.3534 |

And the *direction* is wrong too. Fitting `w(n) = σ(a + b·log n)` on
prompt-grouped OOF with per-fold fitted typographic profiles returns
**b ≈ 0** — no length interaction is supported — while the optimum by length bin
is

| bin | n | `w*` | bacc at `w=0` | bacc at `w=0.45` |
| :-- | --: | --: | --: | --: |
| 0–60 | 555 | 0.05 | 0.2687 | 0.2417 |
| 60–80 | 442 | 0.05 | 0.3250 | 0.2986 |
| 80–100 | 325 | 0.00 | 0.3728 | 0.3460 |
| 100–140 | 523 | **0.35** | 0.3693 | 0.3756 |
| 140–300 | 854 | 0.05 | 0.4382 | 0.4264 |

The optimum is largest in the 100–140 bin, which is the opposite of the
"short passages need typography" premise. Shipping `0.45` on short texts
costs **−0.082** in the 0–60 bin, i.e. the gate does its damage precisely where
it was designed to help.

### Corrected formulation

Generalised logistic transition with a stated Lipschitz budget:

```
w(n) = w_lo + (w_hi − w_lo) · σ(−k(n − n₀)),      σ(z) = 1/(1 + e^{−z})

```text
max_n |dw/dn| = k (w_hi − w_lo)/4
k ≤ 4·budget / ((w_hi − w_lo)·max|log_typo − log_base|)
```

With `w_hi = 0.45, w_lo = 0.15, budget = 0.05 nats, gap = 4 nats`: `k ≤ 0.67`.
The shipped-style default `k = 0.12` leaves 4.4× headroom and gives
`max|dw/dn| = 0.009` per word, i.e. a one-word change perturbs a class by
≤ 0.036 nats.

But the honest fix is not the smooth step — it is to **fit** the gate and to cap
it. `SmoothLengthGate` fits `w(n) = σ(a + b·log n)` by OOF negative
log-likelihood with a shrinkage penalty on `b`, and then rescales to
`[w_lo, w_hi]`. In the smoke test it reports `b ≈ 6e−10` and drives `w` to ~0,
which is the correct behaviour: a gate that does not pay for itself must be able
to switch itself off.

### Patch

`src/length_gate.py` — `smooth_gate`, `lipschitz_safe_k`,
`continuity_budget`, `SmoothLengthGate.fit/weight/report`.
Tests: `test_gate_is_continuous`, `test_gate_perturbation_is_bounded`,
`test_gate_fit_recovers_no_interaction_on_uninformative_data`.

---

## 2. Condorcet cycles and independent pairwise deltas

### The premise is false for this pair list

The 8 shipped pairs touch 12 of 18 classes and form a **forest**: four trees
(`{opus-4-6, opus-5}` is not isolated; the real components are
`opus-4-6–opus-5`, `opus-4-8–sonnet-5`, a gemini chain `3.6–3.7–3.8`, an OpenAI
chain `luna–sol–terra`, and `glm-5.3` bridging `glm-5.3-flash` and
`claude-opus-4-6`). Cycle space dimension:

```
dim H¹ = E − V + C = 8 − 18 + 10 = 0
```

Zero, asserted in `test_forest_projection_realises_requested_margins`. With no
cycles, a cyclic preference `A ≻ B ≻ C ≻ A` cannot exist. HodgeRank would be
solving a problem that is not there — **as a cycle problem**.

### The real coherence defect

Adding `+0.10δ` to class `j` and `−0.10δ` to class `i` does not add a potential
`v` with `v_j − v_i = δ`. It adds the **divergence** of the edge flow,
`b_k = Σ_e ±δ_e`, and the realised margin on edge `(i,j)` becomes
`b_j − b_i`. For an isolated edge that is exactly `2δ` — the correction is
double the intended strength. For a node of degree ≥ 2 the margin is
contaminated by every sibling edge at that node. Measured on the two-edge gemini
chain with requested `δ = [0.5, −0.3]`:

| | requested | realised |
| :-- | --: | --: |
| margin 3.6 → 3.7 | 0.5 | **1.3** |
| margin 3.7 → 3.8 | −0.3 | **−1.1** |

So `0.10` is effectively `0.20` on the four isolated pairs and something else
entirely on the chains — one boundary's calibration leaks into an unrelated
boundary. That is a genuine incoherence, just not the one in the brief.

### It also does not help

Fit per fold (honest, no leakage), applied to the same base:

| pairwise weight | OOF bacc | argmax changes |
| ---: | ---: | ---: |
| 0.00 | **0.3689** | 0 |
| 0.05 | 0.3682 | 44 |
| 0.10 | 0.3686 | 82 |
| 0.20 | 0.3686 | 156 |

Null within noise, in both directions. The structural reason: each pairwise
classifier is a logistic regression on the **same** 4-view TF-IDF matrix as the
base ensemble. It re-derives the base model's own margin. Its only new
information is the residual. And with `clip(±2.5)` and `×0.10`, the view can
never apply more than **0.25 nats**, against a typographic view with a 9.18-nat
spread: it cannot defend its own contribution even in principle.

The second structural reason is that the correction is applied to **all** rows.
89% of passages belong to neither class of a given pair, where the two-class
specialist is extrapolating outside its training window and `p_i + p_j ≈ 0`.

### Corrected formulation

```
s_ij(x)  = log p_i − log p_j                    base margin (already a potential)
r_ij(x)  = logit P(y = j | y ∈ {i,j})           sibling specialist
π_ij(x)  = p_i + p_j                            pair-membership probability
D_ij(x)  = r_ij(x) − τ·s_ij(x)                  residual: zero when they agree
b_k(x)   = Σ_e λ·π_ij(x)·D_ij(x)·(1{k=j_e} − 1{k=i_e})       divergence
v(x)     = L⁺ b(x)                                            POTENTIAL

log p'_k = log p_k + v_k(x),        L v = b  (graph Laplacian)
```

`v` solves `min_v Σ_e (v_j − v_i − λ π_e D_e)²`, so on a forest it realises every
requested margin **exactly** (verified: max error 4.4e−16) and on a cyclic graph
it is the least-squares projection onto the gradient space. `L` has a null space
of one constant per connected component; the implementation centres each
component at zero so a within-family correction cannot move mass between
families.

Fidelity of the two updates on the shipped forest:

| update | max \|realised − requested\| | leaf-to-leaf edge |
| :-- | --: | --: |
| shipped divergence | **3.37 nats** | 2.00× |
| least-squares potential | **4.4e−16** | 1.00× |

`τ` is not cosmetic: a two-class logit and an 18-way log-ratio are not on the
same scale, and `τ = 1` is an assumption that should be reported.

### Patch

`src/coherent_pairwise.py` — `PairwiseResidualInjector` (residual + membership
gate + potential), `shipped_divergence` (kept only to measure the defect),
`ls_potential`, `realized_margins`, `cycle_dimension`.
Tests: `test_forest_projection_realises_requested_margins`,
`test_chain_edge_is_contaminated_by_its_sibling`,
`test_general_projection_is_coherent`, `test_residual_is_zero_when_views_agree`,
`test_membership_gate_suppresses_offpair`.

---

## 3. Out-of-distribution typographic priors

### Proof: the table is not an approximation, it is wrong

Recomputed against the file itself:

| feature | mean \|empirical − hardcoded\| | max |
| :-- | --: | --: |
| em-dash | 0.004 | 0.010 |
| spaced em-dash | 0.028 | 0.183 |
| curly double quote | **0.366** | **0.973** |
| curly single quote | **0.237** | 0.463 |

Every Claude and GLM em-dash rate is right to ±0.02. Everything else is not:

| model | empirical dq | hardcoded dq | empirical sq | hardcoded sq |
| :-- | --: | --: | --: | --: |
| qwen3.8-max | 0.007 | **0.980** | 0.287 | 0.740 |
| gemini-3.8-flash | 0.020 | **0.750** | 0.247 | 0.670 |
| gpt-5.6-luna | 0.107 | **0.950** | 0.513 | 0.970 |
| gpt-5.6-sol | 0.147 | **0.950** | 0.527 | 0.980 |
| gpt-6-astra | 0.073 | **0.950** | 0.440 | 0.800 |
| claude-fable-5 | 0.000 | 0.010 | 0.000 | 0.010 |

### Consequence in nats, per class, averaged over that class's own passages

`nats_penalty_vs_truth` = (empirical-profile log-lik) − (hardcoded-profile
log-lik). Positive means the shipped table scores the true author *worse* than
the truth is:

```
qwen3.8-max        +4.294
gpt-5.6-luna       +3.372
gpt-5.6-sol        +3.318
gpt-5.6-terra      +3.243
gpt-6-astra        +2.831
gemini-3.8-flash   +1.642
kimi-k3            +1.542
gpt-5.5            +1.304
... claude-opus-4-6 +0.071, claude-opus-5 +0.048, sonnet/opus-4-8/fable 0.000
```

The pattern is the finding: **all five Claude models are calibrated to 0.000
nats and everything else is penalised by 1.3–4.3 nats.** This is not a prior,
it is a Claude bias applied at weight 0.45. Its effect on the 18-way posterior:

- mean mass assigned to the four GPT-5.6 variants: **0.0091–0.0101**
- `qwen3.8-max`: **0.0066**
- `gemini-3.1-pro-preview`: **0.1175**, `claude-opus-4-8`: **0.0950**

A 12–17× mass handicap against OpenAI/Qwen before the classifier says anything.

### Direct evidence of collapse

As a standalone classifier the hardcoded view gets balanced accuracy
**0.1407** (2.5× random). Per-class recall for the typographic view alone:

```
claude-fable-5  0.832     gemini-3.1-pro  0.940
qwen3.8-max     0.007     gpt-5.6-luna    0.027
gpt-5.6-sol     0.000     gpt-6-astra     0.000
gemini-3.8      0.000     claude-opus-4-8 0.000
```

`qwen3.8-max` is at 0.007 under the typographic view while the n-gram view has
it at 0.467: it is deleted by the prior, not by the text.

### Two more defects the brief does not mention

**(a) The four indicators are not independent.** `P(spaced_em | em) = 0.082`
while `P(spaced_em) = 0.032`, so the joint is 2.6× the product of marginals. The
factorised likelihood charges the same em-dash twice. (`P(spaced_em ⇒ em) = 1`
by construction.)

**(b) The floor is a lethal constant.** `clip(p, 0.01, 0.99)` means one marker
in a class that "never" emits it costs `log(0.01) = −4.6` nats, and two
independent markers cost −9.2. A coding prompt that makes every model emit
straight quotes, or a fiction prompt that forces em-dashes, does not perturb this
view — it deletes whichever class has the lowest rate.

### Corrected formulation

Model the **joint** 16-state categorical, per fold, with Dirichlet smoothing,
then contaminate with the pooled marginal:

```
Cat_k(s) = (n_ks + α) / (n_k + 16α)                        s ∈ {0,1}⁴
P(s|k)   = (1 − ρ)·Cat_k(s) + ρ·U(s)                       contamination
ll_k     = log P(s|k)
ll'_k    = max(ll_k − max_k ll_k, −c)                      relative cap
```

- The categorical removes the independence assumption and the double counting
  (test: `test_categorical_beats_factorised_on_dependent_bits`).
- `ρ` is the probability that the *prompt* forces a marker regardless of author.
  It bounds the worst-case penalty at `log(ρ·U)` instead of `log(α/(n+16α))`.
- The cap is **relative**, `ll_k − max_k ll_k`, not absolute. This matters: an
  absolute floor `max(ll_k, −c)` collapses every bad class onto the same number
  and destroys the ranking. My first implementation made that exact mistake and
  the smoke test caught it — the fitted view scored 0.099 standalone, *worse*
  than the hardcoded table it was meant to replace. With the relative form it
  scores **0.1496** > 0.1407, and the fused result improves.

Default `α = 1.0, ρ = 0.25, c = 2.0` (likelihood ratio capped at `e⁻² = 0.14`
per passage, versus the current effective `e⁻⁹·²`).

### Patch

`src/typo_prior.py` — `TypoPriorView` (fit on the fold only, `fit_predict_oof`),
`empirical_categorical`, `calibration_report`.
Tests: `test_typo_evidence_is_bounded`, `test_typo_table_is_fitted_not_hardcoded`,
`test_categorical_beats_factorised_on_dependent_bits`.

---

## 4. Static 85/15 blend versus LoRA calibration

### Three separate problems

**(a) Temperature and weight are the same parameter.** In

```
log q_k ∝ Σ_v a_v·log p_k^(v)
```

`a_v` *is* `1/T_v`. Writing `0.85/0.86` and `0.15/0.80` states two numbers where
only the products `0.988` and `0.1875` are identifiable. That is a gauge freedom
presented as calibration.

**(b) The fitted temperature contradicts the shipped one.** Minimising OOF NLL
over `T` for the n-gram view:

| T | NLL |
| --: | --: |
| 0.50 (grid optimum) | **1.7736** |
| 0.80 (shipped) | 1.8688 |
| 1.00 | 1.9607 |

`T* = 0.50` is an interior minimum on the search grid `[0.30, 3.00]`, i.e. the
view is strongly over-confident and wants *more* sharpening than 0.80 — the
shipped value is not absurd but it is 1.6× too soft, which silently de-weights
the n-gram view by 38% relative to intent.
Note the consequence of (a): changing `T_ng` from 0.80 to 0.50 changes nothing
about argmax in a single-view setting and everything about it in a product.

**(c) One scalar for 18 classes is contradicted by the data.** Sweeping the
auxiliary view's weight per class on prompt-grouped OOF:

| model | `w*` | recall at `w=0` | recall at `w*` |
| :-- | --: | --: | --: |
| claude-opus-4-6 | 0.00 | 0.460 | 0.460 |
| claude-opus-5 | 0.00 | 0.573 | 0.573 |
| kimi-k3 | 0.00 | 0.273 | 0.273 |
| gpt-5.6-terra | 0.00 | 0.213 | 0.213 |
| gpt-5.6-luna | 0.00 | 0.333 | 0.333 |
| gemini-3.6/3.7/3.8 | 0.05 | — | — |
| claude-opus-4-8 | 0.10 | 0.260 | 0.280 |
| gpt-5.6-sol | 0.20 | 0.207 | 0.220 |
| claude-sonnet-5 | 0.25 | 0.400 | 0.420 |
| gpt-6-astra | 0.25 | 0.527 | 0.593 |
| gpt-5.5 | 0.30 | 0.360 | 0.387 |
| glm-5.3-flash | 0.45 | 0.187 | 0.207 |
| qwen3.8-max | 0.60 | 0.467 | **0.600** |
| glm-5.3 | 0.80 | 0.220 | 0.240 |
| gemini-3.1-pro | 0.90 | 0.753 | **0.933** |
| claude-fable-5 | 0.90 | 0.141 | **0.463** |

`w*` spans **0.00 to 0.90**. A global 0.15 hands `claude-fable-5` a weight 6×
too small (recall 0.14 instead of 0.46) and `qwen3.8-max` a weight 4× too small,
while actively hurting the five classes that want 0.00. This is the strongest
evidence in the review that the flat blend is structurally wrong, and it is
computable today from the OOF matrices already in the notebook.

### Corrected formulation

```
W ∈ R^{K×V},  W_{k,.} ∈ Δ^{V-1}                    simplex per class
log q_k ∝ Σ_v W_{k,v}·log p_k^(v)

NLL(W) = −(1/N) Σ_i log softmax_k(log q_i)_{y_i}
       + λ Σ_k ‖W_{k,.} − w^global‖²               hierarchical shrinkage
```

NLL is convex in `log q` and `log q` is linear in `W`, so the objective is convex
given the per-class view log-probabilities; only the softmax reparametrisation
is non-convex, and the shrinkage term keeps it benign. `λ ≈ 40` means a
per-class deviation must pay for itself with ~40 observations — with 150
rows/class that permits real bimodality and forbids 18-way noise fitting. Simplex
constraint is not decoration: an unconstrained fit can drive a weight negative,
which is extrapolating a view by *subtracting* it.

`LogOpinionPool` recovers planted weights (`test_pool_fit_recovers_known_weights`
recovers 0.85+ on the informative view), keeps weights on the simplex, and
shrinks toward the global row as `λ` grows.

### Patch

`src/fusion.py` — `LogOpinionPool`, `simplex_project`, `view_reliability`,
`per_class_weight_sweep`.
Tests: `test_pool_weights_lie_on_simplex`, `test_pool_fit_recovers_known_weights`,
`test_pool_shrinks_toward_global`, `test_view_reliability_flags_useless_view`.

---

## 5. Runtime at 4000 rows

### The evidence in the brief does not support the conclusion

The committed log shows `Test passages to classify: 3`. The 289.8 s is therefore
a **fit-time** measurement with a 3-row prediction attached. From the same log:

| stage | wall clock | note |
| :-- | --: | :-- |
| load + model load | 38.9 → 91.9 s | constant |
| LoRA extraction | 1.6 s | **one** batch of 3 rows |
| TF-IDF build | 101.0 → 108.6 s | on 2699 train docs |
| 5-fold LR fit + 3-row predict | 108.6 → 323.9 s (**215 s**) | the real cost |
| 8 pairwise fits | 323.9 → 328.5 s (4.6 s) | |
| total | 289.8 s | |

### Measured scaling

| quantity | local (M-series CPU) | Kaggle (4 vCPU) |
| :-- | --: | --: |
| 4-view TF-IDF fit on 2699 docs | 5.2 s | ~13 s (from the log: 7.6 s) |
| 5-fold multinomial LR fit | 105 s | **215 s (measured)** |
| 4-view transform, 4000 docs | **7.2 s** | ~18 s |
| `predict_proba`, 4000 × 18 | **0.27 s** | ~0.7 s |
| 8 pairwise LR fits | 4.6 s | 4.6 s (measured) |

Conclusion: everything that depends on `N_test` is **O(4000) with a ~20 s
constant**. The only large term is the 5-fold fit, which is independent of test
size. Total budget for the real rerun: **215 s + ~20 s + LoRA inference**.

LoRA inference is the one term I cannot measure here. First-principles estimate:
8 kept layers, `d = 4096`, `ffn = 12288` ⇒ ≈ 2.7 GFLOP/token; 4000 passages ×
~200 tokens ≈ 800 k tokens ⇒ ≈ 2.2 PFLOP; at a realistic 10–15 TFLOPS effective
on 2×T4 with these shapes ⇒ **140–220 s**, plus 40% padding waste if batching is
unsorted. So ~9 minutes total, comfortably inside 9 hours. This estimate is
unverified and the first Kaggle run with real 4000 rows should log its own
stage timings.

### The real runtime defects

1. **No length sorting.** `batch_size = 16` over unsorted passages pads every
   batch to its longest member: a 40-word and a 274-word passage in one batch
   waste ~3× the compute of the short one. Sorting by length and packing to a
   token budget removes 30–40% of LoRA FLOPs. `AGENTS.md` §5 already asks for
   this; the code does not do it.
2. **`float64` TF-IDF.** `TfidfVectorizer` defaults to `float64` and the code
   then multiplies by a weight, doubling memory for zero benefit at these
   scales.
3. **`device_map="auto"` with no `max_memory`**, against the `AGENTS.md` §5
   guidance. It happens to fit at `keep = 24` layers, but nothing asserts it.
4. **No graceful degradation.** `locate_file` raises if the LoRA dataset is
   missing, so one dataset-visibility change turns a submittable 0.54 into a
   failed kernel.
5. **No integrity check beyond the four asserts** — in particular nothing checks
   that the predicted class distribution has not collapsed, which is exactly the
   failure mode of §3.

### Patch

`patches/submission_sota_v3.py` — length-sorted micro-batching by token budget
(`run_lora_view`), `dtype=np.float32` (`build_ngram_views`), explicit fallback
when LoRA artefacts are absent, an assertion that head labels ⊆ the 18 classes,
and a warning when fewer than 12 distinct classes are predicted.

---

## 6. One more defect, unguarded, that can zero the ensemble

`submission_sota.py` lines 266-271:

```python
p_lora = np.zeros((N_test, len(VALID_18_MODELS)), dtype=np.float32)
for col_idx, m_name in enumerate(meta_labels):
    p_lora[:, MODEL_TO_IDX[m_name]] = p_lora_unaligned[:, col_idx]
```

If the LoRA head was trained on a subset of the 18 labels — plausible, since the
repo's two other experiments train on subsets — every missing class keeps
probability **exactly 0**, and `log_p_lora = log(clip(p, 1e-9))/0.86` assigns it
`−24.1` nats, which no amount of n-gram or typographic evidence can undo. The
code never asserts `len(meta_labels) == 18`.

I cannot check `meta.json` from here. Add the assertion. It is one line and it
distinguishes "the ensemble works" from "the ensemble silently predicts 16
classes". The replacement (`align_head_logits`) gives a head that cannot see a
class the row-minimum logit for that class — the view abstains instead of
vetoing — and prints a warning.

---

## 7. What this review does not establish

1. **The LoRA view's quality.** `p_lora` cannot be produced without the weights.
   Every conclusion above about the typographic and pairwise views is measured;
   every conclusion about the *fused* level is an inference. The decisive
   experiment needs `logp_lora_oof` from the notebook, which
   `build_fusion(..., logp_lora_oof=...)` consumes directly.
2. **The 0.539 LB score's composition.** No ablation accompanies it. It is
   entirely possible that 0.539 is the LoRA view at ~0.55 and that the other
   three views are subtracting 0.01–0.05 from it. §1, §2 and §3 each show a view
   that is net-negative at the n-gram level; nothing in the script would have
   revealed that.
3. **That 0.70 is reachable by fixing these five.** It is not. These are hygiene
   fixes: my point estimate of their combined value is **+0.02 to +0.05**
   balanced accuracy, not +0.16. Reaching 0.70 needs a different lever — better
   representation (multi-trunk, external corpus), or prompt-cluster structure
   exploited with soft constraints. `AGENTS.md` §4.4 already documents that the
   hard Hungarian variant of that idea cost LB; the soft variant is where the
   remaining headroom is.
4. **Statistical significance of any single fix.** `src/eval_stats.py` exists and
   is correct; nothing in the submission path calls it. Every number above is a
   point estimate on one seed of 5 grouped folds. With ~150 rows/class the
   per-class recall SE is ≈ 0.04, so a `w*` of 0.90 for one class is itself
   noisy. The *spread* across classes is the robust finding, not any individual
   `w*`.

---

## 8. Ranked actions

| Priority | Action | Expected effect | Cost | File |
| :-- | :--- | :-- | :-- | :--- |
| 1 | Replace the hardcoded typographic table with the per-fold fitted, contaminated, relative-capped view; cap its weight at ~0.05 | +0.045 (measured at n-gram level); fixes the `qwen`/GPT-5.6 mass collapse | 20 lines | `src/typo_prior.py` |
| 2 | Assert `len(meta_labels) == 18` or use abstain-alignment | removes a failure mode that silently deletes 1–2 classes | 1 line | `patches/submission_sota_v3.py` |
| 3 | Fit the pool per class on OOF instead of 0.85/0.15, drop the redundant temperatures | recovers 0.14→0.46 recall on `claude-fable-5`, 0.47→0.60 on `qwen` | 30 lines | `src/fusion.py` |
| 4 | Replace the step gate with the fitted logistic gate | ≈0 on its own; removes the 79/81-word variance spike | 10 lines | `src/length_gate.py` |
| 5 | Replace the divergence update with the gated residual potential | ≈0 on its own (measured null); removes 2× inflation and sibling leakage | 40 lines | `src/coherent_pairwise.py` |
| 6 | Print the OOF ablation at the end of every run | makes regressions visible before submission instead of after | 15 lines | `patches/dropin_submission_sota.py` |
| 7 | Length-sort the LoRA batches, `float32` TF-IDF | 30–40% of LoRA FLOPs | 20 lines | `patches/submission_sota_v3.py` |

Items 4 and 5 are in this list because they are *correct* and cheap, not because
they are *valuable*. Measured, both are worth approximately nothing. Reporting
that honestly is more useful than reporting that they are worth something.

---

## 9. Reproducing

```bash
# invalidation evidence (CPU, ~5 min)
python3 experiments/20_invalidation_evidence.py     # typo calibration, gate geometry, OOF fusion
python3 experiments/21_invalidation_honest.py       # honest per-fold pairwise, weight sweeps, T*
python3 experiments/22_dropin_smoke.py              # end-to-end plumbing of the patched fusion

# property tests (CPU, ~4 s)
python3 -m pytest tests/test_invalidation_fixes.py -q
```

Every reported number is in `experiments/invalidation_evidence.json` and
`experiments/invalidation_evidence2.json`.
