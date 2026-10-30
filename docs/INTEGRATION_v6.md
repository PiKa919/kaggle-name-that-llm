# Integration review v6 — sibling_features / test_time_calibration / short_passage_engine

**Scope**: the three modules as they stood at `md5 0694d5f` (sibling_features,
1955 lines), `25e011d` (test_time_calibration, 1058 lines), `4c5739f`
(short_passage_engine, 1487 lines), and
`notebooks/master_sota/submission_sota.py` as rewritten at 03:25.

**Method**: every number below is measured on `data/train.csv` with
`StratifiedGroupKFold(5, shuffle=True, random_state=42)` grouped on
`prompt_id`. Nothing is estimated from a paper. Scripts:
`experiments/23_integration_gate.py` … `29_global_vs_cluster_ot.py`; raw
output in `experiments/integration_gate*.json`. CPU only, per `AGENTS.md` §7.

**The LoRA view is absent from this machine**, so `p_semantic` is the 4-view
TF-IDF view (OOF 0.3686–0.3790). Conclusions about the *operators* transfer;
conclusions about the *final fused level* do not, and are marked as such.

---

## 0. Ledger

| Component | Δ balanced accuracy (OOF) | Verdict |
| :--- | :--- | :--- |
| Per-cluster Sinkhorn-Knopp OT, `reg=0.10, alpha=0.5`, recovered clusters | **+0.113** (95% CI +0.095…+0.131) | **Ship.** 18/18 class recalls improve. |
| … same operator, oracle `prompt_id` clusters | +0.118 | Upper bound; recovery costs 0.005. |
| Global uniform Sinkhorn (**what the notebook does now**) | **−0.002** balanced, −0.013 under a skewed mixture | **Remove.** See §3. |
| `sibling_features` in the 18-way base | +0.010 (95% CI −0.004…+0.026) | Unproven; cheap, keep behind a flag. |
| `sibling_features` in the 8 pairwise heads | **0.0000** (0.6525 → 0.6525) | Do not add. |
| `PairwiseResidualInjector` residual+gated potential, any λ>0 | −0.001 … −0.006 | Skip, or λ ≤ 0.05. |
| `short_passage_engine` gate, best of 12 configs | +0.004 at L0=120 (grid edge); −0.001 at sane L0 | **Do not ship.** |
| `short_passage_engine.entropy_regularized_short_gate` | **−0.007** | Do not ship. |
| `adaptive_neighborhood_centering` (any strength) | 0.4968 → ≤0.4953 | Do not ship. |

Balanced accuracy of the surrogate base: 0.3790. These deltas are *not*
additive and must not be added to 0.539.

---

## 1. Blocker: the review target moved

Both `src/*.py` files and the notebook were rewritten **during** this review:

| file | mtime during review | change |
| :--- | :--- | :--- |
| `src/sibling_features.py` | 02:34:38 | 2168 → 1955 lines; **101 → 93 features** |
| `src/test_time_calibration.py` | 02:34:48 | 1028 → 1058 lines |
| `src/short_passage_engine.py` | 02:51:23 | 1378 → 1487 lines |
| `notebooks/master_sota/submission_sota.py` | 03:25 | 467 → 850+ lines, new dual-trunk + own Sinkhorn |

The brief says "101 fine-grained stylometric features". The shipped file
returns **93** (`N_FEATURES == 93`, 10 pairs). The notebook header already says
93, so the brief is the stale artefact — but the mismatch is exactly the kind
of thing that silently breaks a re-run. Pin the module as a Kaggle dataset and
record its hash in the kernel before the next submission.

Snapshots used for every measurement here: `experiments/snapshot/`.

---

## 2. Where OT belongs: last

Measured, same base, same clusters:

| order | OOF bacc |
| :--- | ---: |
| base → potential → **OT** | **0.4957** |
| base → OT → potential | 0.4923 |

OT belongs **after** the pairwise/HodgeRank stage. Two reasons, one measured
and one structural:

1. A potential projection is per-row (it adds `v(x)` to each row) and OT is
   across rows (it moves mass between rows of a cluster). Applying the row
   operator after the column operator lets the row correction walk back part
   of the marginal constraint that OT just imposed — which is the 0.0034.
2. `src/test_time_calibration.py` already documents this ("transport is the
   last word because it is a projection"). The module is right; the question
   is answered by its own design intent *and* by the number.

Corollary: the length gate and the typographic prior must also run before OT,
because they are per-row. The current notebook does this correctly.

---

## 3. The notebook's current Sinkhorn is a different, worse operator

`submission_sota.py` §8 does one **global** scaling to a **global uniform**
class marginal at full strength:

```python
target = np.full(K, len(P) / K)
for _ in range(100):
    P *= target / np.maximum(P.sum(0), 1e-12)
    P /= np.maximum(P.sum(1, keepdims=True), 1e-12)
```

Three problems:

1. **It is unfalsifiable on train.** train.csv is exactly 150 prompts × 18
   models, so the global class marginal of *any* OOF matrix is uniform by
   construction. A global-uniform projection therefore cannot lose in CV.
   `AGENTS.md` §1.2 states the hidden test mixture is explicitly not equal and
   differs between public and private. The one assumption the operator makes is
   the one CV cannot test.
2. **It loses even where it should win.** Measured, balanced OOF regime:
   0.3790 → 0.3767. Under a class-skewed test mixture: 0.3778 → 0.3664.
3. **It does not achieve its own objective.** After scaling, predicted counts
   still span 0.043–0.070 per class, because argmax of a doubly-stochastic
   matrix is not uniform. It perturbs ranks without equalising recalls.

Per-cluster projection is falsifiable (clusters smaller than 18 make the
uniform column marginal a real assumption), and it wins in every regime:

| test mixture | raw | global α=1 (current) | global α=0.5 | **cluster α=0.5** |
| :--- | ---: | ---: | ---: | ---: |
| balanced | 0.3790 | 0.3767 | 0.3771 | **0.4968** |
| mild skew | 0.3782 | 0.3716 | 0.3746 | **0.4751** |
| strong skew | 0.3778 | 0.3664 | 0.3719 | **0.4573** |

---

## 4. Hyper-parameters

### `reg` (entropic regularisation) — 0.10

Flat plateau, measured at `alpha=0.5` on recovered clusters:

| reg | 0.02 | 0.05 | **0.10** | 0.20 | 0.50 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| bacc | 0.4808 | 0.4853 | **0.4916** | 0.4856 | 0.4719 |

Anything in 0.05–0.20 is defensible; 0.10 is the interior optimum and is what
the module's solver is tuned for.

### `alpha` (blend toward the projected plan) — 0.5

Measured on recovered clusters, `reg=0.10`:

| alpha | 0.2 | 0.35 | **0.50** | 0.65 | 0.80 | 1.0 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| bacc | — | 0.4920 | **0.4916** | — | — | 0.4871 |

and on 75%-subsampled clusters, `reg=0.10`:

| alpha | 0 | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.80 | 1.0 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bacc | .3834 | .4000 | .4137 | .4345 | .4557 | .4560 | .4585 | .4606 | .4573 | .4541 |

`alpha=1.0` is measurably worse than 0.5 in both regimes. Full projection
asserts that every cluster is a complete 18-model round; the hidden test is
subsampled, so a half-strength projection keeps ~96% of the gain and halves the
exposure to a wrong marginal. `DEFAULT_ALPHA = 0.5` in the module is correct —
it is the one default in these three files that the data endorses.

The gain is also robust to subsampling, which was the main risk
(`AGENTS.md` §4.4: hard Hungarian +0.10 CV / −0.009 LB):

| cluster retention ρ | 1.00 | 0.90 | 0.75 | 0.50 |
| :--- | ---: | ---: | ---: | ---: |
| OT gain | +0.118 | +0.103 | +0.087 | +0.070 |
| configs still positive | 25/25 | 25/25 | 25/25 | 25/25 |

What OT needs is *near-complete rounds*, not prompt identity:

| oracle neighbourhood size | 9 | 17 | 35+ |
| :--- | ---: | ---: | ---: |
| bacc | 0.4286 | 0.4968 | 0.4968 |

At 9 rows the gain halves. This is why the clusterer must produce rounds of
~18, not pure fragments of 3.

### Short-passage gate threshold — there is no valid one

The module's premise is that the short view wins *below* some `L0`. Tested per
length bin, it loses in **every** bin, including the one it was built for:

| words | n | base | short view | Δ |
| :--- | ---: | ---: | ---: | ---: |
| 0–60 | 555 | 0.2823 | 0.2141 | **−0.068** |
| 60–80 | 442 | 0.3595 | 0.2381 | −0.121 |
| 80–120 | 609 | 0.3588 | 0.2538 | −0.105 |
| 120–160 | 439 | 0.4419 | 0.3402 | −0.102 |
| 160–300 | 654 | 0.4167 | 0.3271 | −0.090 |

So no `L0` can help: the gate has nothing better to hand over anywhere. The
best of 12 `(L0, tau, w_max)` configurations is `L0=120, tau=10, w_max=0.35`
for **+0.0037** — and `L0=120` is the last point of the sweep, i.e. the optimum
is "apply a small constant weight nearly everywhere", which is not a length
gate. At the intended `L0=80` it is ≈0. `w_max ≤ 0.1` keeps it harmless (+0.001,
inside noise); anything above 0.2 loses.

`entropy_regularized_short_gate` is −0.0074. Do not ship it.

Two likely causes, both fixable but not worth fixing at +0.004: the model is
fitted on ≥80% truncated rows whose *rate* features are corrupted by the
truncation itself, and its OOF score (0.2786) is 0.10 below the base it is
asked to correct.

### Adaptive neighbourhood centering — 0.0

| operator | bacc |
| :--- | ---: |
| none | 0.4957 |
| `strength=0.1, k=8` | 0.4953 |
| oracle prompt-mean subtraction, `strength=0.3` | 0.4945 |

Subtracting the *true* prompt cluster mean is also negative, so this is not a
clustering-quality problem — the operator removes signal along with the topic
bias. `strength=0` is the optimum, which is the module's own identity case.

---

## 5. `sibling_features`: base, not pairwise

### Result

| where | measured |
| :--- | :--- |
| 18-way base, `sparse + 0.1·dense` | **0.3790** vs 0.3686 sparse → **+0.0103**, CI [−0.0037, +0.0263] |
| 8 pairwise heads, mean accuracy | 0.6525 → **0.6525** (5 pairs better, 3 worse, worst −0.047) |
| pairwise injector end-to-end, λ>0 | −0.001 … −0.006 |

Adding it to the pairwise heads buys exactly nothing on average, because each
pairwise head is a logistic regression on the *same* 4-view TF-IDF matrix as
the base: the dense block's information is already inside the head's input.
Put it where it is *not* redundant — the 18-way base — and skip the 8 extra
fits entirely.

The end-to-end pairwise result is the stronger argument: `PairwiseResidualInjector`
with the correct residual + membership gate + least-squares potential scores
**worse than λ=0 at every λ**, with margin fidelity 1.1e−16. The formulation is
right and the view is empty.

### Cost (measured)

| quantity | value |
| :--- | :--- |
| extraction, 2 699 docs | 4.1 s, single-threaded CPU |
| extraction, train+test (6 699) | ~10 s (extrapolated from 8.9 s / 4 000) |
| matrix | 93 × 4 000 float64 = **3.0 MB** |
| 5-fold LR fit, sparse only | 89 s local / 215 s Kaggle |
| 5-fold LR fit, sparse+dense | 199 s local (≈ +110 s) |
| GPU VRAM | **0 GB** — pure numpy/sklearn |

No OOM risk on Dual T4 at any batch size: the block never touches the GPU, and
the sparse matrix is ~5.8 M nnz. The cost is ~110 s of extra CPU in the fit,
not a latency regression on the 9-hour budget.

### Three footguns in the module

1. **Batch-level normalisation.** `extract_sibling_features` z-scores against
   the batch it is given. Calling it separately on train and test puts the two
   matrices on different scales, and there is no `fit`/`transform` pair to
   prevent it. Call it **once on `train + test` concatenated** and split. That
   is what `build_sibling_block` in the patch does.
2. **Routing is not total.** 1 of the notebook's 10 pairs —
   `("glm-5.3", "claude-opus-4-6")` — is not in `SIBLING_PAIRS` and raises
   `KeyError` from `get_pair_feature_indices`. Never call it unguarded.
3. **Routing is not the right selector anyway.** A fold-internal Cohen's-*d*
   ranking over the same 93 columns is leak-free, needs no routing table, and
   adapts to the data. Measured, static routing and fold-fit selection are
   within noise of each other; fold-fit has one fewer failure mode.

---

## 6. What `.70` requires

OT is the only component in this review with a large, significant,
all-classes-positive effect: +0.113 at CI [+0.095, +0.131], 18/18 recalls up,
worst class 0.173 → 0.247. It is a genuine find and it should ship.

It will not take 0.539 to 0.70, for two arithmetic reasons:

1. **The gain shrinks as the base improves.** Measured on the same clusters:
   base 0.3686 → +0.1189; base 0.3790 → +0.1126. Local slope ≈ −0.6 per unit
   of base. A linear extrapolation to 0.54 is invalid (it would cross zero),
   but the sign is unambiguous: OT repairs marginal imbalance, and a stronger
   base has less imbalance to repair. The +0.113 is an upper bound measured on
   the weakest plausible base.
2. **CV→LB transfer for this family is poor and documented.** `AGENTS.md` §4.4:
   the hard version of this idea was +0.10 CV and −0.009 LB. The soft version
   survives subsampling at ρ=0.5 here, which is the reason to expect better
   than the hard version — not a reason to expect CV numbers to print.

Defensible projection: **0.539 → 0.56–0.61**. That is consistent with the
public 0.574–0.595 cluster, which is the band this pipeline should be trying to
join, not 0.70.

0.70 needs a different lever, and it is the one `INVALIDATION_v2.md` §7 already
named: representation. Multi-trunk (8B + 14B) early-layer readouts, an external
multi-model corpus for stage-1 LoRA, or prompt-cluster structure exploited as a
*soft* constraint. No post-processing operator can manufacture 0.16 of
balanced accuracy from a 0.54 posterior; the ceiling of the operator is visible
in the per-class table — OT lifts every class by roughly the same 0.10, which
is a marginal fix, not a discrimination fix. Sibling confusion survives it:
`glm-5.3` 0.173 → 0.247 and `glm-5.3-flash` 0.247 → 0.260 are still the two
worst classes after projection.

---

## 7. Files

| file | role |
| :--- | :--- |
| `patches/v6_marginal_projection.py` | the drop-in: cluster recovery, OT, guards, optional sibling block |
| `experiments/23_integration_gate.py` | OT grid, pairwise routing, placement, λ/weight sweeps |
| `experiments/24_gate_subsample.py` | OT under cluster subsampling, alpha curve, gate sweep |
| `experiments/25_cluster_recovery.py` | short view per length bin; cluster recovery failure on stylometry |
| `experiments/26_prompt_recovery_content.py` | content clustering, kNN OT, oracle neighbourhood size |
| `experiments/27_end_to_end.py` | injector λ sweep, reg×alpha, per-class recall, bootstrap CI |
| `experiments/28_short_recheck.py` | re-run of the 02:51 short-engine rewrite |
| `experiments/29_global_vs_cluster_ot.py` | notebook's global OT vs per-cluster under mixture skew |
| `experiments/snapshot/` | pinned copies of the three modules |

Reproduce: `python3 experiments/27_end_to_end.py` (~50 s, after
`experiments/cache_oof.npz` exists) and `python3 experiments/29_global_vs_cluster_ot.py`.
