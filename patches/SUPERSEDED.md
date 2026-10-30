# Why these five patches are superseded

They were produced by an earlier pass of the same review. Each contains a real
defect, and none of them is imported by the shipped code. They are kept here only
so that the mistakes are on record instead of being rediscovered.

| file | defect |
| :-- | :--- |
| `v1_length_gate.py` | Functionally correct but its docstring contradicts its signature (`k=0.15` vs "k ≈ 0.12"), and it treats the gate as a shape problem. Measured, the shape is worth 0.000; the *level* is worth 0.045. It also leaves the weight unfitted, which is the actual bug. Replaced by `src/length_gate.py`. |
| `v2_condorcet_cycles.py` | Wrong problem. The shipped pair graph is a forest (`cycle_dim = 0`), so no Condorcet cycle exists and HodgeRank is not needed. Its implementation is also wrong: `b[i2] += delta; b[i1] -= delta` accumulates the *divergence*, then `lstsq` on a Laplacian whose pinning row is hardcoded to node 0 even though node 0 (`claude-fable-5`) is isolated in this pair list. Replaced by `src/coherent_pairwise.py`. |
| `v3_typo_robustness.py` | Not a probability model. `dir_alpha*obs*log(p) + (dir_alpha-1)*(1-obs)*log(1-p)` drops the `(1-obs)*log(1-p)` mass entirely for α = 2, so absence of a marker carries no evidence and the score becomes "bonus for present markers", drifting toward whichever class emits them most. It also imports `src.typo_profiles`, which does not exist, and re-uses the miscalibrated hardcoded table. Replaced by `src/typo_prior.py`. |
| `v4_dynamic_blend.py` | Produces invalid mixtures: `w = 0.5 + 0.5*σ_ng/σ_lora` exceeds 1 whenever `σ_ng > σ_lora`, and `1 - w` then goes negative — extrapolating a view by subtracting it. `w = w/w.sum()*K` does not repair that. Replaced by the simplex-constrained `src/fusion.py`. |
| `v5_runtime.py` | The runtime analysis is wrong in the safe direction and the code is wrong in the unsafe one: "2699 rows × 5 folds = 13495 LR fits" (it is 5 fits), and `p_test += clf.predict_proba(X_trn)/5` predicts on **train**, not test. Replaced by the measured breakdown in `docs/INVALIDATION_v2.md` §5 and the batching in `patches/submission_sota_v3.py`. |
| `CRITIQUE.md` | Claims the length gate's discontinuity is the dominant defect (measured: it is worth 0.000; the level is worth 0.045), and claims HodgeRank projection "changes nothing" on a forest (also wrong: the shipped update is a divergence, so projection changes the applied margin from 1.3 to 0.5 nats on the gemini chain). Superseded by `docs/INVALIDATION_v2.md`. |

The current patches are:

- `src/typo_prior.py` — fitted 16-state categorical, contamination mixture, relative evidence cap
- `src/coherent_pairwise.py` — gated residual injected as a least-squares potential
- `src/length_gate.py` — Lipschitz-bounded, OOF-fitted length gate
- `src/fusion.py` — per-class simplex log-opinion pool
- `patches/dropin_submission_sota.py` — the fusion block, with an OOF self-audit
- `patches/submission_sota_v3.py` — the whole submission script, rewired
- `tests/test_invalidation_fixes.py` — 16 property tests over all of the above
