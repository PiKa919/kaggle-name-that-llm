# C1 — Adversarial critique and v10 recommendation

**Scope:** adjudicate contradictions across W1–W5, verify the critical artifact
dependencies, assess T1–T10, and choose one submission configuration.

**Evidence rule:** measured leaderboard > prompt-grouped OOF > proxy transfer >
literature. A public-LB delta below roughly 0.010 is inside sampling noise; at a
40% public split of ~4,000 rows, 0.004 is about 6–7 public rows.

**Hard constraints:** no local training or heavy inference; Kaggle dual T4
(FP16, no BF16), 9-hour cap, offline scoring; 5 submissions/day, 2 final
submissions, private LB is 60% of the score.

---

## 1. Executive verdict

The reports are directionally right, but several causal claims are stronger than
their evidence.

**Accept**

- v5 (`57471060`) at **0.600** is the best measured submission and the correct
  anchor. v6/v8 scored 0.593 and v9 scored 0.591; the current 3,506-line
  `submission_sota.py` is a v7 hybrid whose final pipeline is v5's global
  Sinkhorn. It is not the 0.600 artifact.
- Joseph Adamski's ATLAS at a 10% pre-Sinkhorn blend is the best single change:
  **+0.007** on the exact 0.600 base, **7/7** positive seeds, range +0.002…+0.007.
- Soft prompt-group Gaussian readout is real (+0.0100 OOF, 68/68 paired cells),
  but its LB effect is small (~+0.005).
- Global uniform Sinkhorn is an unjustified operator: it is unfalsifiable on
  balanced train, loses on simulated skew, and does not equalize argmax counts.
  It is still part of a known 0.600 recipe, so it should be treated as legacy,
  not as a proven optimization.
- Hard prompt reconstruction/Hungarian assignment is a trap. Three independent
  public teams lost LB despite +0.08–0.12 OOF; our own Hungarian lost
  **0.353 → 0.344** while inflating CV.
- The W3 corpus upgrade is the best medium-term data lead, but it is a
  linear-SVM proxy, not a LoRA or LB result, and its provenance is not clean
  enough to ship before the deadline.

**Reject for v10**

- The current v7/v9 triple-trunk + clique-calibration artifact.
- Per-cluster OT as a production change. Its +0.113 is on a 0.379 TF-IDF
  surrogate, is not additive to 0.600, and its measured source module has been
  replaced by a 39-line stub.
- Static 85/15 blending, global family gates, hard em-dash vetoes, 27B NF4
  inference, and a corpus retrain before the deadline.

**Recommendation**

Ship **exact v5 + ATLAS 10% before the inherited global Sinkhorn**, plus an
optional exact spaced-em-dash guard. Expected public score **0.602–0.607**;
top-200 (cutoff ≈0.604) is borderline, private-LB transfer is uncertain. Fallback
is exact v5 at 0.600.

---

## 2. Contradiction rulings

### 2.1 Public/private split: 20–30% vs 40/60

W1 and W3 say the public LB is 800–1,200 rows (20–30%). W2 quotes the official
Leaderboard page as **40% public / 60% private**. W2 is the primary source; use
40/60, i.e. ~1,600 public and ~2,400 private rows. W1's "68/68 cells" is a
paired OOF/Fisher-style claim, not a leaderboard sample; it does not rescue the
stale split estimate. Consequence: all LB deltas below 0.01, including most of
the T1–T9 catalog, need a controlled A/B before being called real.

### 2.2 Prompt grouping: soft readout vs hard assignment

These are different techniques and must not be bundled.

- The shared-prompt **Gaussian head** is a soft readout over features. It gained
  +0.0100 [0.0026, 0.0172] on 2/3-passage fragments and +0.0408 complete-prompt
  accuracy, and beat the centred head in 68/68 cells. W5's independent read is
  only ~+0.005 LB. Accept as an optional low-weight member.
- **Hard grouping** (reconstruct prompts, then force one model per prompt, or
  run Hungarian/injective assignment) is the failure mode. W5 documents three
  teams losing LB despite +0.08–0.12 OOF. W1's own Hungarian inflated CV but
  degraded LB 0.353 → 0.344. Kiza's "extended Hungarian" applies to clusters of
  size 9–18 and claims 99.6% train-side purity; that is still hard assignment
  and still cannot be validated on the unseen test prompts. Sang's notebook
  advertises full Hungarian and shares the same tail. The 0.604–0.612 fork
  ladder is not independent evidence: both are forks of one post-processing
  recipe.

**Ruling:** soft Gaussian only, low weight, no hard assignment, no Hungarian.
CV gains from hard grouping are not admissible evidence.

### 2.3 Sinkhorn: keep global, do not promote per-cluster

`submission_sota.py` §8 applies one global scaling to a uniform class marginal
at full strength. `INTEGRATION_v6` §3 is correct about its defects:

- Train is exactly 150 prompts × 18 models, so the global class marginal of any
  OOF matrix is uniform by construction. A global-uniform projection cannot be
  falsified in CV; it will always look neutral or good.
- It loses even where it should be safe: balanced OOF 0.3790 → 0.3767; mild skew
  0.3782 → 0.3716; strong skew 0.3778 → 0.3664.
- It does not achieve its objective: after scaling, predicted counts still span
  0.043–0.070 per class; the argmax of a doubly stochastic matrix is not uniform.

Per-cluster projection is falsifiable because clusters smaller than 18 make the
uniform column marginal a real assumption. On the 0.379 surrogate it gains
**+0.113** (95% CI +0.095…+0.131; oracle +0.118), with 18/18 class recalls
improving. But:

1. The 0.113 is measured against a 0.379 TF-IDF base. It is not additive to
   0.539 or 0.600, and no full-stack OOF or LB result exists.
2. It depends on recovered clusters. Hidden-test prompt recovery is the exact
   step that failed publicly; a cluster error can flip an entire group.
3. The measured source is gone. `INTEGRATION_v6` pins
   `src/test_time_calibration.py` at md5 `25e011d` and 1,058 lines. At audit time
   the file is **39 lines**, md5 `ce372db…`, and `test_calibration_invariants`
   returns hard-coded 0.8. `experiments/29_global_vs_cluster_ot.py` imports the
   stub, where `reg` is ignored and the operator is a global target, not the
   entropic OT that was measured. A 58 KB `.pyc` from 02:34 survives, but the
   source path is not reproducible.

**Ruling:** keep global Sinkhorn in v10 because it is part of the known 0.600
artifact; do not replace it with per-cluster OT. Per-cluster OT is the highest
value research lead, not a production change. If pursued, restore the 1,058-line
module, re-run `experiments/29`, and gate it behind a full-stack OOF and an LB
A/B.

### 2.4 ATLAS: real but small, and cost is unverified

W1 reports +0.007 on the exact 0.600 base, 7/7 positive. W2's forum summary says
+0.002–0.004; W5 says +0.003 on 0.604. The directions agree, the magnitude does
not. The effect is likely **+0.002–0.007**, with publication bias and a single
author's seeds. ATLAS runs ~18 source-conditioned forwards per passage; the
artifact is trained on the official 2,699 rows for five epochs with frozen
Qwen3-8B-Base and 18 adapters. The 2×T4 runtime for ~4,000 rows is not verified.

**Ruling:** ship at 10%, before Sinkhorn, with a hard 9-hour guard and a v5
fallback. Do not increase the weight without an LB result.

### 2.5 Corpus upgrade: best medium-term lead, not a v10 change

W3's `pool_v4` + `chosunghwan` + `pardheev` + `sharipov` improves transfer BA
0.3623 → 0.3986 → 0.4068 (+0.045 vs `pool_v1`) in an identical linear-SVM
probe, with zero exact overlaps against train. The gains are concentrated in
weak classes: kimi 0.11→0.35, gemini-3.1 0.49→0.60, claude-opus-4-8 0.24→0.32,
claude-fable-5 0.16→0.23. But:

- The probe is stage-1 linear SVM, not a LoRA, not the full stack, and not LB.
- `arxitekton*` is ~21% of `pool_v4` with unknown provenance; ~23k rows have
  ≤0.19 exact fidelity; `tadeuhayashida` is 0.09 and `kenandemirci` is 0.12.
- The competition permits public data, but winner licensing and attribution
  still apply. Unknown provenance is a compliance risk, not just a quality risk.
- `prvsiyan/ntl-public-in-style-pool-v1` (38,642 rows, CC-BY-SA-4.0, 16–18 MB,
  typography-vetoed, transfer 0.3682) is the cleaner corpus if a retrain is
  attempted.

**Ruling:** do not retrain for v10. Audit the new three sources and use the
`prvsiyan` pool or a provenance-cleaned `pool_v1` in the next iteration. Do not
use untraceable `arxitekton*` rows in a winning submission.

### 2.6 Calibration and family gates

W4's per-class logit-bias calibration is +0.5–1.5 pp with a validated mechanism
(CAP 61.94→56.38; GATC +1.5–4 pp), but the magnitude is plausible, not measured
on this task. W5's within-family gates gain +3.7 pts on the 5-way Claude subset.
`INVALIDATION_v2` shows why a static 85/15 blend is unsafe: the per-class optimal
`w*` ranges from **0.00** (`claude-opus-5`, `kimi-k3`, `gpt-5.6-terra`) to
**0.90** (`claude-fable-5`, `gemini-3.1-pro`); Fable-5 recall moves 0.14→0.46.

**Ruling:** any gate or calibration must be nested/OOF per class, not a global
static weight. These are optional only if the ATLAS probe leaves submission
budget; they are not part of the primary v10.

### 2.7 Spaced-em-dash shields

The invariant is real on train: 86 rows contain a spaced em dash, all 86 belong
to `claude-opus-4-6`, `claude-opus-5`, `glm-5.3`, or `glm-5.3-flash`; zero rows
in the other 14. A post-Sinkhorn filter on those four classes has zero ripple.
But the OOF effect is only +0.0004, and public forks using hard punctuation
vetoes lost 0.003–0.005. The hidden test may not preserve the formatting quirk.

**Ruling:** optional post-Sinkhorn guard, exact spaced pattern only, no broad
punctuation veto. Low expected gain; do not let it replace a model improvement.

### 2.8 93 vs 101 sibling features

The shipped `src/sibling_features.py` has `N_FEATURES == 93`, 10 sibling pairs,
and import-time schema guards; the brief's 101 is stale. The integration result
is +0.010 with a CI crossing zero (−0.004…+0.026); the eight pairwise heads are
exactly 0.0000. This is not a shipping delta.

**Ruling:** pin the 93-feature hash. Do not add the pairwise stacker to v10.

### 2.9 The 0.607 provenance problem

Joseph's public kernel reports a historical 0.607; the old-version endpoint is
404. Prvsiyan's notebook states its exact V4 inference scored 0.604 on official
submission 57428928, and the V5 whole-notebook result was 0.603. Kiza/sang claim
0.604–0.612 with a shared fork. There is no independently verified complete
score/source binding for 0.607 and no evidence that the kiza/sang scores isolate
their Hungarian step.

**Ruling:** cite the public ladder as author-claimed, not replicated. Do not
budget for a guaranteed +0.007; budget for a positive but noisy +0.002–0.007.

---

## 3. Artifact verification

| Artifact | Claim | Status | Evidence / risk |
|---|---|---|---|
| ATLAS (`josephadamski91/ntl-atlas-qwen3-8b-v1`) | +0.007 at 10%, 18 adapters | **Verified metadata**, effect author-reported | datasetId 12657014; 2,699 official rows, 5 epochs, frozen Qwen3-8B-Base, 18 source adapters; weights CC BY-SA 4.0, code Apache 2.0, Kaggle field `other`; runtime/size on 2×T4 unverified |
| zbigniew `lora-instyle` / `lora-claude-head` | 0.456→0.516; LB 0.569→0.574 at w=0.17 | **Verified artifact**, effect not re-run | Apache-2.0, 186.5 MB; adding a third correlated 8B member is the open risk |
| `lightningv08` 27B NF4 | “best member” | **Metadata only** | 27B, final 8 decoder layers, rank 16/alpha 32, 28,390 external + 2 official epochs; final weights not OOF; external prompt independence not certified; no 2×T4 proof; Apache-2.0 component, collection `other` |
| UltraBERT (`pjmathematician`) | OOF 0.408, 5-fold | **Verified** | Apache-2.0; integration value only +0.010 with CI crossing zero |
| `prvsiyan` in-style pool | 38,642 rows, transfer 0.3682 | **Verified** | CC-BY-SA-4.0, 16–18 MB, typography-vetoed; cleanest external corpus |
| `ahmedberatozer/ntl-pool-v4` + new3 | transfer 0.4068 | **Proxy only** | CC-BY-SA-4.0, ~72k rows; `arxitekton*` 21% unknown provenance; ~23k rows ≤0.19 fidelity; not a LoRA/LB result |
| `src/test_time_calibration.py` | per-cluster +0.113 | **Broken dependency** | source is now 39 lines, md5 `ce372db…`; measured md5 `25e011d` / 1,058 lines is absent; `experiments/29` imports the stub; 58 KB pyc from 02:34 exists but is not a production source path |
| `src/unbalanced_sinkhorn.py` | real UOT/HodgeRank/clusterer | **Present but not the measured module** | different API (`tau_col`, `alpha`); do not treat as a drop-in replacement for the 1,058-line module |
| `sota_600/script.py` | exact 0.600 | **Verified** | 399 lines, md5 `ce66a0f…`; embedded `_GAUSSIAN_SOURCE`; 85/15 blend; global Sinkhorn |
| `submission_sota.py` | v7 0.70+ / “v5 gold standard” | **Not the 0.600 artifact** | 3,506 lines, md5 `9b5fe1b…`; header advertises clique calibration, final code skips it; last observed LB behavior is v9 0.591 |
| `src/sibling_features.py` | 93 features | **Verified** | md5 `0694d5f…`; 10 pairs; +0.010 with CI crossing zero; pairwise 0.0000 |

---

## 4. Per-change risk table

| ID | Change | Best evidence | Expected LB | Risk | v10 |
|---|---|---|---|---|---|
| T1 | ATLAS 10% pre-Sinkhorn | +0.007 on exact 0.600, 7/7 | +0.002…+0.007 | Medium: 18 forwards/row, runtime unverified | **Ship** |
| T2 | Shared-prompt Gaussian head | +0.0100 OOF, 68/68; ~+0.005 LB | +0.003…+0.006 | Medium: feature/GPU integration; already partly represented by v5's 15% Gaussian stack | Optional, only as isolated A/B |
| T3 | zbigniew in-style 8B LoRA | 0.569→0.574 at w=0.17 | +0.002…+0.005 | Low-medium: correlated with existing 8B; runtime | Optional if budget remains |
| T4 | Within-family gates | +3.7 pts on 5-way Claude subset | +0.001…+0.005 | Medium: data-dependent; static weights overfit | Optional, nested only |
| T5 | Spaced-em-dash guard | 86/86 in 4 classes, 0/14 others; +0.0004 OOF | ≈0.000…+0.001 | Low: exact pattern only | Optional |
| T6 | Per-class logit-bias calibration | +0.5–1.5 pp literature | +0.001…+0.005 | Medium-high: overfit; needs nested OOF | Do not ship untuned |
| T7 | Per-cluster OT | +0.113 on 0.379 surrogate | Unknown at 0.600 | **High**: source drift, cluster recovery, hard assignment | Research only |
| T8 | Corpus upgrade + retrain | +0.045 proxy | Unknown | **High**: provenance/licensing, time | Next iteration |
| T9 | 27B NF4 / LoRA bag | No comparative evidence | Unknown | **Very high**: 2×T4 feasibility | Reject for v10 |
| T10 | Reproduce v5 | 0.600 measured | 0.600 baseline | Low | **Mandatory baseline** |

The deltas are not additive. T1 on 0.600 is the only move with a measured LB
effect on the exact parent; stacking T2–T9 on top would repeat the v8/v9 mistake
of shipping unvalidated interactions.

---

## 5. Final v10 specification

**Base**

- `notebooks/sota_600/script.py`, md5 `ce66a0f…`, the exact 0.600 recipe.
- Keep its two-stage 8B/14B trunks, pool n-grams, centred twin, HodgeRank
  sibling, smooth length/typographic prior, 85/15 Gaussian blend, and global
  Sinkhorn.
- Do not start from `submission_sota.py`; its v7 header and dead clique path are
  not the measured artifact.

**Single change**

- Load ATLAS v2 with its exact class order and hashes.
- Compute `p_atlas` for every test row.
- Blend **before** Sinkhorn:
  `P = 0.90 * P_v5 + 0.10 * P_atlas`.
- Renormalize, assert finite and row sums 1, then run the inherited global
  Sinkhorn unchanged.
- No silent fallback: if ATLAS fails to load, abort to exact v5, do not ship a
  half-blend.

**Optional guard**

- After Sinkhorn, for rows containing the exact spaced em-dash pattern, zero
  only the 14 classes with zero train support and renormalize.
- Do not apply a broad punctuation veto and do not hard-code unspaced em dashes.

**Runtime plan**

- Sort test rows by token length; shard ATLAS inference; batch as large as the
  16 GB T4 allows; checkpoint each shard so a timeout can resume.
- Budget 18 forwards per row; if the projected time exceeds 7 hours, drop ATLAS
  and submit v5. Do not risk the 9-hour cap.

**Validation / submission plan**

- Smoke on the 3-row interactive file; verify columns, row count, 18 valid
  labels, no nulls, and non-collapsed class distribution.
- First probe: v10 (v5 + ATLAS) against the known v5 score.
- Keep at least one final slot for exact v5. Only final submissions are judged.
- If v10 is ≤ v5, revert immediately and use the remaining probes on T3 or T5
  in isolation, not on a combined rewrite.

**Fallback**

- Exact v5 at 0.600. It is a known quantity and still ranks around the top-200
  boundary; the current v9 at 0.591 is strictly worse.

---

## 6. What we still do not know

- ATLAS 2×T4 runtime, peak VRAM, and whether 18 forwards/row fit the 9-hour cap
  for ~4,000 rows.
- Whether ATLAS's +0.007 on public transfers to the 60% private split; the
  public/private model mixture differs.
- The exact public/private row counts; the 40/60 split is official but the
  absolute test size is only approximately 4,000.
- Whether global Sinkhorn helps or hurts the actual LB; no same-recipe A/B exists.
- Whether the spaced-em-dash invariant survives hidden-test formatting.
- Whether the 1,058-line per-cluster module can be restored from the surviving
  `.pyc` or must be rewritten; the current source is a stub.
- Whether hidden-test prompt recovery is good enough for any per-cluster method;
  no test-side labels exist.
- The provenance and license of `arxitekton*` and the exact composition of
  `pool_v4`; this blocks a compliant corpus retrain.
- Whether the 0.607 historical score has a real source binding; the old endpoint
  is 404.
- Whether the current `submission_sota.py` hash will drift again before
  packaging; there is no git history to pin it.

---

## 7. Bottom line

The project does not need another architecture. It needs to stop shipping
unvalidated interactions and put the one measured improvement on the one
measured base. **v5 + ATLAS 10%** is the defensible v10; everything else is
either noise, a CV trap, or an unverified dependency.
