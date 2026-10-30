# v14 / v15 — single-change ablations of the v13 regression

## What the leaderboard says

| Ref | Variant | Public LB | Δ vs v10 |
| :--- | :--- | :--- | :--- |
| 57497594 | **v10** (baseline) | **0.606** | — |
| 57471060 | v5 | 0.600 | −0.006 |
| 57662874 | **v15** (lora24 plain reweight only, twins 0) | **0.593** | **−0.013** |
| 57477702 | v8 | 0.593 | −0.013 |
| 57623259 | v13 | 0.592 | −0.014 |
| 57627241 | v12-fast | 0.584 | −0.022 |

v10 remains the best valid submission and the Final Submission fallback. Nothing has
touched the two Final Submission slots.

**v15 is a clean single-change answer:** adding lora24 plain mass (`+0.10 / +0.05 / +0.05`)
on top of v10 costs **0.013**. v10's lora24 weighting is not under-weighted — it is at a local
optimum, and every attempt to add more lora24 evidence (v12-fast twins 0.80/0.20, v13
prompt-centre, v15 plain) lands in the 0.584–0.593 band, i.e. −0.013 to −0.022. That also
means v16's Stage 2 is asking a different question: it *removes* lora24 and substitutes a
14B LoRA pair plus a fine-tuned DeBERTa, rather than re-weighting the existing evidence.

## Exact change set, v10 → v13

A masked diff of the two scripts (`_GAUSSIAN_SOURCE` collapsed, guard collapsed)
leaves only **two** modelling changes. Everything else is instrumentation.

1. **SVC front-end** — three edits shipped as one bundle:
   - pool `pool_v1.parquet` → `curated_pool.parquet` (81,221 rows after dropping
     `train_regen`, requiring `register_ok`, 30–300 words, dedup)
   - `sample_weight = 0.25` on pool rows, 1.0 on official train rows
   - transductive vocabulary: the four TF-IDF vectorizers are now `fit` on
     train + pool + **test**, then `transform` the fit rows
2. **Top-level lora24 re-weight** — a new block after the legacy blend that adds
   `w · log p` for the lora24 members on top of `LOGIT = log(0.85·new + 0.15·legacy)`:

   | member | weight |
   | :--- | :--- |
   | `lora24_public` | +0.10 |
   | `lora24_public_twin` | **+0.70** |
   | `lora24_seed7` | +0.05 |
   | `lora24_seed7_twin` | 0.0 |
   | `lora24_claude` | +0.05 |
   | `lora24_claude_twin` | +0.15 |

   Total added weight ≈ **1.05**, i.e. as large as `W_SVC = 1.021` and comparable to
   the whole normalised base blend it is being added to.

Plus a required, non-modelling change: the checkpoint guard (`find_base` /
`find_checkpoint` now require real weight files and prefer `/kaggle/input/models/`).
`ntl-public-heads` gained a `base_config` directory between Oct 21 and Oct 25, which
is why v13 v1 and v12-fast v1 both died with `OSError: no file named
model.safetensors`. v10's original selection is not reproducible against the current
dataset version, so the guard has to stay in every variant. LoRA anchor parity
(`max abs 0.0000`) confirms the guard resolves the same trunks v10 used.

## Why the re-weight is the prime suspect

- The centred twin gets **0.70**, while the *uncentred* `lora24_public` only gets
  0.177 inside the legacy blend. The centred readout is over-weighted 4×.
- The twin is only added on `GROUP_ROWS` (rows in inferred clusters of size 2–18).
  On the hidden ~4k set that is most rows, so the effective blend is *different for
  grouped vs singleton rows* — exactly the inconsistency that balanced accuracy
  punishes.
- v12-fast moved only `lora24_public_twin 0.70 → 0.80` and `lora24_claude_twin
  0.15 → 0.20` and lost another 0.008. Local slope ≈ −0.08 per unit of twin weight,
  so 0.70 is already deep on the wrong side of zero.
- The twin weights were fitted on grouped OOF (0.596), but OOF has not transferred
  on this competition (curated SVC OOF 0.4667 vs 0.414 also failed to transfer).

## The testing gap that let this ship

On the 3-row public `test.csv` every row is a singleton, so
`GROUP_ROWS.any()` is False and **no twin is ever computed**. The public log shows
only three `REWEIGHT` lines (`lora24_public`, `lora24_seed7`, `lora24_claude`) — the
three `*_twin` keys are absent from `LEGACY_MEMBERS`. The gate therefore validated a
strictly easier code path than the hidden rerun executes. Any future variant that
changes group-conditional behaviour must be gated on more than the 3-row public run.

## The two ablations now running

Both are v13 with **one** change reverted, so each isolates a single candidate.

| Variant | Kernel | Change vs v10 | Tests |
| :--- | :--- | :--- | :--- |
| **v14** | `pikaaa/name-that-llm-v14-curated-svc-only` | curated SVC bundle only, `EXTRA_LORA24 = {}` | does the OOF 0.414→0.4667 SVC gain transfer to the LB? |
| **v15** | `pikaaa/name-that-llm-v15-lora24-plain-reweight-only` | `pool_v1.parquet` SVC restored, re-weight = plain heads only `0.10 / 0.05 / 0.05`, twins 0 | does *any* extra lora24 mass help once the twins are gone? |

Verified masked diffs: v14 vs v10 = SVC block + guard + instrumentation only;
v15 vs v10 = re-weight block + guard + instrumentation only.

Scripts:

```
v14  8eccd0cf4b2cfc0fee3f5336e3dabfff4b5e84c71a056d0f35e323223b3babda
v15  fb78510430b63a6de4f90caa6a0d59a95c945f408e159490fb497aee00352122
```

## Decision matrix (v15 has landed)

| v14 | v15 | Reading | Next |
| :--- | :--- | :--- | :--- |
| > 0.606 | any | curated SVC transfers | ship v14, then try v14 + small twin |
| ≤ 0.606 | > 0.606 | re-weight was fine, SVC was not | ship v15, then v15 + curated SVC off |
| ≤ 0.606 | ≤ 0.606 | both changes are harmful | keep v10, stop touching the blend |
| > 0.606 | > 0.606 | both help | combine into v16 and submit |

**v15 = 0.593, so the second row is dead and the third is live:** the lora24 re-weight is
harmful on its own. Only v14 can still clear 0.606; otherwise the rule holds and v10 stays.

Hard rule: no Final Submission is replaced unless a variant beats **0.606**.

## Runner

`experiments/run_ablation.sh <dir> <kernel-id> <log> <description>` — push (or
`NTL_SKIP_PUSH=1 NTL_VERSION=n`), poll the public run, enforce the 11,880 s runtime
envelope, verify `members_test.npz` + `submission.parquet` integrity against v10's
public output, submit, then poll the hidden rerun to a score.
